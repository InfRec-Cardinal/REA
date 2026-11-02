import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Writable } from "node:stream";
import { setTimeout as delay } from "node:timers/promises";
import { after, before, test } from "node:test";

import { ProgramClient, DEFAULT_TIMEOUTS_MS, DEFAULT_WALL_TIMEOUT_S, timeoutFor } from "../src/program-client.mjs";
import { LiberoClient } from "../src/libero-client.mjs";

// A stand-in backend: echoes the request and its argv, delays when asked, can answer ok:false,
// can emit a non-JSON line to prove the client ignores protocol noise.
const echoBackend = `
import { createInterface } from "node:readline";
const args = process.argv.slice(2);
process.stderr.write("backend stderr noise\\n");
createInterface({ input: process.stdin }).on("line", line => {
  const request = JSON.parse(line);
  const reply = () => process.stdout.write(JSON.stringify({ id: request.id, ok: true, command: request.command, args, options: request }) + "\\n");
  if (request.command === "fail") { process.stdout.write(JSON.stringify({ id: request.id, ok: false, error: "ValueError: nope" }) + "\\n"); return; }
  if (request.command === "noise") { process.stdout.write("this is not json\\n"); reply(); return; }
  if (request.command === "close") { reply(); process.exit(0); }
  setTimeout(reply, Number(request.delay_ms ?? 0));
});
`;


// Reads fd 0 synchronously (no libuv handle on it), so it can close its stdin while staying alive:
// the next write from the client then fails with EPIPE while the backend process is still running.
const closingBackend = `
import { closeSync, readSync, writeSync } from "node:fs";
let buffer = "";
const chunk = Buffer.alloc(65536);
function readLine() {
  for (;;) {
    const newline = buffer.indexOf("\\n");
    if (newline >= 0) { const line = buffer.slice(0, newline); buffer = buffer.slice(newline + 1); return line; }
    let count = 0;
    try { count = readSync(0, chunk); } catch (error) { if (error.code === "EAGAIN") continue; throw error; }
    if (count === 0) return null;
    buffer += chunk.toString("utf8", 0, count);
  }
}
for (;;) {
  const line = readLine();
  if (line === null) break;
  const request = JSON.parse(line);
  const reply = JSON.stringify({ id: request.id, ok: true, command: request.command }) + "\\n";
  if (request.command === "close_stdin") { closeSync(0); writeSync(1, reply); setTimeout(() => process.exit(0), 3000); break; }
  writeSync(1, reply);
}
`;

let directory;
let script;
let closingScript;
const silent = () => new Writable({ write(chunk, encoding, callback) { callback(); } });

before(async () => {
  directory = await mkdtemp(join(tmpdir(), "rea-program-client-"));
  script = join(directory, "echo_backend.mjs");
  await writeFile(script, echoBackend);
  closingScript = join(directory, "closing_backend.mjs");
  await writeFile(closingScript, closingBackend);
});
after(async () => { await rm(directory, { recursive: true, force: true }); });

async function withClient(options, run) {
  const client = new ProgramClient({ python: process.execPath, script, stderr: silent(), ...options });
  try {
    return await run(client);
  } finally {
    if (client.child.exitCode === null) client.child.kill("SIGKILL");
    await client.exit;
  }
}

test("timeoutFor gives every command its own limit and derives run_program from the wall timeout", () => {
  assert.equal(timeoutFor("observe"), 60000);
  assert.equal(timeoutFor("reset"), 300000);
  assert.equal(timeoutFor("run_program", { wall_timeout_s: 1 }), 61000);
  assert.equal(timeoutFor("run_agent", { wall_timeout_s: 600 }), 660000);
  assert.equal(timeoutFor("run_agent", { wall_timeout_s: 600 }, { ...DEFAULT_TIMEOUTS_MS, run_agent: 5 }), 5);
  assert.equal(DEFAULT_WALL_TIMEOUT_S, 900);
  assert.equal(timeoutFor("run_program"), 900 * 1000 + 60000);
  assert.equal(timeoutFor("unknown_command"), 180000);
  assert.equal(timeoutFor("run_program", { wall_timeout_s: 1 }, { ...DEFAULT_TIMEOUTS_MS, run_program: 5 }), 5);
  assert.equal(DEFAULT_TIMEOUTS_MS.run_program, null);
});

test("ProgramClient spawns the backend with extra arguments and round-trips JSON lines", async () => {
  await withClient({ args: ["--aspire-root", "/tmp/aspire", "--max-steps", "4000"] }, async client => {
    assert.ok(client instanceof LiberoClient);
    const reply = await client.request("catalog");
    assert.equal(reply.ok, true);
    assert.deepEqual(reply.args, ["--aspire-root", "/tmp/aspire", "--max-steps", "4000"]);
    const observe = await client.request("observe", { view: "montage" });
    assert.equal(observe.options.view, "montage");
    assert.equal(observe.options.command, "observe");
    assert.equal(observe.id, 2);
    await assert.rejects(client.request("fail"), /ValueError: nope/u);
    const afterNoise = await client.request("noise");
    assert.equal(afterNoise.command, "noise");
    assert.equal(client.pending.size, 0);
  });
});

test("per-command timeouts reject slow replies and leave later requests working", async () => {
  await withClient({ timeouts: { observe: 40 } }, async client => {
    await assert.rejects(client.request("observe", { delay_ms: 400 }), /^Error: program backend observe timed out/u);
    await assert.rejects(client.request("score", { delay_ms: 400 }, { timeoutMs: 40 }), /program backend score timed out after 40 ms/u);
    assert.equal(client.pending.size, 0);
    const quick = await client.request("score", {}, { timeoutMs: 5000 });
    assert.equal(quick.command, "score");
    const untimed = await client.request("observe", { delay_ms: 60 }, { timeoutMs: null });
    assert.equal(untimed.options.delay_ms, 60);
  });
});

test("a backend that closed its stdin rejects the next request with EPIPE instead of crashing the process", async () => {
  await withClient({ script: closingScript }, async client => {
    const closed = await client.request("close_stdin");
    assert.equal(closed.command, "close_stdin");
    let uncaught = null;
    const guard = error => { uncaught = error; };
    process.once("uncaughtException", guard);
    try {
      // The first write fails with EPIPE while the backend lives on; Node then destroys the stream, so a
      // later request is rejected as well (destroyed stream) instead of hanging or crashing.
      await assert.rejects(client.request("observe", { blob: "x".repeat(1 << 17) }, { timeoutMs: 2000 }), /EPIPE/u);
      await assert.rejects(client.request("observe", { view: "montage" }, { timeoutMs: 2000 }), /EPIPE|destroyed/u);
      await delay(50);
      assert.equal(uncaught, null, "the stdin error must be handled by the client");
      assert.equal(client.pending.size, 0);
      assert.equal(client.child.exitCode, null, "the backend was still alive when the write failed");
    } finally {
      process.off("uncaughtException", guard);
    }
  });
});

test("close sends close, ends stdin and rejects requests that were still pending on exit", async () => {
  const client = new ProgramClient({ python: process.execPath, script, stderr: silent() });
  const dangling = client.request("reset", { delay_ms: 5000 });
  const rejected = dangling.then(() => null, error => error.message);
  await client.close();
  assert.notEqual(client.child.exitCode, null);
  assert.match(await rejected, /program backend exited/u);
  await client.close();
});
