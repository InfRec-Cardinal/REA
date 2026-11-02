import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Writable } from "node:stream";
import { setTimeout as delay } from "node:timers/promises";
import { test } from "node:test";

import { LiberoClient, createLiberoExecutor } from "../src/libero-client.mjs";

test("LIBERO executor gates stale frames and retains simulator success", async () => {
  let state = { frame: { libero: { observationId: "1:1" } }, steps: 0, success: false, terminated: false };
  const calls = [];
  const client = { request: async (command, options) => {
    calls.push({ command, ...options });
    return { frame: { libero: { observationId: "1:2" } }, steps: 5, success: false, terminated: false };
  } };
  const executor = createLiberoExecutor({ client, getState: () => state, setState: next => { state = next; }, tokens: ["X", "O"] });
  const frame = state.frame;
  const result = await executor.execute({ token: "O", frame, execute: true });
  assert.equal(result.executed, true);
  assert.deepEqual(calls, [{ command: "step", token: "O" }]);
  assert.equal(state.success, false);
  const stale = await executor.execute({ token: "X", frame, execute: true });
  assert.equal(stale.executed, false);
  assert.deepEqual(stale.gate.reasons, ["stale_observation"]);
  assert.equal(calls.length, 1);
});

test("LIBERO preview and invalid tokens never step the simulator", async () => {
  const state = { frame: { libero: { observationId: "1:1" } }, terminated: false, steps: 0 };
  const executor = createLiberoExecutor({
    client: { request: async () => assert.fail("unexpected simulator step") },
    getState: () => state, setState: () => {}, tokens: ["X"],
  });
  assert.equal((await executor.execute({ token: "X", frame: state.frame, execute: false })).preview, true);
  const invalid = await executor.execute({ token: "BAD", frame: state.frame, execute: true });
  assert.equal(invalid.executed, false);
  assert.deepEqual(invalid.gate.reasons, ["invalid_manipulation_token"]);
});


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

test("LiberoClient turns a write after the backend closed stdin into a rejected request, not an uncaught EPIPE", async () => {
  const directory = await mkdtemp(join(tmpdir(), "rea-libero-client-"));
  const script = join(directory, "backend.mjs");
  await writeFile(script, closingBackend);
  const client = new LiberoClient({ python: process.execPath, script, stderr: new Writable({ write(chunk, encoding, callback) { callback(); } }) });
  let uncaught = null;
  const guard = error => { uncaught = error; };
  process.once("uncaughtException", guard);
  try {
    const closed = await client.request("close_stdin");
    assert.equal(closed.command, "close_stdin");
    // LiberoClient has no write callback: only the stdin 'error' listener can settle this request.
    await assert.rejects(client.request("step", { token: "X" }), /EPIPE/u);
    await delay(50);
    assert.equal(uncaught, null);
    assert.equal(client.pending.size, 0);
    assert.equal(client.child.exitCode, null, "the backend was still alive when the write failed");
  } finally {
    process.off("uncaughtException", guard);
    if (client.child.exitCode === null) client.child.kill("SIGKILL");
    await client.exit;
    await rm(directory, { recursive: true, force: true });
  }
});
