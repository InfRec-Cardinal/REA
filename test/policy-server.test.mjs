import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { createBridge } from "../src/framework/bridge.mjs";
import { createRouteMemoryRouter } from "../src/framework/route-memory.mjs";
import { runRoutedEpisode } from "../src/framework/routed-runtime.mjs";
import { runEpisode } from "../src/framework/runtime.mjs";
import { servePolicy } from "../src/framework/policy-server.mjs";
import { loadConfiguration } from "../src/framework/configuration.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const MOVE = { name: "move", description: "Move by dx", parameters: { type: "object", properties: { dx: { type: "number" } }, required: ["dx"], additionalProperties: false } };
const observation = n => ({ text: "push", state: { x: n }, images: [] });

test("the bridge lets an external evaluator drive runEpisode: observations in, actions out, its result as the outcome", async () => {
  const bridge = createBridge({ actions: [MOVE] });
  bridge.begin();
  const seen = [];
  const policy = { id: "half-step", async decide({ observation: current }) {
    seen.push(current.state.x);
    return { action: { name: "move", arguments: { dx: current.state.x / 2 } } };
  } };
  const run = runEpisode({ environment: bridge.environment, policy, maxSteps: 10 });
  // The evaluator's first observation may arrive before the runtime asks for it.
  assert.deepEqual(await bridge.step({ observation: observation(4) }), { action: { name: "move", arguments: { dx: 2 } } });
  assert.deepEqual(await bridge.step({ observation: observation(2) }), { action: { name: "move", arguments: { dx: 1 } } });
  bridge.end({ success: true });
  const summary = await run;
  assert.deepEqual(seen, [4, 2]);
  assert.equal(summary.steps, 2);
  assert.equal(summary.success, true);
  assert.equal(summary.reason, "environment_limit");
  bridge.finish({ reason: summary.reason });
  assert.deepEqual(await bridge.step({ observation: observation(1) }), { done: true, reason: "environment_limit", summary: null });
});

test("when the runtime finishes first, the waiting evaluator is told the episode is done", async () => {
  const bridge = createBridge({ actions: [MOVE] });
  bridge.begin();
  const policy = { async decide() { return { action: { name: "move", arguments: { dx: 1 } } }; } };
  const run = runEpisode({ environment: bridge.environment, policy, maxSteps: 1 }).then(summary => {
    bridge.finish({ reason: summary.reason, summary: { steps: summary.steps } });
    return summary;
  });
  assert.equal((await bridge.step({ observation: observation(0) })).action.name, "move");
  const done = await bridge.step({ observation: observation(1) });
  assert.equal(done.done, true);
  assert.equal(done.reason, "max_steps");
  assert.equal((await run).steps, 1);
});

test("a routed episode over the bridge reads the evaluator's result as its sealed score", async () => {
  const bridge = createBridge({ actions: [MOVE] });
  bridge.begin();
  const capability = id => ({ id, policy: { async decide() { return { action: { name: "move", arguments: { dx: 1 } } }; } } });
  const run = runRoutedEpisode({
    environment: bridge.environment, task: "push", maxStepsPerRun: 2, system0: [capability("vla"), capability("code")],
    system1: createRouteMemoryRouter({ recoverOn: "always", maxRuns: 2 }),
  });
  for (let step = 0; step < 4; step += 1) assert.equal((await bridge.step({ observation: observation(step) })).action.name, "move");
  bridge.end({ success: true });
  const summary = await run;
  // The evaluator ended the episode while the second capability was running.
  assert.deepEqual(summary.runs.map(item => [item.candidateId, item.status]), [["vla", "budget"], ["code", "truncated"]]);
  assert.equal(summary.success, true);
  assert.deepEqual(summary.score, { success: true });
});

function client(config, options = {}) {
  const input = new PassThrough();
  const output = new PassThrough();
  const replies = [];
  const waiting = [];
  let buffer = "";
  output.on("data", chunk => {
    buffer += chunk;
    let index;
    while ((index = buffer.indexOf("\n")) >= 0) {
      const reply = JSON.parse(buffer.slice(0, index));
      buffer = buffer.slice(index + 1);
      if (waiting.length) waiting.shift()(reply); else replies.push(reply);
    }
  });
  const served = servePolicy({ config, baseDir: root, env: {}, input, output, ...options });
  let id = 0;
  const request = (op, fields = {}) => new Promise(resolve => {
    waiting.push(resolve);
    input.write(`${JSON.stringify({ id: ++id, op, ...fields })}\n`);
  });
  return { request, served, input };
}

const PACKED = { name: "packed_action_chunk", description: "commands", parameters: { type: "object", properties: { commands: { type: "array", items: { type: "array", items: { type: "number" }, minItems: 14, maxItems: 14 }, minItems: 1 } }, required: ["commands"], additionalProperties: false } };
const NAMED = { name: "action_chunk", description: "commands", parameters: { type: "object", properties: { commands: { type: "array", minItems: 1, items: { type: "object" } } }, required: ["commands"], additionalProperties: true } };

test("servePolicy speaks JSON lines: configure, reset, step until done, end with the evaluator's result, close", async context => {
  const output = await mkdtemp(join(tmpdir(), "rea-serve-"));
  context.after(() => rm(output, { recursive: true, force: true }));
  const { config, baseDir } = await loadConfiguration(join(root, "test/fixtures/serve-scripted.config.json"));
  const { request, served } = client(config, { baseDir, outputDir: output });
  const configured = (await request("configure", { actions: [PACKED, NAMED], name: "fixture" })).result;
  assert.equal(configured.mode, "policy");
  assert.equal(configured.id, "fixture");
  assert.equal((await request("step", { observation: observation(0) })).error, "send reset before the first step");
  assert.deepEqual((await request("reset", { task: "stack_bowls", seed: 3, instruction: "Stack the bowls." })).result, { episode: 1 });
  const first = (await request("step", { observation: observation(0) })).result;
  assert.equal(first.action.name, "packed_action_chunk");
  assert.equal(first.action.arguments.commands[0].length, 14);
  assert.equal((await request("step", { observation: observation(1) })).result.action.name, "action_chunk");
  const done = (await request("step", { observation: observation(2) })).result;
  assert.equal(done.done, true);
  assert.equal(done.reason, "max_steps");
  const ended = (await request("end", { result: { success: false } })).result;
  assert.equal(ended.summary.steps, 2);
  assert.deepEqual((await request("bogus")).error, 'unknown op "bogus"');
  assert.deepEqual((await request("close")).result, { closed: true });
  assert.deepEqual(await served, { episodes: 1 });
  assert.deepEqual((await readdir(output)).sort(), ["0001.json", "0001.jsonl"]);
  const record = JSON.parse(await readFile(join(output, "0001.json"), "utf8"));
  assert.deepEqual(record.external, { success: false });
  const trace = (await readFile(join(output, "0001.jsonl"), "utf8")).trim().split("\n").map(line => JSON.parse(line));
  assert.equal(trace[0].type, "reset");
  assert.equal(trace.at(-1).type, "episode_end");
});

test("a new reset ends the running episode, and the CLI writes nothing but protocol lines to stdout", async () => {
  const child = spawn(process.execPath, [join(root, "scripts/rea.mjs"), "serve", join(root, "test/fixtures/serve-scripted.config.json")], { cwd: root, stdio: ["pipe", "pipe", "pipe"], env: { ...process.env, REA_ENV_FILE: "" } });
  let stdout = "";
  child.stdout.on("data", chunk => { stdout += chunk; });
  const lines = [
    { id: 1, op: "configure", actions: [PACKED, NAMED] },
    { id: 2, op: "reset", task: "a" },
    { id: 3, op: "step", observation: observation(0) },
    { id: 4, op: "reset", task: "b" },
    { id: 5, op: "step", observation: observation(0) },
    { id: 6, op: "close" },
  ];
  for (const line of lines) child.stdin.write(`${JSON.stringify(line)}\n`);
  const code = await new Promise(resolve => child.on("close", resolve));
  assert.equal(code, 0);
  const replies = stdout.trim().split("\n").map(line => JSON.parse(line));
  assert.deepEqual(replies.map(reply => [reply.id, reply.ok]), lines.map(line => [line.id, true]));
  assert.equal(replies[3].result.episode, 2);
  assert.equal(replies[4].result.action.name, "packed_action_chunk", "the second episode starts from its first action again");
});
