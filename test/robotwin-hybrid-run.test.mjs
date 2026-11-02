import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import test from "node:test";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { classifyPolicyOutcome, createHybridEnvironment, withCodeDecisionBudget, withCodeReplay } from "../eval/robotwin/hybrid/run.mjs";
import { runAsyncEpisode } from "../src/framework/async-runtime.mjs";
import { createRobotwinCodePolicy } from "../src/policies/robotwin-code.mjs";
import { createRobotwinVlaPolicy } from "../src/policies/robotwin-vla.mjs";
import { createSuccessRouter } from "../src/policies/success-router.mjs";

const instance = { instance_id: "rt500-000-adjust_bottle-clean-00", task: "adjust_bottle", setting: "clean", seed: 100001, instruction: "Adjust the bottle." };
const config = { subset: "/fixture/frozen500", harnessRoot: "/fixture/harness", codeDecisions: 40 };

test("explicit provider content refusal preserves actions and usage while service errors remain unknown", () => {
  const partial = { reason: "error", success: null, steps: 20, actionAttempts: 20,
    info: { nativeActionCount: 20 }, usage: { total_tokens: 313111 },
    error: { code: "codex_error", message: "Codex exited 1: unexpected status 403 Forbidden: This request contains prohibited content." } };
  const result = classifyPolicyOutcome(partial);
  assert.deepEqual(result, { ...partial, reason: "policy_refusal", success: false,
    outcomeClassification: "The provider explicitly refused the request as prohibited content; episode failed without retry." });
  assert.equal(result.error, partial.error);
  for (const message of ["403 Forbidden: Invalid API key", "403 Forbidden: Insufficient credits",
    "429 Too Many Requests", "503 Service Unavailable"]) {
    const infrastructure = { ...partial, error: { code: "codex_error", message } };
    assert.equal(classifyPolicyOutcome(infrastructure), infrastructure);
  }
});

test("resuming a saved provider refusal completes it without restarting the simulator or calling the policy", async context => {
  const root = await mkdtemp(join(tmpdir(), "robotwin-refusal-"));
  context.after(() => rm(root, { recursive: true, force: true }));
  const output = join(root, "run");
  const directory = join(output, instance.instance_id);
  await mkdir(directory, { recursive: true });
  const original = { instance, mode: "code", reason: "error", success: null, steps: 20,
    actionAttempts: 20, usage: { total_tokens: 313111 },
    error: { code: "codex_error", message: "unexpected status 403 Forbidden: This request contains prohibited content." } };
  await writeFile(join(directory, "result.json"), JSON.stringify(original));
  await writeFile(join(directory, "decisions.jsonl"), "retained executed decisions\n");
  await writeFile(join(root, "config.json"), "{}");
  await writeFile(join(root, "instances.json"), JSON.stringify([instance]));
  execFileSync(process.execPath, [fileURLToPath(new URL("../eval/robotwin/hybrid/run.mjs", import.meta.url)),
    "--config", join(root, "config.json"), "--instances", join(root, "instances.json"),
    "--output", output, "--mode", "code"]);
  const result = JSON.parse(await readFile(join(directory, "result.json"), "utf8"));
  assert.deepEqual(result, classifyPolicyOutcome(original));
  assert.equal(await readFile(join(directory, "decisions.jsonl"), "utf8"), "retained executed decisions\n");
  assert.ok(!(await readdir(output)).includes("attempts"));
  const summary = JSON.parse(await readFile(join(output, "summary.json"), "utf8"));
  assert.deepEqual(summary.groups.at(-1), { setting: "all", total: 1, completed: 1, errors: 0,
    successes: 0, successRate: 0, fullPanelSuccessRate: 0 });
});

test("output budget exhaustion is a failed policy outcome without discarding its trace or usage", () => {
  const partial = { reason: "error", success: null, steps: 35, durationMs: 894369,
    usage: { total_tokens: 338375 }, error: { code: "output_limit", message: "Truncated output" } };
  const result = classifyPolicyOutcome(partial);
  assert.equal(result.reason, "policy_output_limit");
  assert.equal(result.success, false);
  assert.equal(result.steps, 35);
  assert.equal(result.durationMs, partial.durationMs);
  assert.equal(result.usage, partial.usage);
  assert.equal(result.error, partial.error);
  const cli = classifyPolicyOutcome({ ...partial, protocol: "rea-bce-vla-code-v2",
    codeProtocol: { version: 2, backend: "codex", maxOutputTokens: null } });
  assert.equal(cli.reason, "policy_output_limit");
  assert.doesNotMatch(cli.outcomeClassification, /2048|fixed/);
  const infrastructure = { ...partial, error: { code: "upstream_503" } };
  assert.equal(classifyPolicyOutcome(infrastructure), infrastructure);
});

test("skip-errors keeps existing unknown outcomes and continues after a new infrastructure failure", async context => {
  const root = await mkdtemp(join(tmpdir(), "robotwin-skip-errors-"));
  context.after(() => rm(root, { recursive: true, force: true }));
  const output = join(root, "run");
  const rows = [0, 1, 2].map(index => ({ ...instance, instance_id: `scene-${index}` }));
  const existing = { instance: rows[0], mode: "vla", reason: "infrastructure_error", success: null,
    steps: 20, error: "temporary provider error" };
  const terminal = { instance: rows[2], mode: "vla", reason: "environment_limit", success: false, steps: 40 };
  for (const result of [existing, terminal]) {
    const directory = join(output, result.instance.instance_id);
    await mkdir(directory, { recursive: true });
    await writeFile(join(directory, "result.json"), JSON.stringify(result));
  }
  await writeFile(join(root, "config.json"), "{}");
  await writeFile(join(root, "instances.json"), JSON.stringify(rows));
  execFileSync(process.execPath, [fileURLToPath(new URL("../eval/robotwin/hybrid/run.mjs", import.meta.url)),
    "--config", join(root, "config.json"), "--instances", join(root, "instances.json"),
    "--output", output, "--mode", "vla", "--skip-errors"]);
  assert.equal(await readFile(join(output, "scene-0/result.json"), "utf8"), JSON.stringify(existing));
  assert.equal(await readFile(join(output, "scene-2/result.json"), "utf8"), JSON.stringify(terminal));
  assert.equal(JSON.parse(await readFile(join(output, "scene-1/result.json"), "utf8")).success, null);
  const summary = JSON.parse(await readFile(join(output, "summary.json"), "utf8"));
  assert.equal(summary.queueExhausted, true);
  assert.equal(summary.processed, 3);
  assert.equal(summary.groups.at(-1).completed, 1);
  assert.equal(summary.groups.at(-1).errors, 2);
  assert.equal(summary.groups.at(-1).fullPanelSuccessRate, null);
  assert.deepEqual(summary.skippedErrors.map(row => row.instance_id), ["scene-0", "scene-1"]);
  assert.ok(!(await readdir(output)).includes("attempts"));
});

function simulatorFixture({ nativeLimit = 1000, successAt = null, rejectCodeActions = false } = {}) {
  let count = 0;
  let decisions = 0;
  const observation = () => ({
    id: `frame-${decisions}`, text: instance.instruction,
    actions: ["move_gripper", "invalid_output", "robotwin_vla_chunk"].map(name => ({ name, parameters: { type: "object" } })),
    images: ["head_camera", "left_camera", "right_camera"].map(name => ({ name, data: "fixture", mimeType: "image/png" })),
    state: { vla: { joint_action: { vector: Array(14).fill(count) }, endpose: {} } },
  });
  const result = () => ({
    observation: observation(), success: successAt !== null && count >= successAt,
    terminated: successAt !== null && count >= successAt,
    truncated: count >= nativeLimit && !(successAt !== null && count >= successAt),
    info: { nativeActionCount: count, nativeStepLimit: nativeLimit,
      ...(rejectCodeActions && decisions ? { execution: { status: "invalid_action", error: "Position outside world coordinate bounds [-2,2] metres", native_action_counter_delta: 0 } } : {}) },
  });
  return {
    requests: [], stops: 0,
    stop() { this.stops++; },
    async request(payload) {
      this.requests.push(structuredClone(payload));
      if (payload.op === "reset") { count = 0; decisions = 0; }
      if (payload.op === "step") {
        decisions++;
        const actionCount = payload.action.name === "robotwin_vla_chunk" ? payload.action.arguments.commands.length : payload.action.name === "invalid_output" || rejectCodeActions ? 0 : 1;
        count = Math.min(nativeLimit, count + actionCount, successAt ?? Infinity);
      }
      return { ok: true, result: payload.op === "close" ? { closed: true } : result() };
    },
  };
}

function codeFixture(error = null) {
  const budgets = [];
  const generator = { decide: async input => {
    budgets.push(structuredClone(input.budget));
    return { action: { name: "run_python", arguments: { code: "move_gripper('left', [0, 0, 1], [1, 0, 0, 0], 1)" } } };
  } };
  const worker = {
    requests: [], stop() {},
    async request(payload) {
      this.requests.push(payload);
      return error ? { ok: false, error } : { ok: true, result: { action: { arm: "left", position: [0, 0, 1], quaternion: [1, 0, 0, 0], opening: 1 } } };
    },
  };
  return { policy: withCodeDecisionBudget(createRobotwinCodePolicy({ generator, worker }), config.codeDecisions), worker, budgets };
}

test("hybrid code selection dispatches Python actions and ends at the historical 40 decisions", async context => {
  const simulator = simulatorFixture();
  const environment = createHybridEnvironment({ config, instance, output: "/fixture/run", worker: simulator });
  const { policy, worker, budgets } = codeFixture();
  context.after(() => policy.close());
  let predictions = 0;
  const router = createSuccessRouter({ fetchImpl: async () => {
    predictions++;
    return { ok: true, json: async () => ({ scores: { medex: 0.4, code: 0.9 }, trained_candidates: ["medex", "code"] }) };
  } });
  const result = await runAsyncEpisode({
    environment,
    system0: [{ id: "medex", kind: "vla", policy: { decide: async () => { throw new Error("Unselected VLA must not execute"); } } }, { id: "code", kind: "code", policy }],
    system1: router.route,
    task: instance.task, seed: instance.seed, instruction: instance.instruction,
    maxSteps: 1500,
  });
  assert.equal(result.reason, "environment_limit");
  assert.equal(result.steps, 40);
  assert.equal(result.info.codeDecisionLimit, 40);
  assert.equal(predictions, 1);
  assert.equal(worker.requests.length, 40);
  assert.equal(budgets[0].stepsRemaining, 40);
  assert.equal(budgets[39].stepsRemaining, 1);
  assert.deepEqual(simulator.requests[0], { op: "reset", subset: config.subset, harnessRoot: config.harnessRoot, instance, output: "/fixture/run" });
  assert.equal(simulator.requests.filter(request => request.op === "step").length, 40);
  assert.equal(simulator.stops, 1);
});

test("Python execution errors consume code decisions without native robot motion", async context => {
  const simulator = simulatorFixture();
  const environment = createHybridEnvironment({ config, instance, output: "/fixture/run", worker: simulator });
  const { policy } = codeFixture("ValueError: invalid target");
  context.after(() => policy.close());
  const result = await runAsyncEpisode({ environment, system0: [{ id: "code", policy }], maxSteps: 1500 });
  assert.equal(result.reason, "environment_limit");
  assert.equal(result.steps, 40);
  assert.equal(result.info.nativeActionCount, 0);
  assert.equal(result.info.codeDecisionLimit, 40);
  assert.ok(simulator.requests.filter(request => request.op === "step").every(request => request.action.name === "invalid_output"));
});

test("a rejected gripper target consumes the code budget and feeds validation feedback to the next decision", async context => {
  const simulator = simulatorFixture({ rejectCodeActions: true });
  const environment = createHybridEnvironment({ config, instance, output: "/fixture/run", worker: simulator });
  const inputs = [];
  const policy = withCodeDecisionBudget({ async decide(input) {
    inputs.push(input);
    return { action: { name: "move_gripper", arguments: {
      arm: "left", position: [0.9608119225189554, 2.7543182725534407, 0.43],
      quaternion: [0.7000007, -7.1e-6, -3.8e-6, 0.7141421], opening: 0,
    } } };
  } }, config.codeDecisions);
  const result = await runAsyncEpisode({ environment, system0: [{ id: "code", policy }], maxSteps: 1500 });
  assert.equal(result.reason, "environment_limit");
  assert.equal(result.success, false);
  assert.equal(result.steps, 40);
  assert.equal(result.info.nativeActionCount, 0);
  assert.equal(inputs[39].budget.stepsRemaining, 1);
  assert.match(JSON.stringify(inputs[1].feedback), /Position outside world coordinate bounds/);
  assert.equal(simulator.requests.filter(request => request.op === "step").length, 40);
});

test("recovery replays the recorded first decision and leaves 39 fresh decisions with its feedback", async context => {
  const directory = await mkdtemp(join(tmpdir(), "robotwin-code-replay-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const replay = { instanceId: instance.instance_id, decisionFile: join(directory, "decisions.jsonl") };
  const saved = { step: 0, candidate: "code", observationId: `${instance.instance_id}:0`,
    action: { name: "move_gripper", arguments: { arm: "left", position: [0.9608119225189554, 2.7543182725534407, 0.43], quaternion: [1, 0, 0, 0], opening: 0 } },
    notes: "Revisit the target geometry.", code: { source: "recorded Python", error: null },
    usage: { prompt_tokens: 100, completion_tokens: 10, total_tokens: 110 }, model: "gpt-6-astra",
  };
  await writeFile(replay.decisionFile, JSON.stringify(saved) + "\n");
  const inputs = [];
  const live = { async decide(input) { inputs.push(input); return { action: saved.action }; } };
  const wrapped = withCodeReplay(live, instance.instance_id, replay);
  const replayed = await wrapped.decide({ step: 0 });
  assert.equal(inputs.length, 0);
  assert.deepEqual(replayed.action, saved.action);
  assert.deepEqual(replayed.usage, saved.usage);
  assert.deepEqual(replayed.replay, { ...replay, sourceStep: 0, apiCalled: false, pythonExecuted: false });
  assert.equal(withCodeReplay(live, "another-instance", replay), live);
  const simulator = simulatorFixture({ rejectCodeActions: true });
  const result = await runAsyncEpisode({
    environment: createHybridEnvironment({ config, instance, output: "/fixture/run", worker: simulator }),
    system0: [{ id: "code", policy: withCodeDecisionBudget(wrapped, 40) }], maxSteps: 1500,
  });
  assert.equal(result.reason, "environment_limit");
  assert.equal(result.steps, 40);
  assert.equal(result.info.nativeActionCount, 0);
  assert.equal(inputs.length, 39);
  assert.equal(inputs[0].step, 1);
  assert.equal(inputs[0].budget.stepsRemaining, 39);
  assert.equal(inputs.at(-1).budget.stepsRemaining, 1);
  assert.match(JSON.stringify(inputs[0].feedback), /Position outside world coordinate bounds/);
  await writeFile(replay.decisionFile, JSON.stringify({ ...saved, observationId: "another-instance:0" }) + "\n");
  await assert.rejects(wrapped.decide({ step: 0 }), /this instance's recorded first code decision/);
});

test("success on code decision 40 remains success", async context => {
  const simulator = simulatorFixture({ successAt: 40 });
  const environment = createHybridEnvironment({ config, instance, output: "/fixture/run", worker: simulator });
  const { policy } = codeFixture();
  context.after(() => policy.close());
  const result = await runAsyncEpisode({ environment, system0: [{ id: "code", policy }], maxSteps: 1500 });
  assert.equal(result.reason, "success");
  assert.equal(result.steps, 40);
  assert.equal(result.success, true);
  assert.equal(result.truncated, false);
});

test("VLA dispatch uses native chunks until the simulator action limit without consuming code budget", async context => {
  const simulator = simulatorFixture({ nativeLimit: 35 });
  const environment = createHybridEnvironment({ config: { ...config, codeDecisions: 1 }, instance, output: "/fixture/run", worker: simulator });
  const worker = {
    requests: [], stop() {},
    async request(payload) {
      this.requests.push(payload);
      return { ok: true, result: payload.op === "infer" ? { actions: Array.from({ length: 16 }, () => Array(14).fill(0.2)), execute_horizon: 16 } : {} };
    },
  };
  const nativePolicy = createRobotwinVlaPolicy({ model: "medex", url: "ws://fixture:8906", worker });
  const budgets = [];
  const policy = { ...nativePolicy, async decide(input) {
    budgets.push(input.budget.stepsRemaining);
    return nativePolicy.decide(input);
  } };
  context.after(() => policy.close());
  const result = await runAsyncEpisode({ environment, system0: [{ id: "medex", kind: "vla", policy }], maxSteps: 1500 });
  assert.equal(result.reason, "environment_limit");
  assert.equal(result.steps, 3);
  assert.equal(result.info.nativeActionCount, 35);
  assert.equal(result.info.codeDecisionLimit, undefined);
  assert.equal(worker.requests.filter(request => request.op === "infer").length, 3);
  assert.deepEqual(budgets, [1500, 1499, 1498]);
  assert.ok(simulator.requests.filter(request => request.op === "step").every(request => request.action.arguments.actionType === "joint"));
});

test("simulator failures remain unknown outcomes rather than completed benchmark failures", async context => {
  const simulator = simulatorFixture();
  const original = simulator.request;
  simulator.request = async function (payload) {
    return payload.op === "step" ? { ok: false, error: "Simulator disconnected" } : original.call(this, payload);
  };
  const environment = createHybridEnvironment({ config, instance, output: "/fixture/run", worker: simulator });
  const { policy } = codeFixture();
  context.after(() => policy.close());
  const result = await runAsyncEpisode({ environment, system0: [{ id: "code", policy }] });
  assert.equal(result.reason, "error");
  assert.equal(result.success, null);
  assert.match(result.error.message, /Simulator disconnected/);
});
