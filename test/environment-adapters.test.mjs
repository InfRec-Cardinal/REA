import assert from "node:assert/strict";
import test from "node:test";

import { createG1Environment } from "../src/environments/g1.mjs";
import { createLiberoEnvironment } from "../src/environments/libero.mjs";
import { createExecutor, createNullActuator } from "../src/executor.mjs";
import { ACTION_TABLE as LIBERO_ACTION_TABLE } from "../src/libero-actions.mjs";

function g1Fixture({ depths = [3, 3, 3], execute = false, nullActuator = false, evaluate = null, waitForCompletion = null, dispatch = { ok: true }, onMove = () => {} } = {}) {
  const captures = [];
  const moves = [];
  const stops = [];
  const camera = {
    async capture(options) {
      const index = captures.length;
      captures.push(options);
      return {
        imageB64: "aW1hZ2U=",
        mimeType: "image/jpeg",
        capturedAt: 1_700_000_000 + index,
        depth: { center_m: depths[Math.min(index, depths.length - 1)], left_m: 3, right_m: 3 },
        privateCameraData: "not policy state",
      };
    },
  };
  const actuator = nullActuator ? createNullActuator() : {
    status: async () => ({ ok: true, ready: true, reasons: [] }),
    move: async motion => { moves.push(motion); onMove(motion); return dispatch; },
    arm: async () => dispatch,
    stop: async () => { stops.push(true); return { ok: true }; },
  };
  const config = { useLidar: false, executeEnabled: true, maxPulseS: 2, depthNearCenterM: 0.55, depthNearSideM: 0.35 };
  const environment = createG1Environment({ camera, executor: createExecutor({ actuator, config }), config, execute, evaluate, waitForCompletion });
  return { environment, captures, moves, stops };
}

test("G1 reset observes without moving and step previews through the real gate by default", async () => {
  const { environment, captures, moves, stops } = g1Fixture({ waitForCompletion: async () => assert.fail("preview must not wait for motion") });
  assert.equal(environment.timing, "realtime");
  const initial = await environment.reset({ task: "approach the doorway", seed: 7 });
  assert.equal(initial.info.physicalReset, false);
  assert.equal(initial.success, null);
  assert.equal(initial.observation.timestampMs, 1_700_000_000_000);
  assert.equal(initial.observation.text, "approach the doorway");
  assert.deepEqual(initial.observation.images, [{ mimeType: "image/jpeg", data: "aW1hZ2U=" }]);
  assert.equal(JSON.stringify(initial.observation.state).includes("privateCameraData"), false);
  const forward = initial.observation.actions.find(action => action.name === "W");
  assert.equal(forward.parameters.properties.durationMs.maximum, 2000);
  assert.equal(forward.parameters.additionalProperties, false);
  const result = await environment.step({ name: "W", arguments: { durationMs: 600 } }, { observation: initial.observation });
  assert.equal(result.info.execution.gate.allowed, true);
  assert.equal(result.info.execution.preview, true);
  assert.equal(result.info.execution.executed, false);
  assert.equal(result.info.completion, "preview");
  assert.equal(result.success, null);
  assert.equal(result.terminated, false);
  assert.equal(result.info.actionsAccepted, 0);
  assert.equal(captures.length, 3);
  assert.equal(captures[1].depthOnly, true);
  assert.notEqual(result.observation.id, initial.observation.id);
  await environment.stop();
  assert.equal(moves.length, 0);
  assert.equal(stops.length, 0);
  await environment.close();
});

test("G1 rechecks live depth and retains missing-actuator readiness", async () => {
  const fixture = g1Fixture({ depths: [3, 0.3, 0.3], execute: true });
  const initial = await fixture.environment.reset({ task: "approach" });
  const result = await fixture.environment.step({ name: "W", arguments: {} }, { observation: initial.observation });
  assert.equal(result.info.execution.gate.allowed, false);
  assert.equal(result.info.execution.error_type, "gate_refused");
  assert.equal(fixture.moves.length, 0);
  assert.equal(fixture.stops.length, 1);
  const missing = g1Fixture({ nullActuator: true });
  const missingInitial = await missing.environment.reset({ task: "approach" });
  assert.equal(missingInitial.observation.state.robot.ready, false);
  const missingStep = await missing.environment.step({ name: "W", arguments: {} });
  assert.equal(missingStep.info.execution.gate.allowed, false);
  assert.equal(missingStep.success, null);
});

test("G1 accepts task outcomes only from evaluate and bounds velocity durations", async () => {
  const evaluations = [];
  const fixture = g1Fixture({
    execute: true,
    waitForCompletion: async () => {},
    evaluate: async input => {
      evaluations.push(input);
      return { terminated: true, success: true, reward: 1, info: { source: "external task check" } };
    },
  });
  const initial = await fixture.environment.reset({ task: "external task" });
  await assert.rejects(fixture.environment.step({ name: "W", arguments: { durationMs: 3000 } }), /durationMs/);
  const result = await fixture.environment.step({ name: "W", arguments: { durationMs: 700 } }, { observation: initial.observation, reasoning: "move ahead" });
  assert.equal(fixture.moves[0].duration, 0.7);
  assert.equal(result.info.actionsAccepted, 1);
  assert.equal(result.info.totalActionsAccepted, 1);
  assert.equal(result.terminated, true);
  assert.equal(result.success, true);
  assert.equal(result.reward, 1);
  assert.equal(evaluations[0].reasoning, "move ahead");
  assert.equal(evaluations[0].before.id, initial.observation.id);
  assert.equal(evaluations[0].observation.id, result.observation.id);
  assert.equal(evaluations[0].execution.executed, true);
});

test("G1 captures after the completion hook and before evaluating the task", async () => {
  const waiting = Promise.withResolvers();
  const release = Promise.withResolvers();
  const completed = [];
  const fixture = g1Fixture({
    execute: true,
    waitForCompletion: async request => {
      assert.equal(request.action.name, "W");
      assert.equal(request.execution.executed, true);
      assert.equal(fixture.captures.length, 2);
      waiting.resolve();
      await release.promise;
      completed.push("wait");
    },
    evaluate: async () => {
      assert.equal(fixture.captures.length, 3);
      assert.deepEqual(completed, ["wait"]);
      completed.push("evaluate");
      return { success: null };
    },
  });
  await fixture.environment.reset({ task: "approach" });
  const pending = fixture.environment.step({ name: "W", arguments: { durationMs: 2000 } });
  await waiting.promise;
  assert.equal(fixture.captures.length, 2);
  release.resolve();
  const result = await pending;
  assert.equal(result.info.completion, "completion_hook");
  assert.equal(result.success, null);
  assert.deepEqual(completed, ["wait", "evaluate"]);
});

test("G1 waits for the bounded pulse window without declaring task completion", async () => {
  const fixture = g1Fixture({ execute: true });
  await fixture.environment.reset({ task: "approach" });
  const started = performance.now();
  const result = await fixture.environment.step({ name: "W", arguments: { durationMs: 400 } });
  assert.ok(performance.now() - started >= 440);
  assert.equal(result.info.completion, "pulse_elapsed");
  assert.equal(result.success, null);
  assert.equal(result.terminated, false);
  assert.equal(fixture.captures.length, 3);
});

test("G1 cancellation interrupts pulse waiting, stops motion, and skips the after image", async () => {
  const accepted = Promise.withResolvers();
  const fixture = g1Fixture({ execute: true, onMove: () => accepted.resolve() });
  const controller = new AbortController();
  await fixture.environment.reset({ task: "approach" });
  const pending = fixture.environment.step({ name: "W", arguments: { durationMs: 2000 } }, { signal: controller.signal });
  const rejected = assert.rejects(pending, { name: "AbortError" });
  await accepted.promise;
  await new Promise(resolve => setImmediate(resolve));
  controller.abort();
  await rejected;
  assert.equal(fixture.moves.length, 1);
  assert.equal(fixture.stops.length, 1);
  assert.equal(fixture.captures.length, 2);
});

test("G1 accepts an explicit cerebellum terminal receipt without another pulse delay", async () => {
  const fixture = g1Fixture({ execute: true, dispatch: { ok: true, kind: "EXECUTION_COMPLETE", intent_id: "intent-1" } });
  const controller = new AbortController();
  await fixture.environment.reset({ task: "approach" });
  const pending = fixture.environment.step({ name: "W", arguments: { durationMs: 2000 } }, { signal: controller.signal });
  const interrupt = setTimeout(() => controller.abort(), 100);
  try {
    const result = await pending;
    assert.equal(result.info.completion, "execution_complete");
    assert.equal(result.success, null);
    assert.equal(fixture.captures.length, 3);
  } finally {
    clearTimeout(interrupt);
  }
});

test("G1 arm acknowledgement remains distinct from observed task completion", async () => {
  const fixture = g1Fixture({ execute: true });
  await fixture.environment.reset({ task: "push" });
  const result = await fixture.environment.step({ name: "P", arguments: {} });
  assert.equal(result.info.completion, "acknowledged");
  assert.equal(result.info.execution.executed, true);
  assert.equal(result.success, null);
});

function liberoResponse({ observationNumber = 1, steps = 0, success = false, terminated = false, horizon = 10 } = {}) {
  return {
    frame: {
      imageB64: "bW9udGFnZQ==",
      mimeType: "image/jpeg",
      capturedAt: "2026-09-08T01:00:00.000Z",
      libero: {
        observationId: `1:${observationNumber}`,
        task: { suite: "libero_10_task", index: 2, language: "put the bowl on the plate" },
        proprio: { eefPosition: [0, 0, 0.4] },
      },
    },
    steps, success, terminated, horizon,
    repeatSteps: 5,
    amplitude: 0.5,
    initialStateSource: "official_pruned_heldout",
    initialStateIndex: 3,
  };
}

function liberoFixture(responses) {
  const requests = [];
  let closed = false;
  const client = {
    async request(command, options) {
      requests.push({ command, options });
      return responses.shift();
    },
    async close() { closed = true; },
  };
  return { environment: createLiberoEnvironment({ client }), requests, isClosed: () => closed };
}

test("LIBERO forwards episode settings and uses montage, proprioception, and simulator outcomes", async () => {
  const fixture = liberoFixture([
    liberoResponse(),
    liberoResponse({ observationNumber: 2, steps: 3, terminated: true, success: true }),
  ]);
  assert.equal(fixture.environment.timing, "turn-based");
  const task = { suite: "libero_10_task", task: 2, horizon: 10, repeatSteps: 5, amplitude: 0.5 };
  const initial = await fixture.environment.reset({ task, seed: 1003 });
  assert.deepEqual(fixture.requests[0], { command: "reset", options: { ...task, seed: 1003 } });
  assert.equal(initial.info.controlSteps, 0);
  assert.equal(initial.observation.timestampMs, Date.parse("2026-09-08T01:00:00.000Z"));
  assert.deepEqual(initial.observation.state.proprio, { eefPosition: [0, 0, 0.4] });
  assert.deepEqual(initial.observation.images, [{ mimeType: "image/jpeg", data: "bW9udGFnZQ==" }]);
  assert.equal(initial.observation.actions.length, 15);
  assert.equal(initial.observation.actions.find(action => action.name === "R").description, LIBERO_ACTION_TABLE.R.description);
  assert.equal(LIBERO_ACTION_TABLE.R.action, "world_z_negative");
  assert.ok(LIBERO_ACTION_TABLE.R.description.includes("world -z"));
  for (const action of initial.observation.actions) assert.equal(action.parameters.additionalProperties, false);
  const result = await fixture.environment.step({ name: "O", arguments: { repeatSteps: 3, amplitude: 0.2 } }, { observation: initial.observation });
  assert.deepEqual(fixture.requests[1], { command: "step", options: { token: "O", repeatSteps: 3, amplitude: 0.2 } });
  assert.equal(result.terminated, true);
  assert.equal(result.truncated, false);
  assert.equal(result.success, true);
  assert.equal(result.reward, null);
  assert.equal(result.info.controlSteps, 3);
  assert.equal(result.info.totalControlSteps, 3);
  await fixture.environment.close();
  assert.equal(fixture.isClosed(), true);
});

test("LIBERO distinguishes horizon exhaustion from success and rejects stale observations", async () => {
  const fixture = liberoFixture([
    liberoResponse({ horizon: 5 }),
    liberoResponse({ observationNumber: 2, steps: 2, horizon: 5 }),
    liberoResponse({ observationNumber: 3, steps: 5, horizon: 5, terminated: true }),
  ]);
  const initial = await fixture.environment.reset({ task: { suite: "libero_10_task", task: 2 }, seed: 1003 });
  const middle = await fixture.environment.step({ name: "X", arguments: { repeatSteps: 2 } }, { observation: initial.observation });
  assert.equal(middle.success, false);
  assert.equal(middle.terminated, false);
  await assert.rejects(fixture.environment.step({ name: "W", arguments: {} }, { observation: initial.observation }), /current observation/);
  assert.equal(fixture.requests.length, 2);
  const final = await fixture.environment.step({ name: "W", arguments: {} }, { observation: middle.observation });
  assert.equal(final.terminated, false);
  assert.equal(final.truncated, true);
  assert.equal(final.success, false);
  assert.equal(final.info.controlSteps, 3);
  assert.equal(final.info.totalControlSteps, 5);
  assert.equal(final.info.backendTerminated, true);
});

test("LIBERO preserves an early backend termination and observe advances only observation identity", async () => {
  const fixture = liberoFixture([
    liberoResponse(),
    liberoResponse({ observationNumber: 2 }),
    liberoResponse({ observationNumber: 3, steps: 2, terminated: true }),
  ]);
  const initial = await fixture.environment.reset({ task: { suite: "libero_10_task", task: 2 }, seed: 1003 });
  const observed = await fixture.environment.observe();
  assert.notEqual(observed.id, initial.observation.id);
  assert.equal(observed.state.controlSteps, 0);
  assert.equal(fixture.requests[1].command, "observe");
  const result = await fixture.environment.step({ name: "X", arguments: {} }, { observation: observed });
  assert.equal(result.success, false);
  assert.equal(result.terminated, true);
  assert.equal(result.truncated, false);
  await assert.rejects(fixture.environment.step({ name: "X", arguments: {} }), /finished/);
});
