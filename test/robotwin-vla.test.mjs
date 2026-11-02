import assert from "node:assert/strict";
import test from "node:test";

import { createRobotwinVlaPolicy } from "../src/policies/robotwin-vla.mjs";

function observation(value = 0, nativeNames = false) {
  return {
    id: `frame-${value}`,
    text: "Place the cube in the tray.",
    images: [
      { name: nativeNames ? "right_camera" : "right_wrist", data: "right", mimeType: "image/png" },
      { name: nativeNames ? "head_camera" : "head", data: "head", mimeType: "image/png" },
      { name: nativeNames ? "left_camera" : "left_wrist", data: "left", mimeType: "image/png" },
    ],
    state: { vla: {
      joint_action: { vector: Array(14).fill(value), left_arm: Array(6).fill(value), right_arm: Array(6).fill(value), left_gripper: 1, right_gripper: 1 },
      endpose: { left_endpose: [0, 0, 0, 1, 0, 0, 0], right_endpose: [0, 0, 0, 1, 0, 0, 0], left_gripper: 1, right_gripper: 1 },
    } },
  };
}

function workerFixture(infer = () => ({ actions: [Array(14).fill(0.25)], execute_horizon: 1 }), metadata = {}) {
  return {
    requests: [], stops: 0,
    stop() { this.stops++; },
    async request(payload) {
      this.requests.push(structuredClone(payload));
      return { ok: true, result: payload.op === "infer" ? await infer(payload) : payload.op === "configure" ? metadata : { reset: true } };
    },
  };
}

test("native VLA preserves named images, robot state, instruction and per-episode seed", async context => {
  const worker = workerFixture();
  const policy = createRobotwinVlaPolicy({ model: "medex", url: "ws://localhost:9000", worker });
  context.after(() => policy.close());
  await policy.reset();
  assert.equal(worker.requests.length, 0);
  const current = observation(0.5);
  const decision = await policy.decide({ observation: current, instruction: "Complete the task efficiently" });
  assert.deepEqual(worker.requests.map(request => request.op), ["configure", "reset", "infer"]);
  assert.deepEqual(worker.requests[1], { op: "reset", policy_seed: 42 });
  assert.deepEqual(worker.requests[2], {
    op: "infer", instruction: current.text,
    joint_action: current.state.vla.joint_action, endpose: current.state.vla.endpose,
    images: {
      head_camera: { data: "head", mimeType: "image/png" },
      left_camera: { data: "left", mimeType: "image/png" },
      right_camera: { data: "right", mimeType: "image/png" },
    },
  });
  assert.deepEqual(decision.action, { name: "robotwin_vla_chunk", arguments: { commands: [Array(14).fill(0.25)], actionType: "joint" } });
  await policy.reset();
  await policy.decide({ observation: observation(1, true) });
  assert.deepEqual(worker.requests.map(request => request.op), ["configure", "reset", "infer", "reset", "infer"]);
});

test("seven model action formats retain native dimensions and execution horizons", async context => {
  const horizons = { pi05: 50, gigabrain: 30, spatial_forcing: 50, xvla: 30, xwam: 32, patchwam: 16, medex: 16 };
  for (const [model, horizon] of Object.entries(horizons)) {
    const dimension = ["xvla", "xwam"].includes(model) ? 16 : 14;
    const worker = workerFixture(() => ({ actions: Array.from({ length: 50 }, (_, index) => Array(dimension).fill(index + 2)), execute_horizon: horizon, inference_s: 0.1 }));
    const policy = createRobotwinVlaPolicy({ model, url: "ws://localhost:9000", worker });
    context.after(() => policy.close());
    const decision = await policy.decide({ observation: observation() });
    assert.equal(decision.action.arguments.commands.length, horizon, model);
    assert.equal(decision.action.arguments.actionType, dimension === 16 ? "ee" : "joint", model);
    assert.equal(decision.action.arguments.commands[0][0], 2);
    assert.equal(decision.vla.inferenceS, 0.1);
  }
});

test("configured chunk limit shortens the server horizon and fresh observations trigger fresh inference", async context => {
  const worker = workerFixture(payload => ({ actions: Array.from({ length: 5 }, () => payload.joint_action.vector), execute_horizon: 4 }));
  const policy = createRobotwinVlaPolicy({ model: "pi05", url: "ws://localhost:9000", worker, chunkSteps: 2 });
  context.after(() => policy.close());
  assert.deepEqual((await policy.decide({ observation: observation(1) })).action.arguments.commands, [Array(14).fill(1), Array(14).fill(1)]);
  assert.deepEqual((await policy.decide({ observation: observation(2) })).action.arguments.commands, [Array(14).fill(2), Array(14).fill(2)]);
  assert.equal(worker.requests.filter(request => request.op === "infer").length, 2);
});

test("service metadata supplies the native action contract when prediction omits it", async context => {
  const worker = workerFixture(() => ({ actions: [Array(16).fill(0.2), Array(16).fill(0.3)] }), { action_type: "ee", execute_horizon: 1 });
  const policy = createRobotwinVlaPolicy({ model: "xvla", url: "ws://localhost:9000", worker });
  context.after(() => policy.close());
  const decision = await policy.decide({ observation: observation() });
  assert.equal(decision.action.arguments.actionType, "ee");
  assert.equal(decision.action.arguments.commands.length, 1);
});

test("incompatible native action dimensions report an error before dispatch", async context => {
  const worker = workerFixture(() => ({ actions: [Array(14).fill(0.2)], execute_horizon: 1 }));
  const policy = createRobotwinVlaPolicy({ model: "xwam", url: "ws://localhost:9000", worker });
  context.after(() => policy.close());
  await assert.rejects(policy.decide({ observation: observation() }), /16-dimensional ee actions/);
});

test("cancelling native inference closes its worker and preserves the cancellation reason", async context => {
  let entered;
  let rejectInference;
  const started = new Promise(resolve => { entered = resolve; });
  const worker = workerFixture(() => new Promise((resolve, reject) => { rejectInference = reject; entered(); }));
  worker.stop = function () { this.stops++; rejectInference?.(new Error("worker stopped")); };
  const policy = createRobotwinVlaPolicy({ model: "medex", url: "ws://localhost:9000", worker });
  context.after(() => policy.close());
  const controller = new AbortController();
  const pending = policy.decide({ observation: observation(), signal: controller.signal });
  await started;
  controller.abort(new Error("episode cancelled"));
  await assert.rejects(pending, /episode cancelled/);
  assert.equal(worker.stops, 1);
});
