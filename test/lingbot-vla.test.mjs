import assert from "node:assert/strict";
import test from "node:test";

import { createLingBotVlaPolicy } from "../src/policies/lingbot-vla.mjs";
import { createPolicy, listPolicies } from "../src/framework/registry.mjs";
import { runAsyncEpisode } from "../src/framework/async-runtime.mjs";

const TRAJECTORY = {
  name: "rpent_qpos_trajectory",
  parameters: {
    type: "object",
    properties: { commands: { type: "array", items: { type: "array", items: { type: "number" } } } },
    required: ["commands"],
    additionalProperties: false,
  },
};

function observation(value = 0) {
  return {
    id: `frame-${value}`,
    text: "Place the cube in the tray.",
    state: { robot_state: { qpos_target14: Array(14).fill(value) } },
    // Camera order deliberately differs from the model's feature order.
    images: [
      { name: "right_wrist", data: "cmlnaHQ=", mimeType: "image/jpeg" },
      { name: "head", data: "aGVhZA==", mimeType: "image/png" },
      { name: "left_wrist", data: "bGVmdA==", mimeType: "image/png" },
    ],
    actions: [TRAJECTORY],
  };
}

function workerFixture(infer = () => ({ action: [Array(14).fill(0.1)] })) {
  return {
    requests: [],
    stops: 0,
    health() { return { running: this.stops === 0 }; },
    stop() { this.stops += 1; },
    async request(payload, timeoutMs) {
      this.requests.push({ payload: structuredClone(payload), timeoutMs });
      return { ok: true, result: payload.op === "infer" ? await infer(payload) : {} };
    },
  };
}

test("LingBot uses named RoboTwin cameras, target joints and the official flat feature keys", async context => {
  const worker = workerFixture(() => ({ action: [Array(14).fill(0.25)], server_timing: { inference_ms: 12 } }));
  const policy = createLingBotVlaPolicy({ worker, env: {} });
  context.after(() => policy.close());
  assert.equal(worker.requests.length, 0);
  const current = observation(0.5);
  const decision = await policy.decide({ observation: current, instruction: current.text });
  assert.deepEqual(worker.requests.map(item => item.payload.op), ["configure", "reset", "infer"]);
  assert.deepEqual(worker.requests[0].payload, { op: "configure", url: "ws://127.0.0.1:8006", robot: "robotwin", timeoutMs: 30000 });
  assert.deepEqual(worker.requests[2].payload, {
    op: "infer",
    images: {
      "observation.images.cam_high": { data: "aGVhZA==", mimeType: "image/png" },
      "observation.images.cam_left_wrist": { data: "bGVmdA==", mimeType: "image/png" },
      "observation.images.cam_right_wrist": { data: "cmlnaHQ=", mimeType: "image/jpeg" },
    },
    states: { "observation.state": Array(14).fill(0.5) },
    task: current.text,
  });
  assert.deepEqual(decision.action, { name: TRAJECTORY.name, arguments: { commands: [Array(14).fill(0.25)] } });
  assert.equal(decision.model, "lingbot-vla-2.0");
  assert.deepEqual(decision.vla, { controlSteps: 1, serverTiming: { inference_ms: 12 } });
});

test("LingBot preserves the environment task and appends distinct controller instructions", async context => {
  const worker = workerFixture();
  const policy = createLingBotVlaPolicy({ worker });
  context.after(() => policy.close());
  const current = observation();
  await policy.decide({ observation: current, instruction: "Align the left gripper first." });
  await policy.decide({ observation: current, instruction: current.text });
  await policy.decide({ observation: current });
  assert.deepEqual(worker.requests.filter(item => item.payload.op === "infer").map(item => item.payload.task), [
    `${current.text}\n\nAlign the left gripper first.`, current.text, current.text,
  ]);
});

test("custom feature maps support nested paths and literal dotted state keys", async context => {
  const worker = workerFixture();
  const policy = createLingBotVlaPolicy({
    worker,
    robot: "custom_robot",
    imageMap: { "observation.images.front": "overhead" },
    stateMap: {
      "observation.state.arm": "state.arm.joints",
      "observation.state.gripper": ["state", "gripper.position"],
    },
  });
  context.after(() => policy.close());
  const current = {
    text: "Pick up the cup.",
    images: [{ name: "overhead", data: "ZnJvbnQ=", mimeType: "image/png" }],
    state: { arm: { joints: [0.2, 0.4] }, "gripper.position": [0.8] },
  };
  await policy.decide({ observation: current });
  assert.equal(worker.requests[0].payload.robot, "custom_robot");
  assert.deepEqual(worker.requests.at(-1).payload.states, { "observation.state.arm": [0.2, 0.4], "observation.state.gripper": [0.8] });
  assert.deepEqual(worker.requests.at(-1).payload.images, { "observation.images.front": { data: "ZnJvbnQ=", mimeType: "image/png" } });
});

test("each action boundary infers from fresh state and discards the unused prediction tail", async context => {
  const worker = workerFixture(payload => ({ action: Array.from({ length: 5 }, (_, index) => [payload.states["observation.state"][0], index]) }));
  const policy = createLingBotVlaPolicy({ worker, chunkSteps: 2 });
  context.after(() => policy.close());
  assert.deepEqual((await policy.decide({ observation: observation(1) })).action.arguments.commands, [[1, 0], [1, 1]]);
  assert.deepEqual((await policy.decide({ observation: observation(2) })).action.arguments.commands, [[2, 0], [2, 1]]);
  assert.equal(worker.requests.filter(item => item.payload.op === "infer").length, 2);
});

test("episode reset clears remote policy state while retaining its configured connection", async context => {
  const worker = workerFixture();
  const policy = createLingBotVlaPolicy({ worker });
  context.after(() => policy.close());
  await policy.reset();
  await policy.decide({ observation: observation(1) });
  await policy.reset();
  await policy.decide({ observation: observation(2) });
  assert.deepEqual(worker.requests.map(item => item.payload.op), ["configure", "reset", "infer", "reset", "infer"]);
});

test("split output features concatenate in configured order before mapping named joints", async context => {
  const worker = workerFixture(() => ({ "action.left": [[1, 2], [3, 4]], "action.right": [[5, 6], [7, 8]] }));
  const policy = createLingBotVlaPolicy({
    worker,
    actionKeys: ["action.right", "action.left"],
    jointNames: ["right_arm", "right_gripper", "left_arm", "left_gripper"],
    actionName: "robot_trajectory",
    actionArgument: "positions",
  });
  context.after(() => policy.close());
  assert.deepEqual((await policy.decide({ observation: observation() })).action, {
    name: "robot_trajectory",
    arguments: { positions: [
      { right_arm: 5, right_gripper: 6, left_arm: 1, left_gripper: 2 },
      { right_arm: 7, right_gripper: 8, left_arm: 3, left_gripper: 4 },
    ] },
  });
});

test("single-action drivers receive one vector from either a flat prediction or a chunk", async context => {
  let calls = 0;
  const worker = workerFixture(() => ({ action: ++calls === 1 ? [0.1, 0.2] : [[0.3, 0.4], [0.5, 0.6]] }));
  const policy = createLingBotVlaPolicy({ worker, actionFormat: "single", actionName: "joint_position", actionArgument: "positions" });
  context.after(() => policy.close());
  assert.deepEqual((await policy.decide({ observation: observation() })).action, { name: "joint_position", arguments: { positions: [0.1, 0.2] } });
  assert.deepEqual((await policy.decide({ observation: observation() })).action, { name: "joint_position", arguments: { positions: [0.3, 0.4] } });
});

test("incompatible state and prediction shapes report the field instead of returning commands", async context => {
  const worker = workerFixture(() => ({ action: [[1, 2], [3]] }));
  const policy = createLingBotVlaPolicy({ worker });
  context.after(() => policy.close());
  const current = observation();
  current.state.robot_state.qpos_target14 = [0, NaN];
  await assert.rejects(policy.decide({ observation: current }), /observation\.state.*numeric vector/);
  assert.equal(worker.requests.length, 0);
  await assert.rejects(policy.decide({ observation: observation() }), /action dimensions differ/);

  const invalidOutput = createLingBotVlaPolicy({ worker: workerFixture(() => ({ action: [[0, Infinity]] })) });
  context.after(() => invalidOutput.close());
  await assert.rejects(invalidOutput.decide({ observation: observation() }), /action.*numeric vector/);
});

test("cancelling pending inference stops its worker and rejects the decision", { timeout: 2000 }, async context => {
  let entered;
  let rejectInference;
  const inferStarted = new Promise(resolve => { entered = resolve; });
  const worker = workerFixture(() => new Promise((_, reject) => { rejectInference = reject; entered(); }));
  worker.stop = function () { this.stops += 1; rejectInference?.(new Error("worker stopped")); };
  const policy = createLingBotVlaPolicy({ worker });
  context.after(() => policy.close());
  const controller = new AbortController();
  const decision = policy.decide({ observation: observation(), signal: controller.signal });
  await inferStarted;
  controller.abort(new Error("episode cancelled"));
  await assert.rejects(decision, /episode cancelled/);
  assert.equal(worker.stops, 1);
});

test("the built-in policy registry defers setup and passes endpoint environment defaults", async context => {
  assert.ok(listPolicies().some(item => item.type === "lingbot-vla"));
  const worker = workerFixture();
  const policy = await createPolicy({ type: "lingbot-vla", options: { worker, requestTimeoutMs: 9000 } }, {
    env: { LINGBOT_VLA_URL: "ws://lingbot.example:8006", LINGBOT_VLA_PYTHON: "/unused/python" },
  });
  context.after(() => policy.close());
  assert.equal(worker.requests.length, 0);
  await policy.reset();
  assert.deepEqual(worker.requests[0], { payload: { op: "configure", url: "ws://lingbot.example:8006", robot: "robotwin", timeoutMs: 9000 }, timeoutMs: 9000 });
  await policy.close();
  assert.equal(worker.stops, 1);
});

test("async System2 switches from SDK to LingBot at the next action boundary with fresh observations", async context => {
  const worker = workerFixture();
  const policy = createLingBotVlaPolicy({ worker, chunkSteps: 2 });
  context.after(() => policy.close());
  const waitAction = { name: "wait", parameters: { type: "object", properties: {}, additionalProperties: false } };
  let finishThinking;
  let step = 0;
  const dispatched = [];
  const current = () => ({ ...observation(step), actions: [waitAction, TRAJECTORY] });
  const environment = {
    id: "robotwin-fixture",
    reset: async () => ({ observation: current() }),
    async step(action) {
      dispatched.push(action);
      step += 1;
      if (step === 1) {
        finishThinking({ candidateId: "lingbot", directive: "Align the left gripper first." });
        await new Promise(resolve => setImmediate(resolve));
      }
      return { observation: current(), terminated: step === 2, success: step === 2 };
    },
  };
  const summary = await runAsyncEpisode({
    environment,
    system0: [
      { id: "sdk", kind: "sdk", policy: { decide: async () => ({ action: { name: "wait", arguments: {} } }) } },
      { id: "lingbot", kind: "vla", policy },
    ],
    system2: { think: async () => new Promise(resolve => { finishThinking = resolve; }) },
    instruction: "Place the cube in the tray.",
    thinkEverySteps: 10,
    maxSteps: 3,
    maxDurationMs: 2000,
  });
  assert.equal(summary.reason, "success", JSON.stringify(summary.error));
  assert.deepEqual(dispatched.map(action => action.name), ["wait", TRAJECTORY.name]);
  assert.equal(summary.asyncMetrics.routeChanges, 1);
  assert.equal(summary.asyncMetrics.adviceApplied, 1);
  const inference = worker.requests.find(item => item.payload.op === "infer").payload;
  assert.deepEqual(inference.states["observation.state"], Array(14).fill(1));
  assert.match(inference.task, /Place the cube in the tray\./);
  assert.match(inference.task, /System2 directive for this action: Align the left gripper first\./);
});
