import assert from "node:assert/strict";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { delimiter } from "node:path";

import { createRobotEnvironment } from "../src/environments/robot.mjs";
import { createAlohaEnvironment } from "../src/environments/aloha.mjs";
import { createLeRobotEnvironment } from "../src/environments/lerobot.mjs";
import { createOpenRALEnvironment } from "../src/environments/openral.mjs";
import { createRPentEnvironment } from "../src/environments/rpent.mjs";
import { runEpisode } from "../src/framework/runtime.mjs";
import { createEnvironment } from "../src/framework/registry.mjs";

function fixtureDriver(names) {
  const calls = [];
  let positions = Object.fromEntries(names.map(name => [name, 0]));
  const action = { name: "robot_command", parameters: { type: "object", properties: { values: {
    type: "object", properties: Object.fromEntries(names.map(name => [name, { type: "number", minimum: -1, maximum: 1 }])), required: names, additionalProperties: false,
  } }, required: ["values"], additionalProperties: false } };
  const snapshot = () => ({ timestampMs: 12345, state: { positions }, images: [{ name: "wrist", mimeType: "image/png", data: "image" }] });
  return {
    calls,
    async describe() { return { actions: [action], metadata: { jointNames: names }, timing: "realtime" }; },
    async reset(context) { calls.push(["reset", context.task]); return snapshot(); },
    async observe() { return snapshot(); },
    async step(command, context) { calls.push(["step", command, context.execute]); if (context.execute) positions = command.arguments.values; return { ...snapshot(), info: { submitted: context.execute } }; },
    async stop() { calls.push(["stop"]); return { supported: false }; },
    async close() { calls.push(["close"]); },
  };
}

test("one robot adapter handles named single-arm and dual-arm commands without core changes", async () => {
  for (const names of [["shoulder", "gripper"], ["left.elbow", "left.gripper", "right.elbow", "right.gripper"]]) {
    const driver = fixtureDriver(names);
    const environment = createRobotEnvironment({ driver, execute: true });
    const result = await runEpisode({ environment, maxSteps: 1, policy: { async decide({ observation }) {
      assert.deepEqual(observation.state.robot.jointNames, names);
      assert.equal(observation.images[0].name, "wrist");
      return { action: { name: "robot_command", arguments: { values: Object.fromEntries(names.map(name => [name, 0.25])) } } };
    } } });
    assert.equal(result.success, null);
    assert.equal(result.reason, "max_steps");
    assert.equal(result.info.submitted, true);
    assert.equal(driver.calls.filter(call => call[0] === "step").length, 1);
    assert.equal(driver.calls.at(-1)[0], "stop");
    await environment.close();
  }
});

test("preview keeps physical state and outcomes separate and rejects stale or invalid joint commands", async () => {
  const driver = fixtureDriver(["joint"]);
  const environment = createRobotEnvironment({ driver });
  const initial = await environment.reset();
  const action = { name: "robot_command", arguments: { values: { joint: 0.5 } } };
  const next = await environment.step(action, { observation: initial.observation });
  assert.equal(next.observation.state.positions.joint, 0);
  assert.equal(next.success, null);
  assert.equal(driver.calls.at(-1)[2], false);
  await assert.rejects(environment.step(action, { observation: initial.observation }), /current observation/u);
  await assert.rejects(environment.step({ ...action, arguments: { values: { joint: 2 } } }), /<= 1/u);
  await environment.stop();
  assert.equal(driver.calls.some(call => call[0] === "stop"), false);
  await environment.close();
});

test("a task evaluator can score a robot observation independently of command acknowledgement", async () => {
  const environment = createRobotEnvironment({ driver: fixtureDriver(["joint"]), execute: true, evaluate: ({ observation }) => ({ terminated: true, success: observation.state.positions.joint === 0.25 }) });
  await environment.reset();
  const next = await environment.step({ name: "robot_command", arguments: { values: { joint: 0.25 } } });
  assert.equal(next.success, true);
  assert.equal(next.terminated, true);
  await environment.close();
});

function workerFixture() {
  const calls = [];
  let running = true;
  return {
    calls,
    health: () => ({ running }),
    stop() { running = false; },
    async request(payload) {
      calls.push(payload);
      if (payload.op === "configure") return { ok: true, result: { actions: [{ name: "joint_position", parameters: { type: "object" } }], metadata: { cameraNames: ["top", "left_wrist"] } } };
      return { ok: true, result: { state: { qpos: Array(14).fill(0) }, info: { execute: payload.execute } } };
    },
  };
}

test("ALOHA uses one lazy Python driver and sim defaults to execution while real defaults to preview", async () => {
  for (const mode of ["sim", "real"]) {
    const worker = workerFixture();
    const env = createAlohaEnvironment({ mode, worker, actRoot: "/path/to/act" });
    assert.equal(worker.calls.length, 0);
    await env.reset({ task: { taskName: "sim_transfer_cube_scripted" }, seed: 5 });
    await env.step({ name: "joint_position", arguments: { positions: Array(14).fill(0) } });
    assert.equal(worker.calls[0].driver, "aloha");
    assert.equal(worker.calls[0].options.executeEnabled, mode === "sim");
    assert.equal(worker.calls.at(-1).execute, mode === "sim");
    await env.close();
    assert.equal(worker.calls.at(-1).op, "close");
  }
});

test("LeRobot forwards hardware config and an interrupted worker is not restarted", async () => {
  const worker = workerFixture();
  const env = createLeRobotEnvironment({ worker, robotConfig: { type: "so101_follower", port: "fixture" } });
  await env.reset();
  assert.equal(worker.calls[0].driver, "lerobot");
  assert.equal(worker.calls[0].options.robotConfig.type, "so101_follower");
  let rejectRequest;
  worker.request = () => new Promise((resolve, reject) => { rejectRequest = reject; });
  worker.stop = () => rejectRequest?.(new Error("worker exited"));
  const controller = new AbortController();
  const pending = env.step({ name: "joint_position", arguments: {} }, { signal: controller.signal });
  controller.abort();
  await assert.rejects(pending, { name: "AbortError" });
  await assert.rejects(env.observe(), /interrupted/u);
  await env.close();
});

test("OpenRAL and RPent forward platform configuration through the common lazy worker", async () => {
  for (const [create, options, kind, execute] of [
    [createOpenRALEnvironment, { robotId: "aloha_bimanual", mode: "sim" }, "openral", true],
    [createOpenRALEnvironment, { robotId: "so101_follower", mode: "real" }, "openral", false],
    [createRPentEnvironment, { robot: "robotwin", endpoint: "socket://localhost:8803" }, "rpent", true],
  ]) {
    const worker = workerFixture();
    const environment = create({ ...options, worker });
    assert.equal(worker.calls.length, 0);
    await environment.reset({ seed: 3 });
    assert.equal(worker.calls[0].driver, kind);
    assert.equal(worker.calls[0].options.executeEnabled, execute);
    for (const [key, value] of Object.entries(options)) assert.equal(worker.calls[0].options[key], value);
    await environment.step({ name: "joint_position", arguments: {} });
    assert.equal(worker.calls.at(-1).execute, execute);
    await environment.close();
    assert.equal(worker.calls.at(-1).op, "close");
  }
});

test("official OpenRAL single-arm and bimanual MuJoCo HALs run through the Python bridge", {
  skip: !process.env.OPENRAL_TEST_PHYSICS,
}, async () => {
  for (const robotId of ["franka_panda", "aloha_bimanual"]) {
    const environment = createOpenRALEnvironment({ openralRoot: process.env.OPENRAL_TEST_ROOT, robotId, cameraNames: [] });
    try {
      const initial = await environment.reset();
      const positions = initial.observation.state.position;
      assert.equal(positions.length, initial.observation.state.robot.jointNames.length);
      const next = await environment.step({ name: "hal_sequence", arguments: { commands: [
        { controlMode: "joint_position", values: positions },
        { controlMode: "joint_position", values: positions },
      ] } });
      assert.equal(next.info.controlSteps, 2);
      assert.equal(next.info.executed, true);
      assert.equal(next.success, null);
      assert.equal(next.terminated, false);
    } finally {
      await environment.close();
    }
  }
});

test("the registry passes its environment to the real Python bridge and preserves actual commands", async context => {
  const parentPath = process.env.PYTHONPATH;
  const runtimeEnv = { ...process.env, PYTHONPATH: [fileURLToPath(new URL("./fixtures", import.meta.url)), parentPath].filter(Boolean).join(delimiter) };
  const env = await createEnvironment({ type: "lerobot", options: { robotFactory: "named_robot:make_robot", factoryOptions: { arms: ["left", "right"] }, execute: true, controlHz: 100 } }, { env: runtimeEnv });
  context.after(() => env.close());
  const initial = await env.reset({ task: "factory integration" });
  assert.equal(process.env.PYTHONPATH, parentPath);
  assert.equal(initial.observation.images[0].name, "top");
  assert.equal(initial.observation.images[0].mimeType, "image/png");
  const next = await env.step({ name: "robot_trajectory", arguments: { commands: [
    { "left.joint.pos": 0.25, "right.joint.pos": 0.5 },
    { "left.joint.pos": 0.75, "right.joint.pos": 0.25 },
  ] } });
  assert.equal(next.info.controlSteps, 2);
  assert.deepEqual(next.info.actualActions.at(-1), { "left.joint.pos": 0.5, "right.joint.pos": 0.25 });
  assert.equal(next.success, null);
  assert.equal(next.terminated, false);
  const stop = await env.stop();
  assert.equal(stop.supported, false);
});

test("official ACT simulator runs both ALOHA task families through the common episode runner", {
  skip: !process.env.ALOHA_TEST_ACT_ROOT,
}, async context => {
  const environment = createAlohaEnvironment({
    actRoot: process.env.ALOHA_TEST_ACT_ROOT,
    imageWidth: 64, imageHeight: 48,
  });
  context.after(() => environment.close());
  for (const task of ["sim_transfer_cube_scripted", "sim_insertion_scripted"]) {
    const result = await runEpisode({
      environment, task, seed: 3, maxSteps: 1,
      policy: { id: "hold-position-smoke", async decide({ observation }) {
        assert.equal(observation.state.qpos.length, 14);
        assert.equal(observation.state.robot.jointNames.length, 14);
        assert.equal(observation.images[0].name, "top");
        assert.equal(Buffer.from(observation.images[0].data, "base64").subarray(1, 4).toString(), "PNG");
        assert.equal(Object.hasOwn(observation.state, "env_state"), false);
        return { action: { name: "joint_trajectory", arguments: { positions: Array.from({ length: 3 }, () => [...observation.state.qpos]) } } };
      } },
    });
    assert.equal(result.error, null);
    assert.equal(result.steps, 1);
    assert.equal(result.info.controlSteps, 3);
    assert.equal(result.info.totalControlSteps, 3);
    assert.equal(result.success, false);
    assert.equal(result.reason, "max_steps");
  }
});
