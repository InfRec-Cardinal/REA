import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { createRegistry } from "../src/framework/registry.mjs";
import { runEpisode } from "../src/framework/runtime.mjs";

const examples = fileURLToPath(new URL("../examples/robots", import.meta.url));
const python = process.env.PYTHON ?? "python3";

test("a Python driver class runs behind the robot contract through the python-robot type", async () => {
  const registry = createRegistry();
  const environment = await registry.createEnvironment({ type: "python-robot", options: {
    driver: "template_driver:TemplateDriver", python, execute: true, options: { joints: 2 },
  } }, { env: { ...process.env, PYTHONPATH: examples } });
  const policy = { id: "scripted", async decide({ observation }) {
    assert.deepEqual(observation.state.robot.jointNames, ["joint_0", "joint_1"]);
    return { action: { name: "move_joints", arguments: { positions: [0.5, -0.25] } } };
  } };
  try {
    const summary = await runEpisode({ environment, policy, maxSteps: 2 });
    assert.equal(summary.steps, 2);
    assert.deepEqual(summary.finalObservation.state.positions, [0.5, -0.25]);
  } finally {
    await environment.close();
  }
});

test("python-robot refuses a driver that is not package.module:Class", async () => {
  const registry = createRegistry();
  const environment = await registry.createEnvironment({ type: "python-robot", options: { driver: "template_driver" } });
  await assert.rejects(environment.reset({}), /package\.module:ClassOrFactory/u);
});
