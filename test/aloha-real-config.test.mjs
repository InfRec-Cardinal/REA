import test from "node:test";
import assert from "node:assert/strict";
import configuration from "../examples/aloha-real.config.mjs";
import { validateConfiguration } from "../src/framework/configuration.mjs";

test("ALOHA real entry point previews hardware commands and keeps task language", () => {
  const config = validateConfiguration(configuration({ env: {
    REA_MODEL: "example-model", REA_INSTRUCTION: "Place the object in the tray.",
    ALOHA_ROOT: "/robot/aloha", ALOHA_PYTHON: "/robot/bin/python",
  } }));
  assert.equal(config.environment.options.mode, "real");
  assert.equal(config.environment.options.execute, false);
  assert.equal(config.environment.options.alohaRoot, "/robot/aloha");
  assert.equal(config.policy.type, "chat");
  assert.equal(config.episodes[0].task.instruction, config.episodes[0].instruction);
});

test("ALOHA VLA entry point maps measured state and native trajectory actions", () => {
  const config = validateConfiguration(configuration({ env: {
    ALOHA_POLICY: "lingbot-vla", LINGBOT_VLA_ROBOT: "custom-aloha", ROBOT_EXECUTE: "true",
  } }));
  assert.equal(config.environment.options.execute, true);
  assert.equal(config.policy.type, "lingbot-vla");
  assert.equal(config.policy.options.robot, "custom-aloha");
  assert.deepEqual(config.policy.options.stateMap, { "observation.state": "state.qpos" });
  assert.equal(config.policy.options.actionName, "joint_trajectory");
  assert.equal(config.policy.options.actionArgument, "positions");
  assert.equal(config.policy.options.imageMap["observation.images.cam_left_wrist"], "cam_left_wrist");
});


test("ALOHA hybrid registers both VLAs and Python code with a background router", () => {
  const config = validateConfiguration(configuration({ env: {
    ALOHA_POLICY: "hybrid", REA_MODEL: "code-model", REA_ROUTER_MODEL: "router-model",
    LINGBOT_VLA_ROBOT: "custom-aloha", PI05_URL: "ws://pi05-host:8000",
    LINGBOT_VLA_URL: "ws://lingbot-host:8006", REA_CODE_PYTHON: "/code/python",
  } }));
  assert.equal(config.policy, undefined);
  assert.equal(config.environment.options.execute, false);
  assert.deepEqual(config.systems.system0.map(({ id, policy }) => [id, policy.type]), [
    ["pi05", "pi05"], ["lingbot-vla", "lingbot-vla"], ["code", "aloha-code"],
  ]);
  assert.equal(config.systems.system0[0].policy.options.url, "ws://pi05-host:8000");
  assert.equal(config.systems.system0[1].policy.options.url, "ws://lingbot-host:8006");
  assert.equal(config.systems.system0[2].policy.options.python, "/code/python");
  assert.equal(config.systems.mode, "async");
  assert.equal(config.systems.thinkingSchedule, "after_dispatch");
  assert.equal(config.systems.thinkEverySteps, 4);
  assert.equal(config.systems.minControllerSteps, 4);
  assert.equal(config.systems.system1, undefined);
  assert.equal(config.systems.system2.options.model, "router-model");
  assert.equal(config.systems.system2.options.exportName, "createAlohaRouterThinker");
  assert.equal(config.systems.thinkingTimeoutMs, config.budget.decisionTimeoutMs);
});

test("ALOHA pi0.5 can run directly without an unused chat model or LingBot configuration", () => {
  const config = validateConfiguration(configuration({ env: { ALOHA_POLICY: "pi05" } }));
  assert.equal(config.policy.type, "pi05");
  assert.equal(config.systems, undefined);
});


test("ALOHA hybrid accepts separate router and code models without REA_MODEL", () => {
  const config = configuration({ env: {
    ALOHA_POLICY: "hybrid", REA_CODE_MODEL: "python-generator",
    REA_ROUTER_MODEL: "controller-selector", LINGBOT_VLA_ROBOT: "custom-aloha",
  } });
  assert.equal(config.systems.system0.find(candidate => candidate.id === "code").policy.options.model, "python-generator");
  assert.equal(config.systems.system2.options.model, "controller-selector");
});

test("ALOHA routing mode and observation cadence can be configured", () => {
  const env = {
    ALOHA_POLICY: "pi05-code", REA_MODEL: "example-model", ALOHA_ROUTING_MODE: "sync",
    ALOHA_ROUTE_EVERY_STEPS: "2", ALOHA_ROUTE_MAX_AGE_STEPS: "4", REA_DECISION_TIMEOUT_MS: "30000",
    ALOHA_MIN_CONTROLLER_STEPS: "3",
  };
  const config = validateConfiguration(configuration({ env }));
  assert.equal(config.systems.mode, "sync");
  assert.equal(config.systems.thinkingSchedule, "before_route");
  assert.equal(config.systems.thinkEverySteps, 2);
  assert.equal(config.systems.minControllerSteps, 3);
  assert.equal(config.systems.maxAdviceAgeSteps, 4);
  assert.equal(config.systems.thinkingTimeoutMs, 30000);
  assert.throws(() => configuration({ env: { ...env, ALOHA_ROUTING_MODE: "unknown" } }), /ALOHA_ROUTING_MODE/);
  const sync = configuration({ env: { ALOHA_POLICY: "pi05-code", REA_MODEL: "example-model", ALOHA_ROUTING_MODE: "sync" } });
  assert.equal(sync.systems.thinkEverySteps, 1);
  assert.equal(sync.systems.minControllerSteps, 1);
});
