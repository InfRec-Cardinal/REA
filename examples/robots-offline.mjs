import { createRobotEnvironment } from "../src/environments/robot.mjs";

// Synthetic named-joint driver: no Python, serial ports, model or hardware.
function fixtureDriver(arms) {
  const names = arms.flatMap(arm => ["shoulder", "elbow", "gripper"].map(joint => `${arm}.${joint}`));
  const properties = Object.fromEntries(names.map(name => [name, { type: "number", minimum: name.endsWith("gripper") ? 0 : -1, maximum: 1 }]));
  const command = { type: "object", properties, required: names, additionalProperties: false };
  let positions;
  const actions = [{ name: "robot_trajectory", description: "Set named joints in sequence; synthetic radians and normalized gripper values", parameters: {
    type: "object", properties: { commands: { type: "array", minItems: 1, maxItems: 10, items: command } }, required: ["commands"], additionalProperties: false,
  } }];
  const snapshot = () => ({ timestampMs: Date.now(), state: { positions }, images: [], success: null });
  return {
    async describe() { return { actions, timing: "turn-based", metadata: { fixture: true, arms, jointNames: names } }; },
    async reset() { positions = Object.fromEntries(names.map(name => [name, 0])); return snapshot(); },
    async observe() { return snapshot(); },
    async step(action, { execute }) {
      const commands = action.arguments.commands;
      if (execute) positions = { ...commands.at(-1) };
      return { ...snapshot(), truncated: true, info: { controlSteps: execute ? commands.length : 0, preview: !execute, fixture: true } };
    },
    async stop() { return { supported: true, fixture: true }; },
    async close() {},
  };
}

export default {
  metadata: { fixture: true, purpose: "Named single-arm and bimanual action-chunk interface demonstration" },
  episodes: [{ id: "single-arm", task: ["arm"] }, { id: "bimanual", task: ["left", "right"] }],
  budget: { maxSteps: 1 },
  createEnvironment({ task }) { return createRobotEnvironment({ id: `fixture:${task.join("+")}`, driver: fixtureDriver(task), execute: true }); },
  createPolicy() {
    return { id: "scripted-fixture", async decide({ observation }) {
      const values = observation.state.positions;
      return { action: { name: "robot_trajectory", arguments: { commands: [
        Object.fromEntries(Object.keys(values).map(name => [name, 0.1])),
        Object.fromEntries(Object.keys(values).map(name => [name, 0.2])),
      ] } } };
    } };
  },
};
