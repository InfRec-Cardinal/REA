const CLICK = { name: "click", description: "Select a grid cell", parameters: { type: "object", properties: { x: { type: "integer", minimum: 0, maximum: 3 }, y: { type: "integer", minimum: 0, maximum: 3 } }, required: ["x", "y"], additionalProperties: false } };
const MOVE = { name: "move_joints", description: "Set two joint positions", parameters: { type: "object", properties: { positions: { type: "array", items: { type: "number", minimum: -1, maximum: 1 }, minItems: 2, maxItems: 2 } }, required: ["positions"], additionalProperties: false } };

export default {
  episodes: [
    { id: "grid-click", task: "grid", seed: 1 },
    { id: "two-joint-robot", task: "joints", seed: 1 },
  ],
  budget: { maxSteps: 3, maxDurationMs: 10000 },
  createEnvironment({ task }) {
    let done = false;
    const observation = () => ({
      id: done ? "1" : "0",
      timestampMs: Date.now(),
      text: "Synthetic interface fixture; target is deliberately visible.",
      state: { target: task === "grid" ? [2, 1] : [0.25, -0.5], reached: done },
      actions: [task === "grid" ? CLICK : MOVE],
    });
    return {
      id: `fixture:${task}`,
      timing: "turn-based",
      async reset() { done = false; return { observation: observation(), success: false }; },
      async step(action) {
        const target = observation().state.target;
        done = task === "grid" ? action.arguments.x === target[0] && action.arguments.y === target[1] : action.arguments.positions.every((value, index) => value === target[index]);
        return { observation: observation(), terminated: done, success: done, reward: done ? 1 : 0 };
      },
    };
  },
  createPolicy() {
    return {
      id: "fixture-visible-target",
      async decide({ observation, actions }) {
        const target = observation.state.target;
        return { action: { name: actions[0].name, arguments: actions[0].name === "click" ? { x: target[0], y: target[1] } : { positions: target } } };
      },
    };
  },
};
