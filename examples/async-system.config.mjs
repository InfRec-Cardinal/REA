import { createAsyncSystemDemo } from "./async-system-demo.mjs";

// Same System0/1/2 runtime through the existing configuration-based CLI and API.
export default {
  name: "async-system-demonstration",
  episodes: [{ id: "reach-around-fixture", seed: 0, instruction: "Reach the target around the fixture." }],
  budget: { maxSteps: 24, maxDurationMs: 30000 },
  createEnvironment() { return createAsyncSystemDemo().environment; },
  createSystems() {
    const { environment, ...systems } = createAsyncSystemDemo();
    return systems;
  },
};
