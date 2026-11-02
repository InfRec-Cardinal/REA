import { createRoutedSystemDemo } from "./routed-system-demo.mjs";

// REA's routed episode through the configuration-based CLI and API: route memory picks the first program, the judge
// runs only when a program returns, the ladder escalates, and success is read once at the end.
export default {
  name: "routed-system-demonstration",
  episodes: [{ id: "open-drawer", task: "open-drawer", seed: 0, instruction: "Open the drawer fully." }],
  budget: { maxSteps: 20, maxDurationMs: 30000 },
  createEnvironment() { return createRoutedSystemDemo().environment; },
  createSystems() {
    const { environment, ...systems } = createRoutedSystemDemo();
    return systems;
  },
};
