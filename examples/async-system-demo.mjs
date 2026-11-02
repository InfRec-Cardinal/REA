import { setTimeout as sleep } from "node:timers/promises";
import { pathToFileURL } from "node:url";
import { runAsyncEpisode } from "../src/framework/async-runtime.mjs";

const MOVE = {
  name: "move", description: "Move the simulated end effector by a bounded increment.",
  parameters: { type: "object", properties: { dx: { type: "number" }, dy: { type: "number" } }, required: ["dx", "dy"], additionalProperties: false },
};

/** A deterministic scheduling demonstration, not a robot or LIBERO benchmark. */
export function createAsyncSystemDemo({ mode = "async", brainMs = 1350, actionMs = 240, policyMs = 45, ...overrides } = {}) {
  let x = 0.06;
  let y = 0;
  let step = 0;
  const observation = () => ({
    id: `demo-${step}`, timestampMs: Date.now(), actions: [MOVE],
    text: "Reach the target at (1, 0.3). A fixture at x=0.48 blocks the direct approach; pass above y=0.23.",
    state: { x, y, target: { x: 1, y: 0.3 }, obstacle: { x: 0.48, y: 0 }, progress: Math.min(1, x), demonstration: true },
  });
  const environment = {
    id: "async-scheduling-demonstration", actions: [MOVE],
    async reset() { x = 0.06; y = 0; step = 0; return { observation: observation(), success: false }; },
    async step(action, { signal }) {
      await sleep(actionMs, undefined, { signal });
      const nextX = x + action.arguments.dx;
      if (y >= 0.23 || nextX < 0.43) x = Math.min(1, nextX);
      y = Math.min(0.3, Math.max(0, y + action.arguments.dy));
      step += 1;
      const success = x >= 0.99 && y >= 0.23;
      return { observation: observation(), terminated: success, success, reward: success ? 1 : 0 };
    },
    async stop() {},
  };
  const candidate = (id, kind, decide) => ({ id, kind, policy: { id, async decide(input) {
    await sleep(policyMs, undefined, { signal: input.signal });
    return { action: { name: "move", arguments: decide(input.observation.state) }, thought: input.directive ?? "Execute current local controller." };
  } } });
  return {
    environment,
    system0: [
      candidate("sdk-reach", "sdk", () => ({ dx: 0.065, dy: 0 })),
      candidate("vla-detour", "vla", () => ({ dx: 0.03, dy: 0.15 })),
      candidate("motion-servo", "motion", () => ({ dx: 0.085, dy: 0 })),
    ],
    system1: async ({ observation: current, advice, signal }) => {
      await sleep(18, undefined, { signal });
      if (current.state.y >= 0.23) return { candidateId: "motion-servo", reason: "Clear of the fixture; motor controller completes the reach." };
      if (advice?.candidateId) return { candidateId: advice.candidateId, reason: advice.reason };
      return { candidateId: "sdk-reach", reason: "Begin the approach while the brain evaluates the scene." };
    },
    system2: { id: "demo-brain", async think({ signal }) {
      await sleep(brainMs, undefined, { signal });
      return { candidateId: "vla-detour", directive: "Lift above y=0.23, then continue to the target.", reason: "The direct approach intersects the fixture; use VLA for the detour." };
    } },
    mode,
    task: "reach-around-fixture", seed: 0,
    instruction: "Reach the target. Start with the available controller and incorporate the brain's route correction when ready.",
    maxSteps: 24, maxDurationMs: 30000, thinkEverySteps: 5, maxAdviceAgeSteps: 24,
    ...overrides,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const mode = process.argv.includes("--sync") ? "sync" : "async";
  const summary = await runAsyncEpisode(createAsyncSystemDemo({ mode, onEvent: event => {
    if (process.argv.includes("--events")) process.stdout.write(`${JSON.stringify(event)}\n`);
  } }));
  process.stdout.write(`${JSON.stringify({ demonstration: true, ...summary }, null, 2)}\n`);
}
