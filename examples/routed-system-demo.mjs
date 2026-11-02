import { pathToFileURL } from "node:url";

import { buildRouteTable, createRouteMemoryRouter } from "../src/framework/route-memory.mjs";
import { runRoutedEpisode } from "../src/framework/routed-runtime.mjs";

const CONTINUE = {
  name: "continue_program", description: "Run the selected frozen program for one control block.",
  parameters: { type: "object", properties: {}, additionalProperties: false },
};

// Development evidence, as tools/route-table.mjs would read it from one development evaluation per program:
// "quick" opens the drawer in fewer steps when it works, "thorough" is slower and always works.
export const DEVELOPMENT_EPISODES = [
  { task: "open-drawer", candidateId: "quick", success: true, steps: 3 },
  { task: "open-drawer", candidateId: "quick", success: true, steps: 3 },
  { task: "open-drawer", candidateId: "thorough", success: true, steps: 5 },
  { task: "open-drawer", candidateId: "thorough", success: true, steps: 5 },
];

/** A deterministic routed-episode demonstration, not a robot or benchmark: a drawer that frozen programs push open. */
export function createRoutedSystemDemo({ stiffness = 0.45, ...overrides } = {}) {
  let open = 0;
  let active = null;
  let blocks = 0;
  const programs = { quick: { push: 0.6 - stiffness / 2, blocks: 3 }, thorough: { push: 0.25, blocks: 5 } };
  const observation = () => ({
    id: `drawer-${blocks}`, timestampMs: Date.now(), actions: [CONTINUE],
    text: `The drawer is ${Math.round(open * 100)}% open.`,
    state: { open: Number(open.toFixed(3)), demonstration: true },
  });
  let left = 0;
  const environment = {
    id: "routed-drawer-demonstration", timing: "turn-based", resumable: true, actions: [CONTINUE],
    async reset() { open = 0; active = null; blocks = 0; return { observation: observation() }; },
    async step(action, { reasoning }) {
      const candidateId = reasoning.route.candidateId;
      if (active !== candidateId) { active = candidateId; left = programs[candidateId].blocks; }
      open = Math.min(1, open + programs[candidateId].push / programs[candidateId].blocks * 1.25);
      blocks += 1;
      left -= 1;
      const status = left === 0 ? "returned" : "running";
      if (status === "returned") active = null;
      return { observation: observation(), info: { capability: { candidateId, status } } };
    },
    async stop() {},
    // The sealed score: the only place the demonstration reads its success condition.
    async score() { return { success: open >= 0.99, open }; },
  };
  const program = id => ({ id, kind: "program", description: `frozen program ${id}`, policy: { id, async decide() {
    return { action: { name: "continue_program", arguments: {} }, thought: `continue ${id}` };
  } } });
  return {
    environment,
    mode: "routed",
    system0: [program("quick"), program("thorough")],
    system1: createRouteMemoryRouter({ table: buildRouteTable(DEVELOPMENT_EPISODES, { candidates: ["quick", "thorough"], createdAt: "2026-01-01T00:00:00.000Z" }), recoverOn: "judge" }),
    // A scripted stand-in for a vision-language judge: it reads the rendered description, never the score.
    system2: { id: "demo-judge", async judge({ observation }) {
      const percent = Number(/(\d+)% open/u.exec(observation.text)?.[1] ?? 0);
      return { verdict: percent >= 99 ? "done" : "not_done", done: percent >= 99, confidence: 0.9, rationale: observation.text };
    } },
    task: "open-drawer", seed: 0, instruction: "Open the drawer fully.",
    maxSteps: 20, maxDurationMs: 30000,
    ...overrides,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const summary = await runRoutedEpisode(createRoutedSystemDemo({ onEvent: event => {
    if (process.argv.includes("--events")) process.stdout.write(`${JSON.stringify(event)}\n`);
  } }));
  process.stdout.write(`${JSON.stringify({ demonstration: true, ...summary }, null, 2)}\n`);
}
