import assert from "node:assert/strict";
import test from "node:test";

import { createEmbodiedAgent, createRegistry } from "../src/framework/index.mjs";
import { createVlmJudge } from "../src/framework/judge-gate.mjs";
import { buildRouteTable, createRouteMemoryRouter } from "../src/framework/route-memory.mjs";
import { runRoutedEpisode } from "../src/framework/routed-runtime.mjs";
import { createRoutedSystemDemo } from "../examples/routed-system-demo.mjs";

const CONTINUE = { name: "continue", description: "Advance the selected capability", parameters: { type: "object", properties: {}, additionalProperties: false } };

// A resumable drawer: each capability pushes for a fixed number of steps and then returns. Only score() reads success.
function drawerEnvironment({ resumable = true, push = { quick: [0.3, 2], thorough: [0.25, 4] } } = {}) {
  const log = { scores: 0, steps: [] };
  let open = 0;
  let active = null;
  let left = 0;
  const observation = () => ({
    id: `drawer-${log.steps.length}`, timestampMs: Date.now(), actions: [CONTINUE], text: `drawer ${Math.round(open * 100)}% open`,
    state: { open }, images: [{ name: "front", mimeType: "image/jpeg", data: Buffer.from(`open=${open.toFixed(2)}`).toString("base64") }],
  });
  return {
    log,
    environment: {
      id: "drawer", resumable, actions: [CONTINUE],
      async reset() { open = 0; active = null; left = 0; return { observation: observation() }; },
      async step(action, { reasoning }) {
        const candidate = reasoning.route.candidateId;
        if (active !== candidate) { active = candidate; left = push[candidate][1]; }
        open = Math.min(1, open + push[candidate][0]);
        left -= 1;
        log.steps.push(candidate);
        const returned = left === 0;
        if (returned) active = null;
        return { observation: observation(), terminated: !resumable && returned, info: { capability: { candidateId: candidate, status: returned ? "returned" : "running" } } };
      },
      async stop() {},
      async score() { log.scores += 1; return { success: open >= 0.99 }; },
    },
  };
}

const capability = id => ({ id, kind: "program", policy: { id, async decide(input) {
  assert.equal("success" in input.feedback, false, "capabilities never see success");
  return { action: { name: "continue", arguments: {} } };
} } });

const judgeFromImage = calls => ({ async judge({ observation, initialObservation }) {
  calls.push({ observation: observation.id, initial: initialObservation.id });
  const open = Number(Buffer.from(observation.images[0].data, "base64").toString().split("=")[1]);
  return { verdict: open >= 0.99 ? "done" : "not_done", done: open >= 0.99, confidence: 0.9 };
} });

const developmentTable = () => buildRouteTable([
  { task: "drawer", candidateId: "quick", success: true, steps: 2 },
  { task: "drawer", candidateId: "thorough", success: true, steps: 4 },
], { candidates: ["quick", "thorough"] });

test("route memory runs first, the judge runs only at boundaries, the ladder escalates and success is read once", async () => {
  const { environment, log } = drawerEnvironment();
  const judged = [];
  const events = [];
  const summary = await runRoutedEpisode({
    environment, task: "drawer", system0: [capability("quick"), capability("thorough")],
    system1: createRouteMemoryRouter({ table: developmentTable(), recoverOn: "judge" }),
    system2: judgeFromImage(judged), onEvent: event => events.push(event),
  });
  assert.deepEqual(log.steps, ["quick", "quick", "thorough", "thorough", "thorough", "thorough"]);
  assert.deepEqual(summary.runs.map(run => [run.candidateId, run.status, run.steps, run.verdict.verdict]), [["quick", "returned", 2, "not_done"], ["thorough", "returned", 4, "done"]]);
  assert.equal(judged.length, 2, "one judge call per capability boundary");
  assert.deepEqual(judged.map(call => call.initial), ["drawer-0", "drawer-0"]);
  assert.equal(summary.reason, "router_stop");
  assert.equal(summary.stopReason, "judge_done");
  assert.equal(summary.success, true);
  assert.equal(log.scores, 1);
  assert.equal(summary.steps, 6);
  const order = events.map(event => `${event.type}.${event.status}`).filter(key => key.startsWith("judge") || key.startsWith("run"));
  assert.deepEqual(order, ["run.started", "run.completed", "judge.started", "judge.completed", "run.started", "run.completed", "judge.started", "judge.completed"]);
});

test("all runs share one step budget; the episode is still scored when the budget ends", async () => {
  const { environment, log } = drawerEnvironment();
  const summary = await runRoutedEpisode({
    environment, task: "drawer", system0: [capability("quick"), capability("thorough")], maxSteps: 3,
    system1: createRouteMemoryRouter({ table: developmentTable(), recoverOn: "always" }),
  });
  assert.equal(summary.reason, "max_steps");
  assert.deepEqual(summary.runs.map(run => [run.candidateId, run.status, run.steps]), [["quick", "returned", 2], ["thorough", "budget", 1]]);
  assert.equal(summary.success, false);
  assert.equal(log.scores, 1);
});

test("maxStepsPerRun bounds a capability that never returns, and recoverOn exception escalates after it", async () => {
  const { environment } = drawerEnvironment({ push: { quick: [0.1, 100], thorough: [0.5, 2] } });
  const summary = await runRoutedEpisode({
    environment, task: "drawer", system0: [capability("quick"), capability("thorough")], maxStepsPerRun: 3,
    system1: createRouteMemoryRouter({ table: developmentTable(), recoverOn: "exception" }),
  });
  assert.deepEqual(summary.runs.map(run => [run.candidateId, run.status, run.steps]), [["quick", "budget", 3], ["thorough", "returned", 2]]);
  assert.equal(summary.stopReason, "run_returned");
  assert.equal(summary.success, true);
});

test("a non-resumable environment ends the episode when it terminates", async () => {
  const { environment, log } = drawerEnvironment({ resumable: false });
  const summary = await runRoutedEpisode({
    environment, task: "drawer", system0: [capability("quick"), capability("thorough")],
    system1: createRouteMemoryRouter({ table: developmentTable(), recoverOn: "always" }),
  });
  assert.equal(summary.reason, "terminated");
  assert.deepEqual(summary.runs.map(run => run.candidateId), ["quick"]);
  assert.equal(log.scores, 1);
});

test("a failed run reports its error, and a policy error leaves the episode unscored", async () => {
  const failing = drawerEnvironment();
  failing.environment.step = async (action, { reasoning }) => ({ observation: { id: "x", actions: [CONTINUE], images: [] }, info: { capability: { candidateId: reasoning.route.candidateId, status: "failed", error: "IK failed" } } });
  const summary = await runRoutedEpisode({ environment: failing.environment, task: "drawer", system0: [capability("quick")], system1: createRouteMemoryRouter({ recoverOn: "exception", maxRuns: 2 }) });
  assert.deepEqual(summary.runs.map(run => [run.status, run.error]), [["failed", "IK failed"], ["failed", "IK failed"]]);
  assert.match(summary.stopReason, /^ladder_exhausted/u);
  const broken = drawerEnvironment();
  const error = await runRoutedEpisode({ environment: broken.environment, task: "drawer", system0: [{ id: "quick", policy: { async decide() { throw new Error("policy crashed"); } } }] });
  assert.equal(error.reason, "error");
  assert.equal(error.success, null);
  assert.equal(broken.log.scores, 0, "no score after an error");
});

test("the VLM judge sends the current and the reset image and applies its threshold", async () => {
  const frames = [];
  const judge = createVlmJudge({ model: "judge-model", threshold: 0.8, infer: async (frame, options) => {
    frames.push([frame.imageB64, options.initialFrame?.imageB64, options.instruction]);
    return { done: true, confidence: frames.length === 1 ? 0.7 : 0.9, rationale: "drawer open" };
  } });
  const observation = { images: [{ data: "bm93", mimeType: "image/jpeg" }] };
  const initialObservation = { images: [{ data: "c3RhcnQ=", mimeType: "image/jpeg" }] };
  assert.equal((await judge.judge({ observation, initialObservation, instruction: "open the drawer" })).verdict, "not_done");
  assert.equal((await judge.judge({ observation, initialObservation, instruction: "open the drawer" })).verdict, "done");
  assert.deepEqual(frames[0], ["bm93", "c3RhcnQ=", "open the drawer"]);
  assert.equal((await judge.judge({ observation: { images: [] } })).verdict, "error");
  const failing = createVlmJudge({ model: "m", infer: async () => { throw new Error("HTTP 503"); } });
  assert.deepEqual(await failing.judge({ observation }), { verdict: "error", error: "HTTP 503", usage: null });
});

test("createEmbodiedAgent runs routed systems from configuration with a route-memory System 1", async () => {
  const { environment, log } = drawerEnvironment();
  const judged = [];
  const agent = await createEmbodiedAgent({
    name: "routed-drawer",
    episodes: [{ id: "drawer-1", task: "drawer", seed: 0 }],
    createEnvironment: () => environment,
    createSystems: () => ({
      mode: "routed",
      system0: [capability("quick"), capability("thorough")],
      system1: createRouteMemoryRouter({ table: developmentTable() }),
      system2: judgeFromImage(judged),
    }),
  });
  try {
    const summary = await agent.run();
    assert.equal(summary.mode, "routed");
    assert.equal(summary.success, true);
    assert.equal(log.scores, 1);
  } finally {
    await agent.close();
  }
});

test("the routed demonstration escalates through the whole ladder before the judge says done", async () => {
  const summary = await runRoutedEpisode(createRoutedSystemDemo());
  assert.deepEqual(summary.runs.map(run => [run.candidateId, run.verdict.verdict]), [["quick", "not_done"], ["thorough", "not_done"], ["quick", "done"]]);
  assert.equal(summary.success, true);
  assert.equal(summary.score.open, 1);
});

test("registry environments forward resumable and score() to routed episodes", async () => {
  const drawer = drawerEnvironment();
  const registry = createRegistry();
  registry.registerEnvironment("drawer", () => drawer.environment, { description: "test drawer" });
  const environment = await registry.createEnvironment({ type: "drawer" });
  const summary = await runRoutedEpisode({
    environment, task: "drawer", system0: [capability("quick"), capability("thorough")],
    system1: createRouteMemoryRouter({ table: developmentTable(), recoverOn: "judge" }), system2: judgeFromImage([]),
  });
  assert.equal(environment.resumable, true);
  assert.equal(summary.success, true);
  assert.equal(drawer.log.scores, 1);
  await environment.close();
});
