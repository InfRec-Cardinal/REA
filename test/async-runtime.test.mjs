import assert from "node:assert/strict";
import test from "node:test";

import { runAsyncEpisode } from "../src/framework/async-runtime.mjs";

const MOVE = { name: "move", parameters: { type: "object", properties: { direction: { type: "string", enum: ["left", "right"] } }, required: ["direction"], additionalProperties: false } };
const flush = () => new Promise(resolve => setImmediate(resolve));
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const capability = (id, direction) => ({ id, kind: "sdk", description: `${direction} movement`, policy: { async decide({ directive }) { return { action: { name: "move", arguments: { direction } }, thought: directive ?? "continue", usage: { total_tokens: 2 } }; } } });
const system0 = () => [capability("fast", "left"), capability("corrected", "right")];

function fixture({ stepLimit = 3, holdSteps = false } = {}) {
  let phase = 0;
  const observation = () => ({ id: String(phase), state: { phase }, actions: [MOVE], timestampMs: Date.now() });
  const executed = [];
  const pending = [];
  const arrivals = Array.from({ length: stepLimit }, () => Promise.withResolvers());
  return {
    id: "async-fixture", executed, pending, arrivals, stopped: 0,
    async reset() { phase = 0; return { observation: observation(), success: false }; },
    async step(action, { signal }) {
      const index = executed.length;
      executed.push(action.arguments.direction);
      const deferred = Promise.withResolvers();
      pending.push({ ...deferred, signal });
      arrivals[index]?.resolve();
      if (holdSteps) await deferred.promise;
      phase += 1;
      return { observation: observation(), terminated: phase === stepLimit, success: phase === stepLimit, reward: phase === stepLimit ? 1 : 0, info: { phase } };
    },
    async stop() { this.stopped += 1; },
  };
}

test("System 1 keeps routing and System 0 keeps executing while System 2 thinks, then corrects at the next boundary", async () => {
  const environment = fixture({ holdSteps: true });
  const thought = Promise.withResolvers();
  const completed = Promise.withResolvers();
  const events = [];
  const transitions = [];
  const running = runAsyncEpisode({
    environment, system0: system0(), maxSteps: 3, thinkEverySteps: 100,
    system2: { think: input => {
      assert.equal(input.candidates[1].description, "right movement");
      return thought.promise;
    } },
    onEvent: event => { events.push(event); if (event.type === "thinking" && event.status === "completed") completed.resolve(); },
    onTransition: record => transitions.push(record),
  });
  await environment.arrivals[0].promise;
  assert.deepEqual(environment.executed, ["left"]);
  await pause(5);
  environment.pending[0].resolve();
  await environment.arrivals[1].promise;
  assert.deepEqual(environment.executed, ["left", "left"]);
  thought.resolve({ candidateId: "corrected", directive: "move to the other side", usage: { total_tokens: 8 } });
  await completed.promise;
  assert.deepEqual(environment.executed, ["left", "left"], "completed advice does not interrupt an active action");
  environment.pending[1].resolve();
  await environment.arrivals[2].promise;
  assert.deepEqual(environment.executed, ["left", "left", "right"]);
  environment.pending[2].resolve();
  const summary = await running;
  assert.equal(summary.success, true);
  assert.equal(summary.asyncMetrics.routedWhileThinking, 2);
  assert.equal(summary.asyncMetrics.corrections, 1);
  assert.equal(summary.asyncMetrics.adviceApplied, 1);
  assert.equal(summary.asyncMetrics.blockedThinkingMs, 0);
  assert.ok(summary.asyncMetrics.resetMs >= 0);
  assert.equal(summary.asyncMetrics.firstActionAfterResetMs, summary.asyncMetrics.firstActionMs - summary.asyncMetrics.resetMs);
  assert.ok(summary.asyncMetrics.overlapMs > 0);
  assert.ok(summary.asyncMetrics.overlapMs <= summary.asyncMetrics.environmentMs);
  assert.deepEqual(summary.usage, { total_tokens: 14 });
  assert.equal(summary.usageComplete, true);
  assert.equal(summary.decisionCalls, 4);
  assert.deepEqual(summary.usageCompleteBySystem, { system0: true, system1: null, system2: true });
  const applied = events.find(event => event.status === "applied");
  assert.equal(applied.step, 2);
  assert.equal(applied.actionId, "action-3");
  assert.deepEqual(transitions.map(record => record.type), ["reset", "step", "step", "step", "episode_end"]);
  assert.equal(transitions[3].route.adviceId, "think-1");
  assert.equal(environment.stopped, 1);
});

test("sync uses the same brain and capabilities but waits before the first action", async () => {
  const environment = fixture({ stepLimit: 1 });
  const thought = Promise.withResolvers();
  const started = Promise.withResolvers();
  const running = runAsyncEpisode({
    environment, system0: system0(), mode: "sync", maxSteps: 1,
    system2: { think: () => { started.resolve(); return thought.promise; } },
  });
  await started.promise;
  await pause(5);
  assert.deepEqual(environment.executed, []);
  thought.resolve({ candidateId: "corrected" });
  const summary = await running;
  assert.deepEqual(environment.executed, ["right"]);
  assert.equal(summary.asyncMetrics.overlapMs, 0);
  assert.equal(summary.asyncMetrics.routedWhileThinking, 0);
  assert.ok(summary.asyncMetrics.blockedThinkingMs > 0);
  assert.ok(summary.asyncMetrics.firstActionMs >= summary.asyncMetrics.blockedThinkingMs);
  assert.equal(summary.asyncMetrics.corrections, 0, "initial deliberate choice is not a mid-episode correction");
});

test("a slow brain is not restarted each step and its next request receives the latest queued observation", async () => {
  const environment = fixture({ stepLimit: 4, holdSteps: true });
  const thoughts = [];
  const secondStarted = Promise.withResolvers();
  const running = runAsyncEpisode({ environment, system0: system0(), maxSteps: 4, system2: { think: input => {
    const deferred = Promise.withResolvers();
    thoughts.push({ ...deferred, step: input.step, signal: input.signal });
    if (thoughts.length === 2) secondStarted.resolve();
    return deferred.promise;
  } } });
  await environment.arrivals[0].promise;
  environment.pending[0].resolve();
  await environment.arrivals[1].promise;
  environment.pending[1].resolve();
  await environment.arrivals[2].promise;
  assert.equal(thoughts.length, 1);
  assert.equal(thoughts[0].signal.aborted, false);
  thoughts[0].resolve({ candidateId: "corrected" });
  await secondStarted.promise;
  assert.deepEqual(thoughts.map(request => request.step), [0, 2]);
  environment.pending[2].resolve();
  await environment.arrivals[3].promise;
  environment.pending[3].resolve();
  const summary = await running;
  assert.equal(summary.asyncMetrics.system2Calls, 2);
  assert.equal(thoughts[1].signal.aborted, true);
  assert.deepEqual(environment.executed, ["left", "left", "left", "right"]);
});

test("advice older than the permitted step age is recorded stale and never applied", async () => {
  const environment = fixture({ stepLimit: 2, holdSteps: true });
  const thought = Promise.withResolvers();
  const events = [];
  const completed = Promise.withResolvers();
  const running = runAsyncEpisode({
    environment, system0: system0(), maxSteps: 2, maxAdviceAgeSteps: 0, thinkEverySteps: 100,
    system2: { think: () => thought.promise },
    onEvent: event => { events.push(event); if (event.status === "completed" && event.type === "thinking") completed.resolve(); },
  });
  await environment.arrivals[0].promise;
  thought.resolve({ candidateId: "corrected" });
  await completed.promise;
  environment.pending[0].resolve();
  await environment.arrivals[1].promise;
  environment.pending[1].resolve();
  const summary = await running;
  assert.deepEqual(environment.executed, ["left", "left"]);
  assert.equal(summary.asyncMetrics.adviceApplied, 0);
  assert.ok(events.some(event => event.type === "thinking" && event.status === "stale" && event.reason === "observation_age"));
  assert.equal(events.some(event => event.status === "applied"), false);
});

test("cancellation stops both lanes and late results cannot publish events or dispatch actions", async () => {
  const environment = fixture({ stepLimit: 1, holdSteps: true });
  const controller = new AbortController();
  const thought = Promise.withResolvers();
  const events = [];
  let thinkingSignal;
  const running = runAsyncEpisode({
    environment, system0: system0(), signal: controller.signal,
    system2: { think: ({ signal }) => { thinkingSignal = signal; return thought.promise; } },
    onEvent: event => events.push(event),
  });
  await environment.arrivals[0].promise;
  controller.abort(new Error("operator cancelled"));
  const summary = await running;
  assert.equal(summary.reason, "cancelled");
  assert.equal(summary.steps, 0);
  assert.equal(summary.actionAttempts, 1);
  assert.equal(thinkingSignal.aborted, true);
  assert.equal(environment.pending[0].signal.aborted, true);
  assert.equal(environment.stopped, 1);
  assert.equal(events.at(-1).type, "episode");
  assert.equal(events.at(-1).status, "cancelled");
  const count = events.length;
  thought.resolve({ candidateId: "corrected" });
  environment.pending[0].resolve();
  await flush();
  assert.equal(events.length, count);
  assert.deepEqual(environment.executed, ["left"]);
});

test("a thinking timeout releases sync routing, while an episode deadline cancels the whole run", async () => {
  const environment = fixture({ stepLimit: 1 });
  const events = [];
  const timedOut = await runAsyncEpisode({
    environment, system0: system0(), mode: "sync", thinkingTimeoutMs: 5,
    system2: { think: () => new Promise(() => {}) }, onEvent: event => events.push(event),
  });
  assert.equal(timedOut.success, true);
  assert.equal(timedOut.asyncMetrics.thinkingFailures, 1);
  assert.ok(events.some(event => event.status === "cancelled" && event.reason === "thinking_timeout"));
  const deadline = await runAsyncEpisode({
    environment: fixture({ holdSteps: true }), system0: system0(), maxDurationMs: 10,
    system2: { think: () => new Promise(() => {}) },
  });
  assert.equal(deadline.reason, "deadline");
  assert.equal(deadline.truncated, true);
  assert.equal(deadline.success, null);
});

test("a router that ignores an alternative proposal does not mark it applied", async () => {
  const events = [];
  const summary = await runAsyncEpisode({
    environment: fixture({ stepLimit: 1 }), system0: system0(), mode: "sync",
    system1: async () => ({ candidateId: "fast", reason: "local feedback takes precedence" }),
    system2: { async think() { return { candidateId: "corrected", directive: "go right" }; } },
    onEvent: event => events.push(event),
  });
  assert.equal(summary.asyncMetrics.adviceApplied, 0);
  assert.equal(events.some(event => event.status === "applied"), false);
  assert.ok(events.some(event => event.type === "thinking" && event.status === "stale" && event.reason === "episode_ended"));
});

test("a new proposal arriving during a decision waits for the following action boundary", async () => {
  const environment = fixture({ stepLimit: 2 });
  const decision = Promise.withResolvers();
  const decisionStarted = Promise.withResolvers();
  const thought = Promise.withResolvers();
  const thinkingCompleted = Promise.withResolvers();
  const fast = capability("fast", "left");
  fast.policy.decide = async () => { decisionStarted.resolve(); await decision.promise; return { action: { name: "move", arguments: { direction: "left" } } }; };
  const running = runAsyncEpisode({
    environment, system0: [fast, capability("corrected", "right")], thinkEverySteps: 100,
    system2: { think: () => thought.promise },
    onEvent: event => { if (event.type === "thinking" && event.status === "completed") thinkingCompleted.resolve(); },
  });
  await decisionStarted.promise;
  thought.resolve({ candidateId: "corrected" });
  await thinkingCompleted.promise;
  decision.resolve();
  const summary = await running;
  assert.deepEqual(environment.executed, ["left", "right"]);
  assert.equal(summary.asyncMetrics.corrections, 1);
});

test("a decision timeout prevents a late System 0 result from executing", async () => {
  const environment = fixture({ stepLimit: 1 });
  const decision = Promise.withResolvers();
  const summary = await runAsyncEpisode({ environment, decisionTimeoutMs: 5, system0: [{ id: "slow", policy: { decide: () => decision.promise } }] });
  decision.resolve({ action: { name: "move", arguments: { direction: "left" } } });
  await flush();
  assert.equal(summary.reason, "deadline");
  assert.equal(environment.executed.length, 0);
});

test("System 2 gets its own thinking budget and failed inference usage remains in the report", async () => {
  let thinkingBudget;
  const summary = await runAsyncEpisode({
    environment: fixture({ stepLimit: 1 }), system0: system0(), mode: "sync", decisionTimeoutMs: 15, thinkingTimeoutMs: 500,
    system2: { async think({ budget }) {
      thinkingBudget = budget;
      throw Object.assign(new Error("model returned an incomplete recommendation"), { usage: { total_tokens: 17 } });
    } },
  });
  assert.equal(thinkingBudget.decisionMs, 500);
  assert.equal(thinkingBudget.thinkingMs, 500);
  assert.deepEqual(summary.usage, { total_tokens: 19 });
  assert.deepEqual(summary.usageBySystem.system2, { total_tokens: 17 });
  assert.equal(summary.usageCompleteBySystem.system2, true);
  assert.equal(summary.asyncMetrics.thinkingFailures, 1);
  assert.equal(summary.success, true);
});

test("same-controller directives count as corrections only when the policy reports using them", async () => {
  for (const consumesDirective of [false, true]) {
    const environment = fixture({ stepLimit: 2, holdSteps: true });
    const thought = Promise.withResolvers();
    const completed = Promise.withResolvers();
    const running = runAsyncEpisode({
      environment, thinkEverySteps: 100,
      system0: [{ id: "fast", policy: { async decide({ directive }) {
        return { action: { name: "move", arguments: { direction: consumesDirective && directive ? "right" : "left" } }, usedAdvice: consumesDirective && Boolean(directive) };
      } } }],
      system2: { think: () => thought.promise },
      onEvent: event => { if (event.type === "thinking" && event.status === "completed") completed.resolve(); },
    });
    await environment.arrivals[0].promise;
    thought.resolve({ candidateId: "fast", directive: "adjust the same controller to the right" });
    await completed.promise;
    environment.pending[0].resolve();
    await environment.arrivals[1].promise;
    environment.pending[1].resolve();
    const summary = await running;
    assert.equal(summary.asyncMetrics.adviceApplied, 1);
    assert.equal(summary.asyncMetrics.routeChanges, 0);
    assert.equal(summary.asyncMetrics.corrections, Number(consumesDirective));
    assert.deepEqual(environment.executed, ["left", consumesDirective ? "right" : "left"]);
  }
});
