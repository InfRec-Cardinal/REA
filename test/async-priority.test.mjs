import assert from "node:assert/strict";
import test from "node:test";

import { runAsyncEpisode } from "../src/framework/async-runtime.mjs";

const MOVE = {
  name: "move",
  parameters: {
    type: "object", properties: { direction: { type: "string", enum: ["left", "right"] } },
    required: ["direction"], additionalProperties: false,
  },
};
const flush = () => new Promise(resolve => setImmediate(resolve));
const action = direction => ({ name: "move", arguments: { direction } });
const capability = (id, direction) => ({ id, policy: { async decide() { return { action: action(direction) }; } } });

function heldEnvironment(limit, onDispatch = () => {}) {
  let phase = 0;
  const observation = () => ({ id: String(phase), state: { phase }, actions: [MOVE], timestampMs: Date.now() });
  const arrivals = Array.from({ length: limit }, () => Promise.withResolvers());
  const releases = Array.from({ length: limit }, () => Promise.withResolvers());
  return {
    id: "execution-priority-fixture", arrivals, releases, executed: [], signals: [], stopped: 0,
    async reset() { return { observation: observation() }; },
    async step(command, { signal }) {
      const index = this.executed.length;
      this.executed.push(command.arguments.direction);
      this.signals.push(signal);
      onDispatch();
      arrivals[index].resolve();
      await releases[index].promise;
      phase += 1;
      return { observation: observation(), terminated: phase === limit };
    },
    async stop() { this.stopped += 1; },
  };
}

function lifetime(context, environment) {
  const controller = new AbortController();
  context.after(() => {
    controller.abort(new Error("test cleanup"));
    for (const release of environment.releases) release.resolve();
  });
  return controller;
}

test("execution priority lets System0 decide and dispatch before starting System2, and cancellation does not wait for it", { timeout: 5000 }, async context => {
  const order = [];
  const environment = heldEnvironment(2, () => order.push("dispatch"));
  const controller = lifetime(context, environment);
  const decisionStarted = Promise.withResolvers();
  const decision = Promise.withResolvers();
  const thinkingStarted = Promise.withResolvers();
  const thought = Promise.withResolvers();
  let thinkingSignal;
  const running = runAsyncEpisode({
    environment, maxSteps: 2, thinkingSchedule: "after_dispatch", signal: controller.signal,
    system0: [{ id: "fast", policy: { async decide() {
      order.push("decide");
      decisionStarted.resolve();
      await decision.promise;
      return { action: action("left") };
    } } }],
    system2: { think({ signal }) {
      order.push("think");
      thinkingSignal = signal;
      thinkingStarted.resolve();
      return thought.promise;
    } },
  });
  await decisionStarted.promise;
  await flush();
  assert.deepEqual(order, ["decide"], "background inference must not start ahead of System0's first action");
  decision.resolve();
  await thinkingStarted.promise;
  assert.deepEqual(order, ["decide", "dispatch", "think"]);
  controller.abort(new Error("operator cancelled"));
  const summary = await running;
  assert.equal(summary.reason, "cancelled");
  assert.equal(thinkingSignal.aborted, true);
  assert.equal(environment.signals[0].aborted, true);
  assert.equal(environment.stopped, 1);
  thought.resolve({ candidateId: "fast" });
  environment.releases[0].resolve();
  await flush();
  assert.deepEqual(environment.executed, ["left"], "late advice does not restart the action loop");
});

test("execution priority skips busy thinking slots without a queued restart, then uses the next dispatch's observation", { timeout: 5000 }, async context => {
  const environment = heldEnvironment(6);
  const controller = lifetime(context, environment);
  const thoughts = [];
  const started = Array.from({ length: 2 }, () => Promise.withResolvers());
  const completed = Promise.withResolvers();
  const running = runAsyncEpisode({
    environment, maxSteps: 6, thinkingSchedule: "after_dispatch", thinkEverySteps: 2, signal: controller.signal,
    system0: [capability("fast", "left")],
    system2: { think(input) {
      const reply = Promise.withResolvers();
      thoughts.push({ ...reply, input });
      started[thoughts.length - 1]?.resolve();
      return reply.promise;
    } },
    onEvent(event) { if (event.type === "thinking" && event.status === "completed") completed.resolve(); },
  });
  await started[0].promise;
  environment.releases[0].resolve();
  await environment.arrivals[1].promise;
  environment.releases[1].resolve();
  await environment.arrivals[2].promise;
  assert.equal(thoughts.length, 1, "an occupied thinking slot is skipped");
  thoughts[0].resolve({ candidateId: "fast" });
  await completed.promise;
  await flush();
  assert.equal(thoughts.length, 1, "finishing a request must not immediately launch queued work");
  environment.releases[2].resolve();
  await environment.arrivals[3].promise;
  assert.equal(thoughts.length, 1, "an unscheduled action does not start thinking");
  environment.releases[3].resolve();
  await started[1].promise;
  assert.equal(environment.executed.length, 5, "System0 dispatched the next eligible action first");
  assert.deepEqual(thoughts.map(({ input }) => [input.step, input.observation.state.phase]), [[0, 0], [4, 4]]);
  environment.releases[4].resolve();
  await environment.arrivals[5].promise;
  environment.releases[5].resolve();
  const summary = await running;
  assert.equal(summary.error, null);
  assert.equal(summary.asyncMetrics.system2Calls, 2);
  assert.equal(summary.asyncMetrics.blockedThinkingMs, 0);
  assert.equal(thoughts[1].input.signal.aborted, true, "episode completion cancels unfinished background work");
});

test("the default selector gives each controller four completed action blocks before accepting a switch", { timeout: 5000 }, async context => {
  const environment = heldEnvironment(9);
  const controller = lifetime(context, environment);
  const completed = Array.from({ length: 9 }, () => Promise.withResolvers());
  const transitions = [];
  const running = runAsyncEpisode({
    environment, maxSteps: 9, thinkingSchedule: "after_dispatch", minControllerSteps: 4, signal: controller.signal,
    system0: [capability("fast", "left"), capability("corrected", "right")],
    system2: { async think({ step }) { return { candidateId: step < 4 ? "corrected" : "fast" }; } },
    onEvent(event) {
      if (event.type === "thinking" && event.status === "completed") completed[event.sourceStep].resolve();
    },
    onTransition: record => transitions.push(record),
  });
  for (let index = 0; index < 9; index += 1) {
    await completed[index].promise;
    environment.releases[index].resolve();
  }
  const summary = await running;
  assert.equal(summary.error, null);
  assert.deepEqual(environment.executed, ["left", "left", "left", "left", "right", "right", "right", "right", "left"]);
  assert.equal(summary.asyncMetrics.routeChanges, 2);
  const steps = transitions.filter(record => record.type === "step");
  assert.deepEqual(steps.slice(1, 4).map(record => record.route.deferredCandidateId), ["corrected", "corrected", "corrected"]);
  assert.deepEqual(steps.slice(5, 8).map(record => record.route.deferredCandidateId), ["fast", "fast", "fast"]);
  assert.equal(steps[4].route.candidateId, "corrected");
  assert.equal(steps[8].route.candidateId, "fast");
});
