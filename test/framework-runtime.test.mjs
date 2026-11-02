import assert from "node:assert/strict";
import test from "node:test";

import { validateAction } from "../src/framework/actions.mjs";
import { runEpisode } from "../src/framework/runtime.mjs";
import { createRoutedPolicy } from "../src/framework/routing.mjs";

const CLICK = { name: "click", parameters: { type: "object", properties: { x: { type: "integer", minimum: 0, maximum: 63 }, y: { type: "integer", minimum: 0, maximum: 63 } }, required: ["x", "y"], additionalProperties: false } };
const MOVE = { name: "joint_target", parameters: { type: "object", properties: { position: { type: "array", items: { type: "number", minimum: -2, maximum: 2 }, minItems: 2, maxItems: 2 } }, required: ["position"], additionalProperties: false } };
const FINISH = { name: "confirm", parameters: { type: "object", properties: {}, additionalProperties: false } };

function environment(initialAction = CLICK) {
  let step = 0;
  const observation = () => ({ id: String(step), timestampMs: Date.now() - 30, text: "public observation", state: { phase: step }, actions: [step === 0 ? initialAction : FINISH] });
  return {
    id: "fixture",
    timing: "turn-based",
    executed: [],
    stopped: 0,
    async reset() {
      step = 0;
      return { observation: observation(), success: false };
    },
    async step(action) {
      this.executed.push(action);
      step += 1;
      return { observation: observation(), terminated: step === 2, success: step === 2, reward: step === 2 ? 1 : 0 };
    },
    async stop() { this.stopped += 1; },
  };
}

test("one runtime handles coordinate actions, dynamic action sets, notes and authoritative success", async () => {
  const fixture = environment();
  const transitions = [];
  const policy = {
    id: "scripted",
    async decide({ observation, actions, notes, history }) {
      if (observation.state.phase === 0) {
        assert.equal(notes, "");
        return { action: { name: "click", arguments: { x: 21, y: 9 } }, notes: "click moved to phase 1", usage: { total_tokens: 10 } };
      }
      assert.equal(actions[0].name, "confirm");
      assert.equal(notes, "click moved to phase 1");
      assert.equal(history[0].action.arguments.x, 21);
      return { action: { name: "confirm" }, usage: { total_tokens: 20 } };
    },
  };
  const summary = await runEpisode({ environment: fixture, policy, onTransition: record => transitions.push(record) });
  assert.equal(summary.success, true);
  assert.equal(summary.steps, 2);
  assert.equal(summary.reason, "success");
  assert.deepEqual(summary.usage, { total_tokens: 30 });
  assert.equal(summary.usageComplete, true);
  assert.equal(fixture.stopped, 1);
  assert.equal(transitions[1].timing.observationAgeAtDispatchMs >= 30, true);
  assert.deepEqual(transitions.map(record => record.type), ["reset", "step", "step", "episode_end"]);
});

test("continuous vectors use the same loop without token conversion", async () => {
  const fixture = environment(MOVE);
  const summary = await runEpisode({
    environment: fixture,
    policy: { async decide({ observation }) { return { action: observation.state.phase ? { name: "confirm" } : { name: "joint_target", arguments: { position: [0.2, -1.3] } } }; } },
  });
  assert.equal(summary.success, true);
  assert.deepEqual(fixture.executed[0].arguments.position, [0.2, -1.3]);
  assert.equal(summary.usage, null);
  assert.equal(summary.usageComplete, false);
});

test("invalid coordinates are surfaced instead of executing a substituted action", async () => {
  const fixture = environment();
  const summary = await runEpisode({ environment: fixture, policy: { async decide() { return { action: { name: "click", arguments: { x: 99, y: 0 } } }; } } });
  assert.equal(summary.reason, "error");
  assert.equal(summary.error.code, "invalid_action");
  assert.equal(summary.success, null);
  assert.equal(fixture.executed.length, 0);
  assert.throws(() => validateAction({ name: "joint_target", arguments: { position: [1, 2, 3] } }, [MOVE]), /at most 2/);
});

test("timeouts prevent late decisions from executing and report truncation", async () => {
  const fixture = environment();
  const delayed = Promise.withResolvers();
  const summary = await runEpisode({ environment: fixture, decisionTimeoutMs: 10, policy: { decide: () => delayed.promise } });
  delayed.resolve({ action: { name: "click", arguments: { x: 1, y: 1 } } });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(summary.reason, "deadline");
  assert.equal(summary.truncated, true);
  assert.equal(summary.success, null);
  assert.equal(summary.modelCalls, 1);
  assert.equal(summary.usageComplete, false);
  assert.equal(fixture.executed.length, 0);
  assert.equal(fixture.stopped, 1);
});

test("step budget does not turn an unfinished episode into a win", async () => {
  const fixture = environment();
  const summary = await runEpisode({ environment: fixture, maxSteps: 1, policy: { async decide() { return { action: { name: "click", arguments: { x: 1, y: 1 } } }; } } });
  assert.equal(summary.reason, "max_steps");
  assert.equal(summary.steps, 1);
  assert.equal(summary.truncated, true);
  assert.equal(summary.terminated, false);
  assert.equal(summary.success, false);
});

test("reusing an environment resets episode notes and rejects overlapping runs", async () => {
  const fixture = environment();
  const pending = Promise.withResolvers();
  const running = runEpisode({ environment: fixture, maxSteps: 1, policy: { decide: () => pending.promise } });
  await assert.rejects(runEpisode({ environment: fixture, policy: {} }), /already has an episode/);
  pending.resolve({ action: { name: "click", arguments: { x: 1, y: 1 } }, notes: "prior episode" });
  await running;
  await runEpisode({ environment: fixture, maxSteps: 1, policy: { async decide({ notes, history }) {
    assert.equal(notes, "");
    assert.equal(history.length, 0);
    return { action: { name: "click", arguments: { x: 1, y: 1 } } };
  } } });
});

test("routing can escalate on task-specific signals without robot semantics", async () => {
  const routed = createRoutedPolicy({
    reflex: { id: "small", async decide() { return { action: { name: "click", arguments: { x: 0, y: 0 } } }; } },
    deliberate: { id: "large", async decide() { return { action: { name: "click", arguments: { x: 3, y: 5 } } }; } },
    shouldDeliberate: ({ notes }) => notes === "need new hypothesis",
  });
  await routed.reset({});
  const first = await routed.decide({ history: [], notes: "" });
  const second = await routed.decide({ history: [], notes: "need new hypothesis" });
  assert.equal(first.model, "small");
  assert.equal(second.model, "large");
});

test("runtime preserves policy claims and progress reviews in transitions", async () => {
  const fixture = environment();
  const transitions = [];
  let calls = 0;
  await runEpisode({
    environment: fixture,
    onTransition: transition => transitions.push(transition),
    policy: { async decide({ observation }) {
      calls += 1;
      return {
        action: observation.state.phase === 0 ? { name: "click", arguments: { x: 1, y: 1 } } : { name: "confirm" },
        claim: calls === 1 ? { status: "complete", evidence: "the object is visible above the tray" } : undefined,
        review: calls === 2 ? { required: true, reasons: ["completion_claim"] } : undefined,
      };
    } },
  });
  assert.deepEqual(transitions[1].claim, { status: "complete", evidence: "the object is visible above the tray" });
  assert.equal(transitions[1].review, undefined);
  assert.deepEqual(transitions[2].review, { required: true, reasons: ["completion_claim"] });
});

test("parameter validation respects heterogeneous actuator bounds and positive amplitudes", () => {
  const command = { name: "step", parameters: { type: "object", properties: {
    action: { type: "array", prefixItems: [{ type: "number", minimum: -1, maximum: 1 }, { type: "integer", minimum: 0, maximum: 3 }], minItems: 2, maxItems: 2 },
    amplitude: { type: "number", exclusiveMinimum: 0, maximum: 1 },
  }, required: ["action"] } };
  assert.deepEqual(validateAction({ name: "step", arguments: { action: [0.5, 3], amplitude: 0.1 } }, [command]).arguments.action, [0.5, 3]);
  assert.throws(() => validateAction({ name: "step", arguments: { action: [0.5, 4] } }, [command]), /<= 3/);
  assert.throws(() => validateAction({ name: "step", arguments: { action: [0.5, 2], amplitude: 0 } }, [command]), /must be > 0/);
});
