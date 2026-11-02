import assert from "node:assert/strict";
import test from "node:test";

import { compactProgressHistory, createProgressPolicy } from "../src/framework/progress.mjs";
import { createRoutedPolicy } from "../src/framework/routing.mjs";

const action = { name: "hold", arguments: { position: [0.2, 0.1, 0.4], opening: 1 } };

test("progress review keeps task evidence compact and detects repeated holds", async () => {
  const contexts = [];
  const parent = {
    id: "parent",
    async decide(context) { contexts.push(context); return { action, claim: { status: "working", evidence: "still checking" } }; },
  };
  const policy = createProgressPolicy({ policy: parent, reviewInterval: 99, repeatWindow: 3, patience: 3 });
  const history = [1, 2, 3].map(step => ({ step, action, thought: "old conclusion", observation: { state: { frame: step } }, info: { execution: { status: "completed", selected_target_reached: false } } }));
  const result = await policy.decide({ observation: { instruction: "Place the object", state: { frame: 4 } }, history, instruction: "Place the object" });
  assert.equal(result.review.required, true);
  assert.deepEqual(result.review.reasons, ["repeated_action"]);
  assert.match(contexts[0].instruction, /Place the object/u);
  assert.match(contexts[0].instruction, /Review task progress now/u);
  assert.equal(contexts[0].history, history);
  assert.equal(contexts[0].historyForPrompt[0].thought, undefined);
  assert.equal(contexts[0].historyForPrompt[0].observation, undefined);
  assert.equal(contexts[0].historyForPrompt.at(-1).info.execution.selected_target_reached, false);
  assert.deepEqual(result.action, action);
});

test("a completion claim requests the next decision's evidence check without ending an episode", async () => {
  const parent = { calls: 0, async decide() { this.calls += 1; return { action, claim: { status: "complete", evidence: "the gripper appears closed" } }; } };
  const policy = createProgressPolicy({ policy: parent, reviewInterval: 99, repeatWindow: 99 });
  const first = await policy.decide({ observation: {}, history: [] });
  const history = [{ step: 1, action, claim: first.claim, info: { execution: { status: "completed" } } }];
  const second = await policy.decide({ observation: {}, history });
  assert.equal(parent.calls, 2);
  assert.deepEqual(second.review.reasons, ["completion_claim"]);
  assert.deepEqual(second.action, action);
});

test("a required progress review can escalate the routed policy once without a second model call", async () => {
  const calls = [];
  const make = label => ({ id: label, async decide() { calls.push(label); return { action }; } });
  const routed = createRoutedPolicy({ reflex: make("reflex"), deliberate: make("deliberate"), maxDeliberateFraction: 1, progress: false });
  const result = await routed.decide({ history: [], progressReview: { required: true } });
  assert.equal(result.route, "deliberate");
  assert.deepEqual(calls, ["deliberate"]);
  assert.deepEqual(compactProgressHistory([{ step: 1, action, info: { execution: { status: "completed" } } }]), [{ step: 1, action, info: { execution: { status: "completed" } } }]);
});
