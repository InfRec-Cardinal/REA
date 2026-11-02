import test from "node:test";
import assert from "node:assert/strict";
import { createRobotwinCodePolicy } from "../src/policies/robotwin-code.mjs";

test("code policy executes generated Python and keeps usage and source", async () => {
  const source = "move_gripper('left', [0, 0, 1], [1, 0, 0, 0], 1)";
  const command = { arm: "left", position: [0, 0, 1], quaternion: [1, 0, 0, 0], opening: 1 };
  const requests = [];
  const policy = createRobotwinCodePolicy({
    generator: { async decide(input) {
      assert.equal(input.actions[0].name, "run_python");
      return { action: { arguments: { code: source } }, usage: { input_tokens: 12 } };
    } },
    worker: { async request(request) { requests.push(request); return { ok: true, result: { action: command } }; }, stop() {} },
  });
  const observation = { state: { robot_state: {} } };
  const result = await policy.decide({ observation, instruction: "lift" });
  assert.deepEqual(result.action, { name: "move_gripper", arguments: command });
  assert.equal(result.code.source, source);
  assert.equal(result.usage.input_tokens, 12);
  assert.equal(requests[0].observation, observation);
  await policy.close();
});

test("a failed program spends a decision and feeds its error into the next program", async () => {
  const feedback = [];
  const policy = createRobotwinCodePolicy({
    generator: { async decide(input) {
      feedback.push(input.feedback);
      return { action: { arguments: { code: "raise ValueError('bad target')" } } };
    } },
    worker: { async request() { return { ok: false, error: "ValueError: bad target" }; }, stop() {} },
  });
  const first = await policy.decide({ observation: {}, instruction: "lift" });
  await policy.decide({ observation: {}, instruction: "lift" });
  assert.equal(first.action.name, "invalid_output");
  assert.equal(feedback[1].previousCodeError, "ValueError: bad target");
  await policy.close();
});
