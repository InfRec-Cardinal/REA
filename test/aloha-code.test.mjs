import test from "node:test";
import assert from "node:assert/strict";

import { createAlohaCodePolicy } from "../src/policies/aloha-code.mjs";

test("ALOHA code computes a trajectory from measured qpos through a real Python worker", async context => {
  const positions = Array(14).fill(0.1);
  positions[6] = 0.5;
  positions[13] = 0.7;
  const source = "start = np.array(observation['state']['qpos'])\nend = start.copy()\nend[0] += 0.02\njoint_trajectory([start, end])\nprint(instruction)";
  const actions = [{ name: "joint_trajectory", parameters: { type: "object" } }];
  const policy = createAlohaCodePolicy({
    generator: { async decide(input) {
      assert.equal(input.actions[0].name, "run_python");
      assert.match(input.instruction, /ALOHA action schemas/);
      assert.match(input.instruction, /joint_trajectory/);
      assert.equal(input.observation.state.qpos, positions);
      return { action: { name: "run_python", arguments: { code: source } }, usage: { input_tokens: 12 }, notes: "Move left waist a little." };
    } },
  });
  context.after(() => policy.close());
  const result = await policy.decide({ observation: { state: { qpos: positions }, actions }, instruction: "Align the object." });
  assert.equal(result.action.name, "joint_trajectory");
  assert.deepEqual(result.action.arguments.positions[0], positions);
  assert.equal(result.action.arguments.positions[1][0], 0.1 + 0.02);
  assert.deepEqual(result.action.arguments.positions[1].slice(1), positions.slice(1));
  assert.equal(result.code.source, source);
  assert.equal(result.code.stdout, "Align the object.\n");
  assert.equal(result.usage.input_tokens, 12);
  assert.equal(result.notes, "Move left waist a little.");
});

test("ALOHA code errors produce no robot command and expose feedback on the next call", async context => {
  const feedback = [];
  let source = "joint_position([0, 1])";
  const policy = createAlohaCodePolicy({
    generator: { async decide(input) {
      feedback.push(input.feedback);
      return { action: { arguments: { code: source } }, usage: { input_tokens: 4 } };
    } },
  });
  context.after(() => policy.close());
  const input = { observation: { state: { qpos: Array(14).fill(0) } }, instruction: "Hold." };
  await assert.rejects(policy.decide(input), error => error.code === "invalid_action" && /14 absolute positions/.test(error.message));
  source = "joint_position(observation['state']['qpos'])";
  const result = await policy.decide(input);
  assert.match(feedback[1].previousCodeError, /14 absolute positions/);
  assert.equal(result.action.name, "joint_position");
  assert.deepEqual(result.action.arguments.positions, Array(14).fill(0));
});

test("ALOHA code module factory reads its runtime environment and supports worker injection", async context => {
  const requests = [];
  let stopped = false;
  const action = { name: "joint_position", arguments: { positions: Array(14).fill(0) } };
  const policy = createAlohaCodePolicy({
    generator: { async decide() { return { action: { arguments: { code: "joint_position([0] * 14)" } } }; } },
    worker: { async request(payload) { requests.push(payload); return { ok: true, result: { action } }; }, stop() { stopped = true; } },
  }, { env: { REA_CODE_PYTHON: "/unused/injected/python" } });
  context.after(() => policy.close());
  assert.deepEqual((await policy.decide({ observation: { text: "Hold the current position." } })).action, action);
  assert.equal(requests[0].instruction, "Hold the current position.");
  await policy.close();
  assert.equal(stopped, true);
});
