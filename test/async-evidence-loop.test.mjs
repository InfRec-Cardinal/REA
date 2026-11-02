import assert from "node:assert/strict";
import test from "node:test";

import { runAsyncEpisode } from "../src/framework/async-runtime.mjs";
import { createProgressPolicy } from "../src/framework/progress.mjs";

const Q = [0.707, 0, 0.707, 0];
const MOVE = { name: "move_gripper", parameters: { type: "object", properties: { arm: { type: "string", enum: ["left", "right"] }, position: { type: "array", items: { type: "number" } }, quaternion: { type: "array", items: { type: "number" } }, opening: { type: "number" } }, required: ["arm", "position", "quaternion", "opening"], additionalProperties: false } };
const INSTRUCTION = "Hold the light brown cylindrical roller with both arms.";

// A bimanual pose robot whose grippers close on nothing (empty closure), as in grab_roller 075.
function robot() {
  const pose = { left: [-0.12, -0.419, 0.947], right: [0.193, -0.272, 0.879] };
  const opening = { left: 0, right: 0 };
  const executed = [];
  const observation = () => ({
    id: String(executed.length), actions: [MOVE], timestampMs: Date.now(),
    state: {
      left_endpose: [...pose.left, ...Q], right_endpose: [...pose.right, ...Q], left_gripper: opening.left, right_gripper: opening.right,
      measured_finger_joint_positions: { left: [opening.left * 0.045, opening.left * 0.045], right: [opening.right * 0.045, opening.right * 0.045] },
    },
  });
  return {
    id: "pose-robot", executed,
    async reset() { return { observation: observation(), success: false }; },
    async step(action) {
      const { arm, position, opening: open } = action.arguments;
      executed.push(action.arguments);
      pose[arm] = position.slice(0, 3);
      opening[arm] = open;
      return { observation: observation(), success: false, info: { execution: { status: "target_reached" } } };
    },
  };
}

const holder = { id: "holder", kind: "sdk", policy: { async decide({ observation }) {
  const [x, y, z] = observation.state.left_endpose;
  return { action: { name: "move_gripper", arguments: { arm: "left", position: [x + 0.0003, y, z], quaternion: Q, opening: 0 } }, thought: "Hold achieved; keeping the roller." };
} } };

test("async loop judges an idle hold claim and runs the regrasp macro without the policy", async () => {
  const environment = robot();
  const events = [];
  const result = await runAsyncEpisode({ environment, system0: [holder], instruction: INSTRUCTION, maxSteps: 5, loop: true, onEvent: event => events.push(event) });
  assert.equal(environment.executed.length, 5);
  assert.equal(environment.executed[0].opening, 1, "first recovery step reopens");
  assert.ok(environment.executed[0].position[2] > 0.947, "and retreats upward");
  assert.ok(events.some(event => event.type === "loop" && event.status === "judged" && event.reason === "empty_grasp"));
  assert.equal(result.steps, 5);
});

test("controller dwell does not keep an internal recovery macro selected after it finishes", async () => {
  let decisions = 0;
  const result = await runAsyncEpisode({
    environment: robot(), instruction: INSTRUCTION, maxSteps: 7, loop: true,
    minControllerSteps: 10, mode: "sync",
    system0: [{ ...holder, policy: { async decide(input) { decisions += 1; return holder.policy.decide(input); } } }],
    system2: { async think() { return { candidateId: "holder" }; } },
  });
  assert.equal(result.error, null);
  assert.equal(result.steps, 7);
  assert.ok(decisions > 1, "the real controller can be selected again after recovery");
});

test("async loop blocks a no-op, consumes budget, and asks System2 immediately with guidance", async () => {
  const environment = robot();
  const guidance = [];
  const idle = { id: "idle", kind: "sdk", policy: { async decide({ observation }) {
    const [x, y, z] = observation.state.right_endpose;
    return { action: { name: "move_gripper", arguments: { arm: "right", position: [x, y, z + 0.0002], quaternion: Q, opening: observation.state.right_gripper } }, thought: "adjust" };
  } } };
  const result = await runAsyncEpisode({
    environment, system0: [idle], instruction: "Press the red button.", maxSteps: 2, thinkEverySteps: 100, mode: "sync", loop: true,
    system2: { async think(input) { guidance.push(input.guidance); return { candidateId: "idle", directive: "move", reason: "test" }; } },
  });
  assert.equal(environment.executed.length, 0, "near-zero commands never reach the robot");
  assert.equal(result.steps, 2);
  assert.ok(guidance.some(text => /rejected before execution/.test(text ?? "")));
});

test("async runtime without loop is unchanged", async () => {
  const environment = robot();
  await runAsyncEpisode({ environment, system0: [holder], instruction: INSTRUCTION, maxSteps: 2 });
  assert.equal(environment.executed.length, 2);
  assert.equal(environment.executed[0].opening, 0);
});

test("async runtime without loop accepts a System0 capability whose kind is recover", async () => {
  const environment = robot();
  const result = await runAsyncEpisode({ environment, system0: [{ ...holder, kind: "recover" }], instruction: INSTRUCTION, maxSteps: 1 });
  assert.equal(result.reason, "max_steps", JSON.stringify(result.error));
  assert.equal(environment.executed.length, 1);
});

test("async transitions record claims and progress reviews, so a System0 completion claim is reviewed next", async () => {
  const contexts = [];
  const parent = { id: "parent", async decide(context) {
    contexts.push(context);
    const [x, y, z] = context.observation.state.left_endpose;
    const status = contexts.length === 1 ? "complete" : "working";
    return { action: { name: "move_gripper", arguments: { arm: "left", position: [x, y, z + 0.05], quaternion: Q, opening: 0 } }, claim: { status, evidence: "gripper closed" } };
  } };
  const transitions = [];
  const result = await runAsyncEpisode({
    environment: robot(), instruction: "Press the red button.", maxSteps: 2, onTransition: record => { transitions.push(record); },
    system0: [{ id: "chat", kind: "policy", policy: createProgressPolicy({ policy: parent, reviewInterval: 99, repeatWindow: 99 }) }],
  });
  assert.equal(result.steps, 2, JSON.stringify(result.error));
  const steps = transitions.filter(record => record.type === "step");
  assert.deepEqual(steps[0].claim, { status: "complete", evidence: "gripper closed" });
  assert.deepEqual(steps[0].review, { required: false, reasons: [] });
  assert.deepEqual(contexts[1].progressReview, { required: true, reasons: ["completion_claim"] });
  assert.deepEqual(steps[1].review.reasons, ["completion_claim"]);
});
