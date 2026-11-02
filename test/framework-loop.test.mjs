import test from "node:test";
import assert from "node:assert/strict";

import { createRegistry } from "../src/framework/registry.mjs";
import { createRoutedPolicy } from "../src/framework/routing.mjs";
import { loadLoopConfig } from "../src/framework/loop/config.mjs";
import { armEvidence } from "../src/framework/loop/evidence.mjs";
import { moved, stalledSteps } from "../src/framework/loop/progress.mjs";
import { screenAction } from "../src/framework/loop/gate.mjs";
import { judgeCompletion } from "../src/framework/loop/judge.mjs";
import { planRecovery, yawRotate } from "../src/framework/loop/recover.mjs";

const C = loadLoopConfig();
const Q = [0.707, 0, 0.707, 0];
const INSTRUCTION = "Hold the light brown cylindrical roller with both arms.";

// Shapes and magnitudes follow RoboTwin interface v2 (grab_roller 075, Astra run).
function state({ left = [-0.12, -0.419, 0.947], right = [0.193, -0.272, 0.879], lg = 0, rg = 0, lf = 0, rf = 0, jitter = 0 } = {}) {
  return {
    left_endpose: [...left.map(v => v + jitter), ...Q], right_endpose: [...right.map(v => v - jitter), ...Q],
    left_gripper: lg, right_gripper: rg,
    tcp_poses: { left: [left[0], left[1], left[2] - 0.12], right: [right[0], right[1], right[2] - 0.12] },
    measured_finger_joint_positions: { left: [lf, lf], right: [rf, rf] },
  };
}
const move = (arm, position, opening, quaternion = Q) => ({ name: "move_gripper", arguments: { arm, position, quaternion, opening } });
const record = (action, before, after, extra = {}) => ({ action, observation: { state: before }, nextObservation: { state: after }, success: false, reward: null, info: {}, ...extra });

test("progress ignores sub-millimetre jitter but sees real motion", () => {
  assert.equal(moved(state(), state({ jitter: 0.0004 }), C.progress), false);
  assert.equal(moved(state(), state({ left: [-0.12, -0.419, 0.9] }), C.progress), true);
  assert.equal(moved(state({ lg: 1 }), state({ lg: 0 }), C.progress), true);
  assert.equal(moved({ q: [0.1, 0.2] }, { q: [0.1002, 0.2] }, C.progress), false);
  const history = [1, 2, 3].map(i => record(move("left", [-0.12, -0.419, 0.947], 0), state(), state({ jitter: 0.0001 * i })));
  assert.equal(stalledSteps(history, C.progress), 3);
});

test("evidence separates a held object from an empty closure", () => {
  const held = armEvidence(state({ lf: 0.0166 }), null, C.judge);
  assert.equal(held.left.grasp, "held");
  assert.equal(held.right.grasp, "empty");
  assert.equal(armEvidence(state({ lg: 1, lf: 0.045 }), null, C.judge).left.grasp, "open");
  const reported = armEvidence(state(), { gripper_execution: { left: { fingers: [{ actual_m: 0.0178 }] } } }, C.judge);
  assert.equal(reported.left.grasp, "held");
});

test("gate blocks near-zero and repeated commands and reads idle claims", () => {
  const s = state();
  const idle = { action: move("left", [-0.1204, -0.419, 0.947], 0), thought: "keep moving" };
  assert.equal(screenAction({ decision: idle, state: s, history: [], config: C.gate, progress: C.progress }).verdict, "near_zero");
  const claim = { action: move("left", [-0.1204, -0.419, 0.947], 0), thought: "Hold achieved; maintaining the grasp." };
  assert.equal(screenAction({ decision: claim, state: s, history: [], config: C.gate, progress: C.progress }).verdict, "hold_claim");
  const target = move("right", [0.25, -0.3, 0.86], 1);
  const history = [record(target, state(), state({ jitter: 0.0002 }))];
  assert.equal(screenAction({ decision: { action: target }, state: s, history, config: C.gate, progress: C.progress }).verdict, "repeat");
  assert.equal(screenAction({ decision: { action: move("left", [-0.12, -0.419, 0.947], 1) }, state: s, history: [], config: C.gate, progress: C.progress }).verdict, "allow");
});

test("judge rejects the grab_roller false completions", () => {
  const evidenceOf = r => armEvidence(r.nextObservation.state, null, C.judge);
  const empty = judgeCompletion({ instruction: INSTRUCTION, evidence: armEvidence(state(), null, C.judge), evidenceOf, config: C.judge });
  assert.deepEqual([empty.verdict, empty.reason], ["not_complete", "empty_grasp"]);
  const oneArm = judgeCompletion({ instruction: INSTRUCTION, evidence: armEvidence(state({ lf: 0.0166 }), null, C.judge), evidenceOf, config: C.judge });
  assert.deepEqual([oneArm.verdict, oneArm.reason, oneArm.arm], ["not_complete", "missing_arm", "right"]);
  const low = state({ lf: 0.017, rf: 0.017 });
  const notLifted = judgeCompletion({ instruction: INSTRUCTION, evidence: armEvidence(low, null, C.judge), history: [record(null, low, low)], evidenceOf, config: C.judge });
  assert.deepEqual([notLifted.verdict, notLifted.reason], ["not_complete", "not_lifted"]);
  const high = state({ left: [-0.12, -0.419, 1.0], right: [0.193, -0.272, 0.93], lf: 0.017, rf: 0.017 });
  const done = judgeCompletion({ instruction: INSTRUCTION, evidence: armEvidence(high, null, C.judge), history: [record(null, low, low)], evidenceOf, config: C.judge });
  assert.equal(done.verdict, "complete");
  for (const instruction of ["Grab the sauce can with smooth surface and place it beside the kitchenpot for stovetop boiling", "Lift the gray kitchenpot with rounded edges with both arms and put it on the stove"]) {
    assert.equal(judgeCompletion({ instruction, evidence: armEvidence(state(), null, C.judge), evidenceOf, config: C.judge }).verdict, "uncertain", instruction);
  }
  assert.equal(judgeCompletion({ instruction: "Open the microwave.", evidence: armEvidence(state(), null, C.judge), evidenceOf, config: C.judge }).verdict, "uncertain");
});

test("recovery regrasps from the last closing pose with a new yaw", () => {
  const history = [record(move("left", [-0.12, -0.42, 0.8], 0), state(), state())];
  const plan = planRecovery({ verdict: { reason: "empty_grasp", arm: "left" }, evidence: armEvidence(state(), null, C.judge), history, attempt: 0, config: C.recover });
  assert.equal(plan.macro, "regrasp");
  assert.equal(plan.actions.length, 5);
  assert.equal(plan.actions[0].arguments.opening, 1);
  assert.ok(plan.actions[0].arguments.position[2] > 0.947);
  assert.deepEqual(plan.actions[3].arguments.position.slice(0, 2), [-0.12, -0.42]);
  assert.equal(plan.actions[3].arguments.opening, 0);
  assert.notDeepEqual(plan.actions[2].arguments.quaternion, Q);
  const q = yawRotate([1, 0, 0, 0], Math.PI);
  assert.ok(Math.abs(Math.abs(q[3]) - 1) < 1e-9);
  const directive = planRecovery({ verdict: { reason: "missing_arm", arm: "right" }, evidence: armEvidence(state({ lf: 0.0166 }), null, C.judge), history, attempt: 0, config: C.recover });
  assert.equal(directive.macro, "directive");
  assert.match(directive.directive, /right arm/);
});

test("routed loop turns an unverified hold into a recovery macro and then deliberates", async () => {
  const calls = [];
  const hold = { action: move("left", [-0.12, -0.419, 0.947], 0), thought: "Hold achieved." };
  const reflex = { id: "reflex", async decide(ctx) { calls.push(["reflex", ctx.guidance ?? null]); return hold; } };
  const deliberate = { id: "deliberate", async decide(ctx) { calls.push(["deliberate", ctx.guidance ?? null]); return hold; } };
  const routed = createRoutedPolicy({ reflex, deliberate, loop: true });
  await routed.reset({});
  const s = state();
  const history = [record(move("left", [-0.12, -0.42, 0.8], 0), s, s)];
  const first = await routed.decide({ observation: { state: s }, history, instruction: INSTRUCTION, budget: { stepsRemaining: 30 } });
  assert.equal(first.route, "recover");
  assert.equal(first.nodes.judge.reason, "empty_grasp");
  assert.equal(first.nodes.versions.judge, "judge@2");
  const rest = [];
  for (let i = 0; i < 4; i += 1) rest.push(await routed.decide({ observation: { state: s }, history, instruction: INSTRUCTION, budget: { stepsRemaining: 29 - i } }));
  assert.ok(rest.every(r => r.route === "recover"));
  assert.equal(calls.length, 1);
});

test("routed loop re-deliberates a blocked no-op within the same decision", async () => {
  const s = state({ lg: 1, rg: 1, lf: 0.045, rf: 0.045 });
  const reflex = { id: "reflex", async decide() { return { action: move("left", [-0.12, -0.419, 0.947], 1), thought: "adjusting" }; } };
  const deliberate = { id: "deliberate", async decide(ctx) { assert.match(ctx.guidance, /rejected before execution/); return { action: move("left", [-0.12, -0.419, 0.85], 1) }; } };
  const routed = createRoutedPolicy({ reflex, deliberate, loop: { route: { maxDeliberateFraction: 1 } } });
  await routed.reset({});
  const result = await routed.decide({ observation: { state: s }, history: [], instruction: INSTRUCTION, budget: { stepsRemaining: 40 } });
  assert.equal(result.route, "deliberate");
  assert.equal(result.nodes.gate.verdict, "allow");
  assert.equal(result.nodes.route.calls, 2);
});

test("legacy routing escalates on jittering stalls with stallTolerance; the default keeps exact state equality", async () => {
  const make = options => createRoutedPolicy({
    reflex: { id: "r", async decide() { return { action: { name: "noop", arguments: {} } }; } },
    deliberate: { id: "d", async decide() { return { action: { name: "noop", arguments: {} } }; } },
    maxDeliberateFraction: 1,
    ...options,
  });
  const history = [1, 2, 3].map(i => record({ name: "noop", arguments: {} }, { q: [0.1, 0.2] }, { q: [0.1 + 0.0001 * i, 0.2] }));
  assert.equal((await make({ stallTolerance: true }).decide({ history })).route, "deliberate");
  assert.equal((await make({}).decide({ history })).route, "reflex", "jitter is motion for the default exact-equality test");
  assert.equal((await make({ stallTolerance: { vector: 0.00001 } }).decide({ history })).route, "reflex", "a tighter tolerance sees the jitter as motion");
  const frozen = [1, 2, 3].map(() => record({ name: "noop", arguments: {} }, { q: [0.1, 0.2] }, { q: [0.1, 0.2] }));
  assert.equal((await make({}).decide({ history: frozen })).route, "deliberate");
});

test("the judge reads the task instruction when a progress wrapper rewrites context.instruction", async () => {
  const hold = { action: move("left", [-0.12, -0.419, 0.947], 0), thought: "Hold achieved." };
  const fixed = id => ({ id, async decide() { return hold; } });
  const s = state();
  const history = [record(move("left", [-0.12, -0.42, 0.8], 0), s, s)];
  const context = { observation: { state: s }, history, instruction: INSTRUCTION, budget: { stepsRemaining: 30 } };
  const direct = createRoutedPolicy({ reflex: fixed("reflex"), deliberate: fixed("deliberate"), loop: true, progress: true });
  await direct.reset({});
  const first = await direct.decide(context);
  assert.equal(first.route, "recover");
  assert.equal(first.nodes.judge.reason, "empty_grasp");
  const registry = createRegistry();
  registry.registerPolicy("fixed", ({ label }) => fixed(label));
  const routed = await registry.createPolicy({ type: "routed", options: { reflex: { type: "fixed", options: { label: "fast" } }, deliberate: { type: "fixed", options: { label: "careful" } }, loop: true } });
  await routed.reset({});
  const viaRegistry = await routed.decide(context);
  assert.equal(viaRegistry.route, "recover");
  assert.deepEqual([viaRegistry.nodes.judge.verdict, viaRegistry.nodes.judge.reason], ["not_complete", "empty_grasp"]);
  await routed.close();
});

test("a progress review that falls on a recovery macro step deliberates once the macro is done", async () => {
  const calls = [];
  const hold = { action: move("left", [-0.12, -0.419, 0.947], 0), thought: "Hold achieved." };
  const reflex = { id: "reflex", async decide() { calls.push("reflex"); return hold; } };
  const deliberate = { id: "deliberate", async decide() { calls.push("deliberate"); return { action: move("left", [-0.12, -0.419, 0.85], 1) }; } };
  const routed = createRoutedPolicy({ reflex, deliberate, loop: true });
  await routed.reset({});
  const s = state();
  const history = [record(move("left", [-0.12, -0.42, 0.8], 0), s, s)];
  const step = (extra = {}) => routed.decide({ observation: { state: s }, history, instruction: INSTRUCTION, budget: { stepsRemaining: 30 }, ...extra });
  assert.equal((await step()).route, "recover");
  assert.equal((await step({ progressReview: { required: true, reasons: ["periodic"] } })).route, "recover");
  for (let i = 0; i < 3; i += 1) assert.equal((await step()).route, "recover");
  const after = await step();
  assert.equal(after.route, "deliberate");
  assert.equal(after.nodes.route.signals.requested, true);
  assert.deepEqual(calls, ["reflex", "deliberate"]);
});

test("loop config rejects unknown nodes and id changes", () => {
  assert.throws(() => loadLoopConfig({ memory: {} }), /unknown loop node/);
  assert.throws(() => loadLoopConfig({ judge: { id: "other" } }), /has id/);
  assert.equal(loadLoopConfig({ judge: { version: 2, heldMinGap: 0.008 } }).judge.heldMinGap, 0.008);
});
