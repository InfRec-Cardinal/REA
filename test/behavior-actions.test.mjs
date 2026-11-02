import assert from "node:assert/strict";
import test from "node:test";

import {
  R1PRO_ACTION_SPEC, R1PRO_ACTION_LAYOUT, R1PRO_STATE_LAYOUT,
  describeR1ProState, createR1ProHoldAction,
} from "../src/environments/behavior-actions.mjs";
import { validateAction } from "../src/framework/actions.mjs";

test("R1Pro observation naming keeps the official 61-value field order", () => {
  const state = Float32Array.from({ length: 61 }, (_, index) => index);
  const named = describeR1ProState(state);
  assert.deepEqual(named.base_qvel, [0, 1, 2]);
  assert.deepEqual(named.arm_left_qpos, [3, 4, 5, 6, 7, 8, 9]);
  assert.deepEqual(named.eef_left_quat, [20, 21, 22, 23]);
  assert.deepEqual(named.gripper_left_qpos, [24, 25]);
  assert.deepEqual(named.arm_right_qpos, [28, 29, 30, 31, 32, 33, 34]);
  assert.deepEqual(named.gripper_right_qvel, [51, 52]);
  assert.deepEqual(named.trunk_qpos, [53, 54, 55, 56]);
  assert.deepEqual(Object.values(named).flat(), Array.from(state));
  named.arm_left_qpos[0] = 1000;
  assert.equal(state[3], 3);
  assert.equal(R1PRO_STATE_LAYOUT.reduce((count, field) => count + field.size, 0), 61);
});

test("holding uses observed pose rather than joint zeros, velocities or recorded base motion", () => {
  const state = Array(61).fill(0);
  state.splice(0, 3, 0.6, -0.2, 0.4);
  const left = [0.1, 0.2, 0.3, -0.4, 0.5, 0.6, 0.7];
  const right = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7];
  const torso = [1.025, -1.45, -0.47, 0];
  state.splice(3, 7, ...left);
  state.splice(10, 7, ...Array(7).fill(4));
  state.splice(24, 2, 0.05, 0.05);
  state.splice(28, 7, ...right);
  state.splice(49, 2, 0.01, 0.04);
  state.splice(53, 4, ...torso);
  const copy = [...state];
  const action = createR1ProHoldAction(state);
  assert.deepEqual(action.arguments.values, [0, 0, 0, ...torso, ...left, 1, ...right, 0]);
  assert.deepEqual(validateAction(action, [R1PRO_ACTION_SPEC]), action);
  assert.deepEqual(state, copy);
  assert.deepEqual(R1PRO_ACTION_LAYOUT.map(group => [group.name, group.offset, group.size]), [
    ["base", 0, 3], ["torso", 3, 4], ["left_arm", 7, 7],
    ["left_gripper", 14, 1], ["right_arm", 15, 7], ["right_gripper", 22, 1],
  ]);
});

test("action schema enforces controller normalization without treating arm radians as normalized", () => {
  const action = createR1ProHoldAction(Array(61).fill(0));
  action.arguments.values[7] = -2;
  assert.deepEqual(validateAction(action, [R1PRO_ACTION_SPEC]), action);
  for (const index of [0, 1, 2, 14, 22]) {
    const invalid = structuredClone(action);
    invalid.arguments.values[index] = 1.2;
    assert.throws(() => validateAction(invalid, [R1PRO_ACTION_SPEC]), /<= 1/);
  }
  const shortened = { name: "joint_action", arguments: { values: action.arguments.values.slice(1) } };
  assert.throws(() => validateAction(shortened, [R1PRO_ACTION_SPEC]), /at least 23/);
  action.arguments.values[3] = NaN;
  assert.throws(() => validateAction(action, [R1PRO_ACTION_SPEC]), /must be a number/);
});

test("raw-state errors are reported before constructing a hold baseline", () => {
  assert.throws(() => describeR1ProState(Array(60).fill(0)), /61 values/);
  const state = Array(61).fill(0);
  state[3] = "0.1";
  assert.throws(() => createR1ProHoldAction(state), /finite numbers/);
});
