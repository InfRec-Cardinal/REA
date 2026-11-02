import { test } from "node:test";
import assert from "node:assert/strict";
import { failed, refusedFinish, sameAction, shouldDeliberate, stallReason } from "../eval/behavior/online/behavior_routing.mjs";

function record(step, name, args, result, extra = {}) {
  return { step, action: { name, arguments: args }, info: { result, seconds: 10, ...extra }, success: null };
}
const refused = step => record(step, "finish", { arm: 0 },
  "REFUSED: check_object_in_hand(arm=0) returned false, so arm 0 does not hold the object. The episode continues.", { finish: { accepted: false } });

test("progressing skills do not trigger deliberation", () => {
  const history = [
    record(1, "find_object_base_rotate", { object_name: "red radio" }, "true"),
    record(2, "navigate_to_pose", { x: 1, y: 2, yaw: 0.5 }, "true"),
    record(3, "grasp_object", { object_name: "red radio", candidate_id: 0, arm: 0 }, "true"),
  ];
  assert.equal(stallReason({ history }), null);
  assert.equal(shouldDeliberate({ history }), false);
});

test("three failed actions route to deliberation and name the results", () => {
  const history = [
    record(1, "grasp_object", { object_name: "red radio", candidate_id: 0, arm: 0 }, "ERROR RuntimeError: planning failed"),
    record(2, "grasp_object", { object_name: "red radio", candidate_id: 1, arm: 0 }, "false"),
    record(3, "find_object_torso_rotate", { object_name: "red radio" }, "False"),
  ];
  const reason = stallReason({ history }, 3);
  assert.match(reason, /^The last 3 actions did not do what you intended \(/);
  assert.match(reason, /grasp_object: ERROR RuntimeError: planning failed; grasp_object: false; find_object_torso_rotate: False/);
  assert.match(reason, /another grasp candidate, the other arm, navigate closer, or re-detect the object/);
  assert.equal(shouldDeliberate({ history }, 3), true);
});

test("fewer failures than the patience do not route", () => {
  const history = [
    record(1, "navigate_to_pose", { x: 1, y: 2, yaw: 0 }, "true"),
    record(2, "grasp_object", { object_name: "drill", candidate_id: 0, arm: 1 }, "ERROR: call sample_grasp_pose for this object first"),
    record(3, "grasp_object", { object_name: "drill", candidate_id: 2, arm: 1 }, "none"),
  ];
  assert.equal(stallReason({ history }, 3), null);
  assert.match(stallReason({ history }, 2), /^The last 2 actions/);
});

test("a repeated action is caught regardless of argument order", () => {
  const history = [
    record(1, "navigate_to_pose", { x: 1, y: 2, yaw: 0.5 }, "true"),
    record(2, "navigate_to_pose", { yaw: 0.5, y: 2, x: 1 }, "true"),
  ];
  assert.match(stallReason({ history }), /^Your last action repeated the previous one \(navigate_to_pose/);
  assert.equal(sameAction(history[0], history[1]), true);
  assert.equal(sameAction(history[0], record(3, "navigate_to_pose", { x: 1.2, y: 2, yaw: 0.5 }, "true")), false);
});

test("a refused finish is explained", () => {
  const history = [record(1, "lift_arm", { arm: 0 }, "true"), refused(2)];
  const reason = stallReason({ history });
  assert.match(reason, /^finish was refused: check_object_in_hand\(0\) was False/);
  assert.equal(refusedFinish(history[1]), true);
  assert.equal(failed(history[1]), true);
});

test("an accepted finish is not a failure", () => {
  const accepted = record(1, "finish", { arm: 1 }, "ACCEPTED: check_object_in_hand(arm=1) is True; the episode ends now.", { finish: { accepted: true } });
  assert.equal(refusedFinish(accepted), false);
  assert.equal(failed(accepted), false);
});

test("invalid decisions, not-found and none count as failures; numbers and objects do not", () => {
  assert.equal(failed(record(1, "invalid_output", {}, "NO ACTION: the decision produced no valid take_action call; nothing ran.")), true);
  assert.equal(failed(record(1, "find_object_base_rotate", { object_name: "x" }, "not found")), true);
  assert.equal(failed(record(1, "find_object_base_rotate", { object_name: "x" }, "None")), true);
  assert.equal(failed(record(1, "lift_arm", { arm: 0 }, "null")), false);
  assert.equal(failed(record(1, "navigate_to_pose", { x: 0, y: 0, yaw: 0 }, '{"x": 1}')), false);
  // two invalid outputs in a row are failures, not a repeat
  const history = [record(1, "invalid_output", {}, "NO ACTION: x"), record(2, "invalid_output", {}, "NO ACTION: x")];
  assert.equal(stallReason({ history }), null);
});

test("an empty history does not route", () => {
  assert.equal(stallReason({ history: [] }), null);
});
