import assert from "node:assert/strict";
import { test } from "node:test";

import {
  ACTION_TABLE,
  MIN_LINEAR_SPEED,
  TOKENS,
  actionOrStop,
  isValidAction,
  legacyAction,
  motionFor,
  parseToken,
  toToken,
} from "../src/actions.mjs";

test("every token resolves to a record whose token field matches its key", () => {
  for (const token of TOKENS) {
    assert.equal(ACTION_TABLE[token].token, token);
  }
});

test("parseToken accepts the token set and rejects everything else", () => {
  assert.equal(parseToken("W"), "W");
  assert.equal(parseToken(" w "), "W");
  assert.equal(parseToken("WW"), null);
  assert.equal(parseToken("Z"), null);
  assert.equal(parseToken(""), null);
  assert.equal(parseToken(null), null);
  assert.equal(parseToken(7), null);
});

test("legacy names round-trip onto tokens", () => {
  assert.equal(toToken("FORWARD"), "W");
  assert.equal(toToken("TURN_LEFT"), "Q");
  assert.equal(toToken("STOP"), "X");
  assert.equal(toToken({ action: "BACKWARD", duration_ms: 600 }), "S");
  assert.equal(toToken("NOT_AN_ACTION"), null);
});

test("unknown input fails closed to a stop", () => {
  assert.equal(actionOrStop("Z").token, "X");
  assert.equal(actionOrStop(null).token, "X");
  assert.equal(actionOrStop(undefined).kind, "stop");
});

test("legacyAction clamps duration into the published range", () => {
  assert.deepEqual(legacyAction("W", 1000), { action: "FORWARD", duration_ms: 1000 });
  assert.deepEqual(legacyAction("W", 9999), { action: "FORWARD", duration_ms: 1500 });
  assert.deepEqual(legacyAction("W", -5), { action: "FORWARD", duration_ms: 0 });
  // strafes have no legacy name and must degrade to STOP rather than inventing one
  assert.equal(legacyAction("A").action, "STOP");
});

test("isValidAction enforces the exact two-key legacy schema", () => {
  assert.ok(isValidAction({ action: "FORWARD", duration_ms: 500 }));
  assert.ok(!isValidAction({ action: "FORWARD", duration_ms: 500, extra: 1 }));
  assert.ok(!isValidAction({ action: "FORWARD" }));
  assert.ok(!isValidAction({ action: "SIDESTEP", duration_ms: 500 }));
  assert.ok(!isValidAction({ action: "FORWARD", duration_ms: 1501 }));
  assert.ok(!isValidAction({ action: "FORWARD", duration_ms: 1.5 }));
  assert.ok(!isValidAction(null));
});

test("motionFor keeps commanded speed above the walking-controller deadband", () => {
  const motion = motionFor("W", 1000);
  assert.ok(Math.abs(motion.vx) >= MIN_LINEAR_SPEED, "vx must clear the deadband");
  assert.equal(motion.duration, 1);
  // backward keeps its sign while still clearing the floor
  assert.ok(motionFor("S", 600).vx <= -MIN_LINEAR_SPEED);
});

test("motionFor clamps the pulse to the safe envelope", () => {
  assert.equal(motionFor("W", 60_000, { maxDurationS: 2 }).duration, 2);
  assert.equal(motionFor("W", 10).duration, 0.4);
});

test("the forward yaw trim is preserved, not inflated to the yaw floor", () => {
  // -0.075 rad/s is a drift correction. Rounding it up to the 0.2 floor would turn a straight
  // walk into a curve, so the floor must not apply to it.
  assert.equal(motionFor("W", 1000).yaw, ACTION_TABLE.W.yaw);
});

test("motionFor returns null for non-velocity tokens", () => {
  assert.equal(motionFor("X"), null);
  assert.equal(motionFor("P"), null);
  assert.equal(motionFor("R"), null);
});
