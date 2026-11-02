import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEPTH_FLOOR_CENTER_M,
  DEPTH_FLOOR_SIDE_M,
  depthGate,
  nextArmHeld,
  verdict,
} from "../src/gate.mjs";
import HOLONOMIC from "./fixtures/profiles/holonomic.mjs";
import { defineProfile } from "../src/profiles/define.mjs";
import { G1_PROFILE } from "../src/profiles/g1.mjs";
import { LIBERO_PROFILE } from "../src/profiles/libero.mjs";

const READY = { ready: true, reasons: [], fsm_id: 801, soc: 80 };
const NOT_READY = { ready: false, reasons: ["FSM 4 is not in allowed set [801]"] };
const CLEAR_LIDAR = { usable: true, blocked: false, reason: "clear", center: 2.4, left: 1.9, right: 1.8 };
const NEAR_LIDAR = { usable: true, blocked: true, reason: "lidar_center_near", center: 0.6, left: 1.9, right: 1.8 };
const STALE_LIDAR = { usable: false, blocked: true, reason: "lidar_stale" };
const CLEAR_DEPTH = depthGate({ center_m: 1.8, left_m: 1.5, right_m: 1.6 });
const NEAR_DEPTH = depthGate({ center_m: 0.3, left_m: 1.5, right_m: 1.6 });

test("depth veto floors cannot be lowered past contact range", () => {
  const gate = depthGate({ center_m: 0.3, left_m: 1, right_m: 1 }, { nearCenterM: 0.01, nearSideM: 0.01 });
  assert.equal(gate.near_center_m, DEPTH_FLOOR_CENTER_M);
  assert.equal(gate.near_side_m, DEPTH_FLOOR_SIDE_M);
  // 0.30 m clears the 0.25 m floor, so this must be allowed rather than silently clamped shut
  assert.equal(gate.blocked, false);
});

test("an explicit opt-in can lower the veto for contact work", () => {
  assert.equal(depthGate({ center_m: 0.4, left_m: 1, right_m: 1 }).blocked, true);
  assert.equal(depthGate({ center_m: 0.4, left_m: 1, right_m: 1 }, { nearCenterM: 0.3 }).blocked, false);
});

test("missing depth is treated as blocked, not as clear", () => {
  const gate = depthGate(null);
  assert.equal(gate.blocked, true);
  assert.equal(gate.usable, false);
  assert.equal(gate.reason, "depth_missing");
});

test("stopping is always allowed, even with no evidence at all", () => {
  const result = verdict({ token: "X", state: NOT_READY, depth: depthGate(null), lidar: STALE_LIDAR });
  assert.equal(result.allowed, true);
  assert.deepEqual(result.reasons, []);
});

test("a robot that is not walk-ready cannot move", () => {
  const result = verdict({ token: "W", state: NOT_READY, depth: CLEAR_DEPTH, lidar: CLEAR_LIDAR });
  assert.equal(result.allowed, false);
  assert.ok(result.reasons.some((reason) => reason.startsWith("state:")));
});

test("forward is vetoed by depth and by lidar independently", () => {
  assert.equal(verdict({ token: "W", state: READY, depth: CLEAR_DEPTH, lidar: CLEAR_LIDAR }).allowed, true);
  assert.equal(verdict({ token: "W", state: READY, depth: NEAR_DEPTH, lidar: CLEAR_LIDAR }).allowed, false);
  assert.equal(verdict({ token: "W", state: READY, depth: CLEAR_DEPTH, lidar: NEAR_LIDAR }).allowed, false);
});

test("turning in place is not blocked by an obstacle straight ahead", () => {
  // The whole point of a turn when the front is blocked: it is the escape, not the hazard.
  assert.equal(verdict({ token: "Q", state: READY, depth: NEAR_DEPTH, lidar: NEAR_LIDAR }).allowed, true);
  assert.equal(verdict({ token: "E", state: READY, depth: NEAR_DEPTH, lidar: NEAR_LIDAR }).allowed, true);
});

test("a turn still needs a usable lidar unless lidar is switched off", () => {
  assert.equal(verdict({ token: "Q", state: READY, depth: CLEAR_DEPTH, lidar: STALE_LIDAR }).allowed, false);
  assert.equal(
    verdict({ token: "Q", state: READY, depth: CLEAR_DEPTH, lidar: STALE_LIDAR, useLidar: false }).allowed,
    true,
  );
});

test("strafing is gated on the side it moves toward", () => {
  const leftBlocked = depthGate({ center_m: 2, left_m: 0.1, right_m: 2 });
  assert.equal(verdict({ token: "A", state: READY, depth: leftBlocked, lidar: CLEAR_LIDAR }).allowed, false);
  assert.equal(verdict({ token: "D", state: READY, depth: leftBlocked, lidar: CLEAR_LIDAR }).allowed, true);
});

test("a second holding arm pose is refused until the first is released", () => {
  assert.equal(verdict({ token: "P", state: READY, armHeld: null }).allowed, true);
  assert.equal(verdict({ token: "P", state: READY, armHeld: "U" }).allowed, false);
  // releasing is always available, otherwise a held pose could never be cleared
  assert.equal(verdict({ token: "R", state: READY, armHeld: "U" }).allowed, true);
});

test("arm interlock state tracks holding and release", () => {
  assert.equal(nextArmHeld(null, "P"), "P");
  assert.equal(nextArmHeld("P", "R"), null);
  assert.equal(nextArmHeld("P", "W"), "P");
});

test("verdict() fails closed when the depth is missing or was never gated", () => {
  for (const depth of [undefined, null, { center_m: 2.0, left_m: 2.0, right_m: 2.0 }]) {
    for (const token of ["W", "A", "D"]) {
      const result = verdict({ token, state: READY, depth, lidar: CLEAR_LIDAR });
      assert.equal(result.allowed, false, `${token} with ${JSON.stringify(depth)}`);
      assert.ok(result.reasons.includes("depth:depth_missing"));
    }
    assert.equal(verdict({ token: "X", state: READY, depth, lidar: CLEAR_LIDAR }).allowed, true, "a stop is never refused");
    assert.equal(verdict({ token: "Q", state: READY, depth, lidar: CLEAR_LIDAR }).allowed, true, "a turn does not need depth");
  }
  assert.equal(verdict({ token: "W", state: READY, depth: depthGate(null), lidar: CLEAR_LIDAR }).reasons[0], "depth:depth_missing", "the gated form of a missing depth says the same thing");
});

test("a profile without lidar or turns is gated on depth alone and refuses with its own stop token", () => {
  const near = depthGate({ center_m: 0.2, left_m: 2, right_m: 2 }, { profile: HOLONOMIC });
  const clear = depthGate({ center_m: 2, left_m: 2, right_m: 2 }, { profile: HOLONOMIC });
  assert.equal(near.near_center_m, 0.5, "the profile's own threshold");
  assert.equal(verdict({ token: "H", state: NOT_READY, depth: near, lidar: STALE_LIDAR, profile: HOLONOMIC }).allowed, true);
  const refused = verdict({ token: "F", state: READY, depth: near, lidar: STALE_LIDAR, profile: HOLONOMIC });
  assert.equal(refused.allowed, false);
  assert.deepEqual(refused.reasons, ["depth:depth_center_near"], "no lidar reason: the profile has no lidar");
  assert.equal(verdict({ token: "F", state: READY, depth: clear, lidar: STALE_LIDAR, profile: HOLONOMIC }).allowed, true);
  const leftBlocked = depthGate({ center_m: 2, left_m: 0.1, right_m: 2 }, { profile: HOLONOMIC });
  assert.equal(verdict({ token: "L", state: READY, depth: leftBlocked, profile: HOLONOMIC }).allowed, false);
  assert.equal(verdict({ token: "R", state: READY, depth: leftBlocked, profile: HOLONOMIC }).allowed, true);
  assert.equal(verdict({ token: "W", state: READY, depth: clear, profile: HOLONOMIC }).token, "H", "an unknown token is this profile's stop");
  assert.equal(nextArmHeld(null, "F", HOLONOMIC), null, "no arms, nothing is ever held");
  assert.equal(nextArmHeld("Z", "F", HOLONOMIC), "Z", "and an existing value is carried through untouched");
});

test("a profile without sensing checks readiness only", () => {
  assert.equal(verdict({ token: "W", state: READY, depth: undefined, lidar: undefined, profile: LIBERO_PROFILE }).allowed, true);
  assert.equal(verdict({ token: "W", state: NOT_READY, profile: LIBERO_PROFILE }).allowed, false);
  assert.equal(verdict({ token: "X", state: NOT_READY, profile: LIBERO_PROFILE }).allowed, true);
});

// A lidar-only base: no depth camera at all, so the forward token's only evidence is the lidar sector.
const LIDAR_ONLY = defineProfile({
  id: "lidar-only-test",
  name: "lidar-only test base",
  TOKENS: ["H", "F"],
  ACTION_TABLE: {
    H: { token: "H", name: "HALT", kind: "stop", action: "stop", vx: 0, vy: 0, yaw: 0, durationS: 0, defaultMs: 0, veto: { depth: null, lidar: null } },
    F: { token: "F", name: "FORWARD", kind: "velocity", action: "move", vx: 0.2, vy: 0, yaw: 0, durationS: 1, defaultMs: 1000, veto: { depth: null, lidar: "center" } },
  },
  ACTIONS: [],
  roles: { stop: "H", forward: "F" },
  motion: ["F"],
  opposite: {},
  arms: null,
  envelope: { minLinear: 0.1, minYaw: 0.1, pulseRangeS: [0.4, 2.0] },
  sensing: { depth: null, lidar: { ...G1_PROFILE.sensing.lidar }, buckets: { depth: [0.6, 1.5] } },
  capabilities: { boxLift: null, manipulation: { gripper_available: false } },
  legacyDurationMaxMs: null,
  TOKEN_LEGEND: "H halt F forward",
});

test("a lidar-only profile is vetoed by its own lidar sector, with no depth camera to fall back on", () => {
  // The veto sector is the token's veto.lidar, not veto.depth: this profile has no depth veto at all.
  const refused = verdict({ token: "F", state: READY, depth: undefined, lidar: NEAR_LIDAR, profile: LIDAR_ONLY });
  assert.equal(refused.allowed, false, "a lidar_center_near scan must refuse forward");
  assert.deepEqual(refused.reasons, ["lidar:lidar_center_near"]);
  assert.equal(verdict({ token: "F", state: READY, depth: undefined, lidar: CLEAR_LIDAR, profile: LIDAR_ONLY }).allowed, true);
  assert.deepEqual(
    verdict({ token: "F", state: READY, lidar: STALE_LIDAR, profile: LIDAR_ONLY }).reasons,
    ["lidar:lidar_stale", "lidar:lidar_stale"],
    "an unusable and blocked scan reports both the usability and the sector refusal",
  );
  assert.equal(verdict({ token: "H", state: NOT_READY, lidar: NEAR_LIDAR, profile: LIDAR_ONLY }).allowed, true);
});

test("a side lidar veto reads the sector it was given, not the depth sector", () => {
  const sideProfile = defineProfile({
    id: "lidar-side-test",
    name: "lidar-side test base",
    TOKENS: ["H", "F"],
    ACTION_TABLE: {
      H: { token: "H", name: "HALT", kind: "stop", action: "stop", vx: 0, vy: 0, yaw: 0, durationS: 0, defaultMs: 0, veto: { depth: null, lidar: null } },
      F: { token: "F", name: "SIDESTEP", kind: "velocity", action: "move", vx: 0, vy: 0.2, yaw: 0, durationS: 1, defaultMs: 1000, veto: { depth: null, lidar: "right" } },
    },
    ACTIONS: [],
    roles: { stop: "H", strafeRight: "F" },
    motion: ["F"],
    opposite: {},
    arms: null,
    envelope: { minLinear: 0.1, minYaw: 0.1, pulseRangeS: [0.4, 2.0] },
    sensing: { depth: null, lidar: { ...G1_PROFILE.sensing.lidar }, buckets: { depth: [0.6, 1.5] } },
    capabilities: { boxLift: null, manipulation: { gripper_available: false } },
    legacyDurationMaxMs: null,
    TOKEN_LEGEND: "H halt F sidestep",
  });
  const rightNear = { usable: true, blocked: false, reason: "clear", center: 2.4, left: 0.1, right: 0.2 };
  const leftNear = { usable: true, blocked: false, reason: "clear", center: 2.4, left: 0.2, right: 1.8 };
  assert.deepEqual(verdict({ token: "F", state: READY, lidar: rightNear, profile: sideProfile }).reasons, ["lidar:lidar_right_near"]);
  assert.equal(verdict({ token: "F", state: READY, lidar: leftNear, profile: sideProfile }).allowed, true, "the left sector is not this token's business");
});
