import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import { test } from "node:test";

import { defineProfile, isProfile, ProfileError, profileByName, resolveProfile, G1_PROFILE, LIBERO_PROFILE } from "../src/profiles/index.mjs";
import * as actions from "../src/actions.mjs";
import * as libero from "../src/libero-actions.mjs";
import HOLONOMIC from "./fixtures/profiles/holonomic.mjs";

test("the G1 profile is the constants the facade still exports", () => {
  assert.equal(isProfile(G1_PROFILE), true);
  assert.deepEqual(G1_PROFILE.TOKENS, ["X", "W", "S", "A", "D", "Q", "E", "U", "P", "R"]);
  assert.equal(actions.TOKENS, G1_PROFILE.TOKENS);
  assert.equal(actions.ACTION_TABLE, G1_PROFILE.ACTION_TABLE);
  assert.equal(actions.parseToken, G1_PROFILE.parseToken);
  assert.equal(actions.MIN_LINEAR_SPEED, 0.2);
  assert.deepEqual(actions.HOLDING_TOKENS, ["U", "P"]);
  assert.equal(actions.RELEASE_TOKEN, "R");
  assert.deepEqual(G1_PROFILE.roles, { stop: "X", forward: "W", backward: "S", strafeLeft: "A", strafeRight: "D", turnLeft: "Q", turnRight: "E" });
  assert.deepEqual(G1_PROFILE.ACTION_TABLE.W.veto, { depth: "center", lidar: "center" });
  assert.deepEqual(G1_PROFILE.ACTION_TABLE.A.veto, { depth: "left", lidar: "left" });
  assert.deepEqual(G1_PROFILE.ACTION_TABLE.Q.veto, { depth: null, lidar: "usable" });
  assert.deepEqual(G1_PROFILE.ACTION_TABLE.U.veto, { depth: null, lidar: null });
  assert.equal(G1_PROFILE.sensing.lidar.snapshot, "/run/user/1000/g1-livox/obstacles.json");
  assert.deepEqual(G1_PROFILE.sensing.buckets.depth, [0.6, 1.5]);
  assert.equal(Object.isFrozen(G1_PROFILE.ACTION_TABLE.W), true);
  assert.deepEqual(G1_PROFILE.legacyAction("A"), { action: "STOP", duration_ms: 600 }, "strafes have no legacy name");
  assert.deepEqual(G1_PROFILE.motionFor("W", 5000), { vx: 0.3, vy: 0, yaw: -0.075, duration: 2 });
});

test("the LIBERO profile is the manipulation contract the facade still exports", () => {
  assert.equal(libero.LIBERO_ACTION_CONTRACT, LIBERO_PROFILE);
  assert.equal(libero.TOKENS, LIBERO_PROFILE.TOKENS);
  assert.equal(LIBERO_PROFILE.TOKENS.length, 15);
  assert.equal(LIBERO_PROFILE.roles.stop, "X");
  assert.equal(LIBERO_PROFILE.arms, null);
  assert.equal(LIBERO_PROFILE.sensing, null);
  assert.equal(LIBERO_PROFILE.motionFor, null);
  assert.deepEqual(LIBERO_PROFILE.legacyAction("O"), { action: "OPEN_GRIPPER", duration_ms: 250 });
  assert.deepEqual(LIBERO_PROFILE.legacyAction("O", 9000), { action: "OPEN_GRIPPER", duration_ms: 9000 }, "no legacy clamp");
  assert.equal(LIBERO_PROFILE.manipulation.gripper_available, true);
  assert.equal(typeof LIBERO_PROFILE.legend.short, "string");
});

test("the holonomic fixture profile has no turns, arms or lidar", () => {
  assert.deepEqual(HOLONOMIC.TOKENS, ["H", "F", "L", "R"]);
  assert.equal(HOLONOMIC.roles.stop, "H");
  assert.equal(HOLONOMIC.roles.turnLeft, undefined);
  assert.equal(HOLONOMIC.sensing.lidar, null);
  assert.equal(HOLONOMIC.actionOrStop("W").token, "H", "an unknown token resolves to this profile's stop");
  assert.equal(HOLONOMIC.parseToken("f"), "F");
});

test("defineProfile rejects each malformed field with the field's name", () => {
  const base = () => structuredClone({
    id: "t", TOKENS: ["H", "F"], roles: { stop: "H", forward: "F" }, motion: ["F"],
    ACTION_TABLE: {
      H: { token: "H", name: "HALT", kind: "stop", defaultMs: 0 },
      F: { token: "F", name: "FORWARD", kind: "velocity", vx: 0.1, vy: 0, yaw: 0, durationS: 1, defaultMs: 1000 },
    },
  });
  const failing = [
    [spec => { spec.TOKENS = ["H", "ff"]; }, /TOKENS/u],
    [spec => { spec.TOKENS = ["H", "H"]; }, /TOKENS.*unique/u],
    [spec => { delete spec.ACTION_TABLE.F; }, /ACTION_TABLE\.F/u],
    [spec => { spec.ACTION_TABLE.F.token = "H"; }, /ACTION_TABLE\.F\.token/u],
    [spec => { spec.roles.stop = "Z"; }, /roles\.stop/u],
    [spec => { spec.roles.jump = "F"; }, /roles\.jump/u],
    [spec => { spec.motion = ["H"]; }, /motion/u],
    [spec => { spec.opposite = { F: "H" }; }, /opposite/u],
    [spec => { spec.arms = { holding: ["Z"], release: "H" }; }, /arms\.holding/u],
    [spec => { spec.ACTION_TABLE.F.veto = { depth: "center", lidar: null }; }, /veto/u],
    [spec => { spec.sensing = { depth: { nearCenterM: 0.5, nearSideM: 0.3, floorCenterM: 0.6, floorSideM: 0.2 }, lidar: null }; }, /floorCenterM/u],
    [spec => { spec.envelope = { minLinear: 0.1, minYaw: 0.1, pulseRangeS: [2, 1] }; }, /pulseRangeS/u],
    [spec => { spec.legacyDurationMaxMs = -1; }, /legacyDurationMaxMs/u],
  ];
  assert.equal(isProfile(defineProfile(base())), true, "the base spec is valid");
  for (const [mutate, pattern] of failing) {
    const spec = base();
    mutate(spec);
    assert.throws(() => defineProfile(spec), error => error instanceof ProfileError && pattern.test(error.message), pattern.source);
  }
  assert.equal(defineProfile(G1_PROFILE), G1_PROFILE, "an already defined profile passes through");
});

test("profiles resolve by name, by identity and by module path", async () => {
  assert.equal(profileByName(undefined), G1_PROFILE);
  assert.equal(profileByName("g1"), G1_PROFILE);
  assert.equal(profileByName("libero"), LIBERO_PROFILE);
  assert.equal(profileByName("go2"), null);
  assert.equal(await resolveProfile("libero"), LIBERO_PROFILE);
  assert.equal(await resolveProfile(G1_PROFILE), G1_PROFILE);
  const fromModule = await resolveProfile("./fixtures/profiles/holonomic.mjs", { baseDir: fileURLToPath(new URL(".", import.meta.url)) });
  assert.equal(fromModule, HOLONOMIC);
  await assert.rejects(resolveProfile(42), ProfileError);
});
