import { defineProfile } from "./define.mjs";

// The Unitree G1 profile: the single-character tokens the vision model answers with, the sport-service
// motion each one dispatches, the arm interlock, the walking-controller deadband and the gate's depth
// and lidar thresholds. These are measured values from this robot; a second robot gets its own file.
const velocity = (token, name, vx, vy, yaw, durationS, defaultMs, veto) =>
  ({ token, name, kind: "velocity", action: "move", vx, vy, yaw, durationS, defaultMs, veto });
const arm = (token, name, primitive, actionId, holding) =>
  ({ token, name, kind: "arm_primitive", action: primitive, primitive, actionId, holding, defaultMs: 0, veto: { depth: null, lidar: null } });

export const G1_PROFILE = defineProfile({
  id: "unitree-g1",
  name: "Unitree G1",
  TOKENS: ["X", "W", "S", "A", "D", "Q", "E", "U", "P", "R"],
  ACTIONS: ["STOP", "FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT"],
  ACTION_TABLE: {
    X: { token: "X", name: "STOP", kind: "stop", action: "stop", vx: 0, vy: 0, yaw: 0, durationS: 0, defaultMs: 0, veto: { depth: null, lidar: null } },
    // W carries a small yaw trim: the G1 drifts left on a pure-vx command (measured on this robot).
    W: velocity("W", "FORWARD", 0.3, 0, -0.075, 2.0, 1000, { depth: "center", lidar: "center" }),
    S: velocity("S", "BACKWARD", -0.3, 0, 0, 0.6, 600, { depth: null, lidar: "usable" }),
    A: velocity("A", "STRAFE_LEFT", 0, 0.3, 0, 0.6, 600, { depth: "left", lidar: "left" }),
    D: velocity("D", "STRAFE_RIGHT", 0, -0.3, 0, 0.6, 600, { depth: "right", lidar: "right" }),
    Q: velocity("Q", "TURN_LEFT", 0, 0, 0.35, 0.6, 500, { depth: null, lidar: "usable" }),
    E: velocity("E", "TURN_RIGHT", 0, 0, -0.35, 0.6, 500, { depth: null, lidar: "usable" }),
    U: arm("U", "ARM_UP", "both_hands_up", 18, true),
    P: arm("P", "ARM_PUSH", "forward_push", 36, true),
    R: arm("R", "ARM_RELEASE", "release_arm", 99, false),
  },
  roles: { stop: "X", forward: "W", backward: "S", strafeLeft: "A", strafeRight: "D", turnLeft: "Q", turnRight: "E" },
  motion: ["W", "S", "A", "D", "Q", "E"],
  opposite: { Q: "E", E: "Q", A: "D", D: "A", W: "S", S: "W" },
  arms: { holding: ["U", "P"], release: "R" },
  // Speeds sit above the walking-controller deadband on purpose: below about 0.15 m/s (0.2 rad/s)
  // the high-level controller accepts the RPC and never takes a step.
  envelope: { minLinear: 0.2, minYaw: 0.2, pulseRangeS: [0.4, 2.0] },
  sensing: {
    depth: { nearCenterM: 0.55, nearSideM: 0.35, floorCenterM: 0.25, floorSideM: 0.2 },
    lidar: {
      snapshot: "/run/user/1000/g1-livox/obstacles.json",
      maxAgeS: 0.6,
      // Azimuth windows in degrees around the forward axis, not radii: the MID-360 on this robot is
      // self-occluded on the right by the torso.
      center: [-12, 12],
      front: [-30, 30],
      left: [20, 60],
      right: [-40, -15],
      nearCenterM: 0.9,
      nearSideM: 0.45,
      minPointsPerS: 5000,
    },
    buckets: { depth: [0.6, 1.5] },
  },
  capabilities: {
    boxLift: { available: false, reason: "No validated grasp-and-lift primitive is available; clamping is not grasping." },
    manipulation: null,
  },
  legacyDurationMaxMs: 1500,
  TOKEN_LEGEND: {
    full: [
      "X = stop",
      "W = forward at 0.30 m/s for 2.00 s with -0.075 rad/s yaw trim",
      "S = backward at 0.30 m/s for 0.60 s",
      "A = strafe left at 0.30 m/s for 0.60 s",
      "D = strafe right at 0.30 m/s for 0.60 s",
      "Q = turn left at 0.35 rad/s for 0.60 s",
      "E = turn right at 0.35 rad/s for 0.60 s",
      "U = both_hands_up gesture",
      "P = forward_push gesture",
      "R = release/reset arms",
    ].join("\n"),
    short: "X stop W fwd S back A left D right Q turnL E turnR U armsUp P push R release",
    min: "X W S A D Q E U P R",
  },
});
