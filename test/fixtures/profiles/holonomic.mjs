import { defineProfile } from "../../../src/profiles/define.mjs";

// A four-token omnidirectional base with a depth camera and no lidar, no turns and no arms. It exists
// so the tests can prove no core module assumes X, W or a lidar snapshot.
const velocity = (token, name, vx, vy, veto) => ({ token, name, kind: "velocity", action: "move", vx, vy, yaw: 0, durationS: 1, defaultMs: 1000, veto });

export default defineProfile({
  id: "holonomic-test",
  name: "holonomic test base",
  TOKENS: ["H", "F", "L", "R"],
  ACTION_TABLE: {
    H: { token: "H", name: "HALT", kind: "stop", action: "stop", vx: 0, vy: 0, yaw: 0, durationS: 0, defaultMs: 0, veto: { depth: null, lidar: null } },
    F: velocity("F", "FORWARD", 0.2, 0, { depth: "center", lidar: null }),
    L: velocity("L", "LEFT", 0, 0.2, { depth: "left", lidar: null }),
    R: velocity("R", "RIGHT", 0, -0.2, { depth: "right", lidar: null }),
  },
  ACTIONS: [],
  roles: { stop: "H", forward: "F", strafeLeft: "L", strafeRight: "R" },
  motion: ["F", "L", "R"],
  opposite: { L: "R", R: "L" },
  arms: null,
  envelope: { minLinear: 0.1, minYaw: 0.1, pulseRangeS: [0.4, 2.0] },
  sensing: { depth: { nearCenterM: 0.5, nearSideM: 0.3, floorCenterM: 0.25, floorSideM: 0.2 }, lidar: null, buckets: { depth: [0.6, 1.5] } },
  capabilities: { boxLift: null, manipulation: { gripper_available: false } },
  legacyDurationMaxMs: null,
  TOKEN_LEGEND: "H halt F forward L left R right",
});
