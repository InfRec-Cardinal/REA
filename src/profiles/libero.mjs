import { defineProfile } from "./define.mjs";

export const REPEAT_STEPS = 5;
export const ACTION_AMPLITUDE = 0.5;

const definitions = [
  ["X", "HOLD", "hold", "hold", "Hold the current pose and retain gripper state; this does not declare success"],
  ["W", "WORLD_X_POSITIVE", "osc_pose", "world_x_positive", "Move along world +x"],
  ["S", "WORLD_X_NEGATIVE", "osc_pose", "world_x_negative", "Move along world -x"],
  ["A", "WORLD_Y_POSITIVE", "osc_pose", "world_y_positive", "Move along world +y"],
  ["D", "WORLD_Y_NEGATIVE", "osc_pose", "world_y_negative", "Move along world -y"],
  ["U", "WORLD_Z_POSITIVE", "osc_pose", "world_z_positive", "Move along world +z (up)"],
  ["R", "WORLD_Z_NEGATIVE", "osc_pose", "world_z_negative", "Move along world -z (down)"],
  ["Q", "YAW_POSITIVE", "osc_pose", "yaw_positive", "Rotate yaw positively"],
  ["E", "YAW_NEGATIVE", "osc_pose", "yaw_negative", "Rotate yaw negatively"],
  ["P", "CLOSE_GRIPPER", "gripper", "close_gripper", "Close the gripper"],
  ["O", "OPEN_GRIPPER", "gripper", "open_gripper", "Open the gripper"],
  ["I", "ROLL_POSITIVE", "osc_pose", "roll_positive", "Rotate roll positively"],
  ["K", "ROLL_NEGATIVE", "osc_pose", "roll_negative", "Rotate roll negatively"],
  ["J", "PITCH_POSITIVE", "osc_pose", "pitch_positive", "Rotate pitch positively"],
  ["L", "PITCH_NEGATIVE", "osc_pose", "pitch_negative", "Rotate pitch negatively"],
];

export const LIBERO_PROFILE = defineProfile({
  id: "libero-manipulation",
  name: "LIBERO manipulation",
  TOKENS: definitions.map(([token]) => token),
  ACTION_TABLE: Object.fromEntries(definitions.map(([token, name, kind, action, description]) => [
    token,
    { token, name, kind, action, description, defaultMs: 250, repeatSteps: REPEAT_STEPS, amplitude: ACTION_AMPLITUDE },
  ])),
  ACTIONS: [],
  roles: { stop: "X", forward: "W", backward: "S", strafeLeft: "A", strafeRight: "D", turnLeft: "Q", turnRight: "E" },
  motion: ["W", "S", "A", "D", "U", "R", "Q", "E", "I", "K", "J", "L"],
  opposite: { W: "S", S: "W", A: "D", D: "A", U: "R", R: "U", Q: "E", E: "Q", I: "K", K: "I", J: "L", L: "J" },
  arms: null,
  envelope: null,
  sensing: null,
  capabilities: {
    boxLift: null,
    manipulation: { controller: "OSC_POSE", gripper_available: true, repeat_steps: REPEAT_STEPS, amplitude: ACTION_AMPLITUDE },
  },
  legacyDurationMaxMs: null,
  TOKEN_LEGEND: definitions.map(([token, , , , description]) => `${token} = ${description}`).join("\n"),
});
