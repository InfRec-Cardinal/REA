import { LIBERO_PROFILE, REPEAT_STEPS, ACTION_AMPLITUDE } from "./profiles/libero.mjs";

// Facade over the LIBERO manipulation profile (src/profiles/libero.mjs), plus the measured prompt constants of the
// one-token LIBERO controller: per-token motion at amplitude 0.5 over REPEAT_STEPS
// control steps and the camera-frame note. Prompt text only; the controller command is unchanged.
export const { TOKENS, ACTION_TABLE, parseToken, actionForToken, actionOrStop, legacyAction } = LIBERO_PROFILE;
export { REPEAT_STEPS, ACTION_AMPLITUDE, LIBERO_PROFILE };
export const LIBERO_ACTION_CONTRACT = LIBERO_PROFILE;

export const CM_PER_TOKEN = Object.freeze([2.2, 3.2]);
export const DEG_PER_TOKEN = Object.freeze([12, 15]);
export const CAMERA_NOTE =
  "External view (left image): screen RIGHT = world +y, screen UP = world +z (higher) but also world -x (farther from the camera); " +
  "screen DOWN = lower or closer to the camera; about 1 pixel per cm. " +
  "Wrist view (right image): screen UP = world +x and screen RIGHT = world -y, so left/right AND up/down are mirrored relative to the external view: " +
  "a target RIGHT of the gripper in the external view needs A, a target RIGHT of centre in the wrist view needs D.";

// The one-token arm's legend spells out the camera-frame consequence of every token (the profile's generic legend does not).
const LEGEND_DESCRIPTIONS = {
  X: "hold still; also lets the gripper finish closing or opening; never declares success",
  W: "move 2-3 cm toward the external camera (world +x): DOWN on the external view, UP on the wrist view",
  S: "move 2-3 cm away from the external camera (world -x): UP on the external view, DOWN on the wrist view",
  A: "move 2-3 cm RIGHT on the external view (world +y): the scene shifts LEFT in the wrist view",
  D: "move 2-3 cm LEFT on the external view (world -y): the scene shifts RIGHT in the wrist view",
  U: "raise the gripper 2-3 cm (world +z): UP on the external view, the table shrinks in the wrist view",
  R: "lower the gripper 2-3 cm toward the table (world -z): DOWN on the external view, the table grows in the wrist view",
  Q: "yaw 12-15 deg about the vertical axis, positive",
  E: "yaw 12-15 deg about the vertical axis, negative",
  P: "start closing the gripper; send X next so the fingers finish closing before moving",
  O: "start opening the gripper; send X next so the fingers finish opening before moving",
  I: "roll 12-15 deg about world x (the external camera's depth axis), positive",
  K: "roll 12-15 deg about world x, negative",
  J: "pitch 12-15 deg about world y (the external view's left-right axis), positive",
  L: "pitch 12-15 deg about world y, negative",
};
export const TOKEN_LEGEND = TOKENS.map(token => `${token} = ${LEGEND_DESCRIPTIONS[token]}`).join("\n");
