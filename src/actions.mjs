// Facade: the Unitree G1 vocabulary as module constants. New code takes a profile (see
// src/profiles/) instead of importing these; they stay so nothing that imports them has to change.
import { G1_PROFILE } from "./profiles/g1.mjs";

export const {
  TOKENS, ACTIONS, ACTION_TABLE, BOX_LIFT, HOLDING_TOKENS, RELEASE_TOKEN, MIN_LINEAR_SPEED, MIN_YAW_RATE, TOKEN_LEGEND,
  parseToken, toToken, actionForToken, actionOrStop, isValidAction, legacyAction, motionFor,
} = G1_PROFILE;
export { G1_PROFILE };
export default G1_PROFILE;
