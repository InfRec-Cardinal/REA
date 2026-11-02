import { ARMS } from "./evidence.mjs";

// Quaternions are WXYZ. Rotate an orientation about the world z axis.
export function yawRotate(quaternion, angle) {
  const [w, x, y, z] = quaternion;
  const c = Math.cos(angle / 2);
  const s = Math.sin(angle / 2);
  return [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w];
}

/** Where the arm last began closing, taken from its own commands; the recovery anchor. */
export function closingAnchor(history, arm) {
  for (let index = history.length - 1; index >= 0; index -= 1) {
    const args = history[index].action?.arguments;
    if (args?.arm === arm && typeof args.opening === "number" && args.opening <= 0.5 && Array.isArray(args.position)) return { position: args.position, quaternion: args.quaternion };
  }
  return null;
}

const move = (arm, position, quaternion, opening) => ({ name: "move_gripper", arguments: { arm, position: position.map(value => Math.round(value * 10000) / 10000), quaternion, opening } });

/**
 * Plan a recovery macro for a judge verdict. Returns { macro, actions } or { macro: "directive", directive }
 * when the recovery needs perception the rules do not have. Every action is a normal pose command.
 */
export function planRecovery({ verdict, evidence, history, attempt, config }) {
  const macro = config.macros[verdict.reason];
  if (!macro) return null;
  const arm = verdict.arm ?? ARMS.find(item => evidence[item]?.grasp === "empty") ?? ARMS.find(item => closingAnchor(history, item));
  const pose = arm && evidence[arm]?.pose;
  if (macro === "directive" || !pose) {
    const held = ARMS.filter(item => evidence[item]?.grasp === "held");
    return { macro: "directive", directive: verdict.reason === "missing_arm"
      ? `Evidence: ${held.join(" and ")} arm holds the object (finger gap ${held.map(item => evidence[item].gap?.toFixed(4)).join(", ")} m); the ${verdict.arm} arm has no grasp. Keep the holding arm still, bring the ${verdict.arm} arm to the other end of the object, close it, then lift both together.`
      : `Evidence says the task is not complete (${verdict.reason}). Re-plan from the images: approach from a different angle or height and verify a grasp by finger gap before lifting.` };
  }
  const anchor = closingAnchor(history, arm) ?? { position: pose.slice(0, 3), quaternion: pose.slice(3) };
  const [x, y, z] = pose.slice(0, 3);
  const current = pose.slice(3);
  if (macro === "lift") return { macro, arm, actions: [move(arm, [x, y, z + config.liftTest], current, 0)] };
  const direction = attempt % 2 === 0 ? 1 : -1;
  const yaw = yawRotate(anchor.quaternion ?? current, direction * config.yawStep * Math.ceil((attempt + 1) / 2));
  const [ax, ay, az] = anchor.position;
  const grasp = az + config.heightOffset * (attempt + 1);
  return { macro, arm, actions: [
    move(arm, [x, y, z + config.retreat], current, 1),
    move(arm, [ax, ay, grasp + config.approach], yaw, 1),
    move(arm, [ax, ay, grasp], yaw, 1),
    move(arm, [ax, ay, grasp], yaw, 0),
    move(arm, [ax, ay, grasp + config.liftTest], yaw, 0),
  ] };
}
