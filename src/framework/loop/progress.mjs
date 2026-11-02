import { ARMS, armPose } from "./evidence.mjs";

export function quaternionAngle(a, b) {
  const dot = Math.abs(a.reduce((sum, value, index) => sum + value * b[index], 0) / (Math.hypot(...a) * Math.hypot(...b)));
  return 2 * Math.acos(Math.min(1, dot));
}

export const distance = (a, b) => Math.hypot(...a.map((value, index) => value - b[index]));

function numericLeaves(value, out = []) {
  if (typeof value === "number") out.push(Number.isFinite(value) ? value : 0);
  else if (Array.isArray(value)) for (const item of value) numericLeaves(item, out);
  else if (value && typeof value === "object") for (const key of Object.keys(value).sort()) numericLeaves(value[key], out);
  return out;
}

/** Whether a transition changed the robot beyond sensor jitter. Replaces exact JSON equality. */
export function moved(before, after, config) {
  if (!before || !after) return true;
  const poses = ARMS.map(arm => [armPose(before, arm), armPose(after, arm)]).filter(([p, q]) => p && q);
  if (poses.length) {
    if (poses.some(([p, q]) => distance(p.slice(0, 3), q.slice(0, 3)) > config.translation || quaternionAngle(p.slice(3), q.slice(3)) > config.rotation)) return true;
    if (ARMS.some(arm => Math.abs((before[`${arm}_gripper`] ?? 0) - (after[`${arm}_gripper`] ?? 0)) > config.gripper)) return true;
    const fingers = arm => before.measured_finger_joint_positions?.[arm]?.map((value, index) => Math.abs(value - (after.measured_finger_joint_positions?.[arm]?.[index] ?? value))) ?? [];
    return ARMS.some(arm => fingers(arm).some(delta => delta > config.finger));
  }
  const p = numericLeaves(before);
  const q = numericLeaves(after);
  if (p.length !== q.length) return true;
  return p.some((value, index) => Math.abs(value - q[index]) > config.vector);
}

/** Consecutive trailing records that neither succeeded, rewarded, nor moved the robot. */
export function stalledSteps(history, config) {
  let count = 0;
  for (let index = history.length - 1; index >= 0; index -= 1) {
    const record = history[index];
    const blocked = record.info?.gate?.allowed === false || record.info?.execution?.gate?.allowed === false;
    if (!blocked && (record.success === true || record.reward > 0 || moved(record.observation?.state, record.nextObservation?.state, config))) break;
    count += 1;
  }
  return count;
}
