// Normalized per-arm evidence from a pose-controlled bimanual state plus execution feedback.
// Every field comes from the observation and execution feedback only; missing data reads as "unknown".

export const ARMS = ["left", "right"];

const finite = value => typeof value === "number" && Number.isFinite(value);
const vector = (value, length) => Array.isArray(value) && value.length >= length && value.slice(0, length).every(finite) ? value.slice(0, length) : null;
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;

export function armPose(state, arm) {
  return vector(state?.[`${arm}_endpose`], 7) ?? vector(state?.[`${arm}_ee_pose`], 7);
}

export function armEvidence(state, execution, config) {
  const evidence = {};
  for (const arm of ARMS) {
    const pose = armPose(state, arm);
    const tcp = vector(state?.tcp_poses?.[arm], 3);
    const opening = finite(state?.[`${arm}_gripper`]) ? state[`${arm}_gripper`] : null;
    const reported = execution?.gripper_execution?.[arm]?.fingers?.map(finger => finger.actual_m).filter(finite);
    const measured = (state?.measured_finger_joint_positions?.[arm] ?? []).filter(finite);
    const fingers = reported?.length ? reported : measured;
    const gap = fingers.length ? mean(fingers) : null;
    let grasp = "unknown";
    if (opening !== null && opening > config.closedCommandMax) grasp = "open";
    else if (opening !== null && gap !== null) grasp = gap >= config.heldMinGap ? "held" : gap <= config.emptyMaxGap ? "empty" : "unknown";
    evidence[arm] = { pose, z: tcp?.[2] ?? pose?.[2] ?? null, opening, gap, grasp };
  }
  return evidence;
}

export function executionOf(record) {
  return record?.info?.execution ?? null;
}
