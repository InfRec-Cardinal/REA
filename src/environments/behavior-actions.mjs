// BEHAVIOR-1K v3.9.2 default R1Pro contract:
// https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.2/OmniGibson/omnigibson/eval/utils/eval_utils.py
// https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.2/OmniGibson/omnigibson/eval/r1pro.yaml
// https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.2/OmniGibson/omnigibson/robots/robot.py
// Applies to the default robot/controller configuration, not custom challenge robots.

export const R1PRO_STATE_DIM = 61;
export const R1PRO_ACTION_DIM = 23;

export const R1PRO_ACTION_LAYOUT = [
  { name: "base", offset: 0, size: 3, units: "normalized", minimum: -1, maximum: 1,
    description: "Robot-local vx, vy, yaw velocity commands. Multiply by [0.75, 0.75, 1] to obtain [m/s, m/s, rad/s]." },
  { name: "torso", offset: 3, size: 4, units: "rad", description: "Absolute torso joint position targets in torso joint order; not position deltas." },
  { name: "left_arm", offset: 7, size: 7, units: "rad", description: "Absolute left arm joint position targets in joint order; not position deltas." },
  { name: "left_gripper", offset: 14, size: 1, units: "normalized", minimum: -1, maximum: 1, description: "Smooth left gripper position: -1 closed, +1 open." },
  { name: "right_arm", offset: 15, size: 7, units: "rad", description: "Absolute right arm joint position targets in joint order; not position deltas." },
  { name: "right_gripper", offset: 22, size: 1, units: "normalized", minimum: -1, maximum: 1, description: "Smooth right gripper position: -1 closed, +1 open." },
];

export const R1PRO_STATE_LAYOUT = [
  { name: "base_qvel", offset: 0, size: 3, units: ["m/s", "m/s", "rad/s"], frame: "robot_base", description: "Measured robot-local base vx, vy, yaw velocity; not normalized action values." },
  { name: "arm_left_qpos", offset: 3, size: 7, units: "rad" },
  { name: "arm_left_qvel", offset: 10, size: 7, units: "rad/s" },
  { name: "eef_left_pos", offset: 17, size: 3, units: "m", frame: "robot_base", description: "Left end-effector x, y, z position relative to the robot base." },
  { name: "eef_left_quat", offset: 20, size: 4, units: "quaternion", frame: "robot_base", description: "Left end-effector orientation relative to the robot base, ordered x, y, z, w." },
  { name: "gripper_left_qpos", offset: 24, size: 2, units: "m", description: "Measured left finger joint positions; nominally 0 closed to 0.05 open per finger." },
  { name: "gripper_left_qvel", offset: 26, size: 2, units: "m/s" },
  { name: "arm_right_qpos", offset: 28, size: 7, units: "rad" },
  { name: "arm_right_qvel", offset: 35, size: 7, units: "rad/s" },
  { name: "eef_right_pos", offset: 42, size: 3, units: "m", frame: "robot_base", description: "Right end-effector x, y, z position relative to the robot base." },
  { name: "eef_right_quat", offset: 45, size: 4, units: "quaternion", frame: "robot_base", description: "Right end-effector orientation relative to the robot base, ordered x, y, z, w." },
  { name: "gripper_right_qpos", offset: 49, size: 2, units: "m", description: "Measured right finger joint positions; nominally 0 closed to 0.05 open per finger." },
  { name: "gripper_right_qvel", offset: 51, size: 2, units: "m/s" },
  { name: "trunk_qpos", offset: 53, size: 4, units: "rad", description: "Current torso joint positions corresponding to action[3:7]." },
  { name: "trunk_qvel", offset: 57, size: 4, units: "rad/s" },
];

export const R1PRO_ACTION_SPEC = {
  name: "joint_action",
  description: "Command the default BEHAVIOR R1Pro using 23 values: base velocity(3), absolute torso joints(4), absolute left arm joints(7), left gripper(1), absolute right arm joints(7), right gripper(1). Arms and torso use radians, not deltas; base and grippers use normalized [-1,1] commands.",
  parameters: {
    type: "object",
    properties: {
      values: {
        type: "array", minItems: R1PRO_ACTION_DIM, maxItems: R1PRO_ACTION_DIM,
        prefixItems: R1PRO_ACTION_LAYOUT.flatMap(group => Array.from({ length: group.size }, (_, index) => ({
          type: "number",
          description: `${group.name}[${index}] (${group.units}). ${group.description}`,
          ...(group.minimum === undefined ? {} : { minimum: group.minimum, maximum: group.maximum }),
        }))),
        items: false,
      },
    },
    required: ["values"],
    additionalProperties: false,
  },
};

/** Name the fields of one raw 61-value dataset observation without changing units. */
export function describeR1ProState(state) {
  if ((!Array.isArray(state) && !ArrayBuffer.isView(state)) || state.length !== R1PRO_STATE_DIM) {
    throw new TypeError(`R1Pro state must contain ${R1PRO_STATE_DIM} values`);
  }
  const values = Array.from(state);
  if (!values.every(Number.isFinite)) throw new TypeError("R1Pro state values must be finite numbers");
  return Object.fromEntries(R1PRO_STATE_LAYOUT.map(group => [group.name, values.slice(group.offset, group.offset + group.size)]));
}

/**
 * A pose-holding comparison baseline, not a task policy or a demonstrated next action.
 * Stop the base, retain the observed torso and arm positions, and convert finger
 * positions to each gripper's single normalized command. This matches OpenRAL's
 * BEHAVIOR idle_action convention; no assumption of task success is made.
 */
export function createR1ProHoldAction(state) {
  const observed = describeR1ProState(state);
  const gripperCommand = fingers => Math.max(-1, Math.min(1, 2 * ((fingers[0] + fingers[1]) / 2) / 0.05 - 1));
  return {
    name: R1PRO_ACTION_SPEC.name,
    arguments: {
      values: [
        0, 0, 0,
        ...observed.trunk_qpos,
        ...observed.arm_left_qpos,
        gripperCommand(observed.gripper_left_qpos),
        ...observed.arm_right_qpos,
        gripperCommand(observed.gripper_right_qpos),
      ],
    },
  };
}
