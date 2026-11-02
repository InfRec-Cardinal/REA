// Versioned node configurations. Every field here is a parameter that an override may change;
// the invariants in INVARIANTS are not, and loadLoopConfig refuses to change them.

export const DEFAULT_LOOP_CONFIG = Object.freeze({
  progress: { id: "progress", version: 1, translation: 0.005, rotation: 0.05, gripper: 0.05, finger: 0.002, vector: 0.001 },
  gate: {
    id: "gate", version: 1,
    nearZeroTranslation: 0.001, nearZeroRotation: 0.01, gripperChange: 0.05,
    repeatTranslation: 0.002, repeatRotation: 0.01,
    claimPattern: "\\b(done|complete[d]?|achieved|success(ful)?|holding|held|grasped|task (is )?finished|hold (is )?(achieved|maintained)|already (grasped|holding))\\b",
  },
  judge: {
    id: "judge", version: 2,
    closedCommandMax: 0.5, heldMinGap: 0.006, emptyMaxGap: 0.004, liftMin: 0.03,
    bothArmsPattern: "\\bboth (arms|hands|grippers)\\b|\\btwo (arms|hands)\\b|\\bdual[- ]arm\\b|双臂|两只手",
    holdPattern: "\\b(hold|grab|grasp|lift|pick|raise|carry)\\b",
    // A grasp followed by another verb (place, pour, strike, ...) ends released or elsewhere; no end-state rule applies.
    followUpPattern: "\\b(place|put|set|position|hang|pour|beat|strike|hit|pass|hand|move|switch|stack|insert|drop|release|press|push|open|close|spin|turn|shake|throw|rotate|scan|stamp|stir|wipe|sweep|lid|into|onto)\\b",
  },
  recover: {
    id: "recover", version: 1,
    retreat: 0.08, approach: 0.06, yawStep: Math.PI / 6, heightOffset: -0.01, liftTest: 0.03,
    attemptsPerReason: 2, maxFraction: 0.3, abortOnFailures: 2,
    macros: { empty_grasp: "regrasp", no_grasp: "regrasp", not_lifted: "lift", missing_arm: "directive" },
  },
  route: {
    id: "route", version: 1,
    stallSteps: 2, repeatBlocks: 2, failureSteps: 3, maxDeliberateFraction: 0.5, cooldown: 1, maxCallsPerStep: 2,
    failureStatuses: ["planner_failed", "tracking_error", "invalid_action", "not_executed"],
  },
  prompts: {
    id: "prompts", version: 1,
    completion: "Completion requires evidence for every required arm: a commanded-closed gripper whose measured finger gap stays clearly above an empty closure, and, for holding or lifting tasks, a rise of the held end effector. A closed, stationary gripper is not evidence of a grasp. Do not idle in place while the task is unfinished.",
  },
});

export const INVARIANTS = Object.freeze([
  "Observations are unchanged; no simulator success, object pose or termination signal is read.",
  "Every executed action, including recovery macro steps, consumes one decision of the shared budget.",
  "No expert demonstrations and no logic keyed to a task id, seed or instance.",
  "The policy's model and reasoning effort are unchanged.",
]);

export function loadLoopConfig(overrides = {}) {
  const config = {};
  for (const [node, defaults] of Object.entries(DEFAULT_LOOP_CONFIG)) {
    const override = overrides[node] ?? {};
    if (override.id !== undefined && override.id !== defaults.id) throw new TypeError(`loop config ${node} has id ${override.id}`);
    config[node] = { ...defaults, ...override, ...(defaults.macros ? { macros: { ...defaults.macros, ...override.macros } } : {}) };
  }
  for (const node of Object.keys(overrides)) if (!(node in DEFAULT_LOOP_CONFIG)) throw new TypeError(`unknown loop node ${node}`);
  return config;
}

export function configVersions(config) {
  return Object.fromEntries(Object.values(config).map(node => [node.id, `${node.id}@${node.version}`]));
}
