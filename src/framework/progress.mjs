const PROGRESS_INSTRUCTIONS = "Use episode notes to distinguish observed facts, unconfirmed assumptions, the current subgoal, and the next expected change. Compare the previous action's expected effect with the current observation. An accepted or completed action does not establish task progress. Confirm grasping, lifting, and placement from observed changes before treating them as completed. If evidence disagrees with a prior assumption, update it and reconsider the next action. Holding an object or position may still be appropriate; review does not require moving or releasing it.";

const OMITTED_FIELDS = new Set(["images", "imageB64", "thought", "notes"]);
const EXECUTION_FIELDS = ["status", "error", "error_type", "message", "selected_arm", "pose_error", "tracking_error", "target_error", "position_error_m", "rotation_error_rad", "selected_target_reached", "selected_ee_target_reached", "gate"];
const FAILURE_STATUSES = new Set(["planner_failed", "tracking_error", "failed"]);

function promptValue(value) {
  if (Array.isArray(value)) return value.map(promptValue);
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value).filter(([key]) => !OMITTED_FIELDS.has(key)).map(([key, item]) => [key, promptValue(item)]));
}

export function compactProgressHistory(history = []) {
  return history.map((record, index) => {
    const entry = { step: record.step, action: promptValue(record.action) };
    if (index === history.length - 1) {
      if (record.info !== undefined) entry.info = promptValue(record.info);
      return entry;
    }
    const execution = record.info?.execution;
    const info = {};
    if (execution) info.execution = promptValue(Object.fromEntries(EXECUTION_FIELDS.filter(key => execution[key] !== undefined).map(key => [key, execution[key]])));
    if (record.info?.gate !== undefined) info.gate = promptValue(record.info.gate);
    if (record.info?.error !== undefined) info.error = promptValue(record.info.error);
    if (Object.keys(info).length) entry.info = info;
    return entry;
  });
}

function similarAction(first, second, tolerance) {
  if (typeof first === "number" && typeof second === "number") return Math.abs(first - second) <= tolerance;
  if (first === null || second === null || typeof first !== "object" || typeof second !== "object") return first === second;
  if (Array.isArray(first) !== Array.isArray(second)) return false;
  const keys = Object.keys(first);
  return keys.length === Object.keys(second).length && keys.every(key => Object.hasOwn(second, key) && similarAction(first[key], second[key], tolerance));
}

function executionFailed(record) {
  return FAILURE_STATUSES.has(record.info?.execution?.status)
    || record.info?.gate?.allowed === false
    || record.info?.execution?.gate?.allowed === false;
}

export function createProgressPolicy({ policy, reviewInterval = 6, repeatWindow = 5, patience = 3, actionTolerance = 0.001 } = {}) {
  if (!policy || typeof policy.decide !== "function") throw new TypeError("progress policy requires a policy with decide()");
  let decisions = 0;
  let lastReview = -Infinity;
  let reviewedClaim = null;
  return {
    id: `progress:${policy.id ?? "policy"}`,
    async reset(context) {
      decisions = 0;
      lastReview = -Infinity;
      reviewedClaim = null;
      await policy.reset?.(context);
    },
    async decide(context) {
      if (context.progressReview !== undefined) return policy.decide(context);
      decisions += 1;
      const history = context.history ?? [];
      const recent = history.slice(-repeatWindow);
      const reasons = [];
      if (reviewInterval > 0 && decisions % reviewInterval === 0) reasons.push("periodic");
      if (repeatWindow > 0 && recent.length === repeatWindow && recent.every(record => record.action != null)
        && recent.every((record, index) => recent.slice(index + 1).every(other => similarAction(record.action, other.action, actionTolerance)))) reasons.push("repeated_action");
      if (recent.filter(executionFailed).length >= patience) reasons.push("execution_failures");
      const claimRecord = [...history].reverse().find(record => record.claim?.status === "complete");
      if (claimRecord && claimRecord !== reviewedClaim) {
        reasons.push("completion_claim");
        reviewedClaim = claimRecord;
      }
      const required = reasons.length > 0 && (reasons.includes("completion_claim") || decisions - lastReview >= patience);
      const progressReview = { required, reasons: required ? reasons : [] };
      if (required) lastReview = decisions;
      // The unmodified task text, for wrapped policies that parse it (the routed evidence loop's judge).
      const taskInstruction = context.taskInstruction ?? context.instruction ?? context.observation?.instruction;
      const instruction = [context.instruction ?? context.observation?.instruction ?? "Make progress in the environment using its available actions.", PROGRESS_INSTRUCTIONS,
        ...(required ? [`Review task progress now (${reasons.join(", ")}). Check the current observation against the last expected change, revise unsupported assumptions, and choose the next action for the current subgoal. A completion claim is evidence to check, not proof that the environment succeeded.`] : []),
      ].filter(Boolean).join("\n\n");
      const result = await policy.decide({ ...context, history, instruction, taskInstruction, progressReview, historyForPrompt: compactProgressHistory(history) });
      return { ...result, review: progressReview };
    },
    async close() {
      decisions = 0;
      lastReview = -Infinity;
      reviewedClaim = null;
      await policy.close?.();
    },
  };
}
