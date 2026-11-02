import { ARMS } from "./evidence.mjs";

/** Height of each arm when its current uninterrupted grasp began, from history plus the current evidence. */
function graspStartHeights(sequence) {
  const start = {};
  for (const evidence of sequence) {
    for (const arm of ARMS) {
      if (evidence[arm]?.grasp === "held") start[arm] ??= evidence[arm].z;
      else if (evidence[arm]?.grasp === "open" || evidence[arm]?.grasp === "empty") delete start[arm];
    }
  }
  return start;
}

/**
 * Rule judge over shared-observation evidence. Returns
 * { verdict: "complete" | "not_complete" | "uncertain", reason, arm?, detail }.
 */
export function judgeCompletion({ instruction = "", evidence, history = [], evidenceOf, config }) {
  const bothArms = new RegExp(config.bothArmsPattern, "iu").test(instruction);
  const holding = new RegExp(config.holdPattern, "iu").test(instruction) && !new RegExp(config.followUpPattern, "iu").test(instruction);
  const summary = Object.fromEntries(ARMS.map(arm => [arm, { grasp: evidence[arm]?.grasp, gap: evidence[arm]?.gap, z: evidence[arm]?.z }]));
  if (!holding) return { verdict: "uncertain", reason: "no_rule_for_task", detail: summary };
  const held = ARMS.filter(arm => evidence[arm]?.grasp === "held");
  const unknown = ARMS.filter(arm => evidence[arm]?.grasp === "unknown");
  if (bothArms && held.length < 2) {
    const missing = ARMS.find(arm => evidence[arm]?.grasp !== "held");
    if (evidence[missing]?.grasp === "unknown") return { verdict: "uncertain", reason: "grasp_unobserved", arm: missing, detail: summary };
    if (!held.length) return { verdict: "not_complete", reason: ARMS.some(arm => evidence[arm]?.grasp === "empty") ? "empty_grasp" : "no_grasp", arm: missing, detail: summary };
    return { verdict: "not_complete", reason: "missing_arm", arm: missing, detail: summary };
  }
  if (!held.length) {
    if (unknown.length) return { verdict: "uncertain", reason: "grasp_unobserved", detail: summary };
    const empty = ARMS.find(arm => evidence[arm]?.grasp === "empty");
    return { verdict: "not_complete", reason: empty ? "empty_grasp" : "no_grasp", arm: empty ?? null, detail: summary };
  }
  if (holding) {
    const start = graspStartHeights([...history.map(evidenceOf), evidence]);
    const lowest = held.find(arm => !(evidence[arm].z !== null && start[arm] !== undefined && evidence[arm].z - start[arm] >= config.liftMin));
    if (lowest) return { verdict: "not_complete", reason: "not_lifted", arm: lowest, detail: { ...summary, graspStartZ: start } };
  }
  return { verdict: "complete", reason: "evidence_satisfied", detail: summary };
}
