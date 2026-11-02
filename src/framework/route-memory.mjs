import { createHash } from "node:crypto";

// Route memory: REA's System 1 for routed episodes. For every task it remembers how each System 0 capability did on
// development episodes, runs the best one first, and at each capability boundary decides whether to stop or to
// escalate to the next rung of a fixed ladder. The table is built once from development episodes and frozen before
// evaluation; nothing here reads simulator success during an episode.

export const ROUTE_TABLE_SCHEMA = "rea.route-table/v1";
export const ROUTE_RULE = Object.freeze({
  primary: "higher development success rate (successes / scored episodes)",
  tie: "fewer development steps per success (steps of all scored episodes / successes)",
  final: "candidate order",
  ladder: "[first, the other candidates by rank, first], cut to maxRuns",
});
export const RECOVER_ON = Object.freeze(["judge", "exception", "always", "never"]);
export const DEFAULT_MAX_RUNS = 3;

const sha256 = text => createHash("sha256").update(text).digest("hex");

/** The key route memory files a task under: "suite/task" for LIBERO-style tasks, the task string, or its id. */
export function taskKey(task) {
  if (task === null || task === undefined) return "default";
  if (typeof task === "string") return task;
  if (typeof task === "object" && typeof task.task === "string") return task.suite ? `${task.suite}/${task.task}` : task.task;
  if (typeof task === "object" && typeof task.id === "string") return task.id;
  return JSON.stringify(task);
}

/** Development evidence of one capability on one task from its episode results ({ success, steps }). Pure. */
export function candidateEvidence(results) {
  const scored = results.filter(result => typeof result.success === "boolean");
  const n = scored.length;
  const successes = scored.filter(result => result.success).length;
  const steps = scored.reduce((sum, result) => sum + (Number.isFinite(result.steps) ? result.steps : 0), 0);
  return {
    n, successes, unscored: results.length - n,
    rate: n ? successes / n : null,
    steps,
    stepsPerSuccess: successes ? steps / successes : null,
  };
}

/** Capabilities with evidence, best first under ROUTE_RULE; `order` breaks the remaining ties. Pure. */
export function rankCandidates(evidence, order = Object.keys(evidence)) {
  const ids = order.filter(id => evidence[id]?.n > 0);
  return ids.sort((a, b) => {
    const x = evidence[a];
    const y = evidence[b];
    // Exact comparison of successes/n: the candidates may have different numbers of episodes.
    const difference = y.successes * x.n - x.successes * y.n;
    if (difference !== 0) return difference;
    const xs = x.stepsPerSuccess ?? Infinity;
    const ys = y.stepsPerSuccess ?? Infinity;
    if (xs !== ys) return xs < ys ? -1 : 1;
    return order.indexOf(a) - order.indexOf(b);
  });
}

/** The escalation ladder for ranked capabilities: [first, the others by rank, first], at most maxRuns rungs. Pure. */
export function routeLadder(ranked, maxRuns = DEFAULT_MAX_RUNS) {
  if (!ranked.length) return [];
  return [...ranked, ranked[0]].slice(0, Math.max(1, maxRuns));
}

/** sha256 of a table's content (every field except `sha256`). */
export function routeTableSha256(table) {
  const { sha256: _ignored, ...content } = table;
  return sha256(JSON.stringify(content));
}

/**
 * Build a route table from development episode results: [{ task, candidateId, success, steps }], one entry per episode
 * of a single-capability development run. Episodes without a boolean success carry no evidence. Use development
 * episodes only: a table built on evaluation episodes selects on the test set.
 */
export function buildRouteTable(results, { candidates, maxRuns = DEFAULT_MAX_RUNS, createdAt = new Date().toISOString(), sources = [] } = {}) {
  const order = candidates ?? [...new Set(results.map(result => result.candidateId))];
  if (!order.length) throw new Error("a route table needs at least one candidate");
  if (!Number.isInteger(maxRuns) || maxRuns < 1) throw new RangeError("maxRuns must be a positive integer");
  const grouped = new Map();
  for (const result of results) {
    if (!order.includes(result.candidateId)) throw new Error(`result names unknown candidate ${JSON.stringify(result.candidateId)}`);
    const key = taskKey(result.task);
    if (!grouped.has(key)) grouped.set(key, new Map());
    const byCandidate = grouped.get(key);
    if (!byCandidate.has(result.candidateId)) byCandidate.set(result.candidateId, []);
    byCandidate.get(result.candidateId).push(result);
  }
  const tasks = {};
  for (const key of [...grouped.keys()].sort()) {
    const byCandidate = grouped.get(key);
    const evidence = Object.fromEntries(order.filter(id => byCandidate.has(id)).map(id => [id, candidateEvidence(byCandidate.get(id))]));
    const ranked = rankCandidates(evidence, order);
    tasks[key] = { evidence, ranked, choice: ranked[0] ?? null, ladder: routeLadder(ranked, maxRuns) };
  }
  const table = { schema: ROUTE_TABLE_SCHEMA, createdAt, rule: ROUTE_RULE, candidates: order, maxRuns, sources, tasks };
  return { ...table, sha256: routeTableSha256(table) };
}

/** Throw unless `table` is a well-formed route table whose sha256 (when present) matches its content. */
export function verifyRouteTable(table) {
  if (!table || typeof table !== "object") throw new TypeError("route table must be an object");
  if (table.schema !== ROUTE_TABLE_SCHEMA) throw new Error(`route table schema must be ${ROUTE_TABLE_SCHEMA} (got ${table.schema ?? "none"})`);
  if (!table.tasks || typeof table.tasks !== "object") throw new TypeError("route table needs a tasks object");
  for (const [key, entry] of Object.entries(table.tasks)) {
    if (!Array.isArray(entry?.ladder) || entry.ladder.some(id => typeof id !== "string")) throw new TypeError(`route table task ${key} needs a ladder of candidate ids`);
  }
  if (table.sha256 !== undefined && table.sha256 !== routeTableSha256(table)) throw new Error("route table sha256 does not match its content");
  return table;
}

/**
 * System 1 for runRoutedEpisode. route(input) is called before the first run and after every run:
 *   input.runs  completed runs, each { candidateId, status: returned | failed | budget | truncated, steps, verdict }
 *   input.judge async () => verdict from System 2 ({ verdict: done | not_done | error, ... }), or null without a judge
 * It returns { candidateId, reason } for the next run, or { stop: true, reason }.
 *
 * recoverOn decides what triggers the next rung: judge (ask System 2 after every run; escalate unless it says done),
 * exception (escalate only after a failed or truncated run), always (blind retry), never (one run).
 * onJudgeError: what a judge error means under recoverOn judge, escalate (default) or stop.
 */
export function createRouteMemoryRouter({ table = null, recoverOn = "judge", onJudgeError = "escalate", maxRuns = null } = {}) {
  if (table) verifyRouteTable(table);
  if (!RECOVER_ON.includes(recoverOn)) throw new TypeError(`recoverOn must be one of ${RECOVER_ON.join(", ")}`);
  if (!["escalate", "stop"].includes(onJudgeError)) throw new TypeError("onJudgeError must be escalate or stop");
  if (maxRuns !== null && (!Number.isInteger(maxRuns) || maxRuns < 1)) throw new RangeError("maxRuns must be a positive integer");
  const runLimit = maxRuns ?? table?.maxRuns ?? DEFAULT_MAX_RUNS;

  function plan(task, candidateIds) {
    const key = taskKey(task);
    const entry = table?.tasks?.[key];
    const ladder = (entry?.ladder ?? []).filter(id => candidateIds.includes(id)).slice(0, runLimit);
    if (ladder.length) return { key, ladder, reason: `route memory for ${key}: ${entry.choice} first` };
    return { key, ladder: routeLadder(candidateIds, runLimit), reason: `no route evidence for ${key}; candidate order` };
  }

  async function escalation(input) {
    const last = input.runs.at(-1);
    if (recoverOn === "never") return { escalate: false, reason: "single_run" };
    if (recoverOn === "always") return { escalate: true, reason: "blind_retry" };
    if (recoverOn === "exception") {
      return ["failed", "budget", "truncated"].includes(last.status)
        ? { escalate: true, reason: `run_${last.status}` }
        : { escalate: false, reason: "run_returned" };
    }
    if (typeof input.judge !== "function") throw new Error("recoverOn judge needs a System 2 judge");
    const verdict = await input.judge();
    if (verdict?.verdict === "done") return { escalate: false, reason: "judge_done" };
    if (verdict?.verdict === "not_done") return { escalate: true, reason: "judge_not_done" };
    return { escalate: onJudgeError === "escalate", reason: "judge_error" };
  }

  return {
    id: `route-memory:${recoverOn}`,
    plan,
    async route(input) {
      const { ladder, reason } = plan(input.task, input.candidates.map(candidate => candidate.id));
      const runs = input.runs ?? [];
      if (!runs.length) return { candidateId: ladder[0], reason, ladder };
      const decision = await escalation({ ...input, runs });
      if (!decision.escalate) return { stop: true, reason: decision.reason };
      if (runs.length >= ladder.length) return { stop: true, reason: `ladder_exhausted after ${decision.reason}` };
      return { candidateId: ladder[runs.length], reason: decision.reason, ladder };
    },
  };
}
