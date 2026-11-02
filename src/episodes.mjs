// Episode store: every step of every session, appended to one JSONL file and digested per session.
//
// The episodic ring (memory.mjs) is short and dies with its session; the experience store keeps only
// distilled lessons. This is the third tier: the raw record of what each session did, so that
//   - a new session on the same task is told how the last ones went ("12 steps, 3 W refusals at
//     centre near, reviewer STOP x1, ended refused") before its first deliberate decision,
//   - an evaluation can replay refusal rates, escalations and judge scores across sessions offline,
//   - a reviewer verdict or an asynchronous judgement that arrives after a session moved on is still
//     attached to the step it judged.
//
// Records are compact and append-only; the file is compacted to the newest `maxSessions` sessions on
// load. Nothing here is on the control path or reaches the actuator.

import { appendFile, mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

const OUTCOME_ORDER = ["refused", "fail_closed", "dispatch_failed", "too_close", "stalled", "oscillating", "progress", "stopped", "preview", "neutral"];

function round(value, digits = 3) {
  return Number.isFinite(value) ? Number(value.toFixed(digits)) : null;
}

function keywords(taskKey) {
  return new Set(String(taskKey ?? "").split("-").filter(Boolean));
}

/**
 * How alike two task keys are: 2 for the same key, 1 when they share at least half their words (Jaccard
 * >= 0.5: "reach-doorway" ~ "doorway", not "turn-left" ~ "turn-right"), 0 otherwise. "generic" is no task.
 */
export function taskSimilarity(a, b) {
  if (!a || !b || a === "generic" || b === "generic") return 0;
  if (a === b) return 2;
  const wa = keywords(a);
  const wb = keywords(b);
  if (!wa.size || !wb.size) return 0;
  let shared = 0;
  for (const word of wa) if (wb.has(word)) shared += 1;
  const union = wa.size + wb.size - shared;
  return shared > 0 && shared / union >= 0.5 ? 1 : 0;
}

export function createEpisodeStore({ path = null, maxSessions = 200, maxStepsPerSession = 512, now = () => Date.now(), logger = null } = {}) {
  const sessions = new Map(); // id -> { id, taskKey, task, startedAt, endedAt, steps: [], seqs: Set, reviews: [], judgements: [] }
  let queue = Promise.resolve();
  let loaded = path === null;
  let lines = 0;

  function session(id, { taskKey = "generic", task = null } = {}) {
    let entry = sessions.get(id);
    if (!entry) {
      entry = { id, taskKey, task, startedAt: null, endedAt: null, steps: [], seqs: new Set(), reviews: [], judgements: [] };
      sessions.set(id, entry);
    }
    if (taskKey && taskKey !== "generic") entry.taskKey = taskKey;
    if (task) entry.task = task;
    return entry;
  }

  /** Drop the sessions that ended longest ago. A session with no step yet is the newest, not the oldest. */
  function evict() {
    const age = entry => entry.endedAt ?? entry.startedAt ?? Number.POSITIVE_INFINITY;
    while (sessions.size > maxSessions) {
      let oldest = null;
      for (const entry of sessions.values()) if (!oldest || age(entry) < age(oldest)) oldest = entry;
      if (!oldest) break;
      sessions.delete(oldest.id);
    }
  }

  function append(record) {
    if (path === null) return Promise.resolve();
    const line = `${JSON.stringify(record)}\n`;
    const run = async () => {
      await mkdir(dirname(path), { recursive: true });
      await appendFile(path, line, "utf8");
    };
    queue = queue.then(run, run).catch(error => {
      logger?.warn?.({ event: "episodes_append_failed", path, error: error.message });
    });
    return queue;
  }

  function apply(record) {
    if (!record || typeof record !== "object") return false;
    if (record.k === "step" && typeof record.s === "string" && Number.isSafeInteger(record.seq)) {
      const entry = session(record.s, { taskKey: record.t, task: record.task ?? null });
      const step = {
        seq: record.seq,
        token: record.tok ?? null,
        engine: record.eng ?? null,
        candidate: record.cand ?? record.eng ?? null,
        cost: Number.isFinite(record.cost) ? record.cost : null,
        route: record.route ?? null,
        escalated: record.esc === true ? true : record.esc === false ? false : null,
        allowed: record.ok ?? null,
        reasons: Array.isArray(record.why) ? record.why : [],
        executed: record.exe === true,
        outcome: record.out ?? null,
        score: Number.isFinite(record.sc) ? record.sc : null,
        situation: record.sit ?? null,
        centerM: Number.isFinite(record.c) ? record.c : null,
        pose: Array.isArray(record.pose) ? { x: record.pose[0], y: record.pose[1], heading: record.pose[2] } : null,
        revisits: Number.isFinite(record.rv) ? record.rv : 0,
        at: Number.isFinite(record.at) ? record.at : null,
      };
      if (!entry.seqs.has(step.seq)) {
        entry.seqs.add(step.seq);
        entry.steps.push(step);
        // One runaway session (a process-long reactive one, say) keeps its newest steps only.
        while (entry.steps.length > maxStepsPerSession) entry.seqs.delete(entry.steps.shift().seq);
      }
      entry.startedAt = entry.startedAt ?? step.at;
      entry.endedAt = step.at ?? entry.endedAt;
      evict();
      return true;
    }
    if (record.k === "review" && typeof record.s === "string") {
      const entry = session(record.s);
      entry.reviews.push({ seq: record.seq, verdict: record.v, next: record.next ?? null, at: record.at ?? null });
      return true;
    }
    if (record.k === "judgement" && typeof record.s === "string") {
      const entry = session(record.s);
      entry.judgements.push({ seq: record.seq, outcome: record.out, at: record.at ?? null });
      return true;
    }
    return false;
  }

  /**
   * Read the file, then compact it to what is still in memory (the newest sessions, each capped to its
   * newest steps). Read, apply and rewrite run as one job on the append queue, so a record() that
   * arrives while the file is being read is appended after the compaction, never overwritten by it.
   */
  function load() {
    if (path === null) {
      loaded = true;
      return Promise.resolve({ loaded: 0, sessions: 0 });
    }
    const run = async () => {
      let text = "";
      try {
        text = await readFile(path, "utf8");
      } catch (error) {
        if (error.code !== "ENOENT") logger?.warn?.({ event: "episodes_load_failed", path, error: error.message });
        loaded = true;
        return { loaded: 0, sessions: 0 };
      }
      let count = 0;
      let total = 0;
      const parsed = [];
      for (const line of text.split(/\r?\n/u)) {
        if (!line.trim()) continue;
        total += 1;
        try {
          const record = JSON.parse(line);
          parsed.push([line, record]);
          if (apply(record)) count += 1;
        } catch {
          /* a corrupt line is skipped, never fatal */
        }
      }
      lines = total;
      loaded = true;
      const keptLines = parsed
        .filter(([, record]) => {
          const entry = sessions.get(record.s);
          if (!entry) return false;
          return record.k !== "step" || entry.seqs.has(record.seq);
        })
        .map(([line]) => line);
      if (keptLines.length < total) {
        const tmp = `${path}.tmp`;
        await writeFile(tmp, `${keptLines.join("\n")}\n`, "utf8");
        await rename(tmp, path);
        lines = keptLines.length;
      }
      return { loaded: count, sessions: sessions.size };
    };
    const result = queue.then(run, run);
    queue = result.catch(error => {
      loaded = true;
      logger?.warn?.({ event: "episodes_compact_failed", path, error: error.message });
    });
    return result;
  }

  /** One step of one session. `pose` is [x, y, heading] from the spatial memory when present. */
  function record(sessionId, { taskKey = "generic", task = null, seq, token, engine = null, candidate = null, cost = null, route = null, escalated = null, allowed = null, reasons = [], executed = false, outcome = null, score = null, situation = null, centerM = null, pose = null, revisits = 0, at = now() } = {}) {
    if (typeof sessionId !== "string" || !sessionId || !Number.isSafeInteger(seq)) return null;
    const line = {
      k: "step",
      s: sessionId,
      t: taskKey,
      ...(task ? { task: String(task).slice(0, 120) } : {}),
      seq,
      tok: token ?? null,
      eng: engine,
      cand: candidate && candidate !== engine ? candidate : undefined,
      cost: Number.isFinite(cost) ? round(cost, 6) : undefined,
      route,
      esc: escalated === true ? true : escalated === false ? false : undefined,
      ok: allowed,
      why: Array.isArray(reasons) && reasons.length ? reasons.slice(0, 4) : undefined,
      exe: executed === true ? true : undefined,
      out: outcome,
      sc: round(score),
      sit: situation ?? undefined,
      c: round(centerM, 2) ?? undefined,
      pose: pose ? [round(pose.x), round(pose.y), round(pose.heading)] : undefined,
      rv: revisits || undefined,
      at,
    };
    apply(line);
    lines += 1;
    void append(line);
    return line;
  }

  function review(sessionId, { seq, verdict, next = null, at = now() } = {}) {
    if (typeof sessionId !== "string" || !sessionId) return null;
    const line = { k: "review", s: sessionId, seq, v: verdict, next, at };
    apply(line);
    lines += 1;
    void append(line);
    return line;
  }

  function judgement(sessionId, { seq, outcome, at = now() } = {}) {
    if (typeof sessionId !== "string" || !sessionId) return null;
    const line = { k: "judgement", s: sessionId, seq, out: outcome, at };
    apply(line);
    lines += 1;
    void append(line);
    return line;
  }

  /** The shape of one session in a few numbers. */
  function digest(sessionId) {
    const entry = sessions.get(sessionId);
    if (!entry) return null;
    const steps = entry.steps;
    const refusals = steps.filter(step => step.allowed === false);
    const byToken = {};
    for (const step of steps) byToken[step.token ?? "?"] = (byToken[step.token ?? "?"] ?? 0) + 1;
    const refusedWhere = {};
    for (const step of refusals) {
      const where = `${step.token ?? "?"} at ${step.situation ? step.situation.replace(/\|.*$/u, "").replace(/^c:/u, "centre ") : "unknown"}`;
      refusedWhere[where] = (refusedWhere[where] ?? 0) + 1;
    }
    const scores = steps.map(step => step.score).filter(value => Number.isFinite(value));
    const reviews = { KEEP: 0, CORRECT: 0, STOP: 0 };
    for (const item of entry.reviews) if (item.verdict in reviews) reviews[item.verdict] += 1;
    const judgements = {};
    for (const item of entry.judgements) judgements[item.outcome] = (judgements[item.outcome] ?? 0) + 1;
    const last = steps.at(-1) ?? null;
    return {
      session_id: entry.id,
      task: entry.task,
      task_key: entry.taskKey,
      steps: steps.length,
      executed: steps.filter(step => step.executed).length,
      refusals: refusals.length,
      refusal_rate: steps.length ? round(refusals.length / steps.length, 3) : null,
      // The router's own verdict when the step recorded it; for older lines, the reasons that always
      // meant the deliberate engine ran (budget_exhausted means it did not).
      escalations: steps.filter(step => step.escalated ?? (step.route && !["default", "explicit", "memory", "budget_exhausted"].includes(step.route))).length,
      deliberate: steps.filter(step => step.engine === "react").length,
      cost_usd: round(steps.reduce((sum, step) => sum + (Number.isFinite(step.cost) ? step.cost : 0), 0), 6),
      candidates: steps.reduce((acc, step) => ({ ...acc, [step.candidate ?? "?"]: (acc[step.candidate ?? "?"] ?? 0) + 1 }), {}),
      tokens: byToken,
      refused_where: refusedWhere,
      mean_score: scores.length ? round(scores.reduce((sum, value) => sum + value, 0) / scores.length, 3) : null,
      last_outcome: last?.outcome ?? null,
      last_token: last?.token ?? null,
      reviews,
      judgements,
      max_revisits: steps.reduce((max, step) => Math.max(max, step.revisits ?? 0), 0),
      started_at: entry.startedAt,
      ended_at: entry.endedAt,
      duration_s: entry.startedAt && entry.endedAt ? round((entry.endedAt - entry.startedAt) / 1000, 1) : null,
    };
  }

  /**
   * Digests of the most recent other sessions on this task: the same key first, then a similar one
   * (taskSimilarity), newest first. Each digest says which (`match`: "same" | "similar"), and the
   * prompt line names the other task, so the model is never told a neighbouring task was this one.
   * The generic key is no task and recalls nothing.
   */
  function recall({ taskKey = "generic", exclude = null, k = 2 } = {}) {
    const candidates = [...sessions.values()]
      .filter(entry => entry.id !== exclude && entry.steps.length > 0)
      .map(entry => ({ entry, match: taskSimilarity(taskKey, entry.taskKey) }))
      .filter(item => item.match > 0)
      .sort((a, b) => b.match - a.match || (b.entry.endedAt ?? 0) - (a.entry.endedAt ?? 0));
    return candidates.slice(0, Math.max(0, k)).map(item => ({ ...digest(item.entry.id), match: item.match === 2 ? "same" : "similar" }));
  }

  function when(at) {
    if (!Number.isFinite(at) || Math.abs(at) > 8.64e15) return "earlier";
    return new Date(at).toISOString().slice(5, 16).replace("T", " ");
  }

  function describe(d) {
    const label = d.match === "similar" ? `similar task "${String(d.task_key ?? "").slice(0, 40)}", ${when(d.ended_at)}` : when(d.ended_at);
    const parts = [`${d.steps} steps`];
    if (d.refusals) {
      const where = Object.entries(d.refused_where).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([what, count]) => `${count}x ${what}`).join(", ");
      parts.push(`${d.refusals} refused (${where})`);
    }
    if (d.escalations) parts.push(`${d.escalations} escalations`);
    if (d.reviews.STOP || d.reviews.CORRECT) parts.push(`reviewer STOP x${d.reviews.STOP}, CORRECT x${d.reviews.CORRECT}`);
    if (d.max_revisits >= 1) parts.push(`revisited a spot ${d.max_revisits}x`);
    parts.push(`ended ${d.last_outcome ?? "unknown"}${d.last_token ? ` on ${d.last_token}` : ""}`);
    return `Past episode (${label}): ${parts.join("; ")}.`;
  }

  /** A few lines for the deliberate prompt; never on the reflex path. */
  function render(digests, { maxChars = 400 } = {}) {
    if (!digests || digests.length === 0) return "";
    const out = [];
    let used = 0;
    for (const d of digests) {
      const line = describe(d).slice(0, 200);
      if (used + line.length + 1 > maxChars) break;
      out.push(line);
      used += line.length + 1;
    }
    return out.join("\n");
  }

  /** Rotate the file to `<path>.bak-<unix_ms>` (queued before any later write). Returns the backup path or null. */
  function backup() {
    if (path === null) return null;
    const target = `${path}.bak-${Math.floor(now())}`;
    const run = async () => {
      try {
        await rename(path, target);
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
    };
    queue = queue.then(run, run).catch(error => logger?.warn?.({ event: "episodes_backup_failed", path, error: error.message }));
    return target;
  }

  /** Drop every session from memory and truncate the file. Only an evaluation should want this. */
  function clear() {
    sessions.clear();
    lines = 0;
    if (path === null) return Promise.resolve();
    const run = async () => {
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, "", "utf8");
    };
    queue = queue.then(run, run).catch(error => logger?.warn?.({ event: "episodes_clear_failed", path, error: error.message }));
    return queue;
  }

  function stats() {
    let steps = 0;
    for (const entry of sessions.values()) steps += entry.steps.length;
    return { sessions: sessions.size, steps, lines, max_sessions: maxSessions, max_steps_per_session: maxStepsPerSession, path, loaded };
  }

  return {
    load,
    record,
    review,
    judgement,
    digest,
    recall,
    render,
    describe,
    stats,
    clear,
    backup,
    sessions: () => [...sessions.keys()],
    flush: () => queue,
    outcomeOrder: OUTCOME_ORDER,
  };
}
