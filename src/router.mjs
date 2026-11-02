// Per-step engine router, in the Agent-as-a-Router (ACRouter) shape:
//
//   cheap chain  ->  verify  ->  escalate
//
// ACRouter routes a coding task to a cheap model first, verifies the result, and escalates to a
// strong model once `k` cheap attempts applied without resolving. Here a "task" is one control step,
// the cheap chain is the one-token reflex engine (prime / token), the verifier is the fused safety
// gate plus the outcome judge, and the escalation target is the deliberate ReAct engine that thinks,
// consults memory and can look again before committing to a token.
//
// Two escalation paths exist, mirroring ACRouter's `route()` and `run_with_verifier()`:
//   - route()      before the step: escalate when the episodic signals say the reflex is failing
//                  (refusal streak, oscillation, bad judgements, a reviewer flag, a stall) or when the
//                  situation is risky (something near and the last token was motion).
//   - inline       inside the step: when the reflex token is vetoed by the gate and there is time
//                  left in the deadline, ask the deliberate engine once with the veto as observation
//                  and re-gate its answer. This is verify-and-escalate on a single step.
//
// Like ACRouter the router keeps a lightweight memory: per dimension (coarse situation class) an EMA
// of judge scores per engine. When one engine is clearly better in a dimension it is preferred there.
// A budget caps the deliberate fraction so escalation cannot quietly become the default.

import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

import { dimensionOf } from "./memory.mjs";
import { G1_PROFILE } from "./profiles/g1.mjs";

const ROUTER_MEMORY_VERSION = 1;

export const ROUTER_DEFAULTS = Object.freeze({
  cheapChain: ["prime"],
  escalateTo: "react",
  k: 2,
  cooldown: 2,
  maxDeliberateFraction: 0.5,
  riskDepthBucket: "near",
  inlineEscalate: true,
  inlineMinDeadlineMs: 1500,
  memoryMargin: 0.25,
  memoryMinSamples: 4,
  alpha: 0.3,
  // Candidate descriptors (src/candidates.mjs) give ids a tier; without them the escalation engine is the
  // only deliberate id. `mode` says who decides: the rules, an asynchronous model's suggestions, or both.
  candidates: [],
  mode: "rule",
});

export const ROUTER_MODES = Object.freeze(["rule", "model", "hybrid"]);

export function createRouter(options = {}) {
  const { path = null, logger = null, now = () => Date.now(), profile = G1_PROFILE, ...overrides } = options;
  const cfg = { ...ROUTER_DEFAULTS, ...overrides };
  const motion = new Set(profile.motion);
  if (!Array.isArray(cfg.cheapChain) || cfg.cheapChain.length === 0) throw new TypeError("router cheapChain must be a non-empty array");
  if (typeof cfg.escalateTo !== "string" || !cfg.escalateTo) throw new TypeError("router escalateTo is required");
  if (!Number.isSafeInteger(cfg.k) || cfg.k < 1) throw new TypeError("router k must be a positive integer");
  if (!ROUTER_MODES.includes(cfg.mode)) throw new TypeError(`router mode must be one of ${ROUTER_MODES.join(", ")}`);
  const available = cfg.engines ? new Set(cfg.engines) : null;
  const cheap = cfg.cheapChain[0];
  const tiers = new Map((Array.isArray(cfg.candidates) ? cfg.candidates : []).map(c => [c.id, c.tier]));
  const tierOf = id => tiers.get(id) ?? (id === cfg.escalateTo ? "deliberate" : "reflex");

  const state = {
    steps: 0,
    deliberate: 0,
    escalations: 0,
    inlineEscalations: 0,
    cooldownLeft: 0,
    lastReason: null,
    scores: new Map(), // dimension -> engine -> { ema, n }
    observations: 0,
    loaded: path === null,
    candidates: new Map(), // id -> { calls, n, latencyEma, costUsd, scored }
    suggestions: new Map(), // session id -> { candidate, forSeq, reason, meta, offeredAtStep }
    suggested: { offered: 0, used: 0, stale: 0, unavailable: 0, over_budget: 0 },
  };
  let persistQueue = Promise.resolve();

  function has(engine) {
    return available === null || available.has(engine);
  }

  function scoresSnapshot() {
    const scores = {};
    for (const [dimension, entries] of state.scores) {
      scores[dimension] = Object.fromEntries([...entries].map(([engine, record]) => [engine, { ema: Number(record.ema.toFixed(4)), n: record.n }]));
    }
    return scores;
  }

  /**
   * ACRouter's routing memory is the part of the router that learns; losing it on every restart would
   * make each deployment start from scratch. The scores are persisted as one small JSON document next
   * to the experience store (tmp + rename, so a crash mid-write leaves the previous file intact).
   */
  function persist() {
    if (path === null) return Promise.resolve();
    const document = JSON.stringify({ version: ROUTER_MEMORY_VERSION, saved_at: now(), observations: state.observations, scores: scoresSnapshot(), candidates: candidateStats() });
    const run = async () => {
      await mkdir(dirname(path), { recursive: true });
      const tmp = `${path}.tmp`;
      await writeFile(tmp, `${document}\n`, "utf8");
      await rename(tmp, path);
    };
    persistQueue = persistQueue.then(run, run).catch(error => {
      logger?.warn?.({ event: "router_persist_failed", path, error: error.message });
    });
    return persistQueue;
  }

  /** Restore per-dimension scores from a snapshot (the shape `state()` reports). Unknown engines are kept: they may be enabled later. */
  function restore(scores) {
    if (!scores || typeof scores !== "object") return 0;
    let restored = 0;
    for (const [dimension, entries] of Object.entries(scores)) {
      if (!entries || typeof entries !== "object") continue;
      for (const [engine, record] of Object.entries(entries)) {
        const ema = Number(record?.ema);
        const n = Number(record?.n);
        if (!Number.isFinite(ema) || !Number.isSafeInteger(n) || n < 1) continue;
        if (!state.scores.has(dimension)) state.scores.set(dimension, new Map());
        state.scores.get(dimension).set(engine, { ema: Math.max(-1, Math.min(1, ema)), n });
        restored += 1;
      }
    }
    return restored;
  }

  async function load() {
    if (path === null) {
      state.loaded = true;
      return { loaded: 0 };
    }
    let text;
    try {
      text = await readFile(path, "utf8");
    } catch (error) {
      if (error.code !== "ENOENT") logger?.warn?.({ event: "router_load_failed", path, error: error.message });
      state.loaded = true;
      return { loaded: 0 };
    }
    let restored = 0;
    try {
      const document = JSON.parse(text);
      restored = restore(document?.scores);
      if (document?.candidates && typeof document.candidates === "object") {
        for (const [id, record] of Object.entries(document.candidates)) {
          if (!record || typeof record !== "object") continue;
          state.candidates.set(id, { calls: Number(record.calls) || 0, n: Number(record.n) || 0, latencyEma: Number.isFinite(record.latency_ema_s) ? record.latency_ema_s : null, costUsd: Number(record.cost_usd_total) || 0 });
        }
      }
      if (Number.isSafeInteger(document?.observations)) state.observations = document.observations;
    } catch (error) {
      logger?.warn?.({ event: "router_load_failed", path, error: error.message });
    }
    state.loaded = true;
    return { loaded: restored };
  }

  function budgetAllows() {
    if (state.steps < 4) return true;
    return (state.deliberate + 1) / (state.steps + 1) <= cfg.maxDeliberateFraction;
  }

  function bestByMemory(dimension) {
    const entries = state.scores.get(dimension);
    if (!entries) return null;
    let best = null;
    for (const [engine, record] of entries) {
      if (record.n < cfg.memoryMinSamples || !has(engine)) continue;
      if (!best || record.ema > best.ema) best = { engine, ema: record.ema, n: record.n };
    }
    if (!best) return null;
    const cheapRecord = entries.get(cheap);
    const baseline = cheapRecord && cheapRecord.n >= cfg.memoryMinSamples ? cheapRecord.ema : null;
    if (best.engine === cheap) return null;
    if (baseline === null) return null;
    return best.ema - baseline >= cfg.memoryMargin ? best : null;
  }

  function triggers(signals, situation) {
    const out = [];
    if (!signals) return out;
    if (signals.refusalStreak >= cfg.k) out.push("refusal_streak");
    if (signals.badStreak >= cfg.k) out.push("bad_streak");
    if (signals.oscillation >= 2) out.push("oscillation");
    if (signals.stall) out.push("stall");
    if (signals.lastReview === "STOP" || signals.lastReview === "CORRECT") out.push("reviewer_flag");
    if (situation?.center === cfg.riskDepthBucket && motion.has(signals.lastToken)) out.push("risk_near");
    // Spatial memory: back at a place (and heading) already visited twice, or once where a token was
    // refused — a loop the token history alone shows only later.
    if ((signals.recentRevisits ?? 0) >= 2 || ((signals.revisits ?? 0) >= 1 && (signals.refusedHere ?? 0) >= 1)) out.push("revisit");
    return out;
  }

  /**
   * Choose the engine for the next step. `explicit` is a caller-supplied engine (per-request cfg or
   * the service default when routing is disabled); "auto" or null hands the choice to the router.
   */
  /**
   * A suggestion from the asynchronous model router for one step of one session. It is honoured only
   * for that step (or the very next one when no step number was given), never queued.
   */
  function suggest({ candidate, forSeq = null, reason = "model", sessionId = "*", meta = null } = {}) {
    if (typeof candidate !== "string" || !candidate) return false;
    state.suggestions.set(sessionId, { candidate, forSeq, reason, meta, offeredAtStep: state.steps });
    state.suggested.offered += 1;
    return true;
  }

  function takeSuggestion(sessionId, seq) {
    const key = state.suggestions.has(sessionId) ? sessionId : state.suggestions.has("*") ? "*" : null;
    if (key === null) return null;
    const item = state.suggestions.get(key);
    state.suggestions.delete(key);
    const fresh = Number.isFinite(item.forSeq) && Number.isFinite(seq) ? item.forSeq === seq : state.steps - item.offeredAtStep <= 1;
    if (!fresh) {
      state.suggested.stale += 1;
      return null;
    }
    if (!has(item.candidate)) {
      state.suggested.unavailable += 1;
      return null;
    }
    if (tierOf(item.candidate) === "deliberate" && !budgetAllows()) {
      state.suggested.over_budget += 1;
      return null;
    }
    return item;
  }

  function ruleRoute(fired, dimension) {
    if (has(cfg.escalateTo)) {
      if (state.cooldownLeft > 0 && budgetAllows()) {
        return { engine: cfg.escalateTo, reason: "cooldown", escalated: true, triggers: fired, dimension };
      }
      if (fired.length > 0 && budgetAllows()) {
        state.cooldownLeft = cfg.cooldown;
        state.escalations += 1;
        return { engine: cfg.escalateTo, reason: fired[0], escalated: true, triggers: fired, dimension };
      }
    }
    // A learned preference for a deliberate candidate is still deliberation: it spends the same budget.
    const learned = bestByMemory(dimension);
    if (learned && (tierOf(learned.engine) !== "deliberate" || budgetAllows())) {
      return { engine: learned.engine, reason: "memory", escalated: tierOf(learned.engine) === "deliberate", triggers: fired, dimension };
    }
    const engine = cfg.cheapChain.find(has) ?? cheap;
    return { engine, reason: fired.length > 0 ? "budget_exhausted" : "default", escalated: false, triggers: fired, dimension };
  }

  function suggestedRoute(item, fired, dimension) {
    state.suggested.used += 1;
    return { engine: item.candidate, reason: `${item.reason}`, escalated: tierOf(item.candidate) === "deliberate", triggers: fired, dimension, suggested: true, suggestion: item.meta ?? null };
  }

  /**
   * Choose the candidate for the next step. `explicit` is a caller-supplied engine (per-request cfg or
   * the service default when routing is disabled); "auto" or null hands the choice to the router.
   * `seq` and `sessionId` let an asynchronous suggestion be matched to exactly this step.
   */
  function route({ signals = null, situation = null, explicit = null, seq = null, sessionId = "*" } = {}) {
    const dimension = dimensionOf(situation);
    if (explicit && explicit !== "auto") {
      return { engine: explicit, reason: "explicit", escalated: tierOf(explicit) === "deliberate", triggers: [], dimension };
    }
    const fired = triggers(signals, situation);
    const suggestion = cfg.mode === "rule" ? null : takeSuggestion(sessionId, seq);
    if (cfg.mode === "model" && suggestion) return suggestedRoute(suggestion, fired, dimension);
    if (cfg.mode === "hybrid") {
      // Safety triggers outrank the model's opinion; otherwise the model's pick is used before the defaults.
      if (fired.length > 0 || state.cooldownLeft > 0) {
        const ruled = ruleRoute(fired, dimension);
        if (ruled.escalated) return ruled;
      }
      if (suggestion) return suggestedRoute(suggestion, fired, dimension);
    }
    return ruleRoute(fired, dimension);
  }

  /** Record which engine actually ran, so the budget and cooldown reflect reality. */
  function commit(engine, { inline = false } = {}) {
    state.steps += 1;
    state.lastReason = null;
    if (tierOf(engine) === "deliberate") {
      state.deliberate += 1;
      if (state.cooldownLeft > 0) state.cooldownLeft -= 1;
    }
    if (inline) state.inlineEscalations += 1;
    const record = state.candidates.get(engine) ?? { calls: 0, n: 0, latencyEma: null, costUsd: 0 };
    record.calls += 1;
    state.candidates.set(engine, record);
  }

  /**
   * ACRouter `observe()`: fold a judge score into the per-dimension, per-engine memory. Latency and
   * cost, when known, feed the per-candidate statistics the model router reads.
   */
  function observe(dimension, engine, score, { latencyS = null, costUsd = null } = {}) {
    if (typeof engine !== "string" || !engine) return;
    const record = state.candidates.get(engine) ?? { calls: 0, n: 0, latencyEma: null, costUsd: 0 };
    if (Number.isFinite(latencyS) && latencyS >= 0) record.latencyEma = record.latencyEma === null ? latencyS : (1 - cfg.alpha) * record.latencyEma + cfg.alpha * latencyS;
    if (Number.isFinite(costUsd) && costUsd >= 0) record.costUsd += costUsd;
    if (Number.isFinite(score)) record.n += 1;
    state.candidates.set(engine, record);
    if (typeof dimension !== "string" || !Number.isFinite(score)) {
      void persist();
      return;
    }
    if (!state.scores.has(dimension)) state.scores.set(dimension, new Map());
    const entries = state.scores.get(dimension);
    const entry = entries.get(engine) ?? { ema: 0, n: 0 };
    entry.n += 1;
    entry.ema = entry.n === 1 ? score : (1 - cfg.alpha) * entry.ema + cfg.alpha * score;
    entries.set(engine, entry);
    state.observations += 1;
    void persist();
  }

  /** Per-candidate statistics: calls, measured latency, cost so far, and the per-dimension EMA. */
  function candidateStats() {
    const out = {};
    const ids = new Set([...state.candidates.keys(), ...(Array.isArray(cfg.candidates) ? cfg.candidates.map(c => c.id) : []), ...cfg.cheapChain, cfg.escalateTo]);
    for (const id of ids) {
      const record = state.candidates.get(id) ?? { calls: 0, n: 0, latencyEma: null, costUsd: 0 };
      const dimensions = {};
      for (const [dimension, entries] of state.scores) {
        const entry = entries.get(id);
        if (entry) dimensions[dimension] = { ema: Number(entry.ema.toFixed(4)), n: entry.n };
      }
      out[id] = { tier: tierOf(id), calls: record.calls, n: record.n, latency_ema_s: record.latencyEma === null ? null : Number(record.latencyEma.toFixed(4)), cost_usd_total: Number(record.costUsd.toFixed(6)), dimensions };
    }
    return out;
  }

  /** Verify-and-escalate inside one step: the reflex answer was vetoed and there is time to think. */
  function shouldEscalateInline({ engine, token, gate, deadlineLeftMs = Infinity } = {}) {
    if (!cfg.inlineEscalate || !has(cfg.escalateTo)) return false;
    if (engine === cfg.escalateTo) return false;
    if (token === "X" || !gate || gate.allowed !== false) return false;
    if (!(deadlineLeftMs > cfg.inlineMinDeadlineMs)) return false;
    return budgetAllows();
  }

  /** Rotate the persisted scores to `<path>.bak-<unix_ms>` (queued before any later persist). Returns the backup path or null. */
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
    persistQueue = persistQueue.then(run, run).catch(error => logger?.warn?.({ event: "router_backup_failed", path, error: error.message }));
    return target;
  }

  /** Forget everything learned (scores, counters, cooldown). For evaluation's cold arm; persisted too. */
  function reset() {
    state.steps = 0;
    state.deliberate = 0;
    state.escalations = 0;
    state.inlineEscalations = 0;
    state.cooldownLeft = 0;
    state.lastReason = null;
    state.scores = new Map();
    state.observations = 0;
    state.candidates = new Map();
    state.suggestions = new Map();
    state.suggested = { offered: 0, used: 0, stale: 0, unavailable: 0, over_budget: 0 };
    void persist();
  }

  function snapshot() {
    const scores = scoresSnapshot();
    return {
      mode: cfg.mode,
      cheap_chain: [...cfg.cheapChain],
      escalate_to: cfg.escalateTo,
      candidates: candidateStats(),
      suggestions: { ...state.suggested, pending: state.suggestions.size },
      k: cfg.k,
      cooldown: cfg.cooldown,
      max_deliberate_fraction: cfg.maxDeliberateFraction,
      inline_escalate: cfg.inlineEscalate,
      steps: state.steps,
      deliberate: state.deliberate,
      deliberate_fraction: state.steps ? Number((state.deliberate / state.steps).toFixed(3)) : 0,
      escalations: state.escalations,
      inline_escalations: state.inlineEscalations,
      cooldown_left: state.cooldownLeft,
      observations: state.observations,
      memory_path: path,
      loaded: state.loaded,
      scores,
    };
  }

  return {
    route,
    commit,
    observe,
    suggest,
    candidateStats,
    has,
    tierOf,
    mode: cfg.mode,
    shouldEscalateInline,
    state: snapshot,
    load,
    restore,
    reset,
    backup,
    flush: () => persistQueue,
    escalateTo: cfg.escalateTo,
    cheapChain: [...cfg.cheapChain],
  };
}
