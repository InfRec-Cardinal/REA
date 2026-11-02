// The model router: a lightweight LLM that, asynchronously, picks the candidate for the *next* step.
//
// Agent-as-a-Router (ACRouter) routes each task with an LLM that reads the
// task, the candidate models with their cost, and its own accumulated per-dimension quality memory,
// and answers {"model": ..., "reasoning": ...}; epsilon-greedy exploration keeps every candidate
// sampled. Here the "task" is the next control step: the situation the reflex saw, the episodic
// signals (refusal streak, oscillation, stall, last outcome), the step latency budget, and the router's
// per-dimension memory over candidates. The call runs *after* a step, off the control path, and its
// answer is a suggestion the rule router may take for the following step (`router.suggest`). If it is
// late, wrong, or absent, the rule router decides as before — the model can only ever bias who is
// asked, never whether the robot moves.

import { performance } from "node:perf_hooks";

import { InferenceError, apiRoot, isDeepSeek, providerError } from "./provider.mjs";
import { situationPhrase } from "./memory.mjs";

export const ROUTER_SYSTEM_PROMPT = `You route one step of a humanoid robot's navigation controller to one of several candidate models.
Each step, a candidate looks at the forward camera frame and answers with exactly one action token (W forward, Q/E turn, S back, A/D strafe, X stop). A fused safety gate then checks the token against live depth and lidar and may refuse it; a refusal wastes the step.
Your objective: maximise the chance the chosen candidate's token passes the gate and makes progress, at the lowest cost and within the step latency budget. Prefer the cheapest reflex candidate while it is doing well. Escalate to a slower or stronger candidate when the reflex keeps being refused, oscillates, stalls, or the situation is tight (obstacle near). A candidate whose latency exceeds the budget is only acceptable when the robot is already stopped.
Respond with ONLY a JSON object: {"candidate": "<candidate id>", "reasoning": "<one short sentence>"}`;

function fmt(value, digits = 2) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "n/a";
}

/** Expected latency for a candidate: what the router measured, else half its timeout as a guess. */
export function expectedLatencyS(candidate, stats) {
  const ema = stats?.latency_ema_s;
  if (Number.isFinite(ema) && ema > 0) return ema;
  return (candidate.timeoutMs ?? 6000) / 2000;
}

/** Candidates a step may use: reflex always; slower ones only within the budget, or when the robot is stopped. */
export function eligibleCandidates(candidates, statsById, { stepBudgetMs = 10000, stopped = false } = {}) {
  return candidates.filter(c => c.reachable !== false && (c.tier === "reflex" || stopped || expectedLatencyS(c, statsById[c.id]) * 1000 <= stepBudgetMs));
}

export function buildRouterPrompt({ candidates, statsById, dimension, situation, signals, stepBudgetMs, lastStep, memoryLines }) {
  const lines = ["## Candidates (id | tier | model | price $/1M in,out | measured latency s | calls | quality in this situation class)"];
  for (const c of candidates) {
    const s = statsById[c.id] ?? {};
    const dim = s.dimensions?.[dimension];
    const price = c.price ? `${c.price.in},${c.price.out}` : "?";
    lines.push(`- ${c.id} | ${c.tier} | ${c.model} | ${price} | ${fmt(s.latency_ema_s)} | ${s.calls ?? 0} | ${dim ? `${fmt(dim.ema)} (n=${dim.n})` : "no data"}`);
  }
  lines.push("", `## Step to route`, `Situation class: ${dimension}`, `Situation: ${situation ? situationPhrase(situation) : "unknown"}`, `Step latency budget: ${(stepBudgetMs / 1000).toFixed(1)} s`);
  if (signals) {
    lines.push(`Signals: refusal streak ${signals.refusalStreak ?? 0}, bad streak ${signals.badStreak ?? 0}, oscillation ${signals.oscillation ?? 0}, stall ${signals.stall ? "yes" : "no"}, last token ${signals.lastToken ?? "none"}, robot ${signals.lastToken === "X" || !signals.lastToken ? "stopped" : "moving"}`);
  }
  if (lastStep) {
    lines.push(`Last step: candidate ${lastStep.candidate ?? "?"} answered ${lastStep.token ?? "?"}, outcome ${lastStep.outcome ?? "?"} (score ${fmt(lastStep.score)}), latency ${fmt(lastStep.latencyS)} s${lastStep.refused ? ", refused by the gate" : ""}`);
  }
  if (memoryLines?.length) lines.push("", "## Recent lessons", ...memoryLines.slice(0, 3).map(l => `- ${l}`));
  lines.push("", "Choose the candidate for the next step.");
  return lines.join("\n");
}

export function parseRouterAnswer(text, ids) {
  if (typeof text !== "string") return null;
  const tryJson = raw => {
    try {
      const data = JSON.parse(raw);
      const id = typeof data.candidate === "string" ? data.candidate.trim() : typeof data.model === "string" ? data.model.trim() : null;
      if (id && ids.includes(id)) return { candidate: id, reasoning: String(data.reasoning ?? "").slice(0, 200) };
    } catch {
      /* not JSON */
    }
    return null;
  };
  const direct = tryJson(text.trim());
  if (direct) return direct;
  const match = /\{[^{}]*"(?:candidate|model)"[^{}]*\}/su.exec(text);
  if (match) {
    const inner = tryJson(match[0]);
    if (inner) return inner;
  }
  // Longest id mentioned anywhere in the text (so "token@deepseek-v4-pro" beats "token").
  const mentioned = ids.filter(id => text.includes(id)).sort((a, b) => b.length - a.length);
  return mentioned.length ? { candidate: mentioned[0], reasoning: `extracted: ${text.slice(0, 120)}` } : null;
}

/** ACRouter's exploration: under-sampled candidates first, then epsilon-greedy. */
export function chooseWithExploration({ chosen, eligible, statsById, dimension, minTries = 1, epsilon = 0.05, rng = Math.random }) {
  const under = eligible.filter(c => ((statsById[c.id]?.dimensions?.[dimension]?.n) ?? 0) < minTries);
  if (under.length) {
    const pick = under[Math.floor(rng() * under.length)];
    return { candidate: pick.id, reason: `explore:under_sampled`, explored: true };
  }
  if (eligible.length > 1 && rng() < epsilon) {
    const pick = eligible[Math.floor(rng() * eligible.length)];
    return { candidate: pick.id, reason: `explore:epsilon`, explored: true };
  }
  return { candidate: chosen, reason: "model", explored: false };
}

export function createModelRouter({
  apiKey,
  baseUrl,
  dispatcher = undefined,
  model,
  router,
  candidates,
  logger = null,
  epsilon = 0.05,
  minTries = 1,
  deadlineMs = 3000,
  stepBudgetMs = 10000,
  fetchImpl = fetch,
  rng = Math.random,
  now = () => performance.now(),
}) {
  if (!router || typeof router.suggest !== "function") throw new TypeError("model router needs a router with suggest()");
  const list = candidates?.list ?? candidates ?? [];
  const ids = list.map(c => c.id);
  const url = `${apiRoot(baseUrl)}/chat/completions`;
  const inflight = new Set();
  const stats = { calls: 0, suggested: 0, explored: 0, failed: 0, skipped_busy: 0, parse_failed: 0, latency_ema_s: null, usage: { prompt_tokens: 0, completion_tokens: 0 }, last: null };

  async function ask(system, user) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), deadlineMs);
    try {
      const response = await fetchImpl(url, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
        dispatcher,
        signal: controller.signal,
        body: JSON.stringify({
          model,
          stream: false,
          max_tokens: 96,
          temperature: 0,
          messages: [{ role: "system", content: system }, { role: "user", content: user }],
          ...(isDeepSeek(model) ? { thinking: { type: "disabled" } } : {}),
        }),
      });
      if (!response.ok) throw new InferenceError(`router model upstream ${response.status}`, { status: response.status, retryable: true });
      const data = await response.json();
      const message = data.choices?.[0]?.message ?? {};
      const text = typeof message.content === "string" && message.content.trim() ? message.content : String(message.reasoning_content ?? "");
      return { text, usage: data.usage ?? null };
    } finally {
      clearTimeout(timer);
    }
  }

  /** Decide the candidate for step `seq` of `session`, in the background. Resolves to a status object. */
  async function suggestNext(session, { seq, dimension = "unknown", situation = null, signals = null, lastStep = null, memoryLines = [] } = {}) {
    if (!apiKey) return { status: "disabled" };
    const key = session?.id ?? "*";
    if (inflight.has(key)) {
      stats.skipped_busy += 1;
      return { status: "skipped_busy" };
    }
    inflight.add(key);
    const started = now();
    try {
      const statsById = router.candidateStats?.() ?? {};
      const stopped = !signals?.lastToken || signals.lastToken === "X";
      const eligible = eligibleCandidates(list, statsById, { stepBudgetMs, stopped });
      if (eligible.length === 0) return { status: "no_candidates" };
      stats.calls += 1;
      const prompt = buildRouterPrompt({ candidates: eligible, statsById, dimension, situation, signals, stepBudgetMs, lastStep, memoryLines });
      let chosen = null;
      let reasoning = "";
      try {
        const answer = await ask(ROUTER_SYSTEM_PROMPT, prompt);
        if (answer.usage) {
          stats.usage.prompt_tokens += Number(answer.usage.prompt_tokens ?? 0);
          stats.usage.completion_tokens += Number(answer.usage.completion_tokens ?? 0);
        }
        const parsed = parseRouterAnswer(answer.text, eligible.map(c => c.id));
        if (parsed) {
          chosen = parsed.candidate;
          reasoning = parsed.reasoning;
        } else {
          stats.parse_failed += 1;
        }
      } catch (error) {
        stats.failed += 1;
        logger?.warn?.({ event: "router_model_failed", model, error: providerError(error, { apiKey }).message });
      }
      const latencyS = (now() - started) / 1000;
      stats.latency_ema_s = stats.latency_ema_s === null ? latencyS : 0.7 * stats.latency_ema_s + 0.3 * latencyS;
      // No usable answer: exploration alone may still pick an under-sampled candidate; otherwise leave the rule router alone.
      const fallback = chosen ?? eligible[0].id;
      const pick = chooseWithExploration({ chosen: fallback, eligible, statsById, dimension, minTries, epsilon, rng });
      if (!chosen && !pick.explored) return { status: "no_answer", latencyS };
      if (pick.explored) stats.explored += 1;
      const reason = pick.explored ? pick.reason : "model";
      router.suggest({ candidate: pick.candidate, forSeq: seq, reason, sessionId: key, meta: { reasoning, latencyS, model } });
      stats.suggested += 1;
      stats.last = { seq, candidate: pick.candidate, reason, reasoning, latency_s: Number(latencyS.toFixed(3)), dimension };
      logger?.info?.({ event: "router_model_suggestion", session_id: key, seq, candidate: pick.candidate, reason, reasoning, latency_s: Number(latencyS.toFixed(3)) });
      return { status: pick.explored ? "explored" : "suggested", candidate: pick.candidate, reasoning, latencyS };
    } finally {
      inflight.delete(key);
    }
  }

  return { suggestNext, stats: () => ({ model, deadline_ms: deadlineMs, epsilon, min_tries: minTries, ...stats }), model, enabled: Boolean(apiKey) };
}
