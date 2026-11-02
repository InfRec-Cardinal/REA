import { performance } from "node:perf_hooks";

import { validateAction } from "./actions.mjs";
import { createRouteMemoryRouter } from "./route-memory.mjs";
import { observationSummary } from "./runtime.mjs";

const activeEnvironments = new WeakSet();

const deadlineError = message => Object.assign(new Error(message), { code: "deadline" });

async function bounded(operation, timeoutMs, signal) {
  signal.throwIfAborted();
  if (timeoutMs <= 0) throw deadlineError("operation duration budget exhausted");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(deadlineError(`operation exceeded ${Math.ceil(timeoutMs)} ms`)), Math.max(1, timeoutMs));
  const requestSignal = AbortSignal.any([signal, controller.signal]);
  let rejectAborted;
  const aborted = new Promise((resolve, reject) => {
    rejectAborted = () => reject(requestSignal.reason);
    requestSignal.addEventListener("abort", rejectAborted, { once: true });
  });
  try {
    return await Promise.race([Promise.resolve().then(() => operation(requestSignal)), aborted]);
  } finally {
    clearTimeout(timer);
    requestSignal.removeEventListener("abort", rejectAborted);
  }
}

function transitionView(result) {
  return {
    observation: result.observation,
    terminated: result.terminated === true,
    truncated: result.truncated === true,
    success: typeof result.success === "boolean" ? result.success : null,
    reward: Number.isFinite(result.reward) ? result.reward : null,
    info: result.info ?? {},
  };
}

function sumUsage(total, usage) {
  for (const [name, value] of Object.entries(usage ?? {})) {
    if (Number.isFinite(value)) total[name] = (total[name] ?? 0) + value;
  }
}

/**
 * REA's routed episode: System 1 picks the first System 0 capability, the capability runs until it returns, and at
 * that boundary System 1 decides whether to stop or escalate to the next capability, optionally after asking System 2
 * to judge completion from the current images. No capability waits for System 1 or System 2 while it runs. All runs
 * share one step and wall budget, and success is read once, after the episode, from environment.score().
 *
 * A run ends when a transition reports info.capability.status "returned" or "failed", when it is terminated (the
 * environment ends the episode unless it declares resumable: true), when it is truncated, or when it has used
 * maxStepsPerRun steps (status "budget"). The caller owns close(); this function resets policies and stops the
 * environment.
 */
export async function runRoutedEpisode({
  environment,
  system0,
  system1,
  system2,
  task = null,
  seed = 0,
  instruction = "Complete the task with the available capabilities.",
  maxSteps = 200,
  maxDurationMs = 300000,
  maxRuns = 10,
  maxStepsPerRun = null,
  decisionTimeoutMs = 30000,
  judgeTimeoutMs = 60000,
  scoreTimeoutMs = 120000,
  historyLimit = 12,
  maxNotesChars = 4000,
  onEvent = () => {},
  onTransition = async () => {},
  signal,
}) {
  if (!Array.isArray(system0) || system0.length === 0) throw new Error("system0 requires at least one capability");
  if (new Set(system0.map(candidate => candidate.id)).size !== system0.length) throw new Error("system0 capability ids must be unique");
  if (![maxSteps, maxRuns].every(value => Number.isInteger(value) && value >= 1)) throw new RangeError("maxSteps and maxRuns must be positive integers");
  if (maxStepsPerRun !== null && (!Number.isInteger(maxStepsPerRun) || maxStepsPerRun < 1)) throw new RangeError("maxStepsPerRun must be a positive integer");
  if (![maxDurationMs, decisionTimeoutMs, judgeTimeoutMs, scoreTimeoutMs].every(value => value > 0)) throw new RangeError("timeouts must be positive");
  if (system2 && typeof system2.judge !== "function") throw new TypeError("routed mode needs System 2 with judge(input), or no System 2");
  if (activeEnvironments.has(environment)) throw new Error("this environment already has an episode in progress");
  activeEnvironments.add(environment);

  const route = typeof system1 === "function" ? system1
    : system1?.route ? system1.route.bind(system1)
      : createRouteMemoryRouter({ recoverOn: system2 ? "judge" : "exception" }).route;
  const started = performance.now();
  const now = () => performance.now() - started;
  const remainingMs = () => maxDurationMs - now();
  const lifetime = new AbortController();
  const episodeSignal = signal ? AbortSignal.any([signal, lifetime.signal]) : lifetime.signal;
  const episodeTimer = setTimeout(() => lifetime.abort(deadlineError("episode duration budget exhausted")), maxDurationMs);
  const candidates = system0.map(({ id, kind, description }) => ({ id, kind: kind ?? "policy", ...(description === undefined ? {} : { description }) }));
  const runs = [];
  const history = [];
  const usage = { system0: {}, system1: {}, system2: {} };
  const metrics = { routeCalls: 0, judgeCalls: 0, judgeMs: 0, decisionCalls: 0, environmentMs: 0, eventCounts: {} };
  let current;
  let initialObservation = null;
  let steps = 0;
  let notes = "";
  let reason = null;
  let stopReason = null;
  let failure = null;
  let score = null;
  let closed = false;

  const emit = (type, status, details = {}) => {
    if (closed) return;
    const key = `${type}.${status}`;
    metrics.eventCounts[key] = (metrics.eventCounts[key] ?? 0) + 1;
    onEvent({ type, status, atMs: now(), timestampMs: Date.now(), step: steps, ...details });
  };
  const runSummaries = () => runs.map(({ index, candidateId, kind, status, steps: runSteps, reason: runReason, error, verdict }) => ({
    index, candidateId, kind, status, steps: runSteps, reason: runReason, error, verdict,
  }));
  const budget = () => ({ stepsRemaining: maxSteps - steps, remainingMs: remainingMs(), runsRemaining: maxRuns - runs.length });
  const recordTransition = record => bounded(() => onTransition(record), remainingMs(), episodeSignal);

  // System 2, at a boundary only: judge the current observation against the instruction. Never sees success.
  const judge = system2 ? async () => {
    const run = runs.at(-1) ?? null;
    const id = `judge-${++metrics.judgeCalls}`;
    const judgeStarted = now();
    emit("judge", "started", { id, afterRun: run?.index ?? null });
    let verdict;
    try {
      verdict = await bounded(requestSignal => system2.judge({
        observation: current.observation, initialObservation, instruction, task, seed, runs: runSummaries(), budget: budget(), signal: requestSignal,
      }), Math.min(judgeTimeoutMs, remainingMs()), episodeSignal);
    } catch (error) {
      if (episodeSignal.aborted) throw error;
      verdict = { verdict: "error", error: error?.message ?? String(error) };
    }
    metrics.judgeMs += now() - judgeStarted;
    sumUsage(usage.system2, verdict?.usage);
    if (run) run.verdict = verdict;
    emit("judge", verdict?.verdict === "error" ? "failed" : "completed", { id, afterRun: run?.index ?? null, verdict: verdict?.verdict ?? null, confidence: verdict?.confidence ?? null, error: verdict?.error ?? null });
    return verdict;
  } : null;

  async function executeRun(candidate, run, selection) {
    for (;;) {
      episodeSignal.throwIfAborted();
      if (steps >= maxSteps || (maxStepsPerRun !== null && run.steps >= maxStepsPerRun)) {
        run.status = "budget";
        return null;
      }
      const actions = current.observation.actions ?? environment.actions;
      if (!Array.isArray(actions) || actions.length === 0) throw new Error("environment exposed no available actions");
      const input = {
        observation: current.observation, actions, history: [...history], notes, feedback: current.info,
        instruction, task, seed, step: steps, candidates, route: selection,
        run: { index: run.index, candidateId: run.candidateId, steps: run.steps },
        budget: { ...budget(), decisionMs: Math.min(decisionTimeoutMs, remainingMs()) },
      };
      const decisionStarted = now();
      metrics.decisionCalls += 1;
      const decision = await bounded(requestSignal => candidate.policy.decide({ ...input, signal: requestSignal }), Math.min(decisionTimeoutMs, remainingMs()), episodeSignal);
      sumUsage(usage.system0, decision?.usage);
      const action = validateAction(decision.action, actions);
      const environmentStarted = now();
      const next = transitionView(await bounded(requestSignal => environment.step(action, {
        observation: input.observation, signal: requestSignal, reasoning: { thought: decision.thought ?? null, route: selection },
      }), remainingMs(), episodeSignal));
      const environmentMs = now() - environmentStarted;
      metrics.environmentMs += environmentMs;
      steps += 1;
      run.steps += 1;
      if (typeof decision.notes === "string") notes = decision.notes.slice(0, maxNotesChars);
      const capability = next.info?.capability ?? null;
      const record = {
        type: "step", step: steps, run: run.index, candidateId: run.candidateId, action,
        observation: input.observation, nextObservation: next.observation,
        terminated: next.terminated, truncated: next.truncated, success: next.success, reward: next.reward, info: next.info,
        model: decision.model ?? candidate.policy.id ?? candidate.id, thought: decision.thought ?? null, notes,
        usage: decision.usage ?? null, timing: { decisionMs: environmentStarted - decisionStarted, environmentMs },
      };
      current = next;
      history.push({ ...record, observation: observationSummary(input.observation), nextObservation: observationSummary(next.observation) });
      if (history.length > historyLimit) history.splice(0, history.length - historyLimit);
      emit("action", "completed", { candidateId: run.candidateId, run: run.index, capability: capability?.status ?? null, durationMs: environmentMs });
      await recordTransition(record);
      if (next.truncated) {
        run.status = "truncated";
        return "environment_limit";
      }
      if (capability?.status === "failed") {
        run.status = "failed";
        run.error = capability.error ?? null;
        return null;
      }
      if (next.terminated || capability?.status === "returned") {
        run.status = "returned";
        return next.terminated && environment.resumable !== true ? "terminated" : null;
      }
    }
  }

  try {
    emit("episode", "submitted", { mode: "routed", task, seed });
    current = transitionView(await bounded(requestSignal => environment.reset({ task, seed, signal: requestSignal }), remainingMs(), episodeSignal));
    initialObservation = current.observation;
    for (const candidate of system0) {
      if (candidate.policy.reset) await bounded(requestSignal => candidate.policy.reset({ task, seed, environment: environment.id, signal: requestSignal }), remainingMs(), episodeSignal);
    }
    if (system2?.reset) await bounded(requestSignal => system2.reset({ task, seed, environment: environment.id, signal: requestSignal }), remainingMs(), episodeSignal);
    emit("episode", "started", { mode: "routed", task, seed });
    await recordTransition({ type: "reset", task, seed, observation: current.observation, terminated: current.terminated, truncated: current.truncated, info: current.info });

    while (!reason) {
      if (steps >= maxSteps) { reason = "max_steps"; break; }
      if (runs.length >= maxRuns) { reason = "max_runs"; break; }
      metrics.routeCalls += 1;
      const selection = await bounded(requestSignal => route({
        observation: current.observation, initialObservation, instruction, task, seed, candidates,
        runs: runSummaries(), history: [...history], budget: budget(), judge, signal: requestSignal,
      }), Math.min(decisionTimeoutMs + (judge ? judgeTimeoutMs : 0), remainingMs()), episodeSignal);
      sumUsage(usage.system1, selection?.usage);
      if (!selection || selection.stop) {
        reason = "router_stop";
        stopReason = selection?.reason ?? null;
        emit("routing", "stopped", { reason: stopReason, runs: runs.length });
        break;
      }
      const candidate = system0.find(item => item.id === selection.candidateId);
      if (!candidate) throw new Error(`System 1 selected unavailable capability ${JSON.stringify(selection.candidateId)}`);
      const run = { index: runs.length, candidateId: candidate.id, kind: candidate.kind ?? "policy", reason: selection.reason ?? null, status: "running", steps: 0, startStep: steps, error: null, verdict: null, startedAtMs: now(), endedAtMs: null };
      runs.push(run);
      emit("routing", "completed", { candidateId: candidate.id, run: run.index, reason: run.reason });
      emit("run", "started", { candidateId: candidate.id, run: run.index });
      const ended = await executeRun(candidate, run, selection);
      run.endedAtMs = now();
      emit("run", "completed", { candidateId: candidate.id, run: run.index, runStatus: run.status, steps: run.steps });
      if (ended) reason = ended;
    }
  } catch (error) {
    // The wall budget running out is an episode outcome, not an error: the episode is still scored.
    if (error?.code === "deadline" && !signal?.aborted) reason = "deadline";
    else {
      reason = signal?.aborted ? "cancelled" : "error";
      failure = { code: error?.code ?? error?.errorType ?? "episode_error", message: error?.message ?? String(error) };
    }
    const running = runs.at(-1);
    if (running?.status === "running") {
      running.status = reason === "deadline" ? "budget" : "failed";
      running.endedAtMs = now();
    }
  } finally {
    clearTimeout(episodeTimer);
    lifetime.abort(new Error("episode ended"));
    try {
      await environment.stop?.();
    } catch (error) {
      failure ??= { code: "stop_error", message: error?.message ?? String(error) };
      reason = "error";
    }
  }

  // The sealed score: read once, after the last run, never on the control path.
  if (!failure && typeof environment.score === "function") {
    try {
      score = await bounded(() => environment.score(), scoreTimeoutMs, new AbortController().signal);
    } catch (error) {
      failure = { code: "score_error", message: error?.message ?? String(error) };
      reason = "error";
    }
  }
  activeEnvironments.delete(environment);
  closed = true;

  const success = failure ? null : typeof score?.success === "boolean" ? score.success : current?.success ?? null;
  const usageTotal = {};
  for (const value of Object.values(usage)) sumUsage(usageTotal, value);
  const summary = {
    environment: environment.id ?? null, policy: "routed-system", mode: "routed", task, seed, reason, stopReason, steps,
    runs: runSummaries().map((run, index) => ({ ...run, startStep: runs[index].startStep, durationMs: (runs[index].endedAtMs ?? now()) - runs[index].startedAtMs })),
    terminated: current?.terminated ?? false,
    truncated: (current?.truncated ?? false) || ["max_steps", "deadline", "cancelled"].includes(reason),
    success, score, usage: Object.keys(usageTotal).length ? usageTotal : null, usageBySystem: usage,
    durationMs: now(), notes, finalObservation: observationSummary(current?.observation), info: current?.info ?? {}, error: failure,
    routedMetrics: metrics,
  };
  onEvent({ type: "episode", status: reason === "cancelled" ? "cancelled" : failure ? "failed" : "completed", atMs: now(), timestampMs: Date.now(), step: steps, reason, summary });
  await onTransition({ type: "episode_end", summary });
  return summary;
}

