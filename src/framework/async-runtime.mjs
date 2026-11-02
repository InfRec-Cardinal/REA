import { performance } from "node:perf_hooks";

import { validateAction } from "./actions.mjs";
import { configVersions, loadLoopConfig } from "./loop/config.mjs";
import { armEvidence } from "./loop/evidence.mjs";
import { screenAction } from "./loop/gate.mjs";
import { judgeCompletion } from "./loop/judge.mjs";
import { planRecovery } from "./loop/recover.mjs";
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
    return await Promise.race([Promise.resolve().then(() => {
      requestSignal.throwIfAborted();
      return operation(requestSignal);
    }), aborted]);
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
 * System 1 selects a System 0 capability at every action boundary. System 2
 * publishes advice independently, with one active request. after_dispatch gives
 * System0 the first turn and skips busy thinking slots instead of queuing input.
 * A policy/transport must honor its AbortSignal to stop its own external work.
 * The caller owns close(); this function resets policies and stops the environment.
 */
export async function runAsyncEpisode({
  environment,
  system0,
  system1,
  system2,
  mode = "async",
  task = null,
  seed = 0,
  instruction = "Complete the task efficiently using the available capabilities.",
  maxSteps = 200,
  maxDurationMs = 300000,
  decisionTimeoutMs = 30000,
  thinkingTimeoutMs = 60000,
  thinkEverySteps = 1,
  thinkingSchedule = "before_route",
  minControllerSteps = 1,
  maxAdviceAgeSteps = 12,
  historyLimit = 12,
  maxNotesChars = 4000,
  onEvent = () => {},
  onTransition = async () => {},
  loop = null,
  signal,
}) {
  if (!Array.isArray(system0) || system0.length === 0) throw new Error("system0 requires at least one capability");
  if (new Set(system0.map(candidate => candidate.id)).size !== system0.length) throw new Error("system0 capability ids must be unique");
  if (!["async", "sync"].includes(mode)) throw new Error("mode must be async or sync");
  if (!["before_route", "after_dispatch"].includes(thinkingSchedule)) throw new Error("thinkingSchedule must be before_route or after_dispatch");
  if (![maxSteps, thinkEverySteps, minControllerSteps].every(value => Number.isInteger(value) && value >= 1)) throw new RangeError("step budgets, thinkEverySteps and minControllerSteps must be positive integers");
  if (!(maxDurationMs > 0) || !(decisionTimeoutMs > 0) || !(thinkingTimeoutMs > 0) || !(maxAdviceAgeSteps >= 0)) throw new RangeError("timeouts must be positive and maxAdviceAgeSteps nonnegative");
  if (activeEnvironments.has(environment)) throw new Error("this environment already has an episode in progress");
  activeEnvironments.add(environment);

  const started = performance.now();
  const lifetime = new AbortController();
  const episodeSignal = signal ? AbortSignal.any([signal, lifetime.signal]) : lifetime.signal;
  const episodeTimer = setTimeout(() => lifetime.abort(deadlineError("episode duration budget exhausted")), maxDurationMs);
  const candidates = system0.map(({ id, kind, description }) => ({ id, kind: kind ?? "policy", ...(description === undefined ? {} : { description }) }));
  const history = [];
  const thinkingIntervals = [];
  const environmentIntervals = [];
  const usage = { system0: {}, system1: {}, system2: {} };
  const knownUsageCalls = { system0: 0, system1: 0, system2: 0 };
  const metrics = {
    mode, resetMs: null, firstActionMs: null, firstActionAfterResetMs: null, overlapMs: 0, blockedThinkingMs: 0,
    routedWhileThinking: 0, system0Calls: 0, system1Calls: 0, system2Calls: 0,
    corrections: 0, routeChanges: 0, adviceApplied: 0, thinkingFailures: 0,
    thinkingMs: 0, environmentMs: 0, eventCounts: {},
  };
  let current;
  let steps = 0;
  let attempts = 0;
  let notes = "";
  let previousCandidateId = null;
  let controllerSteps = 0;
  let latestAdvice = null;
  let pendingAdvice = null;
  let inFlight = null;
  let queuedInput = null;
  let thinkingRequested = false;
  let closed = false;
  let rewardTotal = 0;
  let rewardKnown = false;
  let reason = "max_steps";
  let failure = null;
  // Evidence loop (optional): gate before dispatch, judge completion claims, queue recovery macros.
  const nodes = loop ? loadLoopConfig(loop === true ? {} : loop) : null;
  const loopState = { queue: [], attempts: {}, recoverySteps: 0, guidance: null };
  const thinkAfterDispatch = mode === "async" && thinkingSchedule === "after_dispatch";

  const now = () => performance.now() - started;
  const remainingMs = () => maxDurationMs - now();
  const emit = (type, status, details = {}) => {
    if (closed) return;
    const key = `${type}.${status}`;
    metrics.eventCounts[key] = (metrics.eventCounts[key] ?? 0) + 1;
    onEvent({ type, status, atMs: now(), timestampMs: Date.now(), step: steps, ...details });
  };
  const captureUsage = (system, result) => {
    if (result?.usage) {
      knownUsageCalls[system] += 1;
      sumUsage(usage[system], result.usage);
    }
  };
  const freshAdvice = () => {
    if (pendingAdvice) {
      if (latestAdvice && !latestAdvice.applied) emit("thinking", "stale", { id: latestAdvice.id, sourceStep: latestAdvice.sourceStep, reason: "superseded" });
      latestAdvice = pendingAdvice;
      pendingAdvice = null;
    }
    if (latestAdvice && (steps - latestAdvice.sourceStep > maxAdviceAgeSteps || (latestAdvice.candidateId && !system0.some(candidate => candidate.id === latestAdvice.candidateId)))) {
      emit("thinking", "stale", { id: latestAdvice.id, sourceStep: latestAdvice.sourceStep, reason: steps - latestAdvice.sourceStep > maxAdviceAgeSteps ? "observation_age" : "unknown_candidate" });
      latestAdvice = null;
    }
    return latestAdvice ? { ...latestAdvice } : null;
  };
  const snapshot = () => ({
    observation: current.observation,
    actions: current.observation.actions ?? environment.actions,
    history: [...history], notes, feedback: current.info,
    instruction, task, seed, step: steps, candidates,
    advice: freshAdvice(), thinking: inFlight !== null, previousCandidateId, guidance: loopState.guidance,
    budget: { stepsRemaining: maxSteps - steps, remainingMs: remainingMs(), decisionMs: Math.min(decisionTimeoutMs, remainingMs()) },
  });
  const recordTransition = record => bounded(() => onTransition(record), remainingMs(), episodeSignal);

  function launchThinking(input) {
    const id = `think-${++metrics.system2Calls}`;
    const controller = new AbortController();
    const request = { id, sourceStep: input.step, controller, interval: { start: now(), end: null }, promise: null };
    inFlight = request;
    thinkingIntervals.push(request.interval);
    emit("thinking", "submitted", { id, sourceStep: input.step });
    emit("thinking", "started", { id, sourceStep: input.step });
    const thinkingMs = Math.min(thinkingTimeoutMs, remainingMs());
    request.promise = bounded(requestSignal => system2.think({ ...input, budget: { ...input.budget, remainingMs: remainingMs(), decisionMs: thinkingMs, thinkingMs }, signal: requestSignal }), thinkingMs, AbortSignal.any([episodeSignal, controller.signal]))
      .then(proposal => {
        if (closed || episodeSignal.aborted) return;
        captureUsage("system2", proposal);
        if (pendingAdvice) emit("thinking", "stale", { id: pendingAdvice.id, sourceStep: pendingAdvice.sourceStep, reason: "superseded" });
        pendingAdvice = { ...proposal, id, sourceStep: input.step, completedAtMs: now(), applied: false };
        emit("thinking", "completed", { id, sourceStep: input.step, candidateId: proposal?.candidateId ?? null, directive: proposal?.directive ?? null, reason: proposal?.reason ?? null, model: proposal?.model ?? null, usage: proposal?.usage ?? null });
      }, error => {
        if (closed || episodeSignal.aborted) return;
        captureUsage("system2", error);
        metrics.thinkingFailures += 1;
        emit("thinking", error?.code === "deadline" ? "cancelled" : "failed", { id, sourceStep: input.step, reason: error?.code === "deadline" ? "thinking_timeout" : "thinking_error", error: error?.message ?? String(error) });
      })
      .finally(() => {
        request.interval.end ??= now();
        if (inFlight === request) inFlight = null;
        if (!thinkAfterDispatch && !closed && !episodeSignal.aborted && queuedInput) {
          const next = queuedInput;
          queuedInput = null;
          launchThinking(next);
        }
      });
    return request.promise;
  }

  const recoveryFailing = () => {
    const recent = history.slice(-nodes.recover.abortOnFailures);
    return recent.length === nodes.recover.abortOnFailures && recent.every(record => record.route?.kind === "recover" && nodes.route.failureStatuses.includes(record.info?.execution?.status));
  };
  // Ask System2 now, outside the regular cadence, so the supervisor's guidance reaches it.
  async function requestThinking() {
    if (!system2) return;
    if (thinkAfterDispatch) {
      thinkingRequested = true;
      return;
    }
    if (inFlight) return;
    const input = snapshot();
    loopState.guidance = null;
    if (mode === "sync") await launchThinking(input);
    else launchThinking(input);
  }
  const evidenceOf = record => armEvidence(record.nextObservation?.state ?? record.observation?.state, record.info?.execution ?? null, nodes.judge);
  async function screenWithLoop({ input, decision }) {
    const state = input.observation?.state;
    const gate = screenAction({ decision, state, history, config: nodes.gate, progress: nodes.progress });
    const trace = { versions: configVersions(nodes), gate: { verdict: gate.verdict, reason: gate.reason ?? null, metrics: gate.metrics ?? null } };
    if (gate.verdict === "allow") return { trace };
    const blocking = gate.verdict !== "hold_claim" || !gate.passThrough;
    const evidence = armEvidence(state, history.at(-1)?.info?.execution ?? null, nodes.judge);
    const verdict = judgeCompletion({ instruction: input.instruction ?? input.observation?.text ?? "", evidence, history, evidenceOf, config: nodes.judge });
    trace.judge = verdict;
    emit("loop", "judged", { gate: gate.verdict, verdict: verdict.verdict, reason: verdict.reason });
    if (verdict.verdict === "not_complete") {
      const attempt = loopState.attempts[verdict.reason] ?? 0;
      const budgetOk = loopState.recoverySteps < nodes.recover.maxFraction * maxSteps;
      const plan = attempt < nodes.recover.attemptsPerReason && budgetOk ? planRecovery({ verdict, evidence, history, attempt, config: nodes.recover }) : null;
      if (plan) loopState.attempts[verdict.reason] = attempt + 1;
      if (plan?.actions?.length) {
        loopState.queue = plan.actions.slice(1);
        loopState.recoverySteps += 1;
        trace.recover = { macro: plan.macro, arm: plan.arm, attempt, steps: plan.actions.length, thought: `judge: ${verdict.reason}; recovery ${plan.macro} on ${plan.arm} arm` };
        return { trace, recovery: plan.actions[0] };
      }
      loopState.guidance = plan?.directive ?? `Evidence says the task is not complete (${verdict.reason}). Recovery attempts for this reason are used up; re-plan from the images.`;
      trace.recover = { macro: plan?.macro ?? "exhausted" };
      await requestThinking();
    } else if (verdict.verdict === "uncertain") {
      loopState.guidance = `A completion claim could not be verified from robot state (${verdict.reason}; evidence ${JSON.stringify(verdict.detail)}). Check the images and continue working unless the task is visibly complete.`;
      await requestThinking();
    } else if (blocking) {
      loopState.guidance = null;
    }
    if (blocking && verdict.verdict !== "complete") {
      loopState.guidance ??= `Your command was rejected before execution: ${gate.reason}.`;
      await requestThinking();
    }
    return { trace, blocked: blocking && verdict.verdict !== "complete" };
  }
  // A blocked command consumes one decision of the budget without moving the robot.
  async function recordBlocked({ input, decision, action, candidate, selected, loopTrace, decisionStarted }) {
    steps += 1;
    if (typeof decision.notes === "string") notes = decision.notes.slice(0, maxNotesChars);
    const record = {
      type: "step", step: steps, action, observation: input.observation, nextObservation: input.observation,
      terminated: false, truncated: false, success: null, reward: null,
      info: { gate: { allowed: false, verdict: loopTrace.gate.verdict, reason: loopTrace.gate.reason } },
      model: decision.model ?? candidate.policy.id ?? candidate.id, route: { ...selected, kind: candidate.kind ?? "policy", adviceId: null },
      thought: decision.thought ?? null, notes,
      ...(decision.claim ? { claim: decision.claim } : {}),
      ...(decision.review ? { review: decision.review } : {}),
      usage: decision.usage ?? null, nodes: loopTrace,
      async: { mode, thinkingAtRoute: input.thinking, thinkingAtDispatch: false, adviceId: null },
      timing: { decisionMs: now() - decisionStarted, environmentMs: 0, observationAgeAtDispatchMs: null },
    };
    history.push({ ...record, observation: observationSummary(input.observation), nextObservation: observationSummary(input.observation) });
    if (history.length > historyLimit) history.splice(0, history.length - historyLimit);
    emit("loop", "blocked", { verdict: loopTrace.gate.verdict, reason: loopTrace.gate.reason });
    await recordTransition(record);
  }

  const route = system1 ?? (async input => {
    const candidateId = input.advice?.candidateId ?? input.previousCandidateId ?? candidates[0].id;
    if (candidateId !== previousCandidateId && controllerSteps < minControllerSteps && candidates.some(candidate => candidate.id === previousCandidateId)) {
      return {
        candidateId: previousCandidateId,
        reason: `Continue ${previousCandidateId}: ${controllerSteps}/${minControllerSteps} action blocks completed before switching`,
        usedAdvice: false,
        deferredCandidateId: candidateId,
      };
    }
    return {
      candidateId,
      reason: input.advice?.reason ?? "continue available capability",
      usedAdvice: Boolean(input.advice && (input.advice.candidateId || input.advice.directive != null)),
    };
  });

  try {
    emit("episode", "submitted", { mode, task, seed });
    episodeSignal.throwIfAborted();
    current = transitionView(await bounded(requestSignal => environment.reset({ task, seed, signal: requestSignal }), remainingMs(), episodeSignal));
    metrics.resetMs = now();
    for (const candidate of system0) {
      if (candidate.policy.reset) await bounded(requestSignal => candidate.policy.reset({ task, seed, environment: environment.id, signal: requestSignal }), remainingMs(), episodeSignal);
    }
    if (system2?.reset) await bounded(requestSignal => system2.reset({ task, seed, environment: environment.id, signal: requestSignal }), remainingMs(), episodeSignal);
    emit("episode", "started", { mode, task, seed });
    await recordTransition({ type: "reset", task, seed, ...current });

    while (steps < maxSteps && !current.terminated && !current.truncated) {
      episodeSignal.throwIfAborted();
      if (system2 && !thinkAfterDispatch && steps % thinkEverySteps === 0) {
        const input = snapshot();
        loopState.guidance = null;
        if (mode === "sync") {
          const waitStarted = now();
          await launchThinking(input);
          metrics.blockedThinkingMs += now() - waitStarted;
          episodeSignal.throwIfAborted();
        } else if (inFlight) {
          queuedInput = input;
        } else {
          launchThinking(input);
        }
      }

      const input = snapshot();
      if (!Array.isArray(input.actions) || input.actions.length === 0) throw new Error("environment exposed no available actions");
      const decisionStarted = now();
      let selected, candidate, decision, usesAdvice = false, directive = null;
      if (nodes && loopState.queue.length && recoveryFailing()) {
        emit("loop", "recovery_aborted", { remaining: loopState.queue.length });
        loopState.queue = [];
        loopState.guidance = "The last recovery motions failed to plan or track. Choose a reachable approach from the images.";
        await requestThinking();
      }
      if (nodes && loopState.queue.length) {
        selected = { candidateId: "loop:recover", reason: "recovery macro" };
        candidate = { id: "loop:recover", kind: "recover", policy: { id: "loop:recover" } };
        decision = { action: loopState.queue.shift(), thought: `recovery macro step (${loopState.queue.length} left)` };
        loopState.recoverySteps += 1;
      } else {
        metrics.system1Calls += 1;
        if (input.thinking) metrics.routedWhileThinking += 1;
        const routeId = `route-${metrics.system1Calls}`;
        emit("routing", "started", { id: routeId, thinking: input.thinking, adviceId: input.advice?.id ?? null });
        selected = await bounded(requestSignal => route({ ...input, signal: requestSignal }), Math.min(decisionTimeoutMs, remainingMs()), episodeSignal).catch(error => {
          captureUsage("system1", error);
          throw error;
        });
        captureUsage("system1", selected);
        candidate = system0.find(item => item.id === selected.candidateId);
        if (!candidate) throw new Error(`System 1 selected unavailable capability ${JSON.stringify(selected.candidateId)}`);
        usesAdvice = Boolean(input.advice && (selected.usedAdvice === true || selected.candidateId === input.advice.candidateId));
        emit("routing", "completed", { id: routeId, candidateId: candidate.id, kind: candidate.kind ?? "policy", reason: selected.reason ?? null, thinking: inFlight !== null, adviceId: input.advice?.id ?? null, usesAdvice });
        metrics.system0Calls += 1;
        directive = usesAdvice ? input.advice?.directive ?? null : null;
        decision = await bounded(requestSignal => candidate.policy.decide({
          ...input,
          instruction: directive === null ? input.instruction : `${input.instruction}\n\nSystem2 directive for this action: ${typeof directive === "string" ? directive : JSON.stringify(directive)}`,
          directive, route: selected, signal: requestSignal,
        }), Math.min(decisionTimeoutMs - (now() - decisionStarted), remainingMs()), episodeSignal).catch(error => {
          captureUsage("system0", error);
          throw error;
        });
        captureUsage("system0", decision);
      }
      const usedAdvice = Boolean(input.advice && (usesAdvice || decision.usedAdvice === true));
      let action = validateAction(decision.action, input.actions);
      let loopTrace = null;
      if (nodes && candidate.kind !== "recover") {
        const outcome = await screenWithLoop({ input, decision, action });
        loopTrace = outcome.trace;
        if (outcome.blocked) {
          await recordBlocked({ input, decision, action, candidate, selected, loopTrace, decisionStarted });
          continue;
        }
        if (outcome.recovery) {
          action = validateAction(outcome.recovery, input.actions);
          candidate = { id: "loop:recover", kind: "recover", policy: { id: "loop:recover" } };
          // The judge rejected the policy's completion claim; the recorded action is the macro's, as in routed policies.
          const { claim: _rejectedClaim, ...kept } = decision;
          decision = { ...kept, thought: loopTrace.recover.thought };
        }
      }
      episodeSignal.throwIfAborted();
      const decisionEnded = now();
      const dispatchAt = Date.now();
      const actionId = `action-${++attempts}`;
      if (usedAdvice && !input.advice.applied) {
        if (latestAdvice?.id === input.advice.id) latestAdvice.applied = true;
        metrics.adviceApplied += 1;
        if (steps > 0 && (previousCandidateId !== candidate.id || decision.usedAdvice === true)) metrics.corrections += 1;
        emit("thinking", "applied", { id: input.advice.id, sourceStep: input.advice.sourceStep, candidateId: candidate.id, actionId });
      }
      if (previousCandidateId && previousCandidateId !== candidate.id) metrics.routeChanges += 1;
      if (previousCandidateId !== candidate.id) controllerSteps = 0;
      previousCandidateId = candidate.id;
      metrics.firstActionMs ??= now();
      metrics.firstActionAfterResetMs ??= metrics.firstActionMs - metrics.resetMs;
      const environmentInterval = { start: now(), end: null };
      environmentIntervals.push(environmentInterval);
      emit("action", "started", { id: actionId, candidateId: candidate.id, kind: candidate.kind ?? "policy", action, thinking: inFlight !== null });
      let next;
      try {
        next = transitionView(await bounded(requestSignal => {
          const execution = environment.step(action, { observation: input.observation, signal: requestSignal, reasoning: { thought: decision.thought ?? null, route: selected, adviceId: usedAdvice ? input.advice.id : null } });
          // Dispatch System0 first. Busy background requests are skipped, not queued;
          // their completion can publish advice but cannot start another request.
          if (thinkAfterDispatch && system2 && !inFlight && (thinkingRequested || steps % thinkEverySteps === 0)) {
            const thinkingInput = snapshot();
            thinkingRequested = false;
            loopState.guidance = null;
            launchThinking(thinkingInput);
          }
          return execution;
        }, remainingMs(), episodeSignal));
      } catch (error) {
        emit("action", episodeSignal.aborted ? "cancelled" : "failed", { id: actionId, candidateId: candidate.id, error: error?.message ?? String(error) });
        throw error;
      } finally {
        environmentInterval.end = now();
      }
      steps += 1;
      controllerSteps += 1;
      if (next.reward !== null) { rewardKnown = true; rewardTotal += next.reward; }
      if (typeof decision.notes === "string") notes = decision.notes.slice(0, maxNotesChars);
      const record = {
        type: "step", step: steps, action, observation: input.observation, nextObservation: next.observation,
        terminated: next.terminated, truncated: next.truncated, success: next.success, reward: next.reward, info: next.info,
        model: decision.model ?? candidate.policy.id ?? candidate.id,
        route: { ...selected, kind: candidate.kind ?? "policy", adviceId: usedAdvice ? input.advice.id : null },
        thought: decision.thought ?? null, notes,
        // As in runEpisode: the progress layer reads completion claims from history.
        ...(decision.claim ? { claim: decision.claim } : {}),
        ...(decision.review ? { review: decision.review } : {}),
        usage: decision.usage ?? null,
        nodes: loopTrace ?? (nodes && candidate.kind === "recover" ? { versions: configVersions(nodes), recover: { remaining: loopState.queue.length } } : null),
        async: { mode, thinkingAtRoute: input.thinking, thinkingAtDispatch: thinkingIntervals.some(interval => interval.start <= environmentInterval.start && (interval.end === null || interval.end >= environmentInterval.start)), adviceId: usedAdvice ? input.advice.id : null },
        timing: { decisionMs: decisionEnded - decisionStarted, environmentMs: environmentInterval.end - environmentInterval.start, observationAgeAtDispatchMs: Number.isFinite(input.observation.timestampMs) ? Math.max(0, dispatchAt - input.observation.timestampMs) : null },
      };
      current = next;
      history.push({ ...record, observation: observationSummary(input.observation), nextObservation: observationSummary(next.observation) });
      if (history.length > historyLimit) history.splice(0, history.length - historyLimit);
      emit("action", "completed", { id: actionId, candidateId: candidate.id, success: next.success, terminated: next.terminated, durationMs: record.timing.environmentMs });
      await recordTransition(record);
    }
    if (current.terminated) reason = current.success === true ? "success" : "terminated";
    else if (current.truncated) reason = "environment_limit";
  } catch (error) {
    reason = signal?.aborted ? "cancelled" : error?.code === "deadline" ? "deadline" : "error";
    failure = { code: error?.code ?? error?.errorType ?? "episode_error", message: error?.message ?? String(error) };
  } finally {
    queuedInput = null;
    if (inFlight) {
      emit("thinking", "cancelled", { id: inFlight.id, sourceStep: inFlight.sourceStep, reason: "episode_ended" });
      inFlight.interval.end ??= now();
      inFlight.controller.abort(new Error("episode ended"));
    }
    if (latestAdvice && !latestAdvice.applied) emit("thinking", "stale", { id: latestAdvice.id, sourceStep: latestAdvice.sourceStep, reason: "episode_ended" });
    if (pendingAdvice) emit("thinking", "stale", { id: pendingAdvice.id, sourceStep: pendingAdvice.sourceStep, reason: "episode_ended" });
    // Close publication before abort settles, so old requests cannot publish into a later run.
    closed = true;
    clearTimeout(episodeTimer);
    lifetime.abort(new Error("episode ended"));
    try {
      await environment.stop?.();
    } catch (error) {
      failure ??= { code: "stop_error", message: error?.message ?? String(error) };
      reason = "error";
    }
    activeEnvironments.delete(environment);
  }

  for (const interval of thinkingIntervals) metrics.thinkingMs += interval.end - interval.start;
  for (const execution of environmentIntervals) {
    metrics.environmentMs += execution.end - execution.start;
    for (const thinking of thinkingIntervals) {
      metrics.overlapMs += Math.max(0, Math.min(execution.end, thinking.end) - Math.max(execution.start, thinking.start));
    }
  }
  const usageTotal = {};
  for (const value of Object.values(usage)) sumUsage(usageTotal, value);
  const decisionCalls = metrics.system0Calls + metrics.system2Calls + (system1 ? metrics.system1Calls : 0);
  const usageKnown = Object.values(knownUsageCalls).reduce((sum, value) => sum + value, 0);
  const usageCompleteBySystem = Object.fromEntries(["system0", "system1", "system2"].map(system => {
    const calls = system === "system1" && !system1 ? 0 : metrics[`${system}Calls`];
    return [system, calls === 0 ? null : calls === knownUsageCalls[system]];
  }));
  const summary = {
    environment: environment.id ?? null, policy: "system-router", mode, task, seed, reason, steps, actionAttempts: attempts,
    terminated: current?.terminated ?? false,
    truncated: (current?.truncated ?? false) || ["max_steps", "deadline", "cancelled"].includes(reason),
    success: failure ? null : current?.success ?? null, reward: rewardKnown ? rewardTotal : null,
    usage: usageKnown === 0 ? null : usageTotal, usageBySystem: usage, usageComplete: decisionCalls > 0 && decisionCalls === usageKnown, usageCompleteBySystem,
    decisionCalls, durationMs: now(), notes, finalObservation: observationSummary(current?.observation), info: current?.info ?? {}, error: failure,
    asyncMetrics: metrics,
  };
  const terminalStatus = reason === "cancelled" ? "cancelled" : failure ? "failed" : "completed";
  metrics.eventCounts[`episode.${terminalStatus}`] = (metrics.eventCounts[`episode.${terminalStatus}`] ?? 0) + 1;
  onEvent({ type: "episode", status: terminalStatus, atMs: now(), timestampMs: Date.now(), step: steps, reason, summary });
  await onTransition({ type: "episode_end", summary });
  return summary;
}
