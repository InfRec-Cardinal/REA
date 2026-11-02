import { performance } from "node:perf_hooks";

import { validateAction } from "./actions.mjs";

const activeEnvironments = new WeakSet();

function timeoutError(message) {
  return Object.assign(new Error(message), { code: "deadline" });
}

async function bounded(operation, timeoutMs, signal) {
  signal?.throwIfAborted();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(timeoutError(`operation exceeded ${Math.ceil(timeoutMs)} ms`)), Math.max(1, timeoutMs));
  const abort = () => controller.abort(signal.reason);
  signal?.addEventListener("abort", abort, { once: true });
  let rejectAborted;
  const aborted = new Promise((resolve, reject) => {
    rejectAborted = () => reject(controller.signal.reason);
    controller.signal.addEventListener("abort", rejectAborted, { once: true });
  });
  try {
    return await Promise.race([operation(controller.signal), aborted]);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
    controller.signal.removeEventListener("abort", rejectAborted);
  }
}

export function observationSummary(observation) {
  if (!observation) return null;
  return { id: observation.id, timestampMs: observation.timestampMs, text: observation.text, state: observation.state };
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

export async function runEpisode({
  environment,
  policy,
  task = null,
  seed = 0,
  instruction = "Explore the environment, infer its rules and goals from observations, and complete the task efficiently.",
  maxSteps = 200,
  maxDurationMs = 300000,
  decisionTimeoutMs = 30000,
  historyLimit = 12,
  maxNotesChars = 4000,
  onTransition = async () => {},
  signal,
}) {
  if (!Number.isInteger(maxSteps) || maxSteps < 1) throw new RangeError("maxSteps must be a positive integer");
  if (!(maxDurationMs > 0) || !(decisionTimeoutMs > 0)) throw new RangeError("episode and decision timeouts must be positive");
  if (activeEnvironments.has(environment)) throw new Error("this environment already has an episode in progress");
  activeEnvironments.add(environment);
  const started = performance.now();
  const history = [];
  let notes = "";
  let current;
  let steps = 0;
  let attempts = 0;
  let rewardTotal = 0;
  let rewardKnown = false;
  let usageCalls = 0;
  let usageKnownCalls = 0;
  const usageTotal = {};
  let reason = "max_steps";
  let failure = null;
  const remainingMs = () => maxDurationMs - (performance.now() - started);
  try {
    current = transitionView(await bounded(requestSignal => environment.reset({ task, seed, signal: requestSignal }), remainingMs(), signal));
    await policy.reset?.({ task, seed, environment: environment.id });
    await onTransition({ type: "reset", task, seed, ...current });
    while (steps < maxSteps && !current.terminated && !current.truncated) {
      if (remainingMs() <= 0) throw timeoutError("episode duration budget exhausted");
      const observation = current.observation;
      const actions = observation.actions ?? environment.actions;
      if (!Array.isArray(actions) || actions.length === 0) throw new Error("environment exposed no available actions");
      const decisionStarted = performance.now();
      const decisionMs = Math.min(decisionTimeoutMs, remainingMs());
      usageCalls += 1;
      const decision = await bounded(requestSignal => policy.decide({
        observation,
        actions,
        history: [...history],
        notes,
        feedback: history.at(-1)?.info ?? null,
        instruction,
        budget: { stepsRemaining: maxSteps - steps, remainingMs: remainingMs(), decisionMs },
        signal: requestSignal,
      }), decisionMs, signal);
      if (decision.usage) {
        usageKnownCalls += 1;
        for (const [name, value] of Object.entries(decision.usage)) {
          if (Number.isFinite(value)) usageTotal[name] = (usageTotal[name] ?? 0) + value;
        }
      }
      const action = validateAction(decision.action, actions);
      const decisionEnded = performance.now();
      signal?.throwIfAborted();
      if (remainingMs() <= 0) throw timeoutError("episode duration budget exhausted before dispatch");
      const dispatchAt = Date.now();
      attempts += 1;
      const next = transitionView(await bounded(requestSignal => environment.step(action, { observation, signal: requestSignal, reasoning: { thought: decision.thought ?? null } }), remainingMs(), signal));
      steps += 1;
      if (next.reward !== null) {
        rewardKnown = true;
        rewardTotal += next.reward;
      }
      if (typeof decision.notes === "string") notes = decision.notes.slice(0, maxNotesChars);
      const record = {
        type: "step",
        step: steps,
        action,
        observation,
        nextObservation: next.observation,
        terminated: next.terminated,
        truncated: next.truncated,
        success: next.success,
        reward: next.reward,
        info: next.info,
        model: decision.model ?? policy.id ?? null,
        route: decision.route ?? null,
        nodes: decision.nodes ?? null,
        thought: decision.thought ?? null,
        notes,
        ...(decision.claim ? { claim: decision.claim } : {}),
        ...(decision.review ? { review: decision.review } : {}),
        usage: decision.usage ?? null,
        timing: {
          decisionMs: decisionEnded - decisionStarted,
          observationAgeAtDispatchMs: Number.isFinite(observation.timestampMs) ? Math.max(0, dispatchAt - observation.timestampMs) : null,
          environmentMs: performance.now() - decisionEnded,
        },
      };
      current = next;
      history.push({ ...record, observation: observationSummary(observation), nextObservation: observationSummary(next.observation) });
      if (history.length > historyLimit) history.splice(0, history.length - historyLimit);
      await onTransition(record);
    }
    if (current.terminated) reason = current.success === true ? "success" : "terminated";
    else if (current.truncated) reason = "environment_limit";
  } catch (error) {
    reason = signal?.aborted ? "cancelled" : error?.code === "deadline" ? "deadline" : "error";
    failure = { code: error?.code ?? error?.errorType ?? "episode_error", message: error?.message ?? String(error) };
  } finally {
    try {
      await environment.stop?.();
    } catch (error) {
      failure ??= { code: "stop_error", message: error.message };
      reason = "error";
    }
    activeEnvironments.delete(environment);
  }
  const summary = {
    environment: environment.id ?? null,
    policy: policy.id ?? null,
    task,
    seed,
    reason,
    steps,
    actionAttempts: attempts,
    terminated: current?.terminated ?? false,
    truncated: (current?.truncated ?? false) || ["max_steps", "deadline", "cancelled"].includes(reason),
    success: failure ? null : current?.success ?? null,
    reward: rewardKnown ? rewardTotal : null,
    usage: usageKnownCalls === 0 ? null : usageTotal,
    usageComplete: usageCalls > 0 && usageCalls === usageKnownCalls,
    modelCalls: usageCalls,
    durationMs: performance.now() - started,
    notes,
    finalObservation: observationSummary(current?.observation),
    info: current?.info ?? {},
    error: failure,
  };
  await onTransition({ type: "episode_end", summary });
  return summary;
}
