// An environment driven from outside. Some benchmark platforms and robot stacks own the control loop and call a policy
// (RoboDojo and RoboTwin through XPolicyLab, challenge evaluators, real-robot deployment stacks). The bridge turns that
// around: the evaluator sends observations, REA's runtime (one policy, async systems or routed systems) runs unchanged
// against the bridge environment, and every action the runtime dispatches goes back to the evaluator.

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

/**
 * createBridge({ id, timing, actions, resumable }) -> { environment, configure, begin, step, end, finish, state }
 *
 *   configure({ name, timing, actions, resumable })   name becomes the environment id
 *
 *   environment          the REA environment the runtime runs on (reset / step wait for the evaluator)
 *   step(payload)        the evaluator delivers { observation, terminated?, truncated?, success?, reward?, info? } and
 *                        receives the next action: { action }, or { done: true, reason, summary } once the runtime ended
 *   end(result)          the evaluator ends the episode; result (e.g. { success }) becomes environment.score()
 *   begin(), finish(x)   episode bookkeeping for the server that runs the runtime
 */
export function createBridge({ id = "bridge", timing = "turn-based", actions = [], resumable = false } = {}) {
  const spec = { id, timing, actions, resumable };
  let episode = 0;
  let sequence = 0;
  let waitingForObservation = null;
  let waitingForAction = null;
  let queuedObservation = null;
  let lastObservation = null;
  let ending = false;
  let result = null;
  let finished = null;

  function transition(payload) {
    const observation = { ...payload.observation };
    observation.id ??= `${spec.id}:${episode}:${++sequence}`;
    observation.timestampMs ??= Date.now();
    observation.actions ??= spec.actions;
    lastObservation = observation;
    return {
      observation,
      terminated: payload.terminated === true,
      truncated: payload.truncated === true,
      success: typeof payload.success === "boolean" ? payload.success : null,
      reward: Number.isFinite(payload.reward) ? payload.reward : null,
      info: payload.info ?? {},
    };
  }

  function endTransition() {
    return {
      observation: lastObservation ?? { id: `${spec.id}:${episode}:end`, timestampMs: Date.now(), text: "", state: {}, images: [], actions: spec.actions },
      terminated: false, truncated: true,
      success: typeof result?.success === "boolean" ? result.success : null,
      reward: null, info: { external: "episode_end" },
    };
  }

  function awaitObservation(signal) {
    if (ending) return Promise.resolve(endTransition());
    if (queuedObservation) {
      const next = queuedObservation;
      queuedObservation = null;
      return Promise.resolve(next);
    }
    const waiter = deferred();
    waitingForObservation = waiter;
    const abort = () => {
      if (waitingForObservation === waiter) waitingForObservation = null;
      waiter.reject(signal.reason);
    };
    signal?.addEventListener("abort", abort, { once: true });
    return waiter.promise.finally(() => signal?.removeEventListener("abort", abort));
  }

  const environment = {
    get id() { return spec.id; },
    get timing() { return spec.timing; },
    get actions() { return spec.actions; },
    get resumable() { return spec.resumable; },
    async reset({ signal } = {}) { return awaitObservation(signal); },
    async step(action, { signal } = {}) {
      // An action nobody waits for (the evaluator gave up on its request) is dropped: the next action the evaluator
      // receives is always decided from the observation it sent last.
      if (waitingForAction) {
        const waiter = waitingForAction;
        waitingForAction = null;
        waiter.resolve({ action });
      }
      return awaitObservation(signal);
    },
    async stop() { return { supported: false, external: true }; },
    async score() { return result; },
    async close() {},
  };

  return {
    environment,
    state: () => ({ episode, waitingForObservation: Boolean(waitingForObservation), waitingForAction: Boolean(waitingForAction), finished }),
    configure({ name: nextName, timing: nextTiming, actions: nextActions, resumable: nextResumable } = {}) {
      if (nextActions !== undefined && !Array.isArray(nextActions)) throw new TypeError("configure.actions must be an array of action specifications");
      if (nextName !== undefined) spec.id = String(nextName);
      if (nextTiming !== undefined) spec.timing = nextTiming;
      if (nextActions !== undefined) spec.actions = nextActions;
      if (nextResumable !== undefined) spec.resumable = nextResumable === true;
      return { ...spec };
    },
    begin() {
      episode += 1;
      sequence = 0;
      waitingForObservation = null;
      waitingForAction = null;
      queuedObservation = null;
      lastObservation = null;
      ending = false;
      result = null;
      finished = null;
      return episode;
    },
    step(payload) {
      if (!payload || typeof payload.observation !== "object" || payload.observation === null) throw new TypeError("step needs an observation object");
      if (finished) return Promise.resolve({ done: true, ...finished });
      const next = transition(payload);
      const waiter = deferred();
      waitingForAction = waiter;
      if (waitingForObservation) {
        const runtime = waitingForObservation;
        waitingForObservation = null;
        runtime.resolve(next);
      } else {
        queuedObservation = next;
      }
      return waiter.promise;
    },
    end(nextResult = null) {
      result = nextResult;
      ending = true;
      if (waitingForObservation) {
        const runtime = waitingForObservation;
        waitingForObservation = null;
        runtime.resolve(endTransition());
      }
    },
    finish(outcome) {
      finished = { reason: outcome?.reason ?? "finished", summary: outcome?.summary ?? null, ...(outcome?.error ? { error: outcome.error } : {}) };
      if (waitingForAction) {
        const waiter = waitingForAction;
        waitingForAction = null;
        waiter.resolve({ done: true, ...finished });
      }
    },
  };
}
