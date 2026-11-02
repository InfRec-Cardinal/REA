import { randomUUID } from "node:crypto";
import { runAsyncEpisode } from "./async-runtime.mjs";
import { observationSummary } from "./runtime.mjs";

/** Start immediately; observe, replay, or cancel without awaiting the episode. */
export function createAsyncSession(options, { run = runAsyncEpisode } = {}) {
  const id = `run-${randomUUID()}`;
  const controller = new AbortController();
  const signal = options.signal ? AbortSignal.any([options.signal, controller.signal]) : controller.signal;
  const startedAt = Date.now();
  const events = [];
  const listeners = new Set();
  let state = "queued";
  let summary = null;
  let latestTransition = null;
  let error = null;

  function publish(event) {
    const entry = { timestampMs: Date.now(), atMs: Date.now() - startedAt, ...event, sessionId: id, sequence: events.length + 1 };
    events.push(entry);
    for (const listener of [...listeners]) listener(entry);
    return entry;
  }

  function status() {
    return { id, state, startedAt, mode: options.mode ?? "async", sequence: events.length, summary, error, latestTransition };
  }

  publish({ type: "session", status: "queued", atMs: 0 });
  const done = Promise.resolve().then(async () => {
    state = signal.aborted ? "cancelling" : "running";
    publish({ type: "session", status: state });
    try {
      summary = await run({
        ...options,
        signal,
        onEvent(event) {
          publish(event);
          return options.onEvent?.(event);
        },
        async onTransition(transition) {
          latestTransition = { ...transition };
          if (transition.observation) latestTransition.observation = observationSummary(transition.observation);
          if (transition.nextObservation) latestTransition.nextObservation = observationSummary(transition.nextObservation);
          if (transition.type !== "episode_end") {
            publish({ type: "transition", status: transition.type, step: transition.step ?? 0, transition: latestTransition });
          }
          await options.onTransition?.(transition);
        },
      });
      state = summary.reason === "cancelled" ? "cancelled" : summary.error ? "failed" : "completed";
      publish({ type: "session", status: state, summary });
      return summary;
    } catch (cause) {
      error = { message: cause.message, code: cause.code ?? "session_error" };
      state = signal.aborted ? "cancelled" : "failed";
      publish({ type: "session", status: state, error });
      throw cause;
    }
  });
  // Keep an unattended submitted session observable even if its runner rejects.
  done.catch(() => {});

  return {
    id,
    done,
    status,
    history({ after = 0 } = {}) { return events.filter(event => event.sequence > after); },
    subscribe(listener, { after = 0, replay = true } = {}) {
      if (replay) for (const event of events) if (event.sequence > after) listener(event);
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    cancel(reason = "session cancelled") {
      if (!["queued", "running", "cancelling"].includes(state)) return false;
      if (state !== "cancelling") {
        state = "cancelling";
        publish({ type: "session", status: state });
        controller.abort(new Error(reason));
      }
      return true;
    },
  };
}
