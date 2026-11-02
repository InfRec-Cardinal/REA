import { fileURLToPath } from "node:url";

import { LineWorker } from "../workers.mjs";

const backendScript = fileURLToPath(new URL("../../scripts/gymnasium_backend.py", import.meta.url));

export function createGymnasiumEnvironment({
  env = process.env,
  python = env.GYM_PYTHON ?? "python3",
  environmentId,
  makeOptions = {},
  modules = [],
  requestTimeoutMs = 120000,
  worker = null,
} = {}) {
  const transport = worker ?? new LineWorker("gymnasium", python, [backendScript], { autoRestart: false, env });
  const options = { environmentId, makeOptions, modules };
  let configured = null;
  let current = null;
  let closed = false;

  async function request(payload, signal) {
    signal?.throwIfAborted();
    if (closed) throw new Error("Gymnasium environment is closed");
    const abort = () => transport.stop();
    signal?.addEventListener("abort", abort, { once: true });
    try {
      const response = await transport.request(payload, requestTimeoutMs).catch(error => {
        transport.stop();
        throw error;
      });
      signal?.throwIfAborted();
      if (!response.ok) throw new Error(response.error ?? "Gymnasium backend request failed");
      return response.result;
    } catch (error) {
      signal?.throwIfAborted();
      throw error;
    } finally {
      signal?.removeEventListener("abort", abort);
    }
  }

  return {
    id: "gymnasium",
    timing: "turn-based",
    async reset({ task, seed = 0, signal } = {}) {
      if (!configured) configured = request({ op: "configure", options }, signal);
      await configured;
      const transition = await request({ op: "reset", task, seed }, signal);
      current = transition.observation;
      return transition;
    },
    async observe({ signal } = {}) {
      current = await request({ op: "observe" }, signal);
      return current;
    },
    async step(action, { observation = current, signal } = {}) {
      if (!current) throw new Error("Reset an episode before stepping");
      const transition = await request({ op: "step", action, observationId: observation?.id }, signal);
      current = transition.observation;
      return transition;
    },
    async close() {
      if (closed) return;
      try {
        if (configured && transport.health().running) await request({ op: "close" });
      } finally {
        closed = true;
        transport.stop();
      }
    },
  };
}
