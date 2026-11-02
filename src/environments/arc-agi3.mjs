import { fileURLToPath } from "node:url";

import { LineWorker } from "../workers.mjs";

const backendScript = fileURLToPath(new URL("../../scripts/arc_agi3_backend.py", import.meta.url));

export function createArcAgi3Environment({
  env = process.env,
  python = env.ARC_PYTHON ?? "python3",
  environmentsDir,
  operationMode = "OFFLINE",
  recordingsDir,
  saveRecording = false,
  endOnGameOver = false,
  scorecardId,
  baseUrl,
  apiKey,
  requestTimeoutMs = 120000,
  worker = null,
} = {}) {
  const mode = operationMode.toUpperCase();
  if (!["OFFLINE", "ONLINE", "COMPETITION"].includes(mode)) throw new Error("operationMode must be OFFLINE, ONLINE, or COMPETITION");
  const transport = worker ?? new LineWorker("arc-agi-3", python, [backendScript], {
    autoRestart: false,
    env: { ...env, OPERATION_MODE: mode },
  });
  const options = { environmentsDir, operationMode: mode, recordingsDir, saveRecording, endOnGameOver, scorecardId, baseUrl, apiKey };
  let configured = null;
  let current = null;
  let closed = false;
  let finalScorecard = null;

  async function request(payload, signal) {
    signal?.throwIfAborted();
    if (closed) throw new Error("ARC environment is closed");
    const abort = () => transport.stop();
    signal?.addEventListener("abort", abort, { once: true });
    try {
      const response = await transport.request(payload, requestTimeoutMs).catch(error => {
        transport.stop();
        throw error;
      });
      signal?.throwIfAborted();
      if (!response.ok) throw new Error(response.error ?? "ARC backend request failed");
      return response.result;
    } catch (error) {
      signal?.throwIfAborted();
      throw error;
    } finally {
      signal?.removeEventListener("abort", abort);
    }
  }

  return {
    id: "arc-agi-3",
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
    async step(action, { observation = current, signal, reasoning } = {}) {
      if (!current) throw new Error("Reset an episode before stepping");
      const transition = await request({ op: "step", action, observationId: observation?.id, reasoning }, signal);
      current = transition.observation;
      return transition;
    },
    async scorecard({ close = false } = {}) {
      if (closed) return finalScorecard;
      finalScorecard = await request({ op: "scorecard", close });
      return finalScorecard;
    },
    async close() {
      if (closed) return;
      try {
        if (configured && transport.health().running) {
          await request({ op: "close" });
        }
      } finally {
        closed = true;
        transport.stop();
      }
    },
  };
}
