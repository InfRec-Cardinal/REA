import { fileURLToPath } from "node:url";

import { LineWorker } from "../workers.mjs";

const script = fileURLToPath(new URL("../../scripts/robot_backend.py", import.meta.url));

export function createPythonRobotDriver({ kind, python = "python3", options = {}, requestTimeoutMs = 120000, worker = null, env = process.env }) {
  // Start only on reset, so importing/configuring a hardware adapter does not connect it.
  let transport = worker;
  let configured;
  let closed = false;
  let interrupted = false;

  async function request(payload, signal) {
    signal?.throwIfAborted();
    if (closed || interrupted) throw new Error("Robot driver is closed or interrupted");
    transport ??= new LineWorker(kind, python, [script], { autoRestart: false, env });
    const abort = () => {
      interrupted = true;
      transport.stop();
    };
    signal?.addEventListener("abort", abort, { once: true });
    try {
      const response = await transport.request(payload, requestTimeoutMs);
      signal?.throwIfAborted();
      if (!response.ok) throw new Error(response.error ?? `${kind} request failed`);
      return response.result;
    } catch (error) {
      if (!transport.health().running || /timeout after/u.test(error.message)) abort();
      signal?.throwIfAborted();
      throw error;
    } finally {
      signal?.removeEventListener("abort", abort);
    }
  }

  return {
    async describe({ signal } = {}) {
      configured ??= request({ op: "configure", driver: kind, options }, signal);
      return configured;
    },
    reset: ({ task, seed, signal }) => request({ op: "reset", task, seed }, signal),
    observe: ({ signal } = {}) => request({ op: "observe" }, signal),
    step: (action, { execute, signal }) => request({ op: "step", action, execute }, signal),
    stop: () => interrupted ? { supported: false, interrupted: true } : request({ op: "stop" }),
    async close() {
      if (closed) return;
      let graceful = false;
      try {
        if (transport?.health().running && !interrupted) {
          await request({ op: "close" });
          graceful = true;
        }
      } finally {
        closed = true;
        if (!graceful) transport?.stop();
      }
    },
  };
}
