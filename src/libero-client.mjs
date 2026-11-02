import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { performance } from "node:perf_hooks";

export class LiberoClient {
  constructor({ python, script, env = process.env, stderr = process.stderr }) {
    this.pending = new Map();
    this.sequence = 0;
    this.child = spawn(python, [script], { env, stdio: ["pipe", "pipe", "pipe"] });
    this.child.stderr.pipe(stderr, { end: false });
    this.lines = createInterface({ input: this.child.stdout });
    this.lines.on("line", line => {
      const response = JSON.parse(line);
      const pending = this.pending.get(response.id);
      if (!pending) return;
      this.pending.delete(response.id);
      clearTimeout(pending.timer);
      if (response.ok) pending.resolve(response);
      else pending.reject(new Error(response.error ?? "LIBERO backend request failed"));
    });
    this.exit = new Promise(resolve => this.child.once("exit", (code, signal) => {
      this.fail(new Error(`LIBERO backend exited: ${code ?? signal}`));
      resolve();
    }));
    this.child.on("error", error => this.fail(error));
    // A write after the backend closed its stdin (crash, OOM kill, teardown) surfaces as an EPIPE
    // 'error' event on the stream; without a listener Node turns it into an uncaught exception.
    this.child.stdin.on("error", error => this.fail(error));
  }

  fail(error) {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }

  request(command, options = {}) {
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`LIBERO ${command} timed out`));
      }, 180000);
      this.pending.set(id, { resolve, reject, timer });
      this.child.stdin.write(`${JSON.stringify({ id, command, ...options })}\n`);
    });
  }

  async close() {
    if (this.child.exitCode !== null) return;
    await this.request("close").catch(() => {});
    this.child.stdin.end();
    await this.exit;
  }
}

export function createLiberoExecutor({ client, getState, setState, tokens }) {
  const tokenSet = new Set(tokens);
  return {
    async execute({ token, frame, execute }) {
      const started = performance.now();
      const state = getState();
      const currentId = state.frame.libero.observationId;
      const observationId = frame?.libero?.observationId;
      const reasons = [];
      if (!tokenSet.has(token)) reasons.push("invalid_manipulation_token");
      if (observationId !== currentId) reasons.push("stale_observation");
      if (state.terminated) reasons.push("episode_finished");
      const gate = { allowed: reasons.length === 0, reasons, observationId, currentId };
      const gateEnded = performance.now();
      let after = state;
      if (execute && gate.allowed) {
        after = await client.request("step", { token });
        setState(after);
      }
      return {
        gate,
        executed: execute && gate.allowed,
        preview: !execute && gate.allowed,
        motion: { token, controller: "OSC_POSE" },
        dispatch: { ok: gate.allowed, steps: after.steps },
        timing: { gate_s: (gateEnded - started) / 1000, exec_s: (performance.now() - gateEnded) / 1000 },
      };
    },
    async stop() { return { ok: true }; },
    async status() { return { ready: !getState().terminated }; },
    async evidence() { return { status: { ready: !getState().terminated }, lidar: null }; },
  };
}
