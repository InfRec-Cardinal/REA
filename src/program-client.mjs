// JSON-lines RPC client for program backends (scripts/aspire_backend.py, scripts/behavior_backend.py, or any
// process that speaks the protocol in docs/extending/program-backends.md).
//
// Same wire protocol and pending-request mechanics as LiberoClient, with two differences program backends
// need: extra command-line arguments for the backend process, and a timeout per command (a frozen program
// legitimately runs for minutes; an observe should not).

import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

import { LiberoClient } from "./libero-client.mjs";

export const DEFAULT_TIMEOUTS_MS = Object.freeze({
  catalog: 180000, reset: 300000, run_program: null, run_agent: null, observe: 60000, score: 60000, close: 30000,
});
// Fallback only: callers normally pass the remaining per-episode wall budget as wall_timeout_s.
export const DEFAULT_WALL_TIMEOUT_S = 900;
const FALLBACK_TIMEOUT_MS = 180000;
const RUN_PROGRAM_GRACE_MS = 60000;

/** Milliseconds a command may take before the client gives up on it; null disables the timer. */
export function timeoutFor(command, options = {}, timeouts = DEFAULT_TIMEOUTS_MS) {
  if (["run_program", "run_agent"].includes(command) && timeouts[command] == null) {
    return (Number(options.wall_timeout_s ?? DEFAULT_WALL_TIMEOUT_S) * 1000) + RUN_PROGRAM_GRACE_MS;
  }
  return command in timeouts ? timeouts[command] : FALLBACK_TIMEOUT_MS;
}

export class ProgramClient extends LiberoClient {
  constructor({ python, script, args = [], env = process.env, stderr = process.stderr, timeouts = DEFAULT_TIMEOUTS_MS }) {
    // LiberoClient's constructor spawns `python [script]` with no room for arguments, so the process
    // is wired up here instead. Returning an object from a derived constructor skips super() while the
    // instance keeps LiberoClient's prototype (fail, close) and passes instanceof checks.
    const client = Object.create(new.target.prototype);
    client.pending = new Map();
    client.sequence = 0;
    client.timeouts = { ...DEFAULT_TIMEOUTS_MS, ...timeouts };
    client.child = spawn(python, [script, ...args], { env, stdio: ["pipe", "pipe", "pipe"] });
    client.child.stderr.pipe(stderr, { end: false });
    client.lines = createInterface({ input: client.child.stdout });
    client.lines.on("line", line => {
      let response;
      try {
        response = JSON.parse(line);
      } catch {
        stderr.write(`program-client: ignoring non-JSON protocol line: ${line.slice(0, 200)}\n`);
        return;
      }
      const pending = client.pending.get(response.id);
      if (!pending) return;
      client.pending.delete(response.id);
      clearTimeout(pending.timer);
      if (response.ok) pending.resolve(response);
      else pending.reject(new Error(response.error ?? `program backend ${pending.command} failed`));
    });
    client.exit = new Promise(resolve => client.child.once("exit", (code, signal) => {
      client.fail(new Error(`program backend exited: ${code ?? signal}`));
      resolve();
    }));
    client.child.on("error", error => client.fail(error));
    // EPIPE on a write after the backend closed stdin must reject the pending requests, not crash
    // the runner (an unhandled stream 'error' event is an uncaught exception).
    client.child.stdin.on("error", error => client.fail(error));
    return client;
  }

  request(command, options = {}, { timeoutMs } = {}) {
    const id = ++this.sequence;
    const limit = timeoutMs === undefined ? timeoutFor(command, options, this.timeouts) : timeoutMs;
    return new Promise((resolve, reject) => {
      const timer = limit == null || !(limit > 0) ? null : setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`program backend ${command} timed out after ${limit} ms`));
      }, limit);
      this.pending.set(id, { resolve, reject, timer, command });
      this.child.stdin.write(`${JSON.stringify({ id, command, ...options })}\n`, error => {
        if (!error) return;
        clearTimeout(timer);
        this.pending.delete(id);
        reject(error);
      });
    });
  }
}
