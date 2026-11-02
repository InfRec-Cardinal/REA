import { spawn } from "node:child_process";
import readline from "node:readline";

/**
 * JSON-lines request/response over a long-lived child process.
 *
 * This is the single biggest latency win: booting a python capture process costs ~1.5 s and DDS init
 * another ~1.5 s, on *every* decision. Keeping both alive turns a capture into a socket round-trip and
 * a motion command into one RPC.
 *
 * One request is in flight at a time per worker (the python side is a simple read-line loop), so
 * calls are serialised through an internal queue.
 */
/**
 * The environment for a process that runs generated code: enough to start Python and find its packages, and none of
 * the parent's credentials. This is not a sandbox; run code policies inside a container or VM when the model is untrusted.
 */
export function codeWorkerEnv(env = process.env) {
  const keep = ["PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "SYSTEMROOT"];
  return Object.fromEntries(keep.filter(name => env[name] !== undefined).map(name => [name, env[name]]));
}

export class LineWorker {
  constructor(name, cmd, args, { env = process.env, onEvent = () => {}, autoRestart = true, readyTimeoutMs = 50 } = {}) {
    this.name = name;
    this.cmd = cmd;
    this.args = args;
    this.env = env;
    this.onEvent = onEvent;
    this.autoRestart = autoRestart;
    this.readyTimeoutMs = readyTimeoutMs;
    this.pending = new Map();
    this.queue = Promise.resolve();
    this.seq = 0;
    this.restarts = 0;
    this.stopped = false;
    this.lastEvent = null;
    this.ready = this.#start();
  }

  #start() {
    this.exited = false;
    // `ready` settles on the worker's ready line, on the grace timeout, or when the process fails to
    // start — a boot that waits on a worker whose interpreter is missing must not hang forever.
    let settled = false;
    let resolveReady;
    const ready = new Promise((resolve) => { resolveReady = resolve; });
    const settle = (value) => {
      if (settled) return;
      settled = true;
      resolveReady(value);
    };
    this.proc = spawn(this.cmd, this.args, { env: this.env, stdio: ["pipe", "pipe", "pipe"] });
    this.proc.stderr.on("data", (chunk) => process.stderr.write(`[${this.name}] ${chunk}`));
    const failed = (reason) => {
      for (const [, entry] of this.pending) entry.reject(new Error(`${this.name} ${reason}`));
      this.pending.clear();
      this.exited = true;
      this.lastError = reason;
      settle(null);
      if (this.autoRestart && !this.stopped) {
        this.restarts += 1;
        // back off a little so a worker that dies on startup cannot spin
        setTimeout(() => {
          if (!this.stopped) this.ready = this.#start();
        }, Math.min(2000, 250 * this.restarts)).unref?.();
      }
    };
    // An unhandled 'error' on a child process (missing interpreter, EPIPE) is fatal to the whole
    // service by default; here it is a failed worker like any other exit.
    this.proc.on("error", (error) => failed(`failed to start or died: ${error.message}`));
    this.proc.stdin.on("error", () => {});
    this.proc.on("exit", (code) => failed(`exited (${code})`));

    const rl = readline.createInterface({ input: this.proc.stdout });
    rl.on("line", (line) => {
      let message;
      try {
        message = JSON.parse(line);
      } catch {
        return;
      }
      if (message.event === "ready") {
        this.lastEvent = message;
        this.onEvent(message);
        settle(message);
        return;
      }
      const entry = this.pending.get(message.id);
      if (entry) {
        this.pending.delete(message.id);
        entry.resolve(message);
      }
    });
    // workers that publish no ready event are considered ready immediately
    setTimeout(() => settle(null), this.readyTimeoutMs).unref?.();
    return ready;
  }

  request(payload, timeoutMs = 10000) {
    const id = `${++this.seq}`;
    const run = () =>
      new Promise((resolve, reject) => {
        if (this.stopped) return reject(new Error(`${this.name} is stopped`));
        if (this.exited) return reject(new Error(`${this.name} is not running`));
        const timer = setTimeout(() => {
          this.pending.delete(id);
          reject(new Error(`${this.name} timeout after ${timeoutMs} ms`));
        }, timeoutMs);
        this.pending.set(id, {
          resolve: (message) => {
            clearTimeout(timer);
            resolve(message);
          },
          reject: (error) => {
            clearTimeout(timer);
            reject(error);
          },
        });
        try {
          this.proc.stdin.write(`${JSON.stringify({ ...payload, id })}\n`);
        } catch (error) {
          this.pending.delete(id);
          clearTimeout(timer);
          reject(new Error(`${this.name} write failed: ${error.message}`));
        }
      });
    const next = this.queue.then(run, run);
    this.queue = next.catch(() => {});
    return next;
  }

  health() {
    return { name: this.name, running: !this.exited && !this.stopped, restarts: this.restarts, pid: this.proc?.pid ?? null, last_error: this.lastError ?? null };
  }

  stop() {
    this.stopped = true;
    try {
      this.proc.kill();
    } catch {
      /* already gone */
    }
  }
}
