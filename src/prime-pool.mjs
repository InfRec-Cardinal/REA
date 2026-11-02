import { InferenceError } from "./provider.mjs";

function poolError(message, errorType = "agent_pool_error") {
  return new InferenceError(message, { errorType, retryable: errorType !== "config_error" });
}

/**
 * A small pool of persistent Prime Agent RPC processes.
 *
 * The old fast harness can keep two decisions in flight for its background pipeline and can race
 * primary/fallback models. A single RPC process is intentionally serial, so pooling preserves both
 * optimisations without cold-starting Prime for every frame.
 */
export function createPrimeAgentPool({ size = 2, createInfer } = {}) {
  if (!Number.isSafeInteger(size) || size < 1 || size > 4) {
    throw poolError("Prime Agent pool size must be an integer between 1 and 4", "config_error");
  }
  if (typeof createInfer !== "function") {
    throw poolError("Prime Agent pool needs an inference factory", "config_error");
  }

  const slots = Array.from({ length: size }, (_, index) => ({
    index,
    infer: createInfer(index),
    busy: false,
  }));
  const waiters = [];
  let cursor = 0;
  let started = false;
  let startPromise = null;
  let closed = false;

  function state() {
    const workers = slots.map(slot => {
      const rpc = slot.infer.state();
      return { index: slot.index, ...rpc, rpcBusy: rpc.busy, busy: slot.busy };
    });
    return {
      status: closed
        ? "closed"
        : workers.every(worker => worker.ready)
          ? "ready"
          : startPromise
            ? "starting"
            : "stopped",
      ready: !closed && workers.every(worker => worker.ready),
      running: workers.some(worker => worker.running),
      busy: workers.some(worker => worker.busy),
      poolSize: slots.length,
      busyWorkers: workers.filter(worker => worker.busy).length,
      queuedRequests: waiters.length,
      workers,
    };
  }

  async function start() {
    if (closed) throw poolError("Prime Agent pool is closed", "config_error");
    if (started && state().ready) return state();
    if (startPromise) return startPromise;
    const current = (async () => {
      try {
        await Promise.all(slots.map(slot => slot.infer.start()));
        started = true;
        return state();
      } catch (error) {
        // Each adapter retires its own failed child. Keep the adapters reusable so a transient
        // provider/process start failure can recover on the next request without a service restart.
        throw error;
      }
    })();
    startPromise = current;
    try {
      return await current;
    } finally {
      if (startPromise === current) startPromise = null;
    }
  }

  function takeIdle() {
    for (let offset = 0; offset < slots.length; offset += 1) {
      const index = (cursor + offset) % slots.length;
      const slot = slots[index];
      if (slot.busy) continue;
      slot.busy = true;
      cursor = (index + 1) % slots.length;
      return slot;
    }
    return null;
  }

  function acquire(signal) {
    if (signal?.aborted) return Promise.reject(poolError("Prime Agent pool wait was aborted", "aborted"));
    const available = takeIdle();
    if (available) return Promise.resolve(available);
    if (closed) return Promise.reject(poolError("Prime Agent pool is closed", "aborted"));
    return new Promise((resolve, reject) => {
      const waiter = { resolve, reject, signal, abort: null };
      if (signal) {
        waiter.abort = () => {
          const index = waiters.indexOf(waiter);
          if (index !== -1) waiters.splice(index, 1);
          reject(poolError("Prime Agent pool wait was aborted", "aborted"));
        };
        signal.addEventListener("abort", waiter.abort, { once: true });
      }
      waiters.push(waiter);
    });
  }

  function release(slot) {
    while (waiters.length > 0) {
      const waiter = waiters.shift();
      if (waiter.abort) waiter.signal.removeEventListener("abort", waiter.abort);
      if (waiter.signal?.aborted) continue;
      waiter.resolve(slot);
      return;
    }
    slot.busy = false;
  }

  async function infer(model, frame, options = {}) {
    await start();
    const slot = await acquire(options.signal);
    try {
      return await slot.infer(model, frame, options);
    } finally {
      release(slot);
    }
  }

  async function close() {
    if (closed) return;
    closed = true;
    const error = poolError("Prime Agent pool closed", "aborted");
    for (const waiter of waiters.splice(0)) {
      if (waiter.abort) waiter.signal?.removeEventListener("abort", waiter.abort);
      waiter.reject(error);
    }
    const owner = slots.at(-1);
    await Promise.allSettled(
      slots.slice(0, -1).map(slot => slot.infer.close({ shutdownDaemon: false })),
    );
    await owner.infer.close({ shutdownDaemon: true });
  }

  async function warmAll(model, frame, options = {}) {
    await start();
    return Promise.all(
      slots.map((_, index) => infer(model, frame, { ...options, task: `${options.task ?? options.instruction ?? "warm"} [worker ${index + 1}]` })),
    );
  }

  return Object.assign(infer, {
    engine: "prime",
    harness: "prime-agent-rpc",
    start,
    close,
    state,
    warm: async () => (await start()).status,
    warmAll,
  });
}
