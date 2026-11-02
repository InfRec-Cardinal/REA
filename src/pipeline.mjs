import { performance } from "node:perf_hooks";

/**
 * Continuous capture -> decide loop.
 *
 * This is what takes the model off the request path. Workers keep deciding in the background; an
 * incoming /v1/act grabs the freshest decision and goes straight to the gate, so the request costs
 * gate + RPC (tens of milliseconds) instead of a full provider round-trip.
 *
 * The decision it executes may be up to `freshMaxS` old. That is a steering-quality trade, not a
 * safety one: the gate re-reads state, depth and lidar at execution time, so a stale decision can
 * only ever be refused, never waved through on stale evidence.
 */
export class DecisionPipeline {
  constructor({ decide, now = () => performance.now(), logger = null }) {
    this.decideFn = decide;
    this.now = now;
    this.logger = logger;
    this.running = false;
    this.body = null;
    this.latest = null;
    this.inflight = null;
    this.loops = 0;
    this.errors = 0;
    this.workers = 0;
    this.generation = 0;
    this.consecutiveFailures = 0;
    this.backoffMaxMs = 2000;
  }

  #loop(generation, staggerMs) {
    const run = async () => {
      await new Promise((resolve) => setTimeout(resolve, staggerMs));
      while (this.running && this.generation === generation) {
        const started = this.now();
        const promise = this.decideFn(this.body)
          .then((decided) => {
            const record = {
              ...decided,
              at: this.now(),
              wallAt: Date.now(),
              decideS: Number(((this.now() - started) / 1000).toFixed(4)),
            };
            // keep the newest successful decision; a failed one must not mask a good older one
            if (record.result?.ok && (!this.latest || record.at > this.latest.at)) this.latest = record;
            return record;
          })
          .catch((error) => {
            this.errors += 1;
            this.logger?.warn?.({ event: "pipeline_decide_failed", error: error?.message ?? String(error) });
            return { result: { ok: false, error: error?.message ?? String(error) }, at: this.now() };
          });
        this.inflight = promise;
        const decided = await promise;
        this.loops += 1;
        // A decision that fails fast (no camera, no key) must not turn the loop into a hot spin that
        // starves the event loop and makes /v1/stop unreachable: back off, capped, until one succeeds.
        if (decided?.result?.ok) {
          this.consecutiveFailures = 0;
          // Yield to the event loop between decisions: an instant decide() (a mock, a cached answer)
          // must not starve timers and HTTP handlers, and a real one costs ~1 s anyway.
          await new Promise((resolve) => setImmediate(resolve));
        } else {
          this.consecutiveFailures += 1;
          const backoff = Math.min(this.backoffMaxMs, 100 * 2 ** Math.min(this.consecutiveFailures, 8));
          await new Promise((resolve) => setTimeout(resolve, backoff));
        }
      }
    };
    void run();
  }

  set(body = {}) {
    const on = body.on !== false;
    const workers = Math.max(1, Math.min(4, Number(body.concurrency ?? this.workers ?? 1)));
    const merged = {
      instruction: body.instruction ?? this.body?.instruction,
      cfg: body.cfg ?? this.body?.cfg ?? {},
    };

    if (on && (!this.running || workers !== this.workers)) {
      this.running = true;
      this.generation += 1;
      this.workers = workers;
      this.body = merged;
      // stagger the workers so their provider calls interleave instead of arriving in a burst
      for (let index = 0; index < workers; index += 1) this.#loop(this.generation, index * (650 / workers));
    } else if (on) {
      this.body = merged;
    } else {
      this.running = false;
      this.workers = 0;
    }
    return this.status();
  }

  status() {
    return {
      running: this.running,
      workers: this.workers,
      loops: this.loops,
      errors: this.errors,
      consecutive_failures: this.consecutiveFailures,
      latest_age_s: this.latest ? Number(((this.now() - this.latest.at) / 1000).toFixed(3)) : null,
      latest_token: this.latest?.result?.token ?? null,
    };
  }

  /** Freshest usable decision, or the one currently in flight, or null. */
  take(freshMaxS) {
    if (this.latest?.result?.ok && (this.now() - this.latest.at) / 1000 <= freshMaxS) return this.latest;
    return null;
  }

  stop() {
    this.running = false;
    this.workers = 0;
  }
}
