import { G1_PROFILE } from "./profiles/g1.mjs";
import { isValidThinkerReview } from "./thinker-infer.mjs";
import { VERSION } from "./version.mjs";

export class ThinkerJobError extends Error {
  constructor(message) {
    super(message);
    this.name = "ThinkerJobError";
  }
}

function validFrame(frame) {
  return Boolean(
    frame &&
      typeof frame === "object" &&
      !Array.isArray(frame) &&
      typeof frame.imageB64 === "string" &&
      frame.imageB64.length > 0,
  );
}

export function thinkerJobError(job, profile = G1_PROFILE) {
  if (job === null || typeof job !== "object" || Array.isArray(job)) return "job must be an object";
  if (typeof job.sessionId !== "string" || job.sessionId.trim().length === 0) {
    return "sessionId must be a non-empty string";
  }
  if (!Number.isSafeInteger(job.seq) || job.seq < 0) return "seq must be a non-negative integer";
  if (typeof job.instruction !== "string" || job.instruction.trim().length === 0) {
    return "instruction must be a non-empty string";
  }
  if (!profile.tokenSet.has(job.token)) return "token is not in the thinker token set";
  if (!("action" in job)) return "action is required";
  if (!validFrame(job.beforeFrame)) return "beforeFrame.imageB64 is required";
  if (job.afterFrame !== undefined && job.afterFrame !== null && !validFrame(job.afterFrame)) {
    return "afterFrame.imageB64 must be a non-empty string";
  }
  return null;
}

export function isValidThinkerJob(job, profile = G1_PROFILE) {
  return thinkerJobError(job, profile) === null;
}

function correctionFor(job, review) {
  return {
    sessionId: job.sessionId,
    basedOnSeq: job.seq,
    verdict: review.verdict,
    next_token: review.next_token,
    brief: review.brief,
    confidence: review.confidence,
  };
}

function log(logger, level, record) {
  const fn = logger?.[level];
  if (typeof fn === "function") fn.call(logger, record);
}

export class ThinkerRuntime {
  constructor({
    infer,
    model,
    callback,
    onCorrection,
    judge = null,
    timeoutMs = 15_000,
    providerConfigured = true,
    logger = console,
    profile = G1_PROFILE,
  }) {
    if (typeof infer !== "function") throw new TypeError("infer must be a function");
    const deliver = callback ?? onCorrection;
    if (typeof deliver !== "function") throw new TypeError("callback must be a function");
    if (judge !== null && typeof judge !== "function") throw new TypeError("judge must be a function when given");
    if (typeof model !== "string" || model.length === 0) throw new TypeError("model is required");
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) {
      throw new TypeError("timeoutMs must be a positive integer");
    }

    this.infer = infer;
    this.model = model;
    this.callback = deliver;
    this.judge = judge;
    this.timeoutMs = timeoutMs;
    this.providerConfigured = Boolean(providerConfigured);
    this.logger = logger;
    this.profile = profile;
    this.closed = false;
    this.generation = 0;
    this.active = null;
    this.pending = null;
    this.draining = false;
    this.idleWaiters = [];
    this.stats = {
      accepted: 0,
      started: 0,
      superseded: 0,
      completed: 0,
      callbacks: 0,
      judgements: 0,
      judgement_failures: 0,
      failures: 0,
      timeouts: 0,
    };
    this.lastCorrection = null;
    this.lastJudgement = null;
    this.lastError = null;
  }

  submit(job) {
    if (this.closed) throw new ThinkerJobError("thinker runtime is closed");
    const error = thinkerJobError(job, this.profile);
    if (error) throw new ThinkerJobError(error);

    const generation = ++this.generation;
    const replaced = Boolean(this.active || this.pending);
    if (this.pending) this.stats.superseded += 1;
    this.pending = { generation, job };
    this.stats.accepted += 1;

    if (this.active && !this.active.abortRequested) {
      this.active.abortRequested = true;
      this.stats.superseded += 1;
      this.active.controller.abort(new Error("superseded by a newer thinker job"));
    }

    queueMicrotask(() => this.#drain());
    return {
      accepted: true,
      sessionId: job.sessionId,
      seq: job.seq,
      generation,
      replaced,
    };
  }

  health() {
    return {
      ok: true,
      ready: !this.closed && this.providerConfigured,
      service: "rea-thinker",
      version: VERSION,
      loopback_only: true,
      actuator_attached: false,
      sdk_attached: false,
      concurrency: 1,
      latest_wins: true,
      judge_attached: this.judge !== null,
      model: this.model,
      busy: this.active !== null,
      pending: this.pending !== null,
      closed: this.closed,
      stats: { ...this.stats },
      last_correction: this.lastCorrection,
      last_judgement: this.lastJudgement,
      last_error: this.lastError,
    };
  }

  status() {
    return this.health();
  }

  waitForIdle() {
    if (!this.draining && this.active === null && this.pending === null) return Promise.resolve();
    return new Promise(resolve => this.idleWaiters.push(resolve));
  }

  async close() {
    if (this.closed) return this.waitForIdle();
    this.closed = true;
    if (this.pending) this.stats.superseded += 1;
    this.pending = null;
    if (this.active && !this.active.abortRequested) {
      this.active.abortRequested = true;
      this.active.controller.abort(new Error("thinker runtime is closing"));
    }
    await this.waitForIdle();
  }

  /** Post the outcome judgement to the brain. Never throws: a lost lesson is a lost lesson, not a control error. */
  async #deliverJudgement(selected, review, judgement, signal) {
    const payload = {
      sessionId: selected.job.sessionId,
      basedOnSeq: selected.job.seq,
      token: selected.job.token,
      verdict: review.verdict,
      next_token: review.next_token,
      confidence: review.confidence,
      outcome: judgement.outcome,
      lesson: judgement.lesson ?? null,
    };
    try {
      await this.judge(payload, { signal, job: selected.job, generation: selected.generation });
      this.stats.judgements += 1;
      this.lastJudgement = payload;
      log(this.logger, "info", {
        event: "thinker_judgement",
        session_id: payload.sessionId,
        based_on_seq: payload.basedOnSeq,
        outcome: payload.outcome,
        lesson: payload.lesson,
      });
    } catch (error) {
      if (signal.aborted) return;
      this.stats.judgement_failures += 1;
      log(this.logger, "warn", {
        event: "thinker_judgement_failed",
        session_id: payload.sessionId,
        seq: payload.basedOnSeq,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  #resolveIdle() {
    if (this.draining || this.active !== null || this.pending !== null) return;
    const waiters = this.idleWaiters.splice(0);
    for (const resolve of waiters) resolve();
  }

  async #drain() {
    if (this.draining) return;
    this.draining = true;
    try {
      while (!this.closed && this.pending) {
        const selected = this.pending;
        this.pending = null;
        const controller = new AbortController();
        const active = {
          ...selected,
          controller,
          abortRequested: false,
          timedOut: false,
        };
        this.active = active;
        this.stats.started += 1;
        const timer = setTimeout(() => {
          active.timedOut = true;
          active.abortRequested = true;
          controller.abort(new Error("thinker inference deadline exceeded"));
        }, this.timeoutMs);

        try {
          const result = await this.infer(this.model, selected.job, {
            signal: controller.signal,
            deadlineMs: this.timeoutMs,
          });
          if (
            controller.signal.aborted ||
            selected.generation !== this.generation ||
            this.closed
          ) {
            continue;
          }
          const review = result?.review ?? result;
          const judgement = result?.judgement ?? null;
          if (!isValidThinkerReview(review, this.profile)) {
            throw new Error("thinker infer returned an invalid review");
          }
          if (review.verdict === "KEEP" && review.next_token !== selected.job.token) {
            throw new Error("KEEP review must preserve the selected token");
          }
          const correction = correctionFor(selected.job, review);
          // Steering first: the correction is time-critical. Its failure (a stale 409, a network
          // error) must not cost the judgement, which is never on time anyway and exists to teach.
          let callbackError = null;
          try {
            await this.callback(correction, {
              signal: controller.signal,
              job: selected.job,
              generation: selected.generation,
            });
          } catch (error) {
            callbackError = error;
          }
          if (
            controller.signal.aborted ||
            selected.generation !== this.generation ||
            this.closed
          ) {
            continue;
          }
          if (judgement && this.judge) {
            await this.#deliverJudgement(selected, review, judgement, controller.signal);
          }
          if (callbackError) throw callbackError;
          this.stats.completed += 1;
          this.stats.callbacks += 1;
          this.lastCorrection = correction;
          this.lastError = null;
          log(this.logger, "info", {
            event: "thinker_correction",
            session_id: correction.sessionId,
            based_on_seq: correction.basedOnSeq,
            verdict: correction.verdict,
            next_token: correction.next_token,
            outcome: judgement?.outcome ?? null,
          });
        } catch (error) {
          if (controller.signal.aborted) {
            if (active.timedOut) {
              this.stats.timeouts += 1;
              this.lastError = "thinker inference deadline exceeded";
              log(this.logger, "warn", {
                event: "thinker_timeout",
                session_id: selected.job.sessionId,
                seq: selected.job.seq,
              });
            }
          } else {
            this.stats.failures += 1;
            this.lastError = error instanceof Error ? error.message : String(error);
            log(this.logger, "warn", {
              event: "thinker_failure",
              session_id: selected.job.sessionId,
              seq: selected.job.seq,
              error: this.lastError,
            });
          }
        } finally {
          clearTimeout(timer);
          if (this.active === active) this.active = null;
        }
      }
    } finally {
      this.draining = false;
      this.#resolveIdle();
      if (!this.closed && this.pending) queueMicrotask(() => this.#drain());
    }
  }
}

export function createThinkerRuntime(options) {
  return new ThinkerRuntime(options);
}
