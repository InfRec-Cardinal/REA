import assert from "node:assert/strict";
import test from "node:test";

import { createThinkerRuntime } from "../src/thinker-runtime.mjs";

const silentLogger = { info() {}, warn() {}, error() {} };

function frame(seq) {
  return { imageB64: Buffer.from(`frame-${seq}`).toString("base64"), mimeType: "image/jpeg" };
}

function job(seq) {
  return {
    sessionId: "session-1",
    seq,
    instruction: "move toward the clear aisle",
    token: "W",
    action: { action: "FORWARD", duration_ms: 500 },
    beforeFrame: frame(seq),
  };
}

function abortableDelay(signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, 10_000);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(signal.reason ?? new Error("aborted"));
      },
      { once: true },
    );
  });
}

test("thinker runtime is single-concurrency latest-wins and cancels old inference", async () => {
  const started = [];
  const callbacks = [];
  let active = 0;
  let maxActive = 0;
  let firstStarted;
  const firstReady = new Promise(resolve => {
    firstStarted = resolve;
  });

  const runtime = createThinkerRuntime({
    model: "deepseek-v4-flash-vision-exp",
    timeoutMs: 2000,
    logger: silentLogger,
    infer: async (_model, selected, { signal }) => {
      started.push(selected.seq);
      active += 1;
      maxActive = Math.max(maxActive, active);
      try {
        if (selected.seq === 1) {
          firstStarted();
          await abortableDelay(signal);
        }
        return {
          review: {
            verdict: selected.seq === 1 ? "KEEP" : "CORRECT",
            next_token: selected.seq === 1 ? "W" : "D",
            brief: "The newest frame supports the revised direction.",
            confidence: 0.9,
          },
        };
      } finally {
        active -= 1;
      }
    },
    callback: async correction => callbacks.push(correction),
  });

  runtime.submit(job(1));
  await firstReady;
  const accepted = runtime.submit(job(2));
  assert.equal(accepted.replaced, true);
  await runtime.waitForIdle();

  assert.deepEqual(started, [1, 2]);
  assert.equal(maxActive, 1);
  assert.deepEqual(callbacks, [
    {
      sessionId: "session-1",
      basedOnSeq: 2,
      verdict: "CORRECT",
      next_token: "D",
      brief: "The newest frame supports the revised direction.",
      confidence: 0.9,
    },
  ]);
  assert.equal(runtime.health().stats.superseded, 1);
  assert.equal(runtime.health().stats.callbacks, 1);
  assert.equal(runtime.health().sdk_attached, false);
  await runtime.close();
});

test("pending thinker jobs collapse to the latest generation before inference starts", async () => {
  const inferred = [];
  const callbacks = [];
  const runtime = createThinkerRuntime({
    model: "reviewer",
    timeoutMs: 1000,
    logger: silentLogger,
    infer: async (_model, selected) => {
      inferred.push(selected.seq);
      return {
        verdict: "STOP",
        next_token: "X",
        brief: "The observation is ambiguous.",
        confidence: 0.7,
      };
    },
    callback: async correction => callbacks.push(correction),
  });

  runtime.submit(job(1));
  runtime.submit(job(2));
  runtime.submit(job(3));
  await runtime.waitForIdle();

  assert.deepEqual(inferred, [3]);
  assert.equal(callbacks[0].basedOnSeq, 3);
  assert.equal(callbacks[0].next_token, "X");
  assert.equal(runtime.health().stats.superseded, 2);
  await runtime.close();
});

test("thinker runtime rejects an invalid job without starting inference", () => {
  let calls = 0;
  const runtime = createThinkerRuntime({
    model: "reviewer",
    infer: async () => {
      calls += 1;
    },
    callback: async () => {},
    logger: silentLogger,
  });
  assert.throws(() => runtime.submit({ ...job(1), token: "FLY" }), /token set/u);
  assert.equal(calls, 0);
});

test("KEEP cannot silently change the selected token", async () => {
  const callbacks = [];
  const runtime = createThinkerRuntime({
    model: "reviewer",
    infer: async () => ({
      verdict: "KEEP",
      next_token: "A",
      brief: "The original action remains suitable.",
      confidence: 0.8,
    }),
    callback: async correction => callbacks.push(correction),
    logger: silentLogger,
  });

  runtime.submit(job(1));
  await runtime.waitForIdle();

  assert.deepEqual(callbacks, []);
  assert.equal(runtime.health().stats.failures, 1);
  assert.match(runtime.health().last_error, /KEEP review must preserve/u);
  await runtime.close();
});

test("the judgement is delivered after the correction, and still when the correction is stale", async () => {
  const callbacks = [];
  const judgements = [];
  const order = [];
  const runtime = createThinkerRuntime({
    model: "reviewer",
    logger: silentLogger,
    infer: async (_model, selected) => ({
      review: { verdict: "CORRECT", next_token: "Q", brief: "door on the left", confidence: 0.9 },
      judgement: { outcome: "no_progress", lesson: "turn before walking when the frame is left of centre" },
    }),
    callback: async correction => {
      order.push("correction");
      callbacks.push(correction);
      if (correction.basedOnSeq === 2) throw new Error("thinker callback returned 409: stale_correction");
    },
    judge: async judgement => {
      order.push("judgement");
      judgements.push(judgement);
    },
  });

  runtime.submit(job(1));
  await runtime.waitForIdle();
  assert.deepEqual(order, ["correction", "judgement"]);
  assert.deepEqual(judgements[0], {
    sessionId: "session-1",
    basedOnSeq: 1,
    token: "W",
    verdict: "CORRECT",
    next_token: "Q",
    confidence: 0.9,
    outcome: "no_progress",
    lesson: "turn before walking when the frame is left of centre",
  });
  assert.equal(runtime.health().stats.judgements, 1);
  assert.equal(runtime.health().judge_attached, true);
  assert.deepEqual(runtime.health().last_judgement, judgements[0]);

  runtime.submit(job(2));
  await runtime.waitForIdle();
  assert.equal(judgements.length, 2, "a stale correction is still a judgement of the step it reviewed");
  assert.equal(judgements[1].basedOnSeq, 2);
  assert.equal(runtime.health().stats.failures, 1, "the stale correction is still accounted as a callback failure");
  assert.equal(runtime.health().stats.callbacks, 1);
  await runtime.close();
});

test("a failing judge channel never becomes a control error, and without a judge nothing is posted", async () => {
  const runtime = createThinkerRuntime({
    model: "reviewer",
    logger: silentLogger,
    infer: async () => ({ review: { verdict: "KEEP", next_token: "W", brief: "fine", confidence: 0.5 }, judgement: { outcome: "progress", lesson: null } }),
    callback: async () => {},
    judge: async () => { throw new Error("brain unreachable"); },
  });
  runtime.submit(job(1));
  await runtime.waitForIdle();
  assert.equal(runtime.health().stats.callbacks, 1);
  assert.equal(runtime.health().stats.failures, 0);
  assert.equal(runtime.health().stats.judgement_failures, 1);
  await runtime.close();

  const plain = createThinkerRuntime({
    model: "reviewer",
    logger: silentLogger,
    infer: async () => ({ review: { verdict: "KEEP", next_token: "W", brief: "fine", confidence: 0.5 }, judgement: { outcome: "progress", lesson: null } }),
    callback: async () => {},
  });
  plain.submit(job(1));
  await plain.waitForIdle();
  assert.equal(plain.health().judge_attached, false);
  assert.equal(plain.health().stats.judgements, 0);
  assert.throws(() => createThinkerRuntime({ model: "m", infer: async () => {}, callback: async () => {}, judge: "nope" }), /judge must be a function/u);
  await plain.close();
});
