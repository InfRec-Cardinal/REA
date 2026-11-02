import assert from "node:assert/strict";
import { test } from "node:test";

import { THINKER_REVIEW_SCHEMA, createThinkerInfer, isValidThinkerReview, splitThinkerOutput } from "../src/thinker-infer.mjs";
import HOLONOMIC from "./fixtures/profiles/holonomic.mjs";
import { reviewSchema } from "../src/thinker-infer.mjs";

function sseData(payload) {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function toolCallChunks(args) {
  const text = JSON.stringify(args);
  const cut = Math.floor(text.length / 2);
  return [
    sseData({ choices: [{ delta: { tool_calls: [{ index: 0, function: { name: "report_review", arguments: text.slice(0, cut) } }] }, finish_reason: null }] }),
    sseData({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: text.slice(cut) } }] }, finish_reason: null }] }),
    sseData({ choices: [{ delta: {}, finish_reason: "tool_calls" }] }),
    "data: [DONE]\n\n",
  ];
}

function provider(args) {
  const requests = [];
  const stub = async (_url, init) => {
    requests.push(JSON.parse(init.body));
    const stream = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        for (const chunk of toolCallChunks(args)) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    });
    return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } });
  };
  return { stub, requests };
}

async function withFetch(stub, run) {
  const original = globalThis.fetch;
  globalThis.fetch = stub;
  try {
    return await run();
  } finally {
    globalThis.fetch = original;
  }
}

const JOB = {
  sessionId: "s1",
  seq: 3,
  instruction: "reach the doorway",
  token: "W",
  action: { action: "FORWARD", duration_ms: 1000 },
  beforeFrame: { imageB64: "YmVmb3Jl", mimeType: "image/jpeg" },
  afterFrame: { imageB64: "YWZ0ZXI=", mimeType: "image/jpeg" },
  feedback: { executed: true, gate: [] },
};

test("the reviewer's tool call splits into a strict correction and an optional judgement", async () => {
  const { stub, requests } = provider({
    verdict: "CORRECT",
    next_token: "Q",
    brief: "the door frame is left of centre",
    confidence: 0.8,
    outcome: "no_progress",
    lesson: "  when the frame is left of centre, turn before walking  ",
    surprise: "ignored",
  });
  const infer = createThinkerInfer({ apiKey: "k", baseUrl: "https://example.test/v1" });
  const result = await withFetch(stub, () => infer("reviewer", JOB, { deadlineMs: 2000 }));
  assert.deepEqual(result.review, { verdict: "CORRECT", next_token: "Q", brief: "the door frame is left of centre", confidence: 0.8 });
  assert.deepEqual(result.judgement, { outcome: "no_progress", lesson: "when the frame is left of centre, turn before walking" });
  assert.ok(isValidThinkerReview(result.review), "the correction keeps its exact four-key shape");

  const body = requests[0];
  assert.equal(body.tools[0].function.name, "report_review");
  assert.deepEqual(body.tools[0].function.parameters, THINKER_REVIEW_SCHEMA);
  assert.deepEqual(body.tools[0].function.parameters.required, ["verdict", "next_token", "brief", "confidence"], "outcome and lesson stay optional");
  const prompt = body.messages[1].content[0].text;
  assert.match(prompt, /outcome: progress if/u);
  assert.match(prompt, /lesson: one sentence/u);
  assert.match(prompt, /never executed/u);
});

test("a review without a judgement, or with an unknown outcome, still steers", async () => {
  const { stub } = provider({ verdict: "KEEP", next_token: "W", brief: "corridor clear", confidence: 0.6, outcome: "excellent" });
  const infer = createThinkerInfer({ apiKey: "k", baseUrl: "https://example.test/v1" });
  const result = await withFetch(stub, () => infer("reviewer", JOB, { deadlineMs: 2000 }));
  assert.equal(result.review.verdict, "KEEP");
  assert.equal(result.judgement, null, "an outcome memory would refuse is dropped, not forwarded");
});

test("a contradictory STOP is not a review, judgement or not", async () => {
  const { stub } = provider({ verdict: "STOP", next_token: "W", brief: "person ahead", confidence: 1, outcome: "unsafe", lesson: "stop" });
  const infer = createThinkerInfer({ apiKey: "k", baseUrl: "https://example.test/v1" });
  await assert.rejects(withFetch(stub, () => infer("reviewer", JOB, { deadlineMs: 2000 })), /without a valid thinker review/u);
});

test("splitThinkerOutput validates both halves", () => {
  assert.equal(splitThinkerOutput(null), null);
  assert.equal(splitThinkerOutput({ verdict: "KEEP", next_token: "W", brief: "ok" }), null, "missing confidence");
  const clean = splitThinkerOutput({ verdict: "STOP", next_token: "X", brief: "hazard", confidence: 1 });
  assert.deepEqual(clean, { review: { verdict: "STOP", next_token: "X", brief: "hazard", confidence: 1 }, judgement: null });
  const judged = splitThinkerOutput({ verdict: "KEEP", next_token: "W", brief: "ok", confidence: 0.5, outcome: "progress" });
  assert.deepEqual(judged.judgement, { outcome: "progress", lesson: null });
  const long = splitThinkerOutput({ verdict: "KEEP", next_token: "W", brief: "ok", confidence: 0.5, outcome: "progress", lesson: "x".repeat(201) });
  assert.equal(long.judgement, null, "a lesson longer than the wire limit is dropped");
  assert.equal(isValidThinkerReview({ verdict: "KEEP", next_token: "W", brief: "ok", confidence: 0.5, outcome: "progress" }), false, "the correction alone must be exactly four keys");
});

test("the review schema and validator follow the profile", () => {
  assert.deepEqual(reviewSchema(HOLONOMIC).properties.next_token.enum, ["H", "F", "L", "R"]);
  assert.equal(isValidThinkerReview({ verdict: "STOP", next_token: "H", brief: "b", confidence: 1 }, HOLONOMIC), true);
  assert.equal(isValidThinkerReview({ verdict: "STOP", next_token: "X", brief: "b", confidence: 1 }, HOLONOMIC), false);
  assert.equal(isValidThinkerReview({ verdict: "KEEP", next_token: "F", brief: "b", confidence: 1 }, HOLONOMIC), true);
  assert.equal(isValidThinkerReview({ verdict: "KEEP", next_token: "W", brief: "b", confidence: 1 }, HOLONOMIC), false);
});
