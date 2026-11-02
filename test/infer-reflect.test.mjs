import assert from "node:assert/strict";
import { test } from "node:test";

import { REFLECTION_SCHEMA, buildReflectPrompt, createReflectInfer, isValidReflection } from "../src/infer-reflect.mjs";

function sseData(payload) {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function provider(args) {
  const requests = [];
  const stub = async (_url, init) => {
    requests.push(JSON.parse(init.body));
    const text = JSON.stringify(args);
    const cut = Math.floor(text.length / 2);
    const chunks = [
      sseData({ choices: [{ delta: { tool_calls: [{ index: 0, function: { name: "report_reflection", arguments: text.slice(0, cut) } }] }, finish_reason: null }] }),
      sseData({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: text.slice(cut) } }] }, finish_reason: null }] }),
      sseData({ choices: [{ delta: {}, finish_reason: "tool_calls" }] }),
      "data: [DONE]\n\n",
    ];
    const stream = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
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

const INPUT = {
  instruction: "reach the doorway",
  situation: { center: "near", left: "far", right: "near", lidar: "clear" },
  trigger: "judge:refused,refusal_streak",
  recent: "#3 W token vetoed(depth:depth_center_near) judge:-0.6 | #4 W token vetoed(depth:depth_center_near) judge:-0.6",
  lessons: "[-] centre near: W was vetoed (depth_center_near); prefer Q",
  token: "W",
};

test("a reflection is one text-only forced tool call that returns a lesson, an alternative and a confidence", async () => {
  const { stub, requests } = provider({ lesson: "  Two vetoes in a row mean the doorway is not ahead; turn toward the open left before walking.  ", alternative: "q", confidence: 0.8, applies_when: "centre near, left far" });
  const infer = createReflectInfer({ apiKey: "k", baseUrl: "https://example.test/v1", maxOutputTokens: 120 });
  const result = await withFetch(stub, () => infer("deepseek-v4-flash-vision-exp", INPUT, { deadlineMs: 3000 }));
  assert.equal(result.lesson, "Two vetoes in a row mean the doorway is not ahead; turn toward the open left before walking.");
  assert.equal(result.alternative, "Q", "the alternative is normalised into the token set");
  assert.equal(result.confidence, 0.8);
  assert.equal(result.appliesWhen, "centre near, left far");
  assert.equal(infer.engine, "reflect");

  const body = requests[0];
  assert.equal(body.max_tokens, 120);
  assert.deepEqual(body.tool_choice, { type: "function", function: { name: "report_reflection" } });
  assert.deepEqual(body.tools[0].function.parameters, REFLECTION_SCHEMA);
  assert.deepEqual(body.thinking, { type: "disabled" });
  assert.equal(body.messages[1].content.includes("image_url"), false, "text only: no frame is sent");
  const prompt = body.messages[1].content;
  assert.ok(prompt.includes("Situation at the last step: centre near, left far, right near, lidar clear"));
  assert.ok(prompt.includes("Why this reflection was triggered: judge:refused,refusal_streak"));
  assert.ok(prompt.includes("#4 W token vetoed"));
  assert.ok(prompt.includes("[-] centre near"));
  assert.ok(prompt.includes("Do not repeat a lesson already known"));
});

test("an invalid reflection is an error, not a lesson", async () => {
  const infer = createReflectInfer({ apiKey: "k", baseUrl: "https://example.test/v1" });
  const { stub } = provider({ lesson: "", confidence: 0.5 });
  await assert.rejects(withFetch(stub, () => infer("m", INPUT, { deadlineMs: 3000 })), /without a valid reflection/u);
  const bad = provider({ lesson: "fine", confidence: 0.5, alternative: "FLY" });
  await assert.rejects(withFetch(bad.stub, () => infer("m", INPUT, { deadlineMs: 3000 })), /without a valid reflection/u);
  await assert.rejects(createReflectInfer({ apiKey: "", baseUrl: "https://example.test/v1" })("m", INPUT), /REA_API_KEY/u);
});

test("reflection validation and the prompt builder", () => {
  assert.equal(isValidReflection({ lesson: "turn first", confidence: 0.5 }), true);
  assert.equal(isValidReflection({ lesson: "turn first", confidence: 0.5, alternative: "Q", applies_when: "near" }), true);
  assert.equal(isValidReflection({ lesson: "turn first" }), false);
  assert.equal(isValidReflection({ lesson: "x".repeat(201), confidence: 0.5 }), false);
  assert.equal(isValidReflection({ lesson: "ok", confidence: 1.5 }), false);
  assert.equal(isValidReflection({ lesson: "ok", confidence: 0.5, applies_when: "x".repeat(81) }), false);
  assert.equal(isValidReflection(null), false);
  const prompt = buildReflectPrompt({ instruction: "go", token: "E" });
  assert.ok(prompt.includes("Situation at the last step: unknown"));
  assert.ok(prompt.includes("Last token: E"));
  assert.ok(prompt.includes("Lessons already known: none"));
});
