import assert from "node:assert/strict";
import { test } from "node:test";

import { buildTokenPrompt, createTokenInfer } from "../src/infer-token.mjs";
import HOLONOMIC from "./fixtures/profiles/holonomic.mjs";

const FRAME = { imageB64: "aGVsbG8=", mimeType: "image/jpeg", depth: { center_m: 1.4 } };

/** Minimal SSE server standing in for the provider. */
function sseServer(chunks, { status = 200 } = {}) {
  return async function fetchStub(url, init) {
    if (String(url).endsWith("/models")) return new Response("{}", { status: 200 });
    if (status !== 200) return new Response("boom", { status });
    const body = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    });
    return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
  };
}

function sseData(payload) {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function delta(content) {
  return sseData({ choices: [{ delta: { content }, finish_reason: null }] });
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

test("the request pins max_tokens to 1 and sends no tools", async () => {
  let sent = null;
  const stub = async (url, init) => {
    sent = JSON.parse(init.body);
    return sseServer([delta("W"), "data: [DONE]\n\n"])(url, init);
  };
  await withFetch(stub, async () => {
    const infer = createTokenInfer({ apiKey: "k", baseUrl: "https://example.test/v1" });
    await infer("deepseek-v4-flash-vision-exp", FRAME, { instruction: "go" });
  });
  assert.equal(sent.max_tokens, 1, "a one-token answer is the whole point of this engine");
  assert.equal(sent.temperature, 0);
  assert.equal(sent.tools, undefined);
  assert.equal(sent.tool_choice, undefined);
  assert.deepEqual(sent.thinking, { type: "disabled" });
});

test("the first content delta resolves the decision", async () => {
  await withFetch(sseServer([delta("W"), delta("ignored"), "data: [DONE]\n\n"]), async () => {
    const infer = createTokenInfer({ apiKey: "k", baseUrl: "https://example.test/v1" });
    const result = await infer("m", FRAME, { instruction: "go" });
    assert.equal(result.token, "W");
    assert.equal(result.source, "content");
    assert.ok(result.tokenReadyS >= 0);
    assert.equal(result.actionReadyS, result.tokenReadyS);
  });
});

test("a token that lands in reasoning_content is still recovered", async () => {
  // DeepSeek sometimes routes very short answers into the reasoning channel even with thinking off.
  const chunks = [sseData({ choices: [{ delta: { reasoning_content: "Q" }, finish_reason: null }] }), "data: [DONE]\n\n"];
  await withFetch(sseServer(chunks), async () => {
    const infer = createTokenInfer({ apiKey: "k", baseUrl: "https://example.test/v1" });
    const result = await infer("deepseek-x", FRAME, { instruction: "go" });
    assert.equal(result.token, "Q");
    assert.equal(result.source, "reasoning_content");
  });
});

test("a stream that never yields a valid token is an error, not a guess", async () => {
  await withFetch(sseServer([delta("hello"), "data: [DONE]\n\n"]), async () => {
    const infer = createTokenInfer({ apiKey: "k", baseUrl: "https://example.test/v1" });
    await assert.rejects(() => infer("m", FRAME, { instruction: "go" }), /without a valid token/u);
  });
});

test("an upstream error is classified and the key is not leaked", async () => {
  await withFetch(sseServer([], { status: 500 }), async () => {
    const infer = createTokenInfer({ apiKey: "super-secret", baseUrl: "https://example.test/v1" });
    await assert.rejects(
      () => infer("m", FRAME, { instruction: "go" }),
      (error) => {
        assert.equal(error.errorType, "upstream_500");
        assert.equal(error.retryable, true);
        assert.ok(!error.message.includes("super-secret"));
        return true;
      },
    );
  });
});

test("a missing API key fails before any request is made", async () => {
  const infer = createTokenInfer({ apiKey: "", baseUrl: "https://example.test/v1" });
  await assert.rejects(() => infer("m", FRAME, { instruction: "go" }), (error) => {
    assert.equal(error.errorType, "config_error");
    assert.equal(error.retryable, false);
    return true;
  });
});

test("every prompt variant states the task and keeps the capability disclaimer where it matters", () => {
  const full = buildTokenPrompt({ instruction: "walk to the door", variant: "full" });
  const short = buildTokenPrompt({ instruction: "walk to the door", variant: "short" });
  const min = buildTokenPrompt({ instruction: "walk to the door", variant: "min" });
  for (const prompt of [full, short, min]) assert.ok(prompt.includes("walk to the door"));
  // dropping the disclaimer is what lets a model talk itself into calling a gesture a grasp
  assert.ok(/grasp|BOX_LIFT/u.test(full));
  assert.ok(/grasp|BOX_LIFT/u.test(short));
});

test("thinker advice reaches the prompt", () => {
  const prompt = buildTokenPrompt({
    instruction: "go",
    advice: { verdict: "CORRECT", next_token: "Q", brief: "wall ahead", confidence: 0.8 },
    variant: "full",
  });
  assert.ok(prompt.includes("wall ahead"));
});

test("the previous step's plan reaches the reflex prompt as advice, capped, and never on the min variant", () => {
  const plan = "expect the door frame at centre after the turn; W only if centre > 1.5 m";
  const short = buildTokenPrompt({ instruction: "walk to the door", variant: "short", directive: plan });
  assert.ok(short.includes(`Plan: ${plan}`));
  assert.ok(short.indexOf("Plan:") < short.indexOf("Advice:"), "the plan sits with the other advice, before the reviewer's");
  const full = buildTokenPrompt({ instruction: "walk to the door", variant: "full", directive: plan });
  assert.ok(full.includes(`Plan carried from the previous step (advice, not a command): ${plan}`));
  const min = buildTokenPrompt({ instruction: "walk to the door", variant: "min", directive: plan });
  assert.equal(min.includes("Plan"), false, "min is the latency floor");
  const long = buildTokenPrompt({ instruction: "x", variant: "short", directive: "y".repeat(500) });
  assert.ok(long.includes(`Plan: ${"y".repeat(120)}\n`), "capped at 120 characters");
  assert.equal(buildTokenPrompt({ instruction: "x", variant: "short", directive: "   " }).includes("Plan:"), false);
});

test("a candidate on another provider is asked at that provider's endpoint with its key and extra body", async () => {
  const seen = [];
  const stub = async (url, init) => {
    seen.push({ url: String(url), auth: init.headers.authorization, body: JSON.parse(init.body) });
    const body = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        controller.enqueue(encoder.encode(delta("W")));
        controller.enqueue(encoder.encode(sseData({ choices: [], usage: { prompt_tokens: 120, completion_tokens: 1 } })));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    });
    return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
  };
  const result = await withFetch(stub, async () => {
    const infer = createTokenInfer({ apiKey: "gateway-key", baseUrl: "https://gateway.test/v1" });
    return infer("gemini-3.8-flash", FRAME, {
      instruction: "go",
      deadlineMs: 2000,
      cfg: { provider: "gemini", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", apiKey: "gk-test", extraBody: { reasoning_effort: "low" } },
    });
  });
  assert.equal(result.token, "W");
  assert.deepEqual(result.usage, { prompt_tokens: 120, completion_tokens: 1 });
  assert.equal(seen.length, 1);
  assert.equal(seen[0].url, "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions");
  assert.equal(seen[0].auth, "Bearer gk-test");
  assert.equal(seen[0].body.model, "gemini-3.8-flash");
  assert.equal(seen[0].body.max_tokens, 1);
  assert.equal(seen[0].body.reasoning_effort, "low");
  assert.equal(seen[0].body.thinking, undefined);

  // without a key for that provider the call fails as config_error before any request, and the
  // gateway's own key is never sent to the other provider
  await assert.rejects(
    withFetch(stub, async () => {
      const infer = createTokenInfer({ apiKey: "gateway-key", baseUrl: "https://gateway.test/v1" });
      return infer("gemini-3.8-flash", FRAME, { instruction: "go", cfg: { provider: "gemini", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", apiKey: "" } });
    }),
    error => error.errorType === "config_error" && /provider gemini/u.test(error.message),
  );
  assert.equal(seen.length, 1);
});

test("the token prompt renders the profile's legend, stop token and capability note", () => {
  const g1 = buildTokenPrompt({ instruction: "go", variant: "short" });
  assert.match(g1, /X stop W fwd/u);
  assert.match(g1, /Output X if unavailable or unclear/u);
  const other = buildTokenPrompt({ instruction: "go", variant: "short", profile: HOLONOMIC });
  assert.match(other, /H halt F forward/u);
  assert.match(other, /Output H if unavailable or unclear/u);
  assert.doesNotMatch(other, /grasp or lift/u, "no boxLift capability declared: no disclaimer");
  assert.equal(buildTokenPrompt({ instruction: "go", variant: "min", profile: HOLONOMIC }).startsWith("H halt F forward"), true);
});
