import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEFAULT_JUDGE_MAX_TOKENS, DEFAULT_JUDGE_THRESHOLD, JUDGE_SYSTEM_PROMPT,
  buildJudgePrompt, createJudgeInfer, judgeVerdict, parseJudgeOutput,
} from "../src/infer-judge.mjs";

const frame = { imageB64: "aGk=", mimeType: "image/jpeg" };
const baseUrl = "https://example.test/v1";
const usage = { prompt_tokens: 700, completion_tokens: 20 };


const JUDGE_MODEL = "deepseek-v4-flash-vision-exp";

test("pickup judge describes the initial head frame and three current front views", async () => {
  let sent;
  const judge = createJudgeInfer({ model: JUDGE_MODEL, apiKey: "test", baseUrl, promptStyle: "pickup", fetchFn: async (_url, request) => {
    sent = JSON.parse(request.body);
    return jsonResponse(completion('{"done":true,"confidence":0.9}'));
  } });
  await judge(frame, { instruction: "Pick up the teddy bear", view: "front", initialFrame: frame });
  const content = sent.messages[1].content;
  const prompt = content[0].text;
  assert.match(prompt, /first is the robot's head camera at the start/u);
  assert.match(prompt, /left = from above head height/u);
  assert.match(prompt, /middle = overview from the robot's front-left/u);
  assert.match(prompt, /right = overview from the robot's front-right/u);
  assert.match(prompt, /It does NOT have to be placed anywhere/u);
  assert.match(prompt, /inside the closed fingers/u);
  assert.equal(content.filter(part => part.type === "image_url").length, 2);
  assert.equal(sent.messages[0].content, JUDGE_SYSTEM_PROMPT);
});

function jsonResponse(body, status = 200) {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

function completion(content, extra = {}) {
  return { choices: [{ message: { role: "assistant", content }, finish_reason: "stop" }], usage, ...extra };
}

async function rejection(promise) {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  assert.fail("expected the judge to throw");
  return null;
}

test("judge sends one non-streaming JSON completion with the frame and parses the fenced answer", async () => {
  let sentUrl;
  let sentHeaders;
  let sent;
  const fetchFn = async (url, request) => {
    sentUrl = url;
    sentHeaders = request.headers;
    sent = JSON.parse(request.body);
    return jsonResponse(completion("```json\n{\"done\": true, \"confidence\": 0.9, \"why\": \"bowl on plate\"}\n```"));
  };
  const judge = createJudgeInfer({ model: JUDGE_MODEL, apiKey: "test", baseUrl, fetchFn });
  const result = await judge(frame, { instruction: "put the bowl on the plate" });
  assert.equal(sentUrl, "https://example.test/v1/chat/completions");
  assert.equal(sentHeaders.authorization, "Bearer test");
  assert.equal(sent.model, JUDGE_MODEL);
  assert.equal(sent.stream, false);
  assert.equal(sent.max_tokens, DEFAULT_JUDGE_MAX_TOKENS);
  assert.equal(sent.max_tokens, 64);
  assert.equal(sent.temperature, 0);
  assert.deepEqual(sent.thinking, { type: "disabled" });
  assert.equal(sent.messages[0].role, "system");
  assert.equal(sent.messages[0].content, JUDGE_SYSTEM_PROMPT);
  const [text, image] = sent.messages[1].content;
  assert.equal(text.type, "text");
  assert.ok(text.text.includes('Instruction given to the robot: "put the bowl on the plate"'));
  assert.ok(text.text.includes("Right half: wrist camera (its left/right is mirrored relative to the external view)."));
  assert.equal(image.image_url.url, "data:image/jpeg;base64,aGk=");
  assert.equal(image.image_url.detail, "high");
  assert.equal(result.done, true);
  assert.equal(result.confidence, 0.9);
  assert.equal(result.rationale, "bowl on plate");
  assert.deepEqual(result.usage, usage);
  assert.equal(result.costUsd, Number(((700 * 0.44 + 20 * 1.32) / 1e6).toFixed(8)));
  assert.equal(result.model, JUDGE_MODEL);
  assert.equal(result.finishReason, "stop");
  assert.equal(result.responseStatus, 200);
  assert.equal(typeof result.latencyS, "number");
  assert.ok(result.outputText.includes('"done": true'));
});

test("non-deepseek models send no thinking key; agentview drops the mirrored-wrist sentence; strings coerce", async () => {
  let sent;
  const fetchFn = async (url, request) => {
    sent = JSON.parse(request.body);
    return jsonResponse(completion('{"done": "no", "confidence": "0.4", "why": "drawer closed"}'));
  };
  const judge = createJudgeInfer({ model: JUDGE_MODEL, apiKey: "test", baseUrl, model: "kimi-k2.5", fetchFn });
  const result = await judge(frame, { instruction: "open the drawer", view: "agentview" });
  assert.equal("thinking" in sent, false);
  assert.equal(sent.model, "kimi-k2.5");
  const prompt = sent.messages[1].content[0].text;
  assert.equal(prompt.includes("mirrored"), false);
  assert.ok(prompt.includes("The image shows the robot workspace now. External front camera."));
  assert.equal(result.done, false);
  assert.equal(result.confidence, 0.4);
  assert.equal(result.rationale, "drawer closed");
  assert.equal(result.costUsd, Number(((700 * 0.6 + 20 * 3.0) / 1e6).toFixed(8)));
});

test("buildJudgePrompt has the exact five lines", () => {
  const lines = buildJudgePrompt({ instruction: "turn on the stove", view: "montage" }).split("\n");
  assert.equal(lines.length, 5);
  assert.equal(lines[0], 'Instruction given to the robot: "turn on the stove"');
  assert.ok(lines[1].startsWith("The image shows the robot workspace now. Left half: external front camera."));
  assert.ok(lines[2].startsWith("Question: judging only from what is visible, has the instruction been completed?"));
  assert.equal(lines[3], "If the target state is hidden or ambiguous, answer done=false or lower your confidence.");
  assert.equal(lines[4], 'Answer exactly: {"done": true|false, "confidence": <number 0.0-1.0>, "why": "<at most 12 words>"}');
});

test("parseJudgeOutput strips fences, takes the first object, coerces and clamps, rejects prose", () => {
  assert.deepEqual(parseJudgeOutput('{"done": true, "confidence": 1.7, "why": "x"}'), { done: true, confidence: 1, rationale: "x" });
  assert.deepEqual(parseJudgeOutput('Sure! {"done": "yes"} and {"done": false}'), { done: true, confidence: 0.5, rationale: null });
  assert.deepEqual(parseJudgeOutput('```\n{"done": "FALSE", "confidence": -2}\n```'), { done: false, confidence: 0, rationale: null });
  assert.equal(parseJudgeOutput('{"done": true, "confidence": 0.9, "why": "x"}').rationale, "x");
  assert.equal(parseJudgeOutput(`{"done": true, "confidence": 0.9, "why": "${"y".repeat(300)}"}`).rationale.length, 200);
  assert.equal(parseJudgeOutput("Done."), null);
  assert.equal(parseJudgeOutput('{"confidence": 0.9}'), null);
  assert.equal(parseJudgeOutput('{"done": "maybe"}'), null);
  assert.equal(parseJudgeOutput('{"done": true, "confidence": "high"}'), null);
  assert.equal(parseJudgeOutput('{"done": true, "confidence": 0.9'), null);
  assert.equal(parseJudgeOutput("[1, 2]"), null);
  assert.equal(parseJudgeOutput(""), null);
  assert.equal(parseJudgeOutput(null), null);
});

test("judgeVerdict applies the confidence threshold", () => {
  assert.equal(DEFAULT_JUDGE_THRESHOLD, 0.6);
  assert.equal(judgeVerdict({ done: true, confidence: 0.59 }), "not_done");
  assert.equal(judgeVerdict({ done: true, confidence: 0.6 }), "done");
  assert.equal(judgeVerdict({ done: false, confidence: 0.99 }), "not_done");
  assert.equal(judgeVerdict({ done: true, confidence: 0.5 }, { threshold: 0.5 }), "done");
  assert.equal(judgeVerdict({ done: true, confidence: 0.5 }, { threshold: 0.51 }), "not_done");
  assert.equal(judgeVerdict({}), "not_done");
});

test("unparsable, empty, upstream and deadline failures are typed InferenceErrors with usage attached", async () => {
  const invalid = await rejection(createJudgeInfer({ model: JUDGE_MODEL, apiKey: "test", baseUrl, fetchFn: async () => jsonResponse(completion("Done.")) })(frame, { instruction: "x" }));
  assert.equal(invalid.name, "InferenceError");
  assert.equal(invalid.errorType, "invalid_judge_output");
  assert.equal(invalid.retryable, true);
  assert.deepEqual(invalid.usage, usage);
  assert.equal(invalid.outputText, "Done.");
  assert.equal(typeof invalid.latencyS, "number");

  const empty = await rejection(createJudgeInfer({ model: JUDGE_MODEL, apiKey: "test", baseUrl, fetchFn: async () => jsonResponse(completion("")) })(frame, { instruction: "x" }));
  assert.equal(empty.errorType, "empty_output");
  assert.equal(empty.retryable, true);
  assert.deepEqual(empty.usage, usage);

  const unauthorised = await rejection(createJudgeInfer({ model: JUDGE_MODEL, apiKey: "test", baseUrl, fetchFn: async () => new Response("bad key", { status: 401 }) })(frame, { instruction: "x" }));
  assert.equal(unauthorised.errorType, "auth_error");
  assert.equal(unauthorised.retryable, false);
  assert.equal(unauthorised.status, 401);
  assert.equal(unauthorised.usage, null);

  const overloaded = await rejection(createJudgeInfer({ model: JUDGE_MODEL, apiKey: "test", baseUrl, fetchFn: async () => new Response("busy", { status: 503 }) })(frame, { instruction: "x" }));
  assert.equal(overloaded.errorType, "upstream_503");
  assert.equal(overloaded.retryable, true);

  const notJson = await rejection(createJudgeInfer({ model: JUDGE_MODEL, apiKey: "test", baseUrl, fetchFn: async () => new Response("<html>", { status: 200 }) })(frame, { instruction: "x" }));
  assert.equal(notJson.errorType, "invalid_response");
  assert.equal(notJson.retryable, true);

  const neverResolves = (url, { signal }) => new Promise((_, reject) => {
    signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })), { once: true });
  });
  const late = await rejection(createJudgeInfer({ model: JUDGE_MODEL, apiKey: "test", baseUrl, deadlineMs: 20, fetchFn: neverResolves })(frame, { instruction: "x" }));
  assert.equal(late.errorType, "deadline");
  assert.equal(late.retryable, true);

  const network = await rejection(createJudgeInfer({ model: JUDGE_MODEL, apiKey: "test", baseUrl, fetchFn: async () => { throw new Error("socket hang up"); } })(frame, { instruction: "x" }));
  assert.equal(network.name, "InferenceError");
  assert.equal(network.retryable, true);
  assert.equal(network.errorType, "stream_error");
});

test("missing credentials or frame are non-retryable config errors and never call fetch", async () => {
  let fetched = 0;
  const fetchFn = async () => { fetched += 1; return jsonResponse(completion('{"done": true}')); };
  const noKey = await rejection(createJudgeInfer({ model: JUDGE_MODEL, apiKey: "", baseUrl, fetchFn })(frame, { instruction: "x" }));
  assert.equal(noKey.errorType, "config_error");
  assert.equal(noKey.retryable, false);
  const noFrame = await rejection(createJudgeInfer({ model: JUDGE_MODEL, apiKey: "test", baseUrl, fetchFn })({ mimeType: "image/jpeg" }, { instruction: "x" }));
  assert.equal(noFrame.errorType, "config_error");
  assert.equal(fetched, 0);
});

test("the caller's abort signal cancels the request", async () => {
  const controller = new AbortController();
  const fetchFn = (url, { signal }) => new Promise((_, reject) => {
    signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })), { once: true });
  });
  const pending = createJudgeInfer({ model: JUDGE_MODEL, apiKey: "test", baseUrl, deadlineMs: 5000, fetchFn })(frame, { instruction: "x", signal: controller.signal });
  controller.abort();
  const error = await rejection(pending);
  assert.equal(error.errorType, "deadline");
});

test("evidence prompt style asks for a description first and the verdict object last", async () => {
  const { buildJudgePrompt, judgeSystemPrompt, JUDGE_PROMPT_STYLES, parseJudgeOutput, createJudgeInfer } = await import("../src/infer-judge.mjs");
  assert.deepEqual([...JUDGE_PROMPT_STYLES], ["default", "evidence", "pickup"]);
  const lines = buildJudgePrompt({ instruction: "Put the bowl on the plate", view: "agentview", style: "evidence" }).split("\n");
  assert.equal(lines[0], 'Instruction given to the robot: "Put the bowl on the plate"');
  assert.ok(lines[2].startsWith("First, in one or two short sentences, state where the object named in the instruction is now"));
  assert.ok(lines[4].startsWith('Finish with exactly one line: {"done": true|false'));
  assert.ok(judgeSystemPrompt("evidence").includes("Give a brief visual description first"));
  assert.equal(judgeSystemPrompt("default"), judgeSystemPrompt());
  assert.deepEqual(parseJudgeOutput('The bowl sits on the plate near the robot base.\n{"done": true, "confidence": 0.8, "why": "bowl on plate"}'),
    { done: true, confidence: 0.8, rationale: "bowl on plate" });
  let sent = null;
  const fetchFn = async (url, init) => {
    sent = JSON.parse(init.body);
    return new Response(JSON.stringify({ choices: [{ message: { content: 'Plate is in front of the stove.\n{"done": true, "confidence": 0.7, "why": "plate at stove front"}' }, finish_reason: "stop" }], usage: { prompt_tokens: 10, completion_tokens: 30 } }), { status: 200, headers: { "content-type": "application/json" } });
  };
  const judge = createJudgeInfer({ model: JUDGE_MODEL, apiKey: "k", baseUrl: "https://example.test/v1", fetchFn, promptStyle: "evidence", maxTokens: 200 });
  const result = await judge({ imageB64: "QUJD" }, { instruction: "Push the plate to the front of the stove", view: "agentview" });
  assert.equal(result.done, true);
  assert.equal(result.promptStyle, "evidence");
  assert.equal(sent.max_tokens, 200);
  assert.ok(sent.messages[0].content.includes("visual description first"));
  assert.ok(sent.messages[1].content[0].text.includes("Finish with exactly one line"));
  assert.throws(() => createJudgeInfer({ model: JUDGE_MODEL, apiKey: "k", baseUrl: "https://example.test/v1", promptStyle: "bogus" }), /promptStyle/u);
});

test("an initial frame turns the request into a before/after comparison with two images", async () => {
  const { buildJudgePrompt, createJudgeInfer } = await import("../src/infer-judge.mjs");
  const lines = buildJudgePrompt({ instruction: "Push the plate to the front of the stove", view: "agentview", style: "evidence", withInitial: true }).split("\n");
  assert.ok(lines[1].startsWith("Two images are attached: the first shows the robot workspace at the start of the episode"));
  assert.ok(lines[2].startsWith("First, in one or two short sentences, compare the two images"));
  assert.ok(lines[3].includes("judging only from what is visible in the second image"));
  assert.ok(!buildJudgePrompt({ instruction: "x", withInitial: false }).includes("Two images"));
  let sent = null;
  const fetchFn = async (url, init) => {
    sent = JSON.parse(init.body);
    return new Response(JSON.stringify({ choices: [{ message: { content: '{"done": false, "confidence": 0.8, "why": "plate did not move"}' }, finish_reason: "stop" }], usage: { prompt_tokens: 10, completion_tokens: 12 } }), { status: 200, headers: { "content-type": "application/json" } });
  };
  const judge = createJudgeInfer({ model: JUDGE_MODEL, apiKey: "k", baseUrl: "https://example.test/v1", fetchFn });
  const result = await judge({ imageB64: "Tk9X", mimeType: "image/jpeg" }, { instruction: "Push the plate", view: "agentview", initialFrame: { imageB64: "U1RBUlQ=", mimeType: "image/png" } });
  assert.equal(result.withInitial, true);
  const parts = sent.messages[1].content;
  assert.equal(parts.length, 3);
  assert.equal(parts[1].image_url.url, "data:image/png;base64,U1RBUlQ=");
  assert.equal(parts[2].image_url.url, "data:image/jpeg;base64,Tk9X");
  assert.ok(parts[0].text.includes("Two images are attached"));
  await assert.rejects(judge({ imageB64: "Tk9X" }, { instruction: "x", initialFrame: {} }), error => error.errorType === "config_error");
});

test("a per-call maxTokens override replaces the configured budget and is echoed on results and errors", async () => {
  const bodies = [];
  let reply = completion('{"done": true, "confidence": 0.8, "why": "ok"}');
  const fetchFn = async (url, request) => { bodies.push(JSON.parse(request.body)); return jsonResponse(reply); };
  const judge = createJudgeInfer({ model: JUDGE_MODEL, apiKey: "test", baseUrl, fetchFn, maxTokens: 200 });
  const first = await judge(frame, { instruction: "x" });
  assert.equal(bodies[0].max_tokens, 200);
  assert.equal(first.maxTokens, 200);
  const second = await judge(frame, { instruction: "x", maxTokens: 400 });
  assert.equal(bodies[1].max_tokens, 400);
  assert.equal(second.maxTokens, 400);
  reply = { choices: [{ message: { role: "assistant", content: "The bowl is" }, finish_reason: "length" }], usage };
  const error = await rejection(judge(frame, { instruction: "x", maxTokens: 800 }));
  assert.equal(error.errorType, "invalid_judge_output");
  assert.equal(error.finishReason, "length");
  assert.equal(error.retryable, true);
  assert.equal(error.maxTokens, 800);
  assert.equal(bodies[2].max_tokens, 800);
});

test("pickup prompt style states the pick-up criterion; the front view describes the three external views and the head-camera start frame", async () => {
  const { buildJudgePrompt, JUDGE_VIEWS } = await import("../src/infer-judge.mjs");
  assert.deepEqual([...JUDGE_VIEWS], ["montage", "agentview", "front"]);
  const text = buildJudgePrompt({ instruction: "Pick up the red radio from the table and hold it in the gripper.", view: "front", style: "pickup", withInitial: true });
  const lines = text.split("\n");
  assert.equal(lines.length, 5);
  assert.ok(lines[1].includes("head camera at the start") && lines[1].includes("three views"), lines[1]);
  assert.ok(lines[2].includes("does NOT have to be placed anywhere") && lines[2].includes("NOT held"), lines[2]);
  assert.ok(lines[3].startsWith("Question:") && lines[3].includes("closed fingers"), lines[3]);
  assert.ok(lines[4].startsWith("Answer exactly:"));
  // default style on the front view keeps the frozen wording apart from the view note
  const plain = buildJudgePrompt({ instruction: "x", view: "front" });
  assert.ok(plain.includes("three views") && plain.includes("Question: judging only from what is visible"));
});
