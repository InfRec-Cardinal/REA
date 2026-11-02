import assert from "node:assert/strict";
import { createServer } from "node:http";
import test from "node:test";

import { buildDecisionPrompt, createChatPolicy } from "../src/framework/policy.mjs";

const EMPTY = { type: "object", properties: {}, additionalProperties: false };
const CLICK = { name: "ACTION6", description: "Click a grid cell", parameters: { type: "object", properties: { x: { type: "integer", minimum: 0, maximum: 63 }, y: { type: "integer", minimum: 0, maximum: 63 } }, required: ["x", "y"], additionalProperties: false } };

async function provider(context, handler) {
  const requests = [];
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    requests.push(body);
    response.writeHead(200, { "content-type": "text/event-stream" });
    const send = (delta, finishReason = null) => response.write(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: finishReason }] })}\n\n`);
    handler({ body, response, send });
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  context.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  return { requests, baseUrl: `http://127.0.0.1:${server.address().port}/v1` };
}

function toolReply(send, response, decision, usage = null) {
  const text = JSON.stringify(decision);
  send({ tool_calls: [{ index: 0, function: { name: "take_action", arguments: text.slice(0, 20) } }] });
  send({ tool_calls: [{ index: 0, function: { arguments: text.slice(20) } }] }, "tool_calls");
  if (usage) response.write(`data: ${JSON.stringify({ choices: [], usage })}\n\n`);
  response.end("data: [DONE]\n\n");
}

test("chat tool policy sends current multimodal state and decodes coordinates, notes, and final usage", async context => {
  const usage = { prompt_tokens: 230, completion_tokens: 24, total_tokens: 254 };
  const upstream = await provider(context, ({ send, response }) => toolReply(send, response, { name: "ACTION6", arguments: { x: 17, y: 29 }, thought: "Test the highlighted cell.", notes: "Blue cells respond to clicks." }, usage));
  const policy = createChatPolicy({ ...upstream, model: "vision", apiKey: "test-only" });
  const currentState = { grid: Array.from({ length: 20 }, () => Array.from({ length: 20 }, (_, index) => index)) };
  const result = await policy.decide({
    observation: { id: "frame-4", timestampMs: 1234, text: "Choose a cell", state: currentState, images: [{ mimeType: "image/png", data: "Y3VycmVudA==" }], actions: [CLICK] },
    instruction: "Solve the environment.",
    history: [{ step: 3, action: { name: "ACTION6", arguments: { x: 2, y: 3 } }, observation: { id: "old", images: [{ mimeType: "image/png", data: "PRIVATE_OLD_IMAGE" }], state: {} }, nextObservation: { id: "frame-4", state: {} }, thought: "Earlier hypothesis" }],
    notes: "Earlier note",
    budget: { stepsRemaining: 12, remainingMs: 45000, decisionMs: 1000 },
  });
  assert.deepEqual(result, { action: { name: "ACTION6", arguments: { x: 17, y: 29 } }, thought: "Test the highlighted cell.", notes: "Blue cells respond to clicks.", usage, model: "vision" });
  const request = upstream.requests[0];
  assert.deepEqual(request.tool_choice, { type: "function", function: { name: "take_action" } });
  assert.deepEqual(request.tools[0].function.parameters.properties.name.enum, ["ACTION6"]);
  const [text, image] = request.messages[1].content;
  assert.match(text.text, /Earlier note/u);
  assert.match(text.text, /Remaining budget: \{"stepsRemaining":12,"remainingMs":45000,"decisionMs":1000\}/u);
  assert.ok(text.text.includes(JSON.stringify(currentState)));
  assert.ok(text.text.includes(JSON.stringify(CLICK.parameters)));
  assert.ok(!JSON.stringify(request).includes("PRIVATE_OLD_IMAGE"));
  assert.equal(image.image_url.url, "data:image/png;base64,Y3VycmVudA==");
});

test("json policy supports vector arguments and bounded recent history", async context => {
  const upstream = await provider(context, ({ send, response }) => {
    send({ content: '{"name":"MOVE","arguments":{"velocity":[0.1,0,0]},"notes":"' });
    send({ content: 'Continue toward the target."}' }, "stop");
    response.end("data: [DONE]\n\n");
  });
  const action = { name: "MOVE", parameters: { type: "object", properties: { velocity: { type: "array", items: { type: "number" }, minItems: 3, maxItems: 3 } }, required: ["velocity"] } };
  const policy = createChatPolicy({ ...upstream, model: "generic", mode: "json", historyLimit: 2 });
  const result = await policy.decide({ observation: { id: "now", state: {}, actions: [action] }, history: [{ step: 1, thought: "discarded-entry" }, { step: 2, thought: "second-entry" }, { step: 3, thought: "third-entry" }] });
  assert.deepEqual(result.action, { name: "MOVE", arguments: { velocity: [0.1, 0, 0] } });
  assert.equal(result.usage, null);
  const request = upstream.requests[0];
  assert.deepEqual(request.response_format, { type: "json_object" });
  assert.equal(request.tools, undefined);
  assert.ok(!request.messages[1].content[0].text.includes("discarded-entry"));
  assert.match(request.messages[1].content[0].text, /second-entry/u);
});

test("token policy requests one token only for parameter-free single-character actions", async context => {
  const upstream = await provider(context, ({ send, response }) => { send({ content: "A" }, "stop"); response.end("data: [DONE]\n\n"); });
  const policy = createChatPolicy({ ...upstream, model: "reflex", mode: "token" });
  const result = await policy.decide({ observation: { id: "one", state: {}, actions: [{ name: "A", parameters: EMPTY }, { name: "B", parameters: EMPTY }] } });
  assert.deepEqual(result.action, { name: "A", arguments: {} });
  assert.equal(result.usage, null);
  assert.equal(upstream.requests[0].max_tokens, 1);
  assert.equal(upstream.requests[0].tools, undefined);
  await assert.rejects(policy.decide({ observation: { actions: [CLICK] } }), /parameter-free single-character/u);
  await assert.rejects(policy.decide({ observation: { actions: [{ ...CLICK, name: "C" }] } }), /parameter-free single-character/u);
  assert.equal(upstream.requests.length, 1);
});

test("policy reports malformed or unavailable actions without substituting an action", async context => {
  let calls = 0;
  const upstream = await provider(context, ({ send, response }) => {
    if (++calls === 1) toolReply(send, response, { name: "UNAVAILABLE", arguments: {} });
    else { send({ tool_calls: [{ index: 0, function: { name: "take_action", arguments: '{"name":' } }] }, "length"); response.end("data: [DONE]\n\n"); }
  });
  const policy = createChatPolicy({ ...upstream, model: "generic" });
  const input = { observation: { actions: [CLICK] } };
  await assert.rejects(policy.decide(input), error => error.errorType === "invalid_action" && error.code === "invalid_action");
  await assert.rejects(policy.decide(input), error => error.code === "output_limit");
});

test("DeepSeek action requests disable implicit thinking and allow an explicit override", async context => {
  const upstream = await provider(context, ({ send, response }) => toolReply(send, response, { name: "ACTION6", arguments: { x: 1, y: 2 } }));
  const input = { observation: { actions: [CLICK] } };
  await createChatPolicy({ ...upstream, model: "deepseek-v4-flash-vision-exp" }).decide(input);
  await createChatPolicy({ ...upstream, model: "deepseek-v4-flash-vision-exp", thinking: { type: "enabled" }, maxOutputTokens: 8192 }).decide(input);
  await createChatPolicy({ ...upstream, model: "other-model" }).decide(input);
  assert.deepEqual(upstream.requests[0].thinking, { type: "disabled" });
  assert.deepEqual(upstream.requests[1].thinking, { type: "enabled" });
  assert.equal(upstream.requests[2].thinking, undefined);
});

test("reasoning-only output exhaustion reports its cause and preserves charged usage", async context => {
  const usage = { prompt_tokens: 30, completion_tokens: 10, total_tokens: 40 };
  const upstream = await provider(context, ({ send, response }) => {
    send({ reasoning_content: "fixture" }, "length");
    response.write(`data: ${JSON.stringify({ choices: [], usage })}\n\n`);
    response.end("data: [DONE]\n\n");
  });
  const policy = createChatPolicy({ ...upstream, model: "reasoning-model", mode: "json" });
  await assert.rejects(policy.decide({ observation: { actions: [CLICK] } }), error => {
    assert.equal(error.code, "output_limit");
    assert.deepEqual(error.usage, usage);
    return true;
  });
});

test("policy observes decision budgets and caller cancellation", async context => {
  const upstream = await provider(context, () => {});
  const policy = createChatPolicy({ ...upstream, model: "generic" });
  const observation = { actions: [CLICK] };
  await assert.rejects(policy.decide({ observation, budget: { decisionMs: 0 } }), /budget exhausted/u);
  await assert.rejects(policy.decide({ observation, signal: AbortSignal.abort(new Error("cancelled by caller")) }), /cancelled by caller/u);
  assert.equal(upstream.requests.length, 0);
  await assert.rejects(policy.decide({ observation, budget: { decisionMs: 50 } }), error => error.code === "deadline");
  assert.equal(upstream.requests.length, 1);
});

test("large robot feedback preserves every transition in the configured history window", async context => {
  const upstream = await provider(context, ({ send, response }) => toolReply(send, response, { name: "ACTION6", arguments: { x: 1, y: 2 } }));
  const policy = createChatPolicy({ ...upstream, model: "vision", historyLimit: 12 });
  const history = Array.from({ length: 13 }, (_, index) => ({
    step: index + 1,
    action: { name: "take_action", arguments: { actuator: "x_cm", value: -2 } },
    info: { execution: { controller_log: "feedback".repeat(12000), status: "completed" } },
    thought: `Check the measured change after step ${index + 1}.`,
    observation: {
      id: `before-${index + 1}`, state: { tcp: [index, 0, 0] },
      images: [{ data: "PRIVATE_OLD_IMAGE" }], imageB64: "PRIVATE_OLD_IMAGE",
    },
    nextObservation: { id: `after-${index + 1}`, state: { tcp: [index - 2, 0, 0] } },
  }));
  await policy.decide({
    observation: { id: "current", state: { phase: "current complete state" }, actions: [CLICK] },
    history,
  });
  const prompt = upstream.requests[0].messages[1].content[0].text;
  const historyJson = prompt.split("Recent transitions: ")[1].split("\n\nFeedback:")[0];
  assert.ok(!historyJson.includes("[history excerpt truncated]"));
  const transitions = JSON.parse(historyJson);
  assert.deepEqual(transitions, history.slice(1).map(({ observation, ...entry }) => ({
    ...entry, observation: { id: observation.id, state: observation.state },
  })));
  assert.match(prompt, /current complete state/u);
  assert.ok(!prompt.includes("PRIVATE_OLD_IMAGE"));
});

test("direct decision prompts keep complete selected history and support disabling the window", () => {
  const selected = [{ step: 2, info: { controller_log: "feedback".repeat(12000) }, nextObservation: { id: "after" } }];
  const input = { observation: { actions: [CLICK] }, history: [{ step: 1 }], historyForPrompt: selected };
  const prompt = buildDecisionPrompt(input);
  const historyJson = prompt.split("Recent transitions: ")[1].split("\n\nFeedback:")[0];
  assert.ok(!historyJson.includes("[history excerpt truncated]"));
  assert.deepEqual(JSON.parse(historyJson), selected);
  assert.ok(buildDecisionPrompt({ ...input, historyLimit: 0 }).includes("Recent transitions: []\n\n"));
});

test("named robot cameras retain their labels beside each image", async context => {
  const upstream = await provider(context, ({ send, response }) => toolReply(send, response, { name: "ACTION6", arguments: { x: 1, y: 2 } }));
  const policy = createChatPolicy({ ...upstream, model: "vision" });
  await policy.decide({ observation: { actions: [CLICK], images: [
    { name: "left_wrist", mimeType: "image/png", data: "bGVmdA==" },
    { name: "right_wrist", mimeType: "image/png", data: "cmlnaHQ=" },
  ] } });
  const content = upstream.requests[0].messages[1].content;
  assert.equal(content[1].text, "Camera: left_wrist");
  assert.equal(content[2].image_url.url, "data:image/png;base64,bGVmdA==");
  assert.equal(content[3].text, "Camera: right_wrist");
  assert.equal(content[4].image_url.url, "data:image/png;base64,cmlnaHQ=");
});

test("chat decisions may report an evidence-based working or complete claim", async context => {
  const upstream = await provider(context, ({ send, response }) => toolReply(send, response, {
    name: "ACTION6",
    arguments: { x: 1, y: 2 },
    claim: { status: "complete", evidence: "the highlighted cell changed" },
  }));
  const policy = createChatPolicy({ ...upstream, model: "vision" });
  const result = await policy.decide({ observation: { actions: [CLICK] } });
  assert.deepEqual(result.claim, { status: "complete", evidence: "the highlighted cell changed" });
});
