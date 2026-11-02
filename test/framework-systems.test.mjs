import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import test from "node:test";

import { createChatPolicy, createChatThinker, createConfiguredSystems, createEmbodiedAgent, createRegistry, inspectConfiguration, runAsyncEpisode } from "../src/framework/index.mjs";

async function fixture(context) {
  const directory = await mkdtemp(join(tmpdir(), "async-systems-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const lifecycle = [];
  const registry = createRegistry();
  registry.registerEnvironment("counter", () => {
    let count = 0;
    const observation = () => ({ id: String(count), timestampMs: Date.now(), state: { count }, actions: [{ name: "move", parameters: { type: "object" } }] });
    return {
      id: "counter", timing: "realtime",
      reset: async () => ({ observation: observation() }),
      step: async (_, { signal }) => {
        await delay(10, null, { signal });
        count++;
        return { observation: observation(), terminated: count === 4, success: count === 4 };
      },
      stop: async () => lifecycle.push("stop"),
      close: async () => lifecycle.push("close-environment"),
    };
  });
  registry.registerPolicy("controller", ({ name }) => ({
    id: name,
    decide: async () => ({ action: { name: "move", arguments: {} } }),
    close: async () => lifecycle.push(`close-${name}`),
  }));
  await writeFile(join(directory, "brain.mjs"), `
    import { setTimeout as delay } from 'node:timers/promises';
    export default () => ({
      async think({ signal }) {
        await delay(3, null, { signal });
        return { candidateId: 'precise', directive: 'Use fine motion.', reason: 'Target is close.' };
      }
    });
  `);
  const config = {
    environment: "counter", systems: {
      system0: ["fast", "precise"].map(id => ({ id, kind: "sdk", policy: { type: "controller", options: { name: id } } })),
      system2: { type: "module", options: { module: "./brain.mjs" } },
      thinkEverySteps: 10,
    },
    episodes: [{ id: "paired:1", seed: 1 }],
    budget: { maxSteps: 5, maxDurationMs: 1000 },
  };
  return { directory, lifecycle, registry, config };
}

test("configured systems run through the public agent and record replayable events in evaluation", async context => {
  const { directory, lifecycle, registry, config } = await fixture(context);
  const agent = await createEmbodiedAgent(config, { baseDir: directory, registry });
  const live = [];
  const summary = await agent.evaluate({ outputDir: "recording", onEvent: event => live.push(event) });
  assert.equal(summary.successes, 1);
  const events = (await readFile(join(summary.outputDir, "0001.events.jsonl"), "utf8")).trim().split("\n").map(JSON.parse);
  const episode = JSON.parse(await readFile(join(summary.outputDir, "0001.json"), "utf8"));
  assert.equal(live.length, events.length);
  assert.ok(events.some(event => event.type === "thinking"));
  assert.ok(events.some(event => event.type === "action" && event.status === "started"));
  assert.ok(episode.asyncMetrics.overlapMs > 0);
  assert.equal(episode.asyncMetrics.blockedThinkingMs, 0);
  assert.ok(lifecycle.includes("close-fast"));
  assert.ok(lifecycle.includes("close-precise"));
  assert.equal(lifecycle.filter(item => item === "close-environment").length, 1);
  await agent.close();
});

test("agent.start exposes events immediately and agent.close cancels the same owned run", async context => {
  const { directory, lifecycle, registry, config } = await fixture(context);
  const agent = await createEmbodiedAgent(config, { baseDir: directory, registry });
  const session = agent.start();
  assert.equal(typeof session.id, "string");
  const enteredAction = new Promise(resolve => session.subscribe(event => {
    if (event.type === "action" && event.status === "started") resolve();
  }));
  await enteredAction;
  await agent.close();
  assert.equal((await session.done).reason, "cancelled");
  assert.ok(lifecycle.includes("stop"));
  assert.ok(lifecycle.includes("close-precise"));
});

test("doctor checks the configured systems without requiring the unused default chat policy", async context => {
  const { directory, registry, config } = await fixture(context);
  const report = await inspectConfiguration(config, { baseDir: directory, registry, env: {} });
  assert.equal(report.ok, true);
  assert.ok(report.checks.some(item => item.name === "System2 brain"));
});

test("factory systems close their controllers and brain after an episode", async () => {
  const closed = [];
  const action = { name: "finish", arguments: {} };
  const observation = { state: {}, actions: [{ name: "finish", parameters: { type: "object" } }] };
  const agent = await createEmbodiedAgent({
    createEnvironment: () => ({
      reset: async () => ({ observation }),
      step: async () => ({ observation, terminated: true, success: true }),
      close: async () => closed.push("environment"),
    }),
    createSystems: () => ({
      system0: [{ id: "sdk", policy: { decide: async () => ({ action }), close: async () => closed.push("sdk") } }],
      system2: { think: async () => ({ candidateId: "sdk" }), close: async () => closed.push("brain") },
    }),
  });
  assert.equal((await agent.run()).success, true);
  assert.deepEqual(closed.sort(), ["brain", "environment", "sdk"]);
  await agent.close();
});

test("chat System2 recommends a controller with usage, and receives its descriptive routing context", async context => {
  let body;
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    body = JSON.parse(Buffer.concat(chunks));
    response.writeHead(200, { "content-type": "text/event-stream" });
    const answer = { name: "recommend_controller", arguments: { candidateId: "vla", directive: "Align the gripper." }, thought: "Precise alignment is needed." };
    response.write(`data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, function: { name: "take_action", arguments: JSON.stringify(answer) } }] }, finish_reason: "tool_calls" }] })}\n\n`);
    response.write(`data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 50, completion_tokens: 20 } })}\n\n`);
    response.end("data: [DONE]\n\n");
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  context.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const brain = createChatThinker({ baseUrl: `http://127.0.0.1:${server.address().port}/v1`, model: "fixture" });
  const result = await brain.think({ observation: { state: {} }, candidates: [{ id: "vla", kind: "vla", description: "Wrist alignment model" }], instruction: "Pick up the cup." });
  assert.equal(result.candidateId, "vla");
  assert.equal(result.directive, "Align the gripper.");
  assert.deepEqual(result.usage, { prompt_tokens: 50, completion_tokens: 20 });
  assert.match(body.messages[1].content[0].text, /Wrist alignment model/);
  assert.match(body.messages[0].content, /System2/);
});

test("a brain setup failure closes the already-created router and System0 policies", async context => {
  const { directory, lifecycle, registry, config } = await fixture(context);
  await writeFile(join(directory, "broken-brain.mjs"), "export default () => { throw new Error('brain setup failed'); };\n");
  await assert.rejects(createConfiguredSystems({
    ...config.systems,
    system1: { route: async () => ({ candidateId: "fast" }), close: async () => lifecycle.push("close-router") },
    system2: { type: "module", options: { module: "./broken-brain.mjs" } },
  }, { registry, baseDir: directory }), /brain setup failed/);
  assert.deepEqual(lifecycle.sort(), ["close-fast", "close-precise", "close-router"]);
});

test("a chat System0 controller receives the task and acknowledged brain directive after the first action", async context => {
  const { registry } = await fixture(context);
  let body;
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    body = JSON.parse(Buffer.concat(chunks));
    response.writeHead(200, { "content-type": "text/event-stream" });
    const answer = { name: "move", arguments: {}, thought: "Apply the corrected approach." };
    response.end(`data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, function: { name: "take_action", arguments: JSON.stringify(answer) } }] }, finish_reason: "tool_calls" }] })}\n\ndata: [DONE]\n\n`);
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  context.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const transitions = [];
  await runAsyncEpisode({
    environment: await registry.createEnvironment({ type: "counter" }),
    instruction: "Pick up the cup.", maxSteps: 2, thinkEverySteps: 10,
    system0: [
      { id: "fast", policy: { decide: async () => ({ action: { name: "move", arguments: {} } }) } },
      { id: "chat", policy: createChatPolicy({ baseUrl: `http://127.0.0.1:${server.address().port}/v1`, model: "controller" }) },
    ],
    system2: { think: async () => ({ candidateId: "chat", directive: "Approach the handle from the right." }) },
    onTransition: record => transitions.push(record),
  });
  assert.deepEqual(transitions.filter(record => record.type === "step").map(record => record.route.candidateId), ["fast", "chat"]);
  assert.match(body.messages[1].content[0].text, /Pick up the cup\./);
  assert.match(body.messages[1].content[0].text, /System2 directive for this action: Approach the handle from the right\./);
});
