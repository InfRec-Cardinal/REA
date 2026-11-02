import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { createEmbodiedAgent, createRegistry, getPreset, loadConfiguration } from "../src/framework/index.mjs";

async function directory(context) {
  const path = await mkdtemp(join(tmpdir(), "fast-agent-"));
  context.after(() => rm(path, { recursive: true, force: true }));
  return path;
}

function trackedRegistry() {
  const registry = createRegistry();
  const events = [];
  for (const type of ["arm", "grid"]) registry.registerEnvironment(type, (options, context) => {
    events.push(["construct", type, options, context.episode.id]);
    let sequence = 0;
    const observation = () => ({ id: `${type}:${++sequence}`, timestampMs: Date.now(), state: {}, actions: [{ name: "finish", parameters: { type: "object" } }] });
    return {
      id: type, timing: "turn-based",
      async reset({ task, seed }) { events.push(["reset", type, task, seed]); return { observation: observation() }; },
      async step() { return { observation: observation(), success: true, terminated: true, reward: 1 }; },
      async stop() { events.push(["stop", type]); },
      async close() { events.push(["close", type]); },
    };
  });
  registry.registerPolicy("finish", () => ({
    id: "finish", decide: async () => ({ action: { name: "finish", arguments: {} } }),
    close: async () => events.push(["policy-close"]),
  }));
  return { registry, events };
}

test("the public API runs a task and a two-environment evaluation from the same recipe", async context => {
  const cwd = await directory(context);
  const agent = await createEmbodiedAgent("demo", { cwd });
  context.after(() => agent.close());
  const single = await agent.run({ task: "joints", seed: 4 });
  assert.equal(single.success, true);
  assert.equal(single.steps, 1);
  assert.equal(single.task, "joints");
  const summary = await agent.evaluate({ outputDir: "batch" });
  assert.equal(summary.outputDir, join(cwd, "batch"));
  assert.equal(summary.episodesRun, 2);
  assert.equal(summary.successes, 2);
  const manifest = JSON.parse(await readFile(join(summary.outputDir, "manifest.json"), "utf8"));
  assert.deepEqual(manifest.episodes.map(item => item.task), ["grid", "joints"]);
  assert.equal(manifest.metadata.framework.environment, "demo");
  const trace = (await readFile(join(summary.outputDir, "0001.jsonl"), "utf8")).trim().split("\n").map(JSON.parse);
  assert.deepEqual(trace.map(item => item.type), ["reset", "step", "episode_end"]);
});

test("configuration files resolve environment references without mutating the source object", async context => {
  const cwd = await directory(context);
  const config = { environment: "demo", policy: { type: "chat", options: { model: { env: "MODEL" }, apiKey: { env: "KEY", default: "" } } }, task: "grid" };
  await writeFile(join(cwd, "recipe.json"), JSON.stringify(config));
  const loaded = await loadConfiguration("recipe.json", { cwd, env: { MODEL: "local-vision" } });
  assert.equal(loaded.baseDir, cwd);
  assert.deepEqual(loaded.config.environment, { type: "demo", options: {} });
  assert.equal(loaded.config.policy.options.model, "local-vision");
  assert.equal(loaded.config.policy.options.apiKey, "");
  assert.equal(loaded.config.episodes[0].task, "grid");
  assert.deepEqual(config.policy.options.model, { env: "MODEL" });
  await assert.rejects(loadConfiguration(config, { env: {} }), /Set MODEL/u);
  await assert.rejects(loadConfiguration({ ...config, version: 9 }, { env: { MODEL: "m" } }), /version 9/u);
});

test("file-based API configurations resolve modules and output beside the configuration", async context => {
  const baseDir = await directory(context);
  const nested = join(baseDir, "recipes");
  await mkdir(nested);
  await writeFile(join(nested, "environment.mjs"), `export default () => {
    let sequence = 0;
    const observation = () => ({ id: String(++sequence), state: {}, actions: [{ name: "finish", parameters: { type: "object" } }] });
    return { id: "local-module", timing: "turn-based", reset: async () => ({ observation: observation() }),
      step: async () => ({ observation: observation(), terminated: true, success: true }) };
  };`);
  await writeFile(join(nested, "config.json"), JSON.stringify({
    environment: { type: "module", options: { module: "./environment.mjs" } },
    policy: { type: "scripted", options: { actions: [{ name: "finish", arguments: {} }] } },
    task: "finish",
  }));
  const agent = await createEmbodiedAgent("./recipes/config.json", { baseDir });
  context.after(() => agent.close());
  assert.equal((await agent.run()).success, true);
  const result = await agent.evaluate({ outputDir: "recording" });
  assert.equal(result.successes, 1);
  assert.equal(result.outputDir, join(nested, "recording"));
});

test("one evaluation switches robot and policy options per episode and closes both resources", async context => {
  const cwd = await directory(context);
  const { registry, events } = trackedRegistry();
  const agent = await createEmbodiedAgent({
    environment: { type: "arm", options: { jointCount: 14 } }, policy: "finish",
    episodes: [
      { id: "dual-arm", task: "transfer", seed: 2 },
      { id: "grid", task: "navigate", seed: 3, environment: { type: "grid", options: { width: 4 } } },
    ],
  }, { cwd, registry });
  const result = await agent.evaluate({ outputDir: "mixed" });
  assert.equal(result.successes, 2);
  assert.deepEqual(events.filter(event => event[0] === "construct").map(event => event.slice(1, 3)), [["arm", { jointCount: 14 }], ["grid", { width: 4 }]]);
  assert.equal(events.filter(event => event[0] === "policy-close").length, 2);
  assert.equal(events.filter(event => event[0] === "close").length, 2);
  await agent.close();
});

test("closing an agent cancels its current decision and releases the policy and environment", async () => {
  const { registry, events } = trackedRegistry();
  let started;
  const ready = new Promise(resolve => { started = resolve; });
  registry.registerPolicy("waiting", () => ({
    decide: () => { started(); return new Promise(() => {}); },
    close: async () => events.push(["policy-close"]),
  }));
  const agent = await createEmbodiedAgent({ environment: "arm", policy: "waiting" }, { registry });
  const running = agent.run();
  await ready;
  assert.throws(() => agent.run(), /already has a run/u);
  await agent.close();
  const result = await running;
  assert.equal(result.reason, "cancelled");
  assert.equal(events.filter(event => event[0] === "stop").length, 1);
  assert.equal(events.filter(event => event[0] === "close").length, 1);
  assert.equal(events.filter(event => event[0] === "policy-close").length, 1);
  assert.throws(() => agent.run(), /closed/u);
});

test("an unstarted agent and a policy setup failure leave no SDK running", async () => {
  const { registry, events } = trackedRegistry();
  const unused = await createEmbodiedAgent({ environment: "arm", policy: "finish" }, { registry });
  await unused.close();
  assert.deepEqual(events, []);
  registry.registerPolicy("broken", () => { throw new Error("model is unavailable"); });
  const agent = await createEmbodiedAgent({ environment: "arm", policy: "broken" }, { registry });
  await assert.rejects(agent.run(), /model is unavailable/u);
  assert.deepEqual(events, [], "lazy environment is not constructed before policy setup completes");
  await agent.close();
});

test("per-episode credentials are used at runtime without being written into the manifest", async context => {
  const cwd = await directory(context);
  const registry = createRegistry();
  let seenKey;
  registry.registerPolicy("credential-fixture", ({ apiKey }, { registry: local }) => {
    seenKey = apiKey;
    return local.createPolicy("demo");
  });
  const config = getPreset("demo");
  config.episodes = [{ id: "one", task: "grid", policy: { type: "credential-fixture", options: { apiKey: "private-value" } } }];
  const agent = await createEmbodiedAgent(config, { cwd, registry });
  const result = await agent.evaluate({ outputDir: "recording" });
  assert.equal(result.successes, 1);
  assert.equal(seenKey, "private-value");
  assert.equal((await readFile(join(result.outputDir, "manifest.json"), "utf8")).includes("private-value"), false);
  await agent.close();
});

test("the unified API still accepts the existing factory-style evaluation configurations", async context => {
  const cwd = await directory(context);
  const { default: factoryConfig } = await import("../examples/eval-fixtures.mjs");
  const agent = await createEmbodiedAgent(factoryConfig, { cwd });
  const result = await agent.evaluate({ outputDir: "legacy" });
  assert.equal(result.episodesRun, 2);
  assert.equal(result.successes, 2);
  await agent.close();
});
