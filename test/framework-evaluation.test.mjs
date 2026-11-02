import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { runEvaluation } from "../src/framework/evaluation.mjs";

const EPISODES = [{ id: "first", task: "first game", seed: 1 }, { id: "second", task: "second game", seed: 2 }];
const OFFICIAL_SCORECARD = { score: 123.45, games: [{ gameId: "first game", score: 98.7 }], closed: true };

async function outputDirectory(context) {
  const directory = await mkdtemp(join(tmpdir(), "fast-evaluation-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

function fixture({ failTask = null, scorecardError = null, closeError = null } = {}) {
  const environments = [];
  const policies = [];
  function createEnvironment() {
    const calls = { resets: [], scorecards: [], closes: 0 };
    let task;
    const observation = () => ({
      id: task,
      timestampMs: Date.now(),
      state: {},
      images: [{ mimeType: "image/png", data: "fixture-image" }],
      actions: [{ name: "finish", description: "finish this game", parameters: { type: "object", properties: {}, additionalProperties: false } }],
    });
    const environment = {
      id: "fake-arc",
      timing: "turn-based",
      async reset(request) {
        task = request.task;
        calls.resets.push(request);
        if (task === failTask) throw new Error("backend unavailable");
        return { observation: observation(), terminated: false, truncated: false, success: null, reward: null, info: {} };
      },
      async step() {
        return { observation: observation(), terminated: true, truncated: false, success: true, reward: null, info: {} };
      },
      async scorecard(options) {
        calls.scorecards.push(options);
        if (scorecardError) throw new Error(scorecardError);
        return OFFICIAL_SCORECARD;
      },
      async close() {
        calls.closes += 1;
        if (closeError) throw new Error(closeError);
      },
    };
    environments.push({ environment, calls });
    return environment;
  }
  function createPolicy(episode, environment) {
    const policy = { id: `policy-${episode.id}`, decide: async () => ({ action: { name: "finish", arguments: {} } }) };
    policies.push({ episode, environment, policy });
    return policy;
  }
  return { createEnvironment, createPolicy, environments, policies };
}

test("shared evaluation resets each game, rebuilds policies, and closes one official scorecard once", async context => {
  const outputDir = await outputDirectory(context);
  const setup = fixture();
  const metadata = { model: "vision-model", policy: "react", environment: "arc-agi3", mode: "online" };
  const summary = await runEvaluation({ episodes: EPISODES, ...setup, outputDir, reuseEnvironment: true, metadata });
  const manifest = JSON.parse(await readFile(join(outputDir, "manifest.json"), "utf8"));
  assert.deepEqual(manifest.metadata, metadata);
  assert.equal(manifest.createEnvironment, undefined);
  assert.equal(manifest.createPolicy, undefined);
  assert.equal(manifest.apiKey, undefined);
  assert.equal(setup.environments.length, 1);
  assert.equal(setup.policies.length, 2);
  assert.notEqual(setup.policies[0].policy, setup.policies[1].policy);
  assert.equal(setup.policies[0].environment, setup.policies[1].environment);
  const calls = setup.environments[0].calls;
  assert.deepEqual(calls.resets.map(request => ({ task: request.task, seed: request.seed })), EPISODES.map(episode => ({ task: episode.task, seed: episode.seed })));
  assert.deepEqual(calls.scorecards, [{ close: true }]);
  assert.equal(calls.closes, 1);
  assert.equal(summary.episodesRun, 2);
  assert.equal(summary.successes, 2);
  assert.equal(summary.scorecardErrors, 0);
  assert.equal(summary.closeErrors, 0);
  assert.equal(summary.officialScorecard.score, 123.45);
  assert.deepEqual(summary.officialScorecard, OFFICIAL_SCORECARD);
  assert.deepEqual(JSON.parse(await readFile(join(outputDir, "summary.json"), "utf8")), summary);
  const first = JSON.parse(await readFile(join(outputDir, "0001.json"), "utf8"));
  assert.equal(first.officialScorecard, undefined);
  assert.equal((await readFile(join(outputDir, "0001.jsonl"), "utf8")).includes("fixture-image"), false);
});

test("default evaluation creates and finalizes an independent environment for each episode", async context => {
  const outputDir = await outputDirectory(context);
  const setup = fixture();
  const summary = await runEvaluation({ episodes: EPISODES, ...setup, outputDir });
  assert.equal(setup.environments.length, 2);
  for (const { calls } of setup.environments) {
    assert.equal(calls.resets.length, 1);
    assert.deepEqual(calls.scorecards, [{ close: true }]);
    assert.equal(calls.closes, 1);
  }
  for (const filename of ["0001.json", "0002.json"]) {
    const result = JSON.parse(await readFile(join(outputDir, filename), "utf8"));
    assert.deepEqual(result.officialScorecard, OFFICIAL_SCORECARD);
  }
  assert.equal(summary.officialScorecard, undefined);
});

test("an episode error stops a shared environment but independent evaluation continues", async context => {
  for (const reuseEnvironment of [false, true]) {
    const outputDir = await outputDirectory(context);
    const setup = fixture({ failTask: EPISODES[0].task });
    const summary = await runEvaluation({ episodes: EPISODES, ...setup, outputDir, reuseEnvironment });
    assert.equal(summary.errors, 1);
    assert.equal(summary.successRate, null);
    assert.equal(summary.episodesRun, reuseEnvironment ? 1 : 2);
    assert.equal(summary.successes, reuseEnvironment ? 0 : 1);
    const first = JSON.parse(await readFile(join(outputDir, "0001.json"), "utf8"));
    assert.equal(first.reason, "error");
    assert.equal(first.success, null);
    assert.equal(first.error.message, "backend unavailable");
    for (const { calls } of setup.environments) assert.equal(calls.closes, 1);
  }
});

test("cancellation stops further episodes and finalizes the shared scorecard", async context => {
  const outputDir = await outputDirectory(context);
  const setup = fixture();
  const controller = new AbortController();
  const summary = await runEvaluation({
    episodes: EPISODES,
    ...setup,
    outputDir,
    reuseEnvironment: true,
    signal: controller.signal,
    createPolicy: (episode, environment) => {
      const policy = setup.createPolicy(episode, environment);
      policy.decide = async () => {
        controller.abort(new Error("evaluation cancelled"));
        return { action: { name: "finish", arguments: {} } };
      };
      return policy;
    },
  });
  assert.equal(summary.episodesRun, 1);
  assert.equal(summary.results[0].reason, "cancelled");
  assert.equal(summary.results[0].success, null);
  assert.equal(summary.successes, 0);
  assert.deepEqual(setup.environments[0].calls.scorecards, [{ close: true }]);
  assert.equal(setup.environments[0].calls.closes, 1);
});

test("scorecard failures are reported without inventing official scores and still close the environment", async context => {
  for (const reuseEnvironment of [false, true]) {
    const outputDir = await outputDirectory(context);
    const setup = fixture({ scorecardError: "official service unavailable" });
    const summary = await runEvaluation({ episodes: EPISODES, ...setup, outputDir, reuseEnvironment });
    const result = reuseEnvironment ? summary : JSON.parse(await readFile(join(outputDir, "0001.json"), "utf8"));
    assert.equal(result.officialScorecard, null);
    assert.equal(result.scorecardError, "official service unavailable");
    assert.equal(summary.scorecardErrors, reuseEnvironment ? 1 : 2);
    assert.equal(summary.closeErrors, 0);
    assert.equal(summary.successRate, null);
    assert.equal(setup.environments[0].calls.closes, 1);
  }
});

test("cleanup failures remain visible instead of becoming successful independent episodes", async context => {
  const outputDir = await outputDirectory(context);
  const setup = fixture({ closeError: "close failed" });
  const summary = await runEvaluation({ episodes: EPISODES.slice(0, 1), ...setup, outputDir });
  assert.equal(summary.errors, 1);
  assert.equal(summary.successes, 0);
  assert.equal(summary.results[0].reason, "error");
  assert.equal(summary.results[0].success, null);
  assert.equal(summary.closeErrors, 1);
  assert.equal(summary.scorecardErrors, 0);
});

test("shared cleanup failures preserve episode outcomes but leave the overall rate incomplete", async context => {
  const outputDir = await outputDirectory(context);
  const setup = fixture({ closeError: "shared close failed" });
  const summary = await runEvaluation({ episodes: EPISODES, ...setup, outputDir, reuseEnvironment: true });
  assert.equal(summary.episodesRun, 2);
  assert.equal(summary.successes, 2);
  assert.equal(summary.errors, 0);
  assert.equal(summary.closeErrors, 1);
  assert.equal(summary.scorecardErrors, 0);
  assert.equal(summary.closeError, "shared close failed");
  assert.equal(summary.successRate, null);
  assert.equal(summary.results[0].success, true);
  assert.equal(summary.results[1].success, true);
});

test("evaluation output never overwrites an existing run or episode trace", async context => {
  const outputDir = await outputDirectory(context);
  const setup = fixture();
  await runEvaluation({ episodes: EPISODES, ...setup, outputDir, reuseEnvironment: true });
  const original = await readFile(join(outputDir, "summary.json"), "utf8");
  await assert.rejects(runEvaluation({ episodes: EPISODES, ...setup, outputDir, reuseEnvironment: true }), { code: "EEXIST" });
  assert.equal(await readFile(join(outputDir, "summary.json"), "utf8"), original);
  assert.equal(setup.environments.length, 1);

  const occupiedDir = await outputDirectory(context);
  await writeFile(join(occupiedDir, "0001.jsonl"), "keep this trace\n");
  await assert.rejects(runEvaluation({ episodes: EPISODES, ...setup, outputDir: occupiedDir }), { code: "EEXIST" });
  assert.equal(await readFile(join(occupiedDir, "0001.jsonl"), "utf8"), "keep this trace\n");
});
