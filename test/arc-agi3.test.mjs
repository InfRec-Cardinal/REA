import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { createArcAgi3Environment } from "../src/environments/arc-agi3.mjs";
import { runEpisode } from "../src/framework/runtime.mjs";

function fixtureWorker() {
  let sequence = 0;
  let stopped = false;
  let observation = null;
  const calls = [];
  return {
    calls,
    health: () => ({ running: !stopped }),
    stop() { stopped = true; },
    async request(payload) {
      calls.push(payload);
      if (payload.op === "configure") return { ok: true, result: { sdkVersion: "0.9.9" } };
      if (payload.op === "observe") return { ok: true, result: observation };
      if (payload.op === "scorecard") return { ok: true, result: { score: 4.25, closed: payload.close } };
      if (payload.op === "close") return { ok: true, result: { closed: true } };
      if (payload.op === "step" && payload.observationId !== observation.id) return { ok: false, error: "Action observation is stale" };
      observation = { id: `arc:1:${++sequence}`, timestampMs: Date.now(), state: { frame: [[[1]]] }, images: [], actions: [{ name: "ACTION6", parameters: { type: "object" } }] };
      return { ok: true, result: { observation, terminated: false, truncated: false, success: null, reward: null, info: {} } };
    },
  };
}

test("ARC adapter defaults offline, preserves transitions, and sends parameterized actions", async () => {
  const worker = fixtureWorker();
  const environment = createArcAgi3Environment({ worker, environmentsDir: "/fixtures" });
  assert.equal(environment.id, "arc-agi-3");
  assert.equal(environment.timing, "turn-based");
  const initial = await environment.reset({ task: { gameId: "fixture", seed: 9 } });
  assert.equal(worker.calls[0].options.operationMode, "OFFLINE");
  assert.equal(worker.calls[0].options.endOnGameOver, false);
  assert.equal(worker.calls[0].options.environmentsDir, "/fixtures");
  assert.deepEqual(worker.calls[1].task, { gameId: "fixture", seed: 9 });
  assert.deepEqual((await environment.observe()).state.frame, [[[1]]]);
  const action = { name: "ACTION6", arguments: { x: 20, y: 13 } };
  const next = await environment.step(action, { observation: initial.observation, reasoning: { thought: "test" } });
  assert.deepEqual(worker.calls.at(-1), { op: "step", action, observationId: initial.observation.id, reasoning: { thought: "test" } });
  assert.notEqual(next.observation.id, initial.observation.id);
  assert.equal(next.reward, null);
  await assert.rejects(environment.step(action, { observation: initial.observation }), /stale/u);
  assert.deepEqual(await environment.scorecard({ close: true }), { score: 4.25, closed: true });
  await environment.close();
  assert.equal(worker.health().running, false);
  assert.deepEqual(await environment.scorecard(), { score: 4.25, closed: true });
  assert.equal(worker.calls.filter(call => call.op === "scorecard").length, 1);
  await environment.close();
});

test("ARC mode changes are explicit and reset does not reconfigure the worker", async () => {
  assert.throws(() => createArcAgi3Environment({ operationMode: "NORMAL" }), /operationMode/u);
  const worker = fixtureWorker();
  const environment = createArcAgi3Environment({ worker, operationMode: "COMPETITION", endOnGameOver: true });
  await environment.reset({ task: "fixture", seed: 3 });
  await environment.reset({ task: "another", seed: 4 });
  assert.equal(worker.calls[0].options.operationMode, "COMPETITION");
  assert.equal(worker.calls[0].options.endOnGameOver, true);
  assert.equal(worker.calls.filter(call => call.op === "configure").length, 1);
  assert.deepEqual(worker.calls.filter(call => call.op === "reset").map(call => call.seed), [3, 4]);
  await environment.close();
  assert.equal(worker.calls.some(call => call.op === "scorecard"), false);
});

test("cancelling an ARC request stops its worker instead of replaying a partial turn", async () => {
  const worker = fixtureWorker();
  const environment = createArcAgi3Environment({ worker });
  await environment.reset({ task: "fixture" });
  const controller = new AbortController();
  let rejectRequest;
  worker.request = () => new Promise((resolve, reject) => { rejectRequest = reject; });
  let stopped = false;
  worker.stop = () => { stopped = true; rejectRequest?.(new Error("worker exited")); };
  worker.health = () => ({ running: !stopped });
  const pending = environment.step({ name: "ACTION6", arguments: { x: 0, y: 0 } }, { signal: controller.signal });
  controller.abort();
  await assert.rejects(pending, { name: "AbortError" });
  assert.equal(stopped, true);
  await environment.close();
});

test("an unavailable ARC Python interpreter returns a request error", async () => {
  const environment = createArcAgi3Environment({ python: "/nonexistent/arc-python", requestTimeoutMs: 500 });
  try {
    await assert.rejects(environment.reset({ task: "fixture" }), /arc-agi-3/u);
  } finally {
    await environment.close();
  }
});

test("official ARC SDK initializes and steps an offline coordinate fixture", {
  skip: !process.env.ARC_TEST_ENVIRONMENTS_DIR,
}, async () => {
  const environment = createArcAgi3Environment({
    environmentsDir: process.env.ARC_TEST_ENVIRONMENTS_DIR,
    recordingsDir: process.env.ARC_TEST_RECORDINGS_DIR ?? "/tmp/fast-arc-sdk-test-recordings",
  });
  try {
    const initial = await environment.reset({ task: "bt33", seed: 7 });
    assert.equal(initial.info.operationMode, "OFFLINE");
    assert.match(initial.info.sdkVersion, /^0\.9\./u);
    assert.ok(initial.observation.state.frame.length > 0);
    assert.equal(initial.observation.images.length, initial.observation.state.frame.length);
    assert.deepEqual(initial.observation.actions.map(action => action.name), ["ACTION6", "RESET"]);
    const card = await environment.scorecard();
    assert.equal(card.total_actions, 0);
    assert.equal(card.environments[0].resets, 0);
    const next = await environment.step({ name: "ACTION6", arguments: { x: 1, y: 1 } }, {
      observation: initial.observation,
      reasoning: { purpose: "offline integration test" },
    });
    assert.deepEqual(next.observation.state.action_input.data, { x: 1, y: 1 });
    assert.equal(next.observation.images[0].mimeType, "image/png");
    assert.equal(Buffer.from(next.observation.images[0].data, "base64").subarray(1, 4).toString(), "PNG");
    assert.equal(next.reward, null);
    const finalCard = await environment.scorecard({ close: true });
    assert.equal(finalCard.total_actions, 1);
    assert.equal(finalCard.environments[0].resets, 0);
  } finally {
    await environment.close();
  }
});

test("official ARC SDK recovers GAME_OVER with one budgeted RESET and then reaches WIN", {
  skip: !process.env.ARC_TEST_ENVIRONMENTS_DIR,
}, async context => {
  const directory = await mkdtemp(join(tmpdir(), "fast-arc-reset-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const gameDirectory = join(directory, "environment_files", "rset");
  await mkdir(gameDirectory, { recursive: true });
  await writeFile(join(gameDirectory, "metadata.json"), JSON.stringify({ game_id: "rset-fixture", title: "Reset fixture", class_name: "Rset", baseline_actions: [2], date_downloaded: "2024-01-01T00:00:00Z" }));
  await writeFile(join(gameDirectory, "rset.py"), `from arcengine import ARCBaseGame, Camera, GameAction, Level

class Rset(ARCBaseGame):
    def __init__(self):
        super().__init__(game_id="rset", levels=[Level(sprites=[], grid_size=(8, 8))], camera=Camera(background=5, letter_box=3), available_actions=[1, 2])

    def step(self):
        if self.action.id == GameAction.ACTION1:
            self.lose()
        elif self.action.id == GameAction.ACTION2:
            self.win()
        self.complete_action()
`);
  for (const endOnGameOver of [false, true]) {
    const environment = createArcAgi3Environment({ environmentsDir: join(directory, "environment_files"), recordingsDir: join(directory, "recordings"), endOnGameOver });
    const transitions = [];
    try {
      const summary = await runEpisode({
        environment,
        task: "rset",
        maxSteps: 3,
        policy: { id: "recovery-fixture", async decide({ history, actions }) {
          if (history.length === 1) assert.deepEqual(actions.map(action => action.name), ["RESET"]);
          return { action: { name: ["ACTION1", "RESET", "ACTION2"][history.length], arguments: {} } };
        } },
        onTransition: transition => transitions.push(transition),
      });
      assert.equal(summary.success, !endOnGameOver);
      assert.equal(summary.steps, endOnGameOver ? 1 : 3);
      const loss = transitions.find(transition => transition.type === "step");
      assert.equal(loss.info.gameOver, true);
      assert.equal(loss.terminated, endOnGameOver);
      assert.equal(loss.success, endOnGameOver ? false : null);
      const card = await environment.scorecard({ close: true });
      assert.equal(card.total_actions, endOnGameOver ? 1 : 3);
      assert.equal(card.environments[0].resets, endOnGameOver ? 0 : 1);
      if (!endOnGameOver) {
        assert.deepEqual(transitions.filter(transition => transition.type === "step").map(transition => transition.action.name), ["ACTION1", "RESET", "ACTION2"]);
        assert.equal(transitions[2].info.gameOver, false);
      }
    } finally {
      await environment.close();
    }
  }
});
