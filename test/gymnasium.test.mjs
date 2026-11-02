import assert from "node:assert/strict";
import { test } from "node:test";

import { createGymnasiumEnvironment } from "../src/environments/gymnasium.mjs";
import { validateAction } from "../src/framework/actions.mjs";

function fixtureWorker() {
  let observation = null;
  let sequence = 0;
  let stopped = false;
  const calls = [];
  return {
    calls,
    health: () => ({ running: !stopped }),
    stop() { stopped = true; },
    async request(payload) {
      calls.push(payload);
      if (payload.op === "configure") return { ok: true, result: { sdkVersion: "1.3.0" } };
      if (payload.op === "close") return { ok: true, result: { closed: true } };
      if (payload.op === "observe") return { ok: true, result: observation };
      if (payload.op === "step" && payload.observationId !== observation.id) return { ok: false, error: "Action observation is stale" };
      observation = { id: `gym:1:${++sequence}`, timestampMs: Date.now(), state: { observation: [0.1, 0.2], info: {} }, images: [], actions: [] };
      return { ok: true, result: { observation, reward: payload.op === "step" ? 0.5 : null, terminated: false, truncated: false, success: null, info: {} } };
    },
  };
}

test("Gym adapter forwards native tasks and parameters through one configured worker", async () => {
  const worker = fixtureWorker();
  const env = createGymnasiumEnvironment({ worker, environmentId: "Fixture-v1", makeOptions: { render_mode: "rgb_array" }, modules: ["custom.registration"] });
  assert.equal(env.id, "gymnasium");
  assert.equal(env.timing, "turn-based");
  const task = { environmentId: "Other-v1", makeOptions: { max_episode_steps: 2 }, resetOptions: { option: 1 } };
  const initial = await env.reset({ task, seed: 8 });
  assert.deepEqual(worker.calls[0].options, { environmentId: "Fixture-v1", makeOptions: { render_mode: "rgb_array" }, modules: ["custom.registration"] });
  assert.deepEqual(worker.calls[1], { op: "reset", task, seed: 8 });
  assert.equal((await env.observe()).id, initial.observation.id);
  const action = { name: "step", arguments: { action: [0.1, -0.2] } };
  const next = await env.step(action, { observation: initial.observation });
  assert.deepEqual(worker.calls.at(-1), { op: "step", action, observationId: initial.observation.id });
  assert.equal(next.reward, 0.5);
  assert.equal(next.success, null);
  await assert.rejects(env.step(action, { observation: initial.observation }), /stale/u);
  await env.reset({ seed: 9 });
  assert.equal(worker.calls.filter(call => call.op === "configure").length, 1);
  await env.close();
  assert.equal(worker.calls.at(-1).op, "close");
  assert.equal(worker.health().running, false);
  await env.close();
  await assert.rejects(env.observe(), /closed/u);
});

test("cancelling an in-flight Gym action stops its worker", async () => {
  const worker = fixtureWorker();
  const env = createGymnasiumEnvironment({ worker, environmentId: "Fixture-v1" });
  await env.reset();
  let rejectRequest;
  worker.request = () => new Promise((resolve, reject) => { rejectRequest = reject; });
  let stopped = false;
  worker.stop = () => { stopped = true; rejectRequest?.(new Error("worker exited")); };
  worker.health = () => ({ running: !stopped });
  const controller = new AbortController();
  const pending = env.step({ name: "step", arguments: { action: 0 } }, { signal: controller.signal });
  controller.abort();
  await assert.rejects(pending, { name: "AbortError" });
  assert.equal(stopped, true);
  await env.close();
});

test("Gym reports an unavailable interpreter without starting a replacement environment", async () => {
  const env = createGymnasiumEnvironment({ python: "/nonexistent/gym-python", environmentId: "CartPole-v1", requestTimeoutMs: 500 });
  try {
    await assert.rejects(env.reset(), /gymnasium/u);
  } finally {
    await env.close();
  }
});

for (const environmentId of ["CartPole-v1", "Pendulum-v1"]) {
  test(`installed Gymnasium runs ${environmentId} through the bridge`, { skip: !process.env.GYM_TEST_SDK }, async () => {
    const env = createGymnasiumEnvironment({ environmentId, makeOptions: { max_episode_steps: 2 } });
    try {
      const initial = await env.reset({ seed: 42 });
      assert.equal(initial.reward, null);
      assert.equal(initial.success, null);
      assert.equal(initial.observation.images.length, 0);
      assert.equal(initial.observation.state.observation.length, environmentId === "CartPole-v1" ? 4 : 3);
      const action = { name: "step", arguments: { action: environmentId === "CartPole-v1" ? 0 : [0] } };
      validateAction(action, initial.observation.actions);
      const invalid = { name: "step", arguments: { action: environmentId === "CartPole-v1" ? 2 : [3] } };
      assert.throws(() => validateAction(invalid, initial.observation.actions), /must be/u);
      await assert.rejects(env.step(invalid), /bounds/u);
      const next = await env.step(action);
      assert.equal(Number.isFinite(next.reward), true);
      assert.equal(next.truncated, false);
      const terminal = await env.step(action);
      assert.equal(terminal.truncated, true);
      assert.equal(terminal.terminated, false);
      assert.equal(terminal.success, null);
      await assert.rejects(env.step(action), /Episode finished/u);
      const restarted = await env.reset({ seed: 42 });
      assert.deepEqual(restarted.observation.state.observation, initial.observation.state.observation);
    } finally {
      await env.close();
    }
  });
}
