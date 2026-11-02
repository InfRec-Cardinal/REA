import test from "node:test";
import assert from "node:assert/strict";
import { createPi05Policy } from "../src/policies/pi05.mjs";

const observation = (value = 0.1) => ({
  text: "Place the cup in the tray.",
  state: { qpos: Array(14).fill(value) },
  images: ["cam_high", "cam_left_wrist", "cam_right_wrist"].map(name => ({ name, data: name, mimeType: "image/png" })),
});

function fixture(infer = () => ({ actions: [Array(14).fill(0.2)] })) {
  return {
    requests: [], stops: 0,
    stop() { this.stops += 1; },
    async request(payload) {
      this.requests.push(structuredClone(payload));
      return { ok: true, result: payload.op === "infer" ? infer(payload) : {} };
    },
  };
}

test("pi0.5 sends Classic ALOHA observations and returns native joint14 trajectories", async context => {
  const worker = fixture(() => ({ actions: [Array(14).fill(0.2), Array(14).fill(0.3)], server_timing: { infer_ms: 8 } }));
  const policy = createPi05Policy({ worker, env: { PI05_URL: "ws://pi05.example:8000" }, chunkSteps: 1 });
  context.after(() => policy.close());
  const decision = await policy.decide({ observation: observation() });
  assert.deepEqual(worker.requests.map(request => request.op), ["configure", "infer"]);
  assert.equal(worker.requests[0].url, "ws://pi05.example:8000");
  assert.deepEqual(worker.requests[1], {
    op: "infer", images: Object.fromEntries(observation().images.map(({ name, ...image }) => [name, image])),
    state: Array(14).fill(0.1), prompt: "Place the cup in the tray.",
  });
  assert.deepEqual(decision.action, { name: "joint_trajectory", arguments: { positions: [Array(14).fill(0.2)] } });
  assert.equal(decision.model, "pi05");
  assert.deepEqual(decision.vla, { controlSteps: 1, predictedSteps: 2, serverTiming: { infer_ms: 8 } });
});

test("pi0.5 rerouting and episode reset request fresh actions without a remote reset command", async context => {
  const worker = fixture(payload => ({ actions: [payload.state, Array(14).fill(0.9)] }));
  const policy = createPi05Policy({ worker, chunkSteps: 1 });
  context.after(() => policy.close());
  const first = await policy.decide({ observation: observation(0.1) });
  await policy.reset();
  const second = await policy.decide({ observation: observation(0.4), instruction: "Use the left arm." });
  assert.deepEqual(first.action.arguments.positions[0], Array(14).fill(0.1));
  assert.deepEqual(second.action.arguments.positions[0], Array(14).fill(0.4));
  assert.deepEqual(worker.requests.map(request => request.op), ["configure", "infer", "infer"]);
  assert.equal(worker.requests.at(-1).prompt, "Place the cup in the tray.\n\nUse the left arm.");
});

test("pi0.5 rejects incompatible observation and action dimensions", async context => {
  const worker = fixture(() => ({ actions: [Array(16).fill(0.2)] }));
  const policy = createPi05Policy({ worker });
  context.after(() => policy.close());
  await assert.rejects(policy.decide({ observation: { ...observation(), state: { qpos: [0.1] } } }), /14 measured joint positions/);
  assert.equal(worker.requests.length, 0);
  await assert.rejects(policy.decide({ observation: observation() }), /shape \[steps, 14\]/);
});
