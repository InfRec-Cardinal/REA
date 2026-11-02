import assert from "node:assert/strict";
import { test } from "node:test";

import { DecisionPipeline } from "../src/pipeline.mjs";

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

test("a decision that fails fast backs off instead of spinning the loop", async () => {
  let calls = 0;
  const pipeline = new DecisionPipeline({ decide: async () => { calls += 1; return { result: { ok: false, errorType: "capture_error" }, frame: null }; }, logger: null });
  pipeline.backoffMaxMs = 200;
  pipeline.set({ on: true, concurrency: 1, instruction: "go" });
  await sleep(350);
  pipeline.stop();
  const status = pipeline.status();
  assert.ok(calls <= 6, `a failing decide() ran ${calls} times in 350 ms; without backoff it would be thousands`);
  assert.ok(status.consecutive_failures >= 2);
  assert.equal(status.latest_token, null, "a failed decision never becomes the latest one");
  await sleep(250);
});

test("a successful decision resets the backoff and becomes the freshest decision", async () => {
  let ok = false;
  let calls = 0;
  const pipeline = new DecisionPipeline({ decide: async () => { calls += 1; return ok ? { result: { ok: true, token: "W" }, frame: {} } : { result: { ok: false }, frame: null }; }, logger: null });
  pipeline.backoffMaxMs = 100;
  pipeline.set({ on: true, concurrency: 1 });
  await sleep(120);
  assert.ok(pipeline.status().consecutive_failures >= 1);
  ok = true;
  await sleep(150);
  assert.equal(pipeline.status().consecutive_failures, 0);
  assert.equal(pipeline.status().latest_token, "W");
  assert.ok(pipeline.take(5)?.result.ok, "the successful decision is fresh");
  pipeline.stop();
  await sleep(50);
});
