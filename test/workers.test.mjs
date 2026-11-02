import assert from "node:assert/strict";
import { test } from "node:test";

import { LineWorker } from "../src/workers.mjs";

test("a worker whose command does not exist fails its requests instead of crashing the process", async () => {
  const worker = new LineWorker("ghost", "/nonexistent/interpreter-that-is-not-there", ["x.py"], { autoRestart: false, readyTimeoutMs: 20 });
  await worker.ready;
  await new Promise(resolve => setTimeout(resolve, 30));
  await assert.rejects(worker.request({ op: "capture" }, 500), /ghost/u);
  const health = worker.health();
  assert.equal(health.running, false);
  assert.match(health.last_error, /failed to start|exited/u);
  worker.stop();
});

test("a worker that exits mid-request rejects the pending request and restarts with backoff", async () => {
  const worker = new LineWorker("flaky", process.execPath, ["-e", 'process.stdout.write(JSON.stringify({event:"ready"})+"\\n"); process.stdin.once("data", () => process.exit(3));'], { readyTimeoutMs: 2000 });
  await worker.ready;
  await assert.rejects(worker.request({ op: "x" }, 2000), /flaky exited \(3\)/u);
  await new Promise(resolve => setTimeout(resolve, 400));
  assert.equal(worker.health().restarts, 1);
  worker.stop();
});

test("code workers start without the parent's credentials", async () => {
  const { codeWorkerEnv } = await import("../src/workers.mjs");
  const env = codeWorkerEnv({ PATH: "/usr/bin", PYTHONPATH: "/opt/site", REA_API_KEY: "secret", OPENAI_API_KEY: "secret", AWS_SECRET_ACCESS_KEY: "secret" });
  assert.deepEqual(env, { PATH: "/usr/bin", PYTHONPATH: "/opt/site" });
});
