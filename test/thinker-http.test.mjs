import assert from "node:assert/strict";
import test from "node:test";

import { createThinkerHttpServer } from "../src/thinker-http.mjs";
import { createThinkerRuntime } from "../src/thinker-runtime.mjs";

const silentLogger = { info() {}, warn() {}, error() {} };

async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  return `http://127.0.0.1:${server.address().port}`;
}

function job(seq = 1) {
  return {
    sessionId: "session-http",
    seq,
    instruction: "keep the aisle centered",
    token: "W",
    action: { action: "FORWARD", duration_ms: 500 },
    beforeFrame: {
      imageB64: Buffer.from("before").toString("base64"),
      mimeType: "image/jpeg",
    },
  };
}

test("thinker health is loopback advisory state and never invokes inference", async t => {
  let inferred = 0;
  const runtime = createThinkerRuntime({
    model: "reviewer",
    providerConfigured: true,
    infer: async () => {
      inferred += 1;
    },
    callback: async () => {},
    logger: silentLogger,
  });
  const server = createThinkerHttpServer({ runtime, logger: silentLogger });
  t.after(async () => {
    await new Promise(resolve => server.close(resolve));
    await runtime.close();
  });
  const base = await listen(server);

  const response = await fetch(`${base}/health`);
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.equal(body.ready, true);
  assert.equal(body.loopback_only, true);
  assert.equal(body.actuator_attached, false);
  assert.equal(body.concurrency, 1);
  assert.equal(body.latest_wins, true);
  assert.equal(inferred, 0);
});

test("POST /v1/review returns 202 before asynchronous inference completes", async t => {
  let release;
  const blocked = new Promise(resolve => {
    release = resolve;
  });
  const callbacks = [];
  const runtime = createThinkerRuntime({
    model: "reviewer",
    timeoutMs: 2000,
    infer: async () => {
      await blocked;
      return {
        review: {
          verdict: "KEEP",
          next_token: "W",
          brief: "The path remains centered and clear.",
          confidence: 0.8,
        },
      };
    },
    callback: async correction => callbacks.push(correction),
    logger: silentLogger,
  });
  const server = createThinkerHttpServer({ runtime, logger: silentLogger });
  t.after(async () => {
    release();
    await runtime.waitForIdle();
    await new Promise(resolve => server.close(resolve));
    await runtime.close();
  });
  const base = await listen(server);

  const response = await fetch(`${base}/v1/review`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(job(4)),
  });
  const body = await response.json();

  assert.equal(response.status, 202);
  assert.equal(body.accepted, true);
  assert.equal(body.asynchronous, true);
  assert.equal(body.sessionId, "session-http");
  assert.equal(body.seq, 4);
  assert.deepEqual(callbacks, []);

  release();
  await runtime.waitForIdle();
  assert.equal(callbacks.length, 1);
  assert.equal(callbacks[0].basedOnSeq, 4);
});

test("POST /v1/review rejects malformed JSON and invalid jobs", async t => {
  const runtime = createThinkerRuntime({
    model: "reviewer",
    infer: async () => {
      throw new Error("must not infer");
    },
    callback: async () => {},
    logger: silentLogger,
  });
  const server = createThinkerHttpServer({ runtime, logger: silentLogger });
  t.after(async () => {
    await new Promise(resolve => server.close(resolve));
    await runtime.close();
  });
  const base = await listen(server);

  const malformed = await fetch(`${base}/v1/review`, {
    method: "POST",
    body: "{",
  });
  assert.equal(malformed.status, 400);
  assert.equal((await malformed.json()).error_type, "request_error");

  const invalid = await fetch(`${base}/v1/review`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ...job(), beforeFrame: null }),
  });
  assert.equal(invalid.status, 400);
  assert.match((await invalid.json()).error, /beforeFrame/u);
});
