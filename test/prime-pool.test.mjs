import assert from "node:assert/strict";
import test from "node:test";

import { createPrimeAgentPool } from "../src/prime-pool.mjs";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function fakeFactory(calls, workers, closeCalls = []) {
  return index => {
    let ready = false;
    let closed = false;
    const infer = async model => {
      const gate = deferred();
      calls.push({ index, model, gate });
      return gate.promise;
    };
    infer.start = async () => {
      ready = true;
      return infer.state();
    };
    infer.close = async options => {
      closeCalls.push({ index, options });
      ready = false;
      closed = true;
    };
    infer.state = () => ({ ready, running: ready, status: closed ? "closed" : (ready ? "ready" : "stopped"), busy: false });
    workers.push(infer);
    return infer;
  };
}

test("keeps two Prime RPC processes hot and preserves two-way pipeline concurrency", async () => {
  const calls = [];
  const workers = [];
  const closeCalls = [];
  const pool = createPrimeAgentPool({ size: 2, createInfer: fakeFactory(calls, workers, closeCalls) });

  const state = await pool.start();
  assert.equal(state.ready, true);
  assert.equal(state.poolSize, 2);

  const first = pool("primary", { imageB64: "a" });
  const second = pool("fallback", { imageB64: "b" });
  const third = pool("primary", { imageB64: "c" });
  await new Promise(resolve => setImmediate(resolve));

  assert.deepEqual(calls.map(call => [call.index, call.model]), [[0, "primary"], [1, "fallback"]]);
  assert.equal(pool.state().busyWorkers, 2);
  assert.equal(pool.state().queuedRequests, 1);

  calls[0].gate.resolve({ token: "W" });
  assert.deepEqual(await first, { token: "W" });
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(calls.map(call => [call.index, call.model]), [[0, "primary"], [1, "fallback"], [0, "primary"]]);

  calls[1].gate.resolve({ token: "X" });
  calls[2].gate.resolve({ token: "Q" });
  assert.deepEqual(await second, { token: "X" });
  assert.deepEqual(await third, { token: "Q" });
  assert.equal(pool.state().busyWorkers, 0);

  await pool.close();
  assert.equal(pool.state().status, "closed");
  assert.equal(workers.length, 2);
  assert.deepEqual(closeCalls, [
    { index: 0, options: { shutdownDaemon: false } },
    { index: 1, options: { shutdownDaemon: true } },
  ]);
});

test("rejects a queued request when its decision deadline aborts", async () => {
  const calls = [];
  const workers = [];
  const pool = createPrimeAgentPool({ size: 1, createInfer: fakeFactory(calls, workers) });
  const first = pool("primary", { imageB64: "a" });
  await new Promise(resolve => setImmediate(resolve));

  const controller = new AbortController();
  const queued = pool("primary", { imageB64: "b" }, { signal: controller.signal });
  controller.abort();
  await assert.rejects(queued, error => error.errorType === "aborted");

  calls[0].gate.resolve({ token: "X" });
  await first;
  await pool.close();
});

test("a transient worker start failure can recover without closing the pool", async () => {
  let starts = 0;
  let closes = 0;
  let ready = false;
  const pool = createPrimeAgentPool({
    size: 1,
    createInfer() {
      const infer = async () => ({ token: "X" });
      infer.start = async () => {
        starts += 1;
        if (starts === 1) throw new Error("temporary start failure");
        ready = true;
        return infer.state();
      };
      infer.close = async () => {
        closes += 1;
        ready = false;
      };
      infer.state = () => ({ ready, running: ready, status: ready ? "ready" : "stopped", busy: false });
      return infer;
    },
  });

  await assert.rejects(pool.start(), /temporary start failure/u);
  assert.equal(closes, 0, "a failed start must not permanently close reusable adapters");
  assert.equal((await pool.start()).ready, true);
  assert.equal((await pool("primary", { imageB64: "a" })).token, "X");
  await pool.close();
  assert.equal(closes, 1);
});
