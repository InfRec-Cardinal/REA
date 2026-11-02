import assert from "node:assert/strict";
import { test } from "node:test";

import { createCorrectionCallback, createJudgementCallback, isLoopbackHost, loadThinkerConfig } from "../thinker.mjs";

const BASE = { REA_API_KEY: "k" };

test("thinker config: the judge channel defaults to loopback, an empty string disables it, the budget fits a brief and a lesson", () => {
  const config = loadThinkerConfig({ ...BASE });
  assert.equal(config.judgeUrl, "http://127.0.0.1:8091/internal/judge/review");
  assert.equal(config.callbackUrl, "http://127.0.0.1:8091/internal/thinker/correction");
  assert.equal(config.maxOutputTokens, 320, "brief (200) + lesson (200) + enums must fit; 160 could truncate the correction away");
  assert.equal(loadThinkerConfig({ ...BASE, G1_THINKER_JUDGE_URL: "" }).judgeUrl, null, "steer only");
  assert.equal(loadThinkerConfig({ ...BASE, G1_THINKER_JUDGE_URL: "http://[::1]:8091/internal/judge/review" }).judgeUrl, "http://[::1]:8091/internal/judge/review");
  assert.equal(loadThinkerConfig({ ...BASE, G1_THINKER_HOST: "::1" }).host, "::1");
  assert.throws(() => loadThinkerConfig({ ...BASE, G1_THINKER_HOST: "0.0.0.0" }), /loopback/u);
  assert.equal(loadThinkerConfig({}).providerConfigured, false);
});

test("loopback checks accept IPv4, IPv6 (bracketed or not) and localhost only", () => {
  for (const host of ["127.0.0.1", "::1", "[::1]", "localhost", "LOCALHOST"]) assert.equal(isLoopbackHost(host), true, host);
  for (const host of ["0.0.0.0", "192.0.2.164", "[fe80::1]", "example.test", "", null]) assert.equal(isLoopbackHost(host), false, String(host));
});

test("the judgement callback posts to loopback only, treats non-2xx as a dropped lesson, and never touches the correction path", async () => {
  assert.throws(() => createJudgementCallback({ url: "http://192.0.2.10:8091/internal/judge/review" }), /loopback/u);
  assert.throws(() => createCorrectionCallback({ url: "http://192.0.2.10:8091/internal/thinker/correction" }), /loopback/u);
  createCorrectionCallback({ url: "http://[::1]:8091/internal/thinker/correction", fetchFn: async () => new Response("{}", { status: 202 }) });

  const posted = [];
  const judge = createJudgementCallback({
    url: "http://[::1]:8091/internal/judge/review",
    timeoutMs: 500,
    fetchFn: async (url, init) => {
      posted.push({ url: String(url), body: JSON.parse(init.body) });
      return posted.length === 1
        ? new Response(JSON.stringify({ ok: true, accepted: true }), { status: 202 })
        : new Response(JSON.stringify({ ok: false, reason: "unknown_session" }), { status: 404 });
    },
  });
  const judgement = { sessionId: "s1", basedOnSeq: 3, token: "W", verdict: "CORRECT", next_token: "Q", confidence: 0.8, outcome: "no_progress", lesson: "turn first" };
  await judge(judgement);
  assert.equal(posted[0].url, "http://[::1]:8091/internal/judge/review");
  assert.deepEqual(posted[0].body, judgement);
  await assert.rejects(judge(judgement), /returned 404/u, "a session the brain no longer holds is a dropped lesson, surfaced to the runtime as a judgement failure");
});
