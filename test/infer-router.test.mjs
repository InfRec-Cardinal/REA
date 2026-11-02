import assert from "node:assert/strict";
import { test } from "node:test";

import { buildRouterPrompt, chooseWithExploration, createModelRouter, eligibleCandidates, parseRouterAnswer } from "../src/infer-router.mjs";
import { createRouter } from "../src/router.mjs";
import { parseCandidateSpec } from "../src/candidates.mjs";

const ENGINES = ["token", "react", "cli"];
const CONFIG = { primaryModel: "deepseek-v4-flash-vision-exp", reactModel: "deepseek-v4-pro", primaryTimeoutMs: 6000, reactTimeoutMs: 8000, cliTimeoutMs: 60000 };
const SPEC = "token@deepseek-v4-flash-vision-exp:reflex, token@kimi-k2.5:deliberate:15000, cli@claude-fable-5-1:deliberate:60000";

function candidates() {
  const list = parseCandidateSpec(SPEC, CONFIG, ENGINES);
  return { list, ids: list.map(c => c.id), reflex: list.filter(c => c.tier === "reflex").map(c => c.id), deliberate: list.filter(c => c.tier === "deliberate").map(c => c.id), byId: new Map(list.map(c => [c.id, c])) };
}

test("slow candidates are eligible only within the step budget, or when the robot is stopped", () => {
  const c = candidates();
  const none = eligibleCandidates(c.list, {}, { stepBudgetMs: 10000 });
  assert.deepEqual(none.map(x => x.id), ["token@deepseek-v4-flash-vision-exp", "token@kimi-k2.5"], "a 15 s timeout guesses 7.5 s, inside the budget; the 60 s CLI does not fit");
  const measured = eligibleCandidates(c.list, { "token@kimi-k2.5": { latency_ema_s: 12 } }, { stepBudgetMs: 10000 });
  assert.deepEqual(measured.map(x => x.id), ["token@deepseek-v4-flash-vision-exp"], "a measured 12 s latency is over budget");
  const stopped = eligibleCandidates(c.list, { "token@kimi-k2.5": { latency_ema_s: 12 } }, { stepBudgetMs: 10000, stopped: true });
  assert.equal(stopped.length, 3, "a stopped robot may wait for anyone");
});

test("the prompt lists candidates with price, latency and per-dimension quality, and the answer parses robustly", () => {
  const c = candidates();
  const prompt = buildRouterPrompt({ candidates: c.list, statsById: { "token@kimi-k2.5": { latency_ema_s: 6.1, calls: 3, dimensions: { "near/clear": { ema: 0.4, n: 3 } } } }, dimension: "near/clear", situation: { center: "near", left: "far", right: "mid", lidar: "clear" }, signals: { refusalStreak: 2, oscillation: 0, stall: false, lastToken: "W" }, stepBudgetMs: 10000, lastStep: { candidate: "token@deepseek-v4-flash-vision-exp", token: "W", outcome: "refused", score: -0.6, latencyS: 1.5, refused: true }, memoryLines: ["centre near: W was vetoed; prefer Q"] });
  assert.match(prompt, /token@kimi-k2\.5 \| deliberate \| kimi-k2\.5 \| 0\.6,3 \| 6\.10 \| 3 \| 0\.40 \(n=3\)/u);
  assert.match(prompt, /refusal streak 2/u);
  assert.match(prompt, /refused by the gate/u);
  assert.match(prompt, /prefer Q/u);
  assert.deepEqual(parseRouterAnswer('{"candidate": "token@kimi-k2.5", "reasoning": "reflex refused twice"}', c.ids), { candidate: "token@kimi-k2.5", reasoning: "reflex refused twice" });
  assert.deepEqual(parseRouterAnswer('Sure: ```json\n{"model":"cli@claude-fable-5-1","reasoning":"stopped"}\n```', c.ids).candidate, "cli@claude-fable-5-1");
  assert.equal(parseRouterAnswer("I would use token@deepseek-v4-flash-vision-exp here", c.ids).candidate, "token@deepseek-v4-flash-vision-exp");
  assert.equal(parseRouterAnswer('{"candidate": "gpt-9"}', c.ids), null);
  assert.equal(parseRouterAnswer("", c.ids), null);
});

test("exploration tries under-sampled candidates first, then epsilon, and otherwise keeps the choice", () => {
  const c = candidates();
  const stats = { "token@deepseek-v4-flash-vision-exp": { dimensions: { "near/clear": { n: 4 } } }, "token@kimi-k2.5": { dimensions: { "near/clear": { n: 0 } } } };
  const first = chooseWithExploration({ chosen: "token@deepseek-v4-flash-vision-exp", eligible: c.list.slice(0, 2), statsById: stats, dimension: "near/clear", minTries: 1, epsilon: 0, rng: () => 0 });
  assert.deepEqual(first, { candidate: "token@kimi-k2.5", reason: "explore:under_sampled", explored: true });
  const sampled = { ...stats, "token@kimi-k2.5": { dimensions: { "near/clear": { n: 1 } } } };
  const keep = chooseWithExploration({ chosen: "token@deepseek-v4-flash-vision-exp", eligible: c.list.slice(0, 2), statsById: sampled, dimension: "near/clear", minTries: 1, epsilon: 0.05, rng: () => 0.5 });
  assert.equal(keep.explored, false);
  const eps = chooseWithExploration({ chosen: "token@deepseek-v4-flash-vision-exp", eligible: c.list.slice(0, 2), statsById: sampled, dimension: "near/clear", minTries: 1, epsilon: 0.05, rng: () => 0.01 });
  assert.equal(eps.reason, "explore:epsilon");
});

test("suggestNext asks the router model off the control path and hands the router a suggestion for the next step", async () => {
  const c = candidates();
  const router = createRouter({ cheapChain: c.reflex, escalateTo: c.deliberate[0], engines: c.ids, candidates: c.list, mode: "hybrid", k: 2 });
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push(JSON.parse(init.body));
    return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content: '{"candidate":"token@kimi-k2.5","reasoning":"two refusals, try the slower model"}' } }], usage: { prompt_tokens: 300, completion_tokens: 20 } }) };
  };
  const mr = createModelRouter({ apiKey: "k", baseUrl: "https://x/v1", model: "deepseek-v4-flash", router, candidates: c, fetchImpl, epsilon: 0, minTries: 0, stepBudgetMs: 10000 });
  const out = await mr.suggestNext({ id: "s1" }, { seq: 4, dimension: "near/clear", situation: { center: "near", lidar: "clear" }, signals: { refusalStreak: 2, lastToken: "W" } });
  assert.equal(out.status, "suggested");
  assert.equal(out.candidate, "token@kimi-k2.5");
  assert.equal(calls[0].model, "deepseek-v4-flash");
  assert.equal(calls[0].thinking.type, "disabled");
  assert.ok(calls[0].messages[1].content.includes("token@kimi-k2.5"));
  // the suggestion is honoured for exactly step 4 of session s1 in hybrid mode with no safety trigger
  const route = router.route({ signals: { refusalStreak: 0 }, situation: { center: "near", lidar: "clear" }, seq: 4, sessionId: "s1" });
  assert.equal(route.engine, "token@kimi-k2.5");
  assert.equal(route.reason, "model");
  assert.equal(route.suggested, true);
  assert.equal(route.escalated, true, "kimi is a deliberate-tier candidate");
  assert.equal(router.state().suggestions.used, 1);
  assert.equal(mr.stats().suggested, 1);
  assert.equal(mr.stats().usage.prompt_tokens, 300);
});

test("a failed or unparseable router model leaves the rule router alone, and busy sessions are skipped", async () => {
  const c = candidates();
  const router = createRouter({ cheapChain: c.reflex, escalateTo: c.deliberate[0], engines: c.ids, candidates: c.list, mode: "hybrid" });
  let mode = "garbage";
  const fetchImpl = async () => {
    if (mode === "fail") return { ok: false, status: 500, json: async () => ({}) };
    return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content: "no idea" } }] }) };
  };
  const mr = createModelRouter({ apiKey: "k", baseUrl: "https://x/v1", router, candidates: c, fetchImpl, epsilon: 0, minTries: 0 });
  assert.equal((await mr.suggestNext({ id: "s" }, { seq: 2, dimension: "far/clear", signals: { lastToken: "W" } })).status, "no_answer");
  mode = "fail";
  assert.equal((await mr.suggestNext({ id: "s" }, { seq: 3, dimension: "far/clear", signals: { lastToken: "W" } })).status, "no_answer");
  assert.equal(mr.stats().failed, 1);
  assert.equal(mr.stats().parse_failed, 1);
  assert.equal(router.route({ signals: { refusalStreak: 0 }, situation: { center: "far", lidar: "clear" }, seq: 3, sessionId: "s" }).reason, "default");
  // concurrency: a second call for the same session while one is in flight is skipped
  let release;
  const slow = createModelRouter({ apiKey: "k", baseUrl: "https://x/v1", router, candidates: c, fetchImpl: () => new Promise(resolve => { release = () => resolve({ ok: true, status: 200, json: async () => ({ choices: [{ message: { content: '{"candidate":"token@deepseek-v4-flash-vision-exp"}' } }] }) }); }), epsilon: 0, minTries: 0 });
  const first = slow.suggestNext({ id: "t" }, { seq: 1, dimension: "far/clear", signals: { lastToken: "W" } });
  const second = await slow.suggestNext({ id: "t" }, { seq: 1, dimension: "far/clear", signals: { lastToken: "W" } });
  assert.equal(second.status, "skipped_busy");
  release();
  assert.equal((await first).status, "suggested");
});
