import assert from "node:assert/strict";
import { test } from "node:test";

import { createRouter } from "../src/router.mjs";
import HOLONOMIC from "./fixtures/profiles/holonomic.mjs";

const CALM = { refusalStreak: 0, badStreak: 0, oscillation: 0, stall: false, lastReview: null, lastToken: "W" };
const FAR = { center: "far", left: "far", right: "far", lidar: "clear" };
const NEAR = { center: "near", left: "far", right: "mid", lidar: "clear" };

test("the cheap chain is the default and an explicit engine always wins", () => {
  const router = createRouter({ cheapChain: ["prime"], escalateTo: "react" });
  const routed = router.route({ signals: CALM, situation: FAR });
  assert.equal(routed.engine, "prime");
  assert.equal(routed.reason, "default");
  assert.equal(routed.escalated, false);
  assert.equal(routed.dimension, "far/clear");
  assert.deepEqual(router.route({ signals: { ...CALM, refusalStreak: 5 }, situation: FAR, explicit: "token" }), {
    engine: "token",
    reason: "explicit",
    escalated: false,
    triggers: [],
    dimension: "far/clear",
  });
  assert.equal(router.route({ signals: CALM, situation: FAR, explicit: "auto" }).engine, "prime");
});

test("k refusals escalate to the deliberate engine and hold it for the cooldown", () => {
  const router = createRouter({ cheapChain: ["prime"], escalateTo: "react", k: 2, cooldown: 2 });
  assert.equal(router.route({ signals: { ...CALM, refusalStreak: 1 }, situation: FAR }).engine, "prime");
  const escalated = router.route({ signals: { ...CALM, refusalStreak: 2 }, situation: FAR });
  assert.equal(escalated.engine, "react");
  assert.equal(escalated.reason, "refusal_streak");
  assert.equal(escalated.escalated, true);
  router.commit("react");
  const held = router.route({ signals: CALM, situation: FAR });
  assert.equal(held.engine, "react");
  assert.equal(held.reason, "cooldown");
  router.commit("react");
  assert.equal(router.route({ signals: CALM, situation: FAR }).engine, "prime", "cooldown expired");
  assert.equal(router.state().escalations, 1);
  assert.equal(router.state().deliberate, 2);
});

test("every escalation trigger is recognised and reported", () => {
  const router = createRouter({ k: 2, cooldown: 0, maxDeliberateFraction: 1 });
  const cases = [
    [{ ...CALM, badStreak: 2 }, FAR, "bad_streak"],
    [{ ...CALM, oscillation: 2 }, FAR, "oscillation"],
    [{ ...CALM, stall: true }, FAR, "stall"],
    [{ ...CALM, lastReview: "CORRECT" }, FAR, "reviewer_flag"],
    [{ ...CALM, lastReview: "STOP" }, FAR, "reviewer_flag"],
    [{ ...CALM, lastToken: "W" }, NEAR, "risk_near"],
    [{ ...CALM, recentRevisits: 2 }, FAR, "revisit"],
    [{ ...CALM, revisits: 1, refusedHere: 1 }, FAR, "revisit"],
  ];
  for (const [signals, situation, expected] of cases) {
    const routed = router.route({ signals, situation });
    assert.equal(routed.engine, "react", expected);
    assert.equal(routed.reason, expected);
    router.commit("react");
    router.commit("prime");
  }
  assert.equal(router.route({ signals: { ...CALM, lastToken: "X" }, situation: NEAR }).engine, "prime", "near but already stopped is not risky");
  assert.equal(router.route({ signals: { ...CALM, revisits: 1, refusedHere: 0 }, situation: FAR }).engine, "prime", "one uneventful revisit is not a loop");
});

test("the deliberate budget caps escalation and reports it", () => {
  const router = createRouter({ k: 1, cooldown: 0, maxDeliberateFraction: 0.25 });
  for (let i = 0; i < 8; i += 1) router.commit("react");
  const routed = router.route({ signals: { ...CALM, refusalStreak: 3 }, situation: FAR });
  assert.equal(routed.engine, "prime");
  assert.equal(routed.reason, "budget_exhausted");
  assert.deepEqual(routed.triggers, ["refusal_streak"]);
  for (let i = 0; i < 40; i += 1) router.commit("prime");
  assert.equal(router.route({ signals: { ...CALM, refusalStreak: 3 }, situation: FAR }).engine, "react", "budget recovers");
});

test("routing memory prefers an engine that keeps scoring better in a dimension", () => {
  const router = createRouter({ cheapChain: ["prime"], escalateTo: "react", cooldown: 0, memoryMinSamples: 3, memoryMargin: 0.25 });
  for (let i = 0; i < 3; i += 1) {
    router.observe("near/clear", "prime", -0.6);
    router.observe("near/clear", "react", 0.3);
  }
  // a stopped last token keeps risk_near quiet, so the memory preference is what decides
  const learned = router.route({ signals: { ...CALM, lastToken: "X" }, situation: NEAR });
  assert.equal(learned.engine, "react");
  assert.equal(learned.reason, "memory");
  assert.equal(router.route({ signals: CALM, situation: NEAR }).reason, "risk_near", "triggers still take precedence over memory");
  assert.equal(router.route({ signals: { ...CALM, lastToken: "X" }, situation: FAR }).engine, "prime", "other dimensions are untouched");
  const snapshot = router.state();
  assert.equal(snapshot.scores["near/clear"].react.n, 3);
  assert.ok(snapshot.scores["near/clear"].react.ema > 0);

  const thin = createRouter({ memoryMinSamples: 3 });
  thin.observe("far/clear", "react", 1);
  assert.equal(thin.route({ signals: { ...CALM, lastToken: "X" }, situation: FAR }).engine, "prime", "one sample is not evidence");
});

test("inline verify-and-escalate only fires for a vetoed reflex token with time left", () => {
  const router = createRouter({ inlineMinDeadlineMs: 1500 });
  const vetoed = { allowed: false, reasons: ["depth:depth_center_near"] };
  assert.equal(router.shouldEscalateInline({ engine: "prime", token: "W", gate: vetoed, deadlineLeftMs: 4000 }), true);
  assert.equal(router.shouldEscalateInline({ engine: "prime", token: "W", gate: vetoed, deadlineLeftMs: 1000 }), false, "not enough time");
  assert.equal(router.shouldEscalateInline({ engine: "react", token: "W", gate: vetoed, deadlineLeftMs: 4000 }), false, "already deliberate");
  assert.equal(router.shouldEscalateInline({ engine: "prime", token: "X", gate: vetoed, deadlineLeftMs: 4000 }), false, "a stop is never escalated");
  assert.equal(router.shouldEscalateInline({ engine: "prime", token: "W", gate: { allowed: true, reasons: [] }, deadlineLeftMs: 4000 }), false);
  const off = createRouter({ inlineEscalate: false });
  assert.equal(off.shouldEscalateInline({ engine: "prime", token: "W", gate: vetoed, deadlineLeftMs: 4000 }), false);
  router.commit("react", { inline: true });
  assert.equal(router.state().inline_escalations, 1);
});

test("an unavailable escalation engine degrades to the cheap chain", () => {
  const router = createRouter({ cheapChain: ["prime", "token"], escalateTo: "react", engines: ["token"] });
  const routed = router.route({ signals: { ...CALM, refusalStreak: 5 }, situation: FAR });
  assert.equal(routed.engine, "token", "first available cheap engine");
  assert.equal(routed.escalated, false);
  assert.equal(router.shouldEscalateInline({ engine: "token", token: "W", gate: { allowed: false, reasons: [] }, deadlineLeftMs: 5000 }), false);
  assert.throws(() => createRouter({ cheapChain: [] }));
  assert.throws(() => createRouter({ k: 0 }));
});

test("routing memory survives a restart through its JSON file", async () => {
  const { mkdtemp, readFile, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dir = await mkdtemp(join(tmpdir(), "nea-router-"));
  const path = join(dir, "nested", "router.json");
  try {
    const router = createRouter({ cheapChain: ["token"], escalateTo: "react", path, now: () => 42 });
    assert.deepEqual(await router.load(), { loaded: 0 }, "a missing file is an empty memory, not an error");
    router.observe("near/clear", "token", -0.6);
    router.observe("near/clear", "token", -0.4);
    router.observe("near/clear", "react", 0.3);
    await router.flush();
    const document = JSON.parse(await readFile(path, "utf8"));
    assert.equal(document.version, 1);
    assert.equal(document.saved_at, 42);
    assert.equal(document.observations, 3);
    assert.equal(document.scores["near/clear"].token.n, 2);

    const reloaded = createRouter({ cheapChain: ["token"], escalateTo: "react", path, memoryMinSamples: 2, memoryMargin: 0.25 });
    assert.deepEqual(await reloaded.load(), { loaded: 2 });
    const state = reloaded.state();
    assert.equal(state.loaded, true);
    assert.equal(state.memory_path, path);
    assert.equal(state.observations, 3);
    assert.equal(state.scores["near/clear"].token.n, 2);
    assert.ok(Math.abs(state.scores["near/clear"].token.ema - (0.7 * -0.6 + 0.3 * -0.4)) < 1e-3);
    // one more react sample crosses memoryMinSamples and the restored baseline makes the preference decide
    reloaded.observe("near/clear", "react", 0.3);
    await reloaded.flush();
    assert.equal(reloaded.route({ signals: { ...CALM, lastToken: "X" }, situation: NEAR }).reason, "memory");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("restore() ignores malformed entries and has() reports engine availability", () => {
  const router = createRouter({ cheapChain: ["token"], escalateTo: "react", engines: ["token", "react"] });
  assert.equal(router.restore({ "far/clear": { token: { ema: "x", n: 1 }, react: { ema: 0.5, n: 0 }, prime: { ema: 2, n: 3 } } }), 1);
  assert.equal(router.state().scores["far/clear"].prime.ema, 1, "restored scores are clamped into [-1, 1]");
  assert.equal(router.restore(null), 0);
  assert.equal(router.has("token"), true);
  assert.equal(router.has("forced"), false);
  const open = createRouter({ cheapChain: ["token"] });
  assert.equal(open.has("anything"), true, "without an engine list every engine is assumed available");
});

test("candidate tiers drive the budget, and suggestions are honoured per mode", () => {
  const candidates = [
    { id: "token@fast", tier: "reflex" },
    { id: "token@slow", tier: "deliberate" },
    { id: "react", tier: "deliberate" },
  ];
  const make = mode => createRouter({ cheapChain: ["token@fast"], escalateTo: "react", engines: candidates.map(c => c.id), candidates, mode, k: 2, cooldown: 0, maxDeliberateFraction: 0.5 });
  const near = { center: "near", lidar: "clear" };
  // rule mode ignores suggestions entirely
  const rule = make("rule");
  rule.suggest({ candidate: "token@slow", forSeq: 1, sessionId: "s" });
  assert.equal(rule.route({ signals: { refusalStreak: 0 }, situation: near, seq: 1, sessionId: "s" }).engine, "token@fast");
  // model mode takes a fresh suggestion, and only for its step
  const model = make("model");
  model.suggest({ candidate: "token@slow", forSeq: 2, sessionId: "s", reason: "model" });
  assert.equal(model.route({ signals: { refusalStreak: 0 }, situation: near, seq: 1, sessionId: "s" }).engine, "token@fast", "the suggestion is for step 2, not 1");
  assert.equal(model.state().suggestions.stale, 1);
  model.suggest({ candidate: "token@slow", forSeq: 2, sessionId: "s", reason: "model" });
  const taken = model.route({ signals: { refusalStreak: 0 }, situation: near, seq: 2, sessionId: "s" });
  assert.deepEqual({ engine: taken.engine, reason: taken.reason, escalated: taken.escalated }, { engine: "token@slow", reason: "model", escalated: true });
  model.commit("token@slow");
  assert.equal(model.state().deliberate, 1, "a deliberate-tier candidate counts against the budget whoever picked it");
  // hybrid: a safety trigger outranks the suggestion; without one the suggestion wins over the default
  const hybrid = make("hybrid");
  hybrid.suggest({ candidate: "token@slow", forSeq: 1, sessionId: "s" });
  const escalated = hybrid.route({ signals: { refusalStreak: 2 }, situation: near, seq: 1, sessionId: "s" });
  assert.deepEqual({ engine: escalated.engine, reason: escalated.reason }, { engine: "react", reason: "refusal_streak" });
  hybrid.commit("react");
  hybrid.suggest({ candidate: "token@slow", forSeq: 2, sessionId: "s" });
  assert.equal(hybrid.route({ signals: { refusalStreak: 0 }, situation: near, seq: 2, sessionId: "s" }).engine, "token@slow");
  // an unknown candidate or one over the deliberate budget is refused, and the router says so
  hybrid.suggest({ candidate: "ghost", forSeq: 3, sessionId: "s" });
  assert.equal(hybrid.route({ signals: { refusalStreak: 0 }, situation: near, seq: 3, sessionId: "s" }).engine, "token@fast");
  assert.equal(hybrid.state().suggestions.unavailable, 1);
  assert.throws(() => createRouter({ cheapChain: ["token"], escalateTo: "react", mode: "oracle" }), /mode/u);
});

test("observe() keeps per-candidate latency and cost statistics that survive a restart", async () => {
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dir = await mkdtemp(join(tmpdir(), "nea-router-"));
  try {
    const path = join(dir, "router.json");
    const router = createRouter({ cheapChain: ["token"], escalateTo: "react", path });
    await router.load();
    router.commit("token");
    router.observe("near/clear", "token", 0.3, { latencyS: 1.5, costUsd: 0.001 });
    router.observe("near/clear", "token", -0.6, { latencyS: 2.5, costUsd: 0.002 });
    const stats = router.candidateStats();
    assert.equal(stats.token.calls, 1);
    assert.equal(stats.token.n, 2);
    assert.ok(Math.abs(stats.token.latency_ema_s - 1.8) < 1e-6, "EMA with alpha 0.3");
    assert.equal(stats.token.cost_usd_total, 0.003);
    assert.equal(stats.token.dimensions["near/clear"].n, 2);
    assert.equal(stats.react.tier, "deliberate");
    await router.flush();
    const again = createRouter({ cheapChain: ["token"], escalateTo: "react", path });
    await again.load();
    assert.equal(again.candidateStats().token.cost_usd_total, 0.003);
    assert.ok(Math.abs(again.candidateStats().token.latency_ema_s - 1.8) < 1e-6);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("the risk trigger uses the profile's motion tokens", () => {
  const router = createRouter({ cheapChain: ["token"], escalateTo: "react", k: 2, cooldown: 0, profile: HOLONOMIC });
  const near = { center: "near", lidar: "clear" };
  const calm = { refusalStreak: 0, badStreak: 0, oscillation: 0, stall: false, lastReview: null };
  assert.equal(router.route({ signals: { ...calm, lastToken: "F" }, situation: near }).reason, "risk_near");
  assert.equal(router.route({ signals: { ...calm, lastToken: "W" }, situation: near }).reason, "default", "W is not motion in this profile");
});
