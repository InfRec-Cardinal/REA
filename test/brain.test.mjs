import assert from "node:assert/strict";
import { test } from "node:test";

import { createAgent } from "../src/agent.mjs";
import { createBrain } from "../src/brain.mjs";
import { createEpisodeStore } from "../src/episodes.mjs";
import { createExperienceMemory } from "../src/memory.mjs";
import { createRouter } from "../src/router.mjs";
import { createSessionStore } from "../src/sessions.mjs";

const NEAR = { center_m: 0.45, left_m: 2.2, right_m: 0.5 };
const FAR = { center_m: 3.0, left_m: 3.0, right_m: 3.0 };

const CONFIG = {
  engine: "token",
  providerConfigured: true,
  primaryModel: "reflex-model",
  fallbackModel: "",
  primaryTimeoutMs: 1000,
  fallbackTimeoutMs: 1000,
  maxOutputTokens: 64,
  promptVariant: "short",
  schema: "compact",
  maxWidth: 320,
  quality: 40,
  filter: "",
  includeDepth: true,
  imageDetail: "low",
  race: false,
  useLidar: false,
  executeEnabled: false,
  agentHarness: "rea-token",
  primeAgentVersion: null,
  brainStepBudgetMs: 10000,
  routerEscalateTo: "react",
  reactModel: "deliberate-model",
  reactTimeoutMs: 5000,
  reactMaxOutputTokens: 160,
  reactMaxIterations: 2,
};

function camera(frames) {
  let index = 0;
  return {
    capture: async () => {
      const depth = frames[Math.min(index, frames.length - 1)];
      index += 1;
      return { imageB64: "aGk=", mimeType: "image/jpeg", depth, width: 320, height: 240 };
    },
  };
}

/** Reflex engine always says W; the deliberate engine says Q and records what it was told. */
function engines({ reflexToken = "W", deliberateToken = "Q", deliberateNext = null } = {}) {
  const calls = { token: [], react: [] };
  const token = async (model, frame, options) => {
    calls.token.push({ model, options });
    return { token: reflexToken, tokenReadyS: 0.1 };
  };
  token.engine = "token";
  const react = async (model, frame, options) => {
    calls.react.push({ model, options });
    return { token: deliberateToken, thought: "left is open, turn left", next: deliberateNext, tokenReadyS: 0.8, source: "react", trace: [{ iteration: 1, action: "finish" }] };
  };
  react.engine = "react";
  return { engines: { token, react }, calls };
}

/** Gate: forward is vetoed when the centre is near; everything else passes. Never actuates. */
function executor() {
  const stops = [];
  return {
    stops,
    async execute({ token, frame, execute }) {
      const depth = frame?.depth ?? null;
      const vetoed = token === "W" && depth && depth.center_m < 0.6;
      const gate = {
        allowed: token === "X" || !vetoed,
        reasons: vetoed ? ["depth:depth_center_near"] : [],
        depth: depth ? { usable: true, blocked: vetoed, reason: vetoed ? "depth_center_near" : "clear", ...depth } : { usable: false, blocked: true, reason: "depth_missing" },
        lidar: { usable: true, blocked: false, reason: "lidar_disabled" },
        state: { ready: true, reasons: [] },
      };
      return {
        token,
        gate,
        executed: false,
        preview: !execute,
        dispatch: null,
        arm_held: null,
        motion: null,
        timing: { gate_s: 0.001, exec_s: null, total_s: 0.001 },
        ...(gate.allowed ? {} : { error_type: "gate_refused", error: gate.reasons.join("; ") }),
      };
    },
    async evidence({ frame }) {
      return { status: { ready: true, reasons: [] }, lidar: { usable: true, blocked: false, reason: "lidar_disabled" }, depth: frame?.depth ?? null };
    },
    async stop() {
      stops.push(Date.now());
      return { ok: true };
    },
    async status() {
      return { ok: true, ready: true, reasons: [] };
    },
    armState: () => ({ held: null, holding_tokens: ["U", "P"] }),
  };
}

function build({ frames, engineOptions = {}, routerOptions = {}, brain: withBrain = true, reflect = null } = {}) {
  const { engines: e, calls } = engines(engineOptions);
  const experience = createExperienceMemory({ now: () => 1_000_000 });
  const router = createRouter({ cheapChain: ["token"], escalateTo: "react", k: 2, cooldown: 1, engines: Object.keys(e), ...routerOptions });
  const brain = withBrain ? createBrain({ experience, router, logger: null, reflect }) : null;
  const sessions = createSessionStore({ idFactory: () => "s1" });
  const agent = createAgent({ camera: camera(frames), engines: e, executor: executor(), config: CONFIG, sessions, brain, logger: { info() {}, warn() {}, error() {} } });
  return { agent, brain, experience, router, calls, sessions };
}

test("a vetoed reflex token is escalated inline to the deliberate engine, re-gated, and remembered", async () => {
  const { agent, experience, router, calls } = build({ frames: [NEAR] });
  const out = await agent.step({ instruction: "reach the doorway" });

  assert.equal(out.ok, true);
  assert.equal(out.token, "Q", "the deliberate answer replaced the vetoed reflex token");
  assert.equal(out.engine, "react");
  assert.equal(out.reasoning, true);
  assert.equal(out.gate.allowed, true);
  assert.equal(out.brain.route, "default", "the reflex engine was routed first");
  assert.equal(out.brain.escalated, true);
  assert.deepEqual({ from: out.brain.inline.from, engine: out.brain.inline.engine, token: out.brain.inline.token, ok: out.brain.inline.ok }, { from: "W", engine: "react", token: "Q", ok: true });
  assert.equal(out.brain.thought, "left is open, turn left");
  assert.equal(out.brain.judge.outcome, "preview");

  assert.equal(calls.token.length, 1);
  assert.equal(calls.react.length, 1);
  assert.equal(calls.react[0].model, "deliberate-model", "the deliberate engine uses its own model");
  assert.equal(calls.react[0].options.cfg.primaryTimeoutMs, 5000);
  assert.ok(calls.react[0].options.context.observation.includes("vetoed W"));
  assert.ok(calls.react[0].options.context.observation.includes("depth_center_near"));
  assert.equal(typeof calls.react[0].options.context.observe, "function");
  assert.equal(typeof calls.react[0].options.context.recall, "function");

  const lessons = experience.all();
  assert.equal(lessons.length, 1, "the refused reflex attempt became a lesson");
  assert.equal(lessons[0].token, "W");
  assert.equal(lessons[0].outcome, "refused");
  assert.equal(lessons[0].alternative, "Q");
  assert.equal(lessons[0].taskKey, "reach-doorway");
  assert.equal(router.state().inline_escalations, 1);
  assert.equal(router.state().steps, 1);
  assert.ok(router.state().scores["near/clear"].token.ema < 0, "the reflex engine was marked down in this dimension");
});

test("lessons reach the reflex prompt and risky situations route straight to the deliberate engine", async () => {
  // Step 1 captures twice (reflex, then the inline deliberate re-decision), both near the obstacle.
  const { agent, calls } = build({ frames: [NEAR, NEAR, FAR, FAR] });
  const first = await agent.step({ instruction: "reach the doorway" });
  assert.equal(first.token, "Q");

  // Step 2: the last step ended near an obstacle with a motion token => risk_near escalates up front.
  const second = await agent.step({ session_id: first.session_id });
  assert.equal(second.seq, 2);
  assert.equal(second.brain.route, "risk_near");
  assert.equal(second.engine, "react");
  assert.equal(second.brain.inline, null, "no veto, so nothing to escalate inline");
  assert.equal(second.brain.lessons_used, 1, "the refusal lesson was recalled for the same task and situation");
  assert.ok(second.brain.experience.includes("W was vetoed"));
  assert.ok(calls.react[1].options.context.lessons.includes("prefer Q"));

  // Step 3: far away, cooldown expired => back to the reflex engine, which now sees the experience block.
  const third = await agent.step({ session_id: first.session_id });
  assert.equal(third.brain.route, "default");
  assert.equal(third.engine, "token");
  assert.equal(third.token, "W");
  assert.equal(third.gate.allowed, true);
  const reflexCall = calls.token.at(-1);
  assert.equal(typeof reflexCall.options.experience, "string");
  assert.ok(reflexCall.options.experience.startsWith("[-]"));
});

test("reviewer verdicts teach the brain even when they arrive too late to steer", async () => {
  const { agent, experience, sessions } = build({ frames: [FAR, FAR] });
  const first = await agent.step({ instruction: "reach the doorway" });
  await agent.step({ session_id: first.session_id });
  assert.equal(sessions.get("s1").seq, 2);

  const stale = agent.acceptCorrection({ sessionId: "s1", basedOnSeq: 1, verdict: "CORRECT", next_token: "Q", brief: "door is on the left", confidence: 0.9 });
  assert.equal(stale.accepted, false);
  assert.equal(stale.reason, "stale_correction");
  assert.equal(stale.learned.outcome, "review_correct");
  const reviewer = experience.all().filter(lesson => lesson.source === "reviewer");
  assert.equal(reviewer.length, 1);
  assert.equal(reviewer[0].token, "W");
  assert.equal(reviewer[0].alternative, "Q");
  assert.ok(reviewer[0].lesson.includes("door is on the left"));
  assert.equal(sessions.get("s1").episodic.get(1).review.verdict, "CORRECT");

  const fresh = agent.acceptCorrection({ sessionId: "s1", basedOnSeq: 2, verdict: "KEEP", next_token: "W", brief: "corridor clear", confidence: 0.7 });
  assert.equal(fresh.accepted, true);
  assert.equal(fresh.learned.outcome, "review_keep");
  const third = await agent.step({ session_id: first.session_id });
  assert.equal(third.advice_used.verdict, "KEEP", "a fresh correction still steers the next step exactly as before");
});

test("an explicit engine bypasses the router, and without a brain the step is unchanged", async () => {
  const { agent, calls } = build({ frames: [FAR] });
  const forced = await agent.step({ instruction: "go", cfg: { engine: "react" } });
  assert.equal(forced.brain.route, "explicit");
  assert.equal(forced.engine, "react");
  assert.equal(calls.token.length, 0);

  const plain = build({ frames: [NEAR], brain: false });
  const out = await plain.agent.step({ instruction: "go" });
  assert.equal(out.brain, null);
  assert.equal(out.token, "W");
  assert.equal(out.gate.allowed, false, "no brain => the veto stands and nothing escalates");
  assert.equal(out.error_type, "gate_refused");
  assert.equal(plain.calls.react.length, 0);
});

test("/v1/arm needs the request-level execute opt-in like every other motion", async () => {
  const { agent } = build({ frames: [FAR] });
  const preview = await agent.arm({ token: "U" });
  assert.equal(preview.preview, true, "a token without execute:true is a gated preview");
  assert.equal(preview.executed, false);
  assert.equal(preview.ok, true, "an allowed preview is a successful answer (HTTP 200), not a 502");
  const raw = await agent.arm({ action_id: 36 });
  assert.equal(raw.ok, false);
  assert.equal(raw.error_type, "execute_required");
  assert.equal(raw.executed, false);
  const bad = await agent.arm({});
  assert.equal(bad.error_type, "request_error");
});

test("external judges can post lessons and the brain state is observable", async () => {
  const { agent, brain } = build({ frames: [FAR] });
  const accepted = agent.brain.learnLesson({
    instruction: "reach the doorway",
    evidence: { depth: NEAR, lidar: { usable: true, reason: "clear" } },
    token: "W",
    outcome: "collision_risk",
    score: -0.9,
    lesson: "offline replay: W here would have clipped the frame",
    engine: "token",
  });
  assert.equal(accepted.accepted, true);
  assert.equal(accepted.size, 1);
  assert.equal(agent.brain.learnLesson({ lesson: "", score: 0, outcome: "x" }).accepted, false);

  const recalled = agent.brain.recall({ instruction: "reach the doorway", situation: { center: "near", left: "far", right: "near", lidar: "clear" } });
  assert.equal(recalled.lessons.length, 1);
  assert.equal(recalled.lessons[0].source, "external");
  assert.ok(recalled.rendered.includes("offline replay"));

  const state = brain.state();
  assert.equal(state.enabled, true);
  assert.equal(state.memory.size, 1);
  assert.equal(state.router.scores["near/clear"].token.n, 1);
  assert.equal(agent.capabilities().brain.enabled, true);
  assert.equal(agent.capabilities().engines.react.deliberate, true);
});

test("/v1/act learns too: vetoes and pulses on the reactive path become lessons and the next act recalls them", async () => {
  const { agent, experience, router, calls } = build({ frames: [NEAR, NEAR, FAR] });
  const first = await agent.act({ instruction: "reach the doorway" });
  assert.equal(first.token, "W");
  assert.equal(first.gate.allowed, false, "act() has no inline escalation; the veto stands");
  assert.equal(first.brain.judge.outcome, "refused");
  assert.equal(first.brain.judge.alternative, "Q");
  assert.equal(first.brain.memory_size, 1);
  assert.equal(calls.token[0].options.experience, null, "nothing to recall yet");
  assert.equal(calls.react.length, 0);

  const lessons = experience.all();
  assert.equal(lessons.length, 1);
  assert.equal(lessons[0].taskKey, "reach-doorway");
  assert.equal(lessons[0].outcome, "refused");
  assert.equal(router.state().steps, 0, "an act never consulted the router, so it is not one of its decisions");
  assert.ok(router.state().scores["near/clear"].token.ema < 0, "but the reflex engine is still scored in this dimension");

  const second = await agent.act({ instruction: "reach the doorway" });
  const reflexCall = calls.token.at(-1);
  assert.equal(typeof reflexCall.options.experience, "string", "the second act carries the refusal lesson");
  assert.ok(reflexCall.options.experience.includes("prefer Q"));
  assert.equal(second.brain.experience, reflexCall.options.experience);

  // a forced token is executed under the same gate and remembered, but never scored as an engine
  const forced = await agent.act({ force_token: "Q" });
  assert.equal(forced.source, "forced");
  assert.equal(forced.brain.judge.outcome, "preview");
  assert.equal(router.state().scores["far/clear"]?.forced, undefined);
  assert.equal(router.state().scores["near/clear"].forced, undefined);
});

test("a fail-closed act is remembered as a fail_closed lesson", async () => {
  const { brain, experience } = build({ frames: [FAR] });
  const { engines: e } = engines();
  const cameraFail = { capture: async () => { throw Object.assign(new Error("no camera"), { errorType: "capture_error" }); } };
  const agent = createAgent({ camera: cameraFail, engines: e, executor: executor(), config: CONFIG, sessions: createSessionStore(), brain, logger: { info() {}, warn() {}, error() {} } });
  const failed = await agent.act({ instruction: "go" });
  assert.equal(failed.ok, false);
  assert.equal(failed.error_type, "capture_error");
  assert.equal(failed.brain.judge.outcome, "capture_error");
  assert.equal(failed.brain.judge.score, 0);
  assert.equal(experience.all().length, 0, "a missing camera teaches nothing and blames no engine");
  assert.deepEqual(brain.router.state().scores, {});

  // an inference failure with a frame in hand is still the engine's fail-closed stop
  const { engines: broken } = engines();
  broken.token = Object.assign(async () => { throw Object.assign(new Error("model said nothing"), { errorType: "invalid_token", retryable: false }); }, { engine: "token" });
  const agent2 = createAgent({ camera: camera([FAR]), engines: broken, executor: executor(), config: CONFIG, sessions: createSessionStore(), brain, logger: { info() {}, warn() {}, error() {} } });
  const failClosed = await agent2.act({ instruction: "go" });
  assert.equal(failClosed.ok, false);
  assert.equal(failClosed.brain.judge.outcome, "fail_closed");
  assert.equal(experience.all().at(-1).outcome, "fail_closed");
  assert.equal(experience.all().at(-1).token, "X");
});

test("the asynchronous judge's outcome teaches memory, overrides a neutral heuristic score, and never steers", async () => {
  const { agent, experience, sessions, router } = build({ frames: [FAR, FAR, FAR] });
  const first = await agent.step({ instruction: "reach the doorway" });
  assert.equal(first.token, "W");
  assert.equal(first.brain.judge.outcome, "preview");
  assert.equal(first.brain.judge.score, 0);

  const rejected = agent.acceptJudgement({ sessionId: "s1", basedOnSeq: 1, outcome: "excellent" });
  assert.equal(rejected.accepted, false);
  assert.equal(rejected.reason, "invalid_judgement");
  assert.equal(agent.acceptJudgement({ sessionId: "nope", basedOnSeq: 1, outcome: "unsafe" }).reason, "unknown_session");
  assert.equal(agent.acceptJudgement({ sessionId: "s1", basedOnSeq: 42, outcome: "unsafe" }).reason, "unknown_seq");

  const learned = agent.acceptJudgement({
    sessionId: "s1",
    basedOnSeq: 1,
    token: "W",
    verdict: "CORRECT",
    next_token: "Q",
    confidence: 0.9,
    outcome: "unsafe",
    lesson: "the doorway is left of centre; a forward pulse here walks into the frame",
  });
  assert.equal(learned.accepted, true);
  assert.equal(learned.learned.outcome, "review_unsafe");
  assert.ok(learned.learned.score < -0.8);
  assert.ok(learned.learned.lesson.includes("walks into the frame"));

  const stored = experience.all().find(lesson => lesson.outcome === "review_unsafe");
  assert.equal(stored.source, "reviewer");
  assert.equal(stored.token, "W");
  assert.equal(stored.alternative, "Q");
  assert.equal(stored.taskKey, "reach-doorway");
  assert.ok(router.state().scores["far/clear"].token.ema < 0, "the engine that proposed the unsafe step is marked down");

  const entry = sessions.get("s1").episodic.get(1);
  assert.equal(entry.judgement.outcome, "unsafe");
  assert.equal(entry.judge.async, true, "a negative asynchronous verdict overrides the neutral heuristic score");
  assert.ok(entry.judge.score < 0);
  assert.equal(sessions.get("s1").correction, null, "a judgement is never a pending correction");

  // the next step recalls the model-authored lesson, ranked above anything the heuristic judge wrote
  const second = await agent.step({ session_id: "s1" });
  assert.ok(second.brain.experience.includes("walks into the frame"));
  assert.equal(second.advice_used, null, "the judgement did not steer");

  // a positive judgement never overrides a negative heuristic one
  const positive = agent.acceptJudgement({ sessionId: "s1", basedOnSeq: 2, outcome: "progress", confidence: 1 });
  assert.equal(positive.accepted, true);
  assert.equal(sessions.get("s1").episodic.get(2).judge.async, undefined);
});

test("the deliberate engine's `next` line empowers exactly the following step, then expires", async () => {
  const next = "after the turn the door frame should sit at centre; go W only if centre > 1.5 m";
  const { agent, calls, sessions } = build({ frames: [NEAR, NEAR, FAR, FAR, FAR], engineOptions: { deliberateNext: next } });
  const first = await agent.step({ instruction: "reach the doorway" });
  assert.equal(first.engine, "react", "the vetoed reflex escalated inline");
  assert.equal(first.brain.next, next, "the deliberate engine left a plan for the next step");
  assert.equal(first.brain.directive, null, "nothing was carried into the first step");
  assert.equal(sessions.get("s1").directive.fromSeq, 1);
  assert.equal(sessions.get("s1").episodic.get(1).next, next);

  // step 2 ends near again => risk_near routes to the deliberate engine, which receives the plan as context
  const second = await agent.step({ session_id: "s1" });
  assert.equal(second.engine, "react");
  assert.deepEqual(second.brain.directive, { text: next, from_seq: 1, engine: "react" });
  assert.equal(calls.react.at(-1).options.directive, next);
  assert.equal(calls.react.at(-1).options.context.directive, next);
  assert.equal(sessions.get("s1").episodic.get(2).directiveUsed, next);

  // step 3: cooldown expired, far => the reflex engine runs and gets the plan step 2 left, as a Plan line
  const third = await agent.step({ session_id: "s1" });
  assert.equal(third.engine, "token");
  assert.equal(calls.token.at(-1).options.directive, next);
  assert.equal(third.brain.directive.from_seq, 2);
  assert.equal(third.brain.next, null, "the reflex engine cannot plan");

  // step 4: the reflex engine left nothing, so nothing is carried
  const fourth = await agent.step({ session_id: "s1" });
  assert.equal(fourth.brain.directive, null);
  assert.equal(calls.token.at(-1).options.directive, null);

  // a directive older than the immediately preceding step is dropped, never applied
  sessions.get("s1").directive = { text: "stale plan", fromSeq: 1, token: "Q", engine: "react" };
  const fifth = await agent.step({ session_id: "s1" });
  assert.equal(fifth.brain.directive, null);
  assert.equal(sessions.get("s1").directive, null, "consumed even when dropped");
});

test("reflection fires on a failing episode, is rate-limited, and only ever writes memory", async () => {
  const seen = [];
  const reflect = {
    model: "reflect-model",
    cooldown: 2,
    infer: async (model, input) => {
      seen.push({ model, input });
      return { lesson: "two vetoes ahead mean the doorway is to the left; turn Q before walking", alternative: "Q", confidence: 0.9 };
    },
  };
  // no inline escalation: engines without react, so the veto stands and the refusal is judged -0.6
  const { agent, brain, experience } = build({ frames: [NEAR, NEAR, NEAR, FAR], reflect, routerOptions: { engines: ["token"], escalateTo: "react" } });
  const first = await agent.step({ instruction: "reach the doorway" });
  assert.equal(first.gate.allowed, false);
  assert.equal(first.brain.reflection.status, "started");
  assert.equal(first.brain.reflection.trigger, "judge:refused");
  await brain.reflections.idle();

  const stored = experience.all().filter(lesson => lesson.source === "reflection");
  assert.equal(stored.length, 1);
  assert.equal(stored[0].outcome, "reflection");
  assert.equal(stored[0].token, "W");
  assert.equal(stored[0].alternative, "Q");
  assert.equal(stored[0].taskKey, "reach-doorway");
  assert.ok(stored[0].score < -0.6 && stored[0].score >= -0.7, "confidence 0.9 => -(0.3 + 0.36)");
  assert.ok(stored[0].lesson.includes("turn Q before walking"));
  assert.equal(seen[0].model, "reflect-model");
  assert.equal(seen[0].input.instruction, "reach the doorway");
  assert.equal(seen[0].input.token, "W");
  assert.ok(seen[0].input.recent.includes("#1 W token vetoed"));
  assert.equal(seen[0].input.situation.center, "near");
  const state = brain.state();
  assert.equal(state.reflection.completed, 1);
  assert.equal(state.reflection.last_trigger, "judge:refused");
  assert.equal(state.summary.reflections, 1);

  // the next veto is inside the cooldown window
  const second = await agent.step({ session_id: first.session_id });
  assert.equal(second.brain.reflection.status, "cooldown");
  assert.ok(second.brain.reflection.trigger.includes("refusal_streak"));
  assert.equal(brain.state().reflection.skipped, 1);

  // the third one is past it and fires again; the fourth (far, clean) has nothing to reflect on
  const third = await agent.step({ session_id: first.session_id });
  assert.equal(third.brain.reflection.status, "started");
  await brain.reflections.idle();
  assert.equal(experience.all().filter(lesson => lesson.source === "reflection").length, 1, "the same lesson consolidates instead of duplicating");
  assert.equal(experience.all().find(lesson => lesson.source === "reflection").hits, 2);
  const fourth = await agent.step({ session_id: first.session_id });
  assert.equal(fourth.brain.reflection, null);

  // the recalled block ranks the reflection above the heuristic refusal when both match
  const recalled = brain.recall({ instruction: "reach the doorway", situation: { center: "near", left: "far", right: "near", lidar: "clear" } });
  assert.equal(recalled.lessons[0].source, "reflection");

  // a failing reflector never surfaces as a step error
  const broken = { model: "m", cooldown: 0, infer: async () => { throw new Error("provider down"); } };
  const other = build({ frames: [NEAR], reflect: broken, routerOptions: { engines: ["token"], escalateTo: "react" } });
  const out = await other.agent.step({ instruction: "go" });
  assert.equal(out.error_type, "gate_refused");
  await other.brain.reflections.idle();
  assert.equal(other.brain.state().reflection.failed, 1);
  assert.match(other.brain.state().reflection.last_error, /provider down/u);
  assert.throws(() => createBrain({ experience, router: other.router, reflect: { model: "m" } }), /needs \{ infer, model \}/u);
});

test("one reviewer call is scored once: the judgement adds its lesson but not a second routing observation", async () => {
  const { agent, experience, router, sessions } = build({ frames: [FAR, FAR] });
  const first = await agent.step({ instruction: "reach the doorway" });
  assert.equal(first.brain.judge.score, 0, "a preview is neutral");
  assert.equal(router.state().scores["far/clear"].token.n, 1);

  const fresh = agent.acceptCorrection({ sessionId: "s1", basedOnSeq: 1, verdict: "CORRECT", next_token: "Q", brief: "door is on the left", confidence: 0.9 });
  assert.equal(fresh.accepted, true);
  assert.equal(router.state().scores["far/clear"].token.n, 2, "the correction scores the engine");
  assert.equal(typeof sessions.get("s1").episodic.get(1).reviewScore, "number");

  const judged = agent.acceptJudgement({ sessionId: "s1", basedOnSeq: 1, token: "W", verdict: "CORRECT", next_token: "Q", confidence: 0.9, outcome: "no_progress", lesson: "turn before walking here" });
  assert.equal(judged.accepted, true);
  assert.equal(router.state().scores["far/clear"].token.n, 2, "the judgement of the same call does not score the engine again");
  const reviewer = experience.all().filter(lesson => lesson.source === "reviewer").map(lesson => lesson.outcome).sort();
  assert.deepEqual(reviewer, ["review_correct", "review_no_progress"], "both halves leave their lesson");
  const entry = sessions.get("s1").episodic.get(1);
  assert.equal(entry.judge.async, true);
  assert.equal(entry.judge.alternative, "Q", "the heuristic alternative survives the override");

  // an unclear judgement with nothing to say is accepted but teaches nothing
  const size = experience.all().length;
  const nothing = agent.acceptJudgement({ sessionId: "s1", basedOnSeq: 1, outcome: "unclear" });
  assert.equal(nothing.accepted, true);
  assert.equal(nothing.learned, null);
  assert.equal(experience.all().length, size);
  assert.equal(router.state().scores["far/clear"].token.n, 2);
});

test("a forward pulse is only judged by a frame captured after it ended", () => {
  const { brain, experience } = build({ frames: [FAR] });
  const session = { id: "retro", task: "go", seq: 0 };
  const executedW = { gate: { allowed: true, reasons: [], lidar: null, state: { ready: true } }, executed: true, preview: false, dispatch: { ok: true }, motion: { vx: 0.3, vy: 0, yaw: 0, duration: 1.0 } };
  const result = { ok: true, token: "W", engine: "token" };
  const before = Date.now();
  brain.record(session, { seq: 1, result, executed: executedW, frame: { depth: { center_m: 2.0, left_m: 2, right_m: 2 } }, engine: "token", taskKey: "go", frameAt: before });
  const first = session.episodic.get(1);
  assert.ok(first.pulseEndAt >= before + 1000, "the pulse ends one second after dispatch");

  // a pipeline frame decided *before* the pulse ended must not be treated as its after-view
  brain.record(session, { seq: 2, result: { ok: true, token: "X", engine: "token" }, executed: { gate: { allowed: true, reasons: [] }, executed: false, preview: true }, frame: { depth: { center_m: 2.0, left_m: 2, right_m: 2 } }, engine: "token", taskKey: "go", frameAt: before + 200 });
  assert.equal(first.afterDepth, undefined, "no retro judgement from a pre-pulse frame");
  assert.equal(experience.all().some(lesson => lesson.outcome === "stalled"), false, "no false stall lesson");

  // a frame from after the pulse ended does judge it
  brain.record(session, { seq: 3, result: { ok: true, token: "X", engine: "token" }, executed: { gate: { allowed: true, reasons: [] }, executed: false, preview: true }, frame: { depth: { center_m: 1.4, left_m: 2, right_m: 2 } }, engine: "token", taskKey: "go", frameAt: first.pulseEndAt + 50 });
  assert.deepEqual(first.afterDepth, { center_m: 1.4, left_m: 2, right_m: 2 });
  assert.deepEqual(first.retro, { score: 0.3, outcome: "progress", seq: 3 }, "judged by the frame two steps later, across the stop");
  assert.equal(first.judge.retro, undefined, "a clean pulse that was already judged positive is not re-learned");

  // another motion in between confounds the depth change: no retro judgement across a turn. The turn's
  // own frame was captured before the pulse ended (a pipeline decision), so it cannot judge it either.
  const other = build({ frames: [FAR] });
  const s2 = { id: "retro2", task: "go", seq: 0 };
  other.brain.record(s2, { seq: 1, result, executed: executedW, frame: { depth: { center_m: 2.0 } }, engine: "token", taskKey: "go", frameAt: before });
  other.brain.record(s2, { seq: 2, result: { ok: true, token: "Q", engine: "token" }, executed: { ...executedW, motion: { vx: 0, vy: 0, yaw: 0.35, duration: 0.6 } }, frame: { depth: { center_m: 2.0 } }, engine: "token", taskKey: "go", frameAt: before + 200 });
  assert.equal(s2.episodic.get(1).afterDepth, undefined, "a pre-pulse frame does not judge the pulse");
  other.brain.record(s2, { seq: 3, result: { ok: true, token: "X", engine: "token" }, executed: { gate: { allowed: true, reasons: [] }, executed: false, preview: true }, frame: { depth: { center_m: 2.0 } }, engine: "token", taskKey: "go", frameAt: before + 10000 });
  assert.equal(s2.episodic.get(1).afterDepth, undefined, "the turn in between confounds the pulse; it is left unjudged");
  assert.equal(other.experience.all().some(lesson => lesson.outcome === "stalled"), false);
});

test("on the reactive path the deliberate engine's `next` line reaches the following act", async () => {
  const next = "after the turn expect the frame at centre; W only if centre > 1.5 m";
  const { agent, calls } = build({ frames: [FAR, FAR, FAR], engineOptions: { deliberateNext: next } });
  const first = await agent.act({ instruction: "reach the doorway", cfg: { engine: "react" } });
  assert.equal(first.engine, "react");
  assert.equal(first.brain.next, next);
  assert.equal(first.brain.directive, null);
  const second = await agent.act({ instruction: "reach the doorway" });
  assert.equal(second.engine, "token");
  assert.equal(calls.token.at(-1).options.directive, next, "the reflex act carries the plan the deliberate act left");
  assert.equal(second.brain.directive, next);
  assert.equal(second.brain.next, null);
  const third = await agent.act({ instruction: "reach the doorway" });
  assert.equal(calls.token.at(-1).options.directive, null, "consumed exactly once");
  assert.equal(third.brain.directive, null);
});

test("requests can only tighten safety: fresh_max_s is capped and use_lidar cannot be switched off", () => {
  const { agent } = build({ frames: [FAR] });
  assert.equal(agent.limits.freshMaxS({ fresh_max_s: 1e9 }), 1.0, "the config ceiling (1.0 s when unset) wins");
  assert.equal(agent.limits.freshMaxS({}), 1.0);
  assert.equal(agent.limits.freshMaxS({ fresh_max_s: 0.2 }), 0.2, "a request may ask for fresher");
  assert.equal(agent.limits.freshMaxS({ fresh_max_s: -3 }), 1.0);
  const capped = createAgent({ camera: camera([FAR]), engines: engines().engines, executor: executor(), config: { ...CONFIG, pipelineFreshMaxS: 0.5 }, sessions: createSessionStore(), logger: { info() {}, warn() {}, error() {} } });
  assert.equal(capped.limits.freshMaxS({ fresh_max_s: 3 }), 0.5);

  const lidarOn = createAgent({ camera: camera([FAR]), engines: engines().engines, executor: executor(), config: { ...CONFIG, useLidar: true }, sessions: createSessionStore(), logger: { info() {}, warn() {}, error() {} } });
  assert.deepEqual(lidarOn.limits.lidar({ use_lidar: false }), { useLidar: true, ignored: true });
  assert.deepEqual(lidarOn.limits.lidar({}), { useLidar: true, ignored: false });
  const permissive = createAgent({ camera: camera([FAR]), engines: engines().engines, executor: executor(), config: { ...CONFIG, useLidar: true, allowRequestLidarOff: true }, sessions: createSessionStore(), logger: { info() {}, warn() {}, error() {} } });
  assert.deepEqual(permissive.limits.lidar({ use_lidar: false }), { useLidar: false, ignored: false });
  assert.deepEqual(agent.limits.lidar({ use_lidar: true }), { useLidar: true, ignored: false }, "a request may require lidar");
});

test("after an inline escalation the gate sees a live depth, not the frame the model saw", async () => {
  // frames: reflex decision (NEAR), the deliberate re-decision (NEAR), the live depth-only read (FAR)
  const { engines: e, calls } = engines();
  const exec = executor();
  const gated = [];
  const original = exec.execute.bind(exec);
  exec.execute = async args => { gated.push(args.frame?.depth ?? null); return original(args); };
  const experience = createExperienceMemory({ now: () => 1_000_000 });
  const router = createRouter({ cheapChain: ["token"], escalateTo: "react", k: 2, cooldown: 1, engines: Object.keys(e) });
  const brain = createBrain({ experience, router, logger: null });
  const agent = createAgent({ camera: camera([NEAR, NEAR, FAR]), engines: e, executor: exec, config: { ...CONFIG, gateLiveDepth: true }, sessions: createSessionStore({ idFactory: () => "s1" }), brain, logger: { info() {}, warn() {}, error() {} } });
  const out = await agent.step({ instruction: "reach the doorway" });
  assert.equal(out.token, "Q");
  assert.equal(out.brain.inline.gate_frame, "live");
  assert.equal(gated.length, 2);
  assert.deepEqual(gated[0], NEAR, "the reflex token was gated on its own frame");
  assert.deepEqual(gated[1], FAR, "the deliberate token was gated on the live depth");
  assert.equal(calls.react.length, 1);

  // without the knob the decision frame is used and the response says so
  const plain = build({ frames: [NEAR, NEAR, FAR] });
  const before = await plain.agent.step({ instruction: "reach the doorway" });
  assert.equal(before.brain.inline.gate_frame, "decision");
});

test("spatial memory: the step's pose is recorded, the prompt says where the robot is, and coming back to a refused spot escalates", async () => {
  // NEAR at the start (W refused), then FAR frames while the robot turns away and comes back.
  const { engines: e, calls } = engines({ reflexToken: "W", deliberateToken: "Q" });
  let reflexToken = "W";
  e.token = Object.assign(async (model, frame, options) => { calls.token.push({ model, options }); return { token: reflexToken, tokenReadyS: 0.1 }; }, { engine: "token" });
  const exec = executor();
  // executed motions: make every allowed velocity token actually execute so the pose moves
  const original = exec.execute.bind(exec);
  exec.execute = async args => {
    const out = await original(args);
    if (out.gate.allowed && args.token !== "X") {
      const motion = args.token === "W" ? { vx: 0.3, vy: 0, yaw: 0, duration: 1 } : args.token === "Q" ? { vx: 0, vy: 0, yaw: Math.PI / 2, duration: 1 } : args.token === "E" ? { vx: 0, vy: 0, yaw: -Math.PI / 2, duration: 1 } : args.token === "S" ? { vx: -0.3, vy: 0, yaw: 0, duration: 1 } : null;
      return { ...out, executed: true, preview: false, motion, dispatch: { ok: true } };
    }
    return out;
  };
  const experience = createExperienceMemory({ now: () => 1_000_000 });
  const router = createRouter({ cheapChain: ["token"], escalateTo: "react", k: 5, cooldown: 0, inlineEscalate: false, engines: Object.keys(e) });
  const brain = createBrain({ experience, router, logger: null });
  const sessions = createSessionStore({ idFactory: () => "s1" });
  const agent = createAgent({ camera: camera([NEAR, FAR, FAR, FAR, FAR, FAR, FAR]), engines: e, executor: exec, config: { ...CONFIG, executeEnabled: true }, sessions, brain, logger: { info() {}, warn() {}, error() {} } });

  const first = await agent.step({ instruction: "reach the doorway" });
  assert.equal(first.gate.allowed, false, "W refused at the start");
  assert.deepEqual(first.brain.spatial.revisits, 0);
  assert.equal(first.brain.where, null, "nothing to say about position before the first step is recorded");

  reflexToken = "Q";
  const second = await agent.step({ session_id: "s1" });
  assert.equal(second.token, "Q");
  assert.match(second.brain.where, /est\. pose \+0\.0 m fwd, \+0\.0 m left, heading \+0°/u, "the second step is planned from the start pose");
  reflexToken = "W";
  const third = await agent.step({ session_id: "s1" });
  assert.match(third.brain.where, /heading \+90°/u);
  reflexToken = "E";
  const fourth = await agent.step({ session_id: "s1" });
  assert.match(fourth.brain.where, /\+0\.0 m fwd, \+0\.3 m left/u);
  reflexToken = "E";
  const fifth = await agent.step({ session_id: "s1" });
  assert.match(fifth.brain.where, /heading \+0°/u);
  // heading -90° now, 0.3 m left of start; W walks back toward the start line
  reflexToken = "W";
  const sixth = await agent.step({ session_id: "s1" });
  assert.match(sixth.brain.where, /heading -90°/u);
  // decided at the start position facing -90° (a different cell), then Q turns back to 0°: the
  // post-step signals already say "back at the refused spot", which is what the next route reads
  reflexToken = "Q";
  const seventh = await agent.step({ session_id: "s1" });
  assert.equal(seventh.brain.route, "default", "facing -90° at the start is not the refused cell");
  assert.deepEqual({ revisits: seventh.brain.spatial.revisits, refusedHere: seventh.brain.spatial.refusedHere }, { revisits: 1, refusedHere: 1 });
  // facing 0° at the start again: the cell where W was refused
  reflexToken = "W";
  const eighth = await agent.step({ session_id: "s1" });
  assert.equal(eighth.brain.route, "revisit", "back at a spot where a token was refused escalates to the deliberate engine");
  assert.equal(eighth.engine, "react");
  assert.match(calls.react.at(-1).options.context.where, /this spot seen 1x before \(#1, 1 refused here\)/u);
  assert.ok(calls.react.at(-1).options.context.where.includes("dead reckoning"));

  const state = agent.brain.session("s1");
  assert.equal(state.steps.length, 8);
  assert.equal(state.steps[0].pose.x, 0);
  assert.equal(state.trail.length, 8);
  assert.ok(state.spatial.travelled_m > 0.5);
  assert.equal(agent.brain.session("nope"), null);
  assert.match(agent.brain.session("act").session_id, /^act-[0-9a-z]+$/u, "the reactive session has one id per process, aliased as act");
  assert.equal(agent.brain.session(agent.brain.session("act").session_id).session_id, agent.brain.session("act").session_id);
});

test("a new session on the same task is told how the last ones went, and every step lands in the episode store", async () => {
  const episodes = createEpisodeStore({ now: () => 1_700_000_000_000 });
  const { engines: e, calls } = engines();
  const experience = createExperienceMemory({ now: () => 1_000_000 });
  const router = createRouter({ cheapChain: ["token"], escalateTo: "react", k: 2, cooldown: 1, engines: Object.keys(e) });
  const brain = createBrain({ experience, router, logger: null, episodes, episodeRecallK: 2 });
  let id = 0;
  const sessions = createSessionStore({ idFactory: () => `s${++id}` });
  const agent = createAgent({ camera: camera([NEAR, NEAR, FAR, FAR, FAR, FAR]), engines: e, executor: executor(), config: CONFIG, sessions, brain, logger: { info() {}, warn() {}, error() {} } });

  // session 1: a refused W escalated inline to Q, then a clean far step
  const first = await agent.step({ instruction: "reach the doorway" });
  assert.equal(first.brain.episodes_used, 0, "nothing to recall on the very first session");
  assert.equal(first.brain.history, null);
  await agent.step({ session_id: first.session_id });
  const kept = agent.acceptCorrection({ sessionId: first.session_id, basedOnSeq: 2, verdict: "KEEP", next_token: "Q", brief: "clear", confidence: 0.7 });
  assert.equal(kept.accepted, true);
  const digest = episodes.digest(first.session_id);
  assert.equal(digest.steps, 2);
  assert.equal(digest.task, "reach the doorway");
  assert.equal(digest.deliberate, 2, "step 1 escalated inline, step 2 was routed risk_near: both ran the deliberate engine");
  assert.equal(digest.escalations, 1, "only step 2's route was an escalation; step 1 was routed default");
  assert.deepEqual(digest.reviews, { KEEP: 1, CORRECT: 0, STOP: 0 });
  assert.deepEqual(digest.tokens, { Q: 2 }, "the vetoed reflex W is not a step of its own; each step records the token that was finally gated");

  // session 2, same task: the deliberate engine (forced) is told about session 1
  const second = await agent.step({ instruction: "reach the doorway", cfg: { engine: "react" } });
  assert.notEqual(second.session_id, first.session_id);
  assert.equal(second.brain.episodes_used, 1);
  assert.match(second.brain.history, /^Past episode \(.*\): 2 steps; 1 escalations; ended preview on Q\.$/u, "previews never executed, so the episode ended on a preview");
  assert.ok(calls.react.at(-1).options.context.history.includes("Past episode"));
  const reflexCall = calls.token.at(-1);
  assert.equal(reflexCall.options.context?.history ?? null, null, "the reflex path never carries episode history");

  // a different task recalls nothing
  const other = await agent.step({ instruction: "go to the kitchen", cfg: { engine: "react" } });
  assert.equal(other.brain.episodes_used, 0);

  assert.equal(agent.brain.episodes({ instruction: "reach the doorway", k: 5 }).digests.length, 2);
  assert.equal(agent.brain.session(first.session_id).digest.steps, 2);
  assert.equal(brain.state().episodes.sessions, 3);
  assert.equal(brain.state().summary.episodes, 3);
});

test("brain.reset() forgets lessons and routing scores but keeps the episode store unless asked", async () => {
  const episodes = createEpisodeStore();
  const { engines: e } = engines();
  const experience = createExperienceMemory({ now: () => 1_000_000 });
  const router = createRouter({ cheapChain: ["token"], escalateTo: "react", k: 2, cooldown: 1, engines: Object.keys(e) });
  const brain = createBrain({ experience, router, logger: null, episodes });
  const agent = createAgent({ camera: camera([NEAR, NEAR, FAR]), engines: e, executor: executor(), config: CONFIG, sessions: createSessionStore({ idFactory: () => "s1" }), brain, logger: { info() {}, warn() {}, error() {} } });
  await agent.step({ instruction: "reach the doorway" });
  assert.ok(experience.stats().size >= 1);
  assert.ok(router.state().observations >= 1);
  assert.equal(episodes.stats().sessions, 1);
  const result = agent.brain.reset({});
  assert.deepEqual({ experience: result.experience, router: result.router, episodes: result.episodes }, { experience: true, router: true, episodes: false });
  assert.ok(result.before.lessons >= 1);
  assert.equal(experience.stats().size, 0);
  assert.equal(router.state().observations, 0);
  assert.deepEqual(router.state().scores, {});
  assert.equal(router.state().steps, 0);
  assert.equal(episodes.stats().sessions, 1, "the episode store is the evaluation's evidence and stays");
  agent.brain.reset({ episodes: true });
  assert.equal(episodes.stats().sessions, 0);
});

test("brain.record: in-place pulses stay one stay, superseded pulses still move the pose, and a refused-spot revisit triggers reflection", async () => {
  const { motionFor } = await import("../src/actions.mjs");
  const reflected = [];
  const reflect = { model: "m", cooldown: 0, minScore: -2, infer: async (model, input) => { reflected.push(input); return { text: "centre near: the door is on the left; turn Q before walking" }; } };
  const experience = createExperienceMemory({ now: () => 1_000_000 });
  const router = createRouter({ cheapChain: ["token"], escalateTo: "react", k: 5, cooldown: 0, inlineEscalate: false, engines: ["token", "react"] });
  const brain = createBrain({ experience, router, logger: null, reflect });
  const session = { id: "s", task: "reach the doorway", seq: 0 };
  const far = { depth: { center_m: 2.5, left_m: 2.5, right_m: 2.5 } };
  const near = { depth: { center_m: 0.5, left_m: 2.5, right_m: 2.5 } };
  let seq = 0;
  const step = (token, { refused = false, frame = far, dispatch = null, executed = !refused, motion = motionFor(token, null, { maxDurationS: 2 }), durationMs = null } = {}) => {
    seq += 1;
    session.seq = seq;
    const plan = brain.plan(session, { instruction: session.task });
    const exec = {
      gate: { allowed: !refused, reasons: refused ? ["depth:depth_center_near"] : [], state: { ready: true, reasons: [] }, depth: frame.depth, lidar: null },
      executed,
      preview: false,
      motion: durationMs ? motionFor(token, durationMs, { maxDurationS: 2 }) : motion,
      dispatch: dispatch ?? (executed ? { ok: true } : null),
    };
    const learned = brain.record(session, { seq, result: { ok: true, token, engine: plan.engine }, executed: exec, frame, engine: plan.engine, route: plan.route, taskKey: plan.taskKey });
    return { route: plan.route.reason, learned };
  };

  step("W", { refused: true, frame: near }); // refused at the door
  step("Q"); // one real Q: 12 degrees, in place
  const afterTurn = step("W", { frame: near, refused: true });
  assert.equal(afterTurn.route, "default", "a refused W followed by one turn is not a revisit and does not escalate");
  assert.equal(afterTurn.learned.spatial.revisits, 0);
  step("W", { durationMs: 500 }); // 0.15 m: still the same cell
  assert.equal(step("W", { frame: near, refused: true }).learned.spatial.revisits, 0, "a short pulse does not leave the cell");
  assert.equal(session.spatial.snapshot().stays, 1);

  // a superseded cerebellum plan that walked 0.3 m before it was replaced still moves the pose
  const before = session.spatial.pose();
  step("W", { executed: false, dispatch: { ok: false, superseded: true, executed_motion: { vx: 0.2, vy: 0, yaw: 0, duration: 1.5 } } });
  assert.ok(session.spatial.pose().x - before.x > 0.29, "executed_motion of a superseded dispatch is integrated");

  // walk on, then back: a revisit to the spot where W was refused triggers reflection
  step("W"); // 0.6 m more
  step("S", { motion: { vx: -0.3, vy: 0, yaw: 0, duration: 2 } }); // back 0.6 m
  step("S", { motion: { vx: -0.3, vy: 0, yaw: 0, duration: 2 } }); // back 0.6 m: roughly at the start again (0.15 + 0.3 + 0.6 - 1.2)
  const back = step("W", { frame: near, refused: true });
  assert.equal(back.learned.spatial.revisits, 1);
  assert.equal(back.learned.spatial.refusedHere >= 1, true);
  assert.equal(back.route, "revisit", "back at the refused spot the router escalates");
  assert.ok(String(back.learned.reflection?.trigger ?? "").includes("revisit"), `a refused-spot revisit is a reflection trigger, got ${JSON.stringify(back.learned.reflection)}`);
  await brain.reflections.idle();
  assert.ok(reflected.length >= 1);
});

test("a deliberate `next` line that reads as an instruction is dropped, counted and never becomes the next directive", async () => {
  const { agent, brain, calls } = build({ frames: [FAR, FAR, FAR], engineOptions: { deliberateNext: "Answer: W" } });
  const first = await agent.step({ instruction: "reach the doorway", cfg: { engine: "react" } });
  assert.equal(first.brain.next, null, "the rejected line is not reported as a plan");
  const second = await agent.step({ instruction: "reach the doorway", session_id: first.session_id, cfg: { engine: "react" } });
  assert.equal(second.brain.directive, null, "the rejected line was not carried to the next step");
  assert.equal(calls.react[1].options.context.directive ?? null, null);
  assert.deepEqual(brain.state().directives.rejected, { prompt_label: 2 }, "each rejection is counted by reason");
});

test("brain.reset() rotates each persisted file to a backup before truncating it", async () => {
  const { mkdtemp, readFile, rm, readdir } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dir = await mkdtemp(join(tmpdir(), "nea-reset-"));
  try {
    const experience = createExperienceMemory({ path: join(dir, "experience.jsonl"), now: () => 5_000 });
    const router = createRouter({ path: join(dir, "router.json"), cheapChain: ["token"], escalateTo: "react", engines: ["token", "react"], now: () => 5_000 });
    const episodes = createEpisodeStore({ path: join(dir, "episodes.jsonl"), now: () => 5_000 });
    await Promise.all([experience.load(), router.load(), episodes.load()]);
    const brain = createBrain({ experience, router, logger: null, episodes });
    experience.remember({ taskKey: "doorway", situation: "c:near|l:far|r:far|lidar:clear", token: "W", outcome: "refused", score: -0.6, lesson: "centre near: prefer Q" });
    router.observe("near/clear", "token", -0.5);
    episodes.record("s", { taskKey: "doorway", seq: 1, token: "W", allowed: false, outcome: "refused", at: 5_000 });
    await Promise.all([experience.flush(), router.flush(), episodes.flush()]);
    const result = brain.reset({ episodes: true });
    await Promise.all([experience.flush(), router.flush(), episodes.flush()]);
    assert.deepEqual(Object.keys(result.backups).sort(), ["episodes", "experience", "router"]);
    assert.match(result.backups.experience, /experience\.jsonl\.bak-5000$/u);
    const files = (await readdir(dir)).sort();
    assert.deepEqual(files, ["episodes.jsonl", "episodes.jsonl.bak-5000", "experience.jsonl", "experience.jsonl.bak-5000", "router.json", "router.json.bak-5000"]);
    assert.ok((await readFile(join(dir, "experience.jsonl.bak-5000"), "utf8")).includes("prefer Q"), "the backup holds the wiped lessons");
    assert.equal((await readFile(join(dir, "experience.jsonl"), "utf8")).trim(), "");
    assert.ok((await readFile(join(dir, "router.json.bak-5000"), "utf8")).includes("near/clear"));
    assert.equal(JSON.parse(await readFile(join(dir, "router.json"), "utf8")).observations, 0);
    assert.equal((await readFile(join(dir, "episodes.jsonl"), "utf8")), "");
    assert.equal(result.before.episodes, 1);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
