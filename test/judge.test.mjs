import assert from "node:assert/strict";
import { test } from "node:test";

import { alternativeFor, externalJudgementError, externalLessonError, judgeOutcome, judgeReview, judgeStep } from "../src/judge.mjs";
import HOLONOMIC from "./fixtures/profiles/holonomic.mjs";

const CLEAR_LIDAR = { usable: true, blocked: false, reason: "clear", left: 2.5, right: 2.5 };

test("a gate refusal is a negative lesson that names a geometric alternative", () => {
  const verdict = judgeStep({
    token: "W",
    engine: "prime",
    gate: { allowed: false, reasons: ["depth:depth_center_near"] },
    depth: { center_m: 0.4, left_m: 2.1, right_m: 0.5 },
    lidar: CLEAR_LIDAR,
    state: { ready: true },
    taskKey: "doorway",
  });
  assert.equal(verdict.outcome, "refused");
  assert.equal(verdict.score, -0.6);
  assert.equal(verdict.alternative, "Q", "more room on the left => turn left");
  assert.equal(verdict.lesson.source, "heuristic");
  assert.equal(verdict.lesson.token, "W");
  assert.equal(verdict.lesson.taskKey, "doorway");
  assert.ok(verdict.lesson.lesson.includes("vetoed (depth_center_near)"));
  assert.ok(verdict.lesson.lesson.includes("prefer Q"));
  assert.equal(verdict.situation.center, "near");
});

test("alternatives back off when both sides are closed and stop when nothing is evident", () => {
  const closed = { center_m: 0.3, left_m: 0.4, right_m: 0.45 };
  assert.equal(alternativeFor("W", { depth: closed, lidar: CLEAR_LIDAR }), "S");
  assert.equal(alternativeFor("W", { depth: closed, lidar: { usable: false } }), "X", "no lidar => do not back into the unknown");
  assert.equal(alternativeFor("W", { depth: { center_m: 0.3, left_m: 0.4, right_m: 1.8 }, lidar: CLEAR_LIDAR }), "E");
  assert.equal(alternativeFor("W", { depth: { center_m: 0.3, left_m: 1.8, right_m: 1.8 }, lidar: { ...CLEAR_LIDAR, left: 0.3 } }), "E", "lidar closes the left");
  assert.equal(alternativeFor("Q", { depth: closed, lidar: CLEAR_LIDAR }), "X");
  assert.equal(alternativeFor("U", { armHeld: "P" }), "R");
  assert.equal(alternativeFor("U", { armHeld: null }), "X");
  assert.equal(alternativeFor("X", {}), "X");
  assert.equal(alternativeFor("W", {}), "X", "no evidence at all => stop");
});

test("executed forward pulses are judged by the depth they actually gained", () => {
  const base = { token: "W", executed: true, dispatchOk: true, gate: { allowed: true, reasons: [] }, lidar: CLEAR_LIDAR, state: { ready: true } };
  const progressed = judgeStep({ ...base, depth: { center_m: 2.0 }, afterDepth: { center_m: 1.4 } });
  assert.equal(progressed.outcome, "progress");
  assert.equal(progressed.score, 0.3);
  const stalled = judgeStep({ ...base, depth: { center_m: 2.0 }, afterDepth: { center_m: 1.99 } });
  assert.equal(stalled.outcome, "stalled");
  assert.ok(stalled.score < 0);
  const tooClose = judgeStep({ ...base, depth: { center_m: 0.7 }, afterDepth: { center_m: 0.35 } });
  assert.equal(tooClose.outcome, "too_close");
  const turned = judgeStep({ ...base, token: "Q", depth: { center_m: 2.0 } });
  assert.equal(turned.outcome, "progress");
  assert.equal(turned.score, 0.2);
  const rejected = judgeStep({ ...base, dispatchOk: false, depth: { center_m: 2.0 } });
  assert.equal(rejected.outcome, "dispatch_failed");
});

test("oscillation, fail-closed stops and previews are recognised", () => {
  const oscillating = judgeStep({
    token: "Q",
    executed: true,
    dispatchOk: true,
    gate: { allowed: true, reasons: [] },
    depth: { center_m: 1.0 },
    signals: { oscillation: 2, lastToken: "E" },
  });
  assert.equal(oscillating.outcome, "oscillating");
  assert.ok(oscillating.lesson.lesson.includes("Q/E"));
  assert.equal(oscillating.score, -0.1, "progress +0.2 and oscillation -0.3");

  const failClosed = judgeStep({ token: "X", ok: false, engine: "prime" });
  assert.equal(failClosed.outcome, "fail_closed");
  assert.ok(failClosed.lesson.lesson.includes("prime"));
  const noCamera = judgeStep({ token: "X", ok: false, engine: "prime", errorType: "capture_error" });
  assert.equal(noCamera.outcome, "capture_error");
  assert.equal(noCamera.score, 0, "a camera failure is not the engine's fault");
  assert.equal(noCamera.lesson, null, "and there was no situation to learn about");
  assert.equal(judgeStep({ token: "X", ok: false, errorType: "config_error" }).lesson, null);
  assert.equal(judgeStep({ token: "X", ok: false, errorType: "deadline" }).outcome, "fail_closed", "a provider deadline is still on the engine");

  const stopped = judgeStep({ token: "X", ok: true, gate: { allowed: true, reasons: [] } });
  assert.equal(stopped.outcome, "stopped");
  assert.equal(stopped.lesson, null, "an ordinary stop teaches nothing");

  const preview = judgeStep({ token: "W", ok: true, gate: { allowed: true, reasons: [] }, executed: false });
  assert.equal(preview.outcome, "preview");
  assert.equal(preview.score, 0);
});

test("reviewer verdicts become weighted lessons, corrections heaviest", () => {
  const depth = { center_m: 0.9, left_m: 2, right_m: 2 };
  const stop = judgeReview({ review: { verdict: "STOP", next_token: "X", brief: "person stepping in", confidence: 1 }, token: "W", depth, taskKey: "doorway" });
  assert.equal(stop.outcome, "review_stop");
  assert.equal(stop.score, -0.8);
  assert.equal(stop.lesson.source, "reviewer");
  assert.ok(stop.lesson.lesson.includes("person stepping in"));

  const correct = judgeReview({ review: { verdict: "CORRECT", next_token: "Q", brief: "door is to the left", confidence: 0.5 }, token: "W", depth });
  assert.equal(correct.outcome, "review_correct");
  assert.equal(correct.alternative, "Q");
  assert.equal(correct.score, -0.375);

  const keep = judgeReview({ review: { verdict: "KEEP", next_token: "W", brief: "clear corridor", confidence: 0 }, token: "W", depth });
  assert.equal(keep.outcome, "review_keep");
  assert.equal(keep.score, 0.1);
  assert.equal(keep.alternative, null);

  assert.equal(judgeReview({ review: { verdict: "MAYBE" }, token: "W" }), null);
  assert.equal(judgeReview({ review: null, token: "W" }), null);
});

test("external lessons are validated before they can reach memory", () => {
  assert.equal(externalLessonError({ lesson: "ok", score: 0.5, outcome: "progress" }), null);
  assert.equal(externalLessonError({ lesson: "ok", score: 0.5, outcome: "progress", token: "W" }), null);
  assert.ok(externalLessonError({ lesson: "ok", score: 0.5, outcome: "progress", token: "Z" }));
  assert.ok(externalLessonError({ lesson: "", score: 0.5, outcome: "progress" }));
  assert.ok(externalLessonError({ lesson: "ok", score: 3, outcome: "progress" }));
  assert.ok(externalLessonError({ lesson: "ok", score: 0, outcome: "" }));
  assert.ok(externalLessonError([]));
});

test("an asynchronous outcome judgement becomes a weighted, model-authored lesson", () => {
  const depth = { center_m: 0.9, left_m: 2, right_m: 2 };
  const unsafe = judgeOutcome({
    judgement: { outcome: "unsafe", lesson: "the person on the right was closing in; stop earlier", confidence: 1, verdict: "STOP", next_token: "X" },
    token: "W",
    depth,
    taskKey: "doorway",
    engine: "token",
  });
  assert.equal(unsafe.outcome, "review_unsafe");
  assert.equal(unsafe.score, -0.9);
  assert.equal(unsafe.alternative, "X");
  assert.equal(unsafe.engine, "token");
  assert.equal(unsafe.lesson.source, "reviewer");
  assert.equal(unsafe.lesson.taskKey, "doorway");
  assert.ok(unsafe.lesson.lesson.startsWith("centre mid"));
  assert.ok(unsafe.lesson.lesson.includes("W was unsafe — the person on the right"));

  const progress = judgeOutcome({ judgement: { outcome: "progress", confidence: 0 }, token: "Q", depth });
  assert.equal(progress.score, 0.2, "confidence 0 halves the outcome score");
  assert.equal(progress.alternative, null);
  assert.ok(progress.lesson.lesson.endsWith("Q made progress"));

  const halfway = judgeOutcome({ judgement: { outcome: "no_progress", verdict: "CORRECT", next_token: "E" }, token: "W", depth });
  assert.equal(halfway.score, -0.3, "a missing confidence counts as 0.5");
  assert.equal(halfway.alternative, "E");
  assert.equal(judgeOutcome({ judgement: { outcome: "no_progress", verdict: "CORRECT", next_token: "W" }, token: "W", depth }).alternative, null, "the same token is not an alternative");

  assert.equal(judgeOutcome({ judgement: { outcome: "unclear" }, token: "W", depth }), null, "unclear with nothing to say is not a judgement");
  const spoken = judgeOutcome({ judgement: { outcome: "unclear", lesson: "glass door: depth and lidar both see through it" }, token: "W", depth });
  assert.ok(spoken.lesson.lesson.includes("glass door"));

  assert.equal(judgeOutcome({ judgement: { outcome: "great" }, token: "W" }), null);
  assert.equal(judgeOutcome({ judgement: null, token: "W" }), null);
});

test("outcome judgements posted over HTTP are validated before they reach memory", () => {
  const base = { sessionId: "s1", basedOnSeq: 3, outcome: "progress" };
  assert.equal(externalJudgementError(base), null);
  assert.equal(externalJudgementError({ ...base, lesson: "ok", confidence: 0.5, verdict: "KEEP", next_token: "W", token: "W" }), null);
  assert.ok(externalJudgementError({ ...base, sessionId: "" }));
  assert.ok(externalJudgementError({ ...base, basedOnSeq: -1 }));
  assert.ok(externalJudgementError({ ...base, basedOnSeq: 1.5 }));
  assert.ok(externalJudgementError({ ...base, outcome: "excellent" }));
  assert.ok(externalJudgementError({ ...base, lesson: "x".repeat(201) }));
  assert.ok(externalJudgementError({ ...base, confidence: 2 }));
  assert.ok(externalJudgementError({ ...base, verdict: "MAYBE" }));
  assert.ok(externalJudgementError({ ...base, next_token: "Z" }));
  assert.ok(externalJudgementError([]));
});

test("alternatives and judgements use the profile's roles and fall back to its stop token", () => {
  const closed = { center_m: 0.3, left_m: 0.4, right_m: 0.45 };
  assert.equal(alternativeFor("F", { depth: { center_m: 0.3, left_m: 2, right_m: 0.4 }, profile: HOLONOMIC }), "H", "no turn role: stop");
  assert.equal(alternativeFor("F", { depth: closed, lidar: { usable: true }, profile: HOLONOMIC }), "H", "no backward role: stop");
  assert.equal(alternativeFor("W", { depth: closed, profile: HOLONOMIC }), "H", "an unknown token is the profile's stop");
  const refused = judgeStep({ token: "F", gate: { allowed: false, reasons: ["depth:depth_center_near"] }, depth: { center_m: 0.4, left_m: 2, right_m: 2 }, state: { ready: true }, profile: HOLONOMIC });
  assert.equal(refused.outcome, "refused");
  assert.equal(refused.alternative, "H");
  assert.ok(refused.lesson.lesson.includes("prefer H"));
  const progressed = judgeStep({ token: "F", executed: true, dispatchOk: true, gate: { allowed: true, reasons: [] }, depth: { center_m: 2.0 }, afterDepth: { center_m: 1.4 }, profile: HOLONOMIC });
  assert.equal(progressed.outcome, "progress", "the forward role gets the depth-gain judgement");
  assert.equal(judgeStep({ token: "H", ok: true, gate: { allowed: true, reasons: [] }, profile: HOLONOMIC }).outcome, "stopped");
  assert.equal(externalJudgementError({ sessionId: "s", basedOnSeq: 1, outcome: "unsafe", token: "F" }, HOLONOMIC), null);
  assert.match(externalJudgementError({ sessionId: "s", basedOnSeq: 1, outcome: "unsafe", token: "W" }, HOLONOMIC), /token set/u);
});
