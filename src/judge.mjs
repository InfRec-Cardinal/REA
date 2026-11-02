// Outcome judge: turns what happened after a decision into a score and a lesson.
//
// Two judges feed the same memory:
//   - the heuristic judge below runs in-process on every step, from evidence that is already at hand
//     (gate verdict, dispatch result, depth before/after, the token history). It costs microseconds.
//   - the asynchronous reviewer (thinker) sees before/after frames and returns KEEP / CORRECT / STOP;
//     `judgeReview()` converts that into a lesson too, whether or not the correction arrived in time
//     to steer the next step. A stale correction is useless for control but still a valid judgement.
//
// The judge never acts. Its output goes to memory (what to ask next time) and to the router (which
// engine to ask). Corrections are weighted heaviest because they are the lessons that would have
// prevented a mistake.

import { situationOf, situationPhrase } from "./memory.mjs";
import { G1_PROFILE } from "./profiles/g1.mjs";

const VERDICTS = new Set(["KEEP", "CORRECT", "STOP"]);

/**
 * What an asynchronous judge may say about an executed step once it has seen the after frame.
 *   progress     the action advanced the task without creating a hazard
 *   no_progress  nothing useful changed (deadband, blocked, oscillation)
 *   unsafe       the action moved toward a hazard or should have been refused
 *   unclear      the evidence does not support a verdict
 */
export const JUDGE_OUTCOMES = Object.freeze(["progress", "no_progress", "unsafe", "unclear"]);
const OUTCOME_SCORE = Object.freeze({ progress: 0.4, no_progress: -0.4, unsafe: -0.9, unclear: 0 });
const OUTCOME_PHRASE = Object.freeze({ progress: "made progress", no_progress: "made no progress", unsafe: "was unsafe", unclear: "had an unclear outcome" });

function clamp(value) {
  return Number(Math.min(1, Math.max(-1, value)).toFixed(3));
}

function metres(value) {
  if (value === null || value === undefined || value === "") return null;
  return Number.isFinite(Number(value)) ? Number(value) : null;
}

/**
 * A safer alternative to a token that was refused or judged bad in this situation. Purely geometric:
 * turn toward the side with more room, back off when both sides are closed, release a held arm pose
 * before proposing another. Returns the profile's stop token when nothing better is evident — a stop is always allowed.
 */
export function alternativeFor(token, { depth = null, lidar = null, armHeld = null, profile = G1_PROFILE } = {}) {
  const record = profile.actionOrStop(token);
  const { stop, forward, backward, strafeLeft, strafeRight, turnLeft, turnRight } = profile.roles;
  if (record.kind === "arm_primitive") return armHeld && profile.arms ? profile.arms.release : stop;
  if (record.token === stop) return stop;

  const left = metres(depth?.left_m);
  const right = metres(depth?.right_m);
  const lidarLeft = metres(lidar?.left);
  const lidarRight = metres(lidar?.right);
  const roomLeft = Math.min(left ?? Infinity, lidarLeft !== null && lidarLeft >= 0 ? lidarLeft : Infinity);
  const roomRight = Math.min(right ?? Infinity, lidarRight !== null && lidarRight >= 0 ? lidarRight : Infinity);
  const leftKnown = Number.isFinite(roomLeft);
  const rightKnown = Number.isFinite(roomRight);

  if ([forward, strafeLeft, strafeRight].includes(record.token)) {
    if (leftKnown || rightKnown) {
      const bestLeft = (leftKnown ? roomLeft : -1) >= (rightKnown ? roomRight : -1);
      const best = bestLeft ? roomLeft : roomRight;
      const turn = bestLeft ? turnLeft : turnRight;
      if (best >= 0.6 && turn) return turn;
      if (lidar?.usable === true && backward) return backward;
    }
    return stop;
  }
  // a refused turn or backward step means the lidar was unusable or the robot was not ready; do not guess a motion
  return stop;
}

function describeGate(gate) {
  return (gate?.reasons ?? []).map(reason => String(reason).replace(/^(depth|lidar|state|arm):/u, "")).join(", ") || "gate";
}

/**
 * Heuristic step judgement.
 *
 *   token          the token that was proposed (after fail-closed resolution)
 *   engine         which engine produced it
 *   ok             whether inference itself succeeded (false => fail-closed X)
 *   errorType      why it failed, when it did: a capture or configuration failure is not the
 *                  engine's doing and there was no situation to learn about, so it teaches nothing
 *   gate           { allowed, reasons } from the executor
 *   executed       whether the actuator was actually driven
 *   dispatchOk     actuator acknowledgement, when executed
 *   depth          depth summary of the frame the model saw
 *   afterDepth     depth summary observed after execution, when known
 *   lidar, state   evidence used by the gate
 *   signals        episodic signals (oscillation etc.)
 *   armHeld        currently held arm pose, for the alternative
 */
export const UNATTRIBUTABLE_ERRORS = Object.freeze(new Set(["capture_error", "config_error", "request_error"]));

export function judgeStep({
  token,
  engine = null,
  ok = true,
  errorType = null,
  gate = null,
  executed = false,
  dispatchOk = null,
  depth = null,
  afterDepth = null,
  lidar = null,
  state = null,
  signals = null,
  armHeld = null,
  taskKey = "generic",
  profile = G1_PROFILE,
} = {}) {
  const record = profile.actionOrStop(token);
  const situation = situationOf({ depth, lidar, state, lastToken: signals?.lastToken ?? null, profile });
  const phrase = situationPhrase(situation);
  const alternative = alternativeFor(record.token, { depth, lidar, armHeld, profile });
  const motion = new Set(profile.motion);
  const forward = profile.roles.forward;
  const stop = profile.roles.stop;
  let score = 0;
  let outcome = "neutral";
  let lesson = null;
  let attributable = true;

  if (!ok && UNATTRIBUTABLE_ERRORS.has(errorType)) {
    // No frame or no provider: the loop stopped fail-closed, but nothing about the model or the
    // situation was learned. No lesson and no routing observation, so a camera outage never
    // changes which engine is asked next.
    outcome = errorType;
    attributable = false;
  } else if (!ok) {
    score -= 0.2;
    outcome = "fail_closed";
    lesson = `${phrase}: ${engine ?? "model"} produced no valid token; the loop stopped fail-closed`;
  } else if (gate && gate.allowed === false) {
    score -= 0.6;
    outcome = "refused";
    lesson = `${phrase}: ${record.token} was vetoed (${describeGate(gate)}); prefer ${alternative}`;
  } else if (executed && dispatchOk === false) {
    score -= 0.4;
    outcome = "dispatch_failed";
    lesson = `${phrase}: ${record.token} passed the gate but the actuator rejected it`;
  } else if (executed && motion.has(record.token)) {
    const before = metres(depth?.center_m);
    const after = metres(afterDepth?.center_m);
    if (record.token === forward && before !== null && after !== null) {
      const delta = before - after;
      if (after < 0.4) {
        score -= 0.3;
        outcome = "too_close";
        lesson = `${phrase}: ${forward} brought the centre depth to ${after.toFixed(2)} m; stop or turn earlier`;
      } else if (Math.abs(delta) < 0.03) {
        score -= 0.3;
        outcome = "stalled";
        lesson = `${phrase}: ${forward} produced no forward progress (deadband or blocked); try ${alternative}`;
      } else {
        score += 0.3;
        outcome = "progress";
        lesson = `${phrase}: ${forward} advanced ${delta.toFixed(2)} m safely`;
      }
    } else {
      score += 0.2;
      outcome = "progress";
      lesson = `${phrase}: ${record.token} executed cleanly`;
    }
  } else if (record.token === stop) {
    outcome = "stopped";
  } else if (!executed) {
    outcome = "preview";
  }

  if (signals && signals.oscillation >= 2 && motion.has(record.token)) {
    score -= 0.3;
    if (outcome === "neutral" || outcome === "progress" || outcome === "preview") {
      outcome = "oscillating";
      lesson = `${phrase}: alternating ${record.token}/${profile.opposite[record.token] ?? "?"} without progress; commit to one side or stop`;
    }
  }

  return {
    score: clamp(score),
    outcome,
    alternative,
    situation,
    attributable,
    lesson: lesson
      ? { taskKey, situation, token: record.token, outcome, score: clamp(score), lesson, alternative, source: "heuristic" }
      : null,
  };
}

/**
 * Convert an asynchronous review into a judgement of the step it reviewed. CORRECT and STOP are the
 * high-value lessons: they name the token that should have been chosen.
 */
export function judgeReview({ review, token, depth = null, lidar = null, state = null, taskKey = "generic", engine = null, profile = G1_PROFILE } = {}) {
  if (!review || typeof review.verdict !== "string") return null;
  const record = profile.actionOrStop(token);
  const situation = situationOf({ depth, lidar, state, profile });
  const phrase = situationPhrase(situation);
  const brief = typeof review.brief === "string" && review.brief.trim() ? review.brief.trim() : null;
  const confidence = Number.isFinite(review.confidence) ? Math.min(1, Math.max(0, review.confidence)) : 0.5;
  let score;
  let outcome;
  let lesson;
  if (review.verdict === "STOP") {
    score = -0.8 * (0.5 + confidence / 2);
    outcome = "review_stop";
    lesson = `${phrase}: reviewer stopped ${record.token}${brief ? ` — ${brief}` : ""}`;
  } else if (review.verdict === "CORRECT") {
    score = -0.5 * (0.5 + confidence / 2);
    outcome = "review_correct";
    lesson = `${phrase}: reviewer replaced ${record.token} with ${review.next_token}${brief ? ` — ${brief}` : ""}`;
  } else if (review.verdict === "KEEP") {
    score = 0.2 * (0.5 + confidence / 2);
    outcome = "review_keep";
    lesson = `${phrase}: reviewer confirmed ${record.token}${brief ? ` — ${brief}` : ""}`;
  } else {
    return null;
  }
  return {
    score: clamp(score),
    outcome,
    situation,
    alternative: review.verdict === "KEEP" ? null : review.next_token ?? null,
    engine,
    lesson: {
      taskKey,
      situation,
      token: record.token,
      outcome,
      score: clamp(score),
      lesson: lesson.slice(0, 240),
      alternative: review.verdict === "KEEP" ? null : review.next_token ?? null,
      source: "reviewer",
    },
  };
}

/**
 * Convert an asynchronous judge's outcome assessment of a past step into a lesson. This is the judge
 * the reviewer becomes once it has seen the after frame: unlike `judgeReview()`, which converts the
 * steering verdict, this carries a model-authored lesson about *what happened*, weighted by the
 * judge's confidence. Nothing here acts; the lesson only changes what is asked next time.
 */
export function judgeOutcome({ judgement, token, depth = null, lidar = null, state = null, taskKey = "generic", engine = null, profile = G1_PROFILE } = {}) {
  if (!judgement || !JUDGE_OUTCOMES.includes(judgement.outcome)) return null;
  const record = profile.actionOrStop(token);
  const situation = situationOf({ depth, lidar, state, profile });
  const phrase = situationPhrase(situation);
  const confidence = Number.isFinite(judgement.confidence) ? Math.min(1, Math.max(0, judgement.confidence)) : 0.5;
  const score = clamp(OUTCOME_SCORE[judgement.outcome] * (0.5 + confidence / 2));
  const text = typeof judgement.lesson === "string" && judgement.lesson.trim() ? judgement.lesson.trim() : null;
  const alternative =
    judgement.verdict && judgement.verdict !== "KEEP" && profile.tokenSet.has(judgement.next_token) && judgement.next_token !== record.token
      ? judgement.next_token
      : null;
  const outcome = `review_${judgement.outcome}`;
  // An unclear outcome with nothing to say is not a judgement at all: no lesson, no routing score.
  if (judgement.outcome === "unclear" && !text) return null;
  const lesson = `${phrase}: ${record.token} ${OUTCOME_PHRASE[judgement.outcome]}${text ? ` — ${text}` : ""}`.slice(0, 240);
  return {
    score,
    outcome,
    situation,
    alternative,
    engine,
    lesson: { taskKey, situation, token: record.token, outcome, score, lesson, alternative, source: "reviewer" },
  };
}

/** Validate an outcome judgement posted about a past session step (loopback endpoint). */
export function externalJudgementError(body, profile = G1_PROFILE) {
  if (body === null || typeof body !== "object" || Array.isArray(body)) return "judgement must be an object";
  if (typeof body.sessionId !== "string" || body.sessionId.trim().length === 0) return "sessionId must be a non-empty string";
  if (!Number.isSafeInteger(body.basedOnSeq) || body.basedOnSeq < 0) return "basedOnSeq must be a non-negative integer";
  if (!JUDGE_OUTCOMES.includes(body.outcome)) return `outcome must be one of ${JUDGE_OUTCOMES.join(", ")}`;
  if (body.lesson !== undefined && body.lesson !== null && (typeof body.lesson !== "string" || body.lesson.length > 200)) return "lesson must be a string of at most 200 characters";
  if (body.confidence !== undefined && body.confidence !== null && (typeof body.confidence !== "number" || !Number.isFinite(body.confidence) || body.confidence < 0 || body.confidence > 1)) {
    return "confidence must be a number in [0, 1]";
  }
  if (body.verdict !== undefined && body.verdict !== null && !VERDICTS.has(body.verdict)) return "verdict must be KEEP, CORRECT or STOP";
  if (body.next_token !== undefined && body.next_token !== null && !profile.tokenSet.has(body.next_token)) return "next_token is not in the token set";
  if (body.token !== undefined && body.token !== null && !profile.tokenSet.has(body.token)) return "token is not in the token set";
  return null;
}

/** Validate a lesson posted by an external judge (loopback endpoint). */
export function externalLessonError(body, profile = G1_PROFILE) {
  if (body === null || typeof body !== "object" || Array.isArray(body)) return "lesson must be an object";
  if (typeof body.lesson !== "string" || body.lesson.trim().length === 0 || body.lesson.length > 240) return "lesson text must be 1..240 characters";
  if (typeof body.score !== "number" || !Number.isFinite(body.score) || body.score < -1 || body.score > 1) return "score must be in [-1, 1]";
  if (typeof body.outcome !== "string" || body.outcome.length === 0 || body.outcome.length > 40) return "outcome is required";
  if (body.token !== undefined && body.token !== null && profile.actionOrStop(body.token).token !== String(body.token).toUpperCase()) return "token is not in the token set";
  return null;
}
