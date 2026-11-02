import { randomUUID } from "node:crypto";

import { G1_PROFILE } from "./profiles/g1.mjs";

const VERDICTS = new Set(["KEEP", "CORRECT", "STOP"]);
const CORRECTION_KEYS = ["basedOnSeq", "brief", "confidence", "next_token", "sessionId", "verdict"];

/**
 * Session store for the closed-loop control route.
 *
 * Each session holds a monotonically increasing `seq` and at most one pending correction from the
 * asynchronous thinker. The correction protocol is deliberately strict — exact key set, verdict/token
 * agreement, and a `basedOnSeq` that must match the session's current sequence — because this is the
 * one path where an out-of-band service influences what the robot does next. A correction that
 * arrives late (the loop has already moved on) is stale by definition and is dropped, not applied.
 */
export function createSessionStore({ idFactory = () => randomUUID(), maxSessions = 64, tokens = null, profile = G1_PROFILE } = {}) {
  const sessions = new Map();
  const tokenSet = new Set(tokens ?? profile.TOKENS);
  const stop = profile.roles.stop;

  function evict() {
    while (sessions.size > maxSessions) {
      const oldest = sessions.keys().next().value;
      sessions.delete(oldest);
    }
  }

  function open(instruction, defaults = {}) {
    const sessionId = idFactory();
    const session = {
      id: sessionId,
      seq: 0,
      task: instruction,
      cfg: defaults.cfg ?? {},
      correction: null,
      lastToken: null,
      createdAt: Date.now(),
    };
    sessions.set(sessionId, session);
    evict();
    return session;
  }

  function get(sessionId) {
    return sessions.get(sessionId) ?? null;
  }

  /** Pending advice for the step about to run, consumed exactly once. */
  function takeAdvice(session) {
    const advice = session.correction?.basedOnSeq === session.seq ? session.correction : null;
    session.correction = null;
    return advice;
  }

  function commit(session, { seq, token }) {
    session.seq = seq;
    session.lastToken = token;
  }

  function acceptCorrection(correction) {
    if (correction === null || typeof correction !== "object" || Array.isArray(correction)) {
      return { accepted: false, reason: "invalid_correction" };
    }
    if (Object.keys(correction).sort().join(" ") !== CORRECTION_KEYS.join(" ")) {
      return { accepted: false, reason: "invalid_correction" };
    }
    if (
      !VERDICTS.has(correction.verdict) ||
      typeof correction.brief !== "string" ||
      correction.brief.length < 1 ||
      correction.brief.length > 200 ||
      typeof correction.confidence !== "number" ||
      !Number.isFinite(correction.confidence) ||
      correction.confidence < 0 ||
      correction.confidence > 1
    ) {
      return { accepted: false, reason: "invalid_correction" };
    }
    // Token validity and STOP/stop-token agreement are properties of the correction itself, so they
    // are checked before staleness: a stale verdict is still learned from, and only a well-formed one may be.
    if (!tokenSet.has(correction.next_token)) return { accepted: false, reason: "invalid_token" };
    if (correction.verdict === "STOP" && correction.next_token !== stop) {
      return { accepted: false, reason: "invalid_correction" };
    }
    const session = sessions.get(correction.sessionId);
    if (!session) return { accepted: false, reason: "unknown_session" };
    if (!Number.isSafeInteger(correction.basedOnSeq) || correction.basedOnSeq !== session.seq) {
      return { accepted: false, reason: "stale_correction" };
    }
    if (correction.verdict === "KEEP" && correction.next_token !== session.lastToken) {
      return { accepted: false, reason: "invalid_correction" };
    }
    session.correction = { ...correction };
    return { accepted: true, sessionId: correction.sessionId, basedOnSeq: correction.basedOnSeq };
  }

  return {
    open,
    get,
    takeAdvice,
    commit,
    acceptCorrection,
    size: () => sessions.size,
  };
}
