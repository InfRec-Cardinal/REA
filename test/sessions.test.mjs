import assert from "node:assert/strict";
import { test } from "node:test";

import { createSessionStore } from "../src/sessions.mjs";
import HOLONOMIC from "./fixtures/profiles/holonomic.mjs";

function correction(overrides = {}) {
  return {
    sessionId: "s1",
    basedOnSeq: 1,
    verdict: "CORRECT",
    next_token: "Q",
    brief: "wall ahead on the left",
    confidence: 0.8,
    ...overrides,
  };
}

function storeAtSeq(seq, lastToken = "W") {
  let n = 0;
  const store = createSessionStore({ idFactory: () => `s${++n}` });
  const session = store.open("go");
  store.commit(session, { seq, token: lastToken });
  return { store, session };
}

test("a correction for the current sequence is accepted and consumed once", () => {
  const { store, session } = storeAtSeq(1);
  assert.equal(store.acceptCorrection(correction()).accepted, true);
  const advice = store.takeAdvice(session);
  assert.equal(advice.next_token, "Q");
  assert.equal(store.takeAdvice(session), null, "advice must not be applied twice");
});

test("a correction for a sequence the loop has moved past is stale", () => {
  const { store } = storeAtSeq(5);
  const result = store.acceptCorrection(correction({ basedOnSeq: 3 }));
  assert.equal(result.accepted, false);
  assert.equal(result.reason, "stale_correction");
});

test("a correction for an unknown session is rejected", () => {
  const { store } = storeAtSeq(1);
  assert.equal(store.acceptCorrection(correction({ sessionId: "nope" })).reason, "unknown_session");
});

test("the correction schema is exact: no missing and no extra keys", () => {
  const { store } = storeAtSeq(1);
  const { brief: _drop, ...missing } = correction();
  assert.equal(store.acceptCorrection(missing).reason, "invalid_correction");
  assert.equal(store.acceptCorrection({ ...correction(), extra: 1 }).reason, "invalid_correction");
  assert.equal(store.acceptCorrection(null).reason, "invalid_correction");
  assert.equal(store.acceptCorrection([]).reason, "invalid_correction");
});

test("a STOP verdict must recommend the stop token", () => {
  const { store } = storeAtSeq(1);
  assert.equal(store.acceptCorrection(correction({ verdict: "STOP", next_token: "W" })).reason, "invalid_correction");
  assert.equal(store.acceptCorrection(correction({ verdict: "STOP", next_token: "X" })).accepted, true);
});

test("a KEEP verdict must preserve the token it is keeping", () => {
  const { store } = storeAtSeq(1, "W");
  assert.equal(store.acceptCorrection(correction({ verdict: "KEEP", next_token: "S" })).reason, "invalid_correction");
  assert.equal(store.acceptCorrection(correction({ verdict: "KEEP", next_token: "W" })).accepted, true);
});

test("a token outside the fixed set is rejected", () => {
  const { store } = storeAtSeq(1);
  assert.equal(store.acceptCorrection(correction({ next_token: "Z" })).reason, "invalid_token");
});

test("confidence and brief are range-checked", () => {
  const { store } = storeAtSeq(1);
  assert.equal(store.acceptCorrection(correction({ confidence: 1.5 })).reason, "invalid_correction");
  assert.equal(store.acceptCorrection(correction({ confidence: Number.NaN })).reason, "invalid_correction");
  assert.equal(store.acceptCorrection(correction({ brief: "" })).reason, "invalid_correction");
  assert.equal(store.acceptCorrection(correction({ brief: "x".repeat(201) })).reason, "invalid_correction");
});

test("the session store evicts rather than growing without bound", () => {
  let n = 0;
  const store = createSessionStore({ idFactory: () => `s${++n}`, maxSessions: 3 });
  for (let i = 0; i < 10; i += 1) store.open("go");
  assert.equal(store.size(), 3);
});

test("a stale correction must still be a well-formed correction before anyone may learn from it", () => {
  const store = createSessionStore({ idFactory: () => "s" });
  const session = store.open("go");
  store.commit(session, { seq: 3, token: "W" });
  const base = { sessionId: "s", basedOnSeq: 1, verdict: "CORRECT", next_token: "Q", brief: "door left", confidence: 0.8 };
  assert.equal(store.acceptCorrection(base).reason, "stale_correction");
  assert.equal(store.acceptCorrection({ ...base, next_token: "Z" }).reason, "invalid_token", "an unknown token is invalid, not merely stale");
  assert.equal(store.acceptCorrection({ ...base, verdict: "STOP", next_token: "W" }).reason, "invalid_correction", "STOP must carry X even when stale");
  assert.equal(store.acceptCorrection({ ...base, basedOnSeq: 3, next_token: "Z" }).reason, "invalid_token");
});

test("a session store built on another profile validates tokens and STOP against that profile", () => {
  let n = 0;
  const store = createSessionStore({ idFactory: () => `h${++n}`, profile: HOLONOMIC });
  const session = store.open("go");
  store.commit(session, { seq: 1, token: "F" });
  const base = { sessionId: session.id, basedOnSeq: 1, brief: "wall", confidence: 0.9 };
  assert.equal(store.acceptCorrection({ ...base, verdict: "CORRECT", next_token: "Q" }).reason, "invalid_token");
  assert.equal(store.acceptCorrection({ ...base, verdict: "STOP", next_token: "X" }).reason, "invalid_token", "X is not a token here");
  assert.equal(store.acceptCorrection({ ...base, verdict: "STOP", next_token: "F" }).reason, "invalid_correction", "STOP must carry this profile's stop");
  assert.equal(store.acceptCorrection({ ...base, verdict: "STOP", next_token: "H" }).accepted, true);
});
