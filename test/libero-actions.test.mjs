import assert from "node:assert/strict";
import { test } from "node:test";

import { createAgent } from "../src/agent.mjs";
import { createSessionStore } from "../src/sessions.mjs";
import { ACTION_TABLE as G1_ACTION_TABLE } from "../src/actions.mjs";
import { ACTION_TABLE, CAMERA_NOTE, CM_PER_TOKEN, DEG_PER_TOKEN, LIBERO_ACTION_CONTRACT, TOKENS, TOKEN_LEGEND, parseToken } from "../src/libero-actions.mjs";
import { LIBERO_PROFILE } from "../src/profiles/libero.mjs";
import HOLONOMIC from "./fixtures/profiles/holonomic.mjs";

const config = { providerConfigured: true, engine: "token", primaryModel: "test-model", fallbackModel: "", primaryTimeoutMs: 1000, useLidar: false };

function buildAgent({ token = "O", error = null, actionContract = LIBERO_ACTION_CONTRACT } = {}) {
  const executed = [];
  const infer = Object.assign(async () => {
    if (error) throw error;
    return { token, tokenReadyS: 0.01, usage: { completion_tokens: 1 }, timings: { token_ready_s: 0.01 } };
  }, { engine: "token" });
  const sessions = createSessionStore({ tokens: TOKENS });
  const agent = createAgent({
    camera: { capture: async () => ({ imageB64: "aGk=", libero: { proprio: {} } }) },
    engines: { token: infer },
    executor: {
      async execute(request) {
        executed.push(request);
        return { executed: true, preview: false, gate: { allowed: true }, dispatch: { ok: true }, timing: { gate_s: 0, exec_s: 0 } };
      },
    },
    config,
    sessions,
    ...(actionContract ? { actionContract } : {}),
  });
  return { agent, sessions, executed };
}

test("LIBERO tokens include gripper and 6D motion without changing G1 records", () => {
  assert.equal(new Set(TOKENS).size, 15);
  for (const token of TOKENS) {
    assert.equal(parseToken(token), token);
    assert.equal(ACTION_TABLE[token].repeatSteps, 5);
  }
  assert.equal(ACTION_TABLE.R.action, "world_z_negative");
  assert.equal(ACTION_TABLE.O.action, "open_gripper");
  assert.equal(ACTION_TABLE.X.action, "hold");
  assert.equal(G1_ACTION_TABLE.R.action, "release_arm");
  assert.equal(G1_ACTION_TABLE.O, undefined);
});

test("the legend text changed but the token semantics and per-token motion constants are fixed", () => {
  const expected = {
    W: "world_x_positive", S: "world_x_negative", A: "world_y_positive", D: "world_y_negative",
    U: "world_z_positive", R: "world_z_negative", Q: "yaw_positive", E: "yaw_negative",
    I: "roll_positive", K: "roll_negative", J: "pitch_positive", L: "pitch_negative",
    P: "close_gripper", O: "open_gripper", X: "hold",
  };
  for (const [token, action] of Object.entries(expected)) {
    assert.equal(ACTION_TABLE[token].action, action);
    assert.equal(ACTION_TABLE[token].kind, token === "X" ? "hold" : ["P", "O"].includes(token) ? "gripper" : "osc_pose");
    assert.equal(ACTION_TABLE[token].defaultMs, 250);
    assert.equal(ACTION_TABLE[token].amplitude, 0.5);
    assert.equal(typeof ACTION_TABLE[token].description, "string");
  }
  const legendLines = TOKEN_LEGEND.split("\n");
  assert.equal(legendLines.length, TOKENS.length);
  for (const token of TOKENS) {
    const matches = legendLines.filter(line => line.startsWith(`${token} = `));
    assert.equal(matches.length, 1, `token ${token} appears exactly once in TOKEN_LEGEND`);
  }
  assert.ok(TOKEN_LEGEND.includes("R = lower the gripper 2-3 cm toward the table (world -z): DOWN on the external view, the table grows in the wrist view"));
  assert.ok(TOKEN_LEGEND.includes("A = move 2-3 cm RIGHT on the external view (world +y): the scene shifts LEFT in the wrist view"));
  assert.ok(TOKEN_LEGEND.includes("P = start closing the gripper; send X next so the fingers finish closing before moving"));
  assert.deepEqual(CM_PER_TOKEN, [2.2, 3.2]);
  assert.deepEqual(DEG_PER_TOKEN, [12, 15]);
  assert.ok(Object.isFrozen(CM_PER_TOKEN) && Object.isFrozen(DEG_PER_TOKEN));
  assert.ok(CAMERA_NOTE.startsWith("External view (left image): screen RIGHT = world +y"));
  assert.ok(CAMERA_NOTE.includes("a target RIGHT of the gripper in the external view needs A, a target RIGHT of centre in the wrist view needs D."));
});

test("the real step lifecycle preserves LIBERO O and R records and session state", async () => {
  for (const token of ["O", "R"]) {
    const { agent, sessions, executed } = buildAgent({ token });
    const first = await agent.step({ instruction: "put the bowl on the plate", execute: true });
    assert.equal(first.ok, true);
    assert.equal(first.token, token);
    assert.equal(first.record, ACTION_TABLE[token]);
    assert.equal(first.action.action, ACTION_TABLE[token].name);
    assert.deepEqual(first.usage, { completion_tokens: 1 });
    assert.deepEqual(first.model_timings, { token_ready_s: 0.01 });
    assert.equal(executed[0].token, token);
    assert.equal(executed[0].durationMs, 250);
    assert.equal(sessions.get(first.session_id).lastToken, token);
    const second = await agent.step({ session_id: first.session_id, execute: true });
    assert.equal(second.seq, 2);
  }
});

test("act and capabilities use the injected manipulation contract", async () => {
  const { agent } = buildAgent();
  const forced = await agent.act({ force_token: "O", execute: true });
  assert.equal(forced.record, ACTION_TABLE.O);
  assert.equal(forced.token, "O");
  const capabilities = agent.capabilities();
  assert.deepEqual(capabilities.tokens, TOKENS);
  assert.equal(capabilities.manipulation.gripper_available, true);
});

test("the default action contract remains G1", async () => {
  const { agent } = buildAgent({ token: "R", actionContract: null });
  const result = await agent.step({ instruction: "release arms", execute: true });
  assert.equal(result.record, G1_ACTION_TABLE.R);
});

test("inference errors remain failed steps and hold does not imply success", async () => {
  const usage = { prompt_tokens: 200, completion_tokens: 1, total_tokens: 201 };
  const { agent, executed } = buildAgent({ error: Object.assign(new Error("provider unavailable"), { retryable: false, usage, outputText: "" }) });
  const result = await agent.step({ instruction: "lift bowl", execute: true });
  assert.equal(result.ok, false);
  assert.equal(result.record, ACTION_TABLE.X);
  assert.equal(result.token, "X");
  assert.equal(result.error, "provider unavailable");
  assert.deepEqual(result.usage, usage);
  assert.equal(result.output_text, "");
  assert.deepEqual(result.attempts[0].usage, usage);
  assert.equal(executed[0].token, "X");
  assert.equal(result.success, undefined);
});

test("a session can accept manipulation token corrections when configured", () => {
  const sessions = createSessionStore({ tokens: TOKENS });
  const session = sessions.open("lift bowl");
  sessions.commit(session, { seq: 1, token: "P" });
  const accepted = sessions.acceptCorrection({ sessionId: session.id, basedOnSeq: 1, verdict: "CORRECT", next_token: "O", brief: "open the gripper", confidence: 0.8 });
  assert.equal(accepted.accepted, true);
  assert.equal(sessions.takeAdvice(session).next_token, "O");
});

test("createAgent accepts a profile; a failed decision falls back to that profile's stop", async () => {
  let calls = 0;
  // the first decision fails (fail-closed path); later ones answer F so the wiggle path is exercised
  const infer = Object.assign(async () => {
    calls += 1;
    if (calls === 1) throw Object.assign(new Error("down"), { retryable: false });
    return { token: "F", tokenReadyS: 0 };
  }, { engine: "token" });
  const agent = createAgent({
    camera: { capture: async () => ({ imageB64: "aGk=", depth: { center_m: 2, left_m: 2, right_m: 2 } }) },
    engines: { token: infer },
    executor: { async execute(request) { return { token: request.token, executed: false, preview: true, gate: { allowed: true, reasons: [] }, dispatch: null, timing: { gate_s: 0, exec_s: null } }; }, armState: () => ({ held: null }), actuatorInfo: () => ({}) },
    config,
    sessions: createSessionStore({ profile: HOLONOMIC }),
    profile: HOLONOMIC,
    logger: { info() {}, warn() {}, error() {} },
  });
  const result = await agent.step({ instruction: "cross the room" });
  assert.equal(result.ok, false);
  assert.equal(result.token, "H");
  assert.equal(result.safe_stop, true);
  const capabilities = agent.capabilities();
  assert.deepEqual(capabilities.tokens, ["H", "F", "L", "R"]);
  assert.deepEqual(capabilities.manipulation, { gripper_available: false });
  const arm = await agent.arm({ execute: true });
  assert.equal(arm.error_type, "request_error");
  assert.match(arm.error, /action_id is required/u, "a profile without arms names no arm tokens");
  assert.equal((await agent.act({ instruction: "go", exec_mode: "wiggle", execute: true })).token, "H", "no turn role: wiggle degrades to the stop");
});

test("the LIBERO profile threads through createAgent exactly as the old contract did", async () => {
  const { agent } = buildAgent({ token: "O", actionContract: LIBERO_PROFILE });
  const result = await agent.step({ instruction: "open", execute: true });
  assert.equal(result.record, LIBERO_PROFILE.ACTION_TABLE.O);
});
