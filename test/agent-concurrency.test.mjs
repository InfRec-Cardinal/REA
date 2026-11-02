import assert from "node:assert/strict";
import test from "node:test";

import { createAgent } from "../src/agent.mjs";
import { loadConfig } from "../src/config.mjs";
import { createSessionStore } from "../src/sessions.mjs";

const FRAME = { imageB64: "aGk=", mimeType: "image/jpeg", depth: { center_m: 3, left_m: 3, right_m: 3 } };

function preview() {
  return {
    gate: { allowed: true, reasons: [] },
    executed: false,
    preview: true,
    timing: { gate_s: 0, exec_s: null },
  };
}

function build({ capture = async () => FRAME, execute = async () => preview() } = {}) {
  const config = loadConfig({ G1_ENGINE: "token", REA_API_KEY: "test-only", REA_MODEL: "test-model", G1_USE_LIDAR: "0" });
  const sessions = createSessionStore();
  const token = async () => ({ token: "X", tokenReadyS: 0 });
  token.engine = "token";
  const agent = createAgent({ camera: { capture }, engines: { token }, executor: { execute }, config, sessions });
  return { agent, sessions };
}

test("a busy session rejects overlapping steps without changing its task or blocking other sessions", async () => {
  const pendingFrame = Promise.withResolvers();
  let captures = 0;
  const { agent, sessions } = build({
    capture: () => {
      captures += 1;
      return captures === 1 ? pendingFrame.promise : Promise.resolve(FRAME);
    },
  });
  const session = sessions.open("first task");
  const independent = sessions.open("independent task");
  const first = agent.step({ session_id: session.id });
  try {
    const busy = await agent.step({ session_id: session.id, instruction: "rejected task" });
    assert.equal(busy.ok, false);
    assert.equal(busy.error_type, "session_busy");
    assert.equal(busy.session_id, session.id);
    assert.equal(session.task, "first task");
    assert.equal(session.seq, 0);
    assert.equal(captures, 1);

    const other = await agent.step({ session_id: independent.id });
    assert.equal(other.ok, true);
    assert.equal(other.seq, 1);
  } finally {
    pendingFrame.resolve(FRAME);
  }

  assert.equal((await first).seq, 1);
  const next = await agent.step({ session_id: session.id });
  assert.equal(next.ok, true);
  assert.equal(next.seq, 2);
  assert.equal(session.seq, 2);
});

test("a failed step releases its session for the next request", async () => {
  let executions = 0;
  const { agent, sessions } = build({
    execute: async () => {
      executions += 1;
      if (executions === 1) throw new Error("executor unavailable");
      return preview();
    },
  });
  const session = sessions.open("retry task");
  await assert.rejects(agent.step({ session_id: session.id }), /executor unavailable/);
  const next = await agent.step({ session_id: session.id });
  assert.equal(next.ok, true);
  assert.equal(next.seq, 1);
  assert.equal(session.seq, 1);
});
