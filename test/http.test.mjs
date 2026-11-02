import assert from "node:assert/strict";
import test from "node:test";

import { createHttpServer } from "../src/http.mjs";
import { VERSION } from "../src/version.mjs";

function fakeAgent({ onStep = async () => ({ ok: true }) } = {}) {
  return {
    status: async () => ({ ok: true, ready: false, fsm_id: 1 }),
    capabilities: () => ({ service: "rea-agent", engines: { prime: { default: true } } }),
    decideOnly: async () => ({ ok: true, token: "X", action: { action: "STOP", duration_ms: 0 }, engine: "prime", timing: {} }),
    act: async () => ({ ok: true }),
    step: onStep,
    sense: async () => ({ ok: true }),
    arm: async () => ({ ok: true }),
    stop: async () => ({ ok: true }),
    acceptCorrection: () => ({ accepted: false, reason: "unknown_session" }),
    pipeline: { status: () => ({ running: false }), set: () => ({ running: false }) },
  };
}

async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  return `http://127.0.0.1:${address.port}`;
}

test("health and capabilities identify the ready Prime Agent RPC harness", async t => {
  const server = createHttpServer({
    agent: fakeAgent(),
    config: {
      providerConfigured: true,
      cameraSocket: null,
      engine: "prime",
      agentHarness: "prime-agent-rpc",
      primeAgentVersion: "0.8.0",
      primaryModel: "deepseek-v4-flash-vision-exp",
      fallbackModel: "gemini-3.5-flash-lite",
      executeEnabled: false,
      actuator: "none",
      persistentControl: false,
      maxOutputTokens: 64,
    },
    harnessState: () => ({ ready: true, modelReady: true, status: "ready", poolSize: 2 }),
  });
  t.after(() => server.close());
  const base = await listen(server);

  const health = await (await fetch(`${base}/health`)).json();
  assert.equal(health.ready, true);
  assert.equal(health.version, VERSION);
  assert.equal(health.harness, "prime-agent-rpc");
  assert.equal(health.harness_version, "0.8.0");
  assert.equal(health.model_output_tokens, 1);
  assert.equal(health.actuator_attached, false);
  assert.equal(health.actuator_kind, "none");
  assert.equal(health.execute_enabled, false);
  assert.equal(health.persistent_control, false);
  assert.deepEqual(health.harness_state, { ready: true, modelReady: true, status: "ready", poolSize: 2 });

  const capabilities = await (await fetch(`${base}/v1/control/capabilities`)).json();
  assert.equal(capabilities.ok, true);
  assert.equal(capabilities.harness, "prime-agent-rpc");
  assert.equal(capabilities.direct, true);
  assert.equal(capabilities.actuator_attached, false);
});

test("website control route is always preview-only even when the caller asks to execute", async t => {
  let received = null;
  const server = createHttpServer({
    agent: fakeAgent({
      onStep: async body => {
        received = body;
        return {
          ok: true,
          session_id: "session-1",
          seq: 1,
          token: "W",
          action: { action: "FORWARD", duration_ms: 1000 },
          model_used: "deepseek-v4-flash-vision-exp",
          engine: "prime",
          executed: false,
          preview: true,
          timing: { total_s: 1.1 },
          error_type: null,
          error: null,
        };
      },
    }),
    config: {
      providerConfigured: true,
      cameraSocket: null,
      engine: "prime",
      agentHarness: "prime-agent-rpc",
      primeAgentVersion: "0.8.0",
      executeEnabled: true,
      actuator: "daemon",
      persistentControl: true,
      maxOutputTokens: 64,
    },
    harnessState: () => ({ ready: true, modelReady: true }),
  });
  t.after(() => server.close());
  const base = await listen(server);

  const response = await fetch(`${base}/v1/control/step`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ instruction: "move toward the opening", execute: true }),
  });
  const body = await response.json();
  assert.equal(response.status, 200);
  assert.equal(received.execute, false);
  assert.equal(body.harness, "prime-agent-rpc");
  assert.equal(body.token, "W");
  assert.equal(body.actuator_attached, false);
  assert.equal(body.executed, false);
});

test("a session that is already stepping returns conflict", async t => {
  const server = createHttpServer({
    agent: fakeAgent({ onStep: async () => ({ ok: false, session_id: "s1", error_type: "session_busy", error: "busy" }) }),
    config: { providerConfigured: true, cameraSocket: null, engine: "token", executeEnabled: false, actuator: "none", persistentControl: false, maxOutputTokens: 64 },
  });
  t.after(() => server.close());
  const base = await listen(server);
  for (const route of ["/v1/step", "/v1/control/step"]) {
    const response = await fetch(`${base}${route}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ session_id: "s1" }) });
    assert.equal(response.status, 409);
    assert.equal((await response.json()).error_type, "session_busy");
  }
});

test("the judge review channel is loopback-only and maps brain answers to statuses", async t => {
  const received = [];
  const agent = {
    ...fakeAgent(),
    brain: { state: () => ({ summary: {} }) },
    acceptJudgement: body => {
      received.push(body);
      if (body.sessionId === "missing") return { accepted: false, reason: "unknown_session" };
      if (body.basedOnSeq === 99) return { accepted: false, reason: "unknown_seq" };
      if (body.outcome === "great") return { accepted: false, reason: "invalid_judgement", error: "outcome" };
      return { accepted: true, sessionId: body.sessionId, basedOnSeq: body.basedOnSeq, learned: { score: -0.9, outcome: "review_unsafe", lesson: "l" } };
    },
  };
  const server = createHttpServer({
    agent,
    config: { providerConfigured: true, cameraSocket: null, engine: "token", agentHarness: "rea-token", executeEnabled: false, actuator: "none", persistentControl: false, maxOutputTokens: 64 },
  });
  t.after(() => server.close());
  const base = await listen(server);
  const post = async body => {
    const response = await fetch(`${base}/internal/judge/review`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    return { status: response.status, body: await response.json() };
  };
  const ok = await post({ sessionId: "s1", basedOnSeq: 1, outcome: "unsafe", lesson: "l" });
  assert.equal(ok.status, 202);
  assert.equal(ok.body.ok, true);
  assert.equal(ok.body.learned.outcome, "review_unsafe");
  assert.equal((await post({ sessionId: "missing", basedOnSeq: 1, outcome: "unsafe" })).status, 404);
  assert.equal((await post({ sessionId: "s1", basedOnSeq: 99, outcome: "unsafe" })).status, 409);
  assert.equal((await post({ sessionId: "s1", basedOnSeq: 1, outcome: "great" })).status, 422);
  assert.equal(received.length, 4);

  const off = createHttpServer({ agent: { ...fakeAgent(), brain: null, acceptJudgement: () => ({ accepted: false }) }, config: { providerConfigured: true, cameraSocket: null, engine: "token", executeEnabled: false, actuator: "none", persistentControl: false, maxOutputTokens: 64 } });
  t.after(() => off.close());
  const offBase = await listen(off);
  const disabled = await fetch(`${offBase}/internal/judge/review`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ sessionId: "s1", basedOnSeq: 1, outcome: "unsafe" }) });
  assert.equal(disabled.status, 404);
  assert.equal((await disabled.json()).error_type, "brain_disabled");
});

test("a configured bearer token guards every route except the health probe", async t => {
  const server = createHttpServer({
    agent: fakeAgent(),
    config: { providerConfigured: true, cameraSocket: null, engine: "token", agentHarness: "rea-token", harnessVersion: "0.2.0", executeEnabled: false, actuator: "none", persistentControl: false, maxOutputTokens: 64, agentToken: "s3cret-token", host: "0.0.0.0" },
  });
  t.after(() => server.close());
  const base = await listen(server);
  const health = await fetch(`${base}/health`);
  assert.equal(health.status, 200);
  const healthBody = await health.json();
  assert.equal(healthBody.auth_required, true);
  assert.equal(healthBody.harness_version, "0.2.0", "a non-Prime engine reports this service's version");

  const open = await fetch(`${base}/v1/capabilities`);
  assert.equal(open.status, 401);
  assert.equal((await open.json()).error_type, "unauthorized");
  const act = await fetch(`${base}/v1/act`, { method: "POST", headers: { "content-type": "application/json", authorization: "Bearer wrong" }, body: "{}" });
  assert.equal(act.status, 401);
  const stop = await fetch(`${base}/v1/stop`, { method: "POST", headers: { authorization: "Bearer s3cret-token" } });
  assert.equal(stop.status, 200);
  const caps = await fetch(`${base}/v1/capabilities`, { headers: { authorization: "bearer s3cret-token" } });
  assert.equal(caps.status, 200, "the scheme is case-insensitive");
  const internal = await fetch(`${base}/internal/thinker/correction`, { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
  assert.equal(internal.status, 401, "loopback internal routes need the token too when one is set");

  const openServer = createHttpServer({ agent: fakeAgent(), config: { providerConfigured: true, cameraSocket: null, engine: "token", executeEnabled: false, actuator: "none", persistentControl: false, maxOutputTokens: 64 } });
  t.after(() => openServer.close());
  const openBase = await listen(openServer);
  assert.equal((await fetch(`${openBase}/v1/capabilities`)).status, 200, "no token configured => the historical open service");
  assert.equal((await (await fetch(`${openBase}/health`)).json()).auth_required, false);
});

test("the brain reset is loopback-only and forgets experience and routing memory, not episodes unless asked", async t => {
  const calls = [];
  const agent = { ...fakeAgent(), brain: { state: () => ({ summary: {} }), reset: options => { calls.push(options); return { experience: options.experience, router: options.router, episodes: options.episodes, before: { lessons: 3 } }; } } };
  const server = createHttpServer({ agent, logger: { info() {}, warn() {}, error() {} }, config: { providerConfigured: true, cameraSocket: null, engine: "token", executeEnabled: false, actuator: "none", persistentControl: false, maxOutputTokens: 64 } });
  t.after(() => server.close());
  const base = await listen(server);
  const unconfirmed = await fetch(`${base}/internal/brain/reset`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({}) });
  assert.equal(unconfirmed.status, 400, "wiping persisted memory has to be confirmed");
  assert.equal((await unconfirmed.json()).error_type, "confirm_required");
  assert.equal(calls.length, 0);
  const response = await fetch(`${base}/internal/brain/reset`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ confirm: true }) });
  assert.equal(response.status, 202);
  const body = await response.json();
  assert.equal(body.ok, true);
  assert.deepEqual(calls[0], { experience: true, router: true, episodes: false });
  await fetch(`${base}/internal/brain/reset`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ confirm: true, experience: false, episodes: true }) });
  assert.deepEqual(calls[1], { experience: false, router: true, episodes: true });
});

test("a request error from a decision maps to 400 on every decision route", async t => {
  const failing = { ok: false, errorType: "request_error", error_type: "request_error", error: "unknown engine 'x'; known: token", token: "X", action: { action: "STOP", duration_ms: 0 }, attempts: [], timing: {} };
  const server = createHttpServer({
    agent: { ...fakeAgent({ onStep: async () => ({ ...failing, session_id: "s1", seq: 1 }) }), decideOnly: async () => failing, act: async () => failing },
    config: { providerConfigured: true, cameraSocket: null, engine: "token", executeEnabled: false, actuator: "none", persistentControl: false, maxOutputTokens: 64 },
    logger: { info() {}, warn() {}, error() {} },
  });
  t.after(() => server.close());
  const base = await listen(server);
  for (const route of ["/v1/action", "/v1/act", "/v1/step"]) {
    const response = await fetch(`${base}${route}`, { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
    assert.equal(response.status, 400, route);
  }
});

test("a provider-free engine reports ready without a key and one output token", async t => {
  const server = createHttpServer({
    agent: fakeAgent(),
    config: { providerConfigured: false, providerRequired: false, cameraSocket: null, engine: "scripted", modelOutputTokens: 1, executeEnabled: false, actuator: "none", persistentControl: false, maxOutputTokens: 64 },
  });
  t.after(() => server.close());
  const base = await listen(server);
  const health = await (await fetch(`${base}/health`)).json();
  assert.equal(health.ready, true);
  assert.equal(health.model_output_tokens, 1);
});

test("the fixture camera counts as ready without a socket", async t => {
  const server = createHttpServer({ agent: fakeAgent(), config: { providerConfigured: true, cameraSocket: "/definitely/not/a/socket", sense: "fixture", engine: "token", executeEnabled: false, actuator: "none", persistentControl: false, maxOutputTokens: 64 } });
  t.after(() => server.close());
  const base = await listen(server);
  const health = await (await fetch(`${base}/health`)).json();
  assert.equal(health.camera_ready, true);
  assert.equal(health.ready, true);
});
