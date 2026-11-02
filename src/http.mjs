import { timingSafeEqual } from "node:crypto";
import { existsSync } from "node:fs";
import { createServer } from "node:http";

import { senseKindOf } from "./config.mjs";
import { outputTokensFor } from "./engines.mjs";
import { SERVICE, VERSION } from "./version.mjs";

const MAX_BODY_BYTES = 8 * 1024 * 1024;

function sendJson(response, status, payload) {
  const data = Buffer.from(`${JSON.stringify(payload)}\n`);
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": data.length,
    "cache-control": "no-store",
  });
  response.end(data);
}

async function readJson(request) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) {
      const error = new Error("request body is too large");
      error.status = 413;
      throw error;
    }
    chunks.push(chunk);
  }
  if (size === 0) return {};
  const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  if (body === null || typeof body !== "object" || Array.isArray(body)) {
    throw new Error("JSON body must be an object");
  }
  return body;
}

function isLoopback(address) {
  return address === "127.0.0.1" || address === "::1" || address === "::ffff:127.0.0.1";
}

/**
 * Shared-secret check. With no token configured every caller is accepted (the historical open
 * service); with one, every route except GET /health needs `Authorization: Bearer <token>`.
 */
export function authorised(request, token) {
  if (!token) return true;
  const match = /^Bearer\s+(\S+)\s*$/iu.exec(request.headers?.authorization ?? "");
  if (!match) return false;
  const offered = Buffer.from(match[1]);
  const expected = Buffer.from(token);
  return offered.length === expected.length && timingSafeEqual(offered, expected);
}

/** Decision-only response, in the shape both ancestors published on /v1/action. */
function decisionBody(result, config = {}) {
  const timing = {
    capture_s: result.timing?.captureS ?? null,
    model_action_ready_s: result.timing?.modelReadyS ?? null,
    action_ready_s: result.timing?.actionReadyS ?? null,
    total_s: result.timing?.totalS ?? null,
  };
  return {
    ok: result.ok,
    harness: config.agentHarness ?? "prime-agent-rpc",
    harness_version: config.harnessVersion ?? config.primeAgentVersion ?? null,
    open_loop: true,
    token: result.token,
    action: result.action,
    model_used: result.model,
    engine: result.engine,
    model_output_tokens: outputTokensFor(result.engine, null),
    fallback: result.fallbackUsed,
    ...timing,
    timing,
    model_timings: result.modelTimings ?? null,
    attempts: result.attempts,
    candidate: result.candidate ?? null,
    usage: result.usage ?? null,
    cost_usd: result.costUsd ?? null,
    error_type: result.errorType,
    error: result.error,
  };
}

function decisionStatus(result) {
  // A malformed request (an engine name nobody registered, say) is the caller's to fix, not ours.
  if (result.errorType === "request_error") return 400;
  if (result.errorType === "config_error") return 503;
  return 502;
}

function actStatus(result) {
  if (result.error_type === "config_error") return 503;
  if (result.error_type === "execute_disabled") return 409;
  if (result.error_type === "request_error") return 400;
  // A gate refusal is a correct, successful answer to "may I move?" — the answer is simply no.
  if (result.error_type === "gate_refused") return 200;
  return result.ok ? 200 : 502;
}

function stepStatus(result) {
  if (result.error_type === "unknown_session") return 404;
  if (result.error_type === "session_busy") return 409;
  if (result.error_type === "request_error") return 400;
  if (result.error_type === "config_error") return 503;
  if (result.error_type === "execute_disabled") return 409;
  if (result.error_type === "gate_refused") return 200;
  return result.ok ? 200 : 502;
}

function correctionStatus(result) {
  if (result.reason === "unknown_session") return 404;
  if (result.reason === "stale_correction") return 409;
  return 422;
}

export function createHttpServer({
  agent,
  config,
  logger = console,
  workers = {},
  harnessState = null,
}) {
  const routes = new Map();

  const post = (path, handler) => routes.set(`POST ${path}`, handler);
  const get = (path, handler) => routes.set(`GET ${path}`, handler);

  get("/health", async () => {
    const cameraReady = senseKindOf(config) === "fixture" ? true : (config.cameraSocket ? existsSync(config.cameraSocket) : true);
    const robot = await agent.status().catch((error) => ({ ok: false, ready: false, error: error.message }));
    const runtime = typeof harnessState === "function" ? harnessState() : null;
    const harnessReady = config.engine !== "prime" || (runtime?.ready === true && runtime?.modelReady === true);
    return {
      status: 200,
      body: {
        ok: true,
        ready: (config.providerConfigured !== false || config.providerRequired === false) && cameraReady && harnessReady,
        service: SERVICE,
        version: VERSION,
        harness: config.agentHarness ?? "prime-agent-rpc",
        harness_version: config.harnessVersion ?? config.primeAgentVersion ?? null,
        harness_state: runtime,
        auth_required: Boolean(config.agentToken),
        host: config.host ?? null,
        engine: config.engine,
        model_output_tokens: config.modelOutputTokens ?? outputTokensFor(config.engine, config.maxOutputTokens),
        primary_model: config.primaryModel,
        fallback_model: config.fallbackModel,
        camera_socket: config.cameraSocket,
        camera_ready: cameraReady,
        actuator_attached: config.actuator !== "none" && config.executeEnabled,
        actuator_kind: config.actuator,
        execute_enabled: config.executeEnabled,
        persistent_control: config.persistentControl,
        provider_configured: config.providerConfigured,
        robot,
        workers: Object.fromEntries(Object.entries(workers).map(([name, worker]) => [name, worker.health?.() ?? null])),
        pipeline: agent.pipeline.status(),
        brain: agent.brain ? { enabled: true, ...agent.brain.state().summary } : { enabled: false },
      },
    };
  });

  get("/v1/capabilities", async () => ({ status: 200, body: { ok: true, ...agent.capabilities() } }));
  get("/v1/control/capabilities", async () => ({
    status: 200,
    body: {
      ok: true,
      ...agent.capabilities(),
      harness: config.agentHarness ?? "prime-agent-rpc",
      harness_version: config.harnessVersion ?? config.primeAgentVersion ?? null,
      direct: true,
      reasoning: false,
      model_output_tokens: 1,
      actuator_attached: false,
    },
  }));

  post("/v1/action", async (body) => {
    const result = await agent.decideOnly(body);
    const payload = decisionBody(result, config);
    (payload.ok ? logger.info : logger.warn).call(logger, {
      event: "decision",
      ok: payload.ok,
      token: payload.token,
      action: payload.action,
      engine: payload.engine,
      model_used: payload.model_used,
      capture_s: payload.capture_s,
      model_action_ready_s: payload.model_action_ready_s,
      total_s: payload.total_s,
      error_type: payload.error_type,
      error: payload.error,
    });
    return { status: payload.ok ? 200 : decisionStatus(result), body: payload };
  });

  post("/v1/act", async (body) => {
    const result = await agent.act(body);
    (result.ok ? logger.info : logger.warn).call(logger, {
      event: "act",
      ok: result.ok,
      source: result.source,
      token: result.token,
      executed: result.executed,
      gate: result.gate?.reasons ?? null,
      total_s: result.timing?.total_s,
      decision_age_s: result.timing?.decision_age_s ?? null,
      error_type: result.error_type,
      error: result.error,
    });
    return { status: actStatus(result), body: result };
  });

  post("/v1/step", async (body) => {
    const result = await agent.step(body);
    (result.ok ? logger.info : logger.warn).call(logger, {
      event: "step",
      ok: result.ok,
      session_id: result.session_id,
      seq: result.seq,
      token: result.token,
      token_source: result.token_source,
      executed: result.executed,
      thinker_queued: result.thinker_queued,
      total_s: result.timing?.total_s,
      error_type: result.error_type,
      error: result.error,
    });
    return { status: stepStatus(result), body: result };
  });

  // Compatibility route for deployments where an external controller owns actuation: even if a caller
  // supplies execute=true, this adapter returns a freshly observed preview token and never moves.
  post("/v1/control/step", async (body) => {
    const result = await agent.step({ ...body, execute: false });
    const payload = {
      ...result,
      harness: config.agentHarness ?? "prime-agent-rpc",
      harness_version: config.harnessVersion ?? config.primeAgentVersion ?? null,
      direct: true,
      reasoning: false,
      model_output_tokens: 1,
      actuator_attached: false,
      executed: false,
    };
    (payload.ok ? logger.info : logger.warn).call(logger, {
      event: "control_preview",
      ok: payload.ok,
      session_id: payload.session_id,
      seq: payload.seq,
      token: payload.token,
      model_used: payload.model_used,
      total_s: payload.timing?.total_s,
      error_type: payload.error_type,
      error: payload.error,
    });
    return { status: stepStatus(payload), body: payload };
  });

  post("/v1/pipeline", async (body) => ({ status: 200, body: { ok: true, ...agent.pipeline.set(body) } }));
  post("/v1/stop", async () => ({ status: 200, body: { ok: true, ...(await agent.stop()) } }));
  post("/v1/sense", async (body) => ({ status: 200, body: await agent.sense(body) }));
  post("/v1/arm", async (body) => {
    const result = await agent.arm(body);
    const status = result.ok
      ? 200
      : result.error_type === "execute_disabled" || result.error_type === "execute_required"
        ? 409
        : result.error_type === "request_error"
          ? 400
          : result.error_type === "gate_refused"
            ? 200
            : 502;
    return { status, body: result };
  });

  post("/internal/thinker/correction", async (body, request) => {
    if (!isLoopback(request.socket.remoteAddress)) {
      return { status: 403, body: { ok: false, error_type: "forbidden", error: "thinker callbacks are loopback-only" } };
    }
    const result = agent.acceptCorrection(body);
    return { status: result.accepted ? 202 : correctionStatus(result), body: { ok: result.accepted, ...result } };
  });

  // ---- brain: memory, router and judge state. Read-only except for the loopback lesson channel.
  get("/v1/brain", async () => {
    if (!agent.brain) return { status: 200, body: { ok: true, enabled: false } };
    return { status: 200, body: { ok: true, ...agent.brain.state() } };
  });
  // Everything the brain holds about one session (or "act" for the reactive path): episodic steps,
  // signals, the dead-reckoning pose and trail. Read-only; what an evaluation script logs per session.
  post("/v1/brain/session", async (body) => {
    if (!agent.brain) return { status: 404, body: { ok: false, error_type: "brain_disabled", error: "the brain is disabled" } };
    if (typeof body.session_id !== "string" || !body.session_id) return { status: 400, body: { ok: false, error_type: "request_error", error: "session_id is required" } };
    const state = agent.brain.session(body.session_id);
    if (!state) return { status: 404, body: { ok: false, error_type: "unknown_session", error: "unknown control session" } };
    return { status: 200, body: { ok: true, ...state } };
  });
  // How the last sessions on a task went, the way the deliberate engine is told before its first step.
  post("/v1/brain/episodes", async (body) => {
    if (!agent.brain) return { status: 404, body: { ok: false, error_type: "brain_disabled", error: "the brain is disabled" } };
    const result = agent.brain.episodes({ instruction: body.instruction ?? "", taskKey: body.task_key ?? null, k: body.k });
    return { status: 200, body: { ok: true, ...result } };
  });
  post("/v1/brain/recall", async (body) => {
    if (!agent.brain) return { status: 404, body: { ok: false, error_type: "brain_disabled", error: "the brain is disabled" } };
    const result = agent.brain.recall({ instruction: body.instruction ?? "", situation: body.situation ?? null, token: body.token ?? null, k: body.k });
    return { status: 200, body: { ok: true, ...result } };
  });
  // The asynchronous reviewer, having seen the after frame, posts its outcome assessment of a past
  // session step (progress / no_progress / unsafe / unclear plus a lesson). Loopback-only, and by
  // construction too late to steer: it only teaches memory and the router.
  post("/internal/judge/review", async (body, request) => {
    if (!isLoopback(request.socket.remoteAddress)) {
      return { status: 403, body: { ok: false, error_type: "forbidden", error: "judge reviews are loopback-only" } };
    }
    if (!agent.brain) return { status: 404, body: { ok: false, error_type: "brain_disabled", error: "the brain is disabled" } };
    const result = agent.acceptJudgement(body);
    const status = result.accepted ? 202 : result.reason === "unknown_session" ? 404 : result.reason === "unknown_seq" ? 409 : 422;
    return { status, body: { ok: result.accepted, ...result } };
  });
  // Forget what was learned (experience and routing memory; episodes only when asked). Loopback-only:
  // the cold arm of an evaluation, never something a remote caller may do to a running robot.
  post("/internal/brain/reset", async (body, request) => {
    if (!isLoopback(request.socket.remoteAddress)) {
      return { status: 403, body: { ok: false, error_type: "forbidden", error: "brain reset is loopback-only" } };
    }
    if (!agent.brain) return { status: 404, body: { ok: false, error_type: "brain_disabled", error: "the brain is disabled" } };
    // Destructive for the deployment's persisted memory (the files are rotated to *.bak first): the
    // caller has to say so.
    if (body.confirm !== true) {
      return { status: 400, body: { ok: false, error_type: "confirm_required", error: "brain reset wipes persisted memory; send {confirm: true}" } };
    }
    const result = agent.brain.reset({ experience: body.experience !== false, router: body.router !== false, episodes: body.episodes === true });
    logger.warn?.({ event: "brain_reset_requested", ...result });
    return { status: 202, body: { ok: true, ...result } };
  });
  // An out-of-process judge (a stronger VLM, an offline evaluator, a human) posts what it learned
  // about a past step. It changes memory and routing only — never the current command.
  post("/internal/judge/lesson", async (body, request) => {
    if (!isLoopback(request.socket.remoteAddress)) {
      return { status: 403, body: { ok: false, error_type: "forbidden", error: "judge lessons are loopback-only" } };
    }
    if (!agent.brain) return { status: 404, body: { ok: false, error_type: "brain_disabled", error: "the brain is disabled" } };
    const result = agent.brain.learnLesson(body, { source: "external" });
    return { status: result.accepted ? 202 : 422, body: { ok: result.accepted, ...result } };
  });

  return createServer(async (request, response) => {
    const url = new URL(request.url ?? "/", "http://localhost");
    const handler = routes.get(`${request.method} ${url.pathname}`);
    if (!handler) {
      sendJson(response, 404, { ok: false, error_type: "not_found", error: "not found" });
      return;
    }
    const healthProbe = request.method === "GET" && url.pathname === "/health";
    if (!healthProbe && !authorised(request, config.agentToken)) {
      sendJson(response, 401, { ok: false, error_type: "unauthorized", error: "a bearer token is required (G1_AGENT_TOKEN)" });
      return;
    }
    let body = {};
    if (request.method === "POST") {
      try {
        body = await readJson(request);
      } catch (error) {
        sendJson(response, error.status ?? 400, {
          ok: false,
          error_type: error.status === 413 ? "request_too_large" : "request_error",
          error: error instanceof SyntaxError ? "malformed JSON" : error.message,
        });
        return;
      }
    }
    try {
      const { status, body: payload } = await handler(body, request);
      sendJson(response, status, payload);
    } catch (error) {
      logger.error?.({ event: "request_failed", path: url.pathname, error: error?.message ?? String(error) });
      sendJson(response, 500, { ok: false, error_type: "service_error", error: error?.message ?? String(error) });
    }
  });
}
