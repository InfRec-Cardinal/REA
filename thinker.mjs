#!/usr/bin/env node
// rea-thinker — the asynchronous half of the agent.
//
// It reviews what the fast loop just decided, against the before and after frames plus the execution
// feedback, and posts a correction back for the *next* decision.
//
// Three properties make it safe to run a slow, thinking model here:
//   - it holds no SDK handle and no actuator; the only thing it can do is post a correction,
//   - the main loop never blocks on it; a review that never arrives simply changes nothing,
//   - it binds to loopback only, and the correction endpoint refuses non-loopback callers.
//
// Latest-wins: a newer job aborts the one in flight, because by the time a review of an old frame
// lands, the robot is somewhere else and the advice is worse than none.

import { pathToFileURL } from "node:url";

import { loadDotEnv, readEnv } from "./src/env.mjs";
import { resolveProfile } from "./src/profiles/index.mjs";
import { createDispatcher } from "./src/provider.mjs";
import { VERSION } from "./src/version.mjs";
import { createThinkerInfer } from "./src/thinker-infer.mjs";
import { createThinkerHttpServer } from "./src/thinker-http.mjs";
import { createThinkerRuntime } from "./src/thinker-runtime.mjs";

const LOOPBACK = new Set(["127.0.0.1", "::1", "localhost"]);

/** URL.hostname keeps IPv6 brackets ("[::1]"); compare the bare address. */
export function isLoopbackHost(hostname) {
  return LOOPBACK.has(String(hostname ?? "").replace(/^\[(.*)\]$/u, "$1").toLowerCase());
}

function integer(env, name, fallback, { minimum = 1, maximum = Number.MAX_SAFE_INTEGER } = {}) {
  const value = Number(env[name] ?? fallback);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be an integer between ${minimum} and ${maximum}`);
  }
  return value;
}

export function loadThinkerConfig(env = process.env) {
  const host = env.G1_THINKER_HOST ?? "127.0.0.1";
  if (!isLoopbackHost(host)) throw new Error("G1_THINKER_HOST must be a loopback address");
  const apiKey = readEnv(env, "REA_API_KEY") ?? "";
  return {
    host,
    port: integer(env, "G1_THINKER_PORT", 8092, { maximum: 65535 }),
    apiKey,
    providerConfigured: apiKey.length > 0,
    baseUrl: readEnv(env, "REA_BASE_URL") ?? "https://api.openai.com/v1",
    model: env.G1_THINKER_MODEL ?? readEnv(env, "REA_MODEL") ?? "",
    timeoutMs: integer(env, "G1_THINKER_TIMEOUT_MS", 12_000),
    // The review tool call now carries a brief (<= 200 chars) *and* a lesson (<= 200 chars) plus the
    // enum fields: ~140 tokens worst case for the arguments alone. 160 could truncate the JSON and lose
    // the correction with it (finish=length), so the budget is doubled.
    maxOutputTokens: integer(env, "G1_THINKER_MAX_OUTPUT_TOKENS", 320, { maximum: 1024 }),
    callbackUrl: env.G1_THINKER_CALLBACK_URL ?? "http://127.0.0.1:8091/internal/thinker/correction",
    callbackTimeoutMs: integer(env, "G1_THINKER_CALLBACK_TIMEOUT_MS", 3000),
    // The judging half of every review goes here. Empty string disables it (reviewer steers only).
    judgeUrl: env.G1_THINKER_JUDGE_URL === "" ? null : (env.G1_THINKER_JUDGE_URL ?? "http://127.0.0.1:8091/internal/judge/review"),
  };
}

function writeLog(level, record) {
  process.stdout.write(`${JSON.stringify({ at: new Date().toISOString(), level, ...record })}\n`);
}

const logger = {
  info: (record) => writeLog("info", record),
  warn: (record) => writeLog("warn", record),
  error: (record) => writeLog("error", record),
};

export function createCorrectionCallback({ url, timeoutMs = 3000, fetchFn = globalThis.fetch }) {
  const parsed = new URL(url);
  if (!isLoopbackHost(parsed.hostname)) throw new Error("thinker callback must target loopback");
  return async function callback(correction, { signal } = {}) {
    const timeoutSignal = AbortSignal.timeout(timeoutMs);
    const requestSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
    const response = await fetchFn(parsed, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(correction),
      signal: requestSignal,
    });
    if (!response.ok) {
      const detail = (await response.text()).trim();
      throw new Error(`thinker callback returned ${response.status}${detail ? `: ${detail.slice(0, 300)}` : ""}`);
    }
  };
}

/**
 * The outcome judgement channel. Same loopback-only rule as the correction callback, but a 4xx from
 * the brain (an unknown session, a step it no longer holds) is a dropped lesson, not an error worth
 * retrying: by the time a judgement exists the robot has moved on.
 */
export function createJudgementCallback({ url, timeoutMs = 3000, fetchFn = globalThis.fetch }) {
  const parsed = new URL(url);
  if (!isLoopbackHost(parsed.hostname)) throw new Error("thinker judge callback must target loopback");
  return async function judge(judgement, { signal } = {}) {
    const timeoutSignal = AbortSignal.timeout(timeoutMs);
    const requestSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
    const response = await fetchFn(parsed, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(judgement),
      signal: requestSignal,
    });
    if (!response.ok) {
      const detail = (await response.text()).trim();
      throw new Error(`thinker judge callback returned ${response.status}${detail ? `: ${detail.slice(0, 300)}` : ""}`);
    }
  };
}

export async function main() {
  loadDotEnv();
  const config = loadThinkerConfig();
  const profile = await resolveProfile(process.env.G1_ROBOT_PROFILE);
  const infer = createThinkerInfer({
    apiKey: config.apiKey,
    baseUrl: config.baseUrl,
    maxOutputTokens: config.maxOutputTokens,
    dispatcher: await createDispatcher(),
    profile,
  });
  const callback = createCorrectionCallback({ url: config.callbackUrl, timeoutMs: config.callbackTimeoutMs });
  const judge = config.judgeUrl ? createJudgementCallback({ url: config.judgeUrl, timeoutMs: config.callbackTimeoutMs }) : null;
  const runtime = createThinkerRuntime({
    infer,
    model: config.model,
    callback,
    judge,
    timeoutMs: config.timeoutMs,
    providerConfigured: config.providerConfigured,
    logger,
    profile,
  });
  const server = createThinkerHttpServer({ runtime, logger });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(config.port, config.host, resolve);
  });
  logger.info({
    event: "thinker_started",
    service: "rea-thinker",
    version: VERSION,
    host: config.host,
    port: config.port,
    loopback_only: true,
    actuator_attached: false,
    sdk_attached: false,
    concurrency: 1,
    latest_wins: true,
    provider_configured: config.providerConfigured,
    model: config.model,
    callback_url: config.callbackUrl,
    judge_url: config.judgeUrl,
  });

  let stopping = false;
  const close = async (signal) => {
    if (stopping) return;
    stopping = true;
    logger.info({ event: "thinker_stopping", signal });
    await new Promise((resolve) => server.close(resolve));
    await runtime.close();
    process.exit(0);
  };
  process.once("SIGINT", () => void close("SIGINT"));
  process.once("SIGTERM", () => void close("SIGTERM"));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    logger.error({
      event: "thinker_start_failed",
      error_type: "config_error",
      error: error instanceof Error ? error.message : String(error),
    });
    process.exitCode = 1;
  });
}
