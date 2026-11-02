import assert from "node:assert/strict";
import { test } from "node:test";

import { decide } from "../src/decide.mjs";

const FRAME = { imageB64: "aGk=", mimeType: "image/jpeg", depth: { center_m: 2 }, width: 320, height: 240 };
const CONFIG = { providerConfigured: true };
const CFG = {
  engine: "token",
  primaryModel: "primary",
  fallbackModel: "fallback",
  primaryTimeoutMs: 500,
  fallbackTimeoutMs: 500,
  race: false,
};

const camera = { capture: async () => FRAME };

function engineReturning(map) {
  const engine = async (model) => {
    const outcome = map[model];
    if (outcome instanceof Error) throw outcome;
    if (typeof outcome === "function") return outcome();
    return outcome;
  };
  engine.engine = "token";
  return { token: engine };
}

test("a successful decision carries the token, the legacy action and timings", async () => {
  const engines = engineReturning({ primary: { token: "W", tokenReadyS: 0.2 } });
  const { result } = await decide({ camera, engines, config: CONFIG, cfg: CFG, instruction: "go" });
  assert.equal(result.ok, true);
  assert.equal(result.token, "W");
  assert.deepEqual(result.action, { action: "FORWARD", duration_ms: 1000 });
  assert.equal(result.model, "primary");
  assert.equal(result.fallbackUsed, false);
  assert.equal(result.timing.modelReadyS, 0.2);
});

test("a retryable primary failure falls back to the second model", async () => {
  const engines = engineReturning({
    primary: Object.assign(new Error("boom"), { errorType: "stream_error", retryable: true }),
    fallback: { token: "Q", tokenReadyS: 0.3 },
  });
  const { result } = await decide({ camera, engines, config: CONFIG, cfg: CFG, instruction: "go" });
  assert.equal(result.ok, true);
  assert.equal(result.token, "Q");
  assert.equal(result.model, "fallback");
  assert.equal(result.fallbackUsed, true);
  assert.equal(result.attempts.length, 2);
});

test("an auth failure does not burn the fallback model", async () => {
  const engines = engineReturning({
    primary: Object.assign(new Error("nope"), { errorType: "auth_error", status: 401, retryable: false }),
    fallback: { token: "W" },
  });
  const { result } = await decide({ camera, engines, config: CONFIG, cfg: CFG, instruction: "go" });
  assert.equal(result.ok, false);
  assert.equal(result.attempts.length, 1, "a bad key will not become a good key on the next model");
  assert.equal(result.errorType, "auth_error");
});

test("both models failing produces a failure with no action", async () => {
  const engines = engineReturning({
    primary: Object.assign(new Error("a"), { retryable: true }),
    fallback: Object.assign(new Error("b"), { retryable: true }),
  });
  const { result } = await decide({ camera, engines, config: CONFIG, cfg: CFG, instruction: "go" });
  assert.equal(result.ok, false);
  assert.equal(result.token, null);
  assert.equal(result.action, null);
});

test("race mode takes the first valid answer and cancels the loser", async () => {
  let fallbackAborted = false;
  const engine = async (model, frame, { signal }) => {
    if (model === "primary") return { token: "W", tokenReadyS: 0.05 };
    return new Promise((resolve, reject) => {
      signal.addEventListener("abort", () => {
        fallbackAborted = true;
        reject(new Error("aborted"));
      });
      setTimeout(() => resolve({ token: "S", tokenReadyS: 5 }), 3000).unref?.();
    });
  };
  engine.engine = "token";
  const { result } = await decide({
    camera,
    engines: { token: engine },
    config: CONFIG,
    cfg: { ...CFG, race: true },
    instruction: "go",
  });
  assert.equal(result.ok, true);
  assert.equal(result.token, "W");
  assert.equal(fallbackAborted, true, "the losing call must be cancelled, not left running");
});

test("a capture failure never reaches the model", async () => {
  let called = false;
  const engines = engineReturning({ primary: { token: "W" } });
  const original = engines.token;
  engines.token = async (...args) => {
    called = true;
    return original(...args);
  };
  engines.token.engine = "token";
  const failing = {
    capture: async () => {
      throw Object.assign(new Error("no camera"), { errorType: "capture_error" });
    },
  };
  const { result } = await decide({ camera: failing, engines, config: CONFIG, cfg: CFG, instruction: "go" });
  assert.equal(result.ok, false);
  assert.equal(result.errorType, "capture_error");
  assert.equal(called, false, "no point asking a model about a frame we do not have");
});

test("an unconfigured provider fails fast without capturing", async () => {
  let captured = false;
  const watching = {
    capture: async () => {
      captured = true;
      return FRAME;
    },
  };
  const { result } = await decide({
    camera: watching,
    engines: engineReturning({}),
    config: { providerConfigured: false },
    cfg: CFG,
    instruction: "go",
  });
  assert.equal(result.ok, false);
  assert.equal(result.errorType, "config_error");
  assert.equal(captured, false);
});

test("a hung model is cut off at the deadline", async () => {
  const engine = async () => new Promise(() => {});
  engine.engine = "token";
  const { result } = await decide({
    camera,
    engines: { token: engine },
    config: CONFIG,
    cfg: { ...CFG, primaryTimeoutMs: 60, fallbackTimeoutMs: 60 },
    instruction: "go",
  });
  assert.equal(result.ok, false);
  assert.equal(result.errorType, "deadline");
});

test("an unknown engine is a request error, not a silent run of the token engine", async () => {
  let captured = false;
  const engines = engineReturning({ primary: { token: "W" } });
  const { result } = await decide({ camera: { capture: async () => { captured = true; return FRAME; } }, engines, config: CONFIG, cfg: { ...CFG, engine: "tokn" }, instruction: "go" });
  assert.equal(result.ok, false);
  assert.equal(result.errorType, "request_error");
  assert.match(result.error, /unknown engine 'tokn'; known: token/u);
  assert.equal(captured, false, "the camera is not touched");
});

test("an engine that needs no provider runs without a key", async () => {
  const scripted = Object.assign(async () => ({ token: "Q", tokenReadyS: 0 }), { engine: "scripted" });
  const { result } = await decide({ camera, engines: { scripted }, config: { providerConfigured: false }, cfg: { ...CFG, engine: "scripted" }, instruction: "go" });
  assert.equal(result.ok, true);
  assert.equal(result.token, "Q");
});
