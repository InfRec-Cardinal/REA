import assert from "node:assert/strict";
import { test } from "node:test";

import { BUILTIN_ENGINE_META, UnknownEngineError, asEngineRegistry, createEngineRegistry, outputTokensFor } from "../src/engines.mjs";
import { loadConfig } from "../src/config.mjs";

const infer = name => Object.assign(async () => ({ token: "X" }), { engine: name });

test("registered engines are looked up by name and unknown names fail loudly with the known list", () => {
  const registry = createEngineRegistry();
  registry.register("token", infer("token"));
  registry.register("mine", infer("mine"), { outputTokens: 1, deliberate: false, needsProvider: false, health: () => ({ custom: true }) });
  assert.equal(registry.has("token"), true);
  assert.deepEqual(registry.names(), ["token", "mine"]);
  assert.equal(registry.meta("token").outputTokens, 1, "built-in meta fills in for known names");
  assert.equal(registry.meta("mine").needsProvider, false);
  assert.throws(() => registry.get("nope"), error => error instanceof UnknownEngineError && error.known.join() === "token,mine" && /unknown engine 'nope'/u.test(error.message));
  assert.throws(() => registry.register("token", infer("token")), /already registered/u);
  assert.throws(() => registry.register("bad", "not a function"), /function/u);
});

test("describe() reproduces the health block for the built-in engines", () => {
  const config = loadConfig({ REA_API_KEY: "x" });
  const registry = createEngineRegistry();
  for (const name of ["prime", "token", "tool", "react"]) registry.register(name, infer(name));
  const described = registry.describe(config);
  assert.deepEqual(Object.keys(described), ["prime", "token", "tool", "react"]);
  assert.deepEqual(described.prime, { harness: "prime-agent-rpc", harness_version: "0.8.0", installed_version: "0.8.0", model_output_tokens: 1, reasoning: false, persistent: true, pool_size: 2, default: true });
  assert.deepEqual(described.token, { model_output_tokens: 1, reasoning: false, default: false });
  assert.deepEqual(described.react, { model_output_tokens: 160, reasoning: "bounded_react", max_iterations: 2, model: config.reactModel, deliberate: true, default: false });
});

test("a plain engines object is wrapped without changing what its known names mean", () => {
  const wrapped = asEngineRegistry({ token: infer("token"), react: infer("react"), odd: infer("odd") });
  assert.equal(wrapped.meta("token").outputTokens, 1);
  assert.equal(wrapped.meta("react").deliberate, true);
  assert.deepEqual(wrapped.meta("odd"), { outputTokens: null, deliberate: false, needsProvider: true, health: wrapped.meta("odd").health });
  assert.equal(asEngineRegistry(wrapped), wrapped, "a registry passes through");
  assert.equal(outputTokensFor("prime", 64), 1);
  assert.equal(outputTokensFor("tool", 64), 64);
  assert.equal(outputTokensFor("forced", null), null);
  assert.deepEqual(Object.keys(BUILTIN_ENGINE_META), ["prime", "token", "tool", "react", "cli", "scripted"]);
});
