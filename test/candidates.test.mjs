import assert from "node:assert/strict";
import { test } from "node:test";

import { buildCandidates, candidateCfg, costUsd, parseCandidateSpec, priceFor, pricesFrom } from "../src/candidates.mjs";

const ENGINES = ["prime", "token", "tool", "react", "cli"];
const CONFIG = { primaryModel: "deepseek-v4-flash-vision-exp", reactModel: "deepseek-v4-pro", primaryTimeoutMs: 6000, reactTimeoutMs: 8000, cliTimeoutMs: 60000, routerCheapChain: ["token"], routerEscalateTo: "react", reactMaxOutputTokens: 160, reactMaxIterations: 2 };

test("without a spec the candidates are the old two arms with engine names as ids", () => {
  const c = buildCandidates(CONFIG, ENGINES);
  assert.deepEqual(c.ids, ["token", "react"]);
  assert.deepEqual(c.reflex, ["token"]);
  assert.deepEqual(c.deliberate, ["react"]);
  assert.equal(c.byId.get("token").model, "deepseek-v4-flash-vision-exp");
  assert.equal(c.byId.get("react").model, "deepseek-v4-pro");
  assert.equal(c.byId.get("react").timeoutMs, 8000);
  assert.equal(c.byId.get("token").price.in, 0.44);
});

test("a spec binds harnesses to models, tiers and deadlines; ids are engine@model", () => {
  const list = parseCandidateSpec("token@deepseek-v4-flash-vision-exp:reflex, token@kimi-k2.5:deliberate:15000, react:deliberate, cli@claude-fable-5-1:deliberate", CONFIG, ENGINES);
  assert.deepEqual(list.map(c => c.id), ["token@deepseek-v4-flash-vision-exp", "token@kimi-k2.5", "react", "cli@claude-fable-5-1"]);
  assert.equal(list[1].timeoutMs, 15000);
  assert.equal(list[1].tier, "deliberate");
  assert.equal(list[2].model, "deepseek-v4-pro", "a deliberate react candidate defaults to the react model");
  assert.equal(list[3].provider, "cli");
  assert.equal(list[3].timeoutMs, 60000);
  assert.deepEqual(list[3].price, { in: 10, out: 50 });
  const json = parseCandidateSpec(JSON.stringify([{ engine: "token", model: "deepseek-v4-pro", tier: "reflex", timeoutMs: 4000 }]), CONFIG, ENGINES);
  assert.equal(json[0].id, "token@deepseek-v4-pro");
  assert.equal(json[0].timeoutMs, 4000);
  assert.throws(() => parseCandidateSpec("nope@x:reflex", CONFIG, ENGINES), /not one of/u);
  assert.throws(() => parseCandidateSpec("token@x:fast", CONFIG, ENGINES), /tier/u);
  assert.throws(() => buildCandidates({ ...CONFIG, routerCandidates: "react:deliberate" }, ENGINES), /reflex-tier/u);
  assert.throws(() => buildCandidates({ ...CONFIG, routerCandidates: "token:reflex, token:reflex" }, ENGINES), /duplicate/u);
});

test("candidateCfg runs the harness on the candidate's model and deadline with no fallback or race", () => {
  const base = { engine: "token", primaryModel: "a", fallbackModel: "b", primaryTimeoutMs: 6000, race: true, maxOutputTokens: 64 };
  const list = parseCandidateSpec("token@deepseek-v4-pro:reflex:2500, react@deepseek-v4-pro:deliberate", CONFIG, ENGINES);
  const reflex = candidateCfg(base, list[0], CONFIG);
  assert.deepEqual({ engine: reflex.engine, primaryModel: reflex.primaryModel, fallbackModel: reflex.fallbackModel, primaryTimeoutMs: reflex.primaryTimeoutMs, race: reflex.race, maxOutputTokens: reflex.maxOutputTokens }, { engine: "token", primaryModel: "deepseek-v4-pro", fallbackModel: "", primaryTimeoutMs: 2500, race: false, maxOutputTokens: 64 });
  const deliberate = candidateCfg(base, list[1], CONFIG);
  assert.equal(deliberate.maxOutputTokens, 160);
  assert.equal(deliberate.reactMaxIterations, 2);
});

test("prices resolve by exact name then longest prefix, and cost follows the provider's token counts", () => {
  assert.deepEqual(priceFor("kimi-k2.5"), { in: 0.6, out: 3.0 });
  assert.deepEqual(priceFor("claude-fable-5-1-20261001"), { in: 10, out: 50 }, "a dated variant prices like its base");
  assert.equal(priceFor("unknown-model"), null);
  assert.equal(costUsd("deepseek-v4-flash-vision-exp", { prompt_tokens: 1000000, completion_tokens: 1000000 }), 1.76);
  assert.equal(costUsd("deepseek-v4-flash-vision-exp", null), null);
  assert.equal(costUsd("unknown", { prompt_tokens: 10 }), null);
  const custom = pricesFrom({ modelPrices: JSON.stringify({ "deepseek-v4-pro": [1, 2], "x": { in: 3, out: 4 } }) });
  assert.deepEqual(priceFor("deepseek-v4-pro", custom), { in: 1, out: 2 });
  assert.deepEqual(priceFor("x", custom), { in: 3, out: 4 });
});

test("a Gemini model resolves to Google's OpenAI-compatible endpoint; the key decides reachability", async () => {
  const { providersFrom, providerFor } = await import("../src/candidates.mjs");
  const noKey = providersFrom({ providerKeys: { gemini: "" } }, {});
  assert.equal(providerFor("gemini-3.8-flash", noKey).name, "gemini");
  assert.equal(providerFor("deepseek-v4-flash-vision-exp", noKey), null);
  const keyed = providersFrom({ providerKeys: { gemini: "gk-test" } }, {});
  assert.equal(providerFor("gemini-3.8-flash", keyed).apiKey, "gk-test");
  assert.equal(providerFor("gemini-3.8-flash", keyed).baseUrl, "https://generativelanguage.googleapis.com/v1beta/openai");
  // environment fallback and G1_PROVIDERS override
  assert.equal(providerFor("gemini-3.8-flash", providersFrom({}, { GEMINI_API_KEY: "env-key" })).apiKey, "env-key");
  const overridden = providersFrom({ providers: JSON.stringify([{ name: "gemini", extraBody: { reasoning_effort: "none" } }, { name: "vertex", prefixes: ["vertex/"], baseUrl: "https://example.test/v1/", apiKey: "vk" }]), providerKeys: { gemini: "gk" } }, {});
  assert.deepEqual(providerFor("gemini-3.8-flash", overridden).extraBody, { reasoning_effort: "none" });
  assert.equal(providerFor("vertex/gemini-3.8-flash", overridden).name, "vertex");
  assert.equal(providerFor("vertex/gemini-3.8-flash", overridden).baseUrl, "https://example.test/v1");

  const config = { ...CONFIG, providerKeys: { gemini: "" } };
  const list = parseCandidateSpec("token@gemini-3.8-flash:reflex, token@deepseek-v4-flash-vision-exp", config, ENGINES);
  assert.equal(list[0].provider, "gemini");
  assert.equal(list[0].reachable, false);
  assert.equal(list[1].provider, "gateway");
  assert.equal(list[1].reachable, true);

  const keyedConfig = { ...CONFIG, providerKeys: { gemini: "gk-test" } };
  const gemini = parseCandidateSpec("token@gemini-3.8-flash", keyedConfig, ENGINES)[0];
  assert.equal(gemini.reachable, true);
  const cfg = candidateCfg({ engine: "token", primaryModel: "x" }, gemini, keyedConfig);
  assert.equal(cfg.provider, "gemini");
  assert.equal(cfg.baseUrl, "https://generativelanguage.googleapis.com/v1beta/openai");
  assert.equal(cfg.apiKey, "gk-test");
  assert.deepEqual(cfg.extraBody, { reasoning_effort: "low" });
  const deepseek = parseCandidateSpec("token@deepseek-v4-flash-vision-exp", keyedConfig, ENGINES)[0];
  assert.equal(candidateCfg({ engine: "token" }, deepseek, keyedConfig).baseUrl, undefined);
});
