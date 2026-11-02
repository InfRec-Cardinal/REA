// Routing candidates: the (engine harness, model) pairs the router may choose between.
//
// The router used to choose between engine *harnesses* (token, react) that all ran the same configured
// model. A candidate binds a harness to a model, a tier and a price, so "route among agents" can mean
// "route among models": a fast one-token reflex, a slower vision model, the ReAct engine on a
// bigger model, or a CLI-backed frontier model (Claude / Codex) that only fits the asynchronous or
// stopped-robot paths. Candidate ids are what the router remembers, scores and suggests.
//
//   G1_ROUTER_CANDIDATES="token@your-fast-vision-model:reflex, token@your-second-vision-model:reflex,
//                         react@your-larger-model:deliberate:15000, cli@your-cli-model:deliberate:60000"
//
// Without the variable the candidates are exactly the old cheap chain and escalation engine on the
// configured models, with the engine names as ids, so nothing changes for existing deployments.

export const TIERS = Object.freeze(["reflex", "deliberate"]);

// Illustrative USD per 1M tokens [input, output] for some public models (list prices, 2026-09), used only for cost
// accounting when the provider reports no cost. Providers bill at their own rates, so treat these as relative weights,
// not invoices; add or override entries with G1_MODEL_PRICES JSON.
export const DEFAULT_PRICES = Object.freeze({
  "deepseek-v4-flash-vision-exp": [0.44, 1.32],
  "deepseek-v4-flash": [0.44, 1.32],
  "deepseek-v4-pro": [1.32, 3.96],
  "qwen3-vl-flash": [0.15, 0.47],
  "qwen3.8-flash": [0.15, 0.47],
  "qwen3.5-flash": [0.065, 0.26],
  "kimi-k2.5": [0.6, 3.0],
  "MiniMax-M2.7": [0.3, 1.2],
  "glm-5": [0.8, 3.0],
  "gemini-3.5-flash-lite": [0.1, 0.4],
  "claude-fable-5-1": [10, 50],
  "claude-fable-5": [10, 50],
  "claude-haiku-4-5-20251001": [1, 5],
  "gpt-6-astra": [10, 50],
  "gpt-5.4-mini": [0.4, 1.6],
});

// Providers other than the default gateway, chosen by model-name prefix. Gemini's OpenAI-compatible
// endpoint speaks the same chat-completions format (streaming, image_url data URLs, usage chunk), so
// a Gemini Flash candidate needs only a key: G1_GEMINI_API_KEY (or GEMINI_API_KEY). `extraBody` is
// merged into every request; Gemini 3.7+ Flash rejects reasoning_effort "none", so "low" is the
// default and G1_PROVIDERS can override it. G1_PROVIDERS (JSON array of {name, prefixes, baseUrl,
// apiKey|apiKeyEnv, extraBody}) adds or replaces entries.
export const BUILTIN_PROVIDERS = Object.freeze([
  Object.freeze({ name: "gemini", prefixes: ["gemini-", "gemini"], baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", apiKeyEnv: ["G1_GEMINI_API_KEY", "GEMINI_API_KEY"], extraBody: { reasoning_effort: "low" } }),
]);

export function providersFrom(config = {}, env = process.env) {
  const extra = config.providers ? (typeof config.providers === "string" ? JSON.parse(config.providers) : config.providers) : [];
  const byName = new Map();
  for (const entry of [...BUILTIN_PROVIDERS, ...extra]) {
    if (!entry || typeof entry.name !== "string") continue;
    const base = byName.get(entry.name) ?? {};
    const keyEnv = [].concat(entry.apiKeyEnv ?? base.apiKeyEnv ?? []);
    const apiKey = entry.apiKey ?? config.providerKeys?.[entry.name] ?? keyEnv.map(k => env[k]).find(v => typeof v === "string" && v.length > 0) ?? base.apiKey ?? "";
    byName.set(entry.name, { ...base, ...entry, apiKeyEnv: keyEnv, prefixes: entry.prefixes ?? base.prefixes ?? [entry.name], baseUrl: (entry.baseUrl ?? base.baseUrl ?? "").replace(/\/+$/u, ""), extraBody: entry.extraBody ?? base.extraBody ?? {}, apiKey });
  }
  return [...byName.values()];
}

/** The provider a model is served by, or null for the default gateway. Longest matching prefix wins. */
export function providerFor(model, providers) {
  if (typeof model !== "string" || !model) return null;
  let best = null;
  for (const provider of providers ?? []) {
    for (const prefix of provider.prefixes ?? []) {
      if (model.startsWith(prefix) && (!best || prefix.length > best.prefix.length)) best = { prefix, provider };
    }
  }
  return best ? best.provider : null;
}

export function pricesFrom(config = {}) {
  const table = { ...DEFAULT_PRICES };
  const raw = config.modelPrices;
  if (raw) {
    const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
    for (const [model, value] of Object.entries(parsed)) {
      const pair = Array.isArray(value) ? value : [value?.in ?? value?.input_per_1m, value?.out ?? value?.output_per_1m];
      if (pair.every(x => Number.isFinite(Number(x)))) table[model] = [Number(pair[0]), Number(pair[1])];
    }
  }
  return table;
}

/** Exact match first, then the longest key that prefixes the model name (so a dated variant still prices). */
export function priceFor(model, prices = DEFAULT_PRICES) {
  if (typeof model !== "string" || !model) return null;
  if (prices[model]) return { in: prices[model][0], out: prices[model][1] };
  let best = null;
  for (const [key, pair] of Object.entries(prices)) {
    if (model.startsWith(key) && (!best || key.length > best.key.length)) best = { key, pair };
  }
  return best ? { in: best.pair[0], out: best.pair[1] } : null;
}

export function costUsd(model, usage, prices = DEFAULT_PRICES) {
  if (!usage || typeof usage !== "object") return null;
  const price = priceFor(model, prices);
  if (!price) return null;
  const inTok = Number(usage.prompt_tokens ?? usage.input_tokens ?? 0);
  const outTok = Number(usage.completion_tokens ?? usage.output_tokens ?? 0);
  if (!Number.isFinite(inTok) || !Number.isFinite(outTok)) return null;
  return Number(((inTok * price.in + outTok * price.out) / 1e6).toFixed(8));
}

function tierTimeout(config, engine, tier) {
  if (engine === "cli") return config.cliTimeoutMs ?? 60000;
  if (tier === "deliberate") return config.reactTimeoutMs ?? 8000;
  return config.primaryTimeoutMs ?? 6000;
}

function make(config, { engine, model = null, tier = "reflex", timeoutMs = null, id = null }, engines) {
  if (typeof engine !== "string" || !engines.includes(engine)) throw new Error(`candidate engine "${engine}" is not one of ${engines.join(", ")}`);
  if (!TIERS.includes(tier)) throw new Error(`candidate tier "${tier}" must be one of ${TIERS.join(", ")}`);
  const resolvedModel = model ?? (tier === "deliberate" && engine !== "cli" ? (config.reactModel ?? config.primaryModel) : config.primaryModel);
  const prices = pricesFrom(config);
  const provider = engine === "cli" ? null : providerFor(resolvedModel, providersFrom(config));
  return Object.freeze({
    id: id ?? (model ? `${engine}@${model}` : engine),
    engine,
    model: resolvedModel,
    tier,
    timeoutMs: Number.isFinite(Number(timeoutMs)) && Number(timeoutMs) > 0 ? Number(timeoutMs) : tierTimeout(config, engine, tier),
    provider: engine === "cli" ? "cli" : (provider?.name ?? "gateway"),
    // false when the model lives on a provider whose key is not configured (the candidate is
    // listed, but every call would fail with config_error until the key arrives)
    reachable: provider ? provider.apiKey.length > 0 : true,
    price: priceFor(resolvedModel, prices),
  });
}

/** `engine@model[:tier[:timeoutMs]]` entries separated by commas, or a JSON array of objects. */
export function parseCandidateSpec(spec, config, engines) {
  const text = String(spec ?? "").trim();
  if (!text) return [];
  if (text.startsWith("[")) {
    return JSON.parse(text).map(item => make(config, item, engines));
  }
  return text.split(",").map(item => item.trim()).filter(Boolean).map(item => {
    const [head, tier, timeoutMs] = item.split(":");
    const at = head.indexOf("@");
    const engine = at === -1 ? head : head.slice(0, at);
    const model = at === -1 ? null : head.slice(at + 1);
    return make(config, { engine, model: model || null, tier: tier || "reflex", timeoutMs: timeoutMs ?? null }, engines);
  });
}

/** The old two-arm shape: every cheap-chain engine on the primary model, the escalation engine on its own. */
export function defaultCandidates(config, engines) {
  const list = [];
  for (const engine of config.routerCheapChain ?? ["token"]) list.push(make(config, { engine, tier: "reflex", id: engine }, engines));
  const escalateTo = config.routerEscalateTo ?? "react";
  if (!list.some(c => c.id === escalateTo)) list.push(make(config, { engine: escalateTo, tier: "deliberate", id: escalateTo }, engines));
  return list;
}

export function buildCandidates(config, engines) {
  const list = config.routerCandidates ? parseCandidateSpec(config.routerCandidates, config, engines) : defaultCandidates(config, engines);
  const ids = list.map(c => c.id);
  if (new Set(ids).size !== ids.length) throw new Error(`duplicate candidate ids in ${ids.join(", ")}`);
  // unreachable candidates (provider key missing) stay listed and addressable, but are not routed to
  const reflex = list.filter(c => c.tier === "reflex" && c.reachable !== false).map(c => c.id);
  const deliberate = list.filter(c => c.tier === "deliberate" && c.reachable !== false).map(c => c.id);
  if (reflex.length === 0) throw new Error("at least one reflex-tier candidate is required");
  return { list, byId: new Map(list.map(c => [c.id, c])), ids, reflex, deliberate };
}

/** The decision cfg a candidate runs with: its harness, its model, its own deadline, no fallback, no race. */
export function candidateCfg(baseCfg, candidate, config = {}) {
  const cfg = {
    ...baseCfg,
    engine: candidate.engine,
    primaryModel: candidate.model,
    fallbackModel: "",
    primaryTimeoutMs: candidate.timeoutMs,
    race: false,
  };
  if (candidate.engine === "react" || candidate.tier === "deliberate") {
    cfg.maxOutputTokens = config.reactMaxOutputTokens ?? baseCfg.maxOutputTokens ?? 160;
    cfg.reactMaxIterations = config.reactMaxIterations ?? 2;
  }
  const provider = candidate.engine === "cli" ? null : providerFor(candidate.model, providersFrom(config));
  if (provider) {
    cfg.provider = provider.name;
    cfg.baseUrl = provider.baseUrl;
    cfg.apiKey = provider.apiKey;
    cfg.extraBody = provider.extraBody;
  }
  return cfg;
}
