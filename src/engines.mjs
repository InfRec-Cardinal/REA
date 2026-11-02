// G1 service engine registry: every inference engine is registered under a name with the facts the service
// needs about it (does it answer in one output token, is it the deliberate tier, does it need the
// gateway key, what /health should say). Nothing string-matches engine names any more.

export class UnknownEngineError extends Error {
  constructor(name, known) {
    super(`unknown engine '${name}'; known: ${known.join(", ")}`);
    this.name = "UnknownEngineError";
    this.errorType = "request_error";
    this.known = [...known];
  }
}

const defaultHealth = () => ({});

export const BUILTIN_ENGINE_META = Object.freeze({
  prime: { outputTokens: 1, deliberate: false, needsProvider: false, health: config => ({ harness: "prime-agent-rpc", harness_version: config.primeAgentVersion ?? null, installed_version: config.primeAgentVersion ?? null, model_output_tokens: 1, reasoning: false, persistent: true, pool_size: config.primeAgentPoolSize }) },
  token: { outputTokens: 1, deliberate: false, needsProvider: true, health: () => ({ model_output_tokens: 1, reasoning: false }) },
  tool: { outputTokens: null, deliberate: false, needsProvider: true, health: config => ({ model_output_tokens: config.maxOutputTokens, reasoning: false }) },
  react: { outputTokens: null, deliberate: true, needsProvider: true, health: config => ({ model_output_tokens: config.reactMaxOutputTokens ?? 160, reasoning: "bounded_react", max_iterations: config.reactMaxIterations ?? 2, model: config.reactModel ?? null, deliberate: true }) },
  cli: { outputTokens: null, deliberate: true, needsProvider: false, health: config => ({ model_output_tokens: null, reasoning: "cli", deliberate: true, claude_bin: config.cliClaudeBin ?? null, codex_bin: config.cliCodexBin ?? null }) },
  scripted: { outputTokens: 1, deliberate: false, needsProvider: false, health: config => ({ model_output_tokens: 1, reasoning: false, offline: true, tokens: config.scriptedTokens ?? null }) },
});

const UNKNOWN_META = Object.freeze({ outputTokens: null, deliberate: false, needsProvider: true, health: defaultHealth });

/** One output token for the one-token engines, `fallback` otherwise (and for names nobody registered). */
export function outputTokensFor(engine, fallback) {
  return BUILTIN_ENGINE_META[engine]?.outputTokens ?? fallback;
}

export function createEngineRegistry() {
  const entries = new Map();
  const registry = {
    register(name, infer, meta = {}) {
      if (typeof name !== "string" || !name) throw new TypeError("engine name must be a non-empty string");
      if (typeof infer !== "function") throw new TypeError(`engine ${name} must be a function`);
      if (entries.has(name)) throw new Error(`engine ${name} is already registered`);
      entries.set(name, { infer, meta: { ...(BUILTIN_ENGINE_META[name] ?? UNKNOWN_META), ...meta } });
      return registry;
    },
    has: name => entries.has(name),
    names: () => [...entries.keys()],
    get(name) {
      const entry = entries.get(name);
      if (!entry) throw new UnknownEngineError(name, registry.names());
      return entry.infer;
    },
    meta: name => entries.get(name)?.meta ?? null,
    describe(config) {
      return Object.fromEntries(registry.names().map(name => [name, { ...entries.get(name).meta.health(config), default: config.engine === name }]));
    },
  };
  Object.defineProperty(registry, "isEngineRegistry", { value: true });
  return registry;
}

/** A registry passes through; a plain `{ name: infer }` object is wrapped with the built-in meta for known names. */
export function asEngineRegistry(value) {
  if (value?.isEngineRegistry === true) return value;
  const registry = createEngineRegistry();
  for (const [name, infer] of Object.entries(value ?? {})) registry.register(name, infer);
  return registry;
}
