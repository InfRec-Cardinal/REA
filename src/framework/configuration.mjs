import { readFile } from "node:fs/promises";
import { dirname, extname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { getPreset, listPresets } from "./presets.mjs";

const plain = value => value !== null && typeof value === "object" && Object.getPrototypeOf(value) === Object.prototype;

export function resolveEnvironmentReferences(value, env = process.env) {
  if (Array.isArray(value)) return value.map(item => resolveEnvironmentReferences(item, env));
  if (!plain(value)) return value;
  if (typeof value.env === "string" && Object.keys(value).every(key => ["env", "default"].includes(key))) {
    if (env[value.env] !== undefined && env[value.env] !== "") return env[value.env];
    if (Object.hasOwn(value, "default")) return resolveEnvironmentReferences(value.default, env);
    throw new Error(`Set ${value.env}, or give its configuration reference a default.`);
  }
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, resolveEnvironmentReferences(item, env)]));
}

function specification(value, label, fallback) {
  const result = typeof value === "string" ? { type: value } : value ?? fallback;
  if (!plain(result) || typeof result.type !== "string" || !result.type.trim()) throw new TypeError(`${label} requires a type, for example {type: "${label === "environment" ? "aloha" : "chat"}"}.`);
  if (result.options !== undefined && !plain(result.options)) throw new TypeError(`${label}.options must be an object.`);
  return { ...result, options: { ...result.options } };
}

function validateBudget(budget, label) {
  if (budget === undefined) return;
  if (!plain(budget)) throw new TypeError(`${label} must be an object.`);
  if (budget.maxSteps !== undefined && (!Number.isInteger(budget.maxSteps) || budget.maxSteps < 1)) throw new RangeError(`${label}.maxSteps must be a positive integer.`);
  for (const key of ["maxDurationMs", "decisionTimeoutMs"]) {
    if (budget[key] !== undefined && (!Number.isFinite(budget[key]) || budget[key] <= 0)) throw new RangeError(`${label}.${key} must be positive milliseconds.`);
  }
}

export function validateConfiguration(input, { registry } = {}) {
  if (!plain(input)) throw new TypeError("Configuration must be an object.");
  if (input.version !== undefined && input.version !== 1) throw new Error(`Unsupported configuration version ${input.version}; use version 1.`);
  const legacy = typeof input.createEnvironment === "function" && (typeof input.createPolicy === "function" || typeof input.createSystems === "function");
  if (!legacy && (input.createEnvironment || input.createPolicy || input.createSystems)) throw new TypeError("Factory configurations require createEnvironment and createPolicy or createSystems.");
  const config = { ...input, version: 1 };
  if (!legacy) {
    config.environment = specification(input.environment, "environment");
    if (!input.systems || input.policy) config.policy = specification(input.policy, "policy", { type: "chat" });
    for (const [kind, spec, entries] of [
      ["environment", config.environment, registry?.listEnvironments?.()],
      ["policy", config.policy, registry?.listPolicies?.()],
    ]) {
      if (spec && entries && !entries.some(item => (item.type ?? item.id) === spec.type)) throw new Error(`Unknown ${kind} type '${spec.type}'. Register it or choose an installed ${kind}.`);
    }
  }
  if (input.systems) {
    const systems = { mode: "async", ...input.systems };
    if (!["async", "sync", "routed"].includes(systems.mode)) throw new TypeError("systems.mode must be async, sync or routed.");
    if (!Array.isArray(systems.system0) || !systems.system0.length) throw new TypeError("systems.system0 requires at least one controller.");
    const ids = new Set();
    systems.system0 = systems.system0.map(candidate => {
      if (typeof candidate.id !== "string" || !candidate.id || ids.has(candidate.id)) throw new TypeError("System0 controllers need unique nonempty ids.");
      ids.add(candidate.id);
      return { ...candidate, policy: specification(candidate.policy, "policy") };
    });
    config.systems = systems;
  }
  config.episodes = input.episodes ?? [{
    id: input.name ?? "episode-1", task: input.task ?? null,
    seed: input.seed ?? 0, instruction: input.instruction,
  }];
  if (!Array.isArray(config.episodes) || !config.episodes.length) throw new TypeError("episodes must be a nonempty array.");
  config.episodes = config.episodes.map((episode, index) => {
    if (!plain(episode)) throw new TypeError(`episodes[${index}] must be an object.`);
    if (episode.seed !== undefined && (!Number.isSafeInteger(episode.seed) || episode.seed < 0)) throw new RangeError(`episodes[${index}].seed must be a nonnegative integer.`);
    validateBudget(episode.budget, `episodes[${index}].budget`);
    return { ...episode, id: episode.id ?? `episode-${index + 1}`, seed: episode.seed ?? 0 };
  });
  validateBudget(config.budget, "budget");
  if (config.reuseEnvironment && config.episodes.some(episode => episode.environment)) throw new Error("Per-episode environments need reuseEnvironment: false.");
  return config;
}

/** Presets and JSON recipes are inert; JS files execute their exported configuration. */
export async function loadConfiguration(input, { env = process.env, cwd = process.cwd(), registry } = {}) {
  let config = input;
  let baseDir = resolve(cwd);
  let source = "inline";
  if (typeof input === "string") {
    if (listPresets().some(preset => (preset.id ?? preset.name) === input)) {
      config = getPreset(input);
      source = `preset:${input}`;
    } else {
      const path = resolve(cwd, input);
      baseDir = dirname(path);
      source = path;
      if (extname(path) === ".json") config = JSON.parse(await readFile(path, "utf8"));
      else if ([".mjs", ".js", ".cjs"].includes(extname(path))) {
        const imported = await import(pathToFileURL(path).href);
        config = typeof imported.default === "function" ? await imported.default({ env, baseDir }) : imported.default;
      } else throw new Error(`Unknown preset or configuration '${input}'. Use rea list, a .json recipe, or a .mjs configuration.`);
    }
  }
  config = resolveEnvironmentReferences(config, env);
  return { config: validateConfiguration(config, { registry }), baseDir, source };
}
