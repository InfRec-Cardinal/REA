import { resolve } from "node:path";

import { readEnv } from "../env.mjs";

import { loadConfiguration, validateConfiguration } from "./configuration.mjs";
import { createRegistry } from "./registry.mjs";
import { runEpisode } from "./runtime.mjs";
import { runEvaluation } from "./evaluation.mjs";
import { runAsyncEpisode } from "./async-runtime.mjs";
import { runRoutedEpisode } from "./routed-runtime.mjs";
import { createAsyncSession } from "./async-session.mjs";
import { createConfiguredSystems } from "./systems.mjs";

function mergeSpecification(base, override) {
  if (override === undefined) return base;
  if (typeof override === "string") return { type: override, options: {} };
  return { ...base, ...override, options: override.type && override.type !== base.type ? { ...override.options } : { ...base.options, ...override.options } };
}

function combineSignals(first, second) {
  return first && second ? AbortSignal.any([first, second]) : first ?? second;
}

/** A configured agent owns each run's environment and closes it after use. */
export async function createEmbodiedAgent(input = "demo", { baseDir, cwd = process.cwd(), env = process.env, registry = createRegistry() } = {}) {
  const loaded = await loadConfiguration(input, { env, cwd: baseDir ?? cwd, registry });
  const config = loaded.config;
  const directory = loaded.baseDir;
  const factoryConfig = typeof config.createEnvironment === "function";
  const usesSystems = Boolean(config.systems || config.createSystems);
  let active = null;
  let activeController = null;
  let closed = false;

  async function createEnvironment(episode) {
    if (factoryConfig) return config.createEnvironment(episode);
    return registry.createEnvironment(mergeSpecification(config.environment, episode.environment), { env, baseDir: directory, episode });
  }

  async function createPolicy(episode, environment) {
    if (factoryConfig) return config.createPolicy(episode, environment);
    return registry.createPolicy(mergeSpecification(config.policy, episode.policy), { env, baseDir: directory, episode, environment });
  }

  async function createSystems(episode, environment) {
    if (config.createSystems) {
      const systems = await config.createSystems(episode, environment);
      if (systems.close) return systems;
      return { ...systems, async close() {
        const resources = new Set([...systems.system0.map(candidate => candidate.policy), systems.system1, systems.system2].filter(Boolean));
        const results = await Promise.allSettled([...resources].map(resource => resource.close?.()));
        const failure = results.find(result => result.status === "rejected");
        if (failure) throw failure.reason;
      } };
    }
    return createConfiguredSystems(config.systems, { env, baseDir: directory, episode, environment, registry });
  }

  function start(operation) {
    if (closed) throw new Error("This embodied agent is closed.");
    if (active) throw new Error("This embodied agent already has a run in progress.");
    const controller = new AbortController();
    activeController = controller;
    active = Promise.resolve().then(() => operation(controller.signal)).finally(() => {
      active = null;
      activeController = null;
    });
    return active;
  }

  function publicMetadata() {
    return {
      ...config.metadata,
      framework: {
        name: "rea-agent", configurationVersion: 1,
        environment: config.environment?.type ?? "factory",
        policy: usesSystems ? "system0/system1/system2" : config.policy?.type ?? "factory",
        ...(usesSystems ? { runtime: config.systems?.mode === "routed" ? "routed-system" : "async-system", mode: config.systems?.mode ?? "async" } : {}),
        model: config.policy?.options?.model ?? readEnv(env, "REA_MODEL") ?? null,
      },
    };
  }

  const agent = {
    config,
    registry,
    /** Run the first configured episode with optional task/seed/instruction/budget overrides. */
    run({ signal, onTransition, onEvent, mode, budget = {}, ...overrides } = {}) {
      return start(async internalSignal => {
        const episode = { ...config.episodes[0], ...overrides };
        const requestSignal = combineSignals(internalSignal, signal);
        let environment;
        let policy;
        let systems;
        let result;
        try {
          requestSignal.throwIfAborted();
          environment = await createEnvironment(episode);
          if (usesSystems) systems = await createSystems(episode, environment);
          else policy = await createPolicy(episode, environment);
          const runner = !usesSystems ? runEpisode : (systems.mode ?? config.systems?.mode) === "routed" ? runRoutedEpisode : runAsyncEpisode;
          result = await runner({
            ...systems,
            ...(mode ? { mode } : {}),
            ...config.budget, ...episode.budget, ...budget,
            environment, policy, task: episode.task, seed: episode.seed,
            instruction: episode.instruction, signal: requestSignal, onTransition, onEvent,
          });
          if (environment.scorecard) result.officialScorecard = await environment.scorecard({ close: true });
          return result;
        } finally {
          try { await (systems ?? policy)?.close?.(); } finally { await environment?.close?.(); }
        }
      });
    },
    /** Start immediately and subscribe to live System0/1/2 events without waiting for completion. */
    start(options = {}) {
      if (!usesSystems) throw new Error("start() requires a systems configuration; use run() for a single policy.");
      return createAsyncSession({ mode: config.systems?.mode ?? "async", ...options }, { run: options => agent.run(options) });
    },
    /** Evaluate the configured episode plan using the shared reports and scorecard lifecycle. */
    evaluate({ outputDir, episodes = config.episodes, budget = {}, signal, onProgress, onEvent, recordImages = config.recordImages ?? false } = {}) {
      return start(async internalSignal => {
        const plan = validateConfiguration({ ...config, episodes, budget: { ...config.budget, ...budget } }, { registry });
        const target = resolve(directory, outputDir ?? config.outputDir ?? `outputs/run-${new Date().toISOString().replaceAll(":", "-")}`);
        const summary = await runEvaluation({
          episodes: plan.episodes, createEnvironment, createPolicy,
          ...(usesSystems ? { createSystems } : {}),
          budget: plan.budget, outputDir: target, metadata: publicMetadata(),
          reuseEnvironment: config.reuseEnvironment ?? (config.environment?.type === "arc-agi3" && !plan.episodes.some(episode => episode.environment)),
          recordImages, onProgress, onEvent, signal: combineSignals(internalSignal, signal),
        });
        return { ...summary, outputDir: target };
      });
    },
    async close() {
      closed = true;
      activeController?.abort(new Error("embodied agent closed"));
      if (active) await active.catch(() => {});
    },
  };
  return agent;
}
