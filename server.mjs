#!/usr/bin/env node
// REA G1 service — a single-token embodied controller for the Unitree G1.
//
// One-token decisions with an asynchronous reviewer and a fail-closed safety gate, served by
// persistent workers over a streaming transport. A one-token answer means the first streamed chunk
// is already the whole decision, and persistent workers mean there is no process start-up around it,
// so the same service serves a considered decision and a reactive one from the same code path.

import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync } from "node:fs";
import { join } from "node:path";

import { createAgent } from "./src/agent.mjs";
import { createBrain } from "./src/brain.mjs";
import { createFixtureCamera, createSubprocessCamera, createWorkerCamera } from "./src/camera.mjs";
import { createCerebellumActuator, createCerebellumClient } from "./src/cerebellum.mjs";
import { loadConfig, senseKindOf } from "./src/config.mjs";
import { createEngineRegistry } from "./src/engines.mjs";
import { loadDotEnv } from "./src/env.mjs";
import {
  createConsoleActuator,
  createDaemonActuator,
  createExecutor,
  createMockActuator,
  createNullActuator,
} from "./src/executor.mjs";
import { buildCandidates, pricesFrom } from "./src/candidates.mjs";
import { createCliInfer } from "./src/infer-cli.mjs";
import { createModelRouter } from "./src/infer-router.mjs";
import { createHttpServer } from "./src/http.mjs";
import { createPrimeAgentInfer } from "./src/infer-prime.mjs";
import { createReactInfer } from "./src/infer-react.mjs";
import { createReflectInfer } from "./src/infer-reflect.mjs";
import { createScriptedInfer, defaultScriptedTokens } from "./src/infer-scripted.mjs";
import { createEpisodeStore } from "./src/episodes.mjs";
import { createTokenInfer } from "./src/infer-token.mjs";
import { createToolInfer } from "./src/infer-tool.mjs";
import { createExperienceMemory } from "./src/memory.mjs";
import { resolveProfile } from "./src/profiles/index.mjs";
import { createPrimeAgentPool } from "./src/prime-pool.mjs";
import { createDispatcher } from "./src/provider.mjs";
import { createRouter } from "./src/router.mjs";
import { createFixtureObstacleSource, createSnapshotObstacleSource } from "./src/sensors/obstacles.mjs";
import { createSessionStore } from "./src/sessions.mjs";
import { createThinkerClient } from "./src/thinker-client.mjs";
import { SERVICE, VERSION } from "./src/version.mjs";
import { LineWorker } from "./src/workers.mjs";

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "::1", "localhost"]);

function writeLog(level, record) {
  process.stdout.write(`${JSON.stringify({ at: new Date().toISOString(), level, ...record })}\n`);
}

const logger = {
  info: (record) => writeLog("info", record),
  warn: (record) => writeLog("warn", record),
  error: (record) => writeLog("error", record),
};

let primeRuntime = null;
let cleanupRuntime = async () => {
  await primeRuntime?.close();
};
let primeModelReady = false;
let primeModelError = null;
let primeModelChecks = {};

function verifyPrimeAgentBinary(config) {
  const result = spawnSync(config.primeAgentBin, ["--version"], {
    cwd: config.primeAgentCwd,
    env: process.env,
    encoding: "utf8",
    timeout: 5000,
  });
  // Some environments print an informational warning to stderr before the version, so parse
  // the final version-shaped line instead of comparing the whole stream.
  const version = `${result.stdout ?? ""}\n${result.stderr ?? ""}`
    .split(/\r?\n/u)
    .map(line => line.trim())
    .findLast(line => /^\d+\.\d+\.\d+$/u.test(line)) ?? "";
  if (result.status !== 0 || version !== config.primeAgentVersion) {
    throw new Error(`Prime Agent ${config.primeAgentVersion} is required; observed ${version || "unavailable"}`);
  }
  return version;
}

/**
 * Prime keeps its credentials in <config dir>/auth.json. The repository ships auth.example.json (an
 * empty object) and never the real file, so a fresh checkout gets its own copy here.
 */
function ensurePrimeAuthFile(config) {
  const target = join(config.primeAgentConfigDir, "auth.json");
  const example = join(config.primeAgentConfigDir, "auth.example.json");
  if (!existsSync(target) && existsSync(example)) copyFileSync(example, target);
}

async function main() {
  loadDotEnv();
  const config = loadConfig();
  const profile = await resolveProfile(config.robotProfile);
  if (config.engine === "prime") {
    ensurePrimeAuthFile(config);
    verifyPrimeAgentBinary(config);
  }
  if (!LOOPBACK_HOSTS.has(config.host) && !config.agentToken) {
    logger.warn({
      event: "unauthenticated_lan_bind",
      host: config.host,
      execute_enabled: config.executeEnabled,
      hint: "set G1_AGENT_TOKEN, or bind G1_AGENT_HOST=127.0.0.1; every route except /health is open to the network",
    });
  }
  const workers = {};

  // ---- sensing
  let camera;
  const sense = senseKindOf(config);
  if (sense === "fixture") {
    camera = createFixtureCamera({ frames: config.fixtureDepth });
    logger.warn({ event: "fixture_camera", frames: config.fixtureDepth.length, hint: "G1_SENSE=fixture: synthetic frames and depth; no robot is observed" });
  } else if (sense === "worker") {
    workers.sense = new LineWorker("sense", config.python, [config.senseWorker], { env: { ...process.env, G1_REALSENSE_SOCKET: config.cameraSocket } });
    camera = createWorkerCamera({ worker: workers.sense, timeoutMs: config.captureTimeoutMs });
  } else {
    camera = createSubprocessCamera({ python: config.python, script: config.senseScript, socketPath: config.cameraSocket, timeoutMs: config.captureTimeoutMs, maxWidth: config.maxWidth, jpegQuality: config.quality });
  }
  const obstacles = config.obstacles === "fixture"
    ? createFixtureObstacleSource(config.fixtureLidar, { profile })
    : createSnapshotObstacleSource({ path: config.lidarSnapshot, profile });

  // ---- actuation
  let actuator;
  if (config.actuator === "daemon" && config.persistentControl) {
    workers.control = new LineWorker("control", config.python, [config.controlDaemon], { env: process.env });
    actuator = createDaemonActuator({ worker: workers.control, config });
  } else if (config.actuator === "console") {
    actuator = createConsoleActuator({ baseUrl: config.consoleBaseUrl, config });
  } else if (config.actuator === "cerebellum") {
    actuator = createCerebellumActuator({
      client: createCerebellumClient({ socketPath: config.cerebellumSocket, timeoutMs: config.controlTimeoutMs }),
      config,
      logger,
    });
  } else if (config.actuator === "mock") {
    actuator = createMockActuator();
    logger.warn({ event: "mock_actuator", hint: "G1_ACTUATOR=mock: an always-ready robot that never moves; evaluation only" });
  } else {
    actuator = createNullActuator();
  }
  const executor = createExecutor({ actuator, config, logger, profile, obstacles });

  // ---- inference engines, all sharing one keep-alive dispatcher
  // The scripted default is the profile's forward role, not a literal "W": a profile without W would
  // otherwise abort at startup on the engine registration below, whatever G1_ENGINE is. Written back
  // onto config so /v1/capabilities engines.scripted.tokens reports what the engine actually replays.
  if (process.env.G1_SCRIPTED_TOKENS === undefined) config.scriptedTokens = defaultScriptedTokens(profile);
  const dispatcher = await createDispatcher();
  const tokenInfer = createTokenInfer({ apiKey: config.apiKey, baseUrl: config.baseUrl, dispatcher, profile });
  const primeInfer = createPrimeAgentPool({
    size: config.primeAgentPoolSize,
    createInfer: () => createPrimeAgentInfer({
      bin: config.primeAgentBin,
      configDir: config.primeAgentConfigDir,
      provider: config.primeAgentProvider,
      model: config.primaryModel,
      cwd: config.primeAgentCwd,
      profile,
    }),
  });
  primeRuntime = primeInfer;
  cleanupRuntime = async () => {
    for (const worker of Object.values(workers)) worker.stop();
    await primeInfer.close();
  };
  const engines = createEngineRegistry()
    .register("prime", primeInfer)
    .register("token", tokenInfer)
    .register("tool", createToolInfer({ apiKey: config.apiKey, baseUrl: config.baseUrl, dispatcher, profile }))
    // The deliberate engine: bounded ReAct, the router's escalation target. Off the reflex path.
    .register("react", createReactInfer({
      apiKey: config.apiKey,
      baseUrl: config.baseUrl,
      dispatcher,
      maxOutputTokens: config.reactMaxOutputTokens,
      maxIterations: config.reactMaxIterations,
      profile,
    }))
    // The offline engine: a fixed token script, no provider, so the demo and CI run the real service.
    .register("scripted", createScriptedInfer({ tokens: config.scriptedTokens, profile }));
  // Frontier models behind official CLIs only (Claude, Codex); asynchronous roles, never the reflex path.
  if (config.cliEngines) engines.register("cli", createCliInfer({ claudeBin: config.cliClaudeBin, codexBin: config.cliCodexBin, effort: config.cliEffort, profile }));
  if (!engines.has(config.engine)) {
    throw new Error(`G1_ENGINE=${config.engine} is not registered${config.engine === "cli" ? "; set G1_CLI_ENGINES=1" : ""}. Registered: ${engines.names().join(", ")}`);
  }

  // ---- brain: persisted experience + per-step router + judge (memory and routing only, no actuator)
  let brain = null;
  if (config.brainEnabled) {
    const experience = createExperienceMemory({ path: config.memoryPath, maxLessons: config.memoryMaxLessons, logger, profile });
    const loaded = await experience.load();
    // Candidates the router chooses between: (engine, model, tier) triples; ids replace bare engine names.
    const candidates = buildCandidates(config, engines.names());
    const router = createRouter({
      cheapChain: candidates.reflex,
      escalateTo: candidates.deliberate[0] ?? config.routerEscalateTo,
      k: config.routerK,
      cooldown: config.routerCooldown,
      maxDeliberateFraction: config.routerMaxDeliberateFraction,
      inlineEscalate: config.routerInlineEscalate,
      engines: candidates.ids,
      candidates: candidates.list,
      mode: config.routerMode,
      path: config.routerMemoryPath,
      logger,
      profile,
    });
    const routing = await router.load();
    // The asynchronous model router (Agent-as-a-Router): suggests the next step's candidate off the control path.
    const modelRouter = config.routerMode !== "rule" && config.providerConfigured
      ? createModelRouter({
          apiKey: config.apiKey,
          baseUrl: config.baseUrl,
          dispatcher,
          model: config.routerModel,
          router,
          candidates,
          logger,
          epsilon: config.routerExplore,
          minTries: config.routerMinTries,
          deadlineMs: config.routerModelTimeoutMs,
          stepBudgetMs: config.brainStepBudgetMs,
        })
      : null;
    // Reflection needs a provider; without a key the brain still judges and remembers heuristically.
    const reflect = config.brainReflect && config.providerConfigured
      ? {
          infer: createReflectInfer({ apiKey: config.apiKey, baseUrl: config.baseUrl, dispatcher, maxOutputTokens: config.reflectMaxOutputTokens, profile }),
          model: config.reflectModel,
          deadlineMs: config.reflectTimeoutMs,
          cooldown: config.reflectCooldown,
        }
      : null;
    // The episode store always exists; with an empty path it lives in memory for the process only.
    const episodes = createEpisodeStore({ path: config.episodesPath, maxSessions: config.episodesMaxSessions, logger });
    const episodesLoaded = await episodes.load();
    brain = createBrain({
      experience,
      router,
      logger,
      recallK: config.memoryRecallK,
      promptMaxChars: config.memoryPromptMaxChars,
      reflect,
      episodicCapacity: config.episodicCapacity,
      episodes,
      episodeRecallK: config.episodesRecallK,
      candidates,
      prices: pricesFrom(config),
      modelRouter,
      profile,
    });
    logger.info({
      event: "brain_ready",
      lessons_loaded: loaded.loaded,
      memory_path: config.memoryPath,
      router_scores_loaded: routing.loaded,
      router_memory_path: config.routerMemoryPath,
      episodes_loaded: episodesLoaded.sessions,
      episodes_path: config.episodesPath,
      episodic_capacity: config.episodicCapacity,
      cheap_chain: candidates.reflex,
      escalate_to: candidates.deliberate[0] ?? config.routerEscalateTo,
      candidates: candidates.ids,
      candidate_providers: Object.fromEntries(candidates.list.map(c => [c.id, c.provider])),
      unreachable_candidates: candidates.list.filter(c => c.reachable === false).map(c => c.id),
      router_mode: config.routerMode,
      router_model: modelRouter ? modelRouter.model : null,
      reflect: reflect ? { model: reflect.model, cooldown: reflect.cooldown } : null,
    });
  }

  // ---- session + advisory thinker
  const sessions = createSessionStore({ profile });
  const thinkerClient = config.thinkerEnabled
    ? createThinkerClient({ baseUrl: config.thinkerBaseUrl, submitTimeoutMs: config.thinkerSubmitTimeoutMs })
    : null;

  // The reviewer must never sit on the control path: submission is fire-and-forget and a failure to
  // reach it is logged, never surfaced as a control error.
  const thinker = thinkerClient
    ? {
        submit(job) {
          const run = async () => {
            let afterFrame = null;
            if (job.captureAfter) {
              afterFrame = await camera.capture({ maxWidth: config.maxWidth, quality: config.quality, filter: config.filter }).catch((error) => {
                logger.warn({ event: "thinker_after_frame_failed", session_id: job.sessionId, seq: job.seq, error: error.message });
                return null;
              });
            }
            const { captureAfter: _drop, ...rest } = job;
            await thinkerClient.submit({ ...rest, afterFrame });
          };
          setImmediate(() => {
            void run().catch((error) => {
              logger.warn({ event: "thinker_submit_failed", session_id: job.sessionId, seq: job.seq, error: error.message });
            });
          });
          return true;
        },
      }
    : null;

  const agent = createAgent({ camera, engines, executor, config, sessions, thinker, brain, logger, profile });

  let primeWarmPromise = null;
  async function warmPrimeModels() {
    if (primeWarmPromise) return primeWarmPromise;
    const current = (async () => {
      const checks = {};
      let frame;
      try {
        frame = await camera.capture({
          maxWidth: config.maxWidth,
          quality: config.quality,
          filter: config.filter,
          needDepth: config.includeDepth,
        });
      } catch (error) {
        const message = error?.message ?? String(error);
        checks.capture = { ready: false, error: message };
        primeModelChecks = checks;
        primeModelReady = false;
        primeModelError = message;
        logger.warn({ event: "prime_model_warm_failed", stage: "capture", error: message });
        return false;
      }

      // Exercise the fallback once, then warm every slot on the primary. This validates both
      // configured models while leaving all resident workers on the latency-critical primary.
      if (config.fallbackModel && config.fallbackModel !== config.primaryModel) {
        try {
          const result = await primeInfer(config.fallbackModel, frame, {
            instruction: "Validate the fallback bounded decision path and output the safe stop token X.",
            deadlineMs: config.fallbackTimeoutMs,
            cfg: { promptVariant: config.promptVariant },
          });
          if (result.token !== "X") throw new Error("fallback warm-up did not return X");
          checks.fallback = { model: config.fallbackModel, ready: true, error: null, workers: 1 };
        } catch (error) {
          checks.fallback = {
            model: config.fallbackModel,
            ready: false,
            error: error?.message ?? String(error),
            workers: 1,
          };
        }
      }

      try {
        const warm = await primeInfer.warmAll(config.primaryModel, frame, {
          instruction: "Warm the primary bounded decision path and output the safe stop token X.",
          deadlineMs: config.primaryTimeoutMs,
          cfg: { promptVariant: config.promptVariant },
        });
        if (!warm.every(result => result.token === "X")) {
          throw new Error("primary warm-up did not return X on every worker");
        }
        checks.primary = {
          model: config.primaryModel,
          ready: true,
          error: null,
          workers: warm.length,
          total_s: warm.map(result => result.totalS),
        };
      } catch (error) {
        checks.primary = {
          model: config.primaryModel,
          ready: false,
          error: error?.message ?? String(error),
          workers: config.primeAgentPoolSize,
        };
      }

      const required = Object.values(checks).filter(check => Object.hasOwn(check, "model"));
      primeModelChecks = checks;
      primeModelReady = required.length > 0 && required.every(check => check.ready === true);
      primeModelError = required
        .filter(check => check.ready !== true)
        .map(check => `${check.model}: ${check.error}`)
        .join("; ") || null;
      if (primeModelReady) {
        logger.info({ event: "prime_models_warm", models: checks });
      } else {
        logger.warn({ event: "prime_model_warm_failed", models: checks, error: primeModelError });
      }
      return primeModelReady;
    })();
    primeWarmPromise = current;
    try {
      return await current;
    } finally {
      if (primeWarmPromise === current) primeWarmPromise = null;
    }
  }

  // ---- bring the workers up before accepting traffic, then keep the provider tunnel warm
  const ready = await Promise.all(Object.values(workers).map((worker) => worker.ready.catch(() => null)));
  logger.info({ event: "workers_ready", workers: Object.keys(workers), details: ready.map((entry) => entry?.status ?? null) });

  if (config.engine === "prime") {
    logger.info({ event: "prime_agent_ready", state: await primeInfer.start() });
    if (config.providerConfigured && config.primeWarmup) {
      await warmPrimeModels();
      if (!primeModelReady && config.warmIntervalS > 0) {
        setInterval(() => {
          if (!primeModelReady) void warmPrimeModels();
        }, config.warmIntervalS * 1000).unref();
      }
    } else {
      primeModelReady = !config.providerConfigured || !config.primeWarmup;
      primeModelChecks = {};
    }
  }

  if (config.engine !== "prime" && config.providerConfigured && config.warmIntervalS > 0) {
    logger.info({ event: "warm", status: await tokenInfer.warm() });
    setInterval(() => {
      void tokenInfer.warm();
    }, config.warmIntervalS * 1000).unref();
  }

  const server = createHttpServer({
    agent,
    config,
    logger,
    workers,
    harnessState: () => ({
      ...primeInfer.state(),
      modelReady: config.engine === "prime" ? primeModelReady : null,
      modelError: config.engine === "prime" ? primeModelError : null,
      models: config.engine === "prime" ? primeModelChecks : null,
      modelVerified: config.engine === "prime" ? config.primeWarmup : null,
    }),
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(config.port, config.host, resolve);
  });

  logger.info({
    event: "service_started",
    service: SERVICE,
    robot_profile: profile.id,
    version: VERSION,
    harness: config.agentHarness,
    harness_version: config.harnessVersion,
    auth_required: Boolean(config.agentToken),
    harness_state: primeInfer.state(),
    host: config.host,
    port: server.address().port,
    engine: config.engine,
    model_output_tokens: config.modelOutputTokens,
    primary_model: config.primaryModel,
    fallback_model: config.fallbackModel,
    actuator: config.actuator,
    execute_enabled: config.executeEnabled,
    persistent_capture: config.persistentCapture,
    persistent_control: config.persistentControl && config.actuator === "daemon",
    use_lidar: config.useLidar,
    thinker: config.thinkerEnabled ? config.thinkerBaseUrl : null,
    brain: brain ? brain.state().summary : null,
    provider_configured: config.providerConfigured,
  });

  let stopping = false;
  const close = (signal) => {
    if (stopping) return;
    stopping = true;
    logger.info({ event: "service_stopping", signal });
    agent.pipeline.stop();
    // Best-effort zero velocity on the way out: never leave a pulse running after the service dies.
    void agent
      .stop()
      .catch(() => {})
      .finally(async () => {
        for (const worker of Object.values(workers)) worker.stop();
        await primeInfer.close();
        await brain?.experience.flush().catch(() => {});
        await brain?.router.flush().catch(() => {});
        await brain?.episodes?.flush().catch(() => {});
        server.close(() => process.exit(0));
        setTimeout(() => process.exit(0), 2000).unref();
      });
  };
  process.once("SIGINT", () => close("SIGINT"));
  process.once("SIGTERM", () => close("SIGTERM"));
}

main().catch(async (error) => {
  await cleanupRuntime();
  logger.error({
    event: "service_start_failed",
    error_type: "config_error",
    error: error instanceof Error ? error.message : String(error),
  });
  process.exitCode = 1;
});
