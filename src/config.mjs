import { fileURLToPath } from "node:url";
import { join } from "node:path";

import { parseFixtureDepth } from "./camera.mjs";
import { BUILTIN_ENGINE_META } from "./engines.mjs";
import { readEnv } from "./env.mjs";
import { parseFixtureLidar } from "./sensors/obstacles.mjs";
import { VERSION } from "./version.mjs";

const DEFAULT_ROOT = fileURLToPath(new URL("../", import.meta.url));
const DEFAULT_SENSE_SCRIPT = fileURLToPath(new URL("../scripts/g1/g1_u2_sense.py", import.meta.url));
const DEFAULT_SENSE_WORKER = fileURLToPath(new URL("../scripts/g1/sense_worker.py", import.meta.url));
const DEFAULT_CONTROL_DAEMON = fileURLToPath(new URL("../scripts/g1/control_daemon.py", import.meta.url));

function integer(env, name, fallback, { minimum = 1, maximum = Number.MAX_SAFE_INTEGER } = {}) {
  const value = Number(env[name] ?? fallback);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be an integer between ${minimum} and ${maximum}`);
  }
  return value;
}

function number(env, name, fallback, { minimum = 0, maximum = Number.MAX_VALUE } = {}) {
  const value = Number(env[name] ?? fallback);
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be a number between ${minimum} and ${maximum}`);
  }
  return value;
}

function boolean(env, name, fallback = false) {
  const raw = env[name];
  if (raw === undefined) return fallback;
  const value = String(raw).trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(value)) return true;
  if (["0", "false", "no", "off"].includes(value)) return false;
  throw new Error(`${name} must be a boolean`);
}

function choice(env, name, fallback, allowed) {
  const value = env[name] ?? fallback;
  if (!allowed.includes(value)) {
    throw new Error(`${name} must be one of ${allowed.join(", ")}`);
  }
  return value;
}

// The registry's meta table is the single list of engine names: config validates against it.
export const ENGINES = Object.keys(BUILTIN_ENGINE_META);

function engineList(env, name, fallback) {
  const raw = env[name];
  const list = (raw === undefined || raw.trim() === "" ? fallback : raw.split(",").map(item => item.trim()).filter(Boolean));
  for (const engine of list) {
    if (!ENGINES.includes(engine)) throw new Error(`${name} contains unknown engine "${engine}"; allowed: ${ENGINES.join(", ")}`);
  }
  if (list.length === 0) throw new Error(`${name} must name at least one engine`);
  return list;
}

export function loadConfig(env = process.env) {
  const apiKey = readEnv(env, "REA_API_KEY") ?? "";
  const engine = choice(env, "G1_ENGINE", "prime", ENGINES);
  const meta = BUILTIN_ENGINE_META[engine];
  const primaryModel = env.G1_PRIMARY_MODEL ?? readEnv(env, "REA_MODEL") ?? "";
  const fallbackModel = env.G1_FALLBACK_MODEL ?? "";
  // The deliberate engine gets a normal token budget, so it may use the fallback (multi-token)
  // model when one is configured; the single-token primary only ever needs one token.
  const reactModel = env.G1_REACT_MODEL ?? (fallbackModel || primaryModel);
  const maxOutputTokens = integer(env, "G1_MAX_OUTPUT_TOKENS", 64, { maximum: 256 });
  // The offline engine's token script: one comma-separated sequence, replayed per decision.
  const scriptedTokens = (env.G1_SCRIPTED_TOKENS ?? "W").split(",").map(item => item.trim()).filter(Boolean);
  if (scriptedTokens.length === 0) throw new Error("G1_SCRIPTED_TOKENS must name at least one token");
  const config = {
    // ---- service
    robotProfile: env.G1_ROBOT_PROFILE ?? "g1",
    host: env.G1_AGENT_HOST ?? env.G1_ACTION_HOST ?? env.HOST ?? "127.0.0.1",
    // 0 means "any free port"; server.mjs logs the one the listener actually bound.
    port: integer(env, "G1_AGENT_PORT", env.G1_ACTION_PORT ?? env.PORT ?? 8091, { minimum: 0, maximum: 65535 }),
    // Shared secret for every route except GET /health. Empty (the default) leaves the service open,
    // which is only safe on loopback; set it whenever the bind address is not loopback.
    agentToken: env.G1_AGENT_TOKEN ?? "",
    version: VERSION,

    // ---- provider
    baseUrl: readEnv(env, "REA_BASE_URL") ?? "https://api.openai.com/v1",
    apiKey,
    providerConfigured: apiKey.length > 0,
    primaryModel,
    // The single-token engines (prime, token) pin max_tokens=1, and many vision models return empty
    // content under that limit. A fallback that cannot satisfy the contract only makes the readiness
    // probe fail, so there is no default fallback: set G1_FALLBACK_MODEL to a model that has been
    // checked with the engine in use.
    fallbackModel,
    primaryTimeoutMs: integer(env, "G1_PRIMARY_TIMEOUT_MS", 6000),
    fallbackTimeoutMs: integer(env, "G1_FALLBACK_TIMEOUT_MS", 8000),
    maxOutputTokens,
    warmIntervalS: integer(env, "G1_WARM_INTERVAL_S", 20, { minimum: 0, maximum: 600 }),

    // ---- Prime Agent persistent RPC harness
    agentHarness: engine === "prime" ? "prime-agent-rpc" : `rea-${engine}`,
    primeAgentVersion: env.G1_PRIME_AGENT_VERSION ?? "0.8.0",
    // What /health reports as harness_version: Prime's version when Prime runs, else this service's.
    harnessVersion: engine === "prime" ? (env.G1_PRIME_AGENT_VERSION ?? "0.8.0") : VERSION,
    primeAgentBin:
      env.PRIME_AGENT_BIN ?? join(DEFAULT_ROOT, "runtime", "prime-agent", "node_modules", ".bin", "prime-agent"),
    primeAgentConfigDir:
      env.PRIME_AGENT_CODING_AGENT_DIR ?? join(DEFAULT_ROOT, "prime-agent-config"),
    primeAgentProvider: env.G1_PRIME_AGENT_PROVIDER ?? "openai-compatible",
    primeAgentCwd: env.G1_PRIME_AGENT_CWD ?? DEFAULT_ROOT,
    primeAgentPoolSize: integer(env, "G1_PRIME_AGENT_POOL_SIZE", 2, { maximum: 4 }),
    primeWarmup: boolean(env, "G1_PRIME_WARMUP", true),

    // ---- decision defaults (every key is overridable per request through `cfg`)
    engine,
    // From the engine's registry meta: does it need the gateway key, and how many output tokens
    // does it answer in. Nothing downstream string-matches engine names to work these out.
    providerRequired: meta.needsProvider,
    modelOutputTokens: meta.outputTokens ?? maxOutputTokens,
    scriptedTokens,
    promptVariant: choice(env, "G1_PROMPT_VARIANT", "short", ["full", "short", "min"]),
    schema: choice(env, "G1_SCHEMA", "compact", ["full", "compact"]),
    maxWidth: integer(env, "G1_IMG_WIDTH", 320, { maximum: 4096 }),
    quality: integer(env, "G1_JPEG_Q", 40, { minimum: 1, maximum: 100 }),
    filter: env.G1_SENSE_FILTER ?? "cartoon",
    includeDepth: boolean(env, "G1_PROMPT_DEPTH", true),
    imageDetail: env.G1_IMAGE_DETAIL === "" ? null : (env.G1_IMAGE_DETAIL ?? "low"),
    race: boolean(env, "G1_RACE", false),

    // ---- sensing
    python: env.G1_SENSE_PYTHON ?? "/usr/bin/python3",
    senseScript: env.G1_SENSE_SCRIPT ?? DEFAULT_SENSE_SCRIPT,
    senseWorker: env.G1_SENSE_WORKER ?? DEFAULT_SENSE_WORKER,
    cameraSocket: env.G1_REALSENSE_SOCKET ?? "/run/g1-realsense/realsense.sock",
    captureTimeoutMs: integer(env, "G1_CAPTURE_TIMEOUT_MS", 5000),
    persistentCapture: boolean(env, "G1_PERSISTENT_CAPTURE", true),
    // worker (persistent sense_worker.py) | subprocess (g1_u2_sense.py per capture) | fixture (in-process, no robot).
    // Unset: G1_PERSISTENT_CAPTURE decides between the two real cameras, as before.
    sense: env.G1_SENSE === undefined ? null : choice(env, "G1_SENSE", "worker", ["worker", "subprocess", "fixture"]),
    fixtureDepth: parseFixtureDepth(env.G1_FIXTURE_DEPTH ?? "2.5,2.5,2.5"),

    // ---- actuation
    controlDaemon: env.G1_CONTROL_DAEMON ?? DEFAULT_CONTROL_DAEMON,
    persistentControl: boolean(env, "G1_PERSISTENT_CONTROL", false),
    // The legacy alias is parsed as a boolean too: "0"/"false"/"off" must never enable actuation.
    executeEnabled: boolean(env, "G1_EXECUTE_ENABLED", boolean(env, "G1_DIRECT_EXECUTE_ENABLED", false)),
    // mock: an always-ready robot that never moves, for offline evaluation of the gate and the brain.
    actuator: choice(env, "G1_ACTUATOR", "none", ["daemon", "console", "cerebellum", "mock", "none"]),
    consoleBaseUrl: env.G1_CONSOLE_BASE_URL ?? "http://127.0.0.1:8080",
    // agent-vla-async cerebellum (Unix-socket JSON RPC on the robot). The bridge re-shapes tokens into
    // the cerebellum's bounded plans and keeps the intent's heartbeat alive while a pulse runs.
    cerebellumSocket: env.G1_CEREBELLUM_SOCKET ?? "/run/user/1000/g1-vla-async/control.sock",
    cerebellumPolicyWaitS: number(env, "G1_CEREBELLUM_POLICY_WAIT_S", 3, { minimum: 0.1, maximum: 60 }),
    cerebellumWaitForPolicy: boolean(env, "G1_CEREBELLUM_WAIT_FOR_POLICY", false),
    cerebellumHeartbeatS: number(env, "G1_CEREBELLUM_HEARTBEAT_S", 0.5, { minimum: 0.1, maximum: 5 }),
    controlTimeoutMs: integer(env, "G1_CONTROL_TIMEOUT_MS", 8000),
    armTimeoutMs: integer(env, "G1_ARM_TIMEOUT_MS", 30000),
    maxPulseS: number(env, "G1_MAX_PULSE_S", 2.0, { minimum: 0.4, maximum: 3.0 }),

    // ---- fused safety gate
    useLidar: boolean(env, "G1_USE_LIDAR", true),
    // A request may require lidar but may not switch it off unless the deployment allows it
    // (contact work with the lidar producer stopped). Off by default: with lidar off, S/Q/E have no
    // obstacle sensing at all.
    allowRequestLidarOff: boolean(env, "G1_ALLOW_REQUEST_LIDAR_OFF", false),
    // Re-read depth from the camera at execution time when the decision was made on an older frame
    // (pipeline mode, inline escalation). The gate then sees the world as it is, not as the model saw it.
    gateLiveDepth: boolean(env, "G1_GATE_LIVE_DEPTH", true),
    lidarSnapshot: env.G1_LIDAR_SNAPSHOT ?? "/run/user/1000/g1-livox/obstacles.json",
    // snapshot: the Livox producer's JSON file at G1_LIDAR_SNAPSHOT | fixture: G1_FIXTURE_LIDAR sector distances
    obstacles: choice(env, "G1_OBSTACLES", "snapshot", ["snapshot", "fixture"]),
    fixtureLidar: parseFixtureLidar(env.G1_FIXTURE_LIDAR ?? "5,5,5"),
    depthNearCenterM: number(env, "G1_DEPTH_NEAR_CENTER_M", 0.55, { minimum: 0.25, maximum: 3 }),
    depthNearSideM: number(env, "G1_DEPTH_NEAR_SIDE_M", 0.35, { minimum: 0.2, maximum: 3 }),

    // ---- pipeline
    pipelineFreshMaxS: number(env, "G1_PIPELINE_FRESH_MAX_S", 1.0, { minimum: 0.05, maximum: 10 }),
    pipelineConcurrency: integer(env, "G1_PIPELINE_CONCURRENCY", 2, { maximum: 4 }),

    // ---- asynchronous thinker (advisory only, never actuates)
    thinkerEnabled: boolean(env, "G1_THINKER_ENABLED", true),
    thinkerBaseUrl: env.G1_THINKER_BASE_URL ?? "http://127.0.0.1:8092",
    thinkerSubmitTimeoutMs: integer(env, "G1_THINKER_SUBMIT_TIMEOUT_MS", 1500),

    // ---- brain: memory + judge + router around /v1/step (never on the actuator path)
    brainEnabled: boolean(env, "G1_BRAIN_ENABLED", true),
    brainStepBudgetMs: integer(env, "G1_BRAIN_STEP_BUDGET_MS", 10000, { minimum: 1000, maximum: 60000 }),
    memoryPath: env.G1_MEMORY_PATH === "" ? null : (env.G1_MEMORY_PATH ?? join(DEFAULT_ROOT, "runtime", "memory", "experience.jsonl")),
    memoryMaxLessons: integer(env, "G1_MEMORY_MAX_LESSONS", 500, { maximum: 100000 }),
    memoryRecallK: integer(env, "G1_MEMORY_RECALL_K", 3, { minimum: 0, maximum: 10 }),
    memoryPromptMaxChars: integer(env, "G1_MEMORY_PROMPT_MAX_CHARS", 240, { minimum: 0, maximum: 2000 }),
    // Per-session episodic ring (steps kept for signals and the deliberate prompt's recent-steps line).
    episodicCapacity: integer(env, "G1_EPISODIC_CAPACITY", 64, { minimum: 8, maximum: 1024 }),
    // Cross-session episode store (every step of every session, JSONL). Empty string disables it.
    episodesPath: env.G1_EPISODES_PATH === "" ? null : (env.G1_EPISODES_PATH ?? join(DEFAULT_ROOT, "runtime", "memory", "episodes.jsonl")),
    episodesMaxSessions: integer(env, "G1_EPISODES_MAX_SESSIONS", 200, { maximum: 100000 }),
    episodesRecallK: integer(env, "G1_EPISODES_RECALL_K", 2, { minimum: 0, maximum: 10 }),
    // ACRouter shape: cheap chain (the reflex engine) -> gate/judge verify -> escalate (deliberate).
    routerCheapChain: engineList(env, "G1_ROUTER_CHEAP_CHAIN", [engine === "react" ? "token" : engine]),
    routerEscalateTo: choice(env, "G1_ROUTER_ESCALATE_TO", "react", ENGINES),
    routerK: integer(env, "G1_ROUTER_K", 2, { maximum: 10 }),
    routerCooldown: integer(env, "G1_ROUTER_COOLDOWN", 2, { minimum: 0, maximum: 20 }),
    routerMaxDeliberateFraction: number(env, "G1_ROUTER_MAX_DELIBERATE_FRACTION", 0.5, { maximum: 1 }),
    routerInlineEscalate: boolean(env, "G1_ROUTER_INLINE_ESCALATE", true),
    // The router's per-dimension engine scores (ACRouter's routing memory) survive restarts here.
    routerMemoryPath: env.G1_ROUTER_MEMORY_PATH === "" ? null : (env.G1_ROUTER_MEMORY_PATH ?? join(DEFAULT_ROOT, "runtime", "memory", "router.json")),
    // Candidates = (engine harness, model) pairs with a tier; see src/candidates.mjs. Empty = the two-arm default.
    routerCandidates: env.G1_ROUTER_CANDIDATES?.trim() || null,
    // rule: the trigger rules decide; model: an asynchronous router model's suggestion decides when it is
    // fresh; hybrid: safety triggers first, then the suggestion, then the rules (recommended with a model).
    routerMode: choice(env, "G1_ROUTER_MODE", "rule", ["rule", "model", "hybrid"]),
    routerModel: env.G1_ROUTER_MODEL ?? readEnv(env, "REA_MODEL") ?? "",
    routerModelTimeoutMs: integer(env, "G1_ROUTER_MODEL_TIMEOUT_MS", 3000, { maximum: 60000 }),
    routerExplore: number(env, "G1_ROUTER_EXPLORE", 0.05, { maximum: 1 }),
    routerMinTries: integer(env, "G1_ROUTER_MIN_TRIES", 1, { minimum: 0, maximum: 10 }),
    // CLI-backed engines (claude -p / codex exec) for models the gateway only serves to official clients.
    cliEngines: boolean(env, "G1_CLI_ENGINES", false),
    cliClaudeBin: env.G1_CLI_CLAUDE_BIN ?? "claude",
    cliCodexBin: env.G1_CLI_CODEX_BIN ?? "codex",
    cliTimeoutMs: integer(env, "G1_CLI_TIMEOUT_MS", 60000, { maximum: 300000 }),
    cliEffort: env.G1_CLI_EFFORT ?? "low",
    // USD per 1M tokens overrides, JSON {"model": [in, out]}; defaults in src/candidates.mjs.
    modelPrices: env.G1_MODEL_PRICES?.trim() || null,
    // extra chat-completions providers by model prefix (see candidates.mjs BUILTIN_PROVIDERS)
    providers: env.G1_PROVIDERS?.trim() || null,
    providerKeys: { gemini: env.G1_GEMINI_API_KEY ?? env.GEMINI_API_KEY ?? "" },
    captureUsage: boolean(env, "G1_CAPTURE_USAGE", true),
    reactModel,
    reactMaxIterations: integer(env, "G1_REACT_MAX_ITERATIONS", 2, { maximum: 4 }),
    reactTimeoutMs: integer(env, "G1_REACT_TIMEOUT_MS", 8000, { maximum: 60000 }),
    reactMaxOutputTokens: integer(env, "G1_REACT_MAX_OUTPUT_TOKENS", 160, { maximum: 1024 }),
    // Reflection (Reflexion): a text-only lesson over the last few steps when the episode goes wrong.
    // Fire-and-forget and rate-limited; it never touches the control path.
    brainReflect: boolean(env, "G1_BRAIN_REFLECT", true),
    reflectModel: env.G1_REFLECT_MODEL ?? reactModel,
    reflectCooldown: integer(env, "G1_REFLECT_COOLDOWN", 3, { minimum: 0, maximum: 50 }),
    reflectTimeoutMs: integer(env, "G1_REFLECT_TIMEOUT_MS", 6000, { maximum: 60000 }),
    reflectMaxOutputTokens: integer(env, "G1_REFLECT_MAX_OUTPUT_TOKENS", 160, { maximum: 512 }),
  };
  // A fixture camera answers 2.5 m and a fixture lidar 5 m whatever is in front of the robot, so the
  // gate would clear a move it has no evidence for. Harmless in preview or against the mock actuator;
  // refused outright when an actuator that can move the robot is armed.
  if (config.executeEnabled && !["mock", "none"].includes(config.actuator)
    && (senseKindOf(config) === "fixture" || config.obstacles === "fixture")) {
    throw new Error("G1_SENSE/G1_OBSTACLES=fixture feed the gate synthetic evidence; not allowed with G1_EXECUTE_ENABLED=1 and a real actuator (use G1_ACTUATOR=mock)");
  }
  return config;
}

/** Which camera implementation the configuration selects. */
export function senseKindOf(config) {
  return config.sense ?? (config.persistentCapture ? "worker" : "subprocess");
}

/** The per-request decision knobs, pulled out of the service config. */
export function defaultDecisionCfg(config) {
  return {
    engine: config.engine,
    promptVariant: config.promptVariant,
    schema: config.schema,
    maxWidth: config.maxWidth,
    quality: config.quality,
    filter: config.filter,
    includeDepth: config.includeDepth,
    imageDetail: config.imageDetail,
    maxOutputTokens: config.maxOutputTokens,
    primaryModel: config.primaryModel,
    fallbackModel: config.fallbackModel,
    primaryTimeoutMs: config.primaryTimeoutMs,
    fallbackTimeoutMs: config.fallbackTimeoutMs,
    race: config.race,
  };
}
