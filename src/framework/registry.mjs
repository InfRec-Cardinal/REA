import { isAbsolute, resolve } from "node:path";
import { homedir } from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";

import { createAlohaEnvironment } from "../environments/aloha.mjs";
import { createArcAgi3Environment } from "../environments/arc-agi3.mjs";
import { createG1Environment } from "../environments/g1.mjs";
import { createGymnasiumEnvironment } from "../environments/gymnasium.mjs";
import { createLeRobotEnvironment } from "../environments/lerobot.mjs";
import { createLiberoEnvironment } from "../environments/libero.mjs";
import { createOpenRALEnvironment } from "../environments/openral.mjs";
import { createRPentEnvironment } from "../environments/rpent.mjs";
import { createRobotEnvironment } from "../environments/robot.mjs";
import { createPythonDriverEnvironment } from "../environments/python-driver.mjs";
import { createFixtureCamera, createSubprocessCamera, createWorkerCamera } from "../camera.mjs";
import { createCerebellumActuator, createCerebellumClient } from "../cerebellum.mjs";
import { loadConfig, senseKindOf } from "../config.mjs";
import { readEnv } from "../env.mjs";
import { createConsoleActuator, createDaemonActuator, createExecutor, createMockActuator, createNullActuator } from "../executor.mjs";
import { LiberoClient } from "../libero-client.mjs";
import { createPi05Policy } from "../policies/pi05.mjs";
import { createAlohaCodePolicy } from "../policies/aloha-code.mjs";
import { createLingBotVlaPolicy } from "../policies/lingbot-vla.mjs";
import { resolveProfile } from "../profiles/index.mjs";
import { createFixtureObstacleSource, createSnapshotObstacleSource } from "../sensors/obstacles.mjs";
import { LineWorker } from "../workers.mjs";
import { createChatPolicy } from "./policy.mjs";
import { createRoutedPolicy } from "./routing.mjs";
import { createProgressPolicy } from "./progress.mjs";

const CLICK = { name: "click", description: "Select a cell in the demo grid", parameters: { type: "object", properties: { x: { type: "integer", minimum: 0, maximum: 3 }, y: { type: "integer", minimum: 0, maximum: 3 } }, required: ["x", "y"], additionalProperties: false } };
const MOVE = { name: "move_joints", description: "Set the demo's two joint positions", parameters: { type: "object", properties: { positions: { type: "array", items: { type: "number", minimum: -1, maximum: 1 }, minItems: 2, maxItems: 2 } }, required: ["positions"], additionalProperties: false } };

function demoEnvironment({ kind: defaultKind = "grid" } = {}) {
  let kind = defaultKind;
  let done = false;
  let episode = 0;
  let sequence = 0;
  const observation = () => ({
    id: `demo:${episode}:${++sequence}`, timestampMs: Date.now(),
    text: "Synthetic interface demo; the target is visible. This is not a benchmark score.",
    state: { target: kind === "grid" ? [2, 1] : [0.25, -0.5], reached: done },
    actions: [kind === "grid" ? CLICK : MOVE],
  });
  return {
    id: "demo", timing: "turn-based",
    async reset({ task } = {}) {
      kind = typeof task === "string" ? task : task?.kind ?? defaultKind;
      if (!["grid", "joints"].includes(kind)) throw new Error("demo task must be grid or joints");
      done = false; episode += 1; sequence = 0;
      return { observation: observation(), success: false, info: { synthetic: true } };
    },
    async observe() { return observation(); },
    async step(action) {
      done = kind === "grid"
        ? action.name === "click" && action.arguments.x === 2 && action.arguments.y === 1
        : action.name === "move_joints" && action.arguments.positions?.length === 2 && action.arguments.positions[0] === 0.25 && action.arguments.positions[1] === -0.5;
      return { observation: observation(), terminated: done, success: done, reward: done ? 1 : 0, info: { synthetic: true, actionsAccepted: 1 } };
    },
  };
}

function demoPolicy() {
  return {
    id: "demo-visible-target",
    async decide({ observation, actions = observation.actions }) {
      const target = observation.state.target;
      const name = actions[0].name;
      if (!Array.isArray(target) || !["click", "move_joints"].includes(name)) throw new Error("demo policy only supports the synthetic demo environment; choose chat or a custom policy");
      return { action: { name, arguments: name === "click" ? { x: target[0], y: target[1] } : { positions: [...target] } } };
    },
  };
}

function scriptedPolicy({ actions, repeat = false } = {}) {
  if (!Array.isArray(actions) || actions.length === 0) throw new Error("scripted policy requires a nonempty actions array");
  let index = 0;
  return {
    id: "scripted",
    async reset() { index = 0; },
    async decide() {
      if (index >= actions.length && !repeat) throw new Error("scripted policy exhausted its actions");
      return { action: structuredClone(actions[index++ % actions.length]) };
    },
  };
}

/** Construction stays inert: SDK processes and device connections begin on first use. */
function lazyEnvironment(factory, description) {
  let pending;
  let instance;
  let closed = false;
  async function get() {
    if (closed) throw new Error("Environment is closed");
    pending ??= Promise.resolve().then(factory).then(value => {
      if (!value || typeof value.reset !== "function" || typeof value.step !== "function") throw new TypeError("Environment factory must return reset() and step()");
      instance = value;
      return value;
    });
    return pending;
  }
  const environment = {
    get id() { return instance?.id ?? description.type; },
    get timing() { return instance?.timing ?? description.timing; },
    get actions() { return instance?.actions; },
    // Routed episodes read these after the first reset: whether a capability may follow a terminated one, and the
    // environment's once-per-episode success read.
    get resumable() { return instance?.resumable ?? description.resumable ?? false; },
    get score() { return instance?.score ? instance.score.bind(instance) : undefined; },
    get scorecard() {
      if (instance?.scorecard) return instance.scorecard.bind(instance);
      if (description.scorecard) return async (...args) => pending ? (await pending).scorecard(...args) : null;
      return undefined;
    },
    async reset(...args) { return (await get()).reset(...args); },
    async observe(...args) {
      const target = await get();
      if (!target.observe) throw new Error(`${environment.id} does not provide observe()`);
      return target.observe(...args);
    },
    async step(...args) {
      if (closed) throw new Error("Environment is closed");
      // A reset environment can dispatch immediately, before background work starts.
      const target = instance ?? await get();
      return target.step(...args);
    },
    async stop(...args) { return instance?.stop?.(...args) ?? { supported: false }; },
    async close() {
      if (closed) return;
      closed = true;
      if (pending) await (await pending.catch(() => null))?.close?.();
    },
  };
  return environment;
}

async function g1Environment(options, context) {
  const env = context.env ?? process.env;
  const config = resolvePaths({ ...loadConfig(env), ...options.config, ...defined({ python: options.python }) }, context);
  if (!options.executor && !["daemon", "console", "cerebellum", "mock", "none"].includes(config.actuator)) throw new Error(`Unsupported G1 actuator: ${config.actuator}`);
  const profile = options.profile ?? await resolveProfile(config.robotProfile, { baseDir: context.baseDir ?? process.cwd() });
  const workers = [];
  let camera = options.camera;
  const sense = senseKindOf(config);
  if (!camera && sense === "fixture") {
    camera = createFixtureCamera({ frames: config.fixtureDepth });
  } else if (!camera && sense === "worker") {
    const worker = new LineWorker("g1-sense", config.python, [config.senseWorker], { env: { ...env, G1_REALSENSE_SOCKET: config.cameraSocket }, autoRestart: false });
    workers.push(worker);
    camera = createWorkerCamera({ worker, timeoutMs: config.captureTimeoutMs });
  }
  camera ??= createSubprocessCamera({ python: config.python, script: config.senseScript, socketPath: config.cameraSocket, timeoutMs: config.captureTimeoutMs, maxWidth: config.maxWidth, jpegQuality: config.quality, env });
  let executor = options.executor;
  if (!executor) {
    let actuator;
    if (config.actuator === "daemon") {
      const worker = new LineWorker("g1-control", config.python, [config.controlDaemon], { env, autoRestart: false });
      workers.push(worker);
      actuator = createDaemonActuator({ worker, config });
    } else if (config.actuator === "console") actuator = createConsoleActuator({ baseUrl: config.consoleBaseUrl, config, ...(context.fetch ? { fetchFn: context.fetch } : {}) });
    else if (config.actuator === "cerebellum") actuator = createCerebellumActuator({ client: createCerebellumClient({ socketPath: config.cerebellumSocket, timeoutMs: config.controlTimeoutMs }), config, logger: context.logger });
    else if (config.actuator === "mock") actuator = createMockActuator();
    else if (config.actuator === "none") actuator = createNullActuator();
    else throw new Error(`Unsupported G1 actuator: ${config.actuator}`);
    const obstacles = config.obstacles === "fixture"
      ? createFixtureObstacleSource(config.fixtureLidar, { profile })
      : createSnapshotObstacleSource({ path: config.lidarSnapshot, profile });
    executor = createExecutor({ actuator, config, logger: context.logger, profile, obstacles });
  }
  const environment = createG1Environment({ ...options, camera, executor, config, profile });
  const close = environment.close;
  environment.close = async () => {
    try { await close(); } finally { for (const worker of workers) worker.stop(); }
  };
  return environment;
}

async function moduleFactory(kind, { module, export: namedExport, exportName = namedExport, ...options }, context) {
  if (!module) throw new Error(`${kind} module requires options.module`);
  const specifier = isAbsolute(module) || module.startsWith(".")
    ? pathToFileURL(resolve(context.baseDir ?? process.cwd(), module)).href : module;
  const loaded = await import(specifier);
  const factory = exportName ? loaded[exportName] : loaded.default ?? loaded[kind === "environment" ? "createEnvironment" : "createPolicy"];
  if (typeof factory !== "function") throw new Error(`${module} must export a ${kind} factory function`);
  return factory(options, context);
}

const description = (type, text, timing, requires = {}) => ({ type, description: text, ...(timing ? { timing } : {}), requires });
const builtinDescriptions = [
  description("demo", "Offline synthetic grid and two-joint interface demo", "turn-based"),
  description("g1", "Unitree G1 camera and bounded motion through daemon, console or cerebellum", "realtime", { pythonEnv: "G1_SENSE_PYTHON", runtime: "G1 camera gateway and configured actuator" }),
  description("libero", "LIBERO manipulation benchmark", "turn-based", { pythonEnv: "LIBERO_PYTHON", pythonModules: ["libero", "robosuite"] }),
  { ...description("arc-agi3", "ARC-AGI-3 official interactive environment", "turn-based", { pythonEnv: "ARC_PYTHON", pythonModules: ["arc_agi"], sourceEnv: "ARC_ENVIRONMENTS_DIR" }), scorecard: true },
  description("gymnasium", "Gymnasium environments with dynamic action spaces", "turn-based", { pythonEnv: "GYM_PYTHON", pythonModules: ["gymnasium", "numpy"] }),
  description("aloha", "ALOHA ACT simulation or classic real robot interface", "turn-based", { pythonEnv: "ALOHA_PYTHON", sourceEnv: "ACT_ROOT", runtime: "ACT for simulation; ALOHA_ROOT and ROS for real hardware" }),
  description("lerobot", "Feature-derived LeRobot robot drivers", "realtime", { pythonEnv: "LEROBOT_PYTHON", pythonModules: ["lerobot"], runtime: "Configured robot ports and calibration" }),
  description("openral", "OpenRAL robot manifests, HALs and scenes", "realtime", { pythonEnv: "OPENRAL_PYTHON", pythonModules: ["openral_hal"], sourceEnv: "OPENRAL_ROOT" }),
  description("rpent", "RPent LIBERO, RoboCasa and RoboTwin environment services", "turn-based", { pythonEnv: "RPENT_PYTHON", pythonModules: ["numpy"], sourceEnv: "RPENT_ROOT", runtime: "Running RPent environment service" }),
  description("robot", "A JavaScript driver implementing the common robot contract", "realtime", { runtime: "options.driver or context.driver" }),
  description("python-robot", "Your Python driver class (package.module:Class) behind the common robot contract", "realtime", { pythonEnv: "ROBOT_PYTHON", runtime: "options.driver importable by that Python" }),
  description("module", "An environment factory exported by a JavaScript module"),
  description("bridge", "An environment driven by an external evaluator that calls REA as a policy (rea serve)", "turn-based", { runtime: "rea serve and an evaluator-side client" }),
];

export const BRIDGE_DESCRIPTION = Object.freeze({ description: "An environment driven by an external evaluator that calls REA as a policy (rea serve)", timing: "turn-based", requires: { runtime: "rea serve and an evaluator-side client" } });

function defined(values) { return Object.fromEntries(Object.entries(values).filter(([, value]) => value !== undefined)); }

function resolvePaths(options, { baseDir = process.cwd() } = {}) {
  const result = { ...options };
  const local = value => resolve(baseDir, value.startsWith("~/") ? resolve(homedir(), value.slice(2)) : value);
  for (const key of ["actRoot", "alohaRoot", "openralRoot", "rpentRoot", "environmentsDir", "recordingsDir", "script", "senseScript", "senseWorker", "controlDaemon", "cameraSocket", "lidarSnapshot", "cerebellumSocket"]) {
    if (typeof result[key] === "string" && result[key]) result[key] = local(result[key]);
  }
  if (typeof result.python === "string" && result.python.includes("/")) result.python = local(result.python);
  for (const key of ["robotYaml", "simEnvYaml"]) {
    if (typeof result[key] === "string" && result[key]) {
      const value = result[key].startsWith("~/") ? resolve(homedir(), result[key].slice(2)) : result[key];
      result[key] = resolve(result.openralRoot ?? baseDir, value);
    }
  }
  return result;
}

function adapterOptions(options, context, defaults = {}) {
  return resolvePaths({ env: context.env ?? process.env, ...defined(defaults), ...options }, context);
}

/** A registry is local to its caller, so extensions never replace another agent's adapters. */
export function createRegistry({ builtins = true } = {}) {
  const environments = new Map();
  const policies = new Map();
  function register(table, name, factory, metadata) {
    if (typeof name !== "string" || !name || typeof factory !== "function") throw new TypeError("Registration requires a name and factory function");
    table.set(name, { factory, metadata: { ...metadata, type: name } });
    return registry;
  }
  function entry(table, spec, kind) {
    const normalized = typeof spec === "string" ? { type: spec } : spec;
    const found = table.get(normalized?.type);
    if (!found) throw new Error(`Unknown ${kind} type: ${normalized?.type ?? "(missing)"}. Available: ${[...table.keys()].join(", ")}`);
    return { ...found, options: normalized.options ?? {} };
  }
  const registry = {
    registerEnvironment(name, factory, metadata = {}) { return register(environments, name, factory, metadata); },
    registerPolicy(name, factory, metadata = {}) { return register(policies, name, factory, metadata); },
    listEnvironments() { return [...environments.values()].map(item => structuredClone(item.metadata)); },
    listPolicies() { return [...policies.values()].map(item => structuredClone(item.metadata)); },
    async createEnvironment(spec, context = {}) {
      const { factory, metadata, options } = entry(environments, spec, "environment");
      return lazyEnvironment(() => factory(options, { ...context, registry }), metadata);
    },
    async createPolicy(spec, context = {}) {
      const { factory, options } = entry(policies, spec, "policy");
      const policy = await factory(options, { ...context, registry });
      if (typeof policy?.decide !== "function") throw new TypeError("Policy factory must return decide()");
      return policy;
    },
  };
  if (builtins) {
    const factories = {
      demo: demoEnvironment,
      g1: g1Environment,
      libero: (options, context) => {
        const env = context.env ?? process.env;
        return createLiberoEnvironment({ client: options.client ?? new LiberoClient(adapterOptions(options, context, { python: env.LIBERO_PYTHON ?? "python3", script: fileURLToPath(new URL("../../scripts/libero_backend.py", import.meta.url)), env })) });
      },
      "arc-agi3": (options, context) => {
        const env = context.env ?? process.env;
        return createArcAgi3Environment(adapterOptions(options, context, { python: env.ARC_PYTHON, environmentsDir: env.ARC_ENVIRONMENTS_DIR, apiKey: env.ARC_API_KEY }));
      },
      gymnasium: (options, context) => createGymnasiumEnvironment(adapterOptions(options, context, { python: (context.env ?? process.env).GYM_PYTHON })),
      aloha: (options, context) => {
        const env = context.env ?? process.env;
        return createAlohaEnvironment(adapterOptions(options, context, { python: env.ALOHA_PYTHON, actRoot: env.ACT_ROOT, alohaRoot: env.ALOHA_ROOT }));
      },
      lerobot: (options, context) => createLeRobotEnvironment(adapterOptions(options, context, { python: (context.env ?? process.env).LEROBOT_PYTHON })),
      openral: (options, context) => {
        const env = context.env ?? process.env;
        return createOpenRALEnvironment(adapterOptions(options, context, { python: env.OPENRAL_PYTHON, openralRoot: env.OPENRAL_ROOT }));
      },
      rpent: (options, context) => {
        const env = context.env ?? process.env;
        return createRPentEnvironment(adapterOptions(options, context, { python: env.RPENT_PYTHON, rpentRoot: env.RPENT_ROOT }));
      },
      robot: (options, context) => {
        const driver = options.driver ?? context.driver;
        if (!driver) throw new Error("robot requires a JavaScript driver in options.driver/context.driver; JSON configs can use type: module to construct one");
        return createRobotEnvironment({ ...options, driver });
      },
      "python-robot": (options, context) => createPythonDriverEnvironment(adapterOptions(options, context, { python: (context.env ?? process.env).ROBOT_PYTHON })),
      module: (options, context) => moduleFactory("environment", options, context),
      bridge: () => { throw new Error("The bridge environment is driven by an external evaluator; serve this configuration with rea serve <config>."); },
    };
    for (const metadata of builtinDescriptions) registry.registerEnvironment(metadata.type, factories[metadata.type], metadata);
    registry.registerPolicy("demo", demoPolicy, { description: "Solve the synthetic demo's visible target" });
    registry.registerPolicy("scripted", scriptedPolicy, { description: "Replay an explicit action sequence" });
    registry.registerPolicy("chat", (options, { env = process.env }) => createChatPolicy({ ...defined({ model: readEnv(env, "REA_MODEL"), baseUrl: readEnv(env, "REA_BASE_URL"), apiKey: readEnv(env, "REA_API_KEY"), mode: readEnv(env, "REA_POLICY_MODE") }), ...options, progress: options.progress ?? true }), { description: "OpenAI-compatible vision policy using tools, JSON or action tokens" });
    registry.registerPolicy("routed", async ({ reflex, deliberate, ...options }, context) => {
      const childSpec = spec => options.progress === false && spec && typeof spec === "object"
        ? { ...spec, options: { ...spec.options, progress: false } }
        : spec;
      const reflexPolicy = await registry.createPolicy(childSpec(reflex), context);
      let deliberatePolicy;
      try { deliberatePolicy = await registry.createPolicy(childSpec(deliberate), context); }
      catch (error) { await reflexPolicy.close?.(); throw error; }
      return {
        ...createRoutedPolicy({ ...options, reflex: reflexPolicy, deliberate: deliberatePolicy, progress: options.progress ?? true }),
        async close() { await Promise.all([reflexPolicy.close?.(), deliberatePolicy.close?.()]); },
      };
    }, { description: "Route between reflex and deliberate policies" });
    registry.registerPolicy("progress", async ({ policy, ...options }, context) => {
      const nested = await registry.createPolicy(policy, context);
      try {
        return createProgressPolicy({ policy: nested, ...options });
      } catch (error) {
        await nested.close?.();
        throw error;
      }
    }, { description: "Review progress and execution evidence around another policy" });
    registry.registerPolicy("module", (options, context) => moduleFactory("policy", options, context), { description: "A policy factory exported by a JavaScript module" });
    registry.registerPolicy("lingbot-vla", (options, { env = process.env }) => createLingBotVlaPolicy({ ...options, env }), { description: "LingBot-VLA 2.0 WebSocket policy with configurable observation and action chunks" });
    registry.registerPolicy("pi05", (options, { env = process.env }) => createPi05Policy({ ...options, env }), { description: "OpenPI pi0.5 WebSocket policy for Classic ALOHA joint trajectories" });
    registry.registerPolicy("aloha-code", (options, { env = process.env }) => createAlohaCodePolicy({ ...options, env }), { description: "Generate Python to calculate Classic ALOHA absolute joint actions" });
  }
  return registry;
}

const defaultRegistry = createRegistry();
export const createEnvironment = (spec, context) => defaultRegistry.createEnvironment(spec, context);
export const createPolicy = (spec, context) => defaultRegistry.createPolicy(spec, context);
export const listEnvironments = () => defaultRegistry.listEnvironments();
export const listPolicies = () => defaultRegistry.listPolicies();
