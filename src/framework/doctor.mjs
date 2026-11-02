import { execFile } from "node:child_process";
import { access, readdir } from "node:fs/promises";
import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { promisify } from "node:util";

import { readEnv } from "../env.mjs";

import { validateConfiguration } from "./configuration.mjs";
import { createRegistry } from "./registry.mjs";

const execute = promisify(execFile);
const PYTHON_PROBE = `import importlib.util, json, sys
request = json.loads(sys.argv[1])
sys.path[:0] = request["paths"]
print(json.dumps({"version": sys.version.split()[0], "modules": {name: importlib.util.find_spec(name) is not None for name in request["modules"]}}))`;

function localPath(value, baseDir) {
  return resolve(baseDir, value.startsWith("~/") ? join(homedir(), value.slice(2)) : value);
}

function placeholder(value) {
  if (typeof value === "string") return value.startsWith("replace-with-");
  if (Array.isArray(value)) return value.some(placeholder);
  return value && typeof value === "object" && Object.values(value).some(placeholder);
}

/** Check local prerequisites only. Never construct an adapter, import an SDK or call a model. */
export async function inspectConfiguration(input, { baseDir = process.cwd(), env = process.env, registry = createRegistry() } = {}) {
  const checks = [];
  const check = (name, status, message) => checks.push({ name, status, message });
  let config;
  try {
    config = validateConfiguration(input, { registry });
    check("configuration", "ok", "Configuration structure is valid.");
  } catch (error) {
    check("configuration", "error", error.message);
    return { ok: false, checks };
  }
  if (placeholder(config)) check("preset values", "error", "Replace the replace-with-* values in the configuration with your task or installed environment.");
  if (typeof config.createEnvironment === "function") {
    check("custom factories", "info", "JavaScript factory dependencies cannot be inspected statically; the factories were not called.");
    return { ok: !checks.some(item => item.status === "error"), checks };
  }
  const spec = config.environment;
  const options = spec.options ?? {};
  const descriptor = registry.listEnvironments().find(item => item.type === spec.type);
  const requires = descriptor?.requires ?? {};
  const sources = [];
  const addSource = (key, value, required = false) => {
    if (value) sources.push({ key, path: localPath(value, baseDir) });
    else if (required) check(key, "error", `Set environment.options.${key} or its documented environment variable.`);
  };
  if (spec.type === "aloha") {
    const real = options.mode === "real";
    addSource(real ? "alohaRoot" : "actRoot", real ? options.alohaRoot ?? env.ALOHA_ROOT : options.actRoot ?? env.ACT_ROOT, true);
  } else if (spec.type === "openral") {
    addSource("openralRoot", options.openralRoot ?? env.OPENRAL_ROOT, !options.robotYaml && !options.halFactory);
  } else if (spec.type === "rpent") addSource("rpentRoot", options.rpentRoot ?? env.RPENT_ROOT);
  else if (spec.type === "arc-agi3") addSource("environmentsDir", options.environmentsDir ?? env.ARC_ENVIRONMENTS_DIR);
  if (options.script) addSource("script", options.script);
  for (const key of ["robotYaml", "simEnvYaml"]) if (options[key]) {
    const root = options.openralRoot ?? env.OPENRAL_ROOT;
    sources.push({ key, path: localPath(options[key], root ? localPath(root, baseDir) : baseDir) });
  }
  if (spec.type === "module") {
    if (typeof options.module !== "string" || !options.module.trim()) check("environment module", "error", "Set environment.options.module to a JavaScript factory module.");
    else if (options.module.startsWith(".") || isAbsolute(options.module)) addSource("environment module", options.module);
  }
  for (const source of sources) {
    try {
      await access(source.path);
      check(source.key, "ok", `${source.path} exists.`);
    } catch {
      check(source.key, "error", `${source.path} does not exist or is not readable.`);
    }
  }
  const modules = [...(requires.pythonModules ?? [])];
  const paths = [];
  if (spec.type === "aloha") {
    modules.push("numpy", options.mode === "real" ? "rospy" : "dm_control");
    const source = sources.find(item => item.key === "actRoot" || item.key === "alohaRoot");
    if (source) paths.push(options.mode === "real" ? join(source.path, "aloha_scripts") : source.path);
    modules.push(options.mode === "real" ? "real_env" : "sim_env");
  }
  if (spec.type === "rpent") {
    modules.push("rpent");
    const source = sources.find(item => item.key === "rpentRoot");
    if (source) paths.push(source.path);
    if (!options.endpoint) check("RPent endpoint", "error", "Set environment.options.endpoint to the running RPent environment service.");
  }
  if (spec.type === "openral") {
    const source = sources.find(item => item.key === "openralRoot");
    if (source) {
      const pythonRoot = join(source.path, "python");
      for (const name of await readdir(pythonRoot).catch(() => [])) paths.push(join(pythonRoot, name, "src"));
    }
  }
  if (requires.pythonEnv || modules.length) {
    const configuredPython = options.python ?? (spec.type === "g1" ? options.config?.python : undefined) ?? env[requires.pythonEnv] ?? (spec.type === "g1" ? "/usr/bin/python3" : "python3");
    const python = configuredPython.includes("/") ? localPath(configuredPython, baseDir) : configuredPython;
    try {
      const { stdout } = await execute(python, ["-c", PYTHON_PROBE, JSON.stringify({ modules: [...new Set(modules)], paths })], { cwd: baseDir, env, timeout: 10000, maxBuffer: 1024 * 1024 });
      const probe = JSON.parse(stdout);
      check("Python", "ok", `${python} (${probe.version}).`);
      for (const [name, present] of Object.entries(probe.modules)) check(`Python module ${name}`, present ? "ok" : "error", present ? "Module is discoverable; SDK initialization was not tested." : `Module is missing from ${python}. Install the adapter's documented requirements in this interpreter.`);
    } catch (error) {
      check("Python", "error", `Could not inspect ${python}: ${error.code ?? error.message}. Set environment.options.python to an installed interpreter, or pass --python to the CLI.`);
    }
  }
  async function inspectPolicy(policy, name = "policy") {
    if (!policy) return;
    if (typeof policy === "string") policy = { type: policy };
    const settings = policy.options ?? {};
    if (["chat", "aloha-code"].includes(policy.type)) {
      const model = settings.model ?? (policy.type === "aloha-code" ? env.REA_CODE_MODEL : undefined) ?? readEnv(env, "REA_MODEL");
      const baseUrl = settings.baseUrl ?? readEnv(env, "REA_BASE_URL");
      check(`${name} model`, model ? "ok" : "error", model ? "Chat model is configured." : "Set policy.options.model or REA_MODEL.");
      check(`${name} base URL`, baseUrl ? "ok" : "error", baseUrl ? "Model API base URL is configured; connectivity was not tested." : "Set policy.options.baseUrl or REA_BASE_URL.");
      if (policy.type === "aloha-code") {
        const python = settings.python ?? env.REA_CODE_PYTHON ?? "python3";
        try {
          const { stdout } = await execute(python, ["-c", PYTHON_PROBE, JSON.stringify({ modules: ["numpy"], paths: [] })], { cwd: baseDir, env, timeout: 10000 });
          const present = JSON.parse(stdout).modules.numpy;
          check(`${name} Python numpy`, present ? "ok" : "error", present ? `NumPy is available in ${python}.` : `Install NumPy in ${python}.`);
        } catch (error) {
          check(`${name} Python`, "error", `Could not inspect ${python}: ${error.code ?? error.message}. Set REA_CODE_PYTHON.`);
        }
      }
    } else if (["lingbot-vla", "pi05"].includes(policy.type)) {
      const pi05 = policy.type === "pi05";
      const label = pi05 ? "pi0.5" : "LingBot";
      const prefix = pi05 ? "PI05" : "LINGBOT_VLA";
      const requirements = pi05 ? "requirements-pi05.txt" : "requirements-lingbot-vla.txt";
      const url = settings.url ?? env[`${prefix}_URL`] ?? (pi05 ? "ws://127.0.0.1:8000" : "ws://127.0.0.1:8006");
      try {
        if (!["ws:", "wss:"].includes(new URL(url).protocol)) throw new Error();
        check(`${name} endpoint`, "ok", `${label} WebSocket endpoint is configured; connectivity was not tested.`);
      } catch {
        check(`${name} endpoint`, "error", `Set policy.options.url or ${prefix}_URL to a ws:// or wss:// endpoint.`);
      }
      const python = settings.python ?? env[`${prefix}_PYTHON`] ?? "python3";
      try {
        const { stdout } = await execute(python, ["-c", PYTHON_PROBE, JSON.stringify({ modules: ["numpy", "PIL", "msgpack", "websockets"], paths: [] })], { cwd: baseDir, env, timeout: 10000, maxBuffer: 1024 * 1024 });
        const probe = JSON.parse(stdout);
        for (const [module, present] of Object.entries(probe.modules)) {
          check(`${name} Python ${module}`, present ? "ok" : "error", present ? `Module is discoverable in ${python}.` : `Install ${requirements} in ${python}.`);
        }
      } catch (error) {
        check(`${name} Python`, "error", `Could not inspect ${python}: ${error.code ?? error.message}. Set ${prefix}_PYTHON to the client interpreter.`);
      }
      check(`${name} mapping`, "info", "Use camera, state and action mappings for the server checkpoint's robot configuration. Model weights and robot connectivity were not tested.");
    } else if (policy.type === "routed") {
      if (!settings.reflex || !settings.deliberate) check(name, "error", `Set ${name}.options.reflex and ${name}.options.deliberate to policy specifications.`);
      await inspectPolicy(settings.reflex, `${name}.reflex`);
      await inspectPolicy(settings.deliberate, `${name}.deliberate`);
    } else if (policy.type === "progress") {
      if (!settings.policy) check(name, "error", `Set ${name}.options.policy to the wrapped policy specification.`);
      await inspectPolicy(settings.policy, `${name}.policy`);
    } else if (policy.type === "module") {
      if (typeof settings.module !== "string" || !settings.module.trim()) check(name, "error", `Set ${name}.options.module to a JavaScript factory module.`);
      else if (settings.module.startsWith(".") || isAbsolute(settings.module)) {
        const path = localPath(settings.module, baseDir);
        try {
          await access(path);
          check(name, "ok", `${path} exists; the module was not loaded.`);
        } catch {
          check(name, "error", `${path} does not exist or is not readable.`);
        }
      } else check(name, "info", "Custom policy module dependencies were not loaded.");
    }
  }
  if (config.systems) {
    for (const candidate of config.systems.system0) await inspectPolicy(candidate.policy, `System0 ${candidate.id}`);
    const { system1, system2, mode } = config.systems;
    if (system1?.type === "route-memory") {
      const table = system1.options?.table;
      if (typeof table === "string") {
        const path = localPath(table, baseDir);
        try {
          await access(path);
          check("System1 route table", "ok", `${path} exists; it is verified when the episode starts.`);
        } catch {
          check("System1 route table", "error", `${path} does not exist or is not readable. Build it with tools/route-table.mjs.`);
        }
      } else if (!table) check("System1 route table", "info", "No route table: capabilities run in configuration order.");
      if ((system1.options?.recoverOn ?? "judge") === "judge" && !system2) check("System1 recoverOn", "error", "recoverOn judge needs a System 2 judge; set systems.system2 or choose recoverOn exception, always or never.");
    } else if (system1?.type) await inspectPolicy(system1, "System1 router");
    if (system2?.type === "judge") {
      const model = system2.options?.model ?? env.REA_JUDGE_MODEL ?? readEnv(env, "REA_MODEL");
      check("System2 judge model", model ? "ok" : "error", model ? "Judge model is configured; connectivity was not tested." : "Set systems.system2.options.model, REA_JUDGE_MODEL or REA_MODEL.");
      if (mode !== "routed") check("System2 judge", "error", "The judge role is used in routed mode; set systems.mode to routed.");
    } else if (system2?.type) await inspectPolicy(system2, "System2 brain");
    check("runtime", "info", mode === "routed"
      ? "Routed System0/1/2 episode: route memory picks the first capability, the judge runs only at capability boundaries, and success is read once after the episode."
      : `System0/1/2 ${mode} orchestration; thinking overlaps execution in async mode.`);
  } else await inspectPolicy(config.policy);
  if (requires.runtime) check("runtime", "info", `${requires.runtime}. Availability must be verified in the target runtime.`);
  for (const episode of config.episodes) {
    if (!episode.environment && !episode.policy) continue;
    const merge = (base, override) => {
      if (!override) return base;
      if (typeof override === "string") return { type: override, options: {} };
      return { ...base, ...override, options: override.type && override.type !== base.type ? { ...override.options } : { ...base.options, ...override.options } };
    };
    const report = await inspectConfiguration({
      ...config,
      environment: merge(config.environment, episode.environment),
      policy: merge(config.policy, episode.policy),
      episodes: [{ ...episode, environment: undefined, policy: undefined }],
    }, { baseDir, env, registry });
    checks.push(...report.checks.map(item => ({ ...item, name: `${episode.id}: ${item.name}` })));
  }
  return { ok: !checks.some(item => item.status === "error"), checks };
}
