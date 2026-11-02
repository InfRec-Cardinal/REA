#!/usr/bin/env node
import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, extname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

import { deprecatedEnvNotices, loadDotEnv, readEnv } from "../src/env.mjs";

export function loadCliEnvironment({ stderr = process.stderr } = {}) {
  const explicit = readEnv(process.env, "REA_ENV_FILE") || process.env.G1_ENV_FILE;
  let loaded;
  if (explicit) loaded = loadDotEnv(explicit);
  else {
    const workspaceFile = resolve(process.cwd(), ".env");
    loaded = loadDotEnv(existsSync(workspaceFile) ? workspaceFile : new URL("../.env", import.meta.url));
  }
  for (const line of deprecatedEnvNotices(process.env)) stderr.write(`${line}\n`);
  return loaded;
}

const SYSTEM_ROLES = [
  ["route-memory", "System 1: development route table, boundary judge gate and escalation ladder (routed mode)"],
  ["judge", "System 2: vision-language completion judge at capability boundaries (routed mode)"],
  ["chat", "System 2: chat model that recommends a controller in the background (async mode)"],
  ["module", "System 1 or 2 from your own JavaScript factory"],
];

export const HELP = `Routed Embodied Agent (REA) — robots and interactive benchmarks

Usage:
  rea list
  rea init <preset> [--output rea.config.json]
  rea run <preset|config.json|config.mjs> [options]
  rea doctor <preset|config.json> [--python executable]
  rea serve <config.json|config.mjs> [--output directory]

Inside the repository, use npm run rea -- <command> with the same arguments.

Run options:
  --output <directory>     Save manifest, episode traces and summary
  --task <task|JSON>       Override each task ID, or replace it with a JSON object
  --seed <integer>         Override the seed for every episode
  --steps <integer>        Override the maximum policy steps per episode
  --instruction <text>    Override the instruction for every episode
  --model <name>           Override the chat model
  --base-url <url>         Override the model API base URL
  --python <executable>    Override the environment's Python interpreter
  --events                Print live System0/1/2 events for systems configurations
  -h, --help              Show this help

serve runs the configured policy or systems for an external evaluator that owns the control loop
(RoboDojo / XPolicyLab, other benchmark harnesses, robot stacks): JSON lines on stdin and stdout,
see docs/extending/policy-server.md.

Examples:
  rea run demo --output outputs/demo
  rea init aloha
  rea doctor rea.config.json
  rea run rea.config.json --steps 10

Model defaults: REA_MODEL, REA_BASE_URL, REA_API_KEY (the pre-release FAST_* names are still read, with a deprecation notice).
doctor checks configuration and local dependencies without connecting to robots or models.
`;

const OPTIONS = {
  output: { type: "string" },
  task: { type: "string" },
  seed: { type: "string" },
  steps: { type: "string" },
  instruction: { type: "string" },
  model: { type: "string" },
  "base-url": { type: "string" },
  python: { type: "string" },
  events: { type: "boolean" },
  help: { type: "boolean", short: "h" },
};
const COMMAND_OPTIONS = {
  list: [],
  init: ["output"],
  run: Object.keys(OPTIONS),
  doctor: ["python"],
  serve: ["output"],
};

export function parseCommand(argv) {
  const { values, positionals } = parseArgs({ args: argv, options: OPTIONS, allowPositionals: true });
  if (values.help || positionals.length === 0) return { command: "help", values };
  const [command, input, ...extra] = positionals;
  if (!Object.hasOwn(COMMAND_OPTIONS, command)) throw new Error(`Unknown command '${command}'. Run rea --help for usage.`);
  if (extra.length || (command === "list" && input)) throw new Error(`Unexpected argument for rea ${command}. Run rea --help for usage.`);
  if (command !== "list" && !input) throw new Error(`rea ${command} requires a ${command === "init" ? "preset" : "preset or configuration file"}.`);
  for (const option of Object.keys(values)) {
    if (!COMMAND_OPTIONS[command].includes(option)) throw new Error(`--${option} is not an option for rea ${command}.`);
  }
  for (const [name, minimum] of [["seed", 0], ["steps", 1]]) {
    if (values[name] === undefined) continue;
    const number = Number(values[name]);
    if (values[name].trim() === "" || !Number.isSafeInteger(number) || number < minimum) throw new Error(`--${name} must be an integer ${minimum === 0 ? "at least 0" : "at least 1"}.`);
    values[name] = number;
  }
  return { command, input, values };
}

export function applyOverrides(config, values) {
  const result = { ...config };
  const episodeOverrides = Object.fromEntries(["seed", "instruction"].filter(key => values[key] !== undefined).map(key => [key, values[key]]));
  if (Object.keys(episodeOverrides).length) {
    if (config.episodes) result.episodes = config.episodes.map(episode => ({ ...episode, ...episodeOverrides }));
    else Object.assign(result, episodeOverrides);
  }
  if (values.task !== undefined) {
    const task = values.task.trim().startsWith("{") ? JSON.parse(values.task) : values.task;
    const replaceTask = previous => {
      if (typeof task !== "string" || !previous || typeof previous !== "object") return typeof previous === "number" && /^\d+$/u.test(task) ? Number(task) : task;
      const key = ["task", "taskName", "gameId", "environmentId", "kind"].find(name => Object.hasOwn(previous, name));
      if (!key) throw new Error("This task has custom fields. Use --task with a JSON object to replace it.");
      return { ...previous, [key]: key === "task" && /^\d+$/u.test(task) ? Number(task) : task };
    };
    if (result.episodes) result.episodes = result.episodes.map(episode => ({ ...episode, task: replaceTask(episode.task) }));
    else result.task = replaceTask(config.task);
  }
  if (values.steps !== undefined) {
    result.budget = { ...config.budget, maxSteps: values.steps };
    if (result.episodes) result.episodes = result.episodes.map(episode => ({ ...episode, budget: { ...episode.budget, maxSteps: values.steps } }));
  }
  if (values.python !== undefined) {
    if (!config.environment || typeof config.createEnvironment === "function") throw new Error("--python requires a declarative environment configuration. Update your JavaScript environment factory directly.");
    result.environment = { ...config.environment, options: { ...config.environment.options, python: values.python } };
    if (result.episodes) result.episodes = result.episodes.map(episode => {
      if (!episode.environment) return episode;
      const environment = typeof episode.environment === "string" ? { type: episode.environment } : episode.environment;
      return { ...episode, environment: { ...environment, options: { ...environment.options, python: values.python } } };
    });
  }
  if (values.model !== undefined || values["base-url"] !== undefined) {
    if (config.policy?.type !== "chat" || typeof config.createPolicy === "function") throw new Error("--model and --base-url require a chat policy configuration. Update your custom or routed policy in the configuration.");
    result.policy = { ...config.policy, options: {
      ...config.policy?.options,
      ...(values.model !== undefined ? { model: values.model } : {}),
      ...(values["base-url"] !== undefined ? { baseUrl: values["base-url"] } : {}),
    } };
    if (result.episodes) result.episodes = result.episodes.map(episode => {
      if (!episode.policy) return episode;
      const policy = typeof episode.policy === "string" ? { type: episode.policy } : episode.policy;
      if (policy.type && policy.type !== "chat") throw new Error("--model and --base-url require chat policies for all overridden episodes.");
      return { ...episode, policy: { ...policy, options: {
        ...policy.options,
        ...(values.model !== undefined ? { model: values.model } : {}),
        ...(values["base-url"] !== undefined ? { baseUrl: values["base-url"] } : {}),
      } } };
    });
  }
  return result;
}

// Importing this module never loads .env, starts a backend or runs a command.
export async function main(argv = process.argv.slice(2), {
  cwd = process.cwd(), env = process.env, stdin = process.stdin, stdout = process.stdout, stderr = process.stderr,
  framework, signals = process,
} = {}) {
  try {
    const { command, input, values } = parseCommand(argv);
    if (command === "help") {
      stdout.write(HELP);
      return 0;
    }
    const api = framework ?? await import("../src/framework/index.mjs");
    if (command === "list") {
      stdout.write("Presets (rea init <preset> or rea run <preset>):\n");
      for (const preset of api.listPresets()) stdout.write(`  ${preset.id.padEnd(14)} ${preset.description}\n`);
      stdout.write("\nEnvironment adapters:\n");
      for (const environment of api.listEnvironments()) stdout.write(`  ${environment.type.padEnd(14)} ${environment.description}\n`);
      stdout.write("\nPolicies (System 0, or policy.type):\n");
      for (const policy of api.listPolicies?.() ?? []) stdout.write(`  ${policy.type.padEnd(14)} ${policy.description ?? ""}\n`);
      stdout.write("\nSystems (systems.mode async | sync | routed):\n");
      for (const [type, text] of SYSTEM_ROLES) stdout.write(`  ${type.padEnd(14)} ${text}\n`);
      return 0;
    }
    if (command === "init") {
      const config = api.getPreset(input);
      const filename = resolve(cwd, values.output ?? "rea.config.json");
      await mkdir(dirname(filename), { recursive: true });
      try {
        await writeFile(filename, `${JSON.stringify(config, null, 2)}\n`, { flag: "wx" });
      } catch (error) {
        if (error.code === "EEXIST") throw new Error(`${filename} already exists. Choose another --output path.`);
        throw error;
      }
      const invocation = env.npm_lifecycle_event === "rea" ? "npm run rea --" : "rea";
      const quotedFilename = `'${filename.replaceAll("'", "'\\''")}'`;
      stdout.write(`Created ${filename}\nNext: ${invocation} doctor ${quotedFilename}\n`);
      return 0;
    }
    if (command === "doctor" && [".mjs", ".js", ".cjs"].includes(extname(input))) throw new Error("doctor cannot statically inspect JavaScript configurations. Use a JSON recipe; this file was not loaded.");
    const loaded = await api.loadConfiguration(input, { cwd, env });
    const config = applyOverrides(loaded.config, values);
    if (command === "serve") {
      // stdout carries only protocol lines; anything a policy prints goes to stderr.
      const log = console.log;
      console.log = (...args) => console.error(...args);
      const stop = () => stdin.destroy?.();
      signals.once("SIGTERM", stop);
      try {
        await api.servePolicy({ config, baseDir: loaded.baseDir, env, input: stdin, output: stdout,
          outputDir: values.output !== undefined ? resolve(cwd, values.output) : null });
        return 0;
      } finally {
        signals.removeListener("SIGTERM", stop);
        console.log = log;
      }
    }
    if (command === "doctor") {
      const report = await api.inspectConfiguration(config, { baseDir: loaded.baseDir, env });
      for (const check of report.checks) stdout.write(`[${check.status}] ${check.name}: ${check.message}\n`);
      stdout.write("Local configuration check only; no robot, simulator, or model connection was opened.\n");
      return report.ok ? 0 : 1;
    }
    const agent = await api.createEmbodiedAgent(config, { baseDir: loaded.baseDir, env });
    const outputDir = values.output !== undefined ? resolve(cwd, values.output)
      : config.outputDir !== undefined ? resolve(loaded.baseDir, config.outputDir)
        : resolve(cwd, `outputs/run-${new Date().toISOString().replaceAll(":", "-")}`);
    const controller = new AbortController();
    const cancel = () => controller.abort(new Error("run interrupted"));
    signals.once("SIGINT", cancel);
    signals.once("SIGTERM", cancel);
    try {
      const summary = await agent.evaluate({
        outputDir, signal: controller.signal,
        ...(values.events ? { onEvent: event => stdout.write(`${JSON.stringify(event)}\n`) } : {}),
        onProgress: result => {
          stdout.write(`${JSON.stringify({ episode: result.id, reason: result.reason, success: result.success, steps: result.steps, error: result.error })}\n`);
          const error = result.error?.message ?? result.scorecardError ?? result.closeError;
          if (error) stderr.write(`${result.id}: ${error}\n`);
          if (error === "chat policy requires baseUrl and model") stderr.write("Set REA_MODEL and REA_BASE_URL in .env, or set policy.options.model and policy.options.baseUrl.\n");
        },
      });
      stdout.write(`${JSON.stringify({ outputDir, ...summary }, null, 2)}\n`);
      return summary.errors || summary.scorecardErrors || summary.closeErrors || controller.signal.aborted ? 1 : 0;
    } finally {
      signals.removeListener("SIGINT", cancel);
      signals.removeListener("SIGTERM", cancel);
      await agent.close();
    }
  } catch (error) {
    stderr.write(`${error.message}\n`);
    return 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  loadCliEnvironment();
  process.exitCode = await main();
}
