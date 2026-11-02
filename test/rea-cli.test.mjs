import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { EventEmitter } from "node:events";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import test from "node:test";

import { applyOverrides, main, parseCommand } from "../scripts/rea.mjs";
import { inspectConfiguration } from "../src/framework/doctor.mjs";
import { createRegistry } from "../src/framework/registry.mjs";

const exec = promisify(execFile);
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

async function directory(context) {
  const cwd = await mkdtemp(join(tmpdir(), "fast-cli-"));
  context.after(() => rm(cwd, { recursive: true, force: true }));
  return cwd;
}

async function invoke(argv, options = {}) {
  let stdout = "";
  let stderr = "";
  const signals = new EventEmitter();
  const code = await main(argv, { env: {}, signals, ...options, stdout: { write(value) { stdout += value; } }, stderr: { write(value) { stderr += value; } } });
  assert.equal(signals.listenerCount("SIGINT"), 0);
  assert.equal(signals.listenerCount("SIGTERM"), 0);
  return { code, stdout, stderr };
}

test("CLI help, catalog and invalid commands need no robot or model configuration", async () => {
  const help = await invoke(["--help"]);
  assert.equal(help.code, 0);
  assert.match(help.stdout, /rea init <preset>/u);
  assert.match(help.stdout, /npm run rea -- <command>/u);
  const catalog = await invoke(["list"]);
  assert.equal(catalog.code, 0);
  for (const name of ["aloha", "openral", "rpent", "arc-agi3", "demo", "python-robot", "lingbot-vla", "route-memory", "judge"]) assert.match(catalog.stdout, new RegExp(name, "u"));
  const invalid = await invoke(["run"]);
  assert.equal(invalid.code, 1);
  assert.match(invalid.stderr, /requires a preset or configuration file/u);
  assert.throws(() => parseCommand(["init", "demo", "--steps", "5"]), /not an option/u);
  assert.throws(() => parseCommand(["run", "demo", "--steps", "1.5"]), /integer/u);
});

test("init writes an editable recipe and preserves an existing user configuration", async context => {
  const cwd = await directory(context);
  const first = await invoke(["init", "aloha"], { cwd, env: { npm_lifecycle_event: "rea" } });
  assert.equal(first.code, 0, first.stderr);
  assert.match(first.stdout, /Next: npm run rea -- doctor/u);
  const filename = join(cwd, "rea.config.json");
  const original = await readFile(filename, "utf8");
  assert.equal(JSON.parse(original).environment.type, "aloha");
  const repeated = await invoke(["init", "demo"], { cwd });
  assert.equal(repeated.code, 1);
  assert.match(repeated.stderr, /already exists/u);
  assert.equal(await readFile(filename, "utf8"), original);
});

test("one command runs the offline preset and saves standard evaluation artifacts", async context => {
  const cwd = await directory(context);
  const result = await invoke(["run", "demo", "--output", "results", "--seed", "4"], { cwd });
  assert.equal(result.code, 0, result.stderr);
  const summary = JSON.parse(await readFile(join(cwd, "results", "summary.json"), "utf8"));
  const manifest = JSON.parse(await readFile(join(cwd, "results", "manifest.json"), "utf8"));
  assert.equal(summary.successes, summary.episodesPlanned);
  assert.equal(summary.errors, 0);
  assert.ok(manifest.episodes.every(episode => episode.seed === 4));
  assert.match(await readFile(join(cwd, "results", "0001.jsonl"), "utf8"), /"action"/u);
  assert.match(result.stdout, /"outputDir"/u);
});

test("JSON recipe paths work outside the repository, including spaces", async context => {
  const cwd = await directory(context);
  const initialized = await invoke(["init", "demo", "--output", "recipes/my agent.json"], { cwd });
  assert.equal(initialized.code, 0, initialized.stderr);
  const executed = await invoke(["run", "recipes/my agent.json", "--output", "my run", "--steps", "1"], { cwd });
  assert.equal(executed.code, 0, executed.stderr);
  const manifest = JSON.parse(await readFile(join(cwd, "my run", "manifest.json"), "utf8"));
  assert.equal(manifest.budget.maxSteps, 1);
  assert.ok(manifest.episodes.every(episode => episode.budget.maxSteps === 1));
});

test("task overrides preserve benchmark settings and allow full JSON task objects", () => {
  const config = {
    environment: { type: "libero", options: {} }, policy: { type: "chat", options: { temperature: 0 } },
    episodes: [{ task: { suite: "libero_10_task", task: 0, horizon: 100 }, seed: 1, budget: { maxSteps: 12 } }],
  };
  const changed = applyOverrides(config, { task: "3", seed: 9, steps: 2, model: "vision-model", python: "/sdk/bin/python", "base-url": "http://localhost:8000/v1" });
  assert.deepEqual(changed.episodes[0].task, { suite: "libero_10_task", task: 3, horizon: 100 });
  assert.equal(changed.episodes[0].budget.maxSteps, 2);
  assert.equal(changed.policy.options.model, "vision-model");
  assert.equal(changed.environment.options.python, "/sdk/bin/python");
  assert.equal(config.episodes[0].task.task, 0);
  assert.deepEqual(applyOverrides(config, { task: '{"suite":"other","task":5}' }).episodes[0].task, { suite: "other", task: 5 });
  for (const key of ["taskName", "gameId"]) assert.deepEqual(applyOverrides({ episodes: [{ task: { [key]: "old", keep: true } }] }, { task: "new" }).episodes[0].task, { [key]: "new", keep: true });
  assert.throws(() => applyOverrides({ createEnvironment() {}, createPolicy() {} }, { model: "ignored" }), /chat policy/u);
});

test("doctor accepts the offline recipe and reports incomplete chat setup without invoking a policy", async () => {
  const valid = await invoke(["doctor", "demo"]);
  assert.equal(valid.code, 0, valid.stderr);
  assert.match(valid.stdout, /no robot, simulator, or model connection was opened/u);
  const report = await inspectConfiguration({ environment: "demo", policy: "chat" }, { env: {} });
  assert.equal(report.ok, false);
  assert.ok(report.checks.some(check => check.name === "policy model" && check.status === "error"));
  const local = await inspectConfiguration({ environment: "demo", policy: { type: "chat", options: { model: "local-model", baseUrl: "http://localhost:8000/v1" } } }, { env: {} });
  assert.equal(local.ok, true, "local model servers can work without an API key");
  const privateConfig = await inspectConfiguration({ environment: "demo", policy: "chat" }, { env: { REA_MODEL: "model", REA_BASE_URL: "https://model.example/v1", REA_API_KEY: "private-test-credential" } });
  assert.doesNotMatch(JSON.stringify(privateConfig), /private-test-credential/u);
  const routed = await inspectConfiguration({ environment: "demo", policy: { type: "routed", options: { reflex: "demo", deliberate: "chat" } } }, { env: {} });
  assert.equal(routed.ok, false);
  assert.ok(routed.checks.some(check => check.name === "policy.deliberate model" && check.status === "error"));
});

test("run explains missing model settings while preserving the failed episode report", async context => {
  const cwd = await directory(context);
  await writeFile(join(cwd, "rea.config.json"), JSON.stringify({ environment: "demo", policy: "chat" }));
  const result = await invoke(["run", "rea.config.json", "--output", "missing-model"], { cwd });
  assert.equal(result.code, 1);
  assert.match(result.stderr, /Set REA_MODEL and REA_BASE_URL/u);
  const episode = JSON.parse(await readFile(join(cwd, "missing-model", "0001.json"), "utf8"));
  assert.equal(episode.error.message, "chat policy requires baseUrl and model");
});

test("doctor inspects Python package availability without importing the package", async context => {
  const cwd = await directory(context);
  const marker = join(cwd, "imported.txt");
  await writeFile(join(cwd, "fast_probe_fixture.py"), `from pathlib import Path\nPath(${JSON.stringify(marker)}).write_text('imported')\n`);
  const registry = createRegistry();
  registry.registerEnvironment("probe", () => { throw new Error("must not construct an environment"); }, { requires: { pythonModules: ["fast_probe_fixture", "fast_package_that_is_not_installed"] } });
  const report = await inspectConfiguration({ environment: "probe", policy: "demo" }, { baseDir: cwd, env: process.env, registry });
  assert.equal(report.ok, false);
  assert.ok(report.checks.some(check => check.name === "Python module fast_probe_fixture" && check.status === "ok"));
  assert.ok(report.checks.some(check => check.name === "Python module fast_package_that_is_not_installed" && check.status === "error"));
  await assert.rejects(readFile(marker), { code: "ENOENT" });
});

test("doctor never loads executable JavaScript configuration files", async context => {
  const cwd = await directory(context);
  await writeFile(join(cwd, "would-connect.mjs"), "throw new Error('configuration was executed');\n");
  const result = await invoke(["doctor", "would-connect.mjs"], { cwd });
  assert.equal(result.code, 1);
  assert.match(result.stderr, /this file was not loaded/u);
  assert.doesNotMatch(result.stderr, /configuration was executed/u);
  const invalid = await inspectConfiguration({ environment: "module", policy: "module" }, { baseDir: cwd, env: {} });
  assert.equal(invalid.ok, false);
  assert.ok(invalid.checks.some(check => check.name === "environment module" && check.status === "error"));
  assert.ok(invalid.checks.some(check => check.name === "policy" && check.status === "error"));
  const readable = await inspectConfiguration({ environment: { type: "module", options: { module: "./would-connect.mjs" } }, policy: { type: "module", options: { module: "./would-connect.mjs" } } }, { baseDir: cwd, env: {} });
  assert.equal(readable.ok, true);
  assert.ok(readable.checks.some(check => check.name === "policy" && check.status === "ok"));
});

test("doctor checks per-episode adapter overrides without calling their factories", async () => {
  const registry = createRegistry();
  registry.registerEnvironment("episode-only", () => { throw new Error("must not start"); }, { requires: { pythonEnv: "MISSING_PYTHON" } });
  const report = await inspectConfiguration({ environment: "demo", policy: "demo", episodes: [{ id: "different-robot", environment: { type: "episode-only", options: { python: "/does-not-exist/fast-python" } } }] }, { env: {}, registry });
  assert.equal(report.ok, false);
  assert.ok(report.checks.some(check => check.name === "different-robot: Python" && check.status === "error"));
});

test("rea run executes an existing JavaScript configuration with the unified runtime", async context => {
  const cwd = await directory(context);
  const output = join(cwd, "legacy");
  const result = await exec(process.execPath, [join(root, "scripts", "rea.mjs"), "run", join(root, "examples", "eval-fixtures.mjs"), "--output", output], { cwd, env: process.env });
  const summary = JSON.parse(await readFile(join(output, "summary.json"), "utf8"));
  assert.equal(summary.successes, 2);
  assert.match(result.stdout, /"episodesRun": 2/u);
});

test("CLI loads a standalone workspace .env, explicit files and existing shell values", async context => {
  const cwd = await directory(context);
  const env = { ...process.env };
  for (const key of ["REA_ENV_FILE", "FAST_ENV_FILE", "G1_ENV_FILE", "REA_MODEL", "FAST_MODEL", "REA_BASE_URL", "FAST_BASE_URL", "REA_API_KEY", "FAST_API_KEY"]) delete env[key];
  await writeFile(join(cwd, ".env"), "REA_MODEL=workspace-model\nREA_BASE_URL=http://127.0.0.1:9/v1\n");
  await writeFile(join(cwd, "rea.config.json"), JSON.stringify({ environment: "demo", policy: "chat" }));
  const checked = await exec(process.execPath, [join(root, "scripts", "rea.mjs"), "doctor", "rea.config.json"], { cwd, env });
  assert.match(checked.stdout, /\[ok\] policy model/u);
  assert.match(checked.stdout, /no robot, simulator, or model connection was opened/u);
  const moduleUrl = pathToFileURL(join(root, "scripts", "rea.mjs")).href;
  const probe = `import { loadCliEnvironment } from ${JSON.stringify(moduleUrl)}; loadCliEnvironment(); process.stdout.write(JSON.stringify({model:process.env.REA_MODEL,baseUrl:process.env.REA_BASE_URL}));`;
  const shell = await exec(process.execPath, ["--input-type=module", "-e", probe], { cwd, env: { ...env, REA_MODEL: "shell-model" } });
  assert.deepEqual(JSON.parse(shell.stdout), { model: "shell-model", baseUrl: "http://127.0.0.1:9/v1" });
  await writeFile(join(cwd, "chosen.env"), "REA_MODEL=chosen-model\nREA_BASE_URL=http://chosen.invalid/v1\n");
  await writeFile(join(cwd, "legacy.env"), "REA_MODEL=legacy-model\nREA_BASE_URL=http://legacy.invalid/v1\n");
  const chosen = await exec(process.execPath, ["--input-type=module", "-e", probe], { cwd, env: { ...env, REA_ENV_FILE: "chosen.env", G1_ENV_FILE: "legacy.env", REA_MODEL: "shell-model" } });
  assert.deepEqual(JSON.parse(chosen.stdout), { model: "shell-model", baseUrl: "http://chosen.invalid/v1" });
  const legacy = await exec(process.execPath, ["--input-type=module", "-e", probe], { cwd, env: { ...env, G1_ENV_FILE: "legacy.env" } });
  assert.deepEqual(JSON.parse(legacy.stdout), { model: "legacy-model", baseUrl: "http://legacy.invalid/v1" });
  // A .env written before the rename still configures the policy; the CLI says so once on stderr.
  await writeFile(join(cwd, ".env"), "FAST_MODEL=old-name\nFAST_BASE_URL=http://old.invalid/v1\n");
  const deprecated = await exec(process.execPath, [join(root, "scripts", "rea.mjs"), "doctor", "rea.config.json"], { cwd, env });
  assert.match(deprecated.stdout, /\[ok\] policy model/u);
  assert.equal(deprecated.stderr, "REA_MODEL is not set; using the deprecated FAST_MODEL (rename it in .env)\nREA_BASE_URL is not set; using the deprecated FAST_BASE_URL (rename it in .env)\n");
});
