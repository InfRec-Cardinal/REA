import assert from "node:assert/strict";
import { test } from "node:test";

import { once } from "node:events";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

import { DEMO_ENV, runServiceDemo } from "../examples/service.mjs";

test("the real server runs end to end on fixtures: the scripted W is allowed, then refused on near depth", { timeout: 30000 }, async () => {
  const { results, log } = await runServiceDemo();
  assert.equal(results.length, 2);
  assert.deepEqual(results.map(result => result.token), ["W", "W"]);
  assert.deepEqual(results.map(result => result.gate_allowed), [true, false]);
  assert.ok(results[1].reasons.includes("depth:depth_center_near"));
  assert.ok(results.every(result => result.preview === true && result.executed === false));
  assert.equal(log.engine, "scripted");
  assert.equal(log.robot_profile, "unitree-g1");
  assert.ok(Number.isInteger(log.port) && log.port > 0, "the bound port is logged when G1_AGENT_PORT=0");
});

test("the demo pins its whole environment and never loads a repository .env", () => {
  assert.equal(DEMO_ENV.G1_ENV_FILE, "");
  assert.equal(DEMO_ENV.G1_AGENT_HOST, "127.0.0.1");
  assert.equal(Object.isFrozen(DEMO_ENV), true);
});

test("the service starts under a profile with no W token and no G1_SCRIPTED_TOKENS", { timeout: 30000 }, async () => {
  // Regression: the scripted engine was registered with a literal "W" whatever the profile, so any
  // profile without that token aborted at startup whichever engine was selected.
  const env = { PATH: process.env.PATH, HOME: process.env.HOME, ...DEMO_ENV };
  delete env.G1_SCRIPTED_TOKENS;
  env.G1_ROBOT_PROFILE = fileURLToPath(new URL("./fixtures/profiles/holonomic.mjs", import.meta.url));
  const server = fileURLToPath(new URL("../server.mjs", import.meta.url));
  const child = spawn(process.execPath, [server], { env, stdio: ["ignore", "pipe", "pipe"] });
  let output = "";
  const started = Promise.withResolvers();
  const timer = setTimeout(() => started.reject(new Error(`server did not start:\n${output}`)), 20000);
  child.stdout.on("data", chunk => {
    output += chunk;
    for (const line of output.split("\n")) {
      if (!line.includes('"event":"service_started"')) continue;
      try { started.resolve(JSON.parse(line)); } catch { /* partial line, keep reading */ }
    }
  });
  child.stderr.on("data", chunk => { output += chunk; });
  child.once("exit", code => started.reject(new Error(`server exited ${code}:\n${output}`)));
  try {
    const log = await started.promise;
    const health = await (await fetch(`http://127.0.0.1:${log.port}/health`)).json();
    assert.equal(health.ok, true, "/health answers under a profile the scripted default had to be derived for");
    assert.equal(health.engine, "scripted");
    assert.equal(log.robot_profile, "holonomic-test");
    const capabilities = await (await fetch(`http://127.0.0.1:${log.port}/v1/capabilities`)).json();
    assert.deepEqual(capabilities.engines.scripted.tokens, ["F"], "the default follows the profile's forward role");
  } finally {
    clearTimeout(timer);
    if (child.exitCode === null && child.signalCode === null) {
      const exited = once(child, "exit");
      child.kill("SIGTERM");
      await Promise.race([exited, new Promise(resolve => setTimeout(resolve, 3000))]);
      if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
    }
  }
});
