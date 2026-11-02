#!/usr/bin/env node
// The whole G1 service, offline: server.mjs with a fixture camera, fixture lidar, the mock actuator and
// the scripted engine. Two preview requests show the fused gate allowing W on a clear frame and
// refusing it on a near one. Needs Node only: no Python, key, robot or network.
import { spawn } from "node:child_process";
import { once } from "node:events";
import { fileURLToPath, pathToFileURL } from "node:url";

const SERVER = fileURLToPath(new URL("../server.mjs", import.meta.url));

export const DEMO_ENV = Object.freeze({
  // an empty path finds no file: nothing from a repository .env reaches the demo, so its environment is exactly this object
  G1_ENV_FILE: "",
  G1_ENGINE: "scripted",
  G1_SCRIPTED_TOKENS: "W",
  G1_PRIMARY_MODEL: "scripted-fixture",
  G1_ACTUATOR: "mock",
  G1_SENSE: "fixture",
  G1_FIXTURE_DEPTH: "2.5,2.5,2.5;0.3,2.5,2.5",
  G1_OBSTACLES: "fixture",
  G1_FIXTURE_LIDAR: "5,5,5",
  G1_AGENT_HOST: "127.0.0.1",
  G1_AGENT_PORT: "0",
  G1_EXECUTE_ENABLED: "0",
  G1_THINKER_ENABLED: "0",
  G1_BRAIN_ENABLED: "0",
  G1_WARM_INTERVAL_S: "0",
  G1_PERSISTENT_CONTROL: "0",
});

export async function runServiceDemo({ node = process.execPath, timeoutMs = 20000 } = {}) {
  const child = spawn(node, [SERVER], { env: { PATH: process.env.PATH, HOME: process.env.HOME, ...DEMO_ENV }, stdio: ["ignore", "pipe", "pipe"] });
  let output = "";
  const started = Promise.withResolvers();
  const timer = setTimeout(() => started.reject(new Error(`server did not start within ${timeoutMs} ms:\n${output}`)), timeoutMs);
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
    const base = `http://127.0.0.1:${log.port}`;
    const results = [];
    for (let step = 1; step <= 2; step += 1) {
      const response = await fetch(`${base}/v1/act`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ instruction: "Approach the doorway.", exec_mode: "none" }) });
      const body = await response.json();
      if (!response.ok && body.error_type !== "gate_refused") throw new Error(`/v1/act returned ${response.status}: ${body.error ?? JSON.stringify(body)}`);
      results.push({ step, token: body.token, gate_allowed: body.gate?.allowed ?? null, reasons: body.gate?.reasons ?? [], executed: body.executed, preview: body.preview });
    }
    return { port: log.port, log, results };
  } finally {
    clearTimeout(timer);
    if (child.exitCode === null && child.signalCode === null) {
      const exited = once(child, "exit");
      child.kill("SIGTERM");
      await Promise.race([exited, new Promise(resolve => setTimeout(resolve, 3000))]);
      if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  runServiceDemo().then(({ results }) => {
    process.stdout.write(`${JSON.stringify({ mode: "service", engine: "scripted", results }, null, 2)}\n`);
    const expected = [true, false];
    if (results.map(result => result.gate_allowed).join() !== expected.join()) {
      process.stderr.write("unexpected gate verdicts\n");
      process.exitCode = 1;
    }
  }).catch(error => {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  });
}
