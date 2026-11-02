import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { createAlohaEnvironment } from "../src/environments/aloha.mjs";
import { createLeRobotEnvironment } from "../src/environments/lerobot.mjs";
import { createOpenRALEnvironment } from "../src/environments/openral.mjs";
import { createRPentEnvironment } from "../src/environments/rpent.mjs";
import { createArcAgi3Environment } from "../src/environments/arc-agi3.mjs";
import { createGymnasiumEnvironment } from "../src/environments/gymnasium.mjs";

// Speak the backend protocol in a real child process without requiring six optional SDKs.
async function interpreterFixture(context) {
  const directory = await mkdtemp(join(tmpdir(), "fast-adapter-env-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const executable = join(directory, "interpreter.mjs");
  await writeFile(executable, `#!${process.execPath}
import { createInterface } from "node:readline";
const lines = createInterface({ input: process.stdin });
let optionsContainEnv = false;
process.stdout.write(JSON.stringify({ event: "ready" }) + "\\n");
lines.on("line", line => {
  const request = JSON.parse(line);
  if (request.op === "configure") optionsContainEnv = Object.hasOwn(request.options, "env");
  const state = { runtimeValue: process.env.REA_ADAPTER_ENV_TEST, operationMode: process.env.OPERATION_MODE, optionsContainEnv };
  const snapshot = { id: "env-fixture", state, images: [], actions: [{ name: "wait", parameters: { type: "object" } }] };
  const result = request.op === "configure" ? { actions: snapshot.actions, metadata: {} } : { ...snapshot, observation: snapshot };
  process.stdout.write(JSON.stringify({ id: request.id, ok: true, result }) + "\\n", () => {
    if (request.op === "close") process.exit(0);
  });
});
`, { mode: 0o755 });
  return executable;
}

test("all Python adapters pass their supplied environment to the child and use its interpreter default", async context => {
  const executable = await interpreterFixture(context);
  const original = process.env.REA_ADAPTER_ENV_TEST;
  for (const [name, create, pythonKey] of [
    ["aloha", createAlohaEnvironment, "ALOHA_PYTHON"],
    ["lerobot", createLeRobotEnvironment, "LEROBOT_PYTHON"],
    ["openral", createOpenRALEnvironment, "OPENRAL_PYTHON"],
    ["rpent", createRPentEnvironment, "RPENT_PYTHON"],
    ["arc-agi3", createArcAgi3Environment, "ARC_PYTHON"],
    ["gymnasium", createGymnasiumEnvironment, "GYM_PYTHON"],
  ]) {
    const env = { ...process.env, [pythonKey]: executable, REA_ADAPTER_ENV_TEST: `runtime-${name}`, OPERATION_MODE: "COMPETITION" };
    const environment = create({ env, operationMode: "OFFLINE" });
    try {
      const initial = await environment.reset({ task: "fixture", seed: 1 });
      assert.equal(initial.observation.state.runtimeValue, `runtime-${name}`);
      assert.equal(initial.observation.state.optionsContainEnv, false);
      assert.equal(initial.observation.state.operationMode, name === "arc-agi3" ? "OFFLINE" : "COMPETITION");
      assert.equal(process.env.REA_ADAPTER_ENV_TEST, original);
    } finally {
      await environment.close();
    }
  }
});
