import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import { createOpenRALEnvironment } from "../src/environments/openral.mjs";
import { readEnv } from "../src/env.mjs";
import { createChatPolicy } from "../src/framework/policy.mjs";

export default async function configuration() {
  if (!process.env.OPENRAL_CONFIG) throw new Error("set OPENRAL_CONFIG to an OpenRAL driver configuration JSON file");
  const options = JSON.parse(await readFile(process.env.OPENRAL_CONFIG, "utf8"));
  if (options.simEnvYaml?.startsWith(".")) options.simEnvYaml = resolve(dirname(process.env.OPENRAL_CONFIG), options.simEnvYaml);
  const model = readEnv(process.env, "REA_MODEL");
  const baseUrl = readEnv(process.env, "REA_BASE_URL");
  if (!model || !baseUrl) throw new Error("set REA_MODEL and REA_BASE_URL");
  const mode = options.mode ?? "sim";
  const execute = process.env.ROBOT_EXECUTE === undefined ? mode === "sim" : process.env.ROBOT_EXECUTE === "true";
  const instruction = process.env.REA_INSTRUCTION ?? "Observe the robot and choose a useful next action using its declared control modes and units.";
  return {
    metadata: { model, baseUrl, mode, execute, robotId: options.robotId, robotYaml: options.robotYaml, simEnvYaml: options.simEnvYaml, policyMode: "tool" },
    episodes: [{ id: options.robotId ?? "openral-task", task: instruction, instruction }],
    budget: { maxSteps: Number(process.env.REA_MAX_STEPS ?? 1), maxDurationMs: 900000, decisionTimeoutMs: 60000 },
    createEnvironment() {
      return createOpenRALEnvironment({ ...options, mode, execute, openralRoot: process.env.OPENRAL_ROOT ?? options.openralRoot });
    },
    createPolicy() { return createChatPolicy({ model, baseUrl, apiKey: readEnv(process.env, "REA_API_KEY") ?? "" }); },
  };
}
