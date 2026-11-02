import { readFile } from "node:fs/promises";

import { createLeRobotEnvironment } from "../src/environments/lerobot.mjs";
import { readEnv } from "../src/env.mjs";
import { createChatPolicy } from "../src/framework/policy.mjs";

export default async function configuration() {
  if (!process.env.LEROBOT_CONFIG) throw new Error("set LEROBOT_CONFIG to a robot driver configuration JSON file");
  const options = JSON.parse(await readFile(process.env.LEROBOT_CONFIG, "utf8"));
  const model = readEnv(process.env, "REA_MODEL");
  const baseUrl = readEnv(process.env, "REA_BASE_URL");
  if (!model || !baseUrl) throw new Error("set REA_MODEL and REA_BASE_URL");
  const execute = process.env.ROBOT_EXECUTE === "true";
  return {
    metadata: { model, baseUrl, execute, robot: options.robotConfig?.type ?? options.robotFactory, policyMode: "tool" },
    episodes: [{ id: "robot-task", task: process.env.REA_INSTRUCTION ?? "Observe the robot and describe a useful next joint command." }],
    budget: { maxSteps: Number(process.env.REA_MAX_STEPS ?? 1), maxDurationMs: 900000, decisionTimeoutMs: 60000 },
    createEnvironment() { return createLeRobotEnvironment({ ...options, execute }); },
    createPolicy() { return createChatPolicy({ model, baseUrl, apiKey: readEnv(process.env, "REA_API_KEY") ?? "" }); },
  };
}
