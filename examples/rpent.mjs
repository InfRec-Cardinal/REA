import { readFile } from "node:fs/promises";

import { createRPentEnvironment } from "../src/environments/rpent.mjs";
import { readEnv } from "../src/env.mjs";
import { createChatPolicy } from "../src/framework/policy.mjs";

export default async function configuration() {
  if (!process.env.RPENT_CONFIG) throw new Error("set RPENT_CONFIG to an RPent environment-service configuration JSON file");
  const options = JSON.parse(await readFile(process.env.RPENT_CONFIG, "utf8"));
  const model = readEnv(process.env, "REA_MODEL");
  const baseUrl = readEnv(process.env, "REA_BASE_URL");
  if (!model || !baseUrl) throw new Error("set REA_MODEL and REA_BASE_URL");
  const execute = process.env.ROBOT_EXECUTE === undefined || process.env.ROBOT_EXECUTE === "true";
  return {
    metadata: { model, baseUrl, execute, robot: options.robot, endpoint: options.endpoint, expectedMeta: options.expectedMeta, policyMode: "tool" },
    episodes: [{ id: options.robot, seed: Number(process.env.REA_SEED ?? 0), instruction: process.env.REA_INSTRUCTION }],
    budget: { maxSteps: Number(process.env.REA_MAX_STEPS ?? 20), maxDurationMs: 900000, decisionTimeoutMs: 60000 },
    createEnvironment() {
      return createRPentEnvironment({ ...options, execute, rpentRoot: process.env.RPENT_ROOT ?? options.rpentRoot });
    },
    createPolicy() { return createChatPolicy({ model, baseUrl, apiKey: readEnv(process.env, "REA_API_KEY") ?? "" }); },
  };
}
