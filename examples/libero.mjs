import { fileURLToPath } from "node:url";

import { createLiberoEnvironment } from "../src/environments/libero.mjs";
import { readEnv } from "../src/env.mjs";
import { createChatPolicy } from "../src/framework/policy.mjs";
import { LiberoClient } from "../src/libero-client.mjs";

export default function configuration() {
  const python = process.env.LIBERO_PYTHON;
  const suite = process.env.LIBERO_SUITE;
  const model = readEnv(process.env, "REA_MODEL");
  const baseUrl = readEnv(process.env, "REA_BASE_URL");
  if (!python || !suite || !model || !baseUrl) throw new Error("set LIBERO_PYTHON, LIBERO_SUITE, REA_MODEL and REA_BASE_URL");
  const tasks = (process.env.LIBERO_TASKS ?? "0").split(",").map(Number);
  const seeds = (process.env.REA_SEEDS ?? "51").split(",").map(Number);
  const horizon = Number(process.env.LIBERO_HORIZON ?? 600);
  return {
    metadata: { model, baseUrl, policyMode: readEnv(process.env, "REA_POLICY_MODE") ?? "tool", suite },
    episodes: tasks.flatMap(task => seeds.map(seed => ({
      id: `${suite}:${task}:${seed}`, seed,
      task: { suite, task, horizon, repeatSteps: 5, amplitude: 0.5 },
    }))),
    budget: { maxSteps: Number(process.env.REA_MAX_STEPS ?? 120), maxDurationMs: 1800000, decisionTimeoutMs: 60000 },
    createEnvironment() {
      return createLiberoEnvironment({ client: new LiberoClient({
        python,
        script: fileURLToPath(new URL("../scripts/libero_backend.py", import.meta.url)),
      }) });
    },
    createPolicy() {
      return createChatPolicy({ model, baseUrl, apiKey: readEnv(process.env, "REA_API_KEY") ?? "", mode: readEnv(process.env, "REA_POLICY_MODE") ?? "tool" });
    },
  };
}
