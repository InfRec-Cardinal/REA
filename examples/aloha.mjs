import { createAlohaEnvironment } from "../src/environments/aloha.mjs";
import { readEnv } from "../src/env.mjs";
import { createChatPolicy } from "../src/framework/policy.mjs";

export default function configuration() {
  const mode = process.env.ALOHA_MODE ?? "sim";
  const model = readEnv(process.env, "REA_MODEL");
  const baseUrl = readEnv(process.env, "REA_BASE_URL");
  if (!model || !baseUrl) throw new Error("set REA_MODEL and REA_BASE_URL");
  const tasks = (process.env.ALOHA_TASKS ?? "sim_transfer_cube_scripted").split(",");
  const seeds = (process.env.REA_SEEDS ?? "0").split(",").map(Number);
  const execute = process.env.ROBOT_EXECUTE === undefined ? mode === "sim" : process.env.ROBOT_EXECUTE === "true";
  return {
    metadata: { model, baseUrl, mode, execute, policyMode: "tool" },
    episodes: tasks.flatMap(taskName => seeds.map(seed => ({
      id: `${taskName}:${seed}`, task: { taskName }, seed,
      instruction: process.env.REA_INSTRUCTION ?? "Complete the bimanual manipulation task using the camera observations and joint positions.",
    }))),
    budget: { maxSteps: Number(process.env.REA_MAX_STEPS ?? 20), maxDurationMs: 900000, decisionTimeoutMs: 60000 },
    createEnvironment() {
      return createAlohaEnvironment({
        mode, execute,
        actRoot: process.env.ACT_ROOT,
        alohaRoot: process.env.ALOHA_ROOT,
        cameraNames: process.env.ALOHA_CAMERAS?.split(","),
      });
    },
    createPolicy() {
      return createChatPolicy({ model, baseUrl, apiKey: readEnv(process.env, "REA_API_KEY") ?? "", maxOutputTokens: Number(process.env.REA_MAX_OUTPUT_TOKENS ?? 2048) });
    },
  };
}
