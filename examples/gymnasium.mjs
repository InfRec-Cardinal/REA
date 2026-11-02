import { createGymnasiumEnvironment } from "../src/environments/gymnasium.mjs";
import { readEnv } from "../src/env.mjs";
import { createChatPolicy } from "../src/framework/policy.mjs";

export default function configuration() {
  const model = readEnv(process.env, "REA_MODEL");
  const baseUrl = readEnv(process.env, "REA_BASE_URL");
  if (!model || !baseUrl) throw new Error("set REA_MODEL and REA_BASE_URL");
  const environmentIds = (process.env.GYM_ENVIRONMENTS ?? "CartPole-v1,Pendulum-v1").split(",").map(value => value.trim()).filter(Boolean);
  const seeds = (process.env.REA_SEEDS ?? "0").split(",").map(Number);
  const makeOptions = JSON.parse(process.env.GYM_MAKE_OPTIONS ?? "{}");
  const modules = (process.env.GYM_MODULES ?? "").split(",").map(value => value.trim()).filter(Boolean);
  return {
    metadata: { model, baseUrl, policyMode: readEnv(process.env, "REA_POLICY_MODE") ?? "tool", makeOptions, modules },
    episodes: environmentIds.flatMap(environmentId => seeds.map(seed => ({ id: `${environmentId}:${seed}`, task: environmentId, seed, instruction: process.env.REA_INSTRUCTION ?? "Infer the control dynamics and maximize the environment's cumulative reward." }))),
    budget: { maxSteps: Number(process.env.REA_MAX_STEPS ?? 200), maxDurationMs: 900000, decisionTimeoutMs: 60000 },
    createEnvironment({ task }) {
      return createGymnasiumEnvironment({ environmentId: task, makeOptions, modules });
    },
    createPolicy() {
      return createChatPolicy({ model, baseUrl, apiKey: readEnv(process.env, "REA_API_KEY") ?? "", mode: readEnv(process.env, "REA_POLICY_MODE") ?? "tool" });
    },
  };
}
