import { createArcAgi3Environment } from "../src/environments/arc-agi3.mjs";
import { readEnv } from "../src/env.mjs";
import { createChatPolicy } from "../src/framework/policy.mjs";

export default function configuration() {
  const model = readEnv(process.env, "REA_MODEL");
  const baseUrl = readEnv(process.env, "REA_BASE_URL");
  if (!model || !baseUrl) throw new Error("set REA_MODEL and REA_BASE_URL for the ARC policy");
  const games = (process.env.ARC_GAMES ?? "").split(",").map(game => game.trim()).filter(Boolean);
  if (!games.length) throw new Error("set ARC_GAMES to a comma-separated list of game IDs");
  const seeds = (process.env.REA_SEEDS ?? "0").split(",").map(Number);
  const operationMode = (process.env.ARC_OPERATION_MODE ?? "OFFLINE").toUpperCase();
  if (operationMode === "COMPETITION" && (seeds.length !== 1 || new Set(games).size !== games.length)) {
    throw new Error("ARC COMPETITION permits one make per game: use unique ARC_GAMES and one REA_SEEDS value");
  }
  return {
    reuseEnvironment: true,
    metadata: { model, baseUrl, policyMode: readEnv(process.env, "REA_POLICY_MODE") ?? "tool", operationMode },
    episodes: games.flatMap(gameId => seeds.map(seed => ({ id: `${gameId}:${seed}`, task: { gameId }, seed }))),
    budget: {
      maxSteps: Number(process.env.REA_MAX_STEPS ?? 200),
      maxDurationMs: Number(process.env.REA_EPISODE_TIMEOUT_MS ?? 900000),
      decisionTimeoutMs: Number(process.env.REA_DECISION_TIMEOUT_MS ?? 60000),
    },
    createEnvironment() {
      return createArcAgi3Environment({
        operationMode,
        environmentsDir: process.env.ARC_ENVIRONMENTS_DIR,
        recordingsDir: process.env.ARC_RECORDINGS_DIR,
        saveRecording: true,
      });
    },
    createPolicy() {
      return createChatPolicy({
        model,
        baseUrl,
        apiKey: readEnv(process.env, "REA_API_KEY") ?? "",
        mode: readEnv(process.env, "REA_POLICY_MODE") ?? "tool",
        maxOutputTokens: Number(process.env.REA_MAX_OUTPUT_TOKENS ?? 2048),
      });
    },
  };
}
