import { createRobotEnvironment } from "./robot.mjs";
import { createPythonRobotDriver } from "./python-robot.mjs";

export function createAlohaEnvironment({
  env = process.env,
  python = env.ALOHA_PYTHON ?? "python3",
  mode = "sim",
  execute = mode === "sim",
  evaluate = null,
  requestTimeoutMs = 120000,
  worker = null,
  ...options
} = {}) {
  if (!["sim", "real"].includes(mode)) throw new Error("ALOHA mode must be sim or real");
  return createRobotEnvironment({
    id: `aloha:${mode}`,
    timing: mode === "sim" ? "turn-based" : "realtime",
    execute,
    evaluate,
    driver: createPythonRobotDriver({ kind: "aloha", python, env, requestTimeoutMs, worker, options: { ...options, mode, executeEnabled: execute } }),
  });
}
