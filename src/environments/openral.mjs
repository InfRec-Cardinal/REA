import { createRobotEnvironment } from "./robot.mjs";
import { createPythonRobotDriver } from "./python-robot.mjs";

export function createOpenRALEnvironment({
  env = process.env,
  python = env.OPENRAL_PYTHON ?? "python3",
  mode = "sim",
  execute = mode === "sim",
  evaluate = null,
  requestTimeoutMs = 120000,
  worker = null,
  ...options
} = {}) {
  if (!["sim", "real"].includes(mode)) throw new Error("OpenRAL mode must be sim or real");
  return createRobotEnvironment({
    id: `openral:${options.robotId ?? "custom"}`,
    execute, evaluate,
    driver: createPythonRobotDriver({ kind: "openral", python, env, requestTimeoutMs, worker, options: { ...options, mode, executeEnabled: execute } }),
  });
}
