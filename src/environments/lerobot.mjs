import { createRobotEnvironment } from "./robot.mjs";
import { createPythonRobotDriver } from "./python-robot.mjs";

export function createLeRobotEnvironment({
  env = process.env,
  python = env.LEROBOT_PYTHON ?? "python3",
  execute = false,
  evaluate = null,
  requestTimeoutMs = 120000,
  worker = null,
  ...options
} = {}) {
  return createRobotEnvironment({
    id: "lerobot", execute, evaluate,
    driver: createPythonRobotDriver({ kind: "lerobot", python, env, requestTimeoutMs, worker, options: { ...options, executeEnabled: execute } }),
  });
}
