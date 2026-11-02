import { createRobotEnvironment } from "./robot.mjs";
import { createPythonRobotDriver } from "./python-robot.mjs";

export function createRPentEnvironment({
  env = process.env,
  python = env.RPENT_PYTHON ?? "python3",
  robot = "libero",
  execute = true,
  evaluate = null,
  requestTimeoutMs = robot === "robotwin" ? 660000 : 180000,
  worker = null,
  ...options
} = {}) {
  return createRobotEnvironment({
    id: `rpent:${robot}`, timing: "turn-based", execute, evaluate,
    driver: createPythonRobotDriver({ kind: "rpent", python, env, requestTimeoutMs, worker, options: { ...options, robot, executeEnabled: execute } }),
  });
}
