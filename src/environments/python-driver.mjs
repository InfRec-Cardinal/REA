import { createPythonRobotDriver } from "./python-robot.mjs";
import { createRobotEnvironment } from "./robot.mjs";

/**
 * A robot or simulator driven by your own Python driver class, without editing this repository:
 * driver = "package.module:ClassOrFactory", importable by `python` (set PYTHONPATH in `env` when needed).
 * The class follows the driver contract in docs/extending/robots.md. Execution stays off unless execute is true.
 */
export function createPythonDriverEnvironment({
  driver, env = process.env, python = env.ROBOT_PYTHON ?? "python3", id = "python-robot", execute = false, evaluate = null,
  requestTimeoutMs = 120000, worker = null, options = {},
} = {}) {
  if (typeof driver !== "string" || !/^[A-Za-z_][\w.]*:[A-Za-z_]\w*$/u.test(driver)) {
    throw new TypeError('python-robot needs options.driver as "package.module:ClassOrFactory"');
  }
  return createRobotEnvironment({
    id, execute, evaluate,
    driver: createPythonRobotDriver({ kind: driver, python, env, requestTimeoutMs, worker, options: { ...options, executeEnabled: execute } }),
  });
}
