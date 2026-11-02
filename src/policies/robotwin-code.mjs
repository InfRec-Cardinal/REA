import { fileURLToPath } from "node:url";
import { createChatPolicy } from "../framework/policy.mjs";
import { LineWorker, codeWorkerEnv } from "../workers.mjs";
import { createRobotwinCodexGenerator } from "./robotwin-codex.mjs";

const script = fileURLToPath(new URL("../../scripts/robotwin_code_worker.py", import.meta.url));
const programAction = {
  name: "run_python",
  description: "Run Python calculations and queue exactly one world-frame gripper target.",
  parameters: {
    type: "object", properties: { code: { type: "string" } }, required: ["code"], additionalProperties: false,
  },
};
export const ROBOTWIN_CODE_SYSTEM_PROMPT = "You control a RoboTwin robot using Python. Read the supplied images and observable robot state, verify the previous execution, and plan one motion. Generate code for run_python. Python has np (NumPy), observation (the supplied observation dictionary), instruction, and move_gripper(arm, position, quaternion, opening). Call move_gripper exactly once. arm is left or right, position is world XYZ metres, quaternion is WXYZ, opening is 0 closed to 1 open. Targets are end-effector poses, not TCP: TCP_position = EE_position + R(WXYZ) @ [0.12,0,0]. observation['state']['robot_state'] contains measured poses and fingers; calibration is in observation['state']['calibration']. Use two camera views to ground targets and calculate geometry with NumPy. Only supplied observations are available; there is no simulator object state or expert policy. If another decision arrives, the task is not finished. Preserve useful plans in notes and revise after failures.";

export function createRobotwinCodePolicy({
  python = "python3", model, apiKey, baseUrl,
  executionTimeoutMs = 10000, worker = null, generator = null,
  backend = "chat", codexPath, provider = "rea", providerBaseUrl, apiKeyFile, workRoot,
  apiRetries,
} = {}) {
  if (!["chat", "codex"].includes(backend)) throw new TypeError(`unsupported robotwin code backend: ${backend}`);
  if (!generator && !model) throw new TypeError("robotwin code policy requires a model");
  const policy = generator ?? (backend === "codex" ? createRobotwinCodexGenerator({
    model, codexPath, provider, providerBaseUrl: providerBaseUrl ?? baseUrl, apiKeyFile, workRoot,
    systemPrompt: ROBOTWIN_CODE_SYSTEM_PROMPT, apiRetries,
  }) : createChatPolicy({
    model, apiKey, baseUrl, mode: "tool", maxOutputTokens: 2048,
    systemPrompt: ROBOTWIN_CODE_SYSTEM_PROMPT,
  }));
  let transport = worker;
  let lastError = null;

  return {
    id: "robotwin-python-code",
    async reset() { lastError = null; },
    async decide(input) {
      const generated = await policy.decide({
        ...input, actions: [programAction],
        feedback: { execution: input.feedback, previousCodeError: lastError },
      });
      // Generated code runs without the parent environment, so it cannot read model or robot credentials.
      transport ??= new LineWorker("robotwin-code", python, [script], { env: codeWorkerEnv(), autoRestart: false });
      const abort = () => transport.stop();
      input.signal?.addEventListener("abort", abort, { once: true });
      let reply;
      try {
        input.signal?.throwIfAborted();
        reply = await transport.request({
          code: generated.action.arguments.code,
          observation: input.observation, instruction: input.instruction,
        }, executionTimeoutMs);
      } finally {
        input.signal?.removeEventListener("abort", abort);
      }
      input.signal?.throwIfAborted();
      lastError = reply.ok ? null : reply.error;
      return {
        ...generated,
        action: reply.ok ? { name: "move_gripper", arguments: reply.result.action } : { name: "invalid_output", arguments: {} },
        code: { source: generated.action.arguments.code, error: lastError, stdout: reply.result?.stdout ?? "" },
      };
    },
    async close() { transport?.stop(); await policy.close?.(); },
  };
}

export default createRobotwinCodePolicy;
