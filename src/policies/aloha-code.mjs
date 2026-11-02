import { fileURLToPath } from "node:url";

import { readEnv } from "../env.mjs";
import { createChatPolicy } from "../framework/policy.mjs";
import { LineWorker, codeWorkerEnv } from "../workers.mjs";

const script = fileURLToPath(new URL("../../scripts/aloha_code_worker.py", import.meta.url));
const programAction = {
  name: "run_python",
  description: "Compute one absolute ALOHA joint target or trajectory from the current observation.",
  parameters: {
    type: "object", properties: { code: { type: "string" } }, required: ["code"], additionalProperties: false,
  },
};

export const ALOHA_CODE_SYSTEM_PROMPT = "You control Classic ROS1 ALOHA using Python. Read the supplied camera images, measured robot state, and previous execution feedback. Generate one run_python program. Python provides np (NumPy), observation, instruction, joint_position(positions), and joint_trajectory(positions). Call exactly one of these two functions exactly once. joint_position accepts one absolute 14-value vector; joint_trajectory accepts 1 to 100 such vectors, submitted one per native control interval. The order is left arm 6 joints, left gripper, right arm 6 joints, right gripper. Arm positions are radians; grippers are normalized from 0 closed to 1 open. observation['state']['qpos'] is the latest measured vector; observation['state']['robot'] describes joint names, units and control interval. Use the supplied action schemas for joint limits. These are absolute joint positions, not Cartesian poses or joint deltas. Begin from measured qpos, preserve joints you do not intend to change, and use short trajectories to obtain fresh feedback. Python only computes an action; the ALOHA environment executes it after validation. Do not import ROS or robot SDKs or command hardware directly. No hidden object state, inverse-kinematics service or calibrated camera geometry is provided. Preserve useful plans in notes and use observed evidence rather than command acknowledgement to assess completion.";

/** Generated Python computes an ALOHA action; the environment owns all robot I/O. */
export function createAlohaCodePolicy(options = {}, context = {}) {
  const env = options.env ?? context.env ?? process.env;
  const {
    python = env.REA_CODE_PYTHON ?? "python3",
    model = env.REA_CODE_MODEL ?? readEnv(env, "REA_MODEL"),
    apiKey = readEnv(env, "REA_API_KEY") ?? "",
    baseUrl = readEnv(env, "REA_BASE_URL"),
    maxOutputTokens = 2048,
    executionTimeoutMs = 10000,
    generator = null, worker = null,
  } = options;
  const policy = generator ?? createChatPolicy({
    model, apiKey, baseUrl, mode: "tool", maxOutputTokens,
    systemPrompt: ALOHA_CODE_SYSTEM_PROMPT,
  });
  let transport = worker;
  let lastError = null;

  function disconnect() {
    transport?.stop();
    transport = null;
  }

  return {
    id: "aloha-python-code",
    async reset(input) { lastError = null; await policy.reset?.(input); },
    async decide(input) {
      input.signal?.throwIfAborted();
      const generated = await policy.decide({
        ...input,
        actions: [programAction],
        instruction: `${input.instruction ?? input.observation?.text ?? "Complete the task."}\n\nALOHA action schemas: ${JSON.stringify(input.actions ?? input.observation?.actions ?? [])}`,
        feedback: { execution: input.feedback, previousCodeError: lastError },
      });
      input.signal?.throwIfAborted();
      const source = generated.action.arguments.code;
      // Generated code runs without the parent environment, so it cannot read model or robot credentials.
      transport ??= new LineWorker("aloha-code", python, [script], { env: codeWorkerEnv(env), autoRestart: false });
      input.signal?.addEventListener("abort", disconnect, { once: true });
      let reply;
      try {
        reply = await transport.request({
          code: source, observation: input.observation,
          instruction: input.instruction ?? input.observation?.text ?? "Complete the task.",
        }, executionTimeoutMs);
        input.signal?.throwIfAborted();
      } catch (error) {
        disconnect();
        input.signal?.throwIfAborted();
        throw error;
      } finally {
        input.signal?.removeEventListener("abort", disconnect);
      }
      lastError = reply.ok ? null : reply.error;
      if (!reply.ok) {
        throw Object.assign(new Error(`ALOHA code execution failed: ${reply.error}`), {
          usage: generated.usage, code: "invalid_action",
        });
      }
      return {
        ...generated, action: reply.result.action,
        code: { source, error: null, stdout: reply.result.stdout ?? "" },
      };
    },
    async close() { disconnect(); await policy.close?.(); },
  };
}

export default createAlohaCodePolicy;
