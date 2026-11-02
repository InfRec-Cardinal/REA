import { fileURLToPath } from "node:url";

import { LineWorker } from "../workers.mjs";

const clientScript = fileURLToPath(new URL("../../scripts/robotwin_vla_client.py", import.meta.url));
// End-effector services of the reference setup; any other service is joint-space unless its metadata says otherwise.
const EE_MODELS = new Set(["xvla", "xwam"]);
const cameras = {
  head_camera: ["head", "head_camera"],
  left_camera: ["left_wrist", "left_camera"],
  right_camera: ["right_wrist", "right_camera"],
};

export function createRobotwinVlaPolicy({
  model,
  url,
  env = process.env,
  python = env.ROBOTWIN_VLA_PYTHON ?? "python3",
  policySeed = 42,
  chunkSteps = null,
  requestTimeoutMs = 120000,
  worker = null,
} = {}) {
  if (typeof model !== "string" || !/^[A-Za-z0-9_.-]+$/u.test(model)) throw new TypeError(`Invalid RoboTwin VLA model id: ${model}`);
  if (!["ws:", "wss:"].includes(new URL(url).protocol)) throw new TypeError("RoboTwin VLA url must use ws:// or wss://");
  if (chunkSteps !== null && (!Number.isInteger(chunkSteps) || chunkSteps < 1)) throw new TypeError("chunkSteps must be a positive integer");
  let transport = worker;
  let metadata = null;
  let ready = false;
  let closed = false;

  function disconnect() {
    transport?.stop();
    transport = null;
    metadata = null;
    ready = false;
  }

  async function request(payload, signal) {
    signal?.throwIfAborted();
    if (closed) throw new Error("RoboTwin VLA policy is closed");
    transport ??= new LineWorker(`robotwin-vla:${model}`, python, [clientScript], { env, autoRestart: false });
    const abort = () => disconnect();
    signal?.addEventListener("abort", abort, { once: true });
    try {
      const reply = await transport.request(payload, requestTimeoutMs);
      signal?.throwIfAborted();
      if (!reply.ok) throw new Error(reply.error ?? "RoboTwin VLA inference failed");
      return reply.result;
    } catch (error) {
      disconnect();
      signal?.throwIfAborted();
      throw error;
    } finally {
      signal?.removeEventListener("abort", abort);
    }
  }

  async function reset({ signal } = {}) {
    signal?.throwIfAborted();
    ready = false;
  }

  async function decide({ observation, instruction, signal }) {
    signal?.throwIfAborted();
    const images = Object.fromEntries(Object.entries(cameras).map(([name, aliases]) => {
      const image = observation.images?.find(item => aliases.includes(item.name));
      if (!image?.data) throw new Error(`RoboTwin VLA camera ${name} is missing`);
      return [name, { data: image.data, mimeType: image.mimeType }];
    }));
    const state = observation.state?.vla;
    if (!state?.joint_action || !state?.endpose) throw new Error("RoboTwin VLA requires state.vla.joint_action and state.vla.endpose");
    const taskInstruction = observation.text || instruction;
    if (!taskInstruction) throw new Error("RoboTwin VLA requires a task instruction");
    metadata ??= await request({ op: "configure", url, timeoutMs: requestTimeoutMs }, signal);
    if (!ready) {
      await request({ op: "reset", policy_seed: policySeed }, signal);
      ready = true;
    }
    const result = await request({
      op: "infer", instruction: taskInstruction, images,
      joint_action: state.joint_action, endpose: state.endpose,
    }, signal);
    const nativeType = result.action_type ?? metadata.action_type ?? (EE_MODELS.has(model) ? "ee" : "qpos");
    const actionType = nativeType === "qpos" ? "joint" : nativeType;
    if (!["joint", "ee"].includes(actionType)) throw new Error(`Unsupported RoboTwin VLA action type: ${nativeType}`);
    const dimension = actionType === "ee" ? 16 : 14;
    const actions = result.actions;
    if (!Array.isArray(actions) || !actions.length || actions.some(row => !Array.isArray(row) || row.length !== dimension || !row.every(Number.isFinite))) {
      throw new Error(`RoboTwin ${model} must return a nonempty chunk of ${dimension}-dimensional ${actionType} actions`);
    }
    const horizon = result.execute_horizon ?? metadata.execute_horizon ?? chunkSteps;
    if (!Number.isInteger(horizon) || horizon < 1) throw new Error("RoboTwin VLA service must specify execute_horizon or configure chunkSteps");
    const commands = actions.slice(0, Math.min(horizon, chunkSteps ?? horizon));
    return {
      action: { name: "robotwin_vla_chunk", arguments: { commands, actionType } },
      model,
      vla: { controlSteps: commands.length, predictedSteps: actions.length, inferenceS: result.inference_s ?? null, policySeed },
    };
  }

  return { id: `robotwin-vla:${model}`, decide, reset, async close() { closed = true; disconnect(); } };
}

export default createRobotwinVlaPolicy;
