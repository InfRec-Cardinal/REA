import { fileURLToPath } from "node:url";
import { LineWorker } from "../workers.mjs";

const clientScript = fileURLToPath(new URL("../../scripts/pi05_client.py", import.meta.url));
const alohaImages = {
  cam_high: "cam_high",
  cam_left_wrist: "cam_left_wrist",
  cam_right_wrist: "cam_right_wrist",
};

/** openpi's ALOHA input/output transforms own joint and gripper conversion.
 * The client sends measured Classic ALOHA qpos and executes absolute joint14 targets.
 */
export function createPi05Policy({
  env = process.env,
  url = env.PI05_URL ?? "ws://127.0.0.1:8000",
  python = env.PI05_PYTHON ?? "python3",
  imageMap = alohaImages,
  statePath = "state.qpos",
  chunkSteps = 8,
  requestTimeoutMs = 30000,
  worker = null,
} = {}) {
  if (!["ws:", "wss:"].includes(new URL(url).protocol)) throw new TypeError("pi0.5 url must use ws:// or wss://");
  if (!Number.isInteger(chunkSteps) || chunkSteps < 1 || chunkSteps > 100) throw new TypeError("chunkSteps must be between 1 and 100");
  let transport = worker;
  let configured = null;
  let closed = false;

  function disconnect() {
    transport?.stop();
    transport = null;
    configured = null;
  }

  async function request(payload, signal) {
    signal?.throwIfAborted();
    if (closed) throw new Error("pi0.5 policy is closed");
    transport ??= new LineWorker("pi05", python, [clientScript], { env, autoRestart: false });
    const abort = () => disconnect();
    signal?.addEventListener("abort", abort, { once: true });
    try {
      const reply = await transport.request(payload, requestTimeoutMs);
      signal?.throwIfAborted();
      if (!reply.ok) throw new Error(reply.error ?? "pi0.5 inference failed");
      return reply.result;
    } catch (error) {
      disconnect();
      signal?.throwIfAborted();
      throw error;
    } finally {
      signal?.removeEventListener("abort", abort);
    }
  }

  async function decide({ observation, instruction, signal }) {
    const images = Object.fromEntries(Object.entries(imageMap).map(([key, camera]) => {
      const image = observation.images?.find(item => item.name === camera);
      if (!image?.data) throw new Error(`pi0.5 camera ${camera} is missing from observation.images`);
      return [key, { data: image.data, mimeType: image.mimeType }];
    }));
    const state = (Array.isArray(statePath) ? statePath : statePath.split(".")).reduce((value, key) => value?.[key], observation);
    if (!Array.isArray(state) || state.length !== 14 || !state.every(Number.isFinite)) {
      throw new TypeError("pi0.5 ALOHA requires 14 measured joint positions at statePath");
    }
    const prompt = [...new Set([observation.text, instruction].filter(value => typeof value === "string" && value.trim()))].join("\n\n");
    if (!prompt) throw new TypeError("pi0.5 requires a task instruction");
    configured ??= request({ op: "configure", url, timeoutMs: requestTimeoutMs }, signal);
    await configured;
    const result = await request({ op: "infer", images, state, prompt }, signal);
    const actions = result.actions;
    if (!Array.isArray(actions) || !actions.length || actions.some(row => !Array.isArray(row) || row.length !== 14 || !row.every(Number.isFinite))) {
      throw new TypeError("pi0.5 ALOHA must return actions with shape [steps, 14]");
    }
    const positions = actions.slice(0, chunkSteps);
    return {
      action: { name: "joint_trajectory", arguments: { positions } },
      model: "pi05",
      vla: { controlSteps: positions.length, predictedSteps: actions.length, serverTiming: result.server_timing ?? null },
    };
  }

  return {
    id: "pi05:aloha", decide,
    // openpi's reset is local; every decision requests a fresh action chunk.
    async reset({ signal } = {}) { signal?.throwIfAborted(); },
    async close() { closed = true; disconnect(); },
  };
}

export default createPi05Policy;
