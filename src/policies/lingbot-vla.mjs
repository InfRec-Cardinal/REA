import { fileURLToPath } from "node:url";

import { LineWorker } from "../workers.mjs";

const clientScript = fileURLToPath(new URL("../../scripts/lingbot_vla_client.py", import.meta.url));
const robotwinImages = {
  "observation.images.cam_high": "head",
  "observation.images.cam_left_wrist": "left_wrist",
  "observation.images.cam_right_wrist": "right_wrist",
};

function readPath(value, path) {
  return (Array.isArray(path) ? path : path.split(".")).reduce((current, key) => current?.[key], value);
}

function vector(value, label) {
  if (!Array.isArray(value) || !value.length || !value.every(Number.isFinite)) {
    throw new TypeError(`${label} must be a nonempty numeric vector`);
  }
  return value;
}

function actionRows(result, keys) {
  const fields = keys.map(key => {
    const value = result[key];
    if (!Array.isArray(value) || !value.length) throw new TypeError(`LingBot response is missing action field ${key}`);
    const rows = Array.isArray(value[0]) ? value : [value];
    for (const row of rows) vector(row, `LingBot ${key} action`);
    if (rows.some(row => row.length !== rows[0].length)) throw new TypeError(`LingBot ${key} action dimensions differ within the chunk`);
    return rows;
  });
  if (fields.some(rows => rows.length !== fields[0].length)) throw new TypeError("LingBot action fields have different chunk lengths");
  return fields[0].map((_, index) => fields.flatMap(rows => rows[index]));
}

/** LingBot-VLA 2.0's official WebSocket protocol, exposed as a regular policy.
 * Defaults match the RoboTwin checkpoint and RPent's qpos action contract.
 * Each decide performs fresh inference; unused predictions aren't replayed after rerouting.
 */
export function createLingBotVlaPolicy({
  env = process.env,
  url = env.LINGBOT_VLA_URL ?? "ws://127.0.0.1:8006",
  python = env.LINGBOT_VLA_PYTHON ?? "python3",
  robot = "robotwin",
  imageMap = robotwinImages,
  stateMap = { "observation.state": "state.robot_state.qpos_target14" },
  actionKeys = ["action"],
  actionName = "rpent_qpos_trajectory",
  actionArgument = "commands",
  actionFormat = "chunk",
  jointNames = null,
  chunkSteps = 8,
  requestTimeoutMs = 30000,
  worker = null,
} = {}) {
  if (!["ws:", "wss:"].includes(new URL(url).protocol)) throw new TypeError("LingBot url must use ws:// or wss://");
  if (!Number.isInteger(chunkSteps) || chunkSteps < 1) throw new TypeError("chunkSteps must be a positive integer");
  if (!(requestTimeoutMs > 0)) throw new TypeError("requestTimeoutMs must be positive");
  if (!["chunk", "single"].includes(actionFormat)) throw new TypeError("actionFormat must be chunk or single");
  if (!Array.isArray(actionKeys) || !actionKeys.length) throw new TypeError("actionKeys must name at least one server output field");
  if (jointNames && (!jointNames.length || new Set(jointNames).size !== jointNames.length)) throw new TypeError("jointNames must contain unique joint names");
  let transport = worker;
  let configured = null;
  let ready = false;
  let closed = false;

  function disconnect() {
    transport?.stop();
    transport = null;
    configured = null;
    ready = false;
  }

  async function request(payload, signal) {
    signal?.throwIfAborted();
    if (closed) throw new Error("LingBot policy is closed");
    transport ??= new LineWorker("lingbot-vla", python, [clientScript], { env, autoRestart: false });
    const abort = () => disconnect();
    signal?.addEventListener("abort", abort, { once: true });
    try {
      const reply = await transport.request(payload, requestTimeoutMs);
      signal?.throwIfAborted();
      if (!reply.ok) throw new Error(reply.error ?? "LingBot inference failed");
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
    configured ??= request({ op: "configure", url, robot, timeoutMs: requestTimeoutMs }, signal);
    await configured;
    ready = false;
    await request({ op: "reset" }, signal);
    ready = true;
  }

  async function decide({ observation, instruction, signal }) {
    signal?.throwIfAborted();
    const images = Object.fromEntries(Object.entries(imageMap).map(([key, camera]) => {
      const image = observation.images?.find(item => item.name === camera);
      if (!image?.data) throw new Error(`LingBot camera ${camera} is missing from observation.images`);
      return [key, { data: image.data, mimeType: image.mimeType }];
    }));
    const states = Object.fromEntries(Object.entries(stateMap).map(([key, path]) => [key, vector(readPath(observation, path), `LingBot state ${key} (${Array.isArray(path) ? path.join(".") : path})`)]));
    // Keep the environment's actual task when the runtime supplies a generic instruction.
    // Controller guidance arrives in instruction, so preserve that guidance as well.
    const task = [...new Set([observation.text, instruction].filter(value => typeof value === "string" && value.trim()))].join("\n\n");
    if (!task) throw new TypeError("LingBot requires a task instruction");
    if (!ready) await reset({ signal });
    const result = await request({ op: "infer", images, states, task }, signal);
    let rows = actionRows(result, actionKeys).slice(0, actionFormat === "single" ? 1 : chunkSteps);
    if (jointNames) {
      if (rows[0].length !== jointNames.length) throw new TypeError("LingBot action dimension must match jointNames");
      rows = rows.map(row => Object.fromEntries(jointNames.map((name, index) => [name, row[index]])));
    }
    return {
      action: { name: actionName, arguments: { [actionArgument]: actionFormat === "single" ? rows[0] : rows } },
      model: "lingbot-vla-2.0",
      vla: { controlSteps: rows.length, serverTiming: result.server_timing ?? null },
    };
  }

  return {
    id: `lingbot-vla:${robot}`, decide, reset,
    async close() { closed = true; disconnect(); },
  };
}
