// Classic ROS1 ALOHA: use the robot host's existing ROS/Interbotix environment.
// The configuration previews commands unless ROBOT_EXECUTE=true is set.
export default function ({ env = process.env } = {}) {
  const kind = env.ALOHA_POLICY ?? "chat";
  if (!["chat", "pi05", "lingbot-vla", "code", "pi05-code", "hybrid"].includes(kind)) {
    throw new Error("ALOHA_POLICY must be chat, pi05, lingbot-vla, code, pi05-code or hybrid");
  }
  const routed = kind === "hybrid" || kind === "pi05-code";
  const routingMode = env.ALOHA_ROUTING_MODE ?? "async";
  if (routed && !["async", "sync"].includes(routingMode)) throw new Error("ALOHA_ROUTING_MODE must be async or sync");
  const codeModel = env.REA_CODE_MODEL ?? env.REA_MODEL;
  const routerModel = env.REA_ROUTER_MODEL ?? env.REA_MODEL;
  if (kind === "chat" && !env.REA_MODEL) throw new Error("Set REA_MODEL for the ALOHA chat policy");
  if ((kind === "code" || routed) && !codeModel) throw new Error("Set REA_CODE_MODEL or REA_MODEL for the ALOHA code policy");
  if (routed && !routerModel) throw new Error("Set REA_ROUTER_MODEL or REA_MODEL for the ALOHA router");
  if (["lingbot-vla", "hybrid"].includes(kind) && !env.LINGBOT_VLA_ROBOT) {
    throw new Error("Set LINGBOT_VLA_ROBOT to the matching ALOHA checkpoint's robot configuration");
  }
  const instruction = env.REA_INSTRUCTION ?? "Complete the bimanual manipulation task using the camera observations and joint positions.";
  const decisionTimeoutMs = Number(env.REA_DECISION_TIMEOUT_MS ?? 120000);
  const chat = {
    type: "chat",
    options: {
      model: env.REA_MODEL,
      ...(env.REA_BASE_URL ? { baseUrl: env.REA_BASE_URL } : {}),
      maxOutputTokens: Number(env.REA_MAX_OUTPUT_TOKENS ?? 2048),
    },
  };
  const lingbot = {
    type: "lingbot-vla",
    options: {
      url: env.LINGBOT_VLA_URL ?? "ws://127.0.0.1:8006",
      python: env.LINGBOT_VLA_PYTHON ?? "python3",
      robot: env.LINGBOT_VLA_ROBOT,
      imageMap: {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_left_wrist": "cam_left_wrist",
        "observation.images.cam_right_wrist": "cam_right_wrist",
      },
      stateMap: { "observation.state": "state.qpos" },
      actionKeys: ["action"],
      actionName: "joint_trajectory",
      actionArgument: "positions",
      actionFormat: "chunk",
      chunkSteps: Number(env.LINGBOT_VLA_CHUNK_STEPS ?? 8),
      requestTimeoutMs: decisionTimeoutMs,
    },
  };
  const pi05 = {
    type: "pi05",
    options: {
      url: env.PI05_URL ?? "ws://127.0.0.1:8000",
      python: env.PI05_PYTHON ?? "python3",
      chunkSteps: Number(env.PI05_CHUNK_STEPS ?? 8),
      requestTimeoutMs: decisionTimeoutMs,
    },
  };
  const code = {
    type: "aloha-code",
    options: {
      model: codeModel,
      ...(env.REA_BASE_URL ? { baseUrl: env.REA_BASE_URL } : {}),
      python: env.REA_CODE_PYTHON ?? "python3",
      executionTimeoutMs: Number(env.REA_CODE_TIMEOUT_MS ?? 10000),
    },
  };
  const controllers = {
    chat, pi05, "lingbot-vla": lingbot, code,
  };
  const systems = {
    mode: routingMode,
    system0: [
      { id: "pi05", kind: "vla", description: env.PI05_DESCRIPTION ?? "pi0.5 checkpoint trained for Classic ALOHA camera observations and absolute joint14 trajectories.", policy: pi05 },
      { id: "lingbot-vla", kind: "vla", description: env.LINGBOT_VLA_DESCRIPTION ?? "LingBot-VLA checkpoint trained for Classic ALOHA camera observations and absolute joint14 trajectories.", policy: lingbot },
      { id: "code", kind: "code", description: "Generate and execute Python calculations using measured joints and images to produce explicit ALOHA joint targets.", policy: code },
    ].filter(({ id }) => kind !== "pi05-code" || id !== "lingbot-vla"),
    // Let System0 dispatch first and complete several blocks before switching.
    // Background selections never preempt an active action.
    system2: {
      type: "module",
      options: {
        module: "../src/policies/aloha-router.mjs",
        exportName: "createAlohaRouterThinker",
        model: routerModel,
        ...(env.REA_BASE_URL ? { baseUrl: env.REA_BASE_URL } : {}),
      },
    },
    thinkingTimeoutMs: decisionTimeoutMs,
    thinkingSchedule: routingMode === "async" ? "after_dispatch" : "before_route",
    thinkEverySteps: Number(env.ALOHA_ROUTE_EVERY_STEPS ?? (routingMode === "async" ? 4 : 1)),
    minControllerSteps: Number(env.ALOHA_MIN_CONTROLLER_STEPS ?? (routingMode === "async" ? 4 : 1)),
    maxAdviceAgeSteps: Number(env.ALOHA_ROUTE_MAX_AGE_STEPS ?? 12),
  };
  return {
    version: 1,
    name: "aloha-real",
    environment: {
      type: "aloha",
      options: {
        mode: "real",
        execute: env.ROBOT_EXECUTE === "true",
        python: env.ALOHA_PYTHON ?? "python3",
        alohaRoot: env.ALOHA_ROOT,
        cameraNames: (env.ALOHA_CAMERAS ?? "cam_high,cam_low,cam_left_wrist,cam_right_wrist").split(",").map(name => name.trim()).filter(Boolean),
        imageEncoding: env.ALOHA_IMAGE_ENCODING ?? "rgb",
        maxSteps: Number(env.ALOHA_MAX_CONTROL_STEPS ?? 1000),
      },
    },
    ...(routed ? { systems } : { policy: controllers[kind] }),
    episodes: [{ id: "aloha-real", task: { instruction }, instruction }],
    budget: {
      maxSteps: Number(env.REA_MAX_STEPS ?? 20),
      maxDurationMs: Number(env.REA_EPISODE_TIMEOUT_MS ?? 900000),
      decisionTimeoutMs,
    },
  };
}
