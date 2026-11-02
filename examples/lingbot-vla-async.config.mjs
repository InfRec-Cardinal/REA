// LingBot-VLA 2.0 as a System0 candidate in the asynchronous System 0/1/2 runtime (docs/async-systems.md),
// against the same real model server + RPent RoboTwin environment as examples/lingbot-vla.config.mjs.
// See docs/policies/lingbot-vla.md for the two service startup commands.
export default function ({ env = process.env } = {}) {
  const taskName = env.RPENT_TASK ?? "adjust_bottle";
  const taskConfig = env.RPENT_TASK_CONFIG ?? "demo_clean";
  const seed = Number(env.REA_SEED ?? 0);
  const requestTimeoutMs = Number(env.LINGBOT_VLA_TIMEOUT_MS ?? 120000);

  return {
    version: 1,
    name: "lingbot-vla-robotwin-async",
    environment: {
      type: "rpent",
      options: {
        robot: "robotwin",
        endpoint: env.RPENT_ENDPOINT ?? "http://127.0.0.1:8803",
        execute: true,
        shutdownOnClose: false,
        expectedMeta: { task_name: taskName, task_config: taskConfig, seed },
      },
    },
    systems: {
      mode: "async",
      system0: [{
        id: "lingbot-vla",
        kind: "vla",
        description: "LingBot-VLA 2.0 RoboTwin policy; absolute dual-arm joint and gripper targets.",
        policy: {
          type: "lingbot-vla",
          options: {
            url: env.LINGBOT_VLA_URL ?? "ws://127.0.0.1:8006",
            python: env.LINGBOT_VLA_PYTHON ?? "python3",
            robot: "robotwin",
            imageMap: {
              "observation.images.cam_high": "head",
              "observation.images.cam_left_wrist": "left_wrist",
              "observation.images.cam_right_wrist": "right_wrist",
            },
            // Same native getters as RoboTwin's joint_action.vector.
            stateMap: { "observation.state": "state.robot_state.qpos_target14" },
            actionKeys: ["action"],
            actionName: "rpent_qpos_trajectory",
            actionArgument: "commands",
            actionFormat: "chunk",
            chunkSteps: Number(env.LINGBOT_VLA_CHUNK_STEPS ?? 8),
            requestTimeoutMs,
          },
        },
      }],
      // The built-in System1 router uses the latest advice, then the previous
      // controller. REA_MODEL enables the background System2 chat agent.
      ...(env.REA_MODEL ? {
        system2: {
          type: "chat",
          options: {
            model: env.REA_MODEL,
            ...(env.REA_BASE_URL ? { baseUrl: env.REA_BASE_URL } : {}),
          },
        },
      } : {}),
      thinkEverySteps: 2,
      maxAdviceAgeSteps: 12,
      thinkingTimeoutMs: 60000,
    },
    episodes: [{
      id: `${taskName}:${seed}`,
      task: { taskName, task_config: taskConfig },
      seed,
      ...(env.REA_INSTRUCTION ? { instruction: env.REA_INSTRUCTION } : {}),
    }],
    budget: {
      maxSteps: Number(env.REA_MAX_STEPS ?? 200),
      maxDurationMs: 600000,
      decisionTimeoutMs: requestTimeoutMs,
    },
  };
}
