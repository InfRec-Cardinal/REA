// A real LingBot-VLA 2.0 server + RPent RoboTwin environment.
// See docs/policies/lingbot-vla.md for the two service startup commands.
export default function ({ env = process.env } = {}) {
  const taskName = env.RPENT_TASK ?? "adjust_bottle";
  const taskConfig = env.RPENT_TASK_CONFIG ?? "demo_clean";
  const seed = Number(env.REA_SEED ?? 0);
  const requestTimeoutMs = Number(env.LINGBOT_VLA_TIMEOUT_MS ?? 120000);

  return {
    version: 1,
    name: "lingbot-vla-robotwin",
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
