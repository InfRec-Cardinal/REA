const chat = { type: "chat", options: {} };
const budget = { maxSteps: 20, maxDurationMs: 900000, decisionTimeoutMs: 60000 };

const presets = {
  demo: {
    description: "Run a visible-target grid and a two-joint fixture with no external dependencies.",
    config: {
      version: 1,
      name: "offline-demo",
      environment: { type: "demo", options: {} },
      policy: { type: "demo", options: {} },
      episodes: [
        { id: "grid", task: "grid", seed: 0 },
        { id: "joints", task: "joints", seed: 0 },
      ],
      budget: { maxSteps: 8, maxDurationMs: 10000, decisionTimeoutMs: 1000 },
    },
  },
  aloha: {
    description: "Control the ACT dual-arm transfer-cube simulator; requires ACT and ALOHA_PYTHON.",
    config: {
      version: 1,
      name: "aloha-transfer-cube",
      environment: { type: "aloha", options: { mode: "sim", execute: true } },
      policy: chat,
      episodes: [{
        id: "transfer-cube:0", task: { taskName: "sim_transfer_cube_scripted" }, seed: 0,
        instruction: "Transfer the cube between the two grippers using the camera observations and joint positions.",
      }],
      budget,
    },
  },
  lerobot: {
    description: "Preview SO-101 follower commands; edit the port and use a calibrated LeRobot setup.",
    config: {
      version: 1,
      name: "so101-preview",
      environment: {
        type: "lerobot",
        options: {
          execute: false,
          robotConfig: {
            type: "so101_follower", id: "so101_follower", port: "/dev/ttyACM0",
            use_degrees: true, cameras: {},
          },
          registrationModules: ["lerobot.robots.so_follower"],
          controlHz: 30,
          units: {
            "shoulder_pan.pos": "degrees", "shoulder_lift.pos": "degrees",
            "elbow_flex.pos": "degrees", "wrist_flex.pos": "degrees",
            "wrist_roll.pos": "degrees", "gripper.pos": "normalized 0 to 100",
          },
        },
      },
      policy: chat,
      episodes: [{ id: "so101-preview", task: "Observe the current robot state and propose a joint command that holds its position.", seed: 0 }],
      budget: { ...budget, maxSteps: 1 },
    },
  },
  openral: {
    description: "Control a Franka Panda MuJoCo HAL; choose another OpenRAL robotId or scene in the config.",
    config: {
      version: 1,
      name: "openral-franka",
      environment: { type: "openral", options: { robotId: "franka_panda", mode: "sim", execute: true, cameraNames: [] } },
      policy: chat,
      episodes: [{ id: "franka-hold", task: "Read the current joint positions and issue one command to hold that position.", seed: 0 }],
      budget: { ...budget, maxSteps: 1 },
    },
  },
  rpent: {
    description: "Connect to a running RPent LIBERO service; also supports RoboCasa and RoboTwin.",
    config: {
      version: 1,
      name: "rpent-libero",
      environment: { type: "rpent", options: { robot: "libero", endpoint: "http://127.0.0.1:8801", execute: true, shutdownOnClose: false } },
      policy: chat,
      episodes: [{ id: "rpent-task:0", seed: 0, instruction: "Complete the task described in the environment observation." }],
      budget,
    },
  },
  libero: {
    description: "Run LIBERO / LIBERO-Pro through scripts/libero_backend.py; requires a Python environment with LIBERO and its assets.",
    config: {
      version: 1,
      name: "libero-long",
      environment: { type: "libero", options: {} },
      policy: chat,
      episodes: [{
        id: "libero_10_task:0:51", seed: 51,
        task: { suite: "libero_10_task", task: 0, horizon: 600, repeatSteps: 5, amplitude: 0.5 },
      }],
      budget: { ...budget, maxSteps: 120, maxDurationMs: 1800000 },
    },
  },
  "arc-agi3": {
    description: "Run the official ARC-AGI-3 SDK offline; replace the gameId with an installed game.",
    config: {
      version: 1,
      name: "arc-agi3-offline",
      environment: { type: "arc-agi3", options: { operationMode: "OFFLINE", saveRecording: true } },
      policy: chat,
      episodes: [{ id: "arc-game:0", task: { gameId: "replace-with-installed-game-id" }, seed: 0 }],
      reuseEnvironment: true,
      budget: { ...budget, maxSteps: 200 },
    },
  },
  gymnasium: {
    description: "Run CartPole through Gymnasium; change environmentId for another registered task.",
    config: {
      version: 1,
      name: "gymnasium-cartpole",
      environment: { type: "gymnasium", options: { environmentId: "CartPole-v1" } },
      policy: chat,
      episodes: [{ id: "CartPole-v1:0", seed: 0, instruction: "Balance the pole for as long as possible using the observed position, velocity and angle." }],
      budget: { ...budget, maxSteps: 200 },
    },
  },
  g1: {
    description: "Preview Unitree G1 actions using the configured camera and executor on the robot gateway.",
    config: {
      version: 1,
      name: "g1-preview",
      environment: { type: "g1", options: { execute: false } },
      policy: chat,
      episodes: [{ id: "g1-preview", task: "Observe the scene and select the stop action.", seed: 0 }],
      budget: { ...budget, maxSteps: 1 },
    },
  },
};

/** Descriptive catalogue only; obtaining a preset never starts a simulator or connects a robot. */
export function listPresets() {
  return Object.entries(presets).map(([id, preset]) => ({
    id,
    description: preset.description,
    environment: preset.config.environment.type,
  }));
}

/** Return an independently editable JSON configuration. */
export function getPreset(id) {
  const preset = Object.hasOwn(presets, id) ? presets[id] : undefined;
  if (!preset) throw new Error(`Unknown preset: ${id}. Available presets: ${Object.keys(presets).join(", ")}`);
  return structuredClone(preset.config);
}
