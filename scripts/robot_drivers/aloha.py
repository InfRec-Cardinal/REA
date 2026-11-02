"""Joint-position adapters for official ACT simulation and classic ROS 1 ALOHA."""

import importlib
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from .common import image_observation, json_value


def _load_sdk(mode, options):
    root_key, env_key = ("actRoot", "ACT_ROOT") if mode == "sim" else ("alohaRoot", "ALOHA_ROOT")
    root = options.get(root_key) or os.environ.get(env_key)
    if not root:
        raise ValueError(f"Set {root_key} or {env_key} to the official source checkout")
    source = Path(root).expanduser().resolve()
    if mode == "real":
        source /= "aloha_scripts"
    sys.path.insert(0, str(source))
    module = importlib.import_module("sim_env" if mode == "sim" else "real_env")
    constants = importlib.import_module("constants")
    import numpy as np
    return SimpleNamespace(module=module, constants=constants, np=np)


class AlohaDriver:
    id = "aloha"

    def __init__(self, options):
        self.options = dict(options)
        self.mode = self.options.get("mode", "sim")
        if self.mode not in ("sim", "real"):
            raise ValueError("ALOHA mode must be sim or real")
        self.timing = "turn-based" if self.mode == "sim" else "realtime"
        self.sdk = self.options.get("_sdk") or _load_sdk(self.mode, self.options)
        self.dt = float(self.sdk.constants.DT)
        defaults = ["top"] if self.mode == "sim" else ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
        camera_names = self.options.get("cameraNames")
        self.camera_names = list(defaults if camera_names is None else camera_names)
        self.image_encoding = self.options.get("imageEncoding", "rgb")
        if self.image_encoding not in ("rgb", "bgr"):
            raise ValueError("imageEncoding must be rgb or bgr")
        self.task_name = self.options.get("taskName", "sim_transfer_cube_scripted")
        self.env = None
        self.timestep = None
        self.closed = False
        self.stopped = False
        self.steps = 0
        self.episode_success = False
        self.max_observed_reward = None
        self._create_env()
        joint_names = list(self.sdk.constants.JOINT_NAMES)
        names = [f"{side}_{name}" for side in ("left", "right") for name in joint_names + ["gripper"]]
        self.metadata = {
            "mode": self.mode,
            "sdk": "tonyzhaozh/act" if self.mode == "sim" else "tonyzhaozh/aloha (classic ROS 1)",
            "controlDt": self.dt,
            "controlHz": 1 / self.dt,
            "jointNames": names,
            "armPositionUnit": "radians",
            "gripperPositionRange": [0, 1],
            "cameraNames": self.camera_names,
            "imageEncoding": self.image_encoding,
        }
        vector = {"type": "array", "minItems": 14, "maxItems": 14, "prefixItems": self._position_specs()}
        self.actions = [
            {"name": "joint_position", "description": "Submit one absolute bilateral joint target for one control period; left 6 arm joints, left gripper, right 6 arm joints, right gripper.",
             "parameters": {"type": "object", "properties": {"positions": vector}, "required": ["positions"], "additionalProperties": False}},
            {"name": "joint_trajectory", "description": "Submit 1 to 100 absolute bilateral joint targets in order, one per control period.",
             "parameters": {"type": "object", "properties": {"positions": {"type": "array", "minItems": 1, "maxItems": 100, "items": vector}}, "required": ["positions"], "additionalProperties": False}},
        ]

    def _create_env(self):
        if self.mode == "sim":
            configs = self.sdk.constants.SIM_TASK_CONFIGS
            if self.task_name not in configs or not self.task_name.startswith(("sim_transfer_cube", "sim_insertion")):
                raise ValueError("taskName must name an ACT sim_transfer_cube or sim_insertion task in SIM_TASK_CONFIGS")
            self.env = self.sdk.module.make_sim_env(self.task_name)
            self.max_steps = int(self.options.get("maxSteps", configs[self.task_name]["episode_len"]))
            self.max_reward = self.env.task.max_reward
            self.env.task.get_observation = lambda physics: {
                "qpos": self.env.task.get_qpos(physics), "qvel": self.env.task.get_qvel(physics)
            }
        else:
            self.env = self.sdk.module.make_real_env(init_node=self.options.get("initNode", True), setup_robots=False)
            self.max_steps = self.options.get("maxSteps")
            self.max_reward = None
            self.env.get_observation = lambda: {
                "qpos": self.env.get_qpos(), "qvel": self.env.get_qvel(), "effort": self.env.get_effort()
            }

    def _position_specs(self):
        specs = []
        if self.mode == "sim":
            limits = json_value(self.env.physics.model.actuator_ctrlrange)
            bounds = [limits[:6], limits[8:14]]
        else:
            bounds = []
            for bot in (self.env.puppet_bot_left, self.env.puppet_bot_right):
                info = bot.arm.group_info
                bounds.append(list(zip(info.joint_lower_limits[:6], info.joint_upper_limits[:6])))
        for arm_bounds in bounds:
            specs.extend({"type": "number", "minimum": float(low), "maximum": float(high)} for low, high in arm_bounds)
            specs.append({"type": "number", "minimum": 0, "maximum": 1})
        return specs

    def _sample_pose(self, seed):
        rng = self.sdk.np.random.RandomState(seed)
        quat = [1, 0, 0, 0]
        if self.task_name.startswith("sim_transfer_cube"):
            return list(rng.uniform([0, .4, .05], [.2, .6, .05])) + quat
        peg = list(rng.uniform([.1, .4, .05], [.2, .6, .05])) + quat
        socket = list(rng.uniform([-.2, .4, .05], [-.1, .6, .05])) + quat
        return peg + socket

    def reset(self, task=None, seed=0):
        if self.closed:
            raise RuntimeError("ALOHA driver is closed")
        if self.mode == "sim":
            requested = task.get("taskName", self.task_name) if isinstance(task, dict) else task or self.task_name
            if requested != self.task_name:
                self._release_env()
                self.task_name = requested
                self._create_env()
            self.sdk.module.BOX_POSE[0] = self._sample_pose(seed)
            self.timestep = self.env.reset()
        else:
            self.timestep = self.env.reset(fake=True)
        self.steps = 0
        self.stopped = False
        self.episode_success = False
        self.max_observed_reward = None
        return self._snapshot({"reset": True, "physicalHome": False})

    def _require_reset(self):
        if self.closed or self.timestep is None:
            raise RuntimeError("Reset the ALOHA driver before observing or stepping")

    def _status(self):
        last = bool(self.timestep.last())
        terminated = last and self.timestep.discount == 0
        truncated = (last and not terminated) or (self.max_steps is not None and self.steps >= self.max_steps and not terminated)
        return terminated, truncated

    def _snapshot(self, extra_info=None):
        if self.mode == "sim":
            observation = self.env.task.get_observation(self.env.physics)
            cameras = {"top": "top", "angle": "angle", "vis": "front_close"}
            frames = {name: self.env.physics.render(height=int(self.options.get("imageHeight", 480)), width=int(self.options.get("imageWidth", 640)), camera_id=cameras.get(name, name)) for name in self.camera_names}
        else:
            observation = self.env.get_observation()
            frames = self.env.get_images() if self.camera_names else {}
            frames = {name: frames[name] for name in self.camera_names}
        if self.image_encoding == "bgr":
            frames = {name: frame[..., ::-1] for name, frame in frames.items()}
        reward = self.timestep.reward
        if self.mode == "sim" and reward is not None and reward >= self.max_reward:
            self.episode_success = True
        terminated, truncated = self._status()
        state = {key: json_value(observation[key]) for key in ("qpos", "qvel", "effort") if key in observation}
        info = {"controlSteps": 0, "totalControlSteps": self.steps, "controlDt": self.dt, "taskName": self.task_name if self.mode == "sim" else None,
                "maxReward": self.max_reward, "lastReward": float(reward) if reward is not None else None, "maxObservedReward": self.max_observed_reward}
        info.update(extra_info or {})
        return {"timestampMs": int(time.time() * 1000), "state": state, "images": image_observation(frames),
                "reward": float(reward) if self.mode == "sim" and reward is not None else None, "terminated": bool(terminated), "truncated": bool(truncated),
                "success": self.episode_success if self.mode == "sim" else None, "info": info}

    def observe(self):
        self._require_reset()
        return self._snapshot()

    def _targets(self, action):
        name = action.get("name")
        if name not in ("joint_position", "joint_trajectory"):
            raise ValueError(f"Unknown ALOHA action: {name}")
        positions = action.get("arguments", {}).get("positions")
        targets = [positions] if name == "joint_position" else positions
        if not isinstance(targets, list) or not 1 <= len(targets) <= 100:
            raise ValueError("positions must contain 1 to 100 joint targets")
        specs = self._position_specs()
        for target in targets:
            if not isinstance(target, list) or len(target) != 14:
                raise ValueError("Each joint target must have 14 positions")
            for index, (value, spec) in enumerate(zip(target, specs)):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Joint position {index} must be a finite number")
                if not spec["minimum"] <= value <= spec["maximum"]:
                    raise ValueError(f"Joint position {index} is outside its SDK limits")
        return targets

    def step(self, action, execute=False):
        self._require_reset()
        targets = self._targets(action)
        if not execute:
            return self._snapshot({"executed": False, "preview": True, "plannedControlSteps": len(targets)})
        if self.options.get("executeEnabled", False) is not True:
            raise ValueError("ALOHA driver was configured for preview")
        if any(self._status()):
            raise RuntimeError("The episode ended; reset before submitting another target")
        submitted = 0
        rewards = []
        self.stopped = False
        for target in targets:
            if self.stopped:
                break
            self.timestep = self.env.step(self.sdk.np.asarray(target, dtype=float))
            self.steps += 1
            submitted += 1
            if self.timestep.reward is not None:
                reward = float(self.timestep.reward)
                rewards.append(reward)
                self.max_observed_reward = reward if self.max_observed_reward is None else max(reward, self.max_observed_reward)
            if self.mode == "sim" and self.timestep.reward is not None and self.timestep.reward >= self.max_reward:
                self.episode_success = True
            if any(self._status()):
                break
        snapshot = self._snapshot({"executed": True, "controlSteps": submitted, "completion": "simulated" if self.mode == "sim" else "submitted"})
        snapshot["reward"] = sum(rewards) if self.mode == "sim" and rewards else None
        return snapshot

    def stop(self):
        self.stopped = True
        return {"supported": False, "pendingTargetsCancelled": True, "physicalStop": False, "message": "The last submitted position target remains active; classic ALOHA supplies no physical stop primitive."}

    def _release_env(self):
        if self.env is not None:
            env, self.env = self.env, None
            if callable(getattr(env, "close", None)):
                env.close()
            elif self.mode == "sim":
                env.physics.free()

    def close(self):
        if self.closed:
            return
        self.stopped = True
        self._release_env()
        if self.mode == "real" and self.options.get("initNode", True) and "rospy" in sys.modules:
            sys.modules["rospy"].signal_shutdown("ALOHA driver closed")
        self.closed = True
