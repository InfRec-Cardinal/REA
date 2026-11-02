"""Adapter for RPent's LIBERO, RoboCasa and RoboTwin environment RPC services."""

import importlib
import math
import os
from pathlib import Path
import sys
import time

from .common import image_observation, json_value


PLATFORMS = {
    "libero": {
        "embodiment": "LIBERO robot (Panda by default; server selects embodiment)",
        "cameras": ["agentview", "robot0_eye_in_hand"],
        "controls": [("rpent", 7, None, "Normalized controller inputs: translation xyz, rotation xyz, gripper. These are not absolute poses.")],
    },
    "robocasa": {
        "embodiment": "PandaOmron",
        "cameras": ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        "controls": [("rpent", 12, None, "PandaOmron composite controller: eef position inputs (3), eef rotation inputs (3), gripper (1), base motion (4), control mode (1). Preserve native controller units.")],
    },
    "robotwin": {
        "embodiment": "RoboTwin aloha-agilex",
        "cameras": ["head", "left_wrist", "right_wrist"],
        "controls": [
            ("rpent_qpos", 14, "qpos", "Left 6 joint positions and gripper, then right 6 joint positions and gripper; joints in radians, grippers normalized 0 closed to 1 open."),
            ("rpent_ee", 16, "ee", "Left world-frame xyz in metres, quaternion qw qx qy qz, gripper; then the same 8 values for the right arm. Grippers normalized 0 closed to 1 open."),
        ],
    },
}


def make_client(options):
    root = options.get("rpentRoot") or os.environ.get("RPENT_ROOT")
    if root:
        sys.path.insert(0, str(Path(root).expanduser().resolve()))
    if options.get("clientFactory"):
        module_name, function_name = options["clientFactory"].split(":", 1)
        return getattr(importlib.import_module(module_name), function_name)(options.get("factoryOptions", {}))
    from rpent.utils.rpc import make_rpc_client
    return make_rpc_client(options["endpoint"])


def _flat(value):
    value = json_value(value)
    if isinstance(value, list):
        return [item for child in value for item in _flat(child)]
    return [] if value is None else [value]


def _state(value):
    if isinstance(value, dict):
        return {
            name: _state(item) for name, item in value.items()
            if not name.endswith(("_image", "_images", "_depth")) and name not in ("images", "frames")
            and not (hasattr(item, "ndim") and item.ndim >= 3)
        }
    if isinstance(value, (list, tuple)):
        return [_state(item) for item in value]
    return json_value(value)


class RPentDriver:
    id = "rpent"
    timing = "turn-based"

    def __init__(self, options):
        self.platform = options.get("robot", "libero")
        if self.platform not in PLATFORMS:
            raise ValueError("RPent implements libero, robocasa and robotwin; Franka and SO-101 are not implemented upstream")
        profile = PLATFORMS[self.platform]
        self.options = dict(options)
        self.timeout = float(options.get("timeoutSeconds", 600 if self.platform == "robotwin" else 120))
        self.client = make_client(options)
        self.server_meta = json_value(self._call("env.get_env_meta"))
        for key, expected in options.get("expectedMeta", {}).items():
            if self.server_meta.get(key) != expected:
                raise ValueError(f"RPent endpoint metadata differs for {key}: expected {expected!r}, got {self.server_meta.get(key)!r}")
        self.camera_names = list(options.get("cameraNames", profile["cameras"]))
        self.actions = []
        self.control_specs = {}
        for prefix, size, action_type, description in profile["controls"]:
            vector = {"type": "array", "minItems": size, "maxItems": size, "items": {"type": "number"}, "description": description}
            for chunk in (False, True):
                name = prefix + ("_trajectory" if chunk else "_action" if prefix == "rpent" else "")
                key = "commands" if chunk else "values"
                schema = {"type": "array", "minItems": 1, "maxItems": 100, "items": vector} if chunk else vector
                self.actions.append({
                    "name": name,
                    "description": ("Execute 1 to 100 native simulation actions; observe at the end. " if chunk else "Execute one native simulation action. ") + description,
                    "parameters": {"type": "object", "properties": {key: schema}, "required": [key], "additionalProperties": False},
                })
                self.control_specs[name] = (size, action_type, chunk)
        self.metadata = {
            "platform": self.platform, "embodiment": profile["embodiment"],
            "server": self.server_meta, "cameraNames": self.camera_names,
            "resetBehavior": "reset configured task; seed is selected when the server starts",
            "resetSeedMode": "exact" if self.platform == "robotwin" else "server controlled; RoboCasa may advance its RNG on repeated resets" if self.platform == "robocasa" else "server controlled",
            "observationStateSource": "latest reset or step result; cameras rendered on observation",
            "chunkExecution": "LIBERO executes the full native chunk and reports per-step terminal flags" if self.platform == "libero" else "stops at native terminal status",
            "stopSupported": False, "serverOwned": bool(options.get("shutdownOnClose", False)),
        }
        self.last_obs = None
        self.last_info = {}
        self.terminated = False
        self.truncated = False
        self.success = None
        self.reward = None
        self.steps = 0
        self.instruction = None
        self.closed = False

    def _call(self, method, args=(), kwargs=None):
        return self.client.call(method, args=args, kwargs=kwargs, timeout_s=self.timeout)

    def reset(self, task=None, seed=0):
        if self.closed:
            raise RuntimeError("RPent driver is closed")
        if "seed" in self.server_meta and seed != self.server_meta["seed"]:
            raise ValueError("RPent env.reset cannot change seed; start an env server with the requested seed")
        requested = task if isinstance(task, dict) else {}
        if task is not None and not isinstance(task, dict):
            requested = {"task" if self.platform == "libero" else "task_name": task}
        for key in ("suite", "task", "task_name", "task_config"):
            if key in requested and str(requested[key]) != str(self.server_meta.get(key)):
                raise ValueError(f"RPent env.reset cannot change {key}; start an env server for that task")
        if "taskName" in requested and requested["taskName"] != self.server_meta.get("task_name"):
            raise ValueError("RPent env.reset cannot change taskName; start an env server for that task")
        result = self._call("env.reset")
        if self.platform == "robocasa":
            self.last_obs, self.last_info = result, {}
        else:
            self.last_obs, self.last_info = result
        self.terminated = self.truncated = False
        self.reward = None
        self.steps = 0
        self.instruction = self._call("env.get_task_language")
        self.success = self._success()
        if self.platform == "robocasa":
            self.terminated = self.success
        return self._snapshot({"reset": True, "seed": seed, "controlSteps": 0})

    def _success(self):
        if self.platform == "robocasa":
            return bool(self._call("env.check_success"))
        if self.platform == "robotwin":
            return bool(self.last_info.get("episode_status", {}).get("eval_success", False))
        return self.terminated

    def _snapshot(self, extra_info=None):
        if self.closed or self.last_obs is None:
            raise RuntimeError("Reset the RPent environment before observing or stepping")
        cameras = {}
        for name in self.camera_names:
            kwargs = {"camera_name": name, "depth": False}
            if self.platform != "robotwin":
                kwargs.update({"height": int(self.options.get("imageHeight", 256)), "width": int(self.options.get("imageWidth", 256))})
            frame = self._call("env.render_camera", kwargs=kwargs)
            cameras[name] = frame[::-1] if self.platform in ("libero", "robocasa") else frame
        state = _state(self.last_obs)
        if self.platform == "robotwin":
            state["robot_state"] = _state(self.last_info.get("robot_state", {}))
            state["episode_status"] = json_value(self.last_info.get("episode_status", {}))
        return {
            "timestampMs": time.time_ns() // 1_000_000,
            "text": self.instruction,
            "state": state, "images": image_observation(cameras),
            "terminated": self.terminated, "truncated": self.truncated,
            "success": self.success, "reward": self.reward,
            "info": {"native": _state(self.last_info), "episodeControlSteps": self.steps, **(extra_info or {})},
        }

    def observe(self):
        return self._snapshot({"stateSource": "last_transition"})

    def _commands(self, action):
        if action.get("name") not in self.control_specs:
            raise ValueError(f"Unknown RPent action: {action.get('name')}")
        size, action_type, chunk = self.control_specs[action["name"]]
        arguments = action.get("arguments", {})
        commands = arguments.get("commands") if chunk else [arguments.get("values")]
        if not isinstance(commands, list) or not 1 <= len(commands) <= 100:
            raise ValueError("RPent trajectory requires 1 to 100 commands")
        for values in commands:
            if not isinstance(values, list) or len(values) != size:
                raise ValueError(f"RPent action requires {size} native values")
            if any(type(value) not in (float, int) or not math.isfinite(value) for value in values):
                raise ValueError("RPent action values must be finite numbers")
        return commands, action_type, chunk

    def _accept_result(self, result):
        if self.platform == "robocasa":
            self.last_obs, reward, done, self.last_info = result
            self.success = self._success()
            self.terminated = self.success
            self.truncated = bool(done) and not self.success
            term, trunc = self.terminated, self.truncated
        else:
            self.last_obs, reward, term, trunc, self.last_info = result
            self.terminated = any(bool(item) for item in _flat(term))
            self.truncated = any(bool(item) for item in _flat(trunc))
            self.success = self._success()
        rewards = _flat(reward)
        self.reward = sum(float(item) for item in rewards) if rewards else None
        return {"rewards": json_value(reward), "terminated": json_value(term), "truncated": json_value(trunc)}

    def step(self, action, execute=False):
        if self.closed or self.last_obs is None:
            raise RuntimeError("Reset the RPent environment before stepping")
        commands, action_type, chunk = self._commands(action)
        if not execute:
            return self._snapshot({"preview": True, "executed": False, "requestedActions": commands, "controlSteps": 0})
        if self.terminated or self.truncated:
            raise RuntimeError("The RPent episode ended; reset before stepping")
        signals = []
        executed = 0
        if self.platform == "robocasa":
            for command in commands:
                signals.append(self._accept_result(self._call("env.step", args=(command,))))
                executed += 1
                if self.terminated or self.truncated:
                    break
            rewards = [float(item) for signal in signals for item in _flat(signal["rewards"])]
            self.reward = sum(rewards) if rewards else None
        else:
            kwargs = {"action_type": action_type} if action_type else {}
            if chunk:
                kwargs["return_all_frames"] = False
            result = self._call("env.chunk_step" if chunk else "env.step", args=(commands if chunk else commands[0],), kwargs=kwargs)
            signals.append(self._accept_result(result))
            executed = int(self.last_info["executed_actions"]) if self.platform == "robotwin" else len(_flat(result[1])) if chunk else 1
        self.steps += executed
        return self._snapshot({
            "preview": False, "executed": True, "completion": "simulated",
            "requestedActions": commands, "requestedControlSteps": len(commands),
            "controlSteps": executed, "nativeSignals": signals,
        })

    def stop(self):
        return {"supported": False, "reason": "RPent env RPC has no stop primitive; turn-based simulation advances only on actions"}

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.options.get("shutdownOnClose", False):
                self._call("shutdown")
        finally:
            self.client.close()
