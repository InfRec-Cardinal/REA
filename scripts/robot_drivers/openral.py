"""OpenRAL 0.3 HAL bridge using upstream robot manifests and construction."""

import importlib
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from .common import image_observation, json_value


CALLBACK_REAL_MODULES = {
    "openral_hal.franka_panda_real", "openral_hal.ur_real", "openral_hal.sawyer_real",
    "openral_hal.aloha", "openral_hal.openarm_real", "openral_hal.ros_control",
}
PAYLOADS = {
    "joint_position": "joint_targets", "joint_velocity": "joint_velocities",
    "joint_torque": "joint_torques", "body_twist": "body_twist",
    "cartesian_delta": "cartesian_delta", "cartesian_twist": "cartesian_twist",
    "gripper_position": "gripper", "gripper_binary": "gripper",
    "composite_mode": "composite_mode",
}
SCALAR_MODES = {"gripper_position", "gripper_binary", "composite_mode"}
UNSET = object()


class ObservedRollout:
    def __init__(self, env):
        self.wrapped = env
        self.observation = None
        self.transition = None
        self.next_reset_seed = UNSET
        self.action_layout = []
        self.action_units = "Native simulator/controller action space; no joint, gripper, pixel, or metre conversions are applied."
        self.action_space = getattr(env, "action_space", None)
        if self.action_space is None:
            self.action_space = getattr(getattr(env, "_env", None), "action_space", None)
        self.action_dim = getattr(env, "action_dim", None)
        if self.action_dim is None:
            self.action_dim = getattr(getattr(env, "_env", None), "action_dim", None)
        spaces = [self.action_space] if self.action_space is not None else []
        scene = getattr(env, "scene", None)
        if getattr(env, "_state_layout", None) == "gr1":
            self.action_dim = 29
            offset = 0
            for name, width in (("waist", 3), ("right_arm", 7), ("left_arm", 7), ("right_hand", 6), ("left_hand", 6)):
                self.action_layout.append({"name": "action." + name, "offset": offset, "size": width})
                offset += width
            dictionary = getattr(self.action_space, "spaces", {})
            spaces = [dictionary[item["name"]] for item in self.action_layout] if dictionary else []
            self.action_units = "Native GR1 BASIC controller values, ordered waist(3), right_arm(7), left_arm(7), right_hand(6), left_hand(6)."
        elif getattr(scene, "id", None) in ("aloha_transfer_cube", "aloha_insertion"):
            self.action_dim = 14
            spaces = []
            self.action_layout = [{"name": "left_arm", "offset": 0, "size": 6}, {"name": "left_gripper", "offset": 6, "size": 1},
                                  {"name": "right_arm", "offset": 7, "size": 6}, {"name": "right_gripper", "offset": 13, "size": 1}]
            self.action_units = "Absolute arm joint radians and normalized grippers 0..1. The official ALOHA backend deliberately ignores gym-aloha's nominal Box(-1,1) bounds."
        if self.action_dim is None and getattr(self.action_space, "shape", None) is not None:
            self.action_dim = math.prod(self.action_space.shape)
        if not self.action_dim:
            raise ValueError("OpenRAL scene must expose action_dim or a native Gym Box action_space")
        self.action_bounds = []
        for space in spaces:
            if hasattr(space, "low") and hasattr(space, "high"):
                self.action_bounds.extend(zip(space.low.reshape(-1), space.high.reshape(-1)))

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def reset(self, seed=None):
        if self.next_reset_seed is not UNSET:
            seed = self.next_reset_seed
            self.next_reset_seed = UNSET
        self.observation = self.wrapped.reset(seed=seed)
        self.transition = None
        return self.observation

    def step(self, action):
        self.transition = self.wrapped.step(action)
        self.observation = self.transition.observation
        return self.transition


def _source_paths(root):
    if root:
        for source in sorted((Path(root).expanduser().resolve() / "python").glob("*/src")):
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))


def _load_sdk(root):
    _source_paths(root)
    from openral_core import Action, RobotDescription
    from openral_core.exceptions import ROSEStopRequested
    from openral_hal.resolver import build_hal
    return SimpleNamespace(Action=Action, RobotDescription=RobotDescription, build_hal=build_hal, ROSEStopRequested=ROSEStopRequested)


def list_openral_robots(root):
    """Describe every upstream manifest without connecting hardware or loading assets."""
    _source_paths(root)
    from openral_core import RobotDescription
    result = []
    for path in sorted((Path(root).expanduser().resolve() / "robots").glob("*/robot.yaml")):
        desc = RobotDescription.from_yaml(str(path))
        bare = bool(desc.hal.sim or desc.sim is not None)
        kinematic = bool(desc.hal.sim and "PandaMobileHAL" in desc.hal.sim)
        real = desc.hal.real
        modes = (["sim"] if bare or not real else []) + (["real"] if real else [])
        result.append({
            "robotId": path.parent.name, "name": desc.name, "jointCount": len(desc.joints),
            "robotYaml": str(path), "modes": modes, "bareSim": bare,
            "bareSimKind": "kinematic" if kinematic else "mujoco" if bare else None,
            "sceneAttachment": True, "simEnvYamlRequired": not bare,
            "simEntrypoint": desc.hal.sim, "realEntrypoint": real,
            "realTransportFactoryRequired": bool(real and real.split(":")[0] in CALLBACK_REAL_MODULES),
            "declaredControlModes": [mode.value for mode in desc.capabilities.supported_control_modes],
        })
    return result


def _factory(path, options):
    module, name = path.split(":", 1)
    return getattr(importlib.import_module(module), name)(options)


class OpenRALDriver:
    id = "openral"

    def __init__(self, options):
        self.options = dict(options)
        self.mode = self.options.get("mode", "sim")
        if self.mode not in ("sim", "real"):
            raise ValueError("OpenRAL mode must be sim or real")
        self.timing = "turn-based" if self.mode == "sim" else "realtime"
        self.root = self.options.get("openralRoot") or os.environ.get("OPENRAL_ROOT")
        self.sdk = self.options.get("_sdk") or _load_sdk(self.root)
        self.control_hz = float(self.options.get("controlHz", 20))
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("controlHz must be positive")
        self.scene_yaml = self._path(self.options.get("simEnvYaml"))
        self.scene_rollout = None
        if self.scene_yaml and self.mode != "sim":
            raise ValueError("simEnvYaml requires mode=sim")
        self.camera_names = self.options.get("cameraNames")
        self.connected = False
        self.closed = False
        self.stopped = False
        self.total_steps = 0
        self.tick_index = 0
        self.hal = self._make_hal()
        self.description = self.hal.description
        self.profile = self._profile()
        self.control_modes = self._control_modes()
        command_schema = {
            "type": "object", "properties": {
                "controlMode": {"type": "string", "enum": self.control_modes},
                "values": {"type": "array", "items": {"type": "number"}, "minItems": 1},
                "eeName": {"type": "string"}, "frameId": {"type": "string"},
                "jointNames": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            }, "required": ["controlMode", "values"], "additionalProperties": False,
        }
        self.actions = [
            {"name": "hal_action", "description": "Submit one native OpenRAL control step. values follows the chosen control mode and the joint order in robot metadata. Gripper and composite commands use one value; body twist is [vx,vy,vz,wx,wy,wz].", "parameters": command_schema},
            {"name": "hal_sequence", "description": "Execute every command in order as an OpenRAL horizon=1 action; real hardware commands are paced at controlHz. Capture cameras after the sequence.",
             "parameters": {"type": "object", "properties": {"commands": {"type": "array", "items": command_schema, "minItems": 1, "maxItems": 100}}, "required": ["commands"], "additionalProperties": False}},
        ]
        if self.profile in ("scene", "openarm_real"):
            self.actions.append({"name": "hal_group", "description": "Submit every control surface for one atomic SDK tick. OpenArm joint slots may supply compact values with matching jointNames, or a full padded vector. Gripper eeName is its native joint name. Every command shares the same tick_index and tick_group_size.",
                                 "parameters": {"type": "object", "properties": {"commands": {"type": "array", "items": command_schema, "minItems": 2, "maxItems": 32}}, "required": ["commands"], "additionalProperties": False}})
        if self.scene_rollout is not None:
            vector = self._scene_vector_schema()
            self.actions = [
                {"name": "scene_action", "description": "Execute one vector in the scene's native action space. Follow scene controller units and bounds; this vector is not assumed to be robot joint angles.",
                 "parameters": {"type": "object", "properties": {"values": vector}, "required": ["values"], "additionalProperties": False}},
                {"name": "scene_trajectory", "description": "Execute 1 to 100 native scene vectors in order, stopping on the SDK's terminal or truncated result.",
                 "parameters": {"type": "object", "properties": {"commands": {"type": "array", "items": vector, "minItems": 1, "maxItems": 100}}, "required": ["commands"], "additionalProperties": False}},
            ]
        self.metadata = {
            "robotId": self.options.get("robotId", self.description.name), "name": self.description.name,
            "sdk": "OpenRAL", "profile": self.profile, "mode": self.mode,
            "halClass": f"{type(self.hal).__module__}:{type(self.hal).__name__}",
            "jointNames": [joint.name for joint in self.description.joints],
            "joints": [{"name": joint.name, "type": _value(joint.joint_type), "positionLimits": json_value(joint.position_limits), "role": getattr(joint, "role", None)} for joint in self.description.joints],
            "controlModes": self.control_modes, "controlHz": self.control_hz,
            "controlUnits": {"joint_position": "Native HAL joint units: revolute radians, prismatic metres; gripper conventions are adapter-specific.",
                             "gripper_position": "Native HAL gripper units; OpenArm group slots use the named joint's units and limits, SO/Galaxea use normalized 0..1.",
                             "body_twist": "[vx,vy,vz,wx,wy,wz], metres/second then radians/second in the robot body frame.",
                             "cartesian_delta": "Native scene controller units; normalized OSC deltas are not converted to metres."},
            "simGrippers": [{"joint": gripper.joint, "readMode": _value(gripper.read_mode), "writeMode": _value(gripper.write_mode), "ctrlRange": json_value(gripper.ctrl_range)} for gripper in getattr(getattr(self.description, "sim", None), "grippers", [])],
            "sceneJointPositionNames": [joint.name for joint in self.description.joints if getattr(joint, "role", None) != "gripper"] if self.profile == "scene" else None,
            "cameraNames": list(self.camera_names or []), "resetBehavior": "observe_only",
            "seedSupported": False, "taskSignals": "SDK task_success when available; HAL has no episode reward or done contract",
        }
        if self.scene_rollout is not None:
            self.metadata.update({"sceneActionDimension": self.scene_rollout.action_dim, "controlModes": [],
                                  "sceneActionLayout": self.scene_rollout.action_layout, "resetBehavior": "scene_reset", "seedSupported": True,
                                  "sceneStateSource": "OpenRAL SimRollout observation.state", "taskSignals": "Native SimRollout StepResult; success only from SDK success fields or task_success",
                                  "controlUnits": {"scene_action": self.scene_rollout.action_units}})

    def _scene_vector_schema(self):
        dimension = int(self.scene_rollout.action_dim)
        schema = {"type": "array", "minItems": dimension, "maxItems": dimension, "items": {"type": "number"}}
        if self.scene_rollout.action_bounds:
            bounds = []
            for low, high in self.scene_rollout.action_bounds:
                item = {"type": "number"}
                if math.isfinite(float(low)):
                    item["minimum"] = float(low)
                if math.isfinite(float(high)):
                    item["maximum"] = float(high)
                bounds.append(item)
            schema["prefixItems"] = bounds
        return schema

    def _make_hal(self):
        if self.options.get("_hal") is not None:
            return self.options["_hal"]
        if self.options.get("halFactory"):
            return _factory(self.options["halFactory"], self.options.get("factoryOptions", {}))
        path = self._path(self.options.get("robotYaml"))
        if not path:
            if not self.root or not self.options.get("robotId"):
                raise ValueError("Provide openralRoot and robotId, robotYaml, or halFactory")
            path = str(Path(self.root).expanduser() / "robots" / self.options["robotId"] / "robot.yaml")
        description = self.sdk.RobotDescription.from_yaml(str(path))
        entrypoint = description.hal.real
        transport = dict(self.options.get("transport", {}))
        if self.mode == "real" and entrypoint and entrypoint.split(":")[0] in CALLBACK_REAL_MODULES:
            raise ValueError("This OpenRAL real HAL requires halFactory='module:function' wired to actual publisher/state callbacks; JSON transport addresses alone do not connect the robot")
        if self.scene_yaml:
            from openral_hal.sim_bringup import build_sim_env_from_yaml
            from openral_hal.sim_attached import SimAttachedHAL
            env, seed = build_sim_env_from_yaml(self.scene_yaml, robot_id_fallback=description.name)
            try:
                self.scene_rollout = ObservedRollout(env)
                return SimAttachedHAL(self.scene_rollout, description, env_reset_seed=seed, env_action_dim=self.scene_rollout.action_dim)
            except Exception:
                env.close()
                raise
        return self.sdk.build_hal(description, mode=self.mode, transport=transport, sim_env_yaml=self.scene_yaml)

    def _path(self, value):
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute() and self.root:
            path = Path(self.root).expanduser() / path
        return str(path.resolve())

    def _profile(self):
        module = type(self.hal).__module__
        name = type(self.hal).__name__
        if self.scene_yaml or name == "SimAttachedHAL":
            return "scene"
        if name == "PandaMobileHAL":
            return "kinematic"
        if module == "openral_hal.openarm_real":
            return "openarm_real"
        return "mujoco" if self.mode == "sim" else "hardware"

    def _control_modes(self):
        declared = [_value(mode) for mode in self.description.capabilities.supported_control_modes]
        if self.profile == "kinematic":
            return ["joint_position", "body_twist", "cartesian_delta", "gripper_position"]
        if self.profile == "scene":
            return list(dict.fromkeys([mode for mode in declared if mode in PAYLOADS] + ["cartesian_delta", "gripper_position"]))
        if self.profile == "mujoco":
            return ["joint_position"] + (["body_twist"] if "body_twist" in declared else [])
        if self.description.name == "galaxea_a1":
            return ["joint_position", "gripper_position"]
        if self.profile == "openarm_real":
            return ["joint_position", "gripper_position"]
        return ["joint_position"]

    def _connect(self):
        if self.closed:
            raise RuntimeError("OpenRAL driver is closed")
        if not self.connected:
            self.hal.connect()
            self.connected = True

    def reset(self, task=None, seed=0):
        if self.scene_rollout is not None:
            if self.closed:
                raise RuntimeError("OpenRAL driver is closed")
            self.scene_rollout.next_reset_seed = seed
            if self.stopped:
                self.hal.reset_estop()
            self.hal.connect()
            self.connected = True
            self.stopped = False
            self.total_steps = 0
            snapshot = self._scene_snapshot()
            snapshot["info"].update({"reset": "scene_reset", "requestedSeed": seed, "seedApplied": True})
            return snapshot
        snapshot = self.observe()
        snapshot["info"].update({"reset": "observe_only", "requestedSeed": seed, "seedApplied": False})
        return snapshot

    def observe(self):
        self._connect()
        if self.scene_rollout is not None:
            return self._scene_snapshot()
        state = self.hal.read_state()
        read_images = getattr(self.hal, "read_images", None)
        images = read_images() if callable(read_images) and self.camera_names != [] else {}
        if self.camera_names is not None:
            images = {name: images[name] for name in self.camera_names if name in images}
        self.metadata["cameraNames"] = list(images)
        success_reader = getattr(self.hal, "task_success", None)
        success = success_reader() if callable(success_reader) else None
        sim_clock = getattr(self.hal, "sim_time_ns", None)
        sim_time = sim_clock() if callable(sim_clock) else None
        return {
            "timestampMs": time.time_ns() // 1_000_000,
            "state": {"jointNames": list(state.name), "position": json_value(state.position), "velocity": json_value(state.velocity), "effort": json_value(state.effort), "stampNs": state.stamp_ns},
            "images": image_observation(images), "terminated": False, "truncated": False,
            "success": None if success is None else bool(success), "reward": None,
            "info": {"controlSteps": 0, "totalControlSteps": self.total_steps, "simTimeNs": sim_time},
        }

    def _scene_snapshot(self):
        obs = self.scene_rollout.observation or {}
        images = obs.get("images", {})
        if self.camera_names is not None:
            images = {name: images[name] for name in self.camera_names if name in images}
        self.metadata["cameraNames"] = list(images)
        result = self.scene_rollout.transition
        info = dict(result.info) if result is not None else {}
        success = info.get("is_success", info.get("success"))
        if success is None:
            success_reader = getattr(self.hal, "task_success", None)
            success = success_reader() if callable(success_reader) else None
        state = {"observation": json_value(obs.get("state"))}
        for key in ("joint_positions", "joint_velocities"):
            if key in obs:
                state[key] = json_value(obs[key])
        task = getattr(self.scene_rollout, "task", None)
        return {"timestampMs": time.time_ns() // 1_000_000, "text": getattr(task, "instruction", None), "state": state, "images": image_observation(images),
                "terminated": bool(result.terminated) if result is not None else False, "truncated": bool(result.truncated) if result is not None else False,
                "success": None if success is None else bool(success), "reward": float(result.reward) if result is not None and result.reward is not None else None,
                "info": {"controlSteps": 0, "totalControlSteps": self.total_steps, "nativeInfo": json_value(info)}}

    def _scene_step(self, action, execute):
        arguments = action.get("arguments", {})
        commands = [arguments.get("values")] if action["name"] == "scene_action" else arguments.get("commands")
        if not isinstance(commands, list) or not 1 <= len(commands) <= 100:
            raise ValueError("scene commands must contain 1 to 100 vectors")
        schema = self._scene_vector_schema()
        for values in commands:
            if not isinstance(values, list) or len(values) != schema["minItems"]:
                raise ValueError(f"Each native scene command requires {schema['minItems']} values")
            for index, value in enumerate(values):
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("Native scene values must be finite numbers")
                bounds = schema.get("prefixItems", [{}] * len(values))[index]
                if value < bounds.get("minimum", -math.inf) or value > bounds.get("maximum", math.inf):
                    raise ValueError(f"Native scene value {index} is outside action_space bounds")
        self._connect()
        submitted = 0
        rewards = []
        if execute:
            if not self.options.get("executeEnabled", False):
                raise ValueError("OpenRAL driver was configured for preview")
            if self.stopped:
                raise RuntimeError("OpenRAL estop was requested")
            import numpy as np
            for values in commands:
                result = self.scene_rollout.step(np.asarray(values, dtype=np.float32))
                submitted += 1
                self.total_steps += 1
                if result.reward is not None:
                    rewards.append(float(result.reward))
                if result.terminated or result.truncated:
                    break
        snapshot = self._scene_snapshot()
        snapshot["reward"] = sum(rewards) if rewards else None
        snapshot["info"].update({"preview": not execute, "executed": bool(execute), "controlSteps": submitted,
                                  "requestedControlSteps": len(commands), "completion": "simulated" if execute else "preview"})
        return snapshot

    def _command(self, command, group=False):
        mode = command.get("controlMode")
        if mode not in self.control_modes:
            raise ValueError(f"Unsupported OpenRAL control mode: {mode}")
        values = command.get("values")
        if not isinstance(values, list) or not values or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise ValueError("values must be a nonempty finite numeric vector")
        if mode in SCALAR_MODES and len(values) != 1:
            raise ValueError(f"{mode} takes exactly one value")
        if mode in ("body_twist", "cartesian_delta", "cartesian_twist") and len(values) != 6:
            raise ValueError(f"{mode} takes six values")
        if self.profile == "openarm_real" and mode == "gripper_position" and not group:
            raise ValueError("OpenArm gripper_position is supported inside hal_group; a standalone action must contain the full joint vector")
        if self.profile == "openarm_real" and group and mode == "joint_position":
            names = command.get("jointNames")
            if names and len(values) == len(names) and len(values) != len(self.description.joints):
                all_names = [joint.name for joint in self.description.joints]
                padded = [0.0] * len(all_names)
                for name, value in zip(names, values):
                    padded[all_names.index(name)] = value
                values = padded
        if mode == "joint_position" and self.profile not in ("scene", "kinematic") and len(values) != len(self.description.joints):
            raise ValueError(f"joint_position takes {len(self.description.joints)} values in description order")
        payload = {"control_mode": mode, "horizon": 1, PAYLOADS[mode]: values if mode in SCALAR_MODES else [values], "stamp_ns": time.time_ns()}
        for name, field in (("eeName", "ee_name"), ("frameId", "frame_id"), ("jointNames", "joint_names")):
            if name in command:
                payload[field] = command[name]
        return self.sdk.Action(**payload)

    def step(self, action, execute=False):
        name = action.get("name")
        if self.scene_rollout is not None:
            if name not in ("scene_action", "scene_trajectory"):
                raise ValueError("This scene uses scene_action or scene_trajectory with native controller vectors")
            return self._scene_step(action, execute)
        arguments = action.get("arguments", {})
        if name == "hal_action":
            commands = [arguments]
        elif name in ("hal_sequence", "hal_group"):
            commands = arguments.get("commands")
            if not isinstance(commands, list) or not 1 <= len(commands) <= 100:
                raise ValueError("commands must contain 1 to 100 commands")
        else:
            raise ValueError(f"Unknown OpenRAL action: {name}")
        if name == "hal_group" and self.profile not in ("scene", "openarm_real"):
            raise ValueError("This HAL has no atomic action group interface")
        native = [self._command(command, group=name == "hal_group") for command in commands]
        self._connect()
        submitted = 0
        if execute:
            if self.options.get("executeEnabled", False) is not True:
                raise ValueError("OpenRAL driver was configured for preview")
            if self.stopped:
                raise RuntimeError("OpenRAL estop was requested; create a new driver after SDK recovery")
            self.tick_index += 1
            started = time.monotonic()
            for command in native:
                if name == "hal_group":
                    command.tick_index = self.tick_index
                    command.tick_group_size = len(native)
                self.hal.send_action(command)
                submitted += 1
                if self.mode == "real" and name != "hal_group":
                    delay = started + submitted / self.control_hz - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
            self.total_steps += 1 if name == "hal_group" else submitted
        snapshot = self.observe()
        snapshot["info"].update({"preview": not execute, "executed": bool(execute), "completion": "submitted" if execute else "preview",
                                  "controlSteps": (1 if name == "hal_group" else submitted) if submitted else 0,
                                  "submittedActions": submitted, "requestedActions": commands})
        return snapshot

    def stop(self):
        if not self.connected:
            return {"supported": True, "invoked": False}
        self.stopped = True
        try:
            self.hal.estop()
        except self.sdk.ROSEStopRequested as error:
            return {"supported": True, "invoked": True, "sdkException": type(error).__name__, "message": str(error)}
        return {"supported": True, "invoked": True}

    def close(self):
        if self.closed:
            return
        try:
            self.hal.disconnect()
        finally:
            env = getattr(self.hal, "env", None)
            close_env = getattr(env, "close", None)
            if callable(close_env):
                close_env()
            self.connected = False
            self.closed = True


def _value(value):
    return value.value if hasattr(value, "value") else value
