"""LeRobot 0.6.1 Robot adapter with native, feature-derived commands."""

import importlib
import importlib.metadata
import math
import time

from .common import image_observation, json_value


def make_robot(options):
    factory_path = options.get("robotFactory")
    if factory_path:
        module_name, function_name = factory_path.split(":", 1)
        factory = getattr(importlib.import_module(module_name), function_name)
        return factory(options.get("factoryOptions", {}))

    config = options.get("robotConfig")
    if not isinstance(config, dict) or not config.get("type"):
        raise ValueError("Provide robotConfig with a LeRobot type, or robotFactory='module:function'")
    import draccus
    from lerobot.robots import RobotConfig, make_robot_from_config

    for module_name in options.get("registrationModules", []):
        importlib.import_module(module_name)
    if config["type"] not in RobotConfig.get_known_choices():
        module_name = "so_follower" if config["type"] in ("so100_follower", "so101_follower") else config["type"]
        importlib.import_module(f"lerobot.robots.{module_name}")
    return make_robot_from_config(draccus.decode(RobotConfig, config))


class LeRobotDriver:
    id = "lerobot"
    timing = "realtime"

    def __init__(self, options):
        self.control_hz = float(options.get("controlHz", 30))
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("controlHz must be positive")
        self.robot = make_robot(options)
        self.features = dict(self.robot.action_features)
        image_features = {
            name: feature for name, feature in self.robot.observation_features.items()
            if isinstance(feature, tuple) and len(feature) == 3
        }
        self.image_fields = list(image_features)
        self.camera_names = [name for name, shape in image_features.items() if shape[2] in (3, 4)]
        units = options.get("units", {})
        properties = {}
        schema_types = {float: "number", int: "integer", bool: "boolean", str: "string"}
        for name, feature_type in self.features.items():
            if feature_type not in schema_types:
                raise ValueError(f"Unsupported non-scalar LeRobot action feature: {name}: {feature_type}")
            description = f"Native LeRobot action field {name}."
            if name in units:
                description += f" Unit: {units[name]}."
            else:
                description += " Uses the robot's configured units; units are not inferred."
            properties[name] = {"type": schema_types[feature_type], "description": description}
        values_schema = {
            "type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False,
        }
        self.actions = [
            {
                "name": "robot_command",
                "description": "Send one native robot command. The SDK receipt reports the command accepted, not motion completion.",
                "parameters": {
                    "type": "object", "properties": {"values": values_schema},
                    "required": ["values"], "additionalProperties": False,
                },
            },
            {
                "name": "robot_trajectory",
                "description": f"Send 1 to 100 native commands in order at {self.control_hz:g} Hz; observe once at the end.",
                "parameters": {
                    "type": "object", "properties": {"commands": {
                        "type": "array", "items": values_schema, "minItems": 1, "maxItems": 100,
                    }},
                    "required": ["commands"], "additionalProperties": False,
                },
            },
        ]
        try:
            sdk_version = importlib.metadata.version("lerobot")
        except importlib.metadata.PackageNotFoundError:
            sdk_version = None
        self.metadata = {
            "robotType": options.get("robotConfig", {}).get("type", type(self.robot).__name__),
            "sdkVersion": sdk_version,
            "actionKeys": list(self.features),
            "cameraNames": self.camera_names,
            "omittedImageFields": [name for name in self.image_fields if name not in self.camera_names],
            "imageEncoding": "RGB/RGBA PNG; depth image fields are omitted",
            "units": dict(units),
            "controlHz": self.control_hz,
            "stopSupported": callable(getattr(self.robot, "stop", None)),
            "resetBehavior": "observe_only",
            "seedSupported": False,
            "calibrateOnConnect": False,
        }

    def _connect(self):
        if not self.robot.is_connected:
            self.robot.connect(calibrate=False)

    def reset(self, task=None, seed=0):
        snapshot = self.observe()
        snapshot["info"].update({"reset": "observe_only", "requestedSeed": seed, "seedApplied": False})
        return snapshot

    def observe(self):
        self._connect()
        observation = self.robot.get_observation()
        cameras = {name: observation[name] for name in self.camera_names if name in observation}
        return {
            "timestampMs": time.time_ns() // 1_000_000,
            "state": json_value({name: value for name, value in observation.items() if name not in self.image_fields}),
            "images": image_observation(cameras),
            "terminated": False,
            "truncated": False,
            "success": None,
            "reward": None,
            "info": {},
        }

    def _commands(self, action):
        name = action["name"]
        arguments = action.get("arguments", {})
        if name == "robot_command":
            commands = [arguments["values"]]
        elif name == "robot_trajectory":
            commands = arguments["commands"]
            if not isinstance(commands, list) or not 1 <= len(commands) <= 100:
                raise ValueError("robot_trajectory requires 1 to 100 commands")
        else:
            raise ValueError(f"Unknown LeRobot action: {name}")
        for command in commands:
            if not isinstance(command, dict) or command.keys() != self.features.keys():
                raise ValueError("Each command must contain exactly the native action_features keys")
            for key, value in command.items():
                feature_type = self.features[key]
                valid = type(value) is feature_type
                if feature_type is float:
                    valid = type(value) in (int, float) and math.isfinite(value)
                if not valid:
                    raise ValueError(f"Invalid value for native action field {key}: expected {feature_type.__name__}")
        return [dict(command) for command in commands]

    def step(self, action, execute=False):
        commands = self._commands(action)
        actual_actions = []
        self._connect()
        started = time.monotonic()
        if execute:
            for command in commands:
                receipt = self.robot.send_action(dict(command))
                last_sent = time.monotonic()
                actual_actions.append(json_value(receipt))
                delay = last_sent + 1 / self.control_hz - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        control_elapsed_ms = (time.monotonic() - started) * 1000
        snapshot = self.observe()
        snapshot["info"].update({
            "preview": not execute,
            "executed": bool(execute),
            "completion": "submitted" if execute else "preview",
            "requestedActions": commands,
            "actualActions": actual_actions,
            "controlSteps": len(actual_actions),
            "requestedControlSteps": len(commands),
            "controlPeriodMs": 1000 / self.control_hz,
            "controlElapsedMs": control_elapsed_ms,
        })
        return snapshot

    def stop(self):
        stop = getattr(self.robot, "stop", None)
        if not callable(stop):
            return {"supported": False, "reason": "LeRobot Robot has no universal stop method"}
        if not self.robot.is_connected:
            return {"supported": True, "invoked": False}
        return {"supported": True, "invoked": True, "result": json_value(stop())}

    def close(self):
        if self.robot.is_connected:
            self.robot.disconnect()
