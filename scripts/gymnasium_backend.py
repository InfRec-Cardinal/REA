import base64
import binascii
from importlib import import_module
from importlib.metadata import version
import json
import math
import os
import struct
import sys
import time
import zlib


def json_value(value):
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def load_sdk(options):
    import gymnasium as gym
    import numpy as np

    for module in options.get("modules", []):
        import_module(module)
    return gym, np, version("gymnasium")


def png_image(frame):
    rows = json_value(frame)
    height = len(rows)
    width = len(rows[0]) if height else 0
    if not width or len(rows[0][0]) not in (3, 4):
        raise ValueError("rgb_array render must return an RGB or RGBA pixel array")
    color_type = 2 if len(rows[0][0]) == 3 else 6
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(channel for pixel in row for channel in pixel) for row in rows)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, color_type, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return {"mimeType": "image/png", "data": base64.b64encode(png).decode("ascii")}


def scalar_schema(low, high, integer=False):
    schema = {"type": "integer" if integer else "number"}
    if math.isfinite(low):
        schema["minimum"] = low
    if math.isfinite(high):
        schema["maximum"] = high
    return schema


def array_schema(low, high, shape, integer=False):
    if not shape:
        return scalar_schema(low, high, integer)
    return {
        "type": "array", "minItems": shape[0], "maxItems": shape[0],
        "prefixItems": [array_schema(low[index], high[index], shape[1:], integer) for index in range(shape[0])],
    }


def map_nested(left, right, operation):
    if isinstance(left, list):
        return [map_nested(a, b, operation) for a, b in zip(left, right)]
    return operation(left, right)


def space_schema(space, spaces):
    if isinstance(space, spaces.Discrete):
        return scalar_schema(int(space.start), int(space.start + space.n - 1), True)
    if isinstance(space, spaces.Box):
        return array_schema(json_value(space.low), json_value(space.high), space.shape, space.dtype.kind in "iu")
    if isinstance(space, spaces.MultiDiscrete):
        low = json_value(space.start)
        high = map_nested(low, json_value(space.nvec), lambda start, count: start + count - 1)
        return array_schema(low, high, space.shape, True)
    if isinstance(space, spaces.MultiBinary):
        def binary_shape(shape):
            if not shape:
                return scalar_schema(0, 1, True)
            return {"type": "array", "minItems": shape[0], "maxItems": shape[0], "items": binary_shape(shape[1:])}
        return binary_shape(space.shape)
    if isinstance(space, spaces.Dict):
        return {
            "type": "object", "properties": {key: space_schema(value, spaces) for key, value in space.spaces.items()},
            "required": list(space.spaces), "additionalProperties": False,
        }
    if isinstance(space, spaces.Tuple):
        return {
            "type": "array", "minItems": len(space.spaces), "maxItems": len(space.spaces),
            "prefixItems": [space_schema(value, spaces) for value in space.spaces],
        }
    raise ValueError(f"Unsupported Gymnasium action space: {type(space).__name__}")


def validate_value(value, schema, path="action"):
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict) or set(value) != set(schema["required"]):
            raise ValueError(f"{path} must contain exactly {schema['required']}")
        for key, item in value.items():
            validate_value(item, schema["properties"][key], f"{path}.{key}")
    elif kind == "array":
        if not isinstance(value, list) or len(value) != schema["minItems"]:
            raise ValueError(f"{path} must have {schema['minItems']} items")
        for index, item in enumerate(value):
            child = schema["prefixItems"][index] if "prefixItems" in schema else schema["items"]
            validate_value(item, child, f"{path}[{index}]")
    else:
        if type(value) not in (int, float) or not math.isfinite(value) or (kind == "integer" and type(value) is not int):
            raise ValueError(f"{path} must be a finite {kind}")
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError(f"{path} is outside its action-space bounds")


def decode_action(value, space, spaces, np):
    if isinstance(space, spaces.Dict):
        return {key: decode_action(value[key], child, spaces, np) for key, child in space.spaces.items()}
    if isinstance(space, spaces.Tuple):
        return tuple(decode_action(item, child, spaces, np) for item, child in zip(value, space.spaces))
    if isinstance(space, (spaces.Box, spaces.MultiDiscrete, spaces.MultiBinary)):
        return np.asarray(value, dtype=space.dtype)
    return value


class GymnasiumBackend:
    def __init__(self, sdk_factory=load_sdk):
        self.sdk_factory = sdk_factory
        self.gym = None
        self.environment = None
        self.observation = None
        self.episode = 0
        self.sequence = 0
        self.finished = False
        self.options = {}

    def configure(self, options):
        if self.gym is not None:
            raise ValueError("Gymnasium backend is already configured")
        self.options = options
        self.gym, self.np, self.sdk_version = self.sdk_factory(options)
        return {"sdkVersion": self.sdk_version}

    def reset(self, request):
        if self.gym is None:
            raise ValueError("Configure the Gymnasium backend before reset")
        task = request.get("task")
        if task is None:
            task = {}
        elif isinstance(task, str):
            task = {"environmentId": task}
        if not isinstance(task, dict):
            raise ValueError("task must be an environmentId string or an options object")
        environment_id = task.get("environmentId", self.options.get("environmentId"))
        if not isinstance(environment_id, str) or not environment_id.strip():
            raise ValueError("environmentId is required")
        seed = request.get("seed", 0)
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a non-negative integer or null")
        make_options = {**self.options.get("makeOptions", {}), **task.get("makeOptions", {})}
        if self.environment is not None:
            self.environment.close()
        self.environment = self.gym.make(environment_id, **make_options)
        self.environment_id = environment_id
        self.seed = seed
        self.action_schema = space_schema(self.environment.action_space, self.gym.spaces)
        raw_observation, info = self.environment.reset(seed=seed, options=task.get("resetOptions"))
        self.episode += 1
        self.sequence = 0
        self.finished = False
        return self.transition(raw_observation, None, False, False, info)

    def transition(self, raw_observation, reward, terminated, truncated, info):
        self.sequence += 1
        self.finished = bool(terminated or truncated)
        native_info = json_value(info)
        images = []
        mode = self.environment.render_mode
        if mode in ("rgb_array", "rgb_array_list"):
            rendered = self.environment.render()
            frames = rendered if mode == "rgb_array_list" else [rendered]
            images = [png_image(frame) for frame in frames if frame is not None]
        self.observation = {
            "id": f"gym:{self.episode}:{self.sequence}",
            "timestampMs": time.time_ns() // 1_000_000,
            "text": f"Gymnasium {self.environment_id}. Choose an action from the environment's action space.",
            "state": {"observation": json_value(raw_observation), "info": native_info},
            "images": images,
            "actions": [{
                "name": "step", "description": "Advance the environment by one action.",
                "parameters": {
                    "type": "object", "properties": {"action": self.action_schema},
                    "required": ["action"], "additionalProperties": False,
                },
            }],
        }
        success = native_info.get("is_success")
        return {
            "observation": self.observation,
            "reward": float(reward) if reward is not None else None,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "success": success if isinstance(success, bool) else None,
            "info": {**native_info, "_gymnasium": {"environmentId": self.environment_id, "seed": self.seed, "sdkVersion": self.sdk_version}},
        }

    def step(self, request):
        if self.observation is None:
            raise ValueError("Reset an episode before stepping")
        if self.finished:
            raise ValueError("Episode finished; reset before stepping")
        if request.get("observationId") != self.observation["id"]:
            raise ValueError("Action observation is stale")
        action = request["action"]
        if not isinstance(action, dict) or action.get("name") != "step":
            raise ValueError("Gymnasium action name must be step")
        arguments = action.get("arguments")
        if not isinstance(arguments, dict) or set(arguments) != {"action"}:
            raise ValueError("step arguments must contain exactly action")
        value = arguments["action"]
        validate_value(value, self.action_schema)
        native_action = decode_action(value, self.environment.action_space, self.gym.spaces, self.np)
        if not self.environment.action_space.contains(native_action):
            raise ValueError("Action is outside the Gymnasium action space")
        return self.transition(*self.environment.step(native_action))

    def close(self):
        if self.environment is not None:
            self.environment.close()
            self.environment = None
        self.observation = None
        return {"closed": True}

    def dispatch(self, request):
        operation = request["op"]
        if operation == "configure":
            return self.configure(request.get("options", {}))
        if operation == "reset":
            return self.reset(request)
        if operation == "observe":
            if self.observation is None:
                raise ValueError("Reset an episode before observing")
            return self.observation
        if operation == "step":
            return self.step(request)
        if operation == "close":
            return self.close()
        raise ValueError(f"Unknown operation: {operation}")


def main():
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    backend = GymnasiumBackend()
    protocol_output.write(json.dumps({"event": "ready"}) + "\n")
    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                response = {"id": request.get("id"), "ok": True, "result": backend.dispatch(request)}
                encoded = json.dumps(response, separators=(",", ":"), allow_nan=False)
            except Exception as error:
                print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
                encoded = json.dumps({"id": request.get("id"), "ok": False, "error": str(error)})
            protocol_output.write(encoded + "\n")
            if request.get("op") == "close":
                break
    finally:
        backend.close()


if __name__ == "__main__":
    main()
