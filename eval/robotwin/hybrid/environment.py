"""JSON-lines access to one live frozen RoboTwin scene for code and VLA policies."""

import base64
from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
from PIL import Image


ACTIONS = [
    {
        "name": "move_gripper",
        "description": "Move one arm to a world EE pose in metres and WXYZ quaternion; opening is 0 closed, 1 open.",
        "parameters": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "quaternion": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                "opening": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["arm", "position", "quaternion", "opening"],
            "additionalProperties": False,
        },
    },
    {
        "name": "robotwin_vla_chunk",
        "description": "Execute a VLA action chunk, stopping at native task success or the task action limit.",
        "parameters": {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                "actionType": {"type": "string", "enum": ["joint", "ee"]},
            },
            "required": ["commands", "actionType"],
            "additionalProperties": False,
        },
    },
    {
        "name": "invalid_output",
        "description": "Record a failed code-policy decision without moving the robot.",
        "parameters": {
            "type": "object",
            "properties": {"error": {"type": "string"}},
            "additionalProperties": False,
        },
    },
]


class RoboTwinEnvironment:
    def __init__(self):
        self.env = None
        self.instance = None
        self.observation = None
        self.success = False
        self.transition = 0

    def dispatch(self, request):
        operation = request["op"]
        if operation == "reset":
            return self.reset(request)
        if operation == "close":
            self.close()
            return {"closed": True}
        if self.env is None:
            raise RuntimeError("Reset the RoboTwin environment before observing or stepping")
        if operation == "observe":
            return self.result()
        if operation == "step":
            return self.step(request["action"])
        raise ValueError(f"Unknown RoboTwin operation: {operation}")

    def reset(self, request):
        self.close()
        subset = Path(request["subset"]).resolve()
        output = Path(request["output"]).resolve()
        output.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(Path(request["harnessRoot"]).resolve()))
        subset_module = importlib.import_module("robotwin_subset")
        self.control = importlib.import_module("robotwin_control")
        self.plain = importlib.import_module("observation_io").plain
        self.instance = request["instance"]
        self.env, config = subset_module.make_subset_env(subset, self.instance, output)
        self.native_step_limit = int(self.env.step_lim)
        self.transition = 0
        self.success = bool(self.env.eval_success or self.env.check_success())
        self.observation = self.env.get_obs()
        (output / "environment-config.json").write_text(
            json.dumps(self.plain(config), indent=2) + "\n", encoding="utf-8"
        )
        image_directory = output / "initial"
        image_directory.mkdir(exist_ok=True)
        for name, camera in self.observation["observation"].items():
            if "rgb" in camera:
                Image.fromarray(np.asarray(camera["rgb"], dtype=np.uint8)).save(image_directory / f"{name}.png")
        router_input = {
            "instance_id": self.instance.get("instance_id", self.instance.get("episode_id")),
            "task": self.instance["task"],
            "seed": self.instance["seed"],
            "task_text": self.instance["instruction"],
            "images": [str(image_directory / f"{name}.png") for name in ("head_camera", "left_camera", "right_camera")],
            "state_text": self.router_state_text(),
        }
        if "output_contract" in request:
            router_input["output_contract"] = request["output_contract"]
        (output / "initial-router-input.json").write_text(
            json.dumps(router_input, indent=2) + "\n", encoding="utf-8"
        )
        return self.result()

    def router_state_text(self):
        return json.dumps({
            "endpose": self.plain(self.observation["endpose"]),
            "joint": self.plain(self.observation["joint_action"]["vector"]),
        })

    def encoded_observation(self):
        images = []
        for name, camera in self.observation["observation"].items():
            if "rgb" not in camera:
                continue
            image = Image.fromarray(np.asarray(camera["rgb"], dtype=np.uint8))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            images.append({
                "name": name,
                "mimeType": "image/png",
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "width": image.width,
                "height": image.height,
            })
        return {
            "id": f"{self.instance.get('instance_id', self.instance.get('episode_id', self.instance['task']))}:{self.transition}",
            "timestampMs": int(time.time() * 1000),
            "text": self.instance["instruction"],
            "images": images,
            "state": {
                "robot_state": self.plain(self.control.robot_state(self.env, self.observation)),
                "vla": {
                    "endpose": self.plain(self.observation["endpose"]),
                    "joint_action": self.plain(self.observation["joint_action"]),
                    # Keep the trained text representation across the Python/JS boundary.
                    "state_text": self.router_state_text(),
                },
                "calibration": self.plain(self.control.camera_calibration(self.observation)),
            },
            "actions": ACTIONS,
        }

    def result(self, execution=None):
        action_count = int(self.env.take_action_cnt)
        return {
            "observation": self.encoded_observation(),
            "success": self.success,
            "terminated": self.success,
            "truncated": not self.success and action_count >= self.native_step_limit,
            "info": {
                "execution": execution,
                "nativeActionCount": action_count,
                "nativeStepLimit": self.native_step_limit,
            },
        }

    def step(self, action):
        before_count = int(self.env.take_action_cnt)
        if self.success or before_count >= self.native_step_limit:
            return self.result()
        arguments = action["arguments"]
        if action["name"] == "invalid_output":
            execution = {
                "status": "invalid_output",
                "error": arguments.get("error"),
                "actual_state": self.control.robot_state(self.env, self.observation),
            }
        elif action["name"] == "move_gripper":
            before = self.control.robot_state(self.env, self.observation)
            try:
                self.control.validate_action(arguments)
            except ValueError as error:
                # Invalid policy targets consume a decision without advancing physics.
                execution = {"status": "invalid_action", "error": str(error), "actual_state": before}
            else:
                # Keep simulator errors separate from policy validation failures.
                execution, self.observation = self.control.execute(
                    self.env, arguments, before, settle_max_steps=200
                )
                self.success = bool(self.env.eval_success or self.env.check_success())
        elif action["name"] == "robotwin_vla_chunk":
            action_type = arguments["actionType"]
            if action_type not in ("joint", "ee"):
                raise ValueError("VLA actionType must be joint or ee")
            commands = np.asarray(arguments["commands"], dtype=np.float32)
            dimension = 14 if action_type == "joint" else 16
            if commands.ndim != 2 or commands.shape[1] != dimension or not len(commands) or not np.isfinite(commands).all():
                raise ValueError(f"VLA commands must be a nonempty finite array with {dimension} columns")
            executed = 0
            for command in commands:
                if self.success or int(self.env.take_action_cnt) >= self.native_step_limit:
                    break
                self.env.take_action(command, action_type="qpos" if action_type == "joint" else "ee")
                executed += 1
                self.success = bool(self.env.eval_success or self.env.check_success())
            self.observation = self.env.get_obs()
            execution = {
                "status": "success" if self.success else "chunk_executed",
                "actionType": action_type,
                "requestedActions": len(commands),
                "executedActions": executed,
                "actual_state": self.control.robot_state(self.env, self.observation),
            }
        else:
            raise ValueError(f"Unknown RoboTwin action: {action['name']}")
        self.transition += 1
        execution = self.plain(execution)
        execution["native_action_counter_delta"] = int(self.env.take_action_cnt) - before_count
        return self.result(execution)

    def close(self):
        if self.env is not None:
            environment, self.env = self.env, None
            environment.close_env(clear_cache=True)


def serve(environment, input_stream, output_stream):
    print(json.dumps({"event": "ready"}), file=output_stream, flush=True)
    try:
        for line in input_stream:
            request = json.loads(line)
            try:
                with redirect_stdout(sys.stderr):
                    result = environment.dispatch(request)
                reply = {"id": request.get("id"), "ok": True, "result": result}
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                reply = {"id": request.get("id"), "ok": False, "error": str(error), "errorType": type(error).__name__}
            print(json.dumps(reply, allow_nan=False), file=output_stream, flush=True)
            if request["op"] == "close":
                break
    finally:
        with redirect_stdout(sys.stderr):
            environment.close()


def main():
    def terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    serve(RoboTwinEnvironment(), sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
