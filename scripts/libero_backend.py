import argparse
import base64
import io
import json
import os
import sys
from datetime import datetime, timezone


SUITES = (
    "libero_goal_swap",
    "libero_goal_task",
    "libero_object_swap",
    "libero_object_task",
    "libero_spatial_swap",
    "libero_spatial_task",
    "libero_10_swap",
    "libero_10_task",
)
TOKEN_AXES = {
    "W": (0, 1),
    "S": (0, -1),
    "A": (1, 1),
    "D": (1, -1),
    "U": (2, 1),
    "R": (2, -1),
    "I": (3, 1),
    "K": (3, -1),
    "J": (4, 1),
    "L": (4, -1),
    "Q": (5, 1),
    "E": (5, -1),
}
PROPRIO_KEYS = {
    "eefPosition": "robot0_eef_pos",
    "eefQuaternion": "robot0_eef_quat",
    "jointPositions": "robot0_joint_pos",
    "gripperPositions": "robot0_gripper_qpos",
}


def token_action(token, gripper, amplitude=0.5):
    token = str(token).strip().upper()
    action = [0.0] * 7
    if token in TOKEN_AXES:
        axis, direction = TOKEN_AXES[token]
        action[axis] = amplitude * direction
    elif token == "P":
        gripper = 1.0
    elif token == "O":
        gripper = -1.0
    elif token != "X":
        raise ValueError(f"Unknown LIBERO action token: {token}")
    action[-1] = gripper
    return action, gripper


def initial_state_index(seed, state_count):
    if not isinstance(seed, int) or not 1 <= seed <= 100:
        raise ValueError("Seed must be an integer between 1 and 100")
    if seed > 50:
        return None
    if seed > state_count:
        raise ValueError(f"No official initial state for held-out seed {seed}")
    return seed - 1


GRIPPER_METRIC_LENGTH = 0.04


def gripper_state(command_closed, fraction, previous_fraction):
    if previous_fraction is not None and abs(fraction - previous_fraction) > 0.02:
        return "moving"
    if fraction >= 0.85:
        return "open"
    if fraction <= 0.08:
        return "closed"
    return "holding" if command_closed else "partial"


def compact_proprio(observation, command_closed, previous_fraction):
    fraction = min(max(float(observation["robot0_gripper_qpos"][0]) / GRIPPER_METRIC_LENGTH, 0.0), 1.0)
    return {
        "eef_cm": [round(float(value) * 100.0, 1) for value in observation["robot0_eef_pos"]],
        "gripper": gripper_state(command_closed, fraction, previous_fraction),
        "gripper_fraction": round(fraction, 3),
        "gripper_command": "close" if command_closed else "open",
    }


def proprio_observation(observation, command_closed=False, previous_fraction=None):
    """Raw robot-only keys (backward compatible) plus the compact fields the prompt reads.

    Object keys (``*-state``, ``<object>_pos``) are never copied: only PROPRIO_KEYS and the
    compact_proprio listing are exposed.
    """
    proprio = {
        field: observation[key].tolist()
        for field, key in PROPRIO_KEYS.items()
        if key in observation
    }
    if "robot0_eef_pos" in observation and "robot0_gripper_qpos" in observation:
        proprio.update(compact_proprio(observation, command_closed, previous_fraction))
    return proprio


def token_budget(steps, horizon, repeat_steps):
    return {
        "tokensTotal": horizon // repeat_steps,
        "tokensUsed": steps // repeat_steps,
        "tokensLeft": (horizon - steps) // repeat_steps,
        "steps": steps,
        "stepsLeft": max(horizon - steps, 0),
        "horizon": horizon,
        "repeatSteps": repeat_steps,
    }


class LiberoBackend:
    def __init__(self, args):
        self.args = args
        self.env = None
        self.observation = None
        self.task = None
        self.steps = 0
        self.episode = 0
        self.observation_number = 0
        self.gripper = -1.0
        self.success = False
        self.terminated = False
        self.horizon = args.horizon
        self.repeat_steps = args.repeat_steps
        self.amplitude = args.amplitude
        self.initial_state_source = None
        self.initial_state_index = None
        self.previous_gripper_fraction = None

    def catalog(self):
        from libero import benchmark

        suites = []
        for suite_name in SUITES:
            suite = benchmark.get_benchmark(suite_name)()
            suites.append({
                "name": suite_name,
                "family": "long" if suite_name.startswith("libero_10_") else "pro",
                "tasks": [
                    {"index": index, "name": suite.get_task(index).name}
                    for index in range(suite.get_num_tasks())
                ],
            })
        return {"suites": suites}

    def reset(self, request):
        import numpy as np
        from libero import benchmark
        from libero.envs import OffScreenRenderEnv

        suite_name = request["suite"]
        if suite_name not in SUITES:
            raise ValueError(f"Unsupported suite: {suite_name}")
        suite = benchmark.get_benchmark(suite_name)()
        task_index = request["task"]
        if isinstance(task_index, str):
            task_index = suite.get_task_names().index(task_index)
        if not 0 <= task_index < suite.get_num_tasks():
            raise ValueError(f"Task index out of range: {task_index}")
        seed = request["seed"]
        state_index = initial_state_index(seed, 50)
        states = None
        if state_index is not None:
            states = suite.get_task_init_states(task_index)
            state_index = initial_state_index(seed, len(states))
        horizon = request.get("horizon", self.args.horizon)
        repeat_steps = request.get("repeatSteps", self.args.repeat_steps)
        amplitude = request.get("amplitude", self.args.amplitude)
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(repeat_steps, int) or not 1 <= repeat_steps <= horizon:
            raise ValueError("repeatSteps must be a positive integer within the horizon")
        if not 0 < amplitude <= 1:
            raise ValueError("amplitude must be in (0, 1]")
        self.close()
        self.horizon = horizon
        self.repeat_steps = repeat_steps
        self.amplitude = amplitude
        self.env = OffScreenRenderEnv(
            bddl_file_name=suite.get_task_bddl_file_path(task_index),
            controller="OSC_POSE",
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=self.args.camera_size,
            camera_widths=self.args.camera_size,
            camera_depths=False,
            control_freq=20,
            horizon=self.horizon + self.args.settle_steps,
        )
        self.env.seed(seed)
        self.observation = self.env.reset()
        self.initial_state_index = state_index
        if state_index is None:
            self.initial_state_source = "seeded_random_development"
        else:
            self.initial_state_source = "official_pruned_heldout"
            self.observation = self.env.set_init_state(states[state_index])
        self.gripper = -1.0
        self.previous_gripper_fraction = None
        action, _ = token_action("X", self.gripper)
        for settle_step in range(self.args.settle_steps):
            self.observation, _, _, _ = self.env.step(np.asarray(action))
        self.task = {
            "suite": suite_name,
            "index": task_index,
            "name": suite.get_task(task_index).name,
            "language": self.env.language_instruction,
            "seed": seed,
        }
        self.steps = 0
        self.episode += 1
        self.observation_number = 0
        self.success = bool(self.env.check_success())
        self.terminated = self.success
        return self.observe()

    def observe(self):
        from PIL import Image, ImageDraw
        import robosuite.macros as macros

        if self.env is None:
            raise ValueError("Reset an episode before observing")
        camera_size = self.args.camera_size
        title_height = 24
        montage = Image.new("RGB", (camera_size * 2, camera_size + title_height), "#161616")
        draw = ImageDraw.Draw(montage)
        for camera_index, (camera, title) in enumerate((
            ("agentview", "External camera (front)"),
            ("robot0_eye_in_hand", "Wrist camera (L/R mirrored)"),
        )):
            rgb = self.observation[f"{camera}_image"]
            if macros.IMAGE_CONVENTION == "opengl":
                rgb = rgb[::-1]
            image = Image.fromarray(rgb)
            montage.paste(image, (camera_index * camera_size, title_height))
            draw.text((camera_index * camera_size + 6, 6), title, fill="white")
        encoded = io.BytesIO()
        montage.save(encoded, format="JPEG", quality=88)
        self.observation_number += 1
        proprio = proprio_observation(self.observation, self.gripper > 0, self.previous_gripper_fraction)
        if "gripper_fraction" in proprio:
            self.previous_gripper_fraction = proprio["gripper_fraction"]
        return {
            "frame": {
                "imageB64": base64.b64encode(encoded.getvalue()).decode("ascii"),
                "mimeType": "image/jpeg",
                "width": montage.width,
                "height": montage.height,
                "capturedAt": datetime.now(timezone.utc).isoformat(),
                "libero": {
                    "task": self.task,
                    "proprio": proprio,
                    "recentHistory": [],
                    "observationId": f"{self.episode}:{self.observation_number}",
                    "budget": token_budget(self.steps, self.horizon, self.repeat_steps),
                },
            },
            "success": self.success,
            "terminated": self.terminated,
            "steps": self.steps,
            "initialStateSource": self.initial_state_source,
            "initialStateIndex": self.initial_state_index,
            "horizon": self.horizon,
            "repeatSteps": self.repeat_steps,
            "amplitude": self.amplitude,
        }

    def step(self, request):
        import numpy as np

        if self.env is None:
            raise ValueError("Reset an episode before stepping")
        if self.terminated:
            return self.observe()
        repeat_steps = request.get("repeatSteps", self.repeat_steps)
        amplitude = request.get("amplitude", self.amplitude)
        if not isinstance(repeat_steps, int) or not 1 <= repeat_steps <= self.horizon:
            raise ValueError("repeatSteps must be a positive integer within the horizon")
        if not 0 < amplitude <= 1:
            raise ValueError("amplitude must be in (0, 1]")
        action, self.gripper = token_action(request["token"], self.gripper, amplitude)
        for control_step in range(repeat_steps):
            self.observation, _, done, _ = self.env.step(np.asarray(action))
            self.steps += 1
            self.success = self.success or bool(self.env.check_success())
            self.terminated = bool(done) or self.success or self.steps >= self.horizon
            if self.terminated:
                break
        return self.observe()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.observation = None
        return {"closed": True}

    def dispatch(self, request):
        command = request["command"]
        if command == "catalog":
            return self.catalog()
        if command == "reset":
            return self.reset(request)
        if command == "observe":
            return self.observe()
        if command == "step":
            return self.step(request)
        if command == "close":
            return self.close()
        raise ValueError(f"Unknown command: {command}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--repeat-steps", type=int, default=5)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--settle-steps", type=int, default=10)
    args = parser.parse_args()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    if os.environ["MUJOCO_GL"] == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    backend = LiberoBackend(args)
    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                response = {"id": request.get("id"), "ok": True, **backend.dispatch(request)}
            except Exception as error:
                print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
                response = {"id": request.get("id"), "ok": False, "error": str(error)}
            protocol_output.write(json.dumps(response, separators=(",", ":")) + "\n")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
