"""The hybrid bridge preserves one scene across policy changes and stops on native outcomes."""

import base64
from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


SOURCE = Path(__file__).resolve().parents[1] / "eval/robotwin/hybrid/environment.py"
SPEC = importlib.util.spec_from_file_location("robotwin_hybrid_environment", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def plain(value):
    if isinstance(value, dict):
        return {name: plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value.tolist() if hasattr(value, "tolist") else value


class FakeEnvironment:
    def __init__(self, limit=400, success_after=None):
        self.step_lim = limit
        self.take_action_cnt = 0
        self.eval_success = False
        self.success_after = success_after
        self.actions = []
        self.closed = False

    def get_obs(self):
        return {
            "observation": {
                name: {"rgb": np.full((12, 18, 3), index, dtype=np.uint8)}
                for index, name in enumerate(("head_camera", "left_camera", "right_camera", "front_camera"))
            },
            "endpose": {"left_endpose": [0, 0, 1, 1, 0, 0, 0], "right_endpose": [0, 0, 1, 1, 0, 0, 0],
                        "left_gripper": 1.0, "right_gripper": 1.0},
            "joint_action": {"vector": np.zeros(14), "left_arm": np.zeros(6), "right_arm": np.zeros(6)},
        }

    def take_action(self, command, action_type):
        self.actions.append((command.tolist(), action_type))
        self.take_action_cnt += 1
        self.eval_success = self.success_after is not None and self.take_action_cnt >= self.success_after

    def check_success(self):
        return self.eval_success

    def close_env(self, clear_cache):
        self.closed = clear_cache


class HybridEnvironmentTest(unittest.TestCase):
    def bridge(self, environment):
        bridge = MODULE.RoboTwinEnvironment()
        bridge.env = environment
        bridge.instance = {"task": "adjust_bottle", "instruction": "Keep the original unseen instruction.", "seed": 100001}
        bridge.observation = environment.get_obs()
        bridge.native_step_limit = environment.step_lim
        bridge.plain = plain
        bridge.control = SimpleNamespace(
            robot_state=lambda env, obs: obs["endpose"],
            camera_calibration=lambda obs: {"head_camera": {"image_size_wh": [18, 12]}},
            validate_action=lambda action: action,
        )
        return bridge

    def test_vla_chunk_stops_at_native_success_and_maps_joint_to_qpos(self):
        environment = FakeEnvironment(success_after=2)
        bridge = self.bridge(environment)
        result = bridge.step({"name": "robotwin_vla_chunk", "arguments": {"commands": [[0] * 14] * 5, "actionType": "joint"}})
        self.assertTrue(result["success"])
        self.assertTrue(result["terminated"])
        self.assertFalse(result["truncated"])
        self.assertEqual(result["info"]["execution"]["executedActions"], 2)
        self.assertEqual([kind for command, kind in environment.actions], ["qpos", "qpos"])
        bridge.step({"name": "robotwin_vla_chunk", "arguments": {"commands": [[0] * 14], "actionType": "joint"}})
        self.assertEqual(environment.take_action_cnt, 2)

    def test_code_to_vla_keeps_scene_and_native_limit(self):
        environment = FakeEnvironment(limit=4)
        bridge = self.bridge(environment)
        settling = []

        def execute(env, action, before, settle_max_steps):
            settling.append(settle_max_steps)
            env.take_action(np.zeros(16), "ee")
            return {"status": "target_reached", "actual_state": before}, env.get_obs()

        bridge.control.execute = execute
        bridge.step({"name": "move_gripper", "arguments": {}})
        result = bridge.step({"name": "robotwin_vla_chunk", "arguments": {"commands": [[0] * 16] * 7, "actionType": "ee"}})
        self.assertEqual(settling, [200])
        self.assertEqual(environment.step_lim, 4)
        self.assertEqual(environment.take_action_cnt, 4)
        self.assertEqual(result["info"]["execution"]["executedActions"], 3)
        self.assertTrue(result["truncated"])
        self.assertFalse(result["success"])

    def test_invalid_code_decision_reports_error_without_physics(self):
        environment = FakeEnvironment()
        bridge = self.bridge(environment)
        result = bridge.step({"name": "invalid_output", "arguments": {"error": "NameError: unknown helper"}})
        self.assertEqual(environment.actions, [])
        self.assertEqual(result["info"]["execution"]["status"], "invalid_output")
        self.assertEqual(result["info"]["execution"]["native_action_counter_delta"], 0)
        self.assertEqual(result["info"]["execution"]["error"], "NameError: unknown helper")
        self.assertEqual(bridge.transition, 1)
        self.assertFalse(result["success"])

    def test_recorded_out_of_bounds_target_returns_feedback_then_allows_next_decision(self):
        environment = FakeEnvironment()
        bridge = self.bridge(environment)
        # A rejected first action consumes its decision without moving the robot.
        rejected = {"arm": "left", "position": [0.9608119225189554, 2.7543182725534407, 0.43],
                    "quaternion": [0.7000007, -7.1e-6, -3.8e-6, 0.7141421], "opening": 0}
        message = "Position outside world coordinate bounds [-2,2] metres"
        bridge.control.validate_action = Mock(side_effect=ValueError(message))
        bridge.control.execute = Mock()
        observation = bridge.observation
        result = bridge.step({"name": "move_gripper", "arguments": rejected})
        bridge.control.validate_action.assert_called_once_with(rejected)
        bridge.control.execute.assert_not_called()
        self.assertEqual(result["info"]["execution"]["status"], "invalid_action")
        self.assertEqual(result["info"]["execution"]["error"], message)
        self.assertEqual(result["info"]["execution"]["native_action_counter_delta"], 0)
        self.assertEqual(bridge.transition, 1)
        self.assertIs(bridge.observation, observation)
        self.assertFalse(result["success"])
        self.assertFalse(result["terminated"])
        self.assertFalse(result["truncated"])

        bridge.control.validate_action = lambda action: action
        bridge.control.execute = Mock(return_value=({"status": "target_reached"}, observation))
        corrected = {**rejected, "position": [0.1, 0.1, 0.8]}
        bridge.step({"name": "move_gripper", "arguments": corrected})
        bridge.control.execute.assert_called_once()
        self.assertEqual(bridge.transition, 2)

    def test_simulator_value_error_is_not_classified_as_invalid_policy_target(self):
        bridge = self.bridge(FakeEnvironment())
        bridge.control.execute = Mock(side_effect=ValueError("simulator state failure"))
        with self.assertRaisesRegex(ValueError, "simulator state failure"):
            bridge.step({"name": "move_gripper", "arguments": {}})
        self.assertEqual(bridge.transition, 0)

    def test_reset_preserves_initial_images_instruction_and_router_state(self):
        environment = FakeEnvironment(limit=700)
        bridge = self.bridge(environment)
        control = bridge.control
        bridge.env = None
        modules = {
            "robotwin_subset": SimpleNamespace(make_subset_env=lambda subset, instance, output: (environment, {"seed": instance["seed"]})),
            "robotwin_control": control,
            "observation_io": SimpleNamespace(plain=plain),
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), patch.object(sys, "path", sys.path.copy()):
            result = bridge.reset({"subset": directory, "instance": bridge.instance, "output": directory, "harnessRoot": directory})
            row = json.loads((Path(directory) / "initial-router-input.json").read_text())
            self.assertEqual(row["task_text"], bridge.instance["instruction"])
            self.assertEqual(len(row["images"]), 3)
            self.assertTrue(all(Path(image).is_absolute() and Path(image).exists() for image in row["images"]))
            self.assertEqual(json.loads(row["state_text"])["joint"], [0] * 14)
            self.assertEqual(result["observation"]["state"]["vla"]["state_text"], row["state_text"])
            self.assertIn('"left_gripper": 1.0', row["state_text"])
            self.assertIn('"joint": [0.0', row["state_text"])
            self.assertEqual(result["info"]["nativeStepLimit"], 700)
            self.assertEqual(len(result["observation"]["images"]), 4)
            for encoded in result["observation"]["images"]:
                image = Image.open(io.BytesIO(base64.b64decode(encoded["data"])))
                self.assertEqual(image.size, (18, 12))

    def test_worker_keeps_simulator_logs_off_protocol(self):
        class LoggingEnvironment:
            def dispatch(self, request):
                print("simulator progress")
                return {"closed": True}

            def close(self):
                print("simulator closed")

        output, errors = io.StringIO(), io.StringIO()
        with redirect_stderr(errors):
            MODULE.serve(LoggingEnvironment(), io.StringIO('{"id":"1","op":"close"}\n'), output)
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(replies[1], {"id": "1", "ok": True, "result": {"closed": True}})
        self.assertIn("simulator progress", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
