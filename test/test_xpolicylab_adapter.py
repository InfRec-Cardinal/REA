"""REA as an XPolicyLab policy, end to end against a real `rea serve` process (no simulator, no XPolicyLab checkout).

The XPolicyLab loop is reproduced from its demo deploy.py: reset, then update_obs / get_action per chunk."""
import importlib.util
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
sys.path.insert(0, str(ROOT / "integrations" / "python"))
import rea_client  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover - the adapter requires NumPy, as XPolicyLab does
    np = None

FIXTURE = ROOT / "test" / "fixtures" / "serve-scripted.config.json"
DUAL = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def load_adapter():
    spec = importlib.util.spec_from_file_location("rea_xpolicylab_model", ROOT / "integrations" / "xpolicylab" / "REA" / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation(value):
    return {
        "instruction": "Stack the bowls.",
        "vision": {"head_camera": {"color": np.full((8, 10, 3), 120, dtype=np.uint8)}, "left_camera": {"color": np.zeros((8, 10, 3), dtype=np.float32)}},
        "state": {"left_arm_joint_state": np.full(6, value, dtype=np.float32), "left_ee_joint_state": np.array([0.2], dtype=np.float32),
                  "right_arm_joint_state": np.full(6, -value, dtype=np.float32), "right_ee_joint_state": np.array([0.3], dtype=np.float32)},
    }


class LayoutTest(unittest.TestCase):
    @unittest.skipUnless(np is not None, "NumPy is required")
    def test_keys_follow_xpolicylab(self):
        adapter = load_adapter()
        dual = adapter.robot_layout("joint", DUAL)
        self.assertEqual([item[0] for item in dual], ["left_arm_joint_state", "right_arm_joint_state"])
        single = adapter.robot_layout("joint", {"arm_dim": [7], "ee_dim": [1]})
        self.assertEqual(single, [("arm_joint_state", "joint_state", 7, "ee_joint_state", 1)])
        pose = adapter.robot_layout("ee", DUAL)
        self.assertEqual([(item[0], item[2]) for item in pose], [("left_ee_pose", 7), ("right_ee_pose", 7)])
        specs = adapter.action_specs("joint", dual, 50)
        self.assertEqual([spec["name"] for spec in specs], ["action_chunk", "packed_action_chunk"])
        self.assertEqual(specs[1]["parameters"]["properties"]["commands"]["items"]["minItems"], 14)

    @unittest.skipUnless(np is not None, "NumPy is required")
    def test_observation_and_action_conversion(self):
        adapter = load_adapter()
        layout = adapter.robot_layout("joint", DUAL)
        converted = adapter.to_rea_observation(observation(0.5), layout, "joint", rea_client.encode_image)
        self.assertEqual(converted["text"], "Stack the bowls.")
        self.assertEqual([image["name"] for image in converted["images"]], ["head_camera", "left_camera"])
        self.assertEqual(len(converted["state"]["packed"]), 14)
        self.assertEqual(converted["state"]["packed"][:7], [0.5] * 6 + [np.float32(0.2).item()])
        packed = adapter.to_xpolicylab_actions({"name": "packed_action_chunk", "arguments": {"commands": [list(range(14))]}}, layout)
        self.assertEqual(packed[0]["right_arm_joint_state"].tolist(), [7, 8, 9, 10, 11, 12])
        self.assertEqual(packed[0]["right_ee_joint_state"].dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "13 values, expected 14"):
            adapter.to_xpolicylab_actions({"name": "packed_action_chunk", "arguments": {"commands": [list(range(13))]}}, layout)
        hold = adapter.hold_action(observation(0.5), layout)
        self.assertEqual(hold[0]["left_arm_joint_state"].tolist(), [0.5] * 6)
        self.assertIsNone(adapter.hold_action({"state": {}}, layout))
        self.assertEqual(adapter.success_of({"success": 1}), True)
        self.assertIsNone(adapter.success_of("unknown"))


@unittest.skipUnless(NODE and np is not None, "node and NumPy are required")
class AdapterEndToEndTest(unittest.TestCase):
    def test_xpolicylab_loop_runs_rea_and_holds_the_measured_state_after_rea_finishes(self):
        adapter = load_adapter()
        model = adapter.Model({"rea_root": str(ROOT), "rea_config": str(FIXTURE), "action_type": "joint", "robot_action_dim_info": DUAL,
                               "bench_name": "RoboDojo", "task_name": "stack_bowls", "seed": "0", "step_timeout_s": 60})
        try:
            model.reset()
            model.update_obs(observation(0.0))
            first = model.get_action()
            self.assertEqual(len(first), 1)
            np.testing.assert_allclose(first[0]["left_arm_joint_state"], [0.0, 0.01, 0.02, 0.03, 0.04, 0.05], rtol=1e-6)
            model.update_obs(observation(0.1))
            second = model.get_action()
            self.assertEqual(len(second), 2, "an action_chunk command list becomes the XPolicyLab chunk")
            self.assertAlmostEqual(second[1]["right_ee_joint_state"].tolist()[0], 0.4, places=6)
            # The fixture's budget is two decisions: REA has finished, so the adapter holds the measured state.
            model.update_obs(observation(0.7))
            held = model.get_action()
            self.assertEqual([round(value, 6) for value in held[0]["left_arm_joint_state"].tolist()], [0.7] * 6)
            model.on_trial_end({"success": True})
            # A second episode starts a fresh REA episode in the same process.
            model.reset()
            model.update_obs(observation(0.0))
            self.assertEqual(len(model.get_action()), 1)
        finally:
            model.close()


@unittest.skipUnless(NODE, "node is required")
class ClientTest(unittest.TestCase):
    def test_session_round_trip_and_errors(self):
        with rea_client.ReaSession(FIXTURE, rea_root=ROOT, timeout_s=60) as session:
            command = {"type": "array", "items": {"type": "number"}, "minItems": 14, "maxItems": 14}
            spec = {"name": "packed_action_chunk", "description": "commands",
                    "parameters": {"type": "object", "properties": {"commands": {"type": "array", "items": command, "minItems": 1}},
                                   "required": ["commands"], "additionalProperties": False}}
            named = {"name": "action_chunk", "description": "commands", "parameters": {"type": "object"}}
            self.assertEqual(session.configure(actions=[spec, named])["mode"], "policy")
            with self.assertRaisesRegex(rea_client.ReaError, "send reset"):
                session.step({"text": "x", "state": {}, "images": []})
            session.reset(task="t", seed=1, instruction="Stack the bowls.")
            reply = session.step({"text": "Stack the bowls.", "state": {}, "images": [rea_client.encode_image([[[1, 2, 3]]], "head")]})
            self.assertEqual(reply["action"]["name"], "packed_action_chunk")
            summary = session.end({"success": True})
            self.assertEqual(summary["summary"]["steps"], 1)

    def test_encode_image_falls_back_to_png_for_nested_lists(self):
        image = rea_client.encode_image([[[255, 0, 0], [0, 255, 0]]], "pixel")
        self.assertIn(image["mimeType"], ("image/png", "image/jpeg"))
        self.assertEqual(image["name"], "pixel")
        self.assertTrue(image["data"])


if __name__ == "__main__":
    unittest.main()
