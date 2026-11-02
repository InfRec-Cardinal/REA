import unittest

import numpy as np

from agent_vla_async.policy import GrootPolicy, NavigationCalibration, SonicPolicyClient
from agent_vla_async.protocol import Intent
from agent_vla_async.sonic import (
    SONIC_ACTION_DIMS,
    SONIC_MODALITY_CONTRACT,
    SONIC_STATE_DIMS,
)


def sonic_modalities():
    return {
        modality: {
            "as_json": {
                "delta_indices": list(config["delta_indices"]),
                "modality_keys": list(config["modality_keys"]),
            }
        }
        for modality, config in SONIC_MODALITY_CONTRACT.items()
    }


class CalibrationTest(unittest.TestCase):
    def test_maps_only_configured_prefix_and_clamps(self):
        calibration = NavigationCalibration(
            vx_scale=2.0,
            yaw_scale=2.0,
            max_vx=0.15,
            max_yaw=0.30,
            steps_per_chunk=1,
        )
        pulses = calibration.map([[0.4, 0.0, -0.4], [0.1, 0.0, 0.1]])
        self.assertEqual(1, len(pulses))
        self.assertEqual(0.15, pulses[0].vx)
        self.assertEqual(-0.30, pulses[0].yaw)

    def test_groot_observation_keeps_two_frame_history(self):
        policy = GrootPolicy()
        first = np.zeros((4, 5, 3), dtype=np.uint8)
        second = np.ones((4, 5, 3), dtype=np.uint8)
        initial = policy._observation("observe", first)
        updated = policy._observation("observe", second)
        self.assertEqual((1, 2, 4, 5, 3), initial["video"]["ego_view"].shape)
        self.assertEqual([["observe"]], initial["language"]["annotation.human.task_description"])
        self.assertEqual(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            initial["state"]["left_wrist_eef_9d"][0, 0].tolist(),
        )
        self.assertTrue(np.array_equal(first, updated["video"]["ego_view"][0, 0]))
        self.assertTrue(np.array_equal(second, updated["video"]["ego_view"][0, 1]))

    def test_real_g1_navigation_is_advisory_even_with_a_calibration(self):
        policy = GrootPolicy(calibration=NavigationCalibration())
        calls = []

        def fake_call(payload, timeout_ms=None):
            calls.append((payload["endpoint"], timeout_ms))
            return (
                {"navigate_command": np.asarray([[[0.1, 0.0, 0.0]]], dtype=np.float32)},
                {},
            )

        policy._call = fake_call
        output = policy.infer(
            Intent.create("observe the door", policy_wait_timeout_s=2.7),
            1,
            np.zeros((4, 5, 3), dtype=np.uint8),
            {"verdict": {"allowed": True}},
        )
        self.assertEqual([], output.pulses)
        self.assertFalse(output.summary["direct_actuation"])
        self.assertEqual("advisory_only", output.summary["policy_role"])
        self.assertEqual([("get_action", 2700)], calls)

    def test_sonic_status_reads_contract_without_requesting_actions(self):
        policy = SonicPolicyClient(host="policy", port=5551)
        calls = []

        def fake_call(payload):
            calls.append(payload["endpoint"])
            if payload["endpoint"] == "ping":
                return True
            return sonic_modalities()

        policy._call = fake_call
        status = policy.sonic_status()
        self.assertEqual(["ping", "get_modality_config"], calls)
        self.assertEqual("SONIC_POLICY_AVAILABLE", status["kind"])
        self.assertEqual("tcp://policy:5551", status["policy_endpoint"])
        self.assertFalse(status["physical_execution"])
        self.assertFalse(status["action_requested"])

    def test_open_loop_validation_uses_exact_neutral_sonic_observation(self):
        policy = SonicPolicyClient(host="sonic", port=5551)
        calls = []
        captured_observation = {}

        def fake_call(payload, timeout_ms=None):
            calls.append(payload["endpoint"])
            if payload["endpoint"] == "ping":
                return {"status": "ok"}
            if payload["endpoint"] == "get_modality_config":
                return sonic_modalities()
            captured_observation.update(payload["data"]["observation"])
            return (
                {
                    key: np.zeros((1, 40, width), dtype=np.float32)
                    for key, width in SONIC_ACTION_DIMS.items()
                },
                {"latency": 0.1},
            )

        policy._call = fake_call
        result = policy.validate_open_loop(
            "touch the door", np.zeros((12, 16, 3), dtype=np.uint8)
        )
        self.assertEqual(["ping", "get_modality_config", "get_action"], calls)
        self.assertTrue(result["ok"])
        self.assertEqual("synthetic_neutral_placeholder", result["state_source"])
        self.assertEqual((1, 1, 12, 16, 3), captured_observation["video"]["ego_view"].shape)
        self.assertEqual(set(SONIC_STATE_DIMS), set(captured_observation["state"]))
        self.assertEqual(
            [0.0, 0.0, -1.0],
            captured_observation["state"]["projected_gravity"][0, 0].tolist(),
        )
        self.assertFalse(result["action_publisher_started_by_runtime"])
        self.assertEqual(0, result["physical_commands_sent"])

    def test_open_loop_validation_never_requests_action_for_wrong_contract(self):
        policy = SonicPolicyClient()
        calls = []

        def fake_call(payload, timeout_ms=None):
            calls.append(payload["endpoint"])
            if payload["endpoint"] == "ping":
                return {"status": "ok"}
            return {
                "action": {
                    "delta_indices": [0],
                    "modality_keys": list(SONIC_ACTION_DIMS),
                }
            }

        policy._call = fake_call
        result = policy.validate_open_loop(
            "test", np.zeros((4, 5, 3), dtype=np.uint8)
        )
        self.assertFalse(result["ok"])
        self.assertEqual(["ping", "get_modality_config"], calls)


if __name__ == "__main__":
    unittest.main()
