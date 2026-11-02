import unittest

import numpy as np

from agent_vla_async.sonic import (
    SONIC_ACTION_DIMS,
    SONIC_MODALITY_CONTRACT,
    SonicPolicyStatus,
    canonical_keys,
    is_sonic_contract,
    validate_action_chunk,
)


def sonic_modalities(prefixed=False):
    prefix = "action." if prefixed else ""
    return {
        "video": {
            "as_json": {
                "delta_indices": [0],
                "modality_keys": ["ego_view"],
            }
        },
        "state": {
            "as_json": {
                "delta_indices": [0],
                "modality_keys": list(
                    SONIC_MODALITY_CONTRACT["state"]["modality_keys"]
                ),
            }
        },
        "action": {
            "as_json": {
                "delta_indices": list(range(40)),
                "modality_keys": [prefix + key for key in SONIC_ACTION_DIMS],
            }
        },
        "language": {
            "as_json": {
                "delta_indices": [0],
                "modality_keys": ["annotation.human.task_description"],
            }
        },
    }


class SonicContractTest(unittest.TestCase):
    def test_prefixed_and_plain_action_keys_match(self):
        self.assertEqual(
            ["motion_token", "left_hand_joints"],
            canonical_keys(["action.motion_token", "left_hand_joints"]),
        )
        self.assertTrue(is_sonic_contract(sonic_modalities()))
        self.assertTrue(is_sonic_contract(sonic_modalities(prefixed=True)))

    def test_policy_status_is_explicitly_open_loop_only(self):
        status = SonicPolicyStatus.from_modalities(sonic_modalities()).to_dict()
        self.assertEqual("SONIC_POLICY_AVAILABLE", status["kind"])
        self.assertEqual("POLICY_OPEN_LOOP", status["scope"])
        self.assertFalse(status["physical_execution"])
        self.assertEqual("not_connected", status["controller_mode"])

    def test_real_g1_contract_is_not_sonic(self):
        status = SonicPolicyStatus.from_modalities(
            {"action": {"modality_keys": ["navigate_command", "left_arm"]}}
        ).to_dict()
        self.assertEqual("SONIC_POLICY_UNAVAILABLE", status["kind"])
        self.assertIn(
            "active_policy_server_is_not_unitree_g1_sonic", status["reasons"]
        )

    def test_three_action_keys_alone_are_not_a_sonic_contract(self):
        partial = {
            "action": {
                "modality_keys": list(SONIC_ACTION_DIMS),
                "delta_indices": [0],
            }
        }
        status = SonicPolicyStatus.from_modalities(partial).to_dict()
        self.assertFalse(status["contract_compatible"])
        self.assertIn("action_delta_indices_mismatch", status["reasons"])
        self.assertIn("missing_video_modality_config", status["reasons"])
        self.assertIn("missing_state_modality_config", status["reasons"])
        self.assertIn("missing_language_modality_config", status["reasons"])

    def test_exact_contract_rejects_missing_state_key(self):
        modalities = sonic_modalities()
        modalities["state"]["as_json"]["modality_keys"].remove("projected_gravity")
        status = SonicPolicyStatus.from_modalities(modalities).to_dict()
        self.assertFalse(status["contract_compatible"])
        self.assertIn("state_missing_keys:projected_gravity", status["reasons"])

    def test_explicit_action_dimensions_must_match(self):
        modalities = sonic_modalities()
        modalities["action"]["as_json"]["dimensions"] = {
            "motion_token": 63,
            "left_hand_joints": 7,
            "right_hand_joints": 7,
        }
        status = SonicPolicyStatus.from_modalities(modalities).to_dict()
        self.assertFalse(status["contract_compatible"])
        self.assertIn("action_dimensions_mismatch", status["reasons"])

    def test_action_chunk_validates_64_plus_7_plus_7(self):
        actions = {
            "action.motion_token": np.zeros((1, 40, 64), dtype=np.float32),
            "left_hand_joints": np.zeros((1, 40, 7), dtype=np.float32),
            "right_hand_joints": np.zeros((1, 40, 7), dtype=np.float32),
        }
        result = validate_action_chunk(actions)
        self.assertEqual("SONIC_ACTION_CHUNK_VALID", result["kind"])
        self.assertEqual(40, result["horizon"])
        self.assertEqual(0, result["physical_commands_sent"])

    def test_action_chunk_rejects_wrong_token_width(self):
        actions = {
            "motion_token": np.zeros((1, 40, 63), dtype=np.float32),
            "left_hand_joints": np.zeros((1, 40, 7), dtype=np.float32),
            "right_hand_joints": np.zeros((1, 40, 7), dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "motion_token"):
            validate_action_chunk(actions)

    def test_action_chunk_rejects_non_finite_values(self):
        actions = {
            "motion_token": np.full((1, 40, 64), np.nan, dtype=np.float32),
            "left_hand_joints": np.zeros((1, 40, 7), dtype=np.float32),
            "right_hand_joints": np.zeros((1, 40, 7), dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_action_chunk(actions)


if __name__ == "__main__":
    unittest.main()
