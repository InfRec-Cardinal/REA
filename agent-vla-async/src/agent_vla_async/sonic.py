"""SONIC policy-contract inspection and open-loop action validation.

This module deliberately stops at the policy boundary.  It does not publish
latent actions or create a low-level G1 controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple


SONIC_EMBODIMENT = "UNITREE_G1_SONIC"
SONIC_ACTION_DIMS = {
    "motion_token": 64,
    "left_hand_joints": 7,
    "right_hand_joints": 7,
}
SONIC_STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "projected_gravity": 3,
}
SONIC_ACTION_HORIZON = 40
SONIC_ACTION_RATE_HZ = 50
SONIC_POLICY_RATE_HZ = 2.5
SONIC_MODALITY_CONTRACT = {
    "video": {
        "delta_indices": [0],
        "modality_keys": ["ego_view"],
    },
    "state": {
        "delta_indices": [0],
        "modality_keys": list(SONIC_STATE_DIMS),
    },
    "action": {
        "delta_indices": list(range(SONIC_ACTION_HORIZON)),
        "modality_keys": list(SONIC_ACTION_DIMS),
    },
    "language": {
        "delta_indices": [0],
        "modality_keys": ["annotation.human.task_description"],
    },
}


def canonical_key(key: Any) -> str:
    """Return a modality key without the optional GR00T ``action.`` prefix."""
    value = str(key)
    return value[7:] if value.startswith("action.") else value


def canonical_keys(keys: Iterable[Any]) -> List[str]:
    return [canonical_key(key) for key in keys]


def canonical_modality_key(modality: str, key: Any) -> str:
    value = str(key)
    prefixes = {
        "action": ("action.",),
        "state": ("state.",),
        "video": ("video.", "observation.images."),
        "language": ("language.",),
    }.get(modality, ())
    for prefix in prefixes:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _config_json(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get("as_json")
    return nested if isinstance(nested, Mapping) else value


def action_contract(modalities: Any) -> Tuple[List[str], Any]:
    """Extract canonical action keys and delta indices from a modality reply."""
    if not isinstance(modalities, Mapping):
        raise ValueError("GR00T modality response is not a mapping")
    action = _config_json(modalities.get("action"))
    keys = canonical_keys(action.get("modality_keys", []))
    return keys, action.get("delta_indices")


def inspect_modality_contract(modalities: Any) -> Dict[str, Any]:
    """Compare a policy reply with the exact official SONIC VLA contract."""
    if not isinstance(modalities, Mapping):
        raise ValueError("GR00T modality response is not a mapping")

    reasons: List[str] = []
    checks: Dict[str, Any] = {}
    for modality, expected in SONIC_MODALITY_CONTRACT.items():
        present = modality in modalities and bool(_config_json(modalities.get(modality)))
        config = _config_json(modalities.get(modality))
        keys = [
            canonical_modality_key(modality, key)
            for key in config.get("modality_keys", [])
        ]
        delta_indices = config.get("delta_indices")
        expected_keys = list(expected["modality_keys"])
        expected_delta = list(expected["delta_indices"])
        missing_keys = [key for key in expected_keys if key not in keys]
        unexpected_keys = [key for key in keys if key not in expected_keys]
        keys_match = keys == expected_keys
        delta_match = delta_indices == expected_delta

        if not present:
            reasons.append("missing_%s_modality_config" % modality)
        if missing_keys:
            reasons.append(
                "%s_missing_keys:%s" % (modality, ",".join(missing_keys))
            )
        if unexpected_keys:
            reasons.append(
                "%s_unexpected_keys:%s" % (modality, ",".join(unexpected_keys))
            )
        if not missing_keys and not unexpected_keys and not keys_match:
            reasons.append("%s_key_order_mismatch" % modality)
        if not delta_match:
            reasons.append("%s_delta_indices_mismatch" % modality)

        checks[modality] = {
            "present": present,
            "modality_keys": keys,
            "delta_indices": delta_indices,
            "expected_modality_keys": expected_keys,
            "expected_delta_indices": expected_delta,
            "keys_compatible": keys_match,
            "delta_indices_compatible": delta_match,
            "compatible": bool(present and keys_match and delta_match),
        }

    action_keys = checks["action"]["modality_keys"]
    action_config = _config_json(modalities.get("action"))
    raw_dims = action_config.get("dimensions", action_config.get("modality_dims"))
    dimensions_source = "policy_modality_config"
    if isinstance(raw_dims, Mapping):
        declared_action_dims = {
            canonical_modality_key("action", key): int(value)
            for key, value in raw_dims.items()
        }
    elif isinstance(raw_dims, (list, tuple)) and len(raw_dims) == len(action_keys):
        declared_action_dims = {
            key: int(width) for key, width in zip(action_keys, raw_dims)
        }
    else:
        declared_action_dims = {
            key: SONIC_ACTION_DIMS[key]
            for key in action_keys
            if key in SONIC_ACTION_DIMS
        }
        dimensions_source = "official_modality_key_contract"
    action_dims_compatible = declared_action_dims == SONIC_ACTION_DIMS
    if not action_dims_compatible:
        reasons.append("action_dimensions_mismatch")
    checks["action"]["declared_dimensions"] = declared_action_dims
    checks["action"]["expected_dimensions"] = dict(SONIC_ACTION_DIMS)
    checks["action"]["dimensions_source"] = dimensions_source
    checks["action"]["dimensions_compatible"] = action_dims_compatible
    checks["action"]["compatible"] = bool(
        checks["action"]["compatible"] and action_dims_compatible
    )

    compatible = all(check["compatible"] for check in checks.values())
    if not compatible:
        reasons.append("active_policy_server_is_not_unitree_g1_sonic")
    return {
        "compatible": compatible,
        "modalities": checks,
        "reasons": reasons,
    }


def is_sonic_contract(modalities: Any) -> bool:
    return bool(inspect_modality_contract(modalities)["compatible"])


@dataclass(frozen=True)
class SonicPolicyStatus:
    """JSON-facing status for the SONIC policy half of the stack."""

    active_embodiment: str
    contract_compatible: bool
    action_keys: List[str]
    delta_indices: Any
    modalities: Dict[str, Any]
    reasons: List[str]

    @classmethod
    def from_modalities(cls, modalities: Any) -> "SonicPolicyStatus":
        inspected = inspect_modality_contract(modalities)
        action = inspected["modalities"]["action"]
        keys = action["modality_keys"]
        delta_indices = action["delta_indices"]
        compatible = bool(inspected["compatible"])
        return cls(
            active_embodiment=SONIC_EMBODIMENT if compatible else "OTHER",
            contract_compatible=compatible,
            action_keys=keys,
            delta_indices=delta_indices,
            modalities=inspected["modalities"],
            reasons=inspected["reasons"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "kind": (
                "SONIC_POLICY_AVAILABLE"
                if self.contract_compatible
                else "SONIC_POLICY_UNAVAILABLE"
            ),
            "scope": "POLICY_OPEN_LOOP",
            "active_embodiment": self.active_embodiment,
            "contract_compatible": self.contract_compatible,
            "action_keys": list(self.action_keys),
            "delta_indices": self.delta_indices,
            "modality_contract": self.modalities,
            "expected_modality_contract": SONIC_MODALITY_CONTRACT,
            "expected_action_dims": dict(SONIC_ACTION_DIMS),
            "expected_state_dims": dict(SONIC_STATE_DIMS),
            "action_horizon": SONIC_ACTION_HORIZON,
            "policy_rate_hz": SONIC_POLICY_RATE_HZ,
            "action_rate_hz": SONIC_ACTION_RATE_HZ,
            "physical_execution": False,
            "controller_mode": "not_connected",
            "reasons": list(self.reasons),
        }


def validate_action_chunk(actions: Any) -> Dict[str, Any]:
    """Validate one SONIC VLA chunk without publishing it.

    Accepted field names may be either plain or prefixed with ``action.``.
    The expected tensor layout is ``[..., horizon, feature_dim]``.
    """
    if not isinstance(actions, Mapping):
        raise ValueError("SONIC action response is not a mapping")

    canonical: Dict[str, Any] = {canonical_key(key): value for key, value in actions.items()}
    missing = [key for key in SONIC_ACTION_DIMS if key not in canonical]
    if missing:
        raise ValueError("missing SONIC action fields: %s" % ", ".join(missing))

    import numpy as np

    shapes: Dict[str, List[int]] = {}
    horizons: List[int] = []
    maximums: Dict[str, float] = {}
    for key, width in SONIC_ACTION_DIMS.items():
        array = np.asarray(canonical[key])
        if array.ndim < 2 or int(array.shape[-1]) != width:
            raise ValueError(
                "%s must end in [..., horizon, %d], got %s"
                % (key, width, list(array.shape))
            )
        if not bool(np.isfinite(array).all()):
            raise ValueError("%s contains non-finite values" % key)
        horizons.append(int(array.shape[-2]))
        shapes[key] = list(array.shape)
        maximums[key] = float(np.max(np.abs(array))) if array.size else 0.0
    if len(set(horizons)) != 1:
        raise ValueError("SONIC action fields have different horizons")
    if horizons[0] != SONIC_ACTION_HORIZON:
        raise ValueError(
            "SONIC action horizon must be %d, got %d"
            % (SONIC_ACTION_HORIZON, horizons[0])
        )
    if maximums["motion_token"] > 1.25:
        raise ValueError("SONIC motion_token exceeds the open-loop bound 1.25")
    return {
        "ok": True,
        "kind": "SONIC_ACTION_CHUNK_VALID",
        "scope": "POLICY_OPEN_LOOP",
        "action_shapes": shapes,
        "absolute_max": maximums,
        "horizon": horizons[0],
        "physical_commands_sent": 0,
    }
