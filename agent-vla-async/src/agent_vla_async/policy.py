"""VLA policy adapters. GR00T inference is remote; execution mapping is optional."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Dict, List, Optional

from .protocol import Intent, MotionPulse, PolicyOutput
from .sonic import SONIC_STATE_DIMS, SonicPolicyStatus, validate_action_chunk


STATE_DIMS = {
    "left_wrist_eef_9d": 9,
    "right_wrist_eef_9d": 9,
    "left_hand": 7,
    "right_hand": 7,
    "left_arm": 7,
    "right_arm": 7,
    "waist": 3,
}

IDENTITY_EEF_9D = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _call_policy_endpoint(
    host: str,
    port: int,
    timeout_ms: int,
    payload: Dict[str, Any],
) -> Any:
    import msgpack
    import msgpack_numpy as mnp
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
    socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
    socket.connect("tcp://%s:%d" % (host, port))
    try:
        socket.send(msgpack.packb(payload, default=mnp.encode))
        response = msgpack.unpackb(socket.recv(), object_hook=mnp.decode, raw=False)
    finally:
        socket.close(linger=0)
        context.term()
    if isinstance(response, dict) and "error" in response:
        raise RuntimeError(str(response["error"]))
    return response


@dataclass(frozen=True)
class NavigationCalibration:
    vx_scale: float = 1.0
    yaw_scale: float = 1.0
    yaw_sign: float = 1.0
    deadband: float = 0.03
    max_vx: float = 0.15
    max_yaw: float = 0.30
    pulse_s: float = 0.30
    steps_per_chunk: int = 1

    @classmethod
    def load(cls, path: str) -> "NavigationCalibration":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def map(self, vectors: Any) -> List[MotionPulse]:
        import numpy as np

        values = np.asarray(vectors, dtype=np.float32)
        while values.ndim > 2:
            values = values[0]
        if values.ndim == 1:
            values = values[None, :]
        pulses: List[MotionPulse] = []
        for vector in values[: self.steps_per_chunk]:
            if vector.size < 3:
                break
            vx = float(vector[0]) * self.vx_scale
            yaw = float(vector[2]) * self.yaw_scale * self.yaw_sign
            vx = 0.0 if abs(vx) < self.deadband else max(-self.max_vx, min(self.max_vx, vx))
            yaw = 0.0 if abs(yaw) < self.deadband else max(-self.max_yaw, min(self.max_yaw, yaw))
            pulses.append(MotionPulse(vx=vx, yaw=yaw, duration_s=self.pulse_s))
        return pulses


class MockPolicy:
    name = "mock-vla"

    def __init__(self, delay_s: float = 0.01):
        self.delay_s = delay_s

    def infer(self, intent: Intent, generation: int, rgb: Any, nav: Dict[str, Any]) -> PolicyOutput:
        time.sleep(self.delay_s)
        return PolicyOutput(
            intent_id=intent.intent_id,
            generation=generation,
            produced_at=time.time(),
            latency_s=self.delay_s,
            policy=self.name,
            summary={"task": intent.task, "source": "deterministic_test_policy"},
            pulses=[MotionPulse(0.08, 0.0, 0.20)],
        )


class SonicPolicyClient:
    """Dedicated SONIC endpoint used for contract and open-loop validation.

    It is intentionally separate from ``GrootPolicy`` so the REAL_G1 advisory
    endpoint can stay on port 5550 while SONIC is probed on port 5551.
    """

    name = "groot-n1.7-unitree-g1-sonic"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5551,
        timeout_ms: int = 15000,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)

    def _call(self, payload: Dict[str, Any], timeout_ms: Optional[int] = None) -> Any:
        return _call_policy_endpoint(
            self.host,
            self.port,
            self.timeout_ms if timeout_ms is None else int(timeout_ms),
            payload,
        )

    def sonic_status(self) -> Dict[str, Any]:
        """Inspect only ping and modality config; never request an action."""
        ping = self._call({"endpoint": "ping"})
        modalities = self._call({"endpoint": "get_modality_config"})
        value = SonicPolicyStatus.from_modalities(modalities).to_dict()
        ping_ready = bool(
            ping is True
            or (
                isinstance(ping, dict)
                and (
                    ping.get("ok") is True
                    or str(ping.get("status", "")).lower() in {"ok", "ready", "success"}
                )
            )
        )
        if not ping_ready:
            value["contract_compatible"] = False
            value["kind"] = "SONIC_POLICY_UNAVAILABLE"
            if "sonic_policy_ping_not_ready" not in value["reasons"]:
                value["reasons"].append("sonic_policy_ping_not_ready")
        value.update(
            {
                "policy_endpoint": "tcp://%s:%d" % (self.host, self.port),
                "policy_ping": ping,
                "policy_ping_ready": ping_ready,
                "policy": self.name,
                "action_requested": False,
            }
        )
        return value

    @staticmethod
    def _neutral_observation(task: str, rgb: Any) -> Dict[str, Any]:
        import numpy as np

        task = task.strip()
        if not task:
            raise ValueError("SONIC validation task must not be empty")
        if rgb is None:
            raise RuntimeError("current RGB frame is required for SONIC validation")
        frame = np.asarray(rgb, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("current RGB frame must have shape [H, W, 3]")
        state = {
            key: np.zeros((1, 1, width), dtype=np.float32)
            for key, width in SONIC_STATE_DIMS.items()
        }
        state["projected_gravity"][0, 0] = [0.0, 0.0, -1.0]
        return {
            "video": {"ego_view": frame[np.newaxis, np.newaxis]},
            "state": state,
            "language": {"annotation.human.task_description": [[task]]},
        }

    def validate_open_loop(self, task: str, rgb: Any) -> Dict[str, Any]:
        """Request one action chunk and validate it without publishing it."""
        status = self.sonic_status()
        if not status.get("contract_compatible", False):
            return {
                "ok": False,
                "kind": "SONIC_OPEN_LOOP_VALIDATION_FAILED",
                "scope": "POLICY_OPEN_LOOP",
                "policy_endpoint": status.get("policy_endpoint"),
                "policy_contract": status,
                "state_source": "synthetic_neutral_placeholder",
                "action_publisher_started_by_runtime": False,
                "controller_called_by_runtime": False,
                "lowcmd_published_by_runtime": False,
                "physical_commands_sent": 0,
                "reasons": list(status.get("reasons", [])),
            }

        observation = self._neutral_observation(task, rgb)
        response = self._call(
            {
                "endpoint": "get_action",
                "data": {"observation": observation, "options": None},
            }
        )
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise RuntimeError("SONIC policy returned an unexpected response")
        actions, info = response
        validation = validate_action_chunk(actions)
        return {
            "ok": True,
            "kind": "SONIC_OPEN_LOOP_VALIDATION_COMPLETE",
            "scope": "POLICY_OPEN_LOOP",
            "policy": self.name,
            "policy_endpoint": "tcp://%s:%d" % (self.host, self.port),
            "policy_contract": status,
            "action_validation": validation,
            "action_shapes": validation["action_shapes"],
            "absolute_max": validation["absolute_max"],
            "state_source": "synthetic_neutral_placeholder",
            "state_shapes": {
                key: [1, 1, width] for key, width in SONIC_STATE_DIMS.items()
            },
            "video_source": "current_rgb",
            "info_keys": sorted(info) if isinstance(info, dict) else [],
            "action_publisher_started_by_runtime": False,
            "controller_called_by_runtime": False,
            "lowcmd_published_by_runtime": False,
            "physical_commands_sent": 0,
        }


class GrootPolicy:
    name = "groot-n1.7-real-g1"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5550,
        timeout_ms: int = 15000,
        calibration: Optional[NavigationCalibration] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.calibration = calibration
        self._previous_rgb: Any = None

    def _call(self, payload: Dict[str, Any], timeout_ms: Optional[int] = None) -> Any:
        effective_timeout_ms = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        return _call_policy_endpoint(
            self.host,
            self.port,
            effective_timeout_ms,
            payload,
        )

    def _observation(self, task: str, rgb: Any) -> Dict[str, Any]:
        import numpy as np

        if rgb is None:
            raise RuntimeError("RGB frame is required for GR00T inference")
        current = np.asarray(rgb, dtype=np.uint8)
        previous = current if self._previous_rgb is None else self._previous_rgb
        video = np.stack((previous, current), axis=0)[None]
        self._previous_rgb = current.copy()
        observation: Dict[str, Any] = {
            "video": {"ego_view": video},
            "state": {},
            "language": {
                "annotation.human.task_description": [[task]],
            },
        }
        for key, size in STATE_DIMS.items():
            value = IDENTITY_EEF_9D if key.endswith("wrist_eef_9d") else [0.0] * size
            observation["state"][key] = np.asarray(value, dtype=np.float32)[None, None]
        return observation

    @staticmethod
    def _shape_summary(actions: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for key, value in actions.items():
            shape = getattr(value, "shape", None)
            summary[key] = list(shape) if shape is not None else type(value).__name__
        return summary

    def infer(self, intent: Intent, generation: int, rgb: Any, nav: Dict[str, Any]) -> PolicyOutput:
        started = time.monotonic()
        response = self._call(
            {
                "endpoint": "get_action",
                "data": {"observation": self._observation(intent.task, rgb), "options": None},
            },
            timeout_ms=max(1, round(intent.policy_wait_timeout_s * 1000.0)),
        )
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise RuntimeError("GR00T returned an unexpected response")
        actions, info = response
        if not isinstance(actions, dict):
            raise RuntimeError("GR00T action payload is not a mapping")
        nav_vectors = actions.get("navigate_command")
        summary = {
            "action_shapes": self._shape_summary(actions),
            "navigation_mapped": False,
            "direct_actuation": False,
            "policy_role": "advisory_only",
            "calibration_ignored_for_execution": self.calibration is not None,
            "state_source": "neutral_placeholder",
            "gate_allowed": bool(nav.get("verdict", {}).get("allowed")),
            "info_keys": sorted(info) if isinstance(info, dict) else [],
        }
        if nav_vectors is not None:
            import numpy as np

            values = np.asarray(nav_vectors)
            summary["navigate_first"] = values.reshape(-1, values.shape[-1])[0].tolist()
        return PolicyOutput(
            intent_id=intent.intent_id,
            generation=generation,
            produced_at=time.time(),
            latency_s=time.monotonic() - started,
            policy=self.name,
            summary=summary,
            pulses=[],
        )
