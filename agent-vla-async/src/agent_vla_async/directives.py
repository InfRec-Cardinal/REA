"""Directive parsing, the depth/lidar/tilt gate and the console client shared by the robot-side mission loops."""

from __future__ import annotations

import asyncio
import json
import math
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


ACTIONS = {"STOP", "FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT"}


@dataclass(frozen=True)
class Directive:
    seq: int
    action: str
    duration_ms: int
    received_at: float
    captured_at: Optional[float] = None
    published_at: Optional[float] = None


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    hard_stop: bool
    reason: str
    narrow: bool = False


def parse_policy_action(raw: Any, received_at: float) -> Directive:
    """Validate one JSON-compatible PI directive.

    The canonical wire shape has ``action`` and ``duration_ms`` at the top
    level.  The benchmark's older ``action: {...}`` result is also accepted.
    """
    if not isinstance(raw, dict) or raw.get("type") != "policy_action":
        raise ValueError("expected type=policy_action")
    body = raw.get("action")
    if isinstance(body, dict):
        action = body.get("action")
        duration_ms = body.get("duration_ms")
    else:
        action = body
        duration_ms = raw.get("duration_ms")
    if action not in ACTIONS:
        raise ValueError("invalid action")
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
        raise ValueError("duration_ms must be an integer")
    if not 0 <= duration_ms <= 1500:
        raise ValueError("duration_ms must be within 0..1500")
    seq = raw.get("seq")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ValueError("seq must be a non-negative integer")
    captured = raw.get("captured_at")
    published = raw.get("published_at")
    return Directive(
        seq=seq,
        action=str(action),
        duration_ms=duration_ms,
        received_at=float(received_at),
        captured_at=float(captured) if isinstance(captured, (int, float)) else None,
        published_at=float(published) if isinstance(published, (int, float)) else None,
    )


def _range(report: Dict[str, Any], name: str) -> Optional[float]:
    value = report.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0.0 else None


def gate_observation(observation: Any, action: str, tilt_limit: float = 0.35) -> GateDecision:
    """Apply the local fail-closed gate without steering around obstacles."""
    if not isinstance(observation, dict):
        return GateDecision(False, True, "observation_unavailable")
    state = observation.get("state")
    depth = observation.get("depth")
    lidar = observation.get("lidar")
    if not isinstance(state, dict) or not state.get("usable") or not state.get("ready"):
        return GateDecision(False, True, "state_unavailable_or_not_ready")
    rpy = state.get("rpy")
    if not isinstance(rpy, (list, tuple)) or len(rpy) < 2:
        return GateDecision(False, True, "state_attitude_unavailable")
    if abs(float(rpy[0])) > tilt_limit or abs(float(rpy[1])) > tilt_limit:
        return GateDecision(False, True, "excessive_tilt")
    for name, report in (("depth", depth), ("lidar", lidar)):
        if not isinstance(report, dict) or not report.get("usable"):
            return GateDecision(False, True, name + "_unavailable")

    center_limits = ((depth, 0.55), (lidar, 0.90))
    if any((value := _range(report, "center_m")) is not None and value < limit
           for report, limit in center_limits):
        return GateDecision(False, True, "center_obstacle")

    sides: List[float] = []
    for report in (depth, lidar):
        for key in ("left_m", "right_m"):
            value = _range(report, key)
            if value is not None:
                sides.append(value)
    if any(value < 0.35 for value in sides):
        return GateDecision(False, True, "side_obstacle")
    narrow = any(0.35 <= value < 0.45 for value in sides)
    if narrow and action != "FORWARD":
        return GateDecision(False, False, "narrow_forward_only", narrow=True)
    return GateDecision(True, False, "narrow" if narrow else "clear", narrow=narrow)


def advance_forward_steps(steps: int, credit_s: float, action: str,
                          lease_s: float, step_s: float) -> tuple:
    """Pure progress update: turns never count as forward steps."""
    if action != "FORWARD":
        return steps, credit_s
    credit_s += lease_s
    while credit_s + 1e-9 >= step_s:
        steps += 1
        credit_s -= step_s
    return steps, credit_s


class ConsoleClient:
    def __init__(self, base_url: str, execute: bool, timeout_s: float = 1.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.execute = execute
        self.timeout_s = timeout_s
        self._lock = asyncio.Lock()

    def _post_blocking(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("ok") is False:
            raise RuntimeError("console rejected " + path)
        return payload

    async def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if not self.execute:
            return {"ok": True, "dry_run": True}
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(None, self._post_blocking, path, body)

    async def drive(self, vx: float, yaw: float, duration: float) -> Dict[str, Any]:
        return await self._post("/api/drive", {
            "vx": vx, "vy": 0.0, "yaw": yaw, "duration": duration,
        })

    async def stop(self) -> Dict[str, Any]:
        return await self._post("/api/stop", {})
