"""Continuous G1 mission counted from alternating physical gait phases."""

from __future__ import annotations

import os
import argparse
import asyncio
import json
import math
import queue
import signal
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .directives import ConsoleClient, GateDecision, parse_policy_action


# Extra sys.path entries for the Unitree SDK (unitree_sdk2_python and its dependencies), os.pathsep-separated in
# UNITREE_SDK2_PYTHONPATH; empty when the SDK is installed in the interpreter itself.
SDK_PATHS = tuple(path for path in os.environ.get("UNITREE_SDK2_PYTHONPATH", "").split(os.pathsep) if path)


@dataclass(frozen=True)
class GaitEvent:
    count: int
    phase: int
    at: float
    relative_hip_pitch: float


class GaitPhaseCounter:
    """Infer alternating large steps from left-minus-right hip pitch."""

    def __init__(
        self,
        baseline: float,
        outer: float = 0.08,
        inner: float = 0.04,
        confirm_frames: int = 3,
        min_interval_s: float = 0.25,
        alpha: float = 0.45,
    ) -> None:
        self.baseline = baseline
        self.outer = outer
        self.inner = inner
        self.confirm_frames = confirm_frames
        self.min_interval_s = min_interval_s
        self.alpha = alpha
        self.filtered = baseline
        self.candidate = 0
        self.candidate_frames = 0
        self.last_phase = 0
        self.last_event_at = float("-inf")
        self.count = 0

    def update(self, hip_difference: float, now: float, moving: bool) -> Optional[GaitEvent]:
        self.filtered = self.alpha * hip_difference + (1.0 - self.alpha) * self.filtered
        relative = self.filtered - self.baseline
        if not moving:
            # The high-level controller can settle with one leg slightly ahead
            # after a stop. Follow that pose until the next continuous run starts.
            self.baseline = self.filtered
            self.candidate = 0
            self.candidate_frames = 0
            return None

        phase = 1 if relative >= self.outer else (-1 if relative <= -self.outer else 0)
        if phase == 0:
            if abs(relative) <= self.inner:
                self.candidate = 0
                self.candidate_frames = 0
            return None
        if phase == self.last_phase:
            return None
        if phase != self.candidate:
            self.candidate = phase
            self.candidate_frames = 1
            return None
        self.candidate_frames += 1
        if self.candidate_frames < self.confirm_frames:
            return None
        if now - self.last_event_at < self.min_interval_s:
            return None

        self.count += 1
        self.last_phase = phase
        self.last_event_at = now
        self.candidate = 0
        self.candidate_frames = 0
        return GaitEvent(self.count, phase, now, relative)


class DdsGaitMonitor:
    """Read-only DDS monitor; it never creates a command publisher."""

    def __init__(
        self, iface: str, baseline_s: float = 1.0, initialize_channel: bool = True
    ) -> None:
        self.iface = iface
        self.baseline_s = baseline_s
        self.initialize_channel = bool(initialize_channel)
        self.ready = threading.Event()
        self.stop_requested = threading.Event()
        self.moving = threading.Event()
        self.events: queue.SimpleQueue[GaitEvent] = queue.SimpleQueue()
        self.error: Optional[str] = None
        self.baseline: Optional[float] = None
        self.counter: Optional[GaitPhaseCounter] = None
        self.thread: Optional[threading.Thread] = None

    @property
    def count(self) -> int:
        return self.counter.count if self.counter is not None else 0

    @property
    def last_event_at(self) -> Optional[float]:
        if self.counter is None or not math.isfinite(self.counter.last_event_at):
            return None
        return self.counter.last_event_at

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_requested.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def drain(self) -> List[GaitEvent]:
        result: List[GaitEvent] = []
        while True:
            try:
                result.append(self.events.get_nowait())
            except queue.Empty:
                return result

    def _run(self) -> None:
        try:
            for path in SDK_PATHS:
                if path not in sys.path:
                    sys.path.insert(0, path)
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

            if self.initialize_channel:
                ChannelFactoryInitialize(0, self.iface)
            subscriber = ChannelSubscriber("rt/lf/lowstate", LowState_)
            subscriber.Init()
            baseline_values: List[float] = []
            deadline = time.monotonic() + self.baseline_s
            last_tick: Optional[int] = None
            while time.monotonic() < deadline and not self.stop_requested.is_set():
                message = subscriber.Read()
                if message is None:
                    time.sleep(0.002)
                    continue
                tick = int(message.tick)
                if tick == last_tick:
                    continue
                last_tick = tick
                baseline_values.append(
                    float(message.motor_state[0].q - message.motor_state[6].q)
                )
            if len(baseline_values) < 10:
                raise RuntimeError("insufficient lowstate frames for gait baseline")
            self.baseline = statistics.median(baseline_values)
            self.counter = GaitPhaseCounter(self.baseline)
            self.ready.set()

            while not self.stop_requested.is_set():
                message = subscriber.Read()
                if message is None:
                    time.sleep(0.002)
                    continue
                tick = int(message.tick)
                if tick == last_tick:
                    continue
                last_tick = tick
                event = self.counter.update(
                    float(message.motor_state[0].q - message.motor_state[6].q),
                    time.monotonic(),
                    self.moving.is_set(),
                )
                if event is not None:
                    self.events.put(event)
        except Exception as exc:
            self.error = "%s: %s" % (type(exc).__name__, exc)
            self.ready.set()


class DirectSportClient:
    """One long-lived SDK2 high-level sport client owned by the mission."""

    def __init__(self, execute: bool, timeout_s: float = 1.0, iface: str = "eth0") -> None:
        self.execute = bool(execute)
        self.timeout_s = float(timeout_s)
        self.iface = iface
        self._client: Any = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        if not self.execute or self._client is not None:
            return
        for path in SDK_PATHS:
            if path not in sys.path:
                sys.path.insert(0, path)
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        ChannelFactoryInitialize(0, self.iface)
        client = LocoClient()
        client.SetTimeout(self.timeout_s)
        client.Init()
        self._client = client

    async def _velocity(self, vx: float, yaw: float, duration: float) -> Dict[str, Any]:
        if not self.execute:
            return {"ok": True, "dry_run": True, "rc": None}
        if self._client is None:
            raise RuntimeError("sport client is not initialized")
        loop = asyncio.get_running_loop()
        async with self._lock:
            rc = int(
                await loop.run_in_executor(
                    None, self._client.SetVelocity, vx, 0.0, yaw, duration
                )
            )
        if rc != 0:
            raise RuntimeError("SetVelocity returned %d" % rc)
        return {"ok": True, "rc": rc}

    async def drive(self, vx: float, yaw: float, duration: float) -> Dict[str, Any]:
        return await self._velocity(vx, yaw, duration)

    async def stop(self) -> Dict[str, Any]:
        return await self._velocity(0.0, 0.0, 0.2)

def _side_min(observation: Any, side: str) -> Optional[float]:
    if not isinstance(observation, dict):
        return None
    values: List[float] = []
    for name in ("depth", "lidar"):
        report = observation.get(name)
        value = report.get(side + "_m") if isinstance(report, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value) and value >= 0.0:
                values.append(value)
    return min(values) if values else None


def _report_distance(report: Any, key: str) -> Optional[float]:
    value = report.get(key) if isinstance(report, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0.0 else None


def continuous_gate_observation(
    observation: Any, tilt_limit: float = 0.35, require_lidar: bool = True
) -> GateDecision:
    """Gate a continuously renewed lease, leaving mid-range obstacles to steering."""
    if not isinstance(observation, dict):
        return GateDecision(False, True, "observation_unavailable")
    state = observation.get("state")
    depth = observation.get("depth")
    lidar = observation.get("lidar")
    if not isinstance(state, dict) or not state.get("usable"):
        return GateDecision(False, True, "state_unavailable")
    if not state.get("ready"):
        return GateDecision(False, True, "state_not_ready")
    rpy = state.get("rpy")
    if not isinstance(rpy, (list, tuple)) or len(rpy) < 2:
        return GateDecision(False, True, "state_attitude_unavailable")
    if abs(float(rpy[0])) > tilt_limit or abs(float(rpy[1])) > tilt_limit:
        return GateDecision(False, True, "excessive_tilt")
    if not isinstance(depth, dict) or not depth.get("usable"):
        return GateDecision(False, True, "depth_unavailable")
    if require_lidar and (not isinstance(lidar, dict) or not lidar.get("usable")):
        return GateDecision(False, True, "lidar_unavailable")

    depth_center = _report_distance(depth, "center_m")
    lidar_center = _report_distance(lidar, "center_m")
    if depth_center is not None and depth_center < 0.50:
        return GateDecision(False, True, "depth_center_obstacle")
    if lidar_center is not None and lidar_center < 0.55:
        return GateDecision(False, True, "lidar_center_obstacle")
    side_values = [
        value
        for report in (depth, lidar)
        for key in ("left_m", "right_m")
        if (value := _report_distance(report, key)) is not None
    ]
    if any(value < 0.25 for value in side_values):
        return GateDecision(False, True, "side_obstacle")
    narrow = (
        any(value < 0.50 for value in side_values)
        or (depth_center is not None and depth_center < 0.90)
        or (lidar_center is not None and lidar_center < 0.90)
    )
    return GateDecision(True, False, "narrow" if narrow else "clear", narrow=narrow)


def local_avoidance_yaw(observation: Any, yaw: float = 0.12) -> float:
    """Steer gently away from a side obstacle while keeping forward motion."""
    depth = observation.get("depth") if isinstance(observation, dict) else None
    left = _report_distance(depth, "left_m")
    right = _report_distance(depth, "right_m")
    if left is None or right is None:
        left = _side_min(observation, "left")
        right = _side_min(observation, "right")
    if left is None or right is None:
        return 0.0
    if right < 0.45 and left - right >= 0.05:
        return yaw
    if left < 0.45 and right - left >= 0.05:
        return -yaw
    return 0.0


class PhysicalStepsMission:
    def __init__(
        self,
        nav: Any,
        console: ConsoleClient,
        gait: DdsGaitMonitor,
        args: argparse.Namespace,
    ) -> None:
        self.nav = nav
        self.console = console
        self.gait = gait
        self.args = args
        self.latest: Any = None
        self.observation: Optional[Dict[str, Any]] = None
        self.forward_latched = bool(args.auto_forward)
        self.forward_steps = 0
        self.lease_count = 0
        self.max_lease_gap_s = 0.0
        self.max_command_latency_s = 0.0
        self.missed_deadlines = 0
        self._last_lease_at: Optional[float] = None
        self.started_at: Optional[float] = None
        self.last_observation_mono: Optional[float] = None
        self.sensor_error_count = 0
        self._usable_reports: Dict[str, Any] = {}
        self._degraded_components: set = set()
        self.terminal = asyncio.Event()
        self.terminal_kind: Optional[str] = None
        self.terminal_reason: Optional[str] = None

    def emit(self, kind: str, **detail: Any) -> None:
        print(json.dumps({
            "type": "mission_event",
            "kind": kind,
            "forward_steps": self.forward_steps,
            **detail,
        }, ensure_ascii=False, separators=(",", ":")), flush=True)

    def finish(self, kind: str, reason: str) -> None:
        if not self.terminal.is_set():
            self.terminal_kind = kind
            self.terminal_reason = reason
            self.terminal.set()

    def effective_observation(self) -> Any:
        """Reuse recent usable reports only across transient reader failures."""
        if not isinstance(self.observation, dict):
            return self.observation
        now = time.monotonic()
        effective = dict(self.observation)
        degraded = set()
        required = [
            ("state", self.args.state_grace_s),
            ("depth", self.args.sensor_grace_s),
        ]
        if not self.args.no_lidar:
            required.append(("lidar", self.args.sensor_grace_s))
        for name, grace_s in required:
            report = self.observation.get(name)
            usable = isinstance(report, dict) and bool(report.get("usable"))
            if usable:
                self._usable_reports[name] = (dict(report), now)
                continue
            degraded.add(name)
            cached = self._usable_reports.get(name)
            if cached is not None and now - cached[1] < grace_s:
                effective[name] = cached[0]

        for name in sorted(degraded - self._degraded_components):
            self.sensor_error_count += 1
            self.emit("SENSOR_DEGRADED", component=name, fallback="recent_usable")
        for name in sorted(self._degraded_components - degraded):
            self.emit("SENSOR_RECOVERED", component=name)
        self._degraded_components = degraded
        return effective

    async def supervised(self, awaitable: Any) -> None:
        try:
            await awaitable
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.finish("ERROR", "%s: %s" % (type(exc).__name__, exc))

    async def read_stdin(self) -> None:
        loop = asyncio.get_running_loop()
        while not self.terminal.is_set():
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if line == "":
                self.finish("HARD_STOP", "stdin_eof")
                return
            try:
                directive = parse_policy_action(json.loads(line), time.monotonic())
                if self.latest is not None and directive.seq <= self.latest.seq:
                    continue
                self.latest = directive
                self.emit("DIRECTIVE", seq=directive.seq, action=directive.action)
                if directive.action in {"FORWARD", "TURN_LEFT", "TURN_RIGHT"}:
                    self.forward_latched = True
                elif directive.action == "STOP":
                    self.finish("HARD_STOP", "policy_stop")
                elif directive.action == "BACKWARD":
                    self.finish("HARD_STOP", "backward_aborts_mission")
            except Exception as exc:
                self.emit("DIRECTIVE", accepted=False, reason=str(exc))

    async def perceive(self) -> None:
        period = 1.0 / self.args.sensor_hz
        loop = asyncio.get_running_loop()
        while not self.terminal.is_set():
            try:
                snapshot = await loop.run_in_executor(None, self.nav.observe)
                self.observation = snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot
                self.last_observation_mono = time.monotonic()
            except Exception as exc:
                self.sensor_error_count += 1
                age = (
                    time.monotonic() - self.last_observation_mono
                    if self.last_observation_mono is not None else float("inf")
                )
                self.emit("SENSOR_DEGRADED", error="%s: %s" % (type(exc).__name__, exc), age_s=age)
                if age >= self.args.sensor_grace_s:
                    self.finish("HARD_STOP", "sensor_stale")
                    await self.console.stop()
                    return
                await asyncio.sleep(period)
                continue
            decision = continuous_gate_observation(
                self.effective_observation(), self.args.tilt_limit, not self.args.no_lidar
            )
            if decision.hard_stop:
                self.finish("HARD_STOP", decision.reason)
                await self.console.stop()
                return
            await asyncio.sleep(period)

    def commanded_yaw(self) -> float:
        reflex = local_avoidance_yaw(self.observation, self.args.avoid_yaw)
        if reflex:
            return reflex
        if self.latest is None or time.monotonic() - self.latest.received_at > 1.5:
            return 0.0
        action = self.latest.action
        if action == "TURN_LEFT":
            return self.args.policy_turn_yaw
        if action == "TURN_RIGHT":
            return -self.args.policy_turn_yaw
        return 0.0

    def consume_gait_events(self) -> Optional[float]:
        latest_at: Optional[float] = None
        for event in self.gait.drain():
            self.forward_steps = min(event.count, self.args.steps)
            latest_at = event.at
            self.emit(
                "PHYSICAL_STEP",
                step=self.forward_steps,
                phase="left" if event.phase > 0 else "right",
                relative_hip_pitch=event.relative_hip_pitch,
            )
            if event.count >= self.args.steps:
                break
        return latest_at

    async def actuate(self) -> None:
        walking_started: Optional[float] = None
        last_progress: Optional[float] = None
        next_refresh: Optional[float] = None
        while not self.terminal.is_set():
            if not self.forward_latched:
                await asyncio.sleep(0.03)
                continue
            event_at = self.consume_gait_events()
            if event_at is not None:
                last_progress = event_at
            if self.forward_steps >= self.args.steps:
                self.finish("MISSION_COMPLETE", "physical_gait_steps_complete")
                break
            decision: GateDecision = continuous_gate_observation(
                self.effective_observation(), self.args.tilt_limit, not self.args.no_lidar
            )
            if decision.hard_stop:
                self.finish("HARD_STOP", decision.reason)
                break
            if not decision.allowed:
                self.finish("HARD_STOP", decision.reason)
                break

            now = time.monotonic()
            if walking_started is None:
                walking_started = now
                last_progress = now
                self.gait.moving.set()
            if now - walking_started >= self.args.max_runtime_s:
                self.finish("HARD_STOP", "physical_step_timeout")
                break

            yaw = self.commanded_yaw()
            command_started = time.monotonic()
            response = await self.console.drive(
                self.args.forward_vx, yaw, self.args.lease_s
            )
            sent_at = time.monotonic()
            self.max_command_latency_s = max(
                self.max_command_latency_s, sent_at - command_started
            )
            if self._last_lease_at is not None:
                lease_gap = sent_at - self._last_lease_at
                self.max_lease_gap_s = max(self.max_lease_gap_s, lease_gap)
            else:
                lease_gap = 0.0
            self._last_lease_at = sent_at
            self.lease_count += 1
            self.emit(
                "MOTION",
                action="FORWARD",
                vx=self.args.forward_vx,
                yaw=yaw,
                lease_s=self.args.lease_s,
                narrow=decision.narrow,
                accepted=bool(response.get("ok", True)),
                lease_count=self.lease_count,
            )
            if lease_gap >= self.args.lease_s:
                self.finish("ERROR", "lease_continuity_lost")
                break

            event_at = self.consume_gait_events()
            if event_at is not None:
                last_progress = event_at
            if self.forward_steps >= self.args.steps:
                self.finish("MISSION_COMPLETE", "physical_gait_steps_complete")
                break
            if last_progress is not None and now - last_progress >= self.args.gait_stall_s:
                self.finish("HARD_STOP", "gait_stalled")
                break
            if next_refresh is None:
                next_refresh = sent_at + self.args.refresh_s
            else:
                next_refresh += self.args.refresh_s
            delay = next_refresh - time.monotonic()
            if delay <= 0:
                self.missed_deadlines += 1
                next_refresh = time.monotonic()
            else:
                await asyncio.sleep(delay)

    async def run(self) -> int:
        self.gait.start()
        deadline = time.monotonic() + 4.0
        while not self.gait.ready.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if not self.gait.ready.is_set() or self.gait.error is not None:
            self.gait.close()
            self.emit("ERROR", reason=self.gait.error or "gait_monitor_timeout")
            return 1

        loop = asyncio.get_running_loop()
        snapshot = await loop.run_in_executor(None, self.nav.observe)
        self.observation = snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot
        self.last_observation_mono = time.monotonic()
        self.effective_observation()
        self.emit(
            "READY",
            execute=self.console.execute,
            target_steps=self.args.steps,
            counter="alternating_hip_pitch_phase",
            hip_baseline=self.gait.baseline,
            sensor_hz=self.args.sensor_hz,
            lease_s=self.args.lease_s,
            refresh_s=self.args.refresh_s,
            auto_forward=self.args.auto_forward,
        )
        self.started_at = time.monotonic()
        tasks = [
            asyncio.create_task(self.supervised(self.perceive())),
            asyncio.create_task(self.supervised(self.actuate())),
        ]
        if not self.args.auto_forward:
            tasks.append(asyncio.create_task(self.supervised(self.read_stdin())))
        try:
            await self.terminal.wait()
        finally:
            self.gait.moving.clear()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            stop_result: Dict[str, Any] = {}
            stop_ok = False
            postflight_ok = False
            try:
                stop_result = await self.console.stop()
                stop_ok = bool(stop_result.get("ok", False))
                if self.console.execute and getattr(self.args, "backend", "sport") == "sport":
                    stop_ok = stop_ok and stop_result.get("rc") == 0
                self.emit(
                    "FINAL_STOP",
                    accepted=stop_ok,
                    stop_rc=stop_result.get("rc"),
                )
                if not stop_ok and self.terminal_kind == "MISSION_COMPLETE":
                    self.terminal_kind = "ERROR"
                    self.terminal_reason = "final_stop_not_confirmed"
            except Exception as exc:
                self.terminal_kind = "ERROR"
                self.terminal_reason = "final_stop_failed:%s" % exc
            try:
                loop = asyncio.get_running_loop()
                if self.console.execute:
                    await asyncio.sleep(0.35)
                deadline = time.monotonic() + max(2.5, self.args.state_grace_s + 0.5)
                postflight: Any = None
                while True:
                    sample = await loop.run_in_executor(None, self.nav.observe)
                    postflight = sample.to_dict() if hasattr(sample, "to_dict") else sample
                    state = postflight.get("state") if isinstance(postflight, dict) else None
                    rpy = state.get("rpy") if isinstance(state, dict) else None
                    postflight_ok = bool(
                        isinstance(state, dict)
                        and state.get("usable")
                        and state.get("ready")
                        and isinstance(rpy, (list, tuple))
                        and len(rpy) >= 2
                        and abs(float(rpy[0])) <= self.args.tilt_limit
                        and abs(float(rpy[1])) <= self.args.tilt_limit
                    )
                    if postflight_ok or time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(0.25)
                self.emit(
                    "POSTFLIGHT",
                    ok=postflight_ok,
                    state=postflight.get("state") if isinstance(postflight, dict) else None,
                    verdict=postflight.get("verdict") if isinstance(postflight, dict) else None,
                )
                if not postflight_ok and self.terminal_kind == "MISSION_COMPLETE":
                    self.terminal_kind = "ERROR"
                    self.terminal_reason = "postflight_not_confirmed"
            except Exception as exc:
                self.emit("POSTFLIGHT_FAILED", error="%s: %s" % (type(exc).__name__, exc))
                if self.terminal_kind == "MISSION_COMPLETE":
                    self.terminal_kind = "ERROR"
                    self.terminal_reason = "postflight_failed:%s" % exc
            self.gait.close()
            self.emit(
                self.terminal_kind or "ERROR",
                reason=self.terminal_reason or "unknown",
                target_steps=self.args.steps,
                lease_count=self.lease_count,
                elapsed_s=round(time.monotonic() - self.started_at, 3)
                if self.started_at is not None else None,
                max_lease_gap_s=round(self.max_lease_gap_s, 4),
                max_command_latency_s=round(self.max_command_latency_s, 4),
                missed_deadlines=self.missed_deadlines,
                sensor_error_count=self.sensor_error_count,
                final_stop_ok=stop_ok,
                stop_rc=stop_result.get("rc"),
                postflight_ok=postflight_ok,
                monitor_count_at_stop=self.gait.count,
            )
        return 0 if self.terminal_kind == "MISSION_COMPLETE" else 1


async def async_main(args: argparse.Namespace) -> int:
    from g1nav.config import LidarCfg, NavCfg
    from g1nav.session import NavSession

    if args.backend == "sport":
        console: Any = DirectSportClient(args.execute, args.http_timeout_s, args.iface)
        gait = DdsGaitMonitor(args.iface, initialize_channel=not args.execute)
    else:
        console = ConsoleClient(args.console_base_url, args.execute, args.http_timeout_s)
        gait = DdsGaitMonitor(args.iface)
    base = NavCfg()
    nav_cfg = NavCfg(
        lidar=LidarCfg(required=not args.no_lidar),
        depth=base.depth,
        motion=base.motion,
    )
    async with NavSession(nav_cfg, execute=False) as nav:
        if args.backend == "sport":
            await console.initialize()
        mission = PhysicalStepsMission(nav, console, gait, args)
        loop = asyncio.get_running_loop()
        registered = []
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                loop.add_signal_handler(
                    sig, mission.finish, "HARD_STOP", "signal_%s" % sig.name.lower()
                )
                registered.append(sig)
            except (NotImplementedError, RuntimeError):
                pass
        try:
            return await mission.run()
        finally:
            for sig in registered:
                loop.remove_signal_handler(sig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--auto-forward",
        action="store_true",
        help="start the bounded forward mission immediately without policy stdin",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--backend", choices=("sport", "console"), default="sport")
    parser.add_argument("--lease-s", type=float, default=0.45)
    parser.add_argument("--refresh-s", type=float, default=0.12)
    parser.add_argument("--sensor-hz", type=float, default=10.0)
    parser.add_argument("--sensor-grace-s", type=float, default=0.6)
    parser.add_argument("--state-grace-s", type=float, default=2.2)
    parser.add_argument("--forward-vx", type=float, default=0.30)
    parser.add_argument("--policy-turn-yaw", type=float, default=0.18)
    parser.add_argument("--avoid-yaw", type=float, default=0.12)
    parser.add_argument("--max-runtime-s", type=float, default=18.0)
    parser.add_argument("--gait-stall-s", type=float, default=1.8)
    parser.add_argument("--tilt-limit", type=float, default=0.35)
    parser.add_argument("--console-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--http-timeout-s", type=float, default=1.0)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.steps <= 40:
        raise SystemExit("--steps must be within [1, 40]")
    speed_limit = 0.40 if args.backend == "sport" else 0.30
    if not 0.0 < args.forward_vx <= speed_limit:
        raise SystemExit("--forward-vx exceeds the selected backend limit")
    if not 0.0 < args.lease_s <= 0.60:
        raise SystemExit("--lease-s must be within (0, 0.60]")
    if not 0.0 < args.refresh_s <= args.lease_s / 2.0:
        raise SystemExit("--refresh-s must be positive and no more than half --lease-s")
    if not 1.0 <= args.sensor_hz <= 30.0:
        raise SystemExit("--sensor-hz must be within [1, 30]")
    if args.sensor_grace_s <= 0 or args.state_grace_s <= 0:
        raise SystemExit("sensor and state grace periods must be positive")
    if args.max_runtime_s <= 0 or args.gait_stall_s <= 0:
        raise SystemExit("runtime and gait stall limits must be positive")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
