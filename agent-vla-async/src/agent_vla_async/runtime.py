"""Long-running asynchronous VLA cerebellum on the robot gateway."""

from __future__ import annotations

import asyncio
from collections import deque
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

from .controller import UnavailableSonicSupervisor
from .execution import SportNavigationExecutor
from .mailbox import LatestValue
from .protocol import (
    Capability,
    Event,
    ExecutionAvailability,
    ExecutionPlan,
    FORWARD_PUSH,
    HYBRID_HIGH_LEVEL,
    Intent,
    NavigationPlan,
    PolicyOutput,
    RELEASE_ARM,
    SPORT_NAVIGATION,
    SONIC_WBC,
    normalize_controller_mode,
)
from .rpc import serve


class CerebellumRuntime:
    def __init__(
        self,
        navigation: Any,
        policy: Any,
        socket_path: str,
        execute_enabled: bool = False,
        arm: Any = None,
        event_limit: int = 256,
        policy_wait_timeout_s: float = 15.0,
        sonic_supervisor: Any = None,
        sonic_policy: Any = None,
    ) -> None:
        self.navigation = navigation
        self.policy = policy
        self.socket_path = socket_path
        self.execute_enabled = bool(execute_enabled)
        self.arm = arm
        self.policy_wait_timeout_s = float(policy_wait_timeout_s)
        self.sonic_policy = sonic_policy
        self.sonic_supervisor = sonic_supervisor or UnavailableSonicSupervisor()
        self._controller_mode = HYBRID_HIGH_LEVEL
        self._last_sonic_supervisor_status = self.sonic_supervisor.status()
        self.events: Deque[Event] = deque(maxlen=event_limit)
        self._active: Optional[Intent] = None
        self._generation = 0
        self._last_heartbeat_mono = 0.0
        self._created_mono = 0.0
        self._intent_changed = asyncio.Event()
        self._plan_changed = asyncio.Event()
        self._outputs: LatestValue[PolicyOutput] = LatestValue()
        self._sequence = 0
        self._tasks = []
        self._server: Any = None
        self._closed = asyncio.Event()
        self._inference_count = 0
        self._discarded_count = 0
        self._preview_count = 0
        self._motion_count = 0
        self._arm_action_count = 0
        self._completed_plan_count = 0
        self._last_output: Optional[PolicyOutput] = None
        self._last_plan_result: Optional[Dict[str, Any]] = None
        self._plan_started_generation = -1

    def _emit(self, kind: str, intent_id: Optional[str] = None, **detail: Any) -> None:
        self._sequence += 1
        self.events.append(Event(self._sequence, kind, time.time(), intent_id, detail))

    def _active_snapshot(self) -> Tuple[Optional[Intent], int]:
        return self._active, self._generation

    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        op = request.get("op")
        if op == "ping":
            return {"ok": True, "service": "agent-vla-cerebellum"}
        if op == "capabilities":
            return await self.capabilities()
        if op == "sonic_status":
            return await self.sonic_status()
        if op == "sonic_validate":
            return await self.sonic_validate(str(request.get("task", "")))
        if op == "controller_status":
            return await self.controller_status()
        if op == "controller_prepare":
            return await self.controller_prepare(request.get("controller_mode"))
        if op == "controller_stop":
            return await self.controller_stop(request.get("controller_mode"))
        if op == "submit":
            try:
                controller_mode = normalize_controller_mode(
                    request.get("controller_mode", HYBRID_HIGH_LEVEL)
                )
            except ValueError as exc:
                return {
                    "ok": False,
                    "kind": "CONTROLLER_MODE_INVALID",
                    "error": str(exc),
                    "motion_commands_sent": 0,
                    "arm_actions_sent": 0,
                }
            plan_value = request.get("plan")
            plan = ExecutionPlan.from_dict(plan_value) if plan_value is not None else None
            execute = bool(request.get("execute", False))
            if controller_mode == SONIC_WBC:
                controller = await self.controller_status()
                return self._execution_unavailable(
                    plan,
                    ["sonic_wbc_actuation_path_not_connected"],
                    scope=SONIC_WBC,
                    controller=controller,
                )
            if self._controller_mode != HYBRID_HIGH_LEVEL:
                return self._execution_unavailable(
                    plan,
                    ["controller_mode_conflict:%s" % self._controller_mode],
                    scope=HYBRID_HIGH_LEVEL,
                )
            if execute and plan is None:
                return self._execution_unavailable(
                    None,
                    ["structured_plan_required"],
                )
            availability = await self._availability()
            if execute and not self.execute_enabled:
                return self._execution_unavailable(
                    plan,
                    ["cerebellum_service_is_running_in_dry_run_mode"],
                    availability,
                )
            if execute and plan is not None:
                missing = availability.missing_for(plan)
                if missing:
                    return self._execution_unavailable(plan, missing, availability)
            intent = Intent.create(
                task=str(request.get("task", "")),
                ttl_s=float(request.get("ttl_s", 120.0)),
                heartbeat_timeout_s=float(request.get("heartbeat_timeout_s", 8.0)),
                inference_period_s=float(request.get("inference_period_s", 0.5)),
                policy_wait_timeout_s=float(
                    request.get("policy_wait_timeout_s", self.policy_wait_timeout_s)
                ),
                execute=execute,
                controller_mode=controller_mode,
                plan=plan,
                intent_id=request.get("intent_id"),
                wait_for_policy=bool(request.get("wait_for_policy", True)),
            )
            previous = self._active
            if previous is not None and previous.execute and self.execute_enabled:
                await self._stop_execution(release_arm=self._arm_action_count > 0)
                self._emit("SUPERSEDE_STOPPED", previous.intent_id)
            self._generation += 1
            self._active = intent
            self._created_mono = time.monotonic()
            self._last_heartbeat_mono = self._created_mono
            await self._outputs.clear()
            if previous is not None:
                self._emit("SUPERSEDED", previous.intent_id, by=intent.intent_id)
            self._emit("ACCEPTED", intent.intent_id, generation=self._generation, execute=intent.execute)
            self._intent_changed.set()
            self._plan_changed.set()
            return {
                "ok": True,
                "kind": "EXECUTION_AVAILABLE" if intent.execute else "PREVIEW_ACCEPTED",
                "scope": HYBRID_HIGH_LEVEL,
                "controller_mode": controller_mode,
                "direct_vla_actuation": False,
                "motion_commands_sent": 0,
                "arm_actions_sent": 0,
                "intent": intent.to_dict(),
                "generation": self._generation,
            }
        if op == "heartbeat":
            intent_id = request.get("intent_id")
            if self._active is None or self._active.intent_id != intent_id:
                return {"ok": False, "error": "intent is not active"}
            self._last_heartbeat_mono = time.monotonic()
            return {"ok": True, "intent_id": intent_id}
        if op == "cancel":
            intent_id = request.get("intent_id")
            if self._active is None:
                return {"ok": True, "cancelled": False}
            if intent_id and self._active.intent_id != intent_id:
                return {"ok": False, "error": "intent is not active"}
            await self._deactivate("CANCELLED")
            return {"ok": True, "cancelled": True}
        if op == "status":
            return {"ok": True, "status": self.status()}
        if op == "events":
            after = int(request.get("after", 0))
            return {
                "ok": True,
                "events": [event.to_dict() for event in self.events if event.sequence > after],
                "last_sequence": self._sequence,
            }
        if op == "shutdown":
            self._closed.set()
            return {"ok": True}
        return {"ok": False, "error": "unknown operation"}

    def status(self) -> Dict[str, Any]:
        active = self._active
        return {
            "active": active.to_dict() if active else None,
            "generation": self._generation,
            "policy": getattr(self.policy, "name", type(self.policy).__name__),
            "execute_enabled": self.execute_enabled,
            "controller_mode": self._controller_mode,
            "controller_readiness": {
                "sonic_wbc": self._last_sonic_supervisor_status,
            },
            "inference_count": self._inference_count,
            "discarded_inference_count": self._discarded_count,
            "preview_pulse_count": self._preview_count,
            "motion_pulse_count": self._motion_count,
            "arm_action_count": self._arm_action_count,
            "completed_plan_count": self._completed_plan_count,
            "last_output": self._last_output.to_dict() if self._last_output else None,
            "last_plan_result": self._last_plan_result,
            "last_sequence": self._sequence,
        }

    def _execution_unavailable(
        self,
        plan: Optional[ExecutionPlan],
        reasons: List[str],
        availability: Optional[ExecutionAvailability] = None,
        scope: str = HYBRID_HIGH_LEVEL,
        controller: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "ok": False,
            "kind": "EXECUTION_UNAVAILABLE",
            "scope": scope,
            "controller_mode": scope,
            "direct_vla_actuation": False,
            "motion_commands_sent": 0,
            "arm_actions_sent": 0,
            "reasons": reasons,
            "missing_capabilities": reasons,
        }
        if plan is not None:
            value["plan"] = plan.to_dict()
        if availability is not None:
            value["availability"] = availability.to_dict()
        if controller is not None:
            value["controller"] = controller
        return value

    async def _availability(self) -> ExecutionAvailability:
        capabilities: List[Capability] = []
        robot_state: Dict[str, Any] = {}
        try:
            observation = self.navigation.observe()
            verdict = observation.get("verdict", {}) if isinstance(observation, dict) else {}
            state = observation.get("state", {}) if isinstance(observation, dict) else {}
            robot_state = state
            nav_available = bool(
                state.get("usable")
                and state.get("ready")
                and verdict.get("allowed")
            )
            capabilities.append(
                Capability(
                    SPORT_NAVIGATION,
                    available=nav_available,
                    provider="unitree_sdk2_high_level_sport",
                    detail={
                        "mode": "bounded_pulse_sequence",
                        "verdict": verdict,
                        "state": state,
                    },
                )
            )
        except Exception as exc:
            capabilities.append(
                Capability(
                    SPORT_NAVIGATION,
                    available=False,
                    provider="unitree_sdk2_high_level_sport",
                    detail={"error": "%s: %s" % (type(exc).__name__, exc)},
                )
            )

        arm_actions: Dict[str, Any] = {}
        arm_error: Optional[str] = None
        if self.arm is not None:
            try:
                loop = asyncio.get_running_loop()
                arm_actions = await loop.run_in_executor(
                    None, lambda: self.arm.list_actions(refresh=True)
                )
            except Exception as exc:
                arm_error = "%s: %s" % (type(exc).__name__, exc)
        for action_name, capability_name in (
            ("forward_push", FORWARD_PUSH),
            ("release_arm", RELEASE_ARM),
        ):
            action = next(
                (
                    item for item in arm_actions.values()
                    if getattr(item, "name", "").strip().replace(" ", "_").casefold()
                    == action_name
                ),
                None,
            )
            allowed_modes = (
                getattr(action, "metadata", {}).get("mode_machine")
                if action is not None
                else None
            )
            mode_compatible = bool(
                not allowed_modes
                or robot_state.get("mode_machine") in allowed_modes
            )
            installed = action is not None and mode_compatible
            detail: Dict[str, Any] = {"action": action_name}
            if allowed_modes:
                detail["allowed_mode_machine"] = list(allowed_modes)
                detail["current_mode_machine"] = robot_state.get("mode_machine")
            if arm_error:
                detail["error"] = arm_error
            capabilities.append(
                Capability(
                    capability_name,
                    available=installed,
                    provider="unitree_sdk2_high_level_arm_action",
                    detail=detail,
                )
            )
        return ExecutionAvailability(
            capabilities=capabilities,
            execute_enabled=self.execute_enabled,
        )

    async def capabilities(self) -> Dict[str, Any]:
        availability = await self._availability()
        value = availability.to_dict()
        value.update(
            {
                "ok": True,
                "schema": "g1.hybrid.capabilities/v1",
                "service_mode": "execute" if self.execute_enabled else "dry_run",
                "controller_mode": self._controller_mode,
                "supported_controller_modes": [HYBRID_HIGH_LEVEL, SONIC_WBC],
                "policy": {
                    "name": getattr(self.policy, "name", type(self.policy).__name__),
                    "role": "advisory_only",
                    "direct_actuation": False,
                },
            }
        )
        return value

    async def sonic_status(self) -> Dict[str, Any]:
        """Inspect SONIC policy readiness; this endpoint never requests actions."""
        inspect = getattr(self.sonic_policy, "sonic_status", None)
        if inspect is None:
            return {
                "ok": True,
                "kind": "SONIC_POLICY_UNAVAILABLE",
                "scope": "POLICY_OPEN_LOOP",
                "policy": (
                    getattr(self.sonic_policy, "name", type(self.sonic_policy).__name__)
                    if self.sonic_policy is not None
                    else None
                ),
                "physical_execution": False,
                "controller_mode": "not_connected",
                "reasons": ["dedicated_sonic_policy_probe_not_configured"],
            }
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, inspect)
        except Exception as exc:
            endpoint = None
            if self.sonic_policy is not None:
                host = getattr(self.sonic_policy, "host", None)
                port = getattr(self.sonic_policy, "port", None)
                if host is not None and port is not None:
                    endpoint = "tcp://%s:%s" % (host, port)
            return {
                "ok": False,
                "kind": "SONIC_POLICY_UNAVAILABLE",
                "scope": "POLICY_OPEN_LOOP",
                "physical_execution": False,
                "controller_mode": "not_connected",
                "policy_endpoint": endpoint,
                "reasons": ["policy_probe_failed"],
                "error": "%s: %s" % (type(exc).__name__, exc),
            }

    async def sonic_validate(self, task: str) -> Dict[str, Any]:
        """Validate one SONIC chunk open-loop; no action publisher is created."""
        validate = getattr(self.sonic_policy, "validate_open_loop", None)
        if validate is None:
            return {
                "ok": False,
                "kind": "SONIC_OPEN_LOOP_VALIDATION_FAILED",
                "scope": "POLICY_OPEN_LOOP",
                "state_source": "synthetic_neutral_placeholder",
                "action_publisher_started_by_runtime": False,
                "controller_called_by_runtime": False,
                "lowcmd_published_by_runtime": False,
                "physical_commands_sent": 0,
                "reasons": ["dedicated_sonic_policy_client_not_configured"],
            }
        if not task.strip():
            return {
                "ok": False,
                "kind": "SONIC_OPEN_LOOP_VALIDATION_FAILED",
                "scope": "POLICY_OPEN_LOOP",
                "action_publisher_started_by_runtime": False,
                "controller_called_by_runtime": False,
                "lowcmd_published_by_runtime": False,
                "physical_commands_sent": 0,
                "reasons": ["validation_task_is_empty"],
            }
        loop = asyncio.get_running_loop()
        try:
            rgb = await self.navigation.capture_rgb()
            return await loop.run_in_executor(None, validate, task, rgb)
        except Exception as exc:
            return {
                "ok": False,
                "kind": "SONIC_OPEN_LOOP_VALIDATION_FAILED",
                "scope": "POLICY_OPEN_LOOP",
                "state_source": "synthetic_neutral_placeholder",
                "action_publisher_started_by_runtime": False,
                "controller_called_by_runtime": False,
                "lowcmd_published_by_runtime": False,
                "physical_commands_sent": 0,
                "reasons": ["sonic_open_loop_validation_failed"],
                "error": "%s: %s" % (type(exc).__name__, exc),
            }

    async def _call_sonic_supervisor(self, operation: str) -> Dict[str, Any]:
        method = getattr(self.sonic_supervisor, operation)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, method)
        if not isinstance(result, dict):
            raise TypeError("SONIC supervisor %s must return an object" % operation)
        self._last_sonic_supervisor_status = result
        return result

    async def controller_status(self) -> Dict[str, Any]:
        """Return both controller modes and their independent readiness."""
        availability = await self._availability()
        policy = await self.sonic_status()
        try:
            supervisor = await self._call_sonic_supervisor("status")
        except Exception as exc:
            supervisor = {
                "ok": False,
                "kind": "SONIC_WBC_UNAVAILABLE",
                "controller_mode": SONIC_WBC,
                "state": "ERROR",
                "prepared": False,
                "physical_execution": False,
                "lowcmd_published_by_runtime": False,
                "lowcmd_published": False,
                "lowcmd_published_scope": "agent_vla_async_runtime_only",
                "controller_started": False,
                "reasons": ["sonic_supervisor_status_failed"],
                "error": "%s: %s" % (type(exc).__name__, exc),
            }
            self._last_sonic_supervisor_status = supervisor

        policy_ready = bool(policy.get("contract_compatible", False))
        supervisor_prepared = bool(supervisor.get("prepared", False))
        sonic_physical = bool(
            policy_ready and supervisor.get("physical_execution", False)
        )
        reasons = list(policy.get("reasons", []))
        for reason in supervisor.get("reasons", []):
            if reason not in reasons:
                reasons.append(reason)
        if not sonic_physical and "sonic_wbc_actuation_path_not_connected" not in reasons:
            reasons.append("sonic_wbc_actuation_path_not_connected")

        hybrid = availability.to_dict()
        hybrid.update(
            {
                "controller_mode": HYBRID_HIGH_LEVEL,
                "selected": self._controller_mode == HYBRID_HIGH_LEVEL,
                "physical_execution": availability.kind == "EXECUTION_AVAILABLE",
                "lowcmd_published_by_runtime": False,
                "lowcmd_published": False,
                "lowcmd_published_scope": "agent_vla_async_runtime_only",
            }
        )
        sonic = {
            "kind": (
                "SONIC_WBC_READY"
                if sonic_physical
                else "SONIC_WBC_PREPARED_OPEN_LOOP"
                if policy_ready and supervisor_prepared
                else "SONIC_WBC_UNAVAILABLE"
            ),
            "controller_mode": SONIC_WBC,
            "selected": self._controller_mode == SONIC_WBC,
            "prepared": bool(policy_ready and supervisor_prepared),
            "policy_ready": policy_ready,
            "physical_execution": sonic_physical,
            "lowcmd_published_by_runtime": False,
            "lowcmd_published": False,
            "lowcmd_published_scope": "agent_vla_async_runtime_only",
            "controller_started_by_runtime": False,
            "policy": policy,
            "supervisor": supervisor,
            "reasons": reasons,
        }
        return {
            "ok": True,
            "schema": "g1.controller-modes/v1",
            "controller_mode": self._controller_mode,
            "supported_modes": [HYBRID_HIGH_LEVEL, SONIC_WBC],
            "modes": {
                HYBRID_HIGH_LEVEL: hybrid,
                SONIC_WBC: sonic,
            },
            "physical_execution": bool(
                hybrid["physical_execution"]
                if self._controller_mode == HYBRID_HIGH_LEVEL
                else sonic_physical
            ),
            "lowcmd_published_by_runtime": False,
            "lowcmd_published": False,
            "lowcmd_published_scope": "agent_vla_async_runtime_only",
        }

    async def controller_prepare(self, value: Any) -> Dict[str, Any]:
        """Prepare controller ownership without starting a robot controller."""
        try:
            mode = normalize_controller_mode(value)
        except ValueError as exc:
            return {"ok": False, "kind": "CONTROLLER_MODE_INVALID", "error": str(exc)}
        if self._active is not None:
            return {
                "ok": False,
                "kind": "CONTROLLER_BUSY",
                "controller_mode": self._controller_mode,
                "active_intent_id": self._active.intent_id,
            }

        previous = self._controller_mode
        if mode == HYBRID_HIGH_LEVEL:
            if previous == SONIC_WBC:
                await self._call_sonic_supervisor("stop")
            self._controller_mode = HYBRID_HIGH_LEVEL
            self._emit(
                "CONTROLLER_MODE_PREPARED",
                previous=previous,
                controller_mode=HYBRID_HIGH_LEVEL,
                physical_execution=self.execute_enabled,
            )
            result = await self.controller_status()
            result.update(
                {
                    "ok": True,
                    "kind": "CONTROLLER_PREPARED",
                    "operation": "prepare",
                }
            )
            return result

        policy = await self.sonic_status()
        try:
            supervisor = await self._call_sonic_supervisor("prepare")
        except Exception as exc:
            supervisor = {
                "ok": False,
                "prepared": False,
                "physical_execution": False,
                "reasons": ["sonic_supervisor_prepare_failed"],
                "error": "%s: %s" % (type(exc).__name__, exc),
            }
            self._last_sonic_supervisor_status = supervisor
        policy_ready = bool(policy.get("contract_compatible", False))
        prepared = bool(supervisor.get("ok") and supervisor.get("prepared") and policy_ready)
        if prepared:
            self._controller_mode = SONIC_WBC
            self._emit(
                "CONTROLLER_MODE_PREPARED",
                previous=previous,
                controller_mode=SONIC_WBC,
                physical_execution=bool(supervisor.get("physical_execution", False)),
            )
        result = await self.controller_status()
        sonic = result["modes"][SONIC_WBC]
        result.update(
            {
                "ok": prepared,
                "kind": (
                    "CONTROLLER_PREPARED"
                    if prepared
                    else "EXECUTION_UNAVAILABLE"
                ),
                "operation": "prepare",
                "requested_mode": SONIC_WBC,
                "physical_execution": bool(sonic["physical_execution"]),
                "reasons": list(sonic["reasons"]),
            }
        )
        return result

    async def controller_stop(self, value: Any = None) -> Dict[str, Any]:
        """Stop the selected mode and return ownership to high-level control."""
        try:
            mode = normalize_controller_mode(value or self._controller_mode)
        except ValueError as exc:
            return {"ok": False, "kind": "CONTROLLER_MODE_INVALID", "error": str(exc)}
        previous = self._controller_mode
        if mode == SONIC_WBC:
            try:
                stopped = await self._call_sonic_supervisor("stop")
            except Exception as exc:
                return {
                    "ok": False,
                    "kind": "CONTROLLER_STOP_FAILED",
                    "controller_mode": SONIC_WBC,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
            if self._controller_mode == SONIC_WBC:
                self._controller_mode = HYBRID_HIGH_LEVEL
        else:
            if self._active is not None:
                await self._deactivate("CONTROLLER_STOPPED")
                stopped = {
                    "ok": True,
                    "source": "active_intent_deactivation",
                }
            else:
                stopped = await self.navigation.stop()
        self._emit(
            "CONTROLLER_STOPPED",
            previous=previous,
            stopped_mode=mode,
            controller_mode=self._controller_mode,
        )
        return {
            "ok": bool(stopped.get("ok", stopped.get("accepted", True))),
            "kind": "CONTROLLER_STOPPED",
            "stopped_mode": mode,
            "controller_mode": self._controller_mode,
            "result": stopped,
            "lowcmd_published_by_runtime": False,
            "lowcmd_published": False,
            "lowcmd_published_scope": "agent_vla_async_runtime_only",
        }

    async def _deactivate(self, kind: str, **detail: Any) -> None:
        intent = self._active
        self._active = None
        self._generation += 1
        await self._outputs.clear()
        self._intent_changed.set()
        self._plan_changed.set()
        if intent is not None:
            self._emit(kind, intent.intent_id, **detail)
            if intent.execute and self.execute_enabled:
                await self._stop_execution(release_arm=False)

    async def _deactivate_if_current(
        self,
        intent: Intent,
        generation: int,
        kind: str,
        **detail: Any,
    ) -> bool:
        """Deactivate only when the plan still owns the active intent."""
        if not self._plan_is_current(intent, generation):
            return False
        await self._deactivate(kind, **detail)
        return True

    async def _stop_execution(self, release_arm: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {"navigation": await self.navigation.stop()}
        if release_arm and self.arm is not None:
            loop = asyncio.get_running_loop()
            try:
                result["arm"] = await loop.run_in_executor(
                    None, lambda: self.arm.execute("release_arm", wait=False)
                )
            except Exception as exc:
                result["arm"] = {
                    "ok": False,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
        return result

    async def _inference_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            intent, generation = self._active_snapshot()
            if intent is None:
                self._intent_changed.clear()
                await self._intent_changed.wait()
                continue
            self._emit("PERCEIVING", intent.intent_id, generation=generation)
            try:
                nav = self.navigation.observe()
                rgb = await self.navigation.capture_rgb()
                self._emit("INFERENCING", intent.intent_id, generation=generation)
                output = await loop.run_in_executor(
                    None, self.policy.infer, intent, generation, rgb, nav
                )
            except Exception as exc:
                current, current_generation = self._active_snapshot()
                if current is not None and current_generation == generation:
                    self._emit(
                        "INFERENCE_FAILED",
                        intent.intent_id,
                        generation=generation,
                        error="%s: %s" % (type(exc).__name__, exc),
                    )
                await asyncio.sleep(intent.inference_period_s)
                continue
            current, current_generation = self._active_snapshot()
            if current is None or current.intent_id != intent.intent_id or current_generation != generation:
                self._discarded_count += 1
                self._emit("INFERENCE_DISCARDED", intent.intent_id, generation=generation)
                continue
            self._inference_count += 1
            self._last_output = output
            await self._outputs.publish(output)
            self._emit(
                "ACTION_READY",
                intent.intent_id,
                generation=generation,
                latency_s=output.latency_s,
                pulses=len(output.pulses),
                summary=output.summary,
            )
            await asyncio.sleep(intent.inference_period_s)

    async def _policy_inspection_loop(self) -> None:
        """Report GR00T outputs without ever routing them to robot actuation."""
        version = 0
        while True:
            version, output = await self._outputs.wait_after(version)
            if output is None:
                continue
            intent, generation = self._active_snapshot()
            if intent is None or output.intent_id != intent.intent_id or output.generation != generation:
                continue
            self._emit(
                "ACTION_INSPECTED",
                intent.intent_id,
                reason="policy_is_advisory_only",
                direct_vla_actuation=False,
                summary=output.summary,
            )
            if not intent.execute and output.pulses:
                self._preview_count += len(output.pulses)

    def _plan_is_current(self, intent: Intent, generation: int) -> bool:
        current, current_generation = self._active_snapshot()
        return bool(
            current is not None
            and current.intent_id == intent.intent_id
            and current_generation == generation
        )

    async def _wait_for_policy(
        self, intent: Intent, generation: int
    ) -> Optional[PolicyOutput]:
        if not self._plan_is_current(intent, generation):
            return None
        version, output = self._outputs.snapshot()
        if output is not None and output.intent_id == intent.intent_id and output.generation == generation:
            return output
        wait_timeout_s = intent.policy_wait_timeout_s
        deadline = time.monotonic() + wait_timeout_s
        while True:
            if not self._plan_is_current(intent, generation):
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "GR00T advisory inference was not ready within %.3fs"
                    % wait_timeout_s
                )
            try:
                version, output = await asyncio.wait_for(
                    self._outputs.wait_after(version), timeout=remaining
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    "GR00T advisory inference was not ready within %.3fs"
                    % wait_timeout_s
                ) from exc
            if not self._plan_is_current(intent, generation):
                return None
            if output is not None and output.intent_id == intent.intent_id and output.generation == generation:
                return output

    async def _execute_navigation_plan(
        self,
        intent: Intent,
        generation: int,
        plan: NavigationPlan,
    ) -> Dict[str, Any]:
        executor = SportNavigationExecutor(
            self.navigation,
            execute_enabled=intent.execute and self.execute_enabled,
        )
        steps: List[Dict[str, Any]] = []
        for index, pulse in enumerate(plan.pulses):
            if not self._plan_is_current(intent, generation):
                return {"ok": False, "reason": "plan_superseded", "steps": steps}
            single = NavigationPlan(pulses=[pulse], target=plan.target)
            self._emit(
                "STAGE_STARTED",
                intent.intent_id,
                stage="sport_pulse",
                index=index,
                pulse=pulse.to_dict(),
            )
            result = await executor.execute(single)
            steps.append(result)
            if not result.get("ok"):
                return {
                    "ok": False,
                    "reason": result.get("reason", "sport_navigation_failed"),
                    "steps": steps,
                }
            if intent.execute:
                self._motion_count += int(result.get("motion_commands_sent", 0))
            else:
                self._preview_count += 1
            self._emit(
                "STAGE_COMPLETE",
                intent.intent_id,
                stage="sport_pulse",
                index=index,
                result=result,
            )
        return {"ok": True, "steps": steps}

    async def _execute_arm_steps(
        self,
        intent: Intent,
        generation: int,
        plan: ExecutionPlan,
    ) -> Dict[str, Any]:
        if not plan.arm_steps:
            return {"ok": True, "steps": []}
        if self.arm is None:
            return {"ok": False, "reason": "arm_backend_unavailable", "steps": []}
        loop = asyncio.get_running_loop()
        results: List[Dict[str, Any]] = []
        for index, step in enumerate(plan.arm_steps):
            if not self._plan_is_current(intent, generation):
                return {"ok": False, "reason": "plan_superseded", "steps": results}
            self._emit(
                "STAGE_STARTED",
                intent.intent_id,
                stage="arm_action",
                index=index,
                action=step.action,
            )
            try:
                if not intent.execute:
                    action = await loop.run_in_executor(
                        None, lambda name=step.action: self.arm.resolve_action(name)
                    )
                    result = {
                        "ok": True,
                        "dry_run": True,
                        "executed": False,
                        "action": action.to_dict(),
                        "rc": None,
                        "state": None,
                        "transport": getattr(self.arm, "transport", "injected"),
                    }
                else:
                    wait = bool(step.parameters.get("wait", True))
                    timeout_value = step.parameters.get("timeout_s")
                    timeout_s = float(timeout_value) if timeout_value is not None else None
                    result = await loop.run_in_executor(
                        None,
                        lambda action=step.action, should_wait=wait, timeout=timeout_s: self.arm.execute(
                            action,
                            wait=should_wait,
                            timeout_s=timeout,
                        ),
                    )
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": "arm_action_failed:%s: %s" % (type(exc).__name__, exc),
                    "steps": results,
                }
            results.append(result)
            if not result.get("ok"):
                return {"ok": False, "reason": "arm_action_refused", "steps": results}
            if intent.execute and result.get("executed"):
                self._arm_action_count += 1
            self._emit(
                "STAGE_COMPLETE",
                intent.intent_id,
                stage="arm_action",
                index=index,
                action=step.action,
                result=result,
            )
        return {"ok": True, "steps": results}

    async def _run_plan(self, intent: Intent, generation: int) -> None:
        plan = intent.plan
        if plan is None:
            return
        motion_count_before = self._motion_count
        arm_count_before = self._arm_action_count
        self._emit(
            "PLAN_STARTED",
            intent.intent_id,
            generation=generation,
            plan=plan.to_dict(),
            execute=intent.execute,
        )
        if intent.wait_for_policy:
            try:
                policy_output = await self._wait_for_policy(intent, generation)
            except Exception as exc:
                if not self._plan_is_current(intent, generation):
                    return
                result = {
                    "ok": False,
                    "kind": "EXECUTION_FAILED" if intent.execute else "PREVIEW_FAILED",
                    "reason": "policy_not_ready:%s: %s" % (type(exc).__name__, exc),
                    "motion_commands_sent": 0,
                    "arm_actions_sent": 0,
                }
                self._last_plan_result = result
                await self._deactivate_if_current(
                    intent, generation, result["kind"], result=result
                )
                return
            if policy_output is None:
                return
            self._emit(
                "POLICY_ADVISORY_READY",
                intent.intent_id,
                latency_s=policy_output.latency_s,
                direct_vla_actuation=False,
            )
        else:
            # The brain that submitted this plan has already decided; the advisory loop keeps
            # running and reporting, but the bounded plan does not wait for it.
            self._emit(
                "POLICY_ADVISORY_SKIPPED",
                intent.intent_id,
                reason="wait_for_policy=false",
                direct_vla_actuation=False,
            )

        navigation_result: Dict[str, Any] = {"ok": True, "steps": []}
        if plan.navigation is not None:
            navigation_result = await self._execute_navigation_plan(
                intent, generation, plan.navigation
            )
        arm_result: Dict[str, Any] = {"ok": True, "steps": []}
        if navigation_result.get("ok") and self._plan_is_current(intent, generation):
            arm_result = await self._execute_arm_steps(intent, generation, plan)

        if not self._plan_is_current(intent, generation):
            return

        ok = bool(navigation_result.get("ok") and arm_result.get("ok"))
        kind = (
            "EXECUTION_COMPLETE" if intent.execute and ok
            else "EXECUTION_FAILED" if intent.execute
            else "PREVIEW_COMPLETE" if ok
            else "PREVIEW_FAILED"
        )
        result = {
            "ok": ok,
            "kind": kind,
            "scope": "HYBRID_HIGH_LEVEL",
            "direct_vla_actuation": False,
            "plan_id": plan.plan_id,
            "motion_commands_sent": (
                self._motion_count - motion_count_before if intent.execute else 0
            ),
            "arm_actions_sent": (
                self._arm_action_count - arm_count_before if intent.execute else 0
            ),
            "navigation": navigation_result,
            "arm": arm_result,
        }
        self._last_plan_result = result
        if ok:
            self._completed_plan_count += 1
        await self._deactivate_if_current(intent, generation, kind, result=result)

    async def _plan_loop(self) -> None:
        while True:
            intent, generation = self._active_snapshot()
            if (
                intent is None
                or intent.plan is None
                or generation == self._plan_started_generation
            ):
                self._plan_changed.clear()
                await self._plan_changed.wait()
                continue
            self._plan_started_generation = generation
            await self._run_plan(intent, generation)

    async def _watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            intent = self._active
            if intent is None:
                continue
            now = time.monotonic()
            if now - self._created_mono > intent.ttl_s:
                await self._deactivate("EXPIRED", reason="ttl")
            elif now - self._last_heartbeat_mono > intent.heartbeat_timeout_s:
                await self._deactivate("EXPIRED", reason="brain_heartbeat")

    async def run(self) -> None:
        async with self.navigation:
            self._server = await serve(self.socket_path, self.handle)
            self._emit("READY", socket=self.socket_path)
            self._tasks = [
                asyncio.create_task(self._inference_loop()),
                asyncio.create_task(self._policy_inspection_loop()),
                asyncio.create_task(self._plan_loop()),
                asyncio.create_task(self._watchdog_loop()),
            ]
            await self._closed.wait()
            if self._active is not None:
                await self._deactivate("SHUTDOWN")
            if self._controller_mode == SONIC_WBC:
                await self._call_sonic_supervisor("stop")
                self._controller_mode = HYBRID_HIGH_LEVEL
            # Python 3.10 and 3.11 asyncio.wait_for can swallow a cancellation that arrives while the
            # awaited future completes (fixed in 3.12), leaving a loop parked on its next wait forever;
            # re-issue the cancel until every loop has actually finished.
            pending = set(self._tasks)
            while pending:
                for task in pending:
                    task.cancel()
                _done, pending = await asyncio.wait(pending, timeout=1.0)
            self._server.close()
            await self._server.wait_closed()
            lock_file = getattr(self._server, "_g1_lock_file", None)
            if lock_file is not None:
                lock_file.close()
            if self.arm is not None:
                self.arm.close()

    async def close(self) -> None:
        self._closed.set()
