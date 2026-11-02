"""Small JSON-facing protocol shared by the brain and cerebellum."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
import uuid
from typing import Any, Dict, List, Optional


SPORT_NAVIGATION = "sport_navigation"
FORWARD_PUSH = "arm_action:forward_push"
RELEASE_ARM = "arm_action:release_arm"

HYBRID_HIGH_LEVEL = "HYBRID_HIGH_LEVEL"
SONIC_WBC = "SONIC_WBC"
CONTROLLER_MODES = (HYBRID_HIGH_LEVEL, SONIC_WBC)

MAX_SPORT_VX = 0.20
MAX_SPORT_YAW = 0.35
MAX_SPORT_PULSE_S = 0.60


@dataclass(frozen=True)
class Intent:
    intent_id: str
    task: str
    created_at: float
    ttl_s: float = 120.0
    heartbeat_timeout_s: float = 8.0
    inference_period_s: float = 0.5
    policy_wait_timeout_s: float = 15.0
    execute: bool = False
    controller_mode: str = HYBRID_HIGH_LEVEL
    plan: Optional["ExecutionPlan"] = None
    # A plan normally waits for one advisory policy output before it runs. A reactive brain that
    # already decided (rea-agent's gated token) sets this False so a bounded pulse is not
    # held behind a slow VLA round-trip; the navigation gate still runs before every pulse.
    wait_for_policy: bool = True

    @classmethod
    def create(
        cls,
        task: str,
        ttl_s: float = 120.0,
        heartbeat_timeout_s: float = 8.0,
        inference_period_s: float = 0.5,
        policy_wait_timeout_s: float = 15.0,
        execute: bool = False,
        controller_mode: str = HYBRID_HIGH_LEVEL,
        plan: Optional["ExecutionPlan"] = None,
        intent_id: Optional[str] = None,
        wait_for_policy: bool = True,
    ) -> "Intent":
        task = task.strip()
        if not task:
            raise ValueError("task must not be empty")
        if (
            ttl_s <= 0
            or heartbeat_timeout_s <= 0
            or inference_period_s <= 0
            or policy_wait_timeout_s <= 0
        ):
            raise ValueError(
                "ttl, heartbeat timeout, inference period and policy wait timeout "
                "must be positive"
            )
        if execute and plan is None:
            raise ValueError("structured execution plan is required for physical execution")
        controller_mode = normalize_controller_mode(controller_mode)
        if controller_mode == SONIC_WBC and plan is not None:
            raise ValueError("hybrid execution plans cannot run in SONIC_WBC mode")
        if plan is not None:
            plan.validate()
        return cls(
            intent_id=intent_id or uuid.uuid4().hex,
            task=task,
            created_at=time.time(),
            ttl_s=float(ttl_s),
            heartbeat_timeout_s=float(heartbeat_timeout_s),
            inference_period_s=float(inference_period_s),
            policy_wait_timeout_s=float(policy_wait_timeout_s),
            execute=bool(execute),
            controller_mode=controller_mode,
            plan=plan,
            wait_for_policy=bool(wait_for_policy),
        )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["plan"] = self.plan.to_dict() if self.plan is not None else None
        return value


def normalize_controller_mode(value: Any) -> str:
    """Normalize one of the two explicit controller ownership modes."""
    mode = str(value or HYBRID_HIGH_LEVEL).strip().upper().replace("-", "_")
    aliases = {
        "HYBRID": HYBRID_HIGH_LEVEL,
        "HIGH_LEVEL": HYBRID_HIGH_LEVEL,
        "SPORT": HYBRID_HIGH_LEVEL,
        "SONIC": SONIC_WBC,
        "WBC": SONIC_WBC,
    }
    mode = aliases.get(mode, mode)
    if mode not in CONTROLLER_MODES:
        raise ValueError(
            "controller_mode must be one of %s" % ", ".join(CONTROLLER_MODES)
        )
    return mode


@dataclass(frozen=True)
class MotionPulse:
    vx: float
    yaw: float
    duration_s: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "MotionPulse":
        return cls(
            vx=float(value.get("vx", 0.0)),
            yaw=float(value.get("yaw", 0.0)),
            duration_s=float(value.get("duration_s", 0.0)),
        )


@dataclass(frozen=True)
class Capability:
    """One high-level action primitive exposed by the hybrid executor."""

    name: str
    available: bool = True
    provider: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "name": self.name,
            "available": self.available,
        }
        if self.provider:
            value["provider"] = self.provider
        if self.detail:
            value["detail"] = dict(self.detail)
        return value


@dataclass(frozen=True)
class NavigationPlan:
    """A bounded sequence for the audited high-level SPORT adapter."""

    pulses: List[MotionPulse] = field(default_factory=list)
    target: str = ""

    def validate(self) -> None:
        if not self.pulses:
            raise ValueError("navigation plan must contain at least one bounded pulse")
        for pulse in self.pulses:
            if pulse.duration_s <= 0:
                raise ValueError("navigation pulse duration must be positive")
            if abs(pulse.vx) > MAX_SPORT_VX:
                raise ValueError("navigation pulse exceeds the SPORT vx envelope")
            if abs(pulse.yaw) > MAX_SPORT_YAW:
                raise ValueError("navigation pulse exceeds the SPORT yaw envelope")
            if pulse.duration_s > MAX_SPORT_PULSE_S:
                raise ValueError("navigation pulse exceeds the SPORT duration envelope")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "pulses": [pulse.to_dict() for pulse in self.pulses],
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "NavigationPlan":
        pulses = value.get("pulses") or []
        if not isinstance(pulses, list):
            raise ValueError("navigation.pulses must be a list")
        return cls(
            pulses=[MotionPulse.from_dict(item) for item in pulses],
            target=str(value.get("target", "")),
        )


@dataclass(frozen=True)
class ArmActionStep:
    """A named SDK2 arm action; it is never a raw joint/torque command."""

    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def capability_name(self) -> str:
        return "arm_action:%s" % self.action

    def validate(self) -> None:
        if not self.action.strip():
            raise ValueError("arm action must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ArmActionStep":
        parameters = value.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("arm action parameters must be an object")
        return cls(action=str(value.get("action", "")), parameters=dict(parameters))


@dataclass(frozen=True)
class ExecutionPlan:
    """Structured hybrid plan produced by an agent-side task planner.

    Base ``REAL_G1`` GR00T output is deliberately not an execution-plan source:
    its relative whole-body action vectors are not compatible with SPORT or a
    named SDK2 arm primitive.
    """

    source: str
    navigation: Optional[NavigationPlan] = None
    arm_steps: List[ArmActionStep] = field(default_factory=list)
    task: str = ""
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def validate(self) -> None:
        source = self.source.strip().lower()
        if not source:
            raise ValueError("execution plan source must not be empty")
        if source == "groot_real_g1":
            raise ValueError("groot_real_g1 output cannot be used as an execution plan")
        if self.navigation is None and not self.arm_steps:
            raise ValueError("execution plan must contain navigation or an arm action")
        if self.navigation is not None:
            self.navigation.validate()
        for step in self.arm_steps:
            step.validate()

    @property
    def required_capabilities(self) -> List[str]:
        capabilities: List[str] = []
        if self.navigation is not None:
            capabilities.append(SPORT_NAVIGATION)
        capabilities.extend(step.capability_name for step in self.arm_steps)
        return capabilities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "task": self.task,
            "source": self.source,
            "required_capabilities": self.required_capabilities,
            "navigation": self.navigation.to_dict() if self.navigation else None,
            "arm_steps": [step.to_dict() for step in self.arm_steps],
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ExecutionPlan":
        if not isinstance(value, dict):
            raise ValueError("execution plan must be an object")
        navigation_value = value.get("navigation")
        if navigation_value is not None and not isinstance(navigation_value, dict):
            raise ValueError("execution plan navigation must be an object")
        arm_values = value.get("arm_steps") or []
        if not isinstance(arm_values, list):
            raise ValueError("execution plan arm_steps must be a list")
        plan = cls(
            source=str(value.get("source", "")),
            navigation=(
                NavigationPlan.from_dict(navigation_value)
                if navigation_value is not None
                else None
            ),
            arm_steps=[ArmActionStep.from_dict(item) for item in arm_values],
            task=str(value.get("task", "")),
            plan_id=str(value.get("plan_id") or uuid.uuid4().hex),
        )
        plan.validate()
        return plan


@dataclass(frozen=True)
class ExecutionAvailability:
    """JSON-facing declaration of the executor's installed primitives."""

    capabilities: List[Capability]
    execute_enabled: bool = False
    scope: str = "HYBRID_HIGH_LEVEL"
    direct_vla_actuation: bool = False

    @property
    def kind(self) -> str:
        return (
            "EXECUTION_AVAILABLE"
            if self.execute_enabled
            and self.capabilities
            and all(item.available for item in self.capabilities)
            else "EXECUTION_UNAVAILABLE"
        )

    def to_dict(self) -> Dict[str, Any]:
        available_names = [item.name for item in self.capabilities if item.available]
        return {
            "kind": self.kind,
            "scope": self.scope,
            "capabilities": available_names,
            "capability_status": [item.to_dict() for item in self.capabilities],
            "direct_vla_actuation": self.direct_vla_actuation,
            "execution_enabled": self.execute_enabled,
            "dry_run": not self.execute_enabled,
            "components_ready": bool(self.capabilities)
            and all(item.available for item in self.capabilities),
        }

    def missing_for(self, plan: ExecutionPlan) -> List[str]:
        available = {item.name for item in self.capabilities if item.available}
        return [name for name in plan.required_capabilities if name not in available]

    def supports(self, plan: ExecutionPlan) -> bool:
        plan.validate()
        return not self.missing_for(plan)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class PolicyOutput:
    intent_id: str
    generation: int
    produced_at: float
    latency_s: float
    policy: str
    summary: Dict[str, Any] = field(default_factory=dict)
    pulses: List[MotionPulse] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["pulses"] = [pulse.to_dict() for pulse in self.pulses]
        return value


@dataclass(frozen=True)
class Event:
    sequence: int
    kind: str
    at: float
    intent_id: Optional[str]
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
