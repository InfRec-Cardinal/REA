"""Controller-mode supervision without owning a robot actuator.

The runtime has two mutually exclusive control-plane modes:

``HYBRID_HIGH_LEVEL``
    The existing audited SPORT and named arm-action executor.

``SONIC_WBC``
    A future external GEAR-SONIC whole-body controller.  This module only
    reports and prepares software readiness; it never launches a controller or
    publishes a robot command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Mapping, Optional, Protocol

from .protocol import SONIC_WBC


SONIC_REQUIRED_COMPONENTS = (
    "task_checkpoint",
    "observation_adapter",
    "controller_binary",
    "controller_transport",
    "action_bridge",
)


@dataclass(frozen=True)
class SonicControllerReadiness:
    """Readiness declaration supplied by an injected SONIC supervisor."""

    configured: bool = False
    components: Mapping[str, bool] = field(default_factory=dict)
    execution_enabled: bool = False
    simulated: bool = False
    reasons: List[str] = field(default_factory=list)

    @property
    def components_ready(self) -> bool:
        return bool(self.configured) and all(
            bool(self.components.get(name, False))
            for name in SONIC_REQUIRED_COMPONENTS
        )

    @property
    def physical_execution_available(self) -> bool:
        return bool(
            self.components_ready
            and self.execution_enabled
            and not self.simulated
        )

    def to_dict(self) -> Dict[str, Any]:
        components = {
            name: bool(self.components.get(name, False))
            for name in SONIC_REQUIRED_COMPONENTS
        }
        reasons = list(self.reasons)
        if not self.configured and "sonic_controller_not_configured" not in reasons:
            reasons.append("sonic_controller_not_configured")
        if self.simulated and "simulation_has_no_physical_actuation" not in reasons:
            reasons.append("simulation_has_no_physical_actuation")
        return {
            "configured": bool(self.configured),
            "components": components,
            "components_ready": self.components_ready,
            "execution_enabled": bool(self.execution_enabled),
            "simulated": bool(self.simulated),
            "physical_execution_available": self.physical_execution_available,
            "reasons": reasons,
        }


@dataclass(frozen=True)
class SonicSupervisorStatus:
    """Snapshot returned by the SONIC software supervisor."""

    state: str
    readiness: SonicControllerReadiness
    prepared: bool = False
    detail: Mapping[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    @property
    def physical_execution_available(self) -> bool:
        return bool(self.prepared and self.readiness.physical_execution_available)

    def to_dict(self, ok: bool = True, operation: str = "status") -> Dict[str, Any]:
        readiness = self.readiness.to_dict()
        reasons = list(readiness["reasons"])
        if self.prepared and not self.physical_execution_available:
            reason = "sonic_physical_execution_not_available"
            if reason not in reasons:
                reasons.append(reason)
        return {
            "ok": bool(ok),
            "kind": (
                "SONIC_WBC_READY"
                if self.physical_execution_available
                else "SONIC_WBC_PREPARED_OPEN_LOOP"
                if self.prepared
                else "SONIC_WBC_UNAVAILABLE"
            ),
            "controller_mode": SONIC_WBC,
            "operation": operation,
            "state": self.state,
            "prepared": bool(self.prepared),
            "readiness": readiness,
            "physical_execution": self.physical_execution_available,
            "lowcmd_published_by_runtime": False,
            "lowcmd_published": False,
            "lowcmd_published_scope": "agent_vla_async_runtime_only",
            "controller_started": False,
            "reasons": reasons,
            "detail": dict(self.detail),
            "updated_at": float(self.updated_at),
        }


class SonicSupervisor(Protocol):
    """Injection boundary for a controller process owned outside this runtime."""

    def status(self) -> Dict[str, Any]:
        ...

    def prepare(self) -> Dict[str, Any]:
        ...

    def stop(self) -> Dict[str, Any]:
        ...


class UnavailableSonicSupervisor:
    """Production default until the external SONIC stack is configured."""

    def __init__(self, reasons: Optional[List[str]] = None) -> None:
        self._reasons = list(
            reasons
            or [
                "task_finetuned_sonic_checkpoint_not_configured",
                "observation_adapter_not_configured",
                "gear_sonic_controller_not_configured",
                "controller_transport_not_configured",
                "sonic_action_bridge_not_configured",
            ]
        )

    def _snapshot(self, state: str = "UNAVAILABLE") -> SonicSupervisorStatus:
        return SonicSupervisorStatus(
            state=state,
            readiness=SonicControllerReadiness(reasons=self._reasons),
            prepared=False,
            detail={"provider": "unconfigured"},
        )

    def status(self) -> Dict[str, Any]:
        return self._snapshot().to_dict()

    def prepare(self) -> Dict[str, Any]:
        return self._snapshot().to_dict(ok=False, operation="prepare")

    def stop(self) -> Dict[str, Any]:
        return self._snapshot(state="STOPPED").to_dict(operation="stop")


class SimulatedSonicSupervisor:
    """In-memory lifecycle for protocol and integration tests.

    It can become prepared, but it deliberately never claims physical
    execution availability and has no command-publishing method.
    """

    def __init__(
        self,
        components: Optional[Mapping[str, bool]] = None,
    ) -> None:
        self._components = dict(
            components
            or {name: True for name in SONIC_REQUIRED_COMPONENTS}
        )
        self._prepared = False

    def _snapshot(self) -> SonicSupervisorStatus:
        readiness = SonicControllerReadiness(
            configured=True,
            components=self._components,
            execution_enabled=False,
            simulated=True,
        )
        return SonicSupervisorStatus(
            state="PREPARED" if self._prepared else "STOPPED",
            readiness=readiness,
            prepared=self._prepared,
            detail={"provider": "in_memory_simulation"},
        )

    def status(self) -> Dict[str, Any]:
        return self._snapshot().to_dict()

    def prepare(self) -> Dict[str, Any]:
        snapshot = self._snapshot()
        if not snapshot.readiness.components_ready:
            return snapshot.to_dict(ok=False, operation="prepare")
        self._prepared = True
        return self._snapshot().to_dict(operation="prepare")

    def stop(self) -> Dict[str, Any]:
        self._prepared = False
        return self._snapshot().to_dict(operation="stop")
