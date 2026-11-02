"""Execution boundary for structured hybrid Agent -> G1 plans.

This module only wraps the already-audited high-level navigation adapter.  It
does not translate GR00T ``REAL_G1`` tensors into robot motion and it does not
implement arm control; named arm actions are dispatched by the separate SDK2
arm adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .protocol import NavigationPlan


@dataclass(frozen=True)
class NavigationStepInspection:
    ok: bool
    reason: str
    move_rc: Optional[int]
    stop_rc: Optional[int]
    postflight_ready: Optional[bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "move_rc": self.move_rc,
            "stop_rc": self.stop_rc,
            "postflight_ready": self.postflight_ready,
        }


def _controller_result(result: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Return the SPORT payload and whether it came through NavSession.step."""

    controller = result.get("controller")
    if isinstance(controller, dict):
        return controller, True
    return result, False


def inspect_navigation_step_result(result: Any) -> NavigationStepInspection:
    """Verify both the NavSession envelope and the underlying SPORT receipt."""

    if not isinstance(result, dict):
        return NavigationStepInspection(False, "step_result_not_an_object", None, None, None)

    controller, wrapped = _controller_result(result)
    if wrapped:
        if result.get("refused"):
            reasons = result.get("reasons") or ["navigation gate refused motion"]
            return NavigationStepInspection(
                False, "navigation_refused:" + "; ".join(map(str, reasons)), None, None, None
            )
        if result.get("ok") is not True:
            return NavigationStepInspection(False, "navigation_step_failed", None, None, None)
        if result.get("moved") is not True:
            return NavigationStepInspection(False, "navigation_step_did_not_move", None, None, None)
    elif controller.get("ok") is not True:
        return NavigationStepInspection(False, "sport_step_failed", None, None, None)

    move_rc = controller.get("move_rc")
    stop_rc = controller.get("stop_rc")
    postflight = controller.get("postflight")
    postflight_ready = postflight.get("ready") if isinstance(postflight, dict) else None

    if move_rc != 0:
        return NavigationStepInspection(False, "move_rc_not_zero", move_rc, stop_rc, postflight_ready)
    if stop_rc != 0:
        return NavigationStepInspection(False, "stop_rc_not_zero", move_rc, stop_rc, postflight_ready)
    if postflight_ready is not True:
        return NavigationStepInspection(
            False, "postflight_not_ready", move_rc, stop_rc, postflight_ready
        )
    return NavigationStepInspection(True, "accepted_stopped_ready", move_rc, stop_rc, True)


class SportNavigationExecutor:
    """Run bounded navigation pulses through an injected navigation adapter.

    Dry-run is the default and does not call ``adapter.step``.  Enabling this
    wrapper is a separate opt-in from configuring the underlying adapter for
    physical execution.
    """

    def __init__(self, navigation: Any, execute_enabled: bool = False) -> None:
        self.navigation = navigation
        self.execute_enabled = bool(execute_enabled)

    async def execute(self, plan: NavigationPlan) -> Dict[str, Any]:
        plan.validate()
        if not self.execute_enabled:
            return {
                "ok": True,
                "kind": "EXECUTION_PREVIEW",
                "dry_run": True,
                "motion_commands_sent": 0,
                "plan": plan.to_dict(),
                "steps": [],
            }

        steps: List[Dict[str, Any]] = []
        for index, pulse in enumerate(plan.pulses):
            try:
                result = await self.navigation.step(pulse)
            except Exception as exc:
                stop_result = await self.navigation.stop()
                return {
                    "ok": False,
                    "kind": "EXECUTION_FAILED",
                    "dry_run": False,
                    "motion_commands_sent": index,
                    "failed_step": index,
                    "reason": "navigation_step_raised:%s: %s" % (type(exc).__name__, exc),
                    "steps": steps,
                    "stop": stop_result,
                }

            inspection = inspect_navigation_step_result(result)
            steps.append(
                {
                    "index": index,
                    "pulse": pulse.to_dict(),
                    "inspection": inspection.to_dict(),
                    "result": result,
                }
            )
            if not inspection.ok:
                stop_result = await self.navigation.stop()
                return {
                    "ok": False,
                    "kind": "EXECUTION_FAILED",
                    "dry_run": False,
                    "motion_commands_sent": index + 1,
                    "failed_step": index,
                    "reason": inspection.reason,
                    "steps": steps,
                    "stop": stop_result,
                }

        return {
            "ok": True,
            "kind": "EXECUTION_COMPLETE",
            "dry_run": False,
            "motion_commands_sent": len(steps),
            "steps": steps,
        }
