import asyncio
import json
import unittest

from agent_vla_async.execution import (
    SportNavigationExecutor,
    inspect_navigation_step_result,
)
from agent_vla_async.protocol import (
    ArmActionStep,
    Capability,
    ExecutionAvailability,
    ExecutionPlan,
    FORWARD_PUSH,
    HYBRID_HIGH_LEVEL,
    Intent,
    MotionPulse,
    NavigationPlan,
    SPORT_NAVIGATION,
    SONIC_WBC,
    normalize_controller_mode,
)


class FakeNavigation:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.steps = []
        self.stops = 0

    async def step(self, pulse):
        self.steps.append(pulse)
        return self.results.pop(0)

    async def stop(self):
        self.stops += 1
        return {"ok": True, "stop_rc": 0}


def successful_nav_session_result():
    return {
        "ok": True,
        "moved": True,
        "refused": False,
        "controller": {
            "ok": True,
            "move_rc": 0,
            "stop_rc": 0,
            "postflight": {"ready": True},
        },
    }


class ProtocolTests(unittest.TestCase):
    def test_hybrid_plan_and_availability_are_json_facing(self):
        plan = ExecutionPlan(
            source="agent_hybrid_planner",
            task="walk to the door and push it",
            navigation=NavigationPlan([MotionPulse(0.1, 0.0, 0.2)], target="door"),
            arm_steps=[ArmActionStep("forward_push")],
        )
        plan.validate()
        self.assertEqual([SPORT_NAVIGATION, FORWARD_PUSH], plan.required_capabilities)

        availability = ExecutionAvailability(
            [Capability(SPORT_NAVIGATION), Capability(FORWARD_PUSH)],
            execute_enabled=True,
        )
        value = json.loads(availability.to_json())
        self.assertEqual("EXECUTION_AVAILABLE", value["kind"])
        self.assertEqual("HYBRID_HIGH_LEVEL", value["scope"])
        self.assertEqual([SPORT_NAVIGATION, FORWARD_PUSH], value["capabilities"])
        self.assertTrue(availability.supports(plan))
        self.assertFalse(value["dry_run"])
        self.assertFalse(value["direct_vla_actuation"])

    def test_dry_run_reports_components_ready_but_execution_unavailable(self):
        availability = ExecutionAvailability(
            [Capability(SPORT_NAVIGATION), Capability(FORWARD_PUSH)],
            execute_enabled=False,
        )
        value = availability.to_dict()
        self.assertEqual("EXECUTION_UNAVAILABLE", value["kind"])
        self.assertTrue(value["components_ready"])
        self.assertTrue(value["dry_run"])

    def test_real_g1_policy_output_is_not_an_execution_plan(self):
        plan = ExecutionPlan(
            source="groot_real_g1",
            navigation=NavigationPlan([MotionPulse(0.1, 0.0, 0.2)]),
        )
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            plan.validate()

    def test_plan_parser_enforces_the_high_level_sport_envelope(self):
        with self.assertRaisesRegex(ValueError, "vx envelope"):
            ExecutionPlan.from_dict(
                {
                    "source": "agent_plan",
                    "navigation": {
                        "pulses": [{"vx": 0.21, "yaw": 0, "duration_s": 0.2}]
                    },
                }
            )

    def test_controller_mode_is_explicit_in_the_intent_protocol(self):
        intent = Intent.create("preview")
        self.assertEqual(HYBRID_HIGH_LEVEL, intent.controller_mode)
        self.assertEqual(SONIC_WBC, normalize_controller_mode("sonic"))
        with self.assertRaisesRegex(ValueError, "controller_mode"):
            normalize_controller_mode("unknown")


class NavigationResultTests(unittest.TestCase):
    def test_accepts_existing_nav_session_envelope(self):
        checked = inspect_navigation_step_result(successful_nav_session_result())
        self.assertTrue(checked.ok)
        self.assertEqual("accepted_stopped_ready", checked.reason)

    def test_accepts_direct_controller_receipt(self):
        checked = inspect_navigation_step_result(
            {"ok": True, "move_rc": 0, "stop_rc": 0, "postflight": {"ready": True}}
        )
        self.assertTrue(checked.ok)

    def test_rejects_incomplete_receipt(self):
        checked = inspect_navigation_step_result({"ok": True, "moved": True})
        self.assertFalse(checked.ok)
        self.assertEqual("move_rc_not_zero", checked.reason)


class SportNavigationExecutorTests(unittest.TestCase):
    def test_default_dry_run_does_not_call_navigation(self):
        navigation = FakeNavigation()
        executor = SportNavigationExecutor(navigation)
        plan = NavigationPlan([MotionPulse(0.1, 0.0, 0.2)])

        result = asyncio.run(executor.execute(plan))

        self.assertTrue(result["ok"])
        self.assertEqual("EXECUTION_PREVIEW", result["kind"])
        self.assertEqual(0, result["motion_commands_sent"])
        self.assertEqual([], navigation.steps)

    def test_enabled_executor_requires_full_sport_receipt(self):
        navigation = FakeNavigation([successful_nav_session_result()])
        executor = SportNavigationExecutor(navigation, execute_enabled=True)
        plan = NavigationPlan([MotionPulse(0.1, 0.0, 0.2)])

        result = asyncio.run(executor.execute(plan))

        self.assertTrue(result["ok"])
        self.assertEqual("EXECUTION_COMPLETE", result["kind"])
        self.assertEqual(1, result["motion_commands_sent"])
        self.assertEqual(0, navigation.stops)

    def test_failed_receipt_stops_and_fails_closed(self):
        bad = successful_nav_session_result()
        bad["controller"]["postflight"]["ready"] = False
        navigation = FakeNavigation([bad])
        executor = SportNavigationExecutor(navigation, execute_enabled=True)
        plan = NavigationPlan([MotionPulse(0.1, 0.0, 0.2)])

        result = asyncio.run(executor.execute(plan))

        self.assertFalse(result["ok"])
        self.assertEqual("postflight_not_ready", result["reason"])
        self.assertEqual(1, navigation.stops)


if __name__ == "__main__":
    unittest.main()
