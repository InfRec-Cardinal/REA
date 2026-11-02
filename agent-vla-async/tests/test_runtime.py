import asyncio
import tempfile
import unittest

from agent_vla_async.controller import SimulatedSonicSupervisor
from agent_vla_async.navigation import FakeNavigationAdapter
from agent_vla_async.policy import MockPolicy
from agent_vla_async.protocol import HYBRID_HIGH_LEVEL, SONIC_WBC
from agent_vla_async.runtime import CerebellumRuntime


def navigation_plan():
    return {
        "source": "agent_plan",
        "task": "bounded test plan",
        "navigation": {
            "target": "test",
            "pulses": [{"vx": 0.08, "yaw": 0.0, "duration_s": 0.2}],
        },
        "arm_steps": [],
    }


class FakeAction:
    def __init__(self, name):
        self.name = name

    def to_dict(self):
        return {"name": self.name}


class FakeArm:
    def __init__(self):
        self.executed = []
        self.closed = False

    def list_actions(self, refresh=False):
        return {
            "forward_push": FakeAction("forward_push"),
            "release_arm": FakeAction("release_arm"),
        }

    def resolve_action(self, name):
        return FakeAction(name)

    def execute(self, name, wait=True, timeout_s=None):
        self.executed.append(name)
        return {
            "ok": True,
            "dry_run": False,
            "executed": True,
            "action": {"name": name},
            "rc": 0,
        }

    def close(self):
        self.closed = True


class FakeSonicPolicy(MockPolicy):
    name = "fake-sonic"

    def __init__(self):
        super().__init__()
        self.validated = []

    def sonic_status(self):
        return {
            "ok": True,
            "kind": "SONIC_POLICY_AVAILABLE",
            "scope": "POLICY_OPEN_LOOP",
            "contract_compatible": True,
            "physical_execution": False,
            "controller_mode": "not_connected",
            "reasons": [],
        }

    def validate_open_loop(self, task, rgb):
        self.validated.append((task, rgb))
        return {
            "ok": True,
            "kind": "SONIC_OPEN_LOOP_VALIDATION_COMPLETE",
            "scope": "POLICY_OPEN_LOOP",
            "state_source": "synthetic_neutral_placeholder",
            "action_shapes": {
                "motion_token": [1, 40, 64],
                "left_hand_joints": [1, 40, 7],
                "right_hand_joints": [1, 40, 7],
            },
            "absolute_max": {
                "motion_token": 0.0,
                "left_hand_joints": 0.0,
                "right_hand_joints": 0.0,
            },
            "action_publisher_started_by_runtime": False,
            "lowcmd_published_by_runtime": False,
            "physical_commands_sent": 0,
        }


async def wait_for(predicate, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached")


class RuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.navigation = FakeNavigationAdapter()
        self.runtime = CerebellumRuntime(
            navigation=self.navigation,
            policy=MockPolicy(delay_s=0.03),
            socket_path=self.temp.name + "/control.sock",
            execute_enabled=False,
        )
        self.task = asyncio.create_task(self.runtime.run())
        await wait_for(lambda: any(event.kind == "READY" for event in self.runtime.events))

    async def asyncTearDown(self):
        await self.runtime.close()
        await self.task
        self.temp.cleanup()

    async def test_dry_run_previews_without_calling_motion(self):
        submitted = await self.runtime.handle(
            {
                "op": "submit",
                "task": "walk toward the door",
                "heartbeat_timeout_s": 1.0,
                "inference_period_s": 0.05,
            }
        )
        self.assertTrue(submitted["ok"])
        await wait_for(lambda: self.runtime.status()["preview_pulse_count"] > 0)
        self.assertEqual([], self.navigation.steps)

    async def test_new_intent_discards_old_inference(self):
        first = await self.runtime.handle(
            {"op": "submit", "task": "first", "heartbeat_timeout_s": 1.0}
        )
        await asyncio.sleep(0.01)
        second = await self.runtime.handle(
            {"op": "submit", "task": "second", "heartbeat_timeout_s": 1.0}
        )
        await wait_for(lambda: self.runtime.status()["discarded_inference_count"] > 0)
        await wait_for(
            lambda: self.runtime.status()["last_output"] is not None
            and self.runtime.status()["last_output"]["intent_id"] == second["intent"]["intent_id"]
        )
        self.assertNotEqual(first["intent"]["intent_id"], second["intent"]["intent_id"])

    async def test_superseded_plan_stops_waiting_without_deactivating_replacement(self):
        self.runtime.policy = MockPolicy(delay_s=0.5)
        first = await self.runtime.handle(
            {
                "op": "submit",
                "task": "old plan",
                "plan": navigation_plan(),
                "heartbeat_timeout_s": 1.0,
                "inference_period_s": 0.01,
                "policy_wait_timeout_s": 0.5,
            }
        )
        await wait_for(
            lambda: any(
                event.kind == "PLAN_STARTED"
                and event.intent_id == first["intent"]["intent_id"]
                for event in self.runtime.events
            )
        )

        second = await self.runtime.handle(
            {
                "op": "submit",
                "task": "replacement plan",
                "plan": navigation_plan(),
                "heartbeat_timeout_s": 1.0,
                "inference_period_s": 0.01,
                "policy_wait_timeout_s": 0.5,
            }
        )
        await wait_for(
            lambda: self.runtime._plan_started_generation == second["generation"],
            timeout=0.2,
        )

        active = self.runtime.status()["active"]
        self.assertIsNotNone(active)
        self.assertEqual(second["intent"]["intent_id"], active["intent_id"])
        self.assertFalse(
            any(
                event.kind == "PREVIEW_FAILED"
                and event.intent_id == second["intent"]["intent_id"]
                for event in self.runtime.events
            )
        )

    async def test_heartbeat_timeout_expires_intent(self):
        await self.runtime.handle(
            {
                "op": "submit",
                "task": "short lease",
                "heartbeat_timeout_s": 0.12,
                "inference_period_s": 0.5,
            }
        )
        await wait_for(lambda: self.runtime.status()["active"] is None, timeout=1.0)
        self.assertTrue(any(event.kind == "EXPIRED" for event in self.runtime.events))

    async def test_superseding_execute_intent_stops_before_replacement(self):
        self.runtime.execute_enabled = True
        first = await self.runtime.handle(
            {
                "op": "submit",
                "task": "first execute intent",
                "execute": True,
                "plan": navigation_plan(),
                "heartbeat_timeout_s": 1.0,
            }
        )
        self.assertTrue(first["ok"])
        second = await self.runtime.handle(
            {"op": "submit", "task": "replacement", "heartbeat_timeout_s": 1.0}
        )
        self.assertTrue(second["ok"])
        self.assertGreaterEqual(self.navigation.stops, 1)
        self.assertTrue(any(event.kind == "SUPERSEDE_STOPPED" for event in self.runtime.events))

    async def test_execute_requires_a_structured_plan(self):
        self.runtime.execute_enabled = True
        response = await self.runtime.handle(
            {"op": "submit", "task": "raw prompt", "execute": True}
        )
        self.assertFalse(response["ok"])
        self.assertEqual("EXECUTION_UNAVAILABLE", response["kind"])
        self.assertEqual(0, response["motion_commands_sent"])

    async def test_plan_can_skip_the_advisory_wait(self):
        self.runtime.execute_enabled = True
        self.runtime.policy = MockPolicy(delay_s=0.8)
        started = asyncio.get_running_loop().time()
        response = await self.runtime.handle(
            {
                "op": "submit",
                "task": "bounded pulse without advisory wait",
                "execute": True,
                "plan": navigation_plan(),
                "heartbeat_timeout_s": 2.0,
                "policy_wait_timeout_s": 5.0,
                "wait_for_policy": False,
            }
        )
        self.assertTrue(response["ok"])
        self.assertFalse(response["intent"]["wait_for_policy"])
        await wait_for(
            lambda: self.runtime.status()["last_plan_result"] is not None,
            timeout=1.0,
        )
        elapsed = asyncio.get_running_loop().time() - started
        self.assertLess(elapsed, 0.6, "the plan did not wait for the 0.8 s policy")
        result = self.runtime.status()["last_plan_result"]
        self.assertEqual("EXECUTION_COMPLETE", result["kind"])
        self.assertEqual(1, result["motion_commands_sent"])
        self.assertEqual(1, len(self.navigation.steps))
        kinds = [event.kind for event in self.runtime.events]
        self.assertIn("POLICY_ADVISORY_SKIPPED", kinds)
        self.assertNotIn("POLICY_ADVISORY_READY", kinds)

    async def test_structured_plan_preview_never_calls_navigation(self):
        response = await self.runtime.handle(
            {
                "op": "submit",
                "task": "preview bounded plan",
                "plan": navigation_plan(),
                "heartbeat_timeout_s": 1.0,
                "inference_period_s": 0.05,
            }
        )
        self.assertTrue(response["ok"])
        await wait_for(
            lambda: self.runtime.status()["last_plan_result"] is not None,
            timeout=1.0,
        )
        result = self.runtime.status()["last_plan_result"]
        self.assertEqual("PREVIEW_COMPLETE", result["kind"])
        self.assertEqual(0, result["motion_commands_sent"])
        self.assertEqual([], self.navigation.steps)

    async def test_plan_wait_uses_the_submitted_policy_timeout(self):
        self.runtime.policy = MockPolicy(delay_s=0.20)
        response = await self.runtime.handle(
            {
                "op": "submit",
                "task": "preview with a short policy deadline",
                "plan": navigation_plan(),
                "heartbeat_timeout_s": 1.0,
                "inference_period_s": 0.5,
                "policy_wait_timeout_s": 0.05,
            }
        )
        self.assertEqual(0.05, response["intent"]["policy_wait_timeout_s"])
        await wait_for(
            lambda: self.runtime.status()["last_plan_result"] is not None,
            timeout=0.5,
        )
        result = self.runtime.status()["last_plan_result"]
        self.assertEqual("PREVIEW_FAILED", result["kind"])
        self.assertIn("within 0.050s", result["reason"])

    async def test_plan_wait_defaults_to_the_daemon_policy_timeout(self):
        self.runtime.policy_wait_timeout_s = 23.0
        response = await self.runtime.handle(
            {
                "op": "submit",
                "task": "preview with daemon default",
                "heartbeat_timeout_s": 1.0,
            }
        )
        self.assertEqual(23.0, response["intent"]["policy_wait_timeout_s"])

    async def test_capabilities_separate_readiness_from_dry_run_mode(self):
        response = await self.runtime.handle({"op": "capabilities"})
        self.assertTrue(response["ok"])
        self.assertEqual("dry_run", response["service_mode"])
        self.assertTrue(response["components_ready"] is False)
        sport = next(
            item for item in response["capability_status"]
            if item["name"] == "sport_navigation"
        )
        self.assertTrue(sport["available"])

    async def test_sonic_status_does_not_enable_a_controller(self):
        response = await self.runtime.handle({"op": "sonic_status"})
        self.assertTrue(response["ok"])
        self.assertEqual("SONIC_POLICY_UNAVAILABLE", response["kind"])
        self.assertFalse(response["physical_execution"])
        self.assertEqual("not_connected", response["controller_mode"])

    async def test_controller_status_exposes_both_modes_without_changing_default(self):
        response = await self.runtime.handle({"op": "controller_status"})
        self.assertTrue(response["ok"])
        self.assertEqual("g1.controller-modes/v1", response["schema"])
        self.assertEqual(HYBRID_HIGH_LEVEL, response["controller_mode"])
        self.assertTrue(response["modes"][HYBRID_HIGH_LEVEL]["selected"])
        sonic = response["modes"][SONIC_WBC]
        self.assertEqual("SONIC_WBC_UNAVAILABLE", sonic["kind"])
        self.assertFalse(sonic["physical_execution"])
        self.assertFalse(response["lowcmd_published_by_runtime"])

    async def test_sonic_probe_is_independent_from_the_main_advisory_policy(self):
        self.runtime.policy = FakeSonicPolicy()
        response = await self.runtime.handle({"op": "sonic_status"})
        self.assertEqual("SONIC_POLICY_UNAVAILABLE", response["kind"])
        self.assertIn(
            "dedicated_sonic_policy_probe_not_configured", response["reasons"]
        )

    async def test_sonic_validate_uses_only_the_dedicated_open_loop_client(self):
        sonic = FakeSonicPolicy()
        self.runtime.sonic_policy = sonic
        response = await self.runtime.handle(
            {"op": "sonic_validate", "task": "touch the door"}
        )
        self.assertTrue(response["ok"])
        self.assertEqual("SONIC_OPEN_LOOP_VALIDATION_COMPLETE", response["kind"])
        self.assertEqual(1, len(sonic.validated))
        self.assertFalse(response["action_publisher_started_by_runtime"])
        self.assertEqual(0, response["physical_commands_sent"])

    async def test_unconfigured_sonic_prepare_is_unavailable_and_keeps_hybrid(self):
        response = await self.runtime.handle(
            {"op": "controller_prepare", "controller_mode": SONIC_WBC}
        )
        self.assertFalse(response["ok"])
        self.assertEqual("EXECUTION_UNAVAILABLE", response["kind"])
        self.assertEqual(HYBRID_HIGH_LEVEL, self.runtime.status()["controller_mode"])
        self.assertFalse(response["physical_execution"])
        self.assertEqual([], self.navigation.steps)

    async def test_simulated_sonic_mode_is_open_loop_and_stoppable(self):
        self.runtime.sonic_policy = FakeSonicPolicy()
        self.runtime.sonic_supervisor = SimulatedSonicSupervisor()
        prepared = await self.runtime.handle(
            {"op": "controller_prepare", "controller_mode": SONIC_WBC}
        )
        self.assertTrue(prepared["ok"])
        self.assertEqual(SONIC_WBC, prepared["controller_mode"])
        self.assertEqual(
            "SONIC_WBC_PREPARED_OPEN_LOOP",
            prepared["modes"][SONIC_WBC]["kind"],
        )
        self.assertFalse(prepared["physical_execution"])

        refused = await self.runtime.handle(
            {
                "op": "submit",
                "task": "SONIC physical task",
                "controller_mode": SONIC_WBC,
                "execute": True,
            }
        )
        self.assertFalse(refused["ok"])
        self.assertEqual("EXECUTION_UNAVAILABLE", refused["kind"])
        self.assertEqual(0, refused["motion_commands_sent"])
        self.assertEqual([], self.navigation.steps)

        stopped = await self.runtime.handle(
            {"op": "controller_stop", "controller_mode": SONIC_WBC}
        )
        self.assertTrue(stopped["ok"])
        self.assertEqual(HYBRID_HIGH_LEVEL, stopped["controller_mode"])

    async def test_execute_plan_uses_explicit_sport_and_named_arm_steps(self):
        arm = FakeArm()
        self.runtime.arm = arm
        self.runtime.execute_enabled = True
        plan = navigation_plan()
        plan["arm_steps"] = [
            {"action": "forward_push"},
            {"action": "release_arm"},
        ]
        response = await self.runtime.handle(
            {
                "op": "submit",
                "task": "execute explicit hybrid plan",
                "plan": plan,
                "execute": True,
                "heartbeat_timeout_s": 1.0,
                "inference_period_s": 0.05,
            }
        )
        self.assertTrue(response["ok"])
        self.assertEqual("EXECUTION_AVAILABLE", response["kind"])
        await wait_for(
            lambda: self.runtime.status()["last_plan_result"] is not None,
            timeout=1.0,
        )
        result = self.runtime.status()["last_plan_result"]
        self.assertEqual("EXECUTION_COMPLETE", result["kind"])
        self.assertEqual(1, result["motion_commands_sent"])
        self.assertEqual(2, result["arm_actions_sent"])
        self.assertEqual(["forward_push", "release_arm"], arm.executed)
        self.assertEqual(1, len(self.navigation.steps))

    async def test_preview_with_execute_backend_only_resolves_arm_actions(self):
        arm = FakeArm()
        self.runtime.arm = arm
        self.runtime.execute_enabled = True
        plan = navigation_plan()
        plan["arm_steps"] = [
            {"action": "forward_push"},
            {"action": "release_arm"},
        ]
        response = await self.runtime.handle(
            {
                "op": "submit",
                "task": "preview explicit hybrid plan",
                "plan": plan,
                "execute": False,
                "heartbeat_timeout_s": 1.0,
                "inference_period_s": 0.05,
            }
        )
        self.assertTrue(response["ok"])
        self.assertEqual("PREVIEW_ACCEPTED", response["kind"])
        await wait_for(
            lambda: self.runtime.status()["last_plan_result"] is not None,
            timeout=1.0,
        )
        result = self.runtime.status()["last_plan_result"]
        self.assertEqual("PREVIEW_COMPLETE", result["kind"])
        self.assertEqual(0, result["motion_commands_sent"])
        self.assertEqual(0, result["arm_actions_sent"])
        self.assertEqual([], self.navigation.steps)
        self.assertEqual([], arm.executed)
        self.assertTrue(all(step["dry_run"] for step in result["arm"]["steps"]))


if __name__ == "__main__":
    unittest.main()
