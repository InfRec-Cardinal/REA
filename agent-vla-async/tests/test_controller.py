import unittest

from agent_vla_async.controller import (
    SONIC_REQUIRED_COMPONENTS,
    SimulatedSonicSupervisor,
    UnavailableSonicSupervisor,
)


class SonicSupervisorTests(unittest.TestCase):
    def test_production_default_is_explicitly_unavailable(self):
        supervisor = UnavailableSonicSupervisor()
        status = supervisor.status()
        self.assertEqual("SONIC_WBC_UNAVAILABLE", status["kind"])
        self.assertFalse(status["prepared"])
        self.assertFalse(status["physical_execution"])
        self.assertFalse(status["controller_started"])
        self.assertFalse(status["lowcmd_published_by_runtime"])
        self.assertEqual(
            "agent_vla_async_runtime_only", status["lowcmd_published_scope"]
        )
        self.assertIn("sonic_controller_not_configured", status["reasons"])

        prepared = supervisor.prepare()
        self.assertFalse(prepared["ok"])
        self.assertEqual("prepare", prepared["operation"])

    def test_simulated_supervisor_supports_prepare_status_stop(self):
        supervisor = SimulatedSonicSupervisor()
        self.assertFalse(supervisor.status()["prepared"])

        prepared = supervisor.prepare()
        self.assertTrue(prepared["ok"])
        self.assertTrue(prepared["prepared"])
        self.assertEqual("SONIC_WBC_PREPARED_OPEN_LOOP", prepared["kind"])
        self.assertTrue(prepared["readiness"]["components_ready"])
        self.assertFalse(prepared["physical_execution"])
        self.assertFalse(prepared["lowcmd_published_by_runtime"])

        stopped = supervisor.stop()
        self.assertTrue(stopped["ok"])
        self.assertEqual("STOPPED", stopped["state"])
        self.assertFalse(stopped["prepared"])

    def test_simulation_refuses_prepare_when_a_component_is_missing(self):
        components = {name: True for name in SONIC_REQUIRED_COMPONENTS}
        components["controller_binary"] = False
        supervisor = SimulatedSonicSupervisor(components)
        response = supervisor.prepare()
        self.assertFalse(response["ok"])
        self.assertFalse(response["readiness"]["components_ready"])


if __name__ == "__main__":
    unittest.main()
