import argparse
import unittest
from unittest import mock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "g1"))

import deploy_to_g1 as deploy  # noqa: E402

OLD_UNITS = ("old-thinker.service", "old-agent.service")


class DeploymentMigrationTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(target="/home/unitree/rea-agent")

    @mock.patch.object(deploy, "LEGACY_SERVICE_NAMES", OLD_UNITS)
    @mock.patch.object(deploy, "systemctl_each")
    def test_legacy_services_stop_main_first_then_disable(self, systemctl_each):
        snapshot = {service_name: {"LoadState": "loaded"} for service_name in deploy.LEGACY_SERVICE_NAMES}

        deploy.stop_legacy_services(self.args, snapshot)

        stopped = tuple(reversed(deploy.LEGACY_SERVICE_NAMES))
        self.assertEqual(
            systemctl_each.call_args_list,
            [
                mock.call(self.args, "stop", stopped),
                mock.call(self.args, "disable", stopped),
            ],
        )

    @mock.patch.object(deploy, "LEGACY_SERVICE_NAMES", OLD_UNITS)
    @mock.patch.object(deploy, "systemctl_each")
    def test_failed_replacement_restores_enabled_active_legacy_units(self, systemctl_each):
        snapshot = {
            service_name: {
                "LoadState": "loaded",
                "ActiveState": "active",
                "UnitFileState": "enabled",
            }
            for service_name in deploy.LEGACY_SERVICE_NAMES
        }

        errors = deploy.restore_legacy_services(self.args, snapshot)

        self.assertEqual(errors, [])
        self.assertEqual(
            systemctl_each.call_args_list,
            [
                mock.call(self.args, "stop", (deploy.MAIN_SERVICE_NAME,)),
                mock.call(self.args, "disable", (deploy.MAIN_SERVICE_NAME,)),
                mock.call(self.args, "stop", (deploy.THINKER_SERVICE_NAME,)),
                mock.call(self.args, "disable", (deploy.THINKER_SERVICE_NAME,)),
            ]
            + [
                call
                for service_name in deploy.LEGACY_SERVICE_NAMES
                for call in (
                    mock.call(self.args, "enable", (service_name,)),
                    mock.call(self.args, "start", (service_name,)),
                )
            ],
        )

    @mock.patch.object(deploy, "process_runtime")
    def test_runtime_validation_binds_pid_cgroup_and_working_directory(self, process_runtime):
        service_name = deploy.MAIN_SERVICE_NAME
        control_group = f"/user.slice/user-1000.slice/user@1000.service/{service_name}"
        process_runtime.return_value = {
            "cwd": self.args.target,
            "target": self.args.target,
            "cgroups": [control_group],
        }
        properties = {
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "MainPID": "4242",
            "ControlGroup": control_group,
            "WorkingDirectory": self.args.target,
        }

        runtime = deploy.validate_service_runtime(self.args, service_name, properties)

        self.assertEqual(runtime["main_pid"], 4242)
        self.assertEqual(runtime["control_group"], control_group)
        process_runtime.assert_called_once_with(self.args, 4242)

    @mock.patch.object(deploy, "process_runtime")
    def test_runtime_validation_rejects_an_old_service_behind_the_same_port(self, process_runtime):
        properties = {
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "MainPID": "3131",
            "ControlGroup": "/user.slice/old-agent.service",
            "WorkingDirectory": "/home/unitree/old-agent",
        }

        with self.assertRaisesRegex(deploy.DeployError, "WorkingDirectory"):
            deploy.validate_service_runtime(self.args, deploy.MAIN_SERVICE_NAME, properties)

        process_runtime.assert_not_called()


class DeploymentDefaultsTests(unittest.TestCase):
    def test_host_and_primary_model_are_required_and_target_follows_the_user(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                deploy.parser().parse_args(["--primary-model", "vision-model"])
            with self.assertRaises(SystemExit):
                deploy.parser().parse_args(["--host", "192.0.2.5"])
        with mock.patch.dict("os.environ", {"G1_PRIMARY_MODEL": "vision-model"}, clear=True):
            args = deploy.parser().parse_args(["--host", "192.0.2.5"])
            self.assertEqual((args.primary_model, args.fallback_model), ("vision-model", ""))
            deploy.apply_defaults(args)
            self.assertEqual(args.user, "unitree")
            self.assertEqual(args.target, "/home/unitree/rea-agent")
            args = deploy.parser().parse_args(["--host", "192.0.2.5", "--user", "robot"])
            deploy.apply_defaults(args)
            self.assertEqual(args.target, "/home/robot/rea-agent")
        with mock.patch.dict("os.environ", {"G1_DEPLOY_HOST": "192.0.2.9", "G1_DEPLOY_TARGET": "/opt/agent", "G1_PRIMARY_MODEL": "vision-model"}, clear=True):
            args = deploy.parser().parse_args([])
            deploy.apply_defaults(args)
            self.assertEqual(args.host, "192.0.2.9")
            self.assertEqual(args.target, "/opt/agent")

    def test_the_bundle_ships_the_scripts_directory(self):
        self.assertNotIn("run_task.sh", deploy.OPTIONAL_BUNDLE_FILES)
        self.assertIn("scripts", deploy.OPTIONAL_BUNDLE_FILES)


if __name__ == "__main__":
    unittest.main()
