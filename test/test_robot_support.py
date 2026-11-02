import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "inspect_robot_support.py"
specification = importlib.util.spec_from_file_location("inspect_robot_support", SOURCE)
support = importlib.util.module_from_spec(specification)
specification.loader.exec_module(support)


class RobotSupportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()

    def manifest(self, directory, value):
        path = self.root / "robots" / directory / "robot.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
        return path

    def robot_package(self, directory, name):
        package = self.root / "robots" / directory
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("raise RuntimeError('Robot packages must not be imported')\n")
        (package / "robot_spec.py").write_text(
            "from unavailable_robot_sdk import RobotSpec\n"
            "raise RuntimeError('Robot specifications must not execute')\n"
            f"def get_robot_spec():\n    return RobotSpec(name={name!r}, prompts=None)\n"
        )
        return package

    def test_openral_reports_all_manifests_and_preserves_directory_identity(self):
        self.manifest("dual_arm", {
            "name": "dual_arm_v2", "embodiment_kind": "bimanual",
            "hal": {"sim": "unavailable:SimHAL", "real": "unavailable:RealHAL"},
        })
        self.manifest("derived_arm", {"name": "derived_arm", "hal": {"sim": None}, "sim": {}})
        self.manifest("scene_arm", {"name": "scene_arm"})
        with patch.dict(sys.modules, {"yaml": SimpleNamespace(safe_load=json.loads)}):
            inventory = support.inspect_openral(self.root)
        self.assertEqual(inventory["count"], 3)
        robots = {robot["id"]: robot for robot in inventory["robots"]}
        self.assertEqual(robots["dual_arm"]["name"], "dual_arm_v2")
        self.assertEqual(robots["dual_arm"]["manifest"], "robots/dual_arm/robot.yaml")
        self.assertEqual(robots["dual_arm"]["path"], str(self.root / "robots/dual_arm/robot.yaml"))
        self.assertEqual(robots["dual_arm"]["hal"], {"sim": "unavailable:SimHAL", "real": "unavailable:RealHAL"})
        self.assertFalse(robots["dual_arm"]["sceneRequired"])
        self.assertTrue(robots["derived_arm"]["sim"]["configured"])
        self.assertFalse(robots["derived_arm"]["sceneRequired"])
        self.assertTrue(robots["scene_arm"]["sceneRequired"])

    def test_asset_presence_checks_local_paths_without_resolving_external_sdks(self):
        path = self.manifest("arm", {
            "name": "arm", "sim": {}, "assets": {
                "urdf": {"ref": "file:arm.urdf"},
                "srdf": "file:shared/arm.srdf",
                "mjcf": "rd:external_description",
                "missing": "file:missing.xml",
            },
        })
        (path.parent / "arm.urdf").write_text("<robot/>")
        (self.root / "shared").mkdir()
        (self.root / "shared/arm.srdf").write_text("<robot/>")
        with patch.dict(sys.modules, {"yaml": SimpleNamespace(safe_load=json.loads)}):
            assets = support.inspect_openral(self.root)["robots"][0]["assets"]
        self.assertTrue(assets["urdf"]["exists"])
        self.assertEqual(assets["urdf"]["path"], str(path.parent / "arm.urdf"))
        self.assertTrue(assets["srdf"]["exists"])
        self.assertEqual(assets["srdf"]["path"], str(self.root / "shared/arm.srdf"))
        self.assertEqual(assets["mjcf"], {
            "ref": "rd:external_description", "path": None, "exists": None, "resolution": "external_not_checked",
        })
        self.assertFalse(assets["missing"]["exists"])

    def test_rpent_reads_declared_names_and_server_presence_without_importing_packages(self):
        first = self.robot_package("backend_one", "declared_robot")
        (first / "env_server.py").write_text("raise RuntimeError('Do not start a server')\n")
        (first / "env_client.py").write_text("")
        second = self.robot_package("backend_two", "another_robot")
        (second / "vla_server.py").write_text("")
        self.robot_package("_private", "hidden")
        (self.root / "robots/documentation_only").mkdir()
        (self.root / "robots/documentation_only/README.md").write_text("Planned robot")
        inventory = support.inspect_rpent(self.root)
        self.assertEqual(inventory["count"], 2)
        self.assertEqual([robot["name"] for robot in inventory["robots"]], ["declared_robot", "another_robot"])
        self.assertEqual(inventory["robots"][0], {
            "id": "backend_one", "name": "declared_robot", "path": str(first), "spec": "robots/backend_one/robot_spec.py",
            "envServer": True, "envClient": True, "vlaServer": False, "vlaClient": False,
        })
        self.assertTrue(inventory["robots"][1]["vlaServer"])

    def test_rpent_cli_produces_json_without_yaml_or_robot_imports(self):
        self.robot_package("sample", "sample")
        process = subprocess.run(
            [sys.executable, "-S", str(SOURCE), "--rpent-root", str(self.root)],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        inventory = json.loads(process.stdout)
        self.assertEqual(list(inventory), ["rpent"])
        self.assertEqual(inventory["rpent"]["count"], 1)
        self.assertEqual(inventory["rpent"]["robots"][0]["name"], "sample")


if __name__ == "__main__":
    unittest.main()
