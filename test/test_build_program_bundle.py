import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("build_program_bundle", Path(__file__).resolve().parents[1] / "tools/build_program_bundle.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class BuildProgramBundleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.library = self.root / "library"
        self.output = self.root / "bundle"
        self.programs = {
            ("libero_goal_task", "put_the_bowl_on_the_stove", "sdk"): b"# untouched line endings\r\nopen_gripper()\r\n",
            ("libero_goal_task", "put_the_bowl_on_the_stove", "harness"): b"close_gripper()\n",
            ("libero_goal_task", "put_the_bowl_on_the_stove", "copy"): b"close_gripper()\n",
            ("libero_spatial_task", "pick_the_bowl", "sdk"): b"goto_home_joint_position()\n",
        }
        for (suite, task, name), code in self.programs.items():
            path = self.library / suite / task / f"{name}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(code)

    def test_copies_program_bytes_and_records_their_digests(self):
        bundle = prepare.prepare_bundle(self.library, self.output, created_at="2026-01-01T00:00:00+00:00")
        self.assertEqual(bundle["schema"], "rea.program-bundle/v1")
        self.assertEqual([(task["suite"], task["task"]) for task in bundle["tasks"]],
                         [("libero_goal_task", "put_the_bowl_on_the_stove"), ("libero_spatial_task", "pick_the_bowl")])
        first = bundle["tasks"][0]["candidates"]
        self.assertEqual(set(first), {"copy", "sdk"}, "a byte-identical candidate is kept once")
        self.assertEqual((self.output / first["sdk"]["path"]).read_bytes(), self.programs[("libero_goal_task", "put_the_bowl_on_the_stove", "sdk")])
        written = json.loads((self.output / "bundle.json").read_text())
        self.assertEqual(written, bundle)

    def test_tasks_file_selects_and_orders_tasks(self):
        tasks = self.root / "tasks.json"
        tasks.write_text(json.dumps([{"suite": "libero_spatial_task", "task": "pick_the_bowl"}]))
        bundle = prepare.prepare_bundle(self.library, self.output, tasks)
        self.assertEqual([task["task"] for task in bundle["tasks"]], ["pick_the_bowl"])

    def test_refuses_missing_tasks_and_existing_bundles(self):
        tasks = self.root / "tasks.json"
        tasks.write_text(json.dumps([{"suite": "libero_goal_task", "task": "missing"}]))
        with self.assertRaisesRegex(ValueError, "No program candidates"):
            prepare.prepare_bundle(self.library, self.output, tasks)
        prepare.prepare_bundle(self.library, self.output)
        with self.assertRaises(FileExistsError):
            prepare.prepare_bundle(self.library, self.output)


if __name__ == "__main__":
    unittest.main()
