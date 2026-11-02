"""Parallel shards cover the frozen panel and never report partial runs as final."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "eval/robotwin/hybrid/parallel.py"
SPEC = importlib.util.spec_from_file_location("robotwin_hybrid_parallel", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ParallelTest(unittest.TestCase):
    def prepare(self, root, workers=8, count=100):
        configs = []
        for index in range(workers):
            path = root / f"config-{index}.json"
            MODULE.write_json(path, {"simulatorEnv": {"CUDA_VISIBLE_DEVICES": str(index)}})
            configs.append(path)
        instances = [{"instance_id": f"episode-{index}", "setting": "clean" if index < count // 2 else "randomized"}
                     for index in range(count)]
        MODULE.write_json(root / "input.json", instances)
        return MODULE.prepare(configs, root / "input.json", root / "output", "code")

    def result(self, plan, index, success=True, reason="success"):
        directory = Path(plan["shards"][index % plan["workers"]]["output"]) / f"episode-{index}"
        directory.mkdir(exist_ok=True)
        MODULE.write_json(directory / "result.json", {"success": success, "reason": reason})

    def test_eight_shards_cover_exactly_the_original_hundred(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, instances = self.prepare(Path(temporary))
            assigned = [item for shard in plan["shards"] for item in shard["instanceIds"]]
            self.assertEqual(len(assigned), 100)
            self.assertEqual(set(assigned), {row["instance_id"] for row in instances})
            self.assertEqual([shard["requested"] for shard in plan["shards"]], [13] * 4 + [12] * 4)
            self.assertEqual(len({shard["output"] for shard in plan["shards"]}), 8)

    def test_pending_and_error_rows_stay_in_full_panel(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, instances = self.prepare(Path(temporary), workers=2, count=4)
            self.result(plan, 0)
            self.result(plan, 2, success=None, reason="infrastructure_error")
            summary = MODULE.summarize(plan, instances, [0, None])
            group = summary["groups"][-1]
            self.assertEqual((group["total"], group["completed"], group["errors"], group["pending"]), (4, 1, 1, 2))
            self.assertIsNone(group["fullPanelSuccessRate"])
            self.assertEqual([row["instance"] for row in summary["results"]], instances)

    def test_full_panel_rate_requires_all_workers_to_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, instances = self.prepare(Path(temporary), workers=2, count=4)
            for index in range(4):
                self.result(plan, index, success=index != 3, reason="success" if index != 3 else "environment_limit")
            self.assertIsNone(MODULE.summarize(plan, instances, [0, None])["groups"][-1]["fullPanelSuccessRate"])
            groups = MODULE.summarize(plan, instances, [0, 0])["groups"]
            self.assertEqual([group["fullPanelSuccessRate"] for group in groups], [1.0, 0.5, 0.75])
            self.assertEqual(groups[-1]["pending"], 0)

    def test_vla_workers_need_independent_services_but_code_can_share(self):
        configs = [{"vlaEndpoints": {candidate: f"ws://localhost:{8900 + index}" for candidate in MODULE.VLA_IDS}}
                   for index in range(2)]
        MODULE.check_vla_endpoints(configs, "hybrid")
        configs[1] = configs[0]
        with self.assertRaisesRegex(ValueError, "share VLA service"):
            MODULE.check_vla_endpoints(configs, "vla")
        MODULE.check_vla_endpoints(configs, "code")

    def test_static_skip_errors_flag_reaches_each_episode_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, instances = self.prepare(root, workers=2, count=4)
            configs = [root / f"config-{index}.json" for index in range(2)]
            plan, _ = MODULE.prepare(configs, root / "input.json", root / "output", "code", skip_errors=True)
            self.assertTrue(all(shard["command"][-1] == "--skip-errors" for shard in plan["shards"]))
            for index in range(4):
                self.result(plan, index, success=None if index == 0 else True,
                            reason="infrastructure_error" if index == 0 else "success")
            summary = MODULE.summarize(plan, instances, [0, 0])
            self.assertEqual(summary["groups"][-1]["errors"], 1)
            self.assertIsNone(summary["groups"][-1]["fullPanelSuccessRate"])


if __name__ == "__main__":
    unittest.main()
