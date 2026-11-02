"""Real short workers exercise queue balancing and explicit restart semantics."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "eval/robotwin/hybrid/dynamic_parallel.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("robotwin_dynamic_parallel", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

WORKER = '''
import argparse,json,os,pathlib,sys,time
p=argparse.ArgumentParser()
for name in ("config","instances","output","mode"):p.add_argument("--"+name)
a=p.parse_args();config=json.loads(pathlib.Path(a.config).read_text())
instance=json.loads(pathlib.Path(a.instances).read_text())[0]
root=pathlib.Path(a.output);directory=root/instance["instance_id"]
if directory.exists():
 attempts=root/"attempts";attempts.mkdir(exist_ok=True)
 directory.rename(attempts/(instance["instance_id"]+"-previous"))
directory.mkdir();started=time.monotonic()
(directory/"started.json").write_text(json.dumps({"pid":os.getpid(),"started":started}))
time.sleep(instance["delay"])
failed=instance.get("fail") and not config.get("recover")
result={"instance":instance,"mode":a.mode,
        "reason":"infrastructure_error" if failed else "policy_refusal" if instance.get("refuse") else "success",
        "success":None if failed else not instance.get("refuse",False),
        "started":started,"ended":time.monotonic(),"pid":os.getpid()}
if failed:result["error"]={"message":"sensitive-key-management-id","code":"codex_error"}
(directory/"result.json").write_text(json.dumps(result))
sys.exit(1 if failed else 0)
'''


class DynamicQueueTest(unittest.TestCase):
    def prepare(self, root, rows, mode="code"):
        configs = [root / f"gpu-{index}.json" for index in range(2)]
        for path in configs:
            MODULE.write_json(path, {})
        manifest = root / "input.json"
        MODULE.write_json(manifest, rows)
        runner = root / "worker.py"
        runner.write_text(WORKER)
        return configs, manifest, runner, MODULE.prepare(configs, manifest, root / "run", mode=mode,
                                                        node=sys.executable, runner=runner)

    def test_fast_gpu_claims_remaining_scenes_before_slow_gpu_finishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{"instance_id": f"scene-{index}", "setting": "clean", "delay": 0.4 if index == 0 else 0.02}
                    for index in range(5)]
            _, _, _, (plan, instances) = self.prepare(root, rows)
            self.assertEqual(MODULE.run(plan, instances, poll_seconds=0.005), 0)
            self.assertEqual(plan["shards"][0]["instanceIds"], ["scene-0"])
            self.assertEqual(plan["shards"][1]["instanceIds"], ["scene-1", "scene-2", "scene-3", "scene-4"])
            summary = json.loads((root / "run/combined-summary.json").read_text())
            self.assertEqual(summary["groups"][-1]["fullPanelSuccessRate"], 1)
            self.assertEqual([row["shard"] for row in summary["results"]], [0, 1, 1, 1, 1])
            slow = summary["results"][0]["result"]
            self.assertLess(summary["results"][-1]["result"]["started"], slow["ended"])
            self.assertEqual(json.loads((root / "run/plan.json").read_text()), plan)

    def test_infrastructure_failure_pauses_queue_and_explicit_restart_preserves_successes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{"instance_id": f"scene-{index}", "setting": "clean", "delay": 0.1 if index == 1 else 0.02,
                     **({"fail": True} if index == 0 else {})} for index in range(4)]
            configs, manifest, runner, (plan, instances) = self.prepare(root, rows)
            self.assertEqual(MODULE.run(plan, instances, poll_seconds=0.005), 1)
            summary = json.loads((root / "run/combined-summary.json").read_text())
            group = summary["groups"][-1]
            self.assertEqual((group["completed"], group["errors"], group["pending"]), (1, 1, 2))
            self.assertIsNone(group["fullPanelSuccessRate"])
            success = root / "run/shard-01/scene-1/result.json"
            original = success.read_bytes()
            original_time = success.stat().st_mtime_ns
            for path in configs:
                MODULE.write_json(path, {"recover": True})
            configs = configs + configs
            plan, instances = MODULE.prepare(configs, manifest, root / "run", node=sys.executable, runner=runner)
            self.assertEqual(plan["workers"], 4)
            self.assertEqual(MODULE.run(plan, instances, poll_seconds=0.005), 0)
            self.assertEqual(success.read_bytes(), original)
            self.assertEqual(success.stat().st_mtime_ns, original_time)
            self.assertTrue((root / "run/shard-00/attempts/scene-0-previous/result.json").exists())
            assigned = [key for shard in plan["shards"] for key in shard["instanceIds"]]
            self.assertEqual(sorted(assigned), [f"scene-{index}" for index in range(4)])
            self.assertEqual(json.loads((root / "run/combined-summary.json").read_text())["groups"][-1]["completed"], 4)
            with self.assertRaisesRegex(ValueError, "cannot shrink"):
                MODULE.prepare(configs[:2], manifest, root / "run", node=sys.executable, runner=runner)

    def test_provider_refusal_counts_as_failure_continues_queue_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{"instance_id": f"scene-{index}", "setting": "clean", "delay": 0.01,
                     "refuse": index == 0} for index in range(4)]
            configs, manifest, runner, (plan, instances) = self.prepare(root, rows)
            self.assertEqual(MODULE.run(plan, instances, poll_seconds=0.005), 0)
            summary = json.loads((root / "run/combined-summary.json").read_text())
            group = summary["groups"][-1]
            self.assertEqual((group["completed"], group["errors"], group["pending"], group["successes"]), (4, 0, 0, 3))
            self.assertEqual(group["fullPanelSuccessRate"], 0.75)
            refusal = root / "run/shard-00/scene-0/result.json"
            original = refusal.read_bytes()
            original_time = refusal.stat().st_mtime_ns
            plan, instances = MODULE.prepare(configs, manifest, root / "run", node=sys.executable, runner=runner)
            self.assertEqual(MODULE.run(plan, instances, poll_seconds=0.005), 0)
            self.assertEqual(refusal.read_bytes(), original)
            self.assertEqual(refusal.stat().st_mtime_ns, original_time)
            self.assertFalse(list((root / "run").glob("shard-*/attempts")))
            summary = json.loads((root / "run/combined-summary.json").read_text())
            self.assertEqual(sum(state["launches"] for state in summary["workerStates"]), 0)

    def test_skip_errors_preserves_attempts_and_exhausts_unattempted_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{"instance_id": f"scene-{index}", "setting": "clean", "delay": 0.1 if index == 1 else 0.01,
                     "fail": index in (0, 3)} for index in range(6)]
            configs, manifest, runner, (plan, instances) = self.prepare(root, rows)
            self.assertEqual(MODULE.run(plan, instances, poll_seconds=0.005), 1)
            old_paths = [root / "run/shard-00/scene-0/result.json", root / "run/shard-01/scene-1/result.json"]
            originals = [(path.read_bytes(), path.stat().st_mtime_ns) for path in old_paths]
            # A previously dispatched scene was interrupted before writing a result.
            interrupted = root / "run/shard-00/scene-2"
            interrupted.mkdir()
            (interrupted / "trajectory.jsonl").write_text('existing trajectory\n')
            plan["shards"][0]["instanceIds"].append("scene-2")
            plan["shards"][0]["requested"] += 1
            MODULE.write_json(root / "run/plan.json", plan)
            plan, instances = MODULE.prepare(configs, manifest, root / "run", node=sys.executable, runner=runner)
            self.assertEqual(MODULE.run(plan, instances, poll_seconds=0.005, skip_errors=True), 0)
            summary_path = root / "run/combined-summary.json"
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["status"], "completed_with_errors")
            self.assertTrue(summary["queueExhausted"])
            self.assertEqual(summary["queueRemaining"], 0)
            self.assertEqual(summary["exitCodes"], [0, 0])
            self.assertEqual(set(summary["skippedErrorIds"]), {"scene-0", "scene-2", "scene-3"})
            self.assertEqual({row["instance_id"]: row["reason"] for row in summary["skippedErrors"]},
                             {"scene-0": "infrastructure_error", "scene-2": "missing_result", "scene-3": "infrastructure_error"})
            group = summary["groups"][-1]
            self.assertEqual((group["completed"], group["errors"], group["pending"]), (3, 3, 0))
            self.assertIsNone(group["fullPanelSuccessRate"])
            self.assertEqual(sum(state["launches"] for state in summary["workerStates"]), 3)
            self.assertNotIn("sensitive-key-management-id", summary_path.read_text())
            self.assertFalse((interrupted / "result.json").exists())
            self.assertEqual((interrupted / "trajectory.jsonl").read_text(), 'existing trajectory\n')
            self.assertEqual([(path.read_bytes(), path.stat().st_mtime_ns) for path in old_paths], originals)
            plan, instances = MODULE.prepare(configs, manifest, root / "run", node=sys.executable, runner=runner)
            self.assertEqual(MODULE.run(plan, instances, poll_seconds=0.005, skip_errors=True), 0)
            final = json.loads(summary_path.read_text())
            self.assertEqual(sum(state["launches"] for state in final["workerStates"]), 0)
            self.assertEqual(final["status"], "completed_with_errors")
            self.assertFalse(list((root / "run").glob("shard-*/attempts")))

    def test_live_config_expansion_keeps_active_hybrid_workers_and_dispatches_new_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{"instance_id": f"scene-{index}", "setting": "clean", "delay": 0.6 if index < 2 else 0.03}
                    for index in range(12)]
            configs, _, _, (plan, instances) = self.prepare(root, rows, mode="hybrid")
            config_list = root / "configs.json"
            MODULE.write_json(config_list, list(map(str, configs)))
            captured, failures = {}, []

            def expand_after_start():
                try:
                    started = [root / f"run/shard-{index:02d}/scene-{index}/started.json" for index in range(2)]
                    deadline = time.monotonic() + 5
                    while not all(path.exists() for path in started):
                        if time.monotonic() > deadline:
                            raise RuntimeError("Initial workers did not start")
                        time.sleep(0.005)
                    captured["pids"] = [json.loads(path.read_text())["pid"] for path in started]
                    captured["config_mtimes"] = [(root / f"run/shard-{index:02d}/config.json").stat().st_mtime_ns
                                                 for index in range(2)]
                    MODULE.write_json(config_list, list(map(str, configs + configs)))
                except Exception as error:
                    failures.append(error)

            updater = threading.Thread(target=expand_after_start)
            updater.start()
            try:
                self.assertEqual(MODULE.run(plan, instances, poll_seconds=0.005, configs_file=config_list), 0)
            finally:
                updater.join()
            self.assertEqual(failures, [])
            self.assertEqual(plan["workers"], 4)
            summary = json.loads((root / "run/combined-summary.json").read_text())
            results = [row["result"] for row in summary["results"]]
            self.assertEqual([row["pid"] for row in results[:2]], captured["pids"])
            self.assertEqual([(root / f"run/shard-{index:02d}/config.json").stat().st_mtime_ns
                              for index in range(2)], captured["config_mtimes"])
            for index in (2, 3):
                self.assertTrue(plan["shards"][index]["instanceIds"])
                first_id = plan["shards"][index]["instanceIds"][0]
                first = next(row for row in results if row["instance"]["instance_id"] == first_id)
                self.assertLess(first["started"], min(row["ended"] for row in results[:2]))
            assigned = [key for shard in plan["shards"] for key in shard["instanceIds"]]
            self.assertEqual(sorted(assigned), sorted(row["instance_id"] for row in rows))
            self.assertEqual(sum(state["launches"] for state in summary["workerStates"]), len(rows))
            self.assertFalse(list((root / "run").glob("shard-*/attempts")))
            self.assertTrue(all(row["mode"] == "hybrid" for row in results))
            self.assertEqual(summary["groups"][-1]["fullPanelSuccessRate"], 1)
            with self.assertRaisesRegex(ValueError, "different scene or mode"):
                MODULE.completed(plan["shards"][0], rows[0], "code")


if __name__ == "__main__":
    unittest.main()
