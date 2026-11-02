import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hybrid_export_labels", ROOT / "eval/robotwin/hybrid/export_labels.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
import train_success_router_code_head as training


class ExportLabelsTest(unittest.TestCase):
    def prepare(self, root, reasons):
        instances = []
        for index, (reason, success) in enumerate(reasons):
            instance = {"instance_id": f"episode-{index}", "task": "adjust_bottle", "setting": "clean", "seed": index}
            instances.append(instance)
            directory = root / instance["instance_id"]
            directory.mkdir()
            (directory / "initial").mkdir()
            for name in ("head", "left", "right"):
                (directory / "initial" / f"{name}.png").write_bytes(b"image")
            (directory / "initial-router-input.json").write_text(json.dumps({
                "task_text": "Lift the bottle upright.",
                "images": [f"initial/{name}.png" for name in ("head", "left", "right")],
                "state_text": '{"endpose": {}, "joint": []}',
            }))
            if reason is not None:
                (directory / "result.json").write_text(json.dumps({
                    "instance": instance, "mode": "code", "reason": reason, "success": success,
                    "codeModel": "test-code-model", "protocol": "rea-bce-vla-code-v1",
                }))
        (root / "instances.json").write_text(json.dumps(instances))
        return instances

    def shard(self, root, instances, workers=8):
        shards = []
        for index in range(workers):
            directory = root / f"shard-{index:02d}"
            directory.mkdir()
            selected = instances[index::workers]
            (directory / "instances.json").write_text(json.dumps(selected))
            for instance in selected:
                (root / instance["instance_id"]).rename(directory / instance["instance_id"])
            shards.append({"index": index, "instanceIds": [row["instance_id"] for row in selected],
                           "output": f"/original/remote/path/shard-{index:02d}"})
        # Artifact copies must resolve relative to the current root, and assignment
        # order must not change the exported scene order.
        (root / "plan.json").write_text(json.dumps({"workers": workers, "shards": shards[::-1]}))

    def test_completed_outcomes_export_labels_and_original_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, [("success", True), ("max_steps", False),
                                ("environment_limit", False), ("terminated", False),
                                ("policy_refusal", False)])
            output = root / "labels.jsonl"
            report = exporter.export_labels(root, output)
            labels = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report["status"], "completed")
            self.assertEqual([label["code_success"] for label in labels], [True, False, False, False, False])
            self.assertEqual(labels[0]["input"]["output_contract"], exporter.OUTPUT_CONTRACT)
            self.assertEqual(labels[0]["policy"], {"model": "test-code-model", "protocol": "rea-bce-vla-code-v1"})
            self.assertTrue(all(Path(path).is_absolute() for label in labels for path in label["input"]["images"]))
            self.assertEqual(labels[0]["input"]["task_text"], "Lift the bottle upright.")
            vla = root / "vla"
            vla.mkdir()
            for label in labels:
                instance = label["instance"]
                episode = vla / instance["instance_id"]
                episode.mkdir()
                (episode / "result.json").write_text(json.dumps({
                    "instance": instance, "mode": "vla", "reason": "success", "success": True,
                }))
                (episode / "decisions.jsonl").write_text(
                    json.dumps({"candidate": "medex", "step": 0}) + "\n"
                    + json.dumps({"candidate": "patchwam", "step": 1}) + "\n"
                )
            paired_output = root / "paired.jsonl"
            paired_report = exporter.export_labels(root, paired_output, vla)
            paired = [json.loads(line) for line in paired_output.read_text().splitlines()]
            self.assertEqual(paired_report["status"], "completed")
            self.assertTrue(all(label["vla_success"] for label in paired))
            self.assertTrue(all(label["vla_selected_model"] == "medex" for label in paired))

    def test_infrastructure_or_missing_episode_prevents_partial_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, [("success", True), ("infrastructure_error", None), (None, None)])
            output = root / "labels.jsonl"
            report = exporter.export_labels(root, output)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["expected"], 3)
            self.assertEqual(report["completed_labels"], 1)
            self.assertEqual([item["instance_id"] for item in report["missing"]], ["episode-1", "episode-2"])
            self.assertFalse(output.exists())

    def test_explicit_partial_export_preserves_selection_excludes_unknown_and_reuses_identical_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instances = self.prepare(root, [("success", True), ("error", None), (None, None), ("policy_refusal", False)])
            selected = [instances[3], instances[1], instances[0]]
            selection = root / "dev.json"
            selection.write_text(json.dumps(selected))
            output = root / "labels.jsonl"
            report = exporter.export_labels(root, output, instances_path=selection, allow_partial=True)
            self.assertEqual(report["status"], "completed_with_errors")
            self.assertEqual(report["expected"], 3)
            self.assertEqual(report["completed_labels"], 2)
            self.assertEqual([row["instance_id"] for row in report["missing"]], ["episode-1"])
            labels = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["instance"] for row in labels], [instances[3], instances[0]])
            self.assertEqual([row["code_success"] for row in labels], [False, True])
            mtime = output.stat().st_mtime_ns
            again = exporter.export_labels(root, output, instances_path=selection, allow_partial=True)
            self.assertTrue(again["reused"])
            self.assertEqual(output.stat().st_mtime_ns, mtime)
            vla = root / "vla"
            vla.mkdir()
            episode = vla / instances[0]["instance_id"]
            episode.mkdir()
            (episode / "result.json").write_text(json.dumps({"instance": instances[0], "mode": "vla", "reason": "success", "success": True}))
            (episode / "decisions.jsonl").write_text(json.dumps({"candidate": "medex"}) + "\n")
            paired = root / "paired.jsonl"
            report = exporter.export_labels(root, paired, vla, selection, allow_partial=True)
            self.assertEqual(report["completed_labels"], 1)
            self.assertEqual([row["instance_id"] for row in report["missing"]], ["episode-3", "episode-1"])
            self.assertEqual(json.loads(paired.read_text())["instance"], instances[0])

    def test_partial_export_does_not_create_empty_training_or_development_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, [("error", None)])
            output = root / "labels.jsonl"
            report = exporter.export_labels(root, output, allow_partial=True)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["completed_labels"], 0)
            self.assertFalse(output.exists())

    def test_mixed_provider_metadata_survives_export_without_entering_model_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instances = self.prepare(root, [("success", True), ("max_steps", False)])
            policies = []
            for instance, model, provider in zip(instances,
                    ("gpt-6-astra", "openai/gpt-6-astra"), ("provider-a", "provider-b")):
                path = root / instance["instance_id"] / "result.json"
                result = json.loads(path.read_text())
                result.update(codeModel=model, protocol="rea-bce-vla-code-v2",
                              codeProvider={"name": provider, "transport": "codex_cli_responses"},
                              codeProtocol={"version": 2, "backend": "codex", "decisionLimit": 40,
                                            "maxOutputTokens": None, "reasoning": "cli_default"})
                path.write_text(json.dumps(result))
                policies.append({"model": model, "protocol": result["protocol"],
                                 "provider": result["codeProvider"], "code_protocol": result["codeProtocol"]})
            output = root / "labels.jsonl"
            self.assertEqual(exporter.export_labels(root, output)["status"], "completed")
            labels = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["policy"] for row in labels], policies)
            for row in labels:
                normalized = training.normalize_row(row, root)
                self.assertEqual(normalized["policy"], row["policy"])
                self.assertEqual(training.model_record(normalized),
                                 dict(row["input"], id=f"adjust_bottle:clean:{row['instance']['seed']}"))

    def test_eight_shards_export_selected_manifest_order_with_local_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instances = self.prepare(root, [("success", True), ("max_steps", False)] * 8)
            self.shard(root, instances)
            selected = [instances[index] for index in (14, 1, 9, 0)]
            selection = root / "dev.json"
            selection.write_text(json.dumps(selected))
            # An unfinished episode outside the requested split does not block it.
            (root / "shard-02/episode-2/result.json").unlink()
            output = root / "dev.jsonl"
            report = exporter.export_labels(root, output, instances_path=selection)
            labels = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report["expected"], 4)
            self.assertEqual(report["status"], "completed")
            self.assertEqual([row["instance"] for row in labels], selected)
            self.assertEqual([row["code_success"] for row in labels], [True, False, False, True])
            self.assertTrue(all(Path(path).is_file() for row in labels for path in row["input"]["images"]))
            self.assertIn("shard-06/episode-14", labels[0]["input"]["images"][0])

    def test_missing_selected_shard_episode_prevents_partial_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instances = self.prepare(root, [("success", True)] * 16)
            self.shard(root, instances)
            (root / "shard-03/episode-11/result.json").unlink()
            output = root / "labels.jsonl"
            report = exporter.export_labels(root, output)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["completed_labels"], 15)
            self.assertEqual(report["missing"], [{"instance_id": "episode-11", "reason": "missing_files",
                                                  "files": ["result.json"]}])
            self.assertFalse(output.exists())

    def test_code_and_vla_use_their_own_shard_assignments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instances = self.prepare(root, [("success", True)] * 16)
            self.shard(root, instances)
            selected = [instances[index] for index in (15, 2, 8)]
            selection = root / "dev.json"
            selection.write_text(json.dumps(selected))
            vla = root / "vla"
            vla.mkdir()
            for index, instance in enumerate(selected):
                episode = vla / instance["instance_id"]
                episode.mkdir()
                (episode / "result.json").write_text(json.dumps({
                    "instance": instance, "mode": "vla", "reason": "success", "success": index != 1,
                }))
                (episode / "decisions.jsonl").write_text(json.dumps({"candidate": f"model-{index}"}) + "\n")
            self.shard(vla, selected, workers=2)
            output = root / "paired.jsonl"
            report = exporter.export_labels(root, output, vla, selection)
            labels = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report["status"], "completed")
            self.assertEqual([row["vla_selected_model"] for row in labels], ["model-0", "model-1", "model-2"])
            self.assertEqual([row["vla_success"] for row in labels], [True, False, True])
            (vla / "shard-00/episode-8/decisions.jsonl").unlink()
            incomplete = root / "incomplete.jsonl"
            report = exporter.export_labels(root, incomplete, vla, selection)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["missing"][0]["instance_id"], "episode-8")
            self.assertEqual(report["missing"][0]["reason"], "missing_vla_files")
            self.assertFalse(incomplete.exists())

    def test_selected_scene_must_match_the_collection_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instances = self.prepare(root, [("success", True)])
            selection = root / "dev.json"
            selection.write_text(json.dumps([dict(instances[0], seed=999)]))
            output = root / "labels.jsonl"
            with self.assertRaisesRegex(ValueError, "does not match the run manifest"):
                exporter.export_labels(root, output, instances_path=selection)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
