import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_success_router_code_head as training


def row(seed, success, setting="clean"):
    return {
        "instance": {"task": "adjust_bottle", "setting": setting, "seed": seed},
        "input": {
            "task_text": "Lift the bottle.",
            "images": ["head.png", "left.png", "right.png"],
            "state_text": '{"endpose": {}, "joint": []}',
            "output_contract": training.OUTPUT_CONTRACT,
        },
        "code_success": success,
    }


class DataProtocolTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(item) + "\n" for item in rows))
        return path

    def test_test_scenes_excluded_without_using_outcomes(self):
        train = self.write("train.jsonl", [row(1, True), row(3, False)])
        dev = self.write("dev.jsonl", [row(2, False), row(3, True)])
        frozen = self.write("test.jsonl", [{"task": "adjust_bottle", "setting": "clean", "seed": 3}])
        datasets, summary = training.prepare_data(train, dev, frozen)
        self.assertEqual([item["instance"]["seed"] for item in datasets["train"]], [1])
        self.assertEqual([item["instance"]["seed"] for item in datasets["dev"]], [2])
        self.assertEqual(summary["train"]["excluded_test_overlap"], 1)
        self.assertEqual(summary["dev"]["excluded_test_overlap"], 1)
        self.assertEqual(summary["test_overlap_after_exclusion"], 0)

    def test_train_dev_overlap_rejected(self):
        train = self.write("train.jsonl", [row(1, True)])
        dev = self.write("dev.jsonl", [row(1, False)])
        frozen = self.write("test.jsonl", [row(9, True)])
        with self.assertRaisesRegex(ValueError, "Train and dev share"):
            training.prepare_data(train, dev, frozen)

    def test_labels_require_actual_boolean_outcomes(self):
        for success in (None, 0.5, "success", 1):
            with self.subTest(success=success), self.assertRaisesRegex(ValueError, "boolean code_success"):
                training.normalize_row(row(1, success), self.root)

    def test_scene_identity_includes_setting(self):
        self.assertNotEqual(training.scene_key(row(1, True)), training.scene_key(row(1, True, "randomized")))

    def test_dry_run_needs_no_model_or_gpu(self):
        train = self.write("train.jsonl", [row(1, True)])
        dev = self.write("dev.jsonl", [row(2, False)])
        frozen = self.write("test.jsonl", [row(9, True)])
        checkpoint = self.root / "source"
        checkpoint.mkdir()
        (checkpoint / "router_config.json").write_text(json.dumps({"model_ids": ["medex"]}))
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/train_success_router_code_head.py"),
            "--checkpoint", str(checkpoint),
            "--train-jsonl", str(train), "--dev-jsonl", str(dev), "--test-instances", str(frozen),
            "--output", str(self.root / "output"), "--dry-run",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["train"]["rows"], 1)
        self.assertFalse((self.root / "output").exists())


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for head fitting")
class CodeHeadFitTest(unittest.TestCase):
    def test_fit_and_checkpoint_preserve_seven_original_outputs(self):
        import torch

        features = torch.tensor([[-2.0, 0.0], [-1.0, 0.5], [1.0, 0.5], [2.0, 0.0]])
        data = {"features": features, "labels": torch.tensor([0.0, 0.0, 1.0, 1.0])}
        code_head, result = training.train_head(data, data, 40, 4, 0.05, 42)
        self.assertLess(result["dev_code_bce"], 0.2)
        original = {"weight": torch.arange(14, dtype=torch.float32).reshape(7, 2),
                    "bias": torch.arange(7, dtype=torch.float32)}
        merged = training.combined_head(original, code_head)
        torch.testing.assert_close(merged["weight"][:7], original["weight"], rtol=0, atol=0)
        torch.testing.assert_close(merged["bias"][:7], original["bias"], rtol=0, atol=0)
        before = torch.nn.functional.linear(features, original["weight"], original["bias"])
        after = torch.nn.functional.linear(features, merged["weight"], merged["bias"])
        torch.testing.assert_close(after[:, :7], before, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            for name in ("adapter", "processor"):
                (source / name).mkdir()
                (source / name / "source.txt").write_text(name)
            model_ids = [f"vla_{index}" for index in range(7)]
            (source / "router_config.json").write_text(json.dumps({
                "schema_version": "robotwin-router-success-v1", "hidden_size": 2,
                "model_ids": model_ids, "metadata": {"model_path": "/original", "epoch": 2},
            }))
            (source / "training_plan.json").write_text(json.dumps({
                "model_ids": model_ids, "arguments": {"lora_r": 8, "seq_length": 8192},
            }))
            torch.save(original, source / "head.pt")
            output = Path(directory) / "trained"
            config = training.save_checkpoint(source, output, code_head, "code", "/deployed",
                                               {"train": {"rows": 4}}, result, {"epochs": 40})
            self.assertEqual(config["model_ids"], [*model_ids, "code"])
            self.assertTrue(config["metadata"]["code_head"]["trained"])
            self.assertEqual(config["metadata"]["source_training"]["epoch"], 2)
            restored = torch.nn.Linear(2, 8)
            restored.load_state_dict(torch.load(output / "head.pt", weights_only=True))
            torch.testing.assert_close(restored(features)[:, :7], before, rtol=0, atol=0)
            self.assertEqual((output / "adapter/source.txt").read_text(), "adapter")
            self.assertEqual((output / "processor/source.txt").read_text(), "processor")

    def test_checkpoint_selection_uses_dev_bce(self):
        import torch

        features = torch.tensor([[-1.0], [1.0]])
        train = {"features": features, "labels": torch.tensor([0.0, 1.0])}
        dev = {"features": features, "labels": torch.tensor([1.0, 0.0])}
        _, result = training.train_head(train, dev, 5, 2, 0.1, 42)
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["dev_code_bce"], min(epoch["dev_code_bce"] for epoch in result["curve"]))

    def test_paired_dev_prefers_routing_success_over_lower_code_bce(self):
        import torch

        dev = {
            "labels": torch.tensor([1.0, 0.0]),
            "vla_logits": torch.logit(torch.tensor([[0.95, 0.2], [0.95, 0.3]])),
        }
        records = [dict(row(seed, success), vla_selected_model="medex", vla_success=vla_success)
                   for seed, success, vla_success in ((1, True, False), (2, False, True))]
        training.attach_paired_dev(dev, records, ["medex", "patchwam"])
        better_routing = training.dev_metrics(torch.logit(torch.tensor([0.96, 0.9])), dev)
        better_bce = training.dev_metrics(torch.logit(torch.tensor([0.9, 0.1])), dev)
        self.assertGreater(better_routing["dev_code_bce"], better_bce["dev_code_bce"])
        self.assertEqual(better_routing["dev_routing_success"], 1.0)
        self.assertEqual(better_bce["dev_routing_success"], 0.5)
        self.assertGreater(training.selection_key(better_routing), training.selection_key(better_bce))
        records[0]["vla_selected_model"] = "patchwam"
        with self.assertRaisesRegex(ValueError, "differs from checkpoint argmax"):
            training.attach_paired_dev(dev, records, ["medex", "patchwam"])


if __name__ == "__main__":
    unittest.main()
