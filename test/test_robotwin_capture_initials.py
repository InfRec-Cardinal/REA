import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "eval/robotwin/hybrid/capture_initials.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("capture_initials", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CaptureTest(unittest.TestCase):
    def test_capture_only_resets_and_closes_without_policy_or_success_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            response = SimpleNamespace(stdout='{"event":"ready"}\n{"id":1,"ok":true,"result":{"success":true}}\n{"id":2,"ok":true}\n')
            with patch.object(MODULE.subprocess, "run", return_value=response) as run:
                status = MODULE.capture_one({"simulatorPython": "python", "subset": "/subset", "harnessRoot": "/harness"},
                                            {"instance_id": "sample"}, Path(temporary), io.StringIO())
            requests = [json.loads(line) for line in run.call_args.kwargs["input"].splitlines()]
            self.assertEqual([request["op"] for request in requests], ["reset", "close"])
            self.assertEqual(status["nativeActions"], 0)
            self.assertNotIn("success", status)
            self.assertFalse((Path(temporary) / "result.json").exists())

    def test_initial_consistency_reports_changed_record_and_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instances = [{"instance_id": "sample"}]
            for side in ("left", "right"):
                directory = root / side / "sample"
                directory.mkdir(parents=True)
                row = {"instance_id": "sample", "task": "task", "seed": 10, "task_text": "Original instruction",
                       "state_text": '{"joint":[1]}', "images": ["head.png", "left.png", "right.png"]}
                for image in row["images"]:
                    (directory / image).write_bytes(b"same image")
                (directory / "initial-router-input.json").write_text(json.dumps(row))
            matched = MODULE.compare_initials(root / "left", root / "right", instances)
            self.assertEqual(matched["status"], "matched")
            (root / "right/sample/head.png").write_bytes(b"different image")
            changed = MODULE.compare_initials(root / "left", root / "right", instances)
            self.assertEqual(changed["differences"], [{"instance_id": "sample", "fields": ["head_camera"]}])


if __name__ == "__main__":
    unittest.main()
