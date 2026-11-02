"""Initial scoring preserves router inputs and never consumes success labels."""

import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORING = load("score_initials", ROOT / "eval/robotwin/hybrid/score_initials.py")
ROUTER = load("success_router_server", ROOT / "scripts/success_router_server.py")


class ScoreInitialsTest(unittest.TestCase):
    def test_flat_and_sharded_inputs_use_original_text_state_and_cameras(self):
        received = []

        def predictor(row):
            received.append(row)
            return {"medex": 0.8, "pi05": 0.4}, {"images": 3}

        service = ROUTER.RouterService(predictor, ["medex", "pi05"], {"test": True})
        with ROUTER.make_server(service, "127.0.0.1", 0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for sharded in (False, True):
                    with self.subTest(sharded=sharded), tempfile.TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        directory = root / "shard-00" / "episode-1" if sharded else root / "episode-1"
                        directory.mkdir(parents=True)
                        instance = {"instance_id": "episode-1", "task": "move_can_pot", "setting": "clean", "seed": 3}
                        state_text = '{"endpose": [1, 2], "joint": [3, 4]}'
                        for camera in SCORING.CAMERAS:
                            (directory / f"{camera}.png").write_bytes(camera.encode())
                        initial = {"instance_id": "episode-1", "task_text": "The original instruction.",
                                   "state_text": state_text, "images": [f"{camera}.png" for camera in SCORING.CAMERAS]}
                        (directory / "initial-router-input.json").write_text(json.dumps(initial))
                        (directory / "result.json").write_text("not JSON: an outcome must never be read")
                        if sharded:
                            (root / "plan.json").write_text(json.dumps({"shards": [{"output": str(directory.parent), "instanceIds": ["episode-1"]}]}))
                        report = SCORING.score_initials(root, [instance], f"http://127.0.0.1:{server.server_port}/predict")
                        self.assertEqual(report["rows"][0]["instance"], instance)
                        self.assertEqual(report["rows"][0]["selected_candidate"], "medex")
                        self.assertEqual(report["selectedCounts"], {"medex": 1})
                        self.assertEqual(received[-1]["task_text"], initial["task_text"])
                        self.assertEqual(received[-1]["state_text"], state_text)
                        self.assertEqual([image.read() for image in received[-1]["images"]], [camera.encode() for camera in SCORING.CAMERAS])
            finally:
                server.shutdown()
                thread.join()

    def test_missing_initials_are_reported_without_requesting_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = SCORING.score_initials(temporary, [{"instance_id": "pending"}], "http://unused.invalid")
            self.assertEqual(report["requested"], 1)
            self.assertEqual(report["scored"], 0)
            self.assertEqual(report["missing"][0]["instance_id"], "pending")


if __name__ == "__main__":
    unittest.main()
