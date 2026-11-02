import base64
import importlib.util
import json
from pathlib import Path
import threading
import tempfile
import unittest
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("success_router_server", ROOT / "scripts/success_router_server.py")
router = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router)


class SourcePathsTest(unittest.TestCase):
    def test_default_sources_are_bundled(self):
        model, encoding = router.source_paths()
        self.assertTrue(model.is_file())
        self.assertTrue(encoding.is_file())
        module = router.load_source("test_success_router_encoding", encoding)
        for name in ("load_processor", "encode_record", "forward_row"):
            self.assertTrue(callable(getattr(module, name)))

    def test_external_sources_support_flat_and_code_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(router.source_paths(root), (root / "model.py", root / "train.py"))
            (root / "code").mkdir()
            (root / "code/model.py").touch()
            self.assertEqual(router.source_paths(root), (root / "code/model.py", root / "code/train.py"))


def observation():
    return {
        "id": "scene:1",
        "text": "Lift the bottle upright.",
        "images": [
            {"name": name, "mimeType": "image/png", "data": base64.b64encode(content).decode()}
            for name, content in (("right_wrist", b"right"), ("head", b"head"), ("left_wrist", b"left"))
        ],
        "state": {
            "endpose": {
                "right_gripper": 1.0, "left_gripper": 1.0,
                "right_endpose": [0.3, 0, 0.9, 1, 0, 0, 0],
                "left_endpose": [-0.3, 0, 0.9, 1, 0, 0, 0],
            },
            "joint": [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0],
            "success": True,
        },
    }


class InputRecordTest(unittest.TestCase):
    def test_training_camera_order_state_format_and_contract(self):
        source = observation()
        row = router.input_record({"observation": source, "instruction": "Choose the next action."})
        self.assertEqual(row["task_text"], source["text"])
        self.assertEqual([image.read() for image in row["images"]], [b"head", b"left", b"right"])
        expected_endpose = {
            name: source["state"]["endpose"][name]
            for name in ("left_endpose", "left_gripper", "right_endpose", "right_gripper")
        }
        self.assertEqual(row["state_text"], json.dumps({"endpose": expected_endpose, "joint": source["state"]["joint"]}))
        self.assertEqual(row["output_contract"], router.OUTPUT_CONTRACT)
        self.assertNotIn("success", row["state_text"])

    def test_native_robot_state_and_preserved_training_text(self):
        source = observation()
        original = source["state"]
        source["state"] = {"robot_state": {"endpose": original["endpose"], "joint_action": {"vector": original["joint"]}}}
        self.assertEqual(json.loads(router.robot_state_text(source))["joint"], original["joint"])
        source["state"]["state_text"] = '{"endpose": "unchanged", "joint": []}'
        self.assertEqual(router.robot_state_text(source), source["state"]["state_text"])

    def test_missing_camera_does_not_produce_incomplete_multimodal_input(self):
        source = observation()
        source["images"] = source["images"][:2]
        with self.assertRaisesRegex(ValueError, "left_camera"):
            router.input_record({"observation": source})

    def test_vla_state_takes_precedence_over_other_robot_state(self):
        source = observation()
        original = source["state"]
        source["state"] = {
            "vla": {"endpose": original["endpose"], "joint_action": {"vector": original["joint"]}},
            "robot_state": {"joint": [99] * 14},
        }
        self.assertEqual(json.loads(router.robot_state_text(source))["joint"], original["joint"])


class RouterServiceTest(unittest.TestCase):
    def setUp(self):
        self.records = []

        def predictor(row):
            self.records.append(row)
            return {"medex": 0.75, "patchwam": 0.9}, {"images": 3, "input_tokens": 825}

        self.service = router.RouterService(predictor, ["medex", "patchwam"], {"training_input_stage": "initial_observation"})

    def test_only_trained_candidates_receive_probabilities(self):
        result = self.service.predict({
            "observation": observation(), "episodeId": "episode:7", "step": 0,
            "candidates": [{"id": "patchwam", "kind": "vla"}, {"id": "code", "kind": "code"}],
        })
        self.assertEqual(result["scores"], {"patchwam": 0.9})
        self.assertEqual(result["trained_candidates"], ["medex", "patchwam"])
        self.assertEqual(result["untrained_candidates"], [{"id": "code", "kind": "code", "reason": "no_trained_head"}])
        self.assertEqual(result["selected_candidate"], "patchwam")
        self.assertEqual(result["episodeId"], "episode:7")
        self.assertEqual(len(self.records), 1)

    def test_untrained_code_alone_does_not_invoke_model(self):
        result = self.service.predict({"candidates": [{"id": "code", "kind": "code"}]})
        self.assertEqual(result["scores"], {})
        self.assertIsNone(result["selected_candidate"])
        self.assertEqual(self.records, [])

    def test_http_prediction_and_health(self):
        with router.make_server(self.service, "127.0.0.1", 0) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(url + "/health") as response:
                    self.assertEqual(json.load(response)["trained_candidates"], ["medex", "patchwam"])
                payload = json.dumps({"observation": observation()}).encode()
                request = Request(url + "/predict", data=payload, headers={"Content-Type": "application/json"})
                with urlopen(request) as response:
                    result = json.load(response)
                self.assertEqual(result["scores"], {"medex": 0.75, "patchwam": 0.9})
                self.assertEqual(result["selected_candidate"], "patchwam")
            finally:
                server.shutdown()
                worker.join()


if __name__ == "__main__":
    unittest.main()
