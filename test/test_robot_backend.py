import base64
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


backend_module = load_module("robot_backend", SCRIPTS / "robot_backend.py")
common_module = load_module("robot_driver_common", SCRIPTS / "robot_drivers" / "common.py")


class Array:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FixtureDriver:
    id = "fixture:arm"
    timing = "realtime"

    def __init__(self):
        self.calls = []
        self.position = 0.0
        self.actions = []
        self.metadata = {"cameraNames": ["wrist"]}

    def snapshot(self):
        return {
            "timestampMs": 1700000000000,
            "state": {"position": self.position},
            "images": [{"name": "wrist", "mimeType": "image/png", "data": "fixture"}],
            "success": None,
            "reward": None,
            "terminated": False,
            "truncated": False,
        }

    def reset(self, **context):
        self.calls.append(("reset", context))
        self.actions = [{
            "name": "joint_position",
            "description": "Command the fixture joint in radians.",
            "parameters": {
                "type": "object",
                "properties": {"position": {"type": "number", "minimum": -1, "maximum": 1}},
                "required": ["position"],
                "additionalProperties": False,
            },
        }]
        self.metadata = {**self.metadata, "jointNames": ["joint"], "jointUnits": "radians"}
        return self.snapshot()

    def observe(self):
        self.calls.append(("observe",))
        return self.snapshot()

    def step(self, action, execute):
        self.calls.append(("step", action, execute))
        if execute:
            self.position = action["arguments"]["position"]
        return {**self.snapshot(), "info": {"submitted": execute}}

    def stop(self):
        self.calls.append(("stop",))
        return {"supported": True, "completion": "hold_requested"}

    def close(self):
        self.calls.append(("close",))


class RobotBackendTests(unittest.TestCase):
    def setUp(self):
        self.driver = FixtureDriver()
        self.factory_calls = []

        def factory(kind, options):
            self.factory_calls.append((kind, options))
            return self.driver

        self.backend = backend_module.RobotBackend(driver_factory=factory)
        self.addCleanup(self.backend.close)
        self.action = {"name": "joint_position", "arguments": {"position": 0.5}}

    def configure(self, execute_enabled=False):
        return self.backend.dispatch({"op": "configure", "driver": "fixture", "options": {"executeEnabled": execute_enabled}})

    def test_configure_and_reset_publish_current_driver_description(self):
        description = self.configure()
        self.assertEqual(self.factory_calls, [("fixture", {"executeEnabled": False})])
        self.assertEqual(description, {
            "id": "fixture:arm", "timing": "realtime", "actions": [], "metadata": {"cameraNames": ["wrist"]},
        })
        task = {"instruction": "Move the joint", "target": 0.5}
        reset = self.backend.dispatch({"op": "reset", "task": task, "seed": 42})
        self.assertEqual(self.driver.calls, [("reset", {"task": task, "seed": 42})])
        self.assertEqual(reset["metadata"]["jointNames"], ["joint"])
        self.assertEqual(reset["actions"][0]["parameters"]["properties"]["position"]["maximum"], 1)
        observed = self.backend.dispatch({"op": "observe"})
        self.assertEqual(observed["actions"], reset["actions"])
        self.assertEqual(observed["images"][0]["name"], "wrist")
        self.assertEqual(observed["timestampMs"], 1700000000000)
        with self.assertRaisesRegex(ValueError, "already configured"):
            self.configure()
        self.assertEqual(len(self.factory_calls), 1)

    def test_preview_passes_explicit_false_and_never_runs_stop(self):
        self.configure()
        self.backend.dispatch({"op": "reset"})
        result = self.backend.dispatch({"op": "step", "action": self.action})
        self.assertEqual(self.driver.calls[-1], ("step", self.action, False))
        self.assertEqual(result["state"]["position"], 0.0)
        self.assertFalse(result["info"]["submitted"])
        self.assertIsNone(result["success"])
        self.assertIsNone(result["reward"])
        before = len(self.driver.calls)
        with self.assertRaisesRegex(ValueError, "configured for preview"):
            self.backend.dispatch({"op": "step", "action": self.action, "execute": True})
        self.assertEqual(self.backend.dispatch({"op": "stop"}), {"supported": False, "preview": True})
        self.assertEqual(len(self.driver.calls), before)

    def test_enabled_driver_still_requires_execute_per_step_and_preserves_outcomes(self):
        self.configure(execute_enabled=True)
        self.backend.dispatch({"op": "reset"})
        preview = self.backend.dispatch({"op": "step", "action": self.action})
        self.assertEqual(preview["state"]["position"], 0.0)
        result = self.backend.dispatch({"op": "step", "action": self.action, "execute": True})
        self.assertEqual(result["state"]["position"], 0.5)
        self.assertTrue(result["info"]["submitted"])
        self.assertIsNone(result["success"])
        self.assertFalse(result["terminated"])
        self.assertEqual(result["actions"], self.driver.actions)
        self.assertEqual(self.backend.dispatch({"op": "stop"}), {"supported": True, "completion": "hold_requested"})
        self.assertEqual(self.driver.calls[-1], ("stop",))

    def test_explicit_close_releases_the_driver_once(self):
        self.configure(execute_enabled=True)
        self.assertEqual(self.backend.dispatch({"op": "close"}), {"closed": True})
        self.backend.close()
        self.assertEqual(self.driver.calls, [("close",)])
        self.assertIsNone(self.backend.driver)

    def test_close_after_failed_sdk_configuration_preserves_cleanup(self):
        def unavailable_sdk(kind, options):
            raise RuntimeError("simulation assets unavailable")

        backend = backend_module.RobotBackend(driver_factory=unavailable_sdk)
        with self.assertRaisesRegex(RuntimeError, "simulation assets unavailable"):
            backend.dispatch({"op": "configure", "driver": "openral"})
        self.assertEqual(backend.dispatch({"op": "close"}), {"closed": True})

    def test_named_rgb_and_rgba_images_have_decodable_pixels(self):
        images = common_module.image_observation({
            "top": Array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]),
            "wrist": Array([[[10, 20, 30, 40]]]),
            "absent": None,
        })
        self.assertEqual([image["name"] for image in images], ["top", "wrist"])
        expected = [
            ((2, 2, 8, 2, 0, 0, 0), bytes([0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 255, 255, 255, 255])),
            ((1, 1, 8, 6, 0, 0, 0), bytes([0, 10, 20, 30, 40])),
        ]
        for image, (header, pixels) in zip(images, expected):
            self.assertEqual(image["mimeType"], "image/png")
            encoded = base64.b64decode(image["data"])
            self.assertEqual(encoded[:8], b"\x89PNG\r\n\x1a\n")
            chunks = {}
            position = 8
            while position < len(encoded):
                length = struct.unpack("!I", encoded[position:position + 4])[0]
                kind = encoded[position + 4:position + 8]
                chunks[kind] = encoded[position + 8:position + 8 + length]
                position += length + 12
            self.assertEqual(struct.unpack("!IIBBBBB", chunks[b"IHDR"]), header)
            self.assertEqual(zlib.decompress(chunks[b"IDAT"]), pixels)
            self.assertIn(b"IEND", chunks)
        values = common_module.json_value({"positions": Array([0.1, 0.2]), "success": Array(True), "nested": (Array(3),)})
        self.assertEqual(json.loads(json.dumps(values, allow_nan=False)), {"positions": [0.1, 0.2], "success": True, "nested": [3]})


PROTOCOL_FIXTURE = """
import importlib.util
from pathlib import Path
import sys

specification = importlib.util.spec_from_file_location("robot_backend_fixture", sys.argv[1])
module = importlib.util.module_from_spec(specification)
specification.loader.exec_module(module)
backend_type = module.RobotBackend

class Driver:
    id = "fixture"
    timing = "realtime"
    actions = []
    metadata = {}

    def __init__(self, kind, options):
        print("SDK connection message", flush=True)

    def close(self):
        with Path(sys.argv[2]).open("a") as marker:
            marker.write("closed\\n")
        print("SDK close message", flush=True)

module.RobotBackend = lambda: backend_type(driver_factory=Driver)
module.main()
"""


class RobotBackendProtocolTests(unittest.TestCase):
    def test_explicit_close_and_eof_release_resources_and_keep_sdk_logs_off_protocol(self):
        for explicit_close in (True, False):
            with self.subTest(explicit_close=explicit_close), tempfile.TemporaryDirectory() as directory:
                marker = Path(directory) / "closed.txt"
                requests = [{"id": "configure", "op": "configure", "driver": "fixture"}]
                if explicit_close:
                    requests.append({"id": "close", "op": "close"})
                process = subprocess.run(
                    [sys.executable, "-c", PROTOCOL_FIXTURE, str(SCRIPTS / "robot_backend.py"), str(marker)],
                    input="".join(json.dumps(request) + "\n" for request in requests),
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                responses = [json.loads(line) for line in process.stdout.splitlines()]
                self.assertEqual(responses[0], {"event": "ready"})
                self.assertTrue(responses[1]["ok"])
                self.assertEqual(responses[1]["result"]["id"], "fixture")
                self.assertEqual(len(responses), len(requests) + 1)
                if explicit_close:
                    self.assertEqual(responses[-1]["result"], {"closed": True})
                self.assertEqual(marker.read_text(), "closed\n")
                self.assertIn("SDK connection message", process.stderr)
                self.assertIn("SDK close message", process.stderr)

    def test_sigterm_closes_a_configured_driver(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "closed.txt"
            process = subprocess.Popen(
                [sys.executable, "-c", PROTOCOL_FIXTURE, str(SCRIPTS / "robot_backend.py"), str(marker)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                self.assertEqual(json.loads(process.stdout.readline()), {"event": "ready"})
                process.stdin.write(json.dumps({"id": "configure", "op": "configure", "driver": "fixture"}) + "\n")
                process.stdin.flush()
                self.assertTrue(json.loads(process.stdout.readline())["ok"])
                process.terminate()
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, stderr)
                self.assertEqual(stdout, "")
                self.assertEqual(marker.read_text(), "closed\n")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


if __name__ == "__main__":
    unittest.main()
