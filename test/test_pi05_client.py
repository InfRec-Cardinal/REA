"""Exercise the real JSON worker against an openpi-compatible loopback server."""

import base64
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pi05_client.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("pi05_client", SCRIPT)
client_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client_module)

if client_module.DEPENDENCY_ERROR is None:
    import msgpack
    import numpy as np
    from PIL import Image
    from websockets.sync.server import serve


def wire_array(value):
    return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}


def wire_decode(value):
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=value[b"dtype"], shape=value[b"shape"])
    return value


@contextmanager
def loopback(handler):
    errors = []

    def checked_handler(socket):
        try:
            handler(socket)
        except Exception as error:
            errors.append(error)

    with serve(checked_handler, "127.0.0.1", 0, compression=None) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=2)
    if errors:
        raise errors[0]


@unittest.skipIf(client_module.DEPENDENCY_ERROR is not None, "Install requirements-pi05.txt")
class Pi05ClientTest(unittest.TestCase):
    def test_worker_sends_chw_images_state_prompt_and_decodes_action_chunks(self):
        pixels = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        encoded = io.BytesIO()
        Image.fromarray(pixels).save(encoded, format="PNG")
        actions = np.arange(42, dtype=np.float32).reshape(3, 14) / 100

        def handler(socket):
            socket.send(msgpack.packb({"model": "pi05-aloha"}))
            for index in range(2):
                observation = msgpack.unpackb(socket.recv(), object_hook=wire_decode)
                self.assertEqual(set(observation), {"images", "state", "prompt"})
                self.assertEqual(observation["prompt"], "Place the cup.")
                self.assertEqual(observation["state"].dtype, np.dtype("float32"))
                np.testing.assert_array_equal(observation["state"], np.full(14, index, dtype=np.float32))
                self.assertEqual(set(observation["images"]), {"cam_high", "cam_left_wrist", "cam_right_wrist"})
                for image in observation["images"].values():
                    self.assertEqual(image.dtype, np.dtype("uint8"))
                    self.assertEqual(image.shape, (3, 1, 2))
                    np.testing.assert_array_equal(image, pixels.transpose(2, 0, 1))
                socket.send(msgpack.packb({"actions": wire_array(actions + index), "server_timing": {"infer_ms": 12}}))

        with loopback(handler) as url:
            with subprocess.Popen([sys.executable, str(SCRIPT)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
                try:
                    self.assertEqual(json.loads(process.stdout.readline()), {"event": "ready"})

                    def request(operation, **payload):
                        process.stdin.write(json.dumps({"id": operation, "op": operation, **payload}) + "\n")
                        process.stdin.flush()
                        response = json.loads(process.stdout.readline())
                        self.assertTrue(response["ok"], response)
                        return response["result"]

                    self.assertEqual(request("configure", url=url, timeoutMs=2000), {"model": "pi05-aloha"})
                    images = {camera: {"data": base64.b64encode(encoded.getvalue()).decode()}
                              for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
                    for index in range(2):
                        self.assertEqual(request("reset"), {"reset": True})
                        result = request("infer", images=images, state=[index] * 14, prompt="Place the cup.")
                        self.assertEqual(result["actions"], (actions + index).tolist())
                        self.assertEqual(result["server_timing"], {"infer_ms": 12})
                    self.assertEqual(request("close"), {"closed": True})
                    self.assertEqual(process.wait(timeout=3), 0, process.stderr.read())
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
