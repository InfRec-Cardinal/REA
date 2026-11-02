"""Protocol integration tests; skip when the optional LingBot packages are absent."""

import base64
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import threading
import unittest
from contextlib import contextmanager

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "lingbot_vla_client.py"
spec = importlib.util.spec_from_file_location("lingbot_vla_client", SCRIPT)
client_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client_module)

if client_module.DEPENDENCY_ERROR is None:
    import msgpack
    import numpy as np
    from PIL import Image
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve


def fixture_array(value):
    """Build upstream wire records independently of the client serializer."""
    return {
        b"__ndarray__": True,
        b"dtype": value.dtype.str,
        b"shape": list(value.shape),
        b"data": value.tobytes(),
    }


def fixture_decode(value):
    if b"__ndarray__" in value:
        return np.ndarray(value[b"shape"], dtype=value[b"dtype"], buffer=value[b"data"])
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


@unittest.skipIf(client_module.DEPENDENCY_ERROR is not None, "Install requirements-lingbot-vla.txt")
class LingBotClientTest(unittest.TestCase):
    def test_json_worker_handshake_reset_and_persistent_inference(self):
        metadata = {"model": "lingbot-vla-v2", "version": {b"__npgeneric__": True, b"dtype": "<i8", b"data": 2}}
        action = np.arange(28, dtype=np.float32).reshape(2, 14)
        pixels = np.array([[[10, 20, 30, 255], [40, 50, 60, 0]]], dtype=np.uint8)
        encoded_image = io.BytesIO()
        Image.fromarray(pixels).save(encoded_image, format="PNG")

        def handler(socket):
            socket.send(msgpack.packb(metadata))
            self.assertEqual(msgpack.unpackb(socket.recv()), {"reset": True, "robo_name": "robotwin"})
            socket.send(msgpack.packb({"action": None, "server_timing": {"infer_ms": 0.1}}))
            for output in (action, action[0]):
                observation = msgpack.unpackb(socket.recv(), object_hook=fixture_decode)
                self.assertEqual(set(observation), {"observation.images.head_camera", "observation.state", "task"})
                image = observation["observation.images.head_camera"]
                self.assertEqual(image.dtype, np.dtype("uint8"))
                np.testing.assert_array_equal(image, pixels[:, :, :3])
                state = observation["observation.state"]
                self.assertEqual(state.dtype, np.dtype("float32"))
                self.assertEqual(state.shape, (14,))
                np.testing.assert_array_equal(state, np.arange(14, dtype=np.float32))
                self.assertEqual(observation["task"], "pick up the red cup")
                socket.send(msgpack.packb({
                    "action": fixture_array(output),
                    "action.left_gripper": fixture_array(np.array([0.5], dtype=np.float32)),
                    "server_timing": {"infer_ms": 12.5},
                }))

        with loopback(handler) as url:
            with subprocess.Popen(
                [sys.executable, str(SCRIPT)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            ) as process:
                try:
                    self.assertEqual(json.loads(process.stdout.readline()), {"event": "ready"})

                    def request(request_id, **payload):
                        process.stdin.write(json.dumps({"id": request_id, **payload}) + "\n")
                        process.stdin.flush()
                        response = json.loads(process.stdout.readline())
                        self.assertEqual(response["id"], request_id)
                        self.assertTrue(response["ok"], response)
                        return response["result"]

                    self.assertEqual(request("1", op="configure", url=url, robot="robotwin", timeoutMs=2000),
                                     {"model": "lingbot-vla-v2", "version": 2})
                    self.assertEqual(request("2", op="reset")["action"], None)
                    for index, output in enumerate((action, action[0]), start=3):
                        result = request(str(index), op="infer", task="pick up the red cup",
                                         images={"observation.images.head_camera": {
                                             "data": base64.b64encode(encoded_image.getvalue()).decode(),
                                             "mimeType": "image/png",
                                         }}, states={"observation.state": list(range(14))})
                        self.assertEqual(result["action"], output.tolist())
                        self.assertEqual(result["action.left_gripper"], [0.5])
                        self.assertEqual(result["server_timing"], {"infer_ms": 12.5})
                    self.assertEqual(request("5", op="close"), {"closed": True})
                    self.assertEqual(process.wait(timeout=3), 0, process.stderr.read())
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=3)

    def test_keepalive_is_disabled_during_blocking_gpu_inference(self):
        def handler(socket):
            socket.send(msgpack.packb({}))
            try:
                socket.recv()
            except ConnectionClosed:
                pass

        client = client_module.LingBotClient()
        with loopback(handler) as url:
            try:
                client.dispatch({"op": "configure", "url": url})
                self.assertIsNone(client.socket.ping_interval)
                self.assertIsNone(client.socket.ping_timeout)
            finally:
                client.close()

    def test_server_text_error_is_reported_and_disconnects(self):
        def handler(socket):
            socket.send(msgpack.packb({}))
            socket.recv()
            socket.send("RuntimeError: model inference failed")

        client = client_module.LingBotClient()
        with loopback(handler) as url:
            client.dispatch({"op": "configure", "url": url})
            with self.assertRaisesRegex(RuntimeError, "model inference failed"):
                client.dispatch({"op": "infer", "images": {}, "states": {}, "task": "test"})
            self.assertIsNone(client.socket)

    def test_metadata_text_error_is_reported(self):
        def handler(socket):
            socket.send("Model is unavailable")

        client = client_module.LingBotClient()
        with loopback(handler) as url:
            with self.assertRaisesRegex(RuntimeError, "Model is unavailable"):
                client.dispatch({"op": "configure", "url": url})
            self.assertIsNone(client.socket)

    def test_timeout_closes_connection_before_another_request(self):
        def handler(socket):
            socket.send(msgpack.packb({}))
            socket.recv()
            try:
                socket.recv()
            except ConnectionClosed:
                pass

        client = client_module.LingBotClient()
        with loopback(handler) as url:
            client.dispatch({"op": "configure", "url": url, "timeoutMs": 100})
            with self.assertRaises(TimeoutError):
                client.dispatch({"op": "infer", "images": {}, "states": {}, "task": "test"})
            with self.assertRaisesRegex(RuntimeError, "Configure"):
                client.dispatch({"op": "reset"})


if __name__ == "__main__":
    unittest.main()
