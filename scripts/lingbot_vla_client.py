"""Persistent JSON-lines bridge to the official LingBot-VLA 2.0 WebSocket server.

Install requirements-lingbot-vla.txt in the Python environment running this file.
Model weights, torch, and the upstream repository are only needed on the server.
"""

import base64
import io
import json
import signal
import sys

try:
    import msgpack
    import numpy as np
    from PIL import Image
    from websockets.sync.client import connect
    DEPENDENCY_ERROR = None
except ImportError as error:
    DEPENDENCY_ERROR = str(error)


def pack_numpy(value):
    """Encode the numeric array markers understood by deploy/msgpack_numpy.py."""
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
        if isinstance(value, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": value.tobytes(),
                b"dtype": value.dtype.str,
                b"shape": value.shape,
            }
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Cannot encode {type(value).__name__} as MessagePack")


def unpack_numpy(value):
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported NumPy dtype: {dtype}")
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


class LingBotClient:
    def __init__(self):
        self.socket = None
        self.connection_context = None
        self.robot = None
        self.timeout = 30.0

    def receive(self):
        response = self.socket.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise RuntimeError(f"LingBot server error: {response}")
        return json_value(msgpack.unpackb(response, object_hook=unpack_numpy))

    def exchange(self, observation):
        if self.socket is None:
            raise RuntimeError("Configure the LingBot client first")
        try:
            self.socket.send(msgpack.packb(observation, default=pack_numpy))
            return self.receive()
        except Exception:
            # A late response after a timeout must never serve the next request.
            self.close()
            raise

    def dispatch(self, request):
        operation = request["op"]
        if operation == "close":
            self.close()
            return {"closed": True}
        if operation == "configure":
            if DEPENDENCY_ERROR:
                raise RuntimeError(
                    "Install requirements-lingbot-vla.txt for the LingBot client: " + DEPENDENCY_ERROR
                )
            if self.socket is not None:
                raise ValueError("LingBot client is already configured")
            self.timeout = float(request.get("timeoutMs", 30000)) / 1000
            if self.timeout <= 0:
                raise ValueError("timeoutMs must be positive")
            self.robot = request.get("robot", "robotwin")
            context = connect(
                request["url"],
                compression=None,
                max_size=None,
                # Official GPU inference blocks the server event loop, including pongs.
                ping_interval=None,
                ping_timeout=None,
                open_timeout=self.timeout,
                close_timeout=min(self.timeout, 1.0),
            )
            self.socket = context.__enter__()
            self.connection_context = context
            try:
                return self.receive()
            except Exception:
                self.close()
                raise
        if self.socket is None:
            raise RuntimeError("Configure the LingBot client first")
        if operation == "reset":
            return self.exchange({"reset": True, "robo_name": self.robot})
        if operation == "infer":
            observation = {}
            for key, image in request["images"].items():
                with Image.open(io.BytesIO(base64.b64decode(image["data"]))) as source:
                    observation[key] = np.array(source.convert("RGB"), dtype=np.uint8)
            for key, state in request["states"].items():
                observation[key] = np.asarray(state, dtype=np.float32)
            observation["task"] = request["task"]
            return self.exchange(observation)
        raise ValueError(f"Unknown LingBot operation: {operation}")

    def close(self):
        if self.socket is not None:
            context, self.connection_context = self.connection_context, None
            self.socket = None
            context.__exit__(None, None, None)


def main():
    client = LingBotClient()

    def terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    print(json.dumps({"event": "ready"}), flush=True)
    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                result = client.dispatch(request)
                response = json.dumps({"id": request.get("id"), "ok": True, "result": result}, allow_nan=False)
            except Exception as error:
                print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
                response = json.dumps({"id": request.get("id"), "ok": False, "error": str(error)})
            print(response, flush=True)
            if request.get("op") == "close":
                break
    finally:
        client.close()


if __name__ == "__main__":
    main()
