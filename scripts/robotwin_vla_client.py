"""JSON-lines bridge for the RoboTwin seven-model native WebSocket services."""

import base64
import io
import json
import signal
import sys

from lingbot_vla_client import DEPENDENCY_ERROR, LingBotClient

if DEPENDENCY_ERROR is None:
    import numpy as np
    from PIL import Image


class RobotwinVlaClient(LingBotClient):
    def dispatch(self, request):
        operation = request["op"]
        if operation == "configure":
            if DEPENDENCY_ERROR:
                raise RuntimeError("Install msgpack, numpy, Pillow and websockets: " + DEPENDENCY_ERROR)
            metadata = super().dispatch(request)
            return {key: metadata[key] for key in ("model", "action_type", "execute_horizon") if key in metadata}
        if operation == "reset":
            return self.exchange({"reset": True, "policy_seed": request["policy_seed"]})
        if operation == "infer":
            observation = {}
            for camera, image in request["images"].items():
                with Image.open(io.BytesIO(base64.b64decode(image["data"]))) as source:
                    observation[camera] = {"rgb": np.array(source.convert("RGB"), dtype=np.uint8)}
            state = {
                field: {key: np.asarray(value, dtype=np.float32) if isinstance(value, list) else value for key, value in request[field].items()}
                for field in ("joint_action", "endpose")
            }
            return self.exchange({"instruction": request["instruction"], "observation": observation, **state})
        return super().dispatch(request)


def main():
    client = RobotwinVlaClient()

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
