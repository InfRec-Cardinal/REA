"""JSON-lines bridge to an openpi server using its ALOHA transforms.

Wire contract: openpi/policies/aloha_policy.py and
openpi-client/openpi_client/websocket_client_policy.py in the upstream repository.
The MessagePack array representation is shared with the LingBot transport.
"""

import base64
import io
import json
import signal
import sys

from lingbot_vla_client import DEPENDENCY_ERROR, LingBotClient

if DEPENDENCY_ERROR is None:
    import numpy as np
    from PIL import Image


class Pi05Client(LingBotClient):
    def dispatch(self, request):
        operation = request["op"]
        if operation == "configure" and DEPENDENCY_ERROR:
            raise RuntimeError("Install requirements-pi05.txt for the pi0.5 client: " + DEPENDENCY_ERROR)
        if operation == "reset":
            # openpi's public WebsocketClientPolicy.reset() sends no wire message.
            return {"reset": True}
        if operation == "infer":
            images = {}
            for camera, image in request["images"].items():
                with Image.open(io.BytesIO(base64.b64decode(image["data"]))) as source:
                    images[camera] = np.array(source.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
            return self.exchange({
                "images": images,
                "state": np.asarray(request["state"], dtype=np.float32),
                "prompt": request["prompt"],
            })
        return super().dispatch(request)


def main():
    client = Pi05Client()

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
