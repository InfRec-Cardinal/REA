"""Call a REA agent as a policy from Python: a client for `rea serve` (standard library only).

Use it where a benchmark harness or a robot stack owns the control loop and asks a policy for actions:

    from rea_client import ReaSession, encode_image

    with ReaSession("/path/to/rea.config.json", rea_root="/path/to/REA") as rea:
        rea.configure(actions=[...])                       # the action specifications the evaluator accepts
        rea.reset(task="stack_bowls", seed=0, instruction="Stack the bowls.")
        while not episode_over:
            reply = rea.step({"text": instruction, "state": {...}, "images": [encode_image(rgb, "head")]})
            if reply.get("done"):
                ...                                        # REA finished the episode: hold or stop
            else:
                execute(reply["action"])                   # {"name": ..., "arguments": {...}}
        summary = rea.end({"success": evaluator_success})

The protocol is in docs/extending/policy-server.md. REA's own logs go to this process's stderr.
"""

import base64
import io
import json
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import zlib
from pathlib import Path

__all__ = ["ReaError", "ReaSession", "encode_image"]


class ReaError(RuntimeError):
    """A request that `rea serve` answered with ok: false, or a server that stopped."""


def _png(rows, channels):
    height, width = len(rows), len(rows[0])

    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    pixels = b"".join(b"\x00" + bytes(int(value) for pixel in row for value in pixel[:channels]) for row in rows)
    header = struct.pack("!IIBBBBB", width, height, 8, 2 if channels == 3 else 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


def encode_image(image, name, quality=90):
    """An observation image {name, mimeType, data} from an H x W x 3 (or 4) uint8 RGB array or nested list.

    JPEG through Pillow when it is installed, otherwise PNG without dependencies (larger, slower to send).
    """
    try:
        from PIL import Image  # type: ignore

        array = image
        if hasattr(array, "dtype") and str(array.dtype) != "uint8":
            array = array.astype("uint8")
        picture = Image.fromarray(array) if hasattr(array, "shape") else None
        if picture is not None:
            buffer = io.BytesIO()
            picture.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
            return {"name": name, "mimeType": "image/jpeg", "data": base64.b64encode(buffer.getvalue()).decode("ascii")}
    except ImportError:
        pass
    rows = image.tolist() if hasattr(image, "tolist") else image
    channels = len(rows[0][0])
    if channels not in (3, 4):
        raise ValueError(f"image {name} must be H x W x RGB or RGBA")
    return {"name": name, "mimeType": "image/png", "data": base64.b64encode(_png(rows, channels)).decode("ascii")}


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class ReaSession:
    """One `rea serve` process. Requests are sequential; every method blocks until its reply arrives."""

    def __init__(self, config, rea_root=None, node=None, env=None, output=None, timeout_s=600.0):
        root = Path(rea_root or os.environ.get("REA_ROOT") or Path(__file__).resolve().parents[2])
        cli = root / "scripts" / "rea.mjs"
        if not cli.is_file():
            raise ReaError(f"{cli} not found; set rea_root or REA_ROOT to a REA checkout")
        node = node or os.environ.get("REA_NODE") or shutil.which("node")
        if not node:
            raise ReaError("node (22.19 or newer) was not found; set node or REA_NODE")
        command = [node, str(cli), "serve", str(config)] + (["--output", str(output)] if output else [])
        self.timeout_s = timeout_s
        self._sequence = 0
        self._replies = queue.Queue()
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
                                         text=True, encoding="utf-8", bufsize=1, env=env, cwd=str(root))
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        for line in self._process.stdout:
            line = line.strip()
            if line:
                self._replies.put(line)
        self._replies.put(None)

    def request(self, op, timeout_s=None, **fields):
        if self._process.poll() is not None:
            raise ReaError(f"rea serve exited with code {self._process.returncode}")
        self._sequence += 1
        message = {**fields, "id": self._sequence, "op": op}
        try:
            self._process.stdin.write(json.dumps(message, default=_json_default) + "\n")
            self._process.stdin.flush()
        except BrokenPipeError as error:
            raise ReaError("rea serve closed its input") from error
        while True:
            try:
                line = self._replies.get(timeout=timeout_s or self.timeout_s)
            except queue.Empty as error:
                raise ReaError(f"no reply to {op} within {timeout_s or self.timeout_s} s") from error
            if line is None:
                raise ReaError(f"rea serve exited before answering {op}")
            reply = json.loads(line)
            if reply.get("id") != self._sequence:
                continue
            if not reply.get("ok"):
                raise ReaError(reply.get("error") or f"{op} failed")
            return reply.get("result")

    def configure(self, actions, timing="turn-based", resumable=False, name=None):
        return self.request("configure", actions=actions, timing=timing, resumable=resumable, **({"name": name} if name else {}))

    def reset(self, task=None, seed=0, instruction=None):
        return self.request("reset", task=task, seed=seed, **({"instruction": instruction} if instruction else {}))

    def step(self, observation, terminated=False, truncated=False, success=None, reward=None, info=None):
        return self.request("step", observation=observation, terminated=terminated, truncated=truncated,
                            success=success, reward=reward, info=info or {})

    def end(self, result=None):
        return self.request("end", result=result)

    def close(self):
        if self._process.poll() is None:
            try:
                self.request("close", timeout_s=60)
            except ReaError:
                pass
            try:
                self._process.stdin.close()
            except OSError:
                pass
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._reader.join(timeout=5)
        if self._process.stdout and not self._process.stdout.closed:
            self._process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


if __name__ == "__main__":
    print(__doc__, file=sys.stderr)
