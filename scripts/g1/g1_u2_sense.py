#!/usr/bin/env python3
"""Read-only RealSense snapshot for the Unitree G1 EDU (U2) gateway host.

Talks to the robot's RealSense frame service over its Unix socket, returns one RGB frame as base64 JPEG plus depth
clearance statistics. Never touches DDS, the sport service, or the arm service.
Runs on the G1's development computer (PC2; Python 3.8, numpy, cv2). ``--dry-run`` needs nothing.
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import struct
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple

DEFAULT_SOCKET = "/run/g1-realsense/realsense.sock"


class SenseError(RuntimeError):
    pass


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = connection.recv(min(remaining, 1 << 20))
        if not chunk:
            raise SenseError("RealSense socket closed early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _request(socket_path: str, operation: str, timeout: float) -> Tuple[Dict[str, Any], bytes]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        try:
            connection.connect(socket_path)
        except OSError as exc:
            raise SenseError("cannot connect to RealSense service at %s: %s" % (socket_path, exc)) from exc
        connection.sendall((json.dumps({"op": operation}) + "\n").encode("utf-8"))
        header_size = struct.unpack("!I", _recv_exact(connection, 4))[0]
        header = json.loads(_recv_exact(connection, header_size).decode("utf-8"))
        if not header.get("ok", True):
            raise SenseError(str(header.get("error") or "RealSense %s failed" % operation))
        color_bytes = int((header.get("color") or {}).get("bytes", 0))
        depth_bytes = int((header.get("depth") or {}).get("bytes", 0))
        payload = _recv_exact(connection, color_bytes + depth_bytes) if color_bytes + depth_bytes else b""
    return header, payload


def _depth_scale(header: Dict[str, Any], socket_path: str, timeout: float) -> float:
    for candidate in (
        (header.get("depth") or {}).get("scale_m"),
        (header.get("depth") or {}).get("depth_scale_m"),
        header.get("depth_scale_m"),
        ((header.get("intrinsics") or {}).get("depth_scale_m")),
    ):
        if candidate:
            return float(candidate)
    try:
        health, _ = _request(socket_path, "health", timeout)
        intrinsics = health.get("intrinsics") or {}
        if intrinsics.get("depth_scale_m"):
            return float(intrinsics["depth_scale_m"])
    except Exception:
        pass
    return 0.001


def depth_stats(depth_m: Any) -> Dict[str, Any]:
    """Clearance statistics from a metric depth image (invalid depth = 0 = blocked)."""
    import numpy as np

    height, width = depth_m.shape[:2]
    rows = slice(int(height * 0.30), int(height * 0.70))

    def window(c0: float, c1: float) -> Dict[str, Any]:
        block = depth_m[rows, int(width * c0):int(width * c1)]
        valid = block[(block > 0.1) & (block < 10.0)]
        total = block.size
        if total == 0:
            return {"median_m": None, "min_m": None, "valid_fraction": 0.0}
        if valid.size == 0:
            return {"median_m": None, "min_m": None, "valid_fraction": 0.0}
        return {
            "median_m": round(float(np.median(valid)), 3),
            "min_m": round(float(np.percentile(valid, 5)), 3),
            "valid_fraction": round(float(valid.size) / float(total), 3),
        }

    center = window(0.40, 0.60)
    left = window(0.05, 0.30)
    right = window(0.70, 0.95)
    front = window(0.25, 0.75)
    return {
        "center_m": center["median_m"],
        "left_m": left["median_m"],
        "right_m": right["median_m"],
        "front_min_m": front["min_m"],
        "center_valid_fraction": center["valid_fraction"],
        "front_valid_fraction": front["valid_fraction"],
    }


def dhash_from_gray(gray: Any) -> str:
    """64-bit difference hash of an 8x9 grayscale array as 16 hex digits.

    Each bit is "the right neighbour is brighter than this pixel" (a rising left-to-right edge); two
    frames of the same view differ in a handful of bits, two different rooms in dozens. Pure numpy so
    it can be tested without OpenCV."""
    import numpy as np

    rows = np.asarray(gray, dtype="float32")
    if rows.shape != (8, 9):
        raise SenseError("dhash needs an 8x9 grayscale array, got %r" % (rows.shape,))
    bits = (rows[:, 1:] > rows[:, :-1]).astype("uint8").reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return "%016x" % value


def scene_signature(color: Any, color_format: str) -> Dict[str, Any]:
    """A coarse, comparable description of what the camera sees, for the memory's situation signature.

    dhash tells "same place as before" apart from "a different doorway that also reads centre-near";
    brightness and the left/right split are cheap extra cues. Never used by the gate."""
    import cv2  # type: ignore
    import numpy as np

    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY if color_format == "rgb8" else cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    width = gray.shape[1]
    left = float(np.mean(gray[:, : width // 2])) / 255.0
    right = float(np.mean(gray[:, width // 2 :])) / 255.0
    return {
        "dhash": dhash_from_gray(small),
        "brightness": round(float(np.mean(gray)) / 255.0, 3),
        "left_right": round(left - right, 3),
    }


def encode_jpeg(color: Any, color_format: str, max_width: int, quality: int) -> Tuple[bytes, int, int]:
    import cv2  # type: ignore

    frame = color
    if color_format == "rgb8":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]
    if max_width and width > max_width:
        scale = max_width / float(width)
        frame = cv2.resize(frame, (max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)
        height, width = frame.shape[:2]
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise SenseError("JPEG encoding failed")
    return encoded.tobytes(), width, height


def snapshot(socket_path: str, timeout: float, max_width: int, quality: int, with_image: bool, color_order: str) -> Dict[str, Any]:
    import numpy as np

    header, payload = _request(socket_path, "capture", timeout)
    color_meta = header.get("color") or {}
    depth_meta = header.get("depth") or {}
    color_bytes = int(color_meta.get("bytes", 0))
    result: Dict[str, Any] = {
        "ok": True,
        "captured_at": time.time(),
        "source": socket_path,
        "header_keys": sorted(header.keys()),
        "color_meta": {k: v for k, v in color_meta.items() if k != "bytes"},
    }
    if color_bytes:
        color = np.frombuffer(payload[:color_bytes], dtype=np.dtype(color_meta["dtype"])).reshape(tuple(color_meta["shape"]))
        fmt = color_order if color_order != "auto" else str(color_meta.get("format") or color_meta.get("encoding") or "bgr8").lower()
        if with_image:
            jpeg, width, height = encode_jpeg(color, fmt, max_width, quality)
            result.update({"image_b64": base64.b64encode(jpeg).decode("ascii"), "width": width, "height": height, "jpeg_bytes": len(jpeg)})
            try:
                result["scene"] = scene_signature(color, fmt)
            except Exception as exc:  # the memory's cue must not cost a capture that already has its JPEG
                result["scene_error"] = f"{type(exc).__name__}: {exc}"
        else:
            result.update({"width": int(color.shape[1]), "height": int(color.shape[0])})
    depth_bytes = int(depth_meta.get("bytes", 0))
    if depth_bytes:
        depth = np.frombuffer(payload[color_bytes:color_bytes + depth_bytes], dtype=np.dtype(depth_meta["dtype"])).reshape(tuple(depth_meta["shape"]))
        scale = _depth_scale(header, socket_path, timeout)
        result["depth"] = depth_stats(depth.astype("float32") * scale)
        result["depth"]["scale_m"] = scale
    else:
        result["depth"] = None
    return result


def _dry_run(with_image: bool) -> Dict[str, Any]:
    return {
        "ok": True,
        "dry_run": True,
        "captured_at": time.time(),
        "width": 640 if with_image else None,
        "height": 480 if with_image else None,
        "image_b64": None,
        "depth": {"center_m": 2.0, "left_m": 1.2, "right_m": 1.2, "front_min_m": 1.5,
                  "center_valid_fraction": 1.0, "front_valid_fraction": 1.0, "scale_m": 0.001},
        "note": "dry-run: synthetic clearance, no camera was read",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="One read-only RealSense snapshot (RGB JPEG + depth clearance).")
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--timeout", type=float, default=4.0)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--max-width", type=int, default=640)
    snap.add_argument("--quality", type=int, default=80)
    snap.add_argument("--no-image", action="store_true")
    snap.add_argument("--color-order", choices=("auto", "rgb8", "bgr8"), default="auto")
    snap.add_argument("--dry-run", action="store_true")
    snap.add_argument("--json", action="store_true", dest="as_json", help="accepted for symmetry; output is always JSON")
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            payload = _dry_run(not args.no_image)
        else:
            payload = snapshot(args.socket, args.timeout, args.max_width, args.quality, not args.no_image, args.color_order)
    except (SenseError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 3
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
