#!/usr/bin/env python3
"""Replay sense worker: serves recorded JPEG frames over the sense-worker JSON-lines protocol.

A drop-in for sense_worker.py when there is no RealSense: point G1_SENSE_WORKER at this file and the
whole service (gate, executor, brain, thinker) runs against recorded frames with a real VLM and no
robot. One JSON request per stdin line, one JSON reply per stdout line, same keys as sense_worker.py:

  request  {"op":"capture","max_width":320,"quality":40,"filter":"","no_depth":false,"depth_only":false,"id":"7"}
  reply    {"ok":true,"image_b64":...,"width":320,"height":240,"jpeg_bytes":n,"depth":{...},"scene":{...},
            "captured_at":t,"timing":{"socket_s":0,"encode_s":e},"id":"7"}

Frames come from --frames (a directory of *.jpg, served in name order and looped). Depth comes from a
sidecar <frame>.json when present, else from --depth-json (a {filename: depth} map), else from the
--depth constant. --lidar-snapshot writes a clear MID-360 ring every 0.2 s so the gate's lidar check
passes; omit it to exercise the lidar_snapshot_missing path.

Never touches a robot. Python 3.8+, stdlib + numpy; Pillow if the frames must be resized/re-encoded
or a scene signature is wanted (without it the raw JPEG bytes are served as-is).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def parse_depth(text: str) -> Dict[str, Any]:
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 3:
        raise SystemExit("--depth wants center,left,right in metres")
    center, left, right = parts
    return {
        "center_m": center,
        "left_m": left,
        "right_m": right,
        "front_min_m": round(min(center, left, right) * 0.9, 3),
        "center_valid_fraction": 1.0,
        "front_valid_fraction": 1.0,
    }


class LidarWriter(threading.Thread):
    """Writes a clear obstacle ring the way the Livox producer does, so lidarGate() reads 'clear'."""

    def __init__(self, path: str, radius_m: float = 4.0, period_s: float = 0.2) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.radius_m = radius_m
        self.period_s = period_s
        self.stop = threading.Event()

    def run(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        while not self.stop.is_set():
            snapshot = {"unix_s": time.time(), "ring_m": [self.radius_m] * 72, "points_per_s": 200000}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(snapshot, handle)
            os.replace(tmp, self.path)
            self.stop.wait(self.period_s)


class Replay:
    def __init__(self, frames_dir: str, depth_default: Dict[str, Any], depth_json: Optional[str]) -> None:
        names = sorted(name for name in os.listdir(frames_dir) if name.lower().endswith((".jpg", ".jpeg")))
        if not names:
            raise SystemExit("no *.jpg frames in %s" % frames_dir)
        self.frames_dir = frames_dir
        self.names: List[str] = names
        self.index = 0
        self.current: Optional[str] = None
        self.depth_default = depth_default
        self.depth_map: Dict[str, Any] = {}
        if depth_json:
            with open(depth_json, "r", encoding="utf-8") as handle:
                self.depth_map = json.load(handle)
        try:
            from PIL import Image  # type: ignore

            self.Image = Image
        except Exception:
            self.Image = None

    def depth_for(self, name: str) -> Dict[str, Any]:
        sidecar = os.path.join(self.frames_dir, os.path.splitext(name)[0] + ".json")
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return self.depth_map.get(name) or self.depth_default

    def scene_for(self, image: Any) -> Optional[Dict[str, Any]]:
        try:
            import numpy as np

            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "g1"))
            import g1_u2_sense as S

            gray = image.convert("L")
            small = np.asarray(gray.resize((9, 8), self.Image.BOX), dtype="float32")
            full = np.asarray(gray, dtype="float32")
            width = full.shape[1]
            return {
                "dhash": S.dhash_from_gray(small),
                "brightness": round(float(full.mean()) / 255.0, 3),
                "left_right": round((float(full[:, : width // 2].mean()) - float(full[:, width // 2 :].mean())) / 255.0, 3),
            }
        except Exception:
            return None

    def capture(self, req: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        if req.get("depth_only"):
            # The gate's live re-read: the depth of the frame most recently served, as a real re-read
            # would see the same scene the model saw; the sequence does not advance.
            name = self.current or self.names[0]
        else:
            name = self.names[self.index % len(self.names)]
            self.index += 1
            self.current = name
        path = os.path.join(self.frames_dir, name)
        out: Dict[str, Any] = {"ok": True, "captured_at": t0, "frame": name}
        if not req.get("depth_only"):
            with open(path, "rb") as handle:
                raw = handle.read()
            width = height = None
            data = raw
            if self.Image is not None:
                image = self.Image.open(io.BytesIO(raw)).convert("RGB")
                max_width = int(req.get("max_width", 640) or 0)
                if max_width and image.width > max_width:
                    image = image.resize((max_width, int(round(image.height * max_width / image.width))))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=int(req.get("quality", 80) or 80))
                data = buffer.getvalue()
                width, height = image.size
                if not req.get("no_scene"):
                    scene = self.scene_for(image)
                    if scene:
                        out["scene"] = scene
            out.update({"image_b64": base64.b64encode(data).decode("ascii"), "width": width, "height": height, "jpeg_bytes": len(data)})
        if not req.get("no_depth"):
            out["depth"] = self.depth_for(name)
        out["timing"] = {"socket_s": 0.0, "encode_s": round(time.time() - t0, 4)}
        return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", default=os.environ.get("G1_REPLAY_FRAMES", ""), help="directory of *.jpg frames")
    parser.add_argument("--depth", default=os.environ.get("G1_REPLAY_DEPTH", "2.5,2.5,2.5"), help="center,left,right metres when no sidecar/map")
    parser.add_argument("--depth-json", default=os.environ.get("G1_REPLAY_DEPTH_JSON") or None)
    parser.add_argument("--lidar-snapshot", default=os.environ.get("G1_REPLAY_LIDAR_SNAPSHOT") or None, help="write a clear ring here every 0.2 s")
    args = parser.parse_args(argv)
    if not args.frames:
        raise SystemExit("--frames (or G1_REPLAY_FRAMES) is required")

    replay = Replay(args.frames, parse_depth(args.depth), args.depth_json)
    lidar = None
    if args.lidar_snapshot:
        lidar = LidarWriter(args.lidar_snapshot)
        lidar.start()
    sys.stdout.write(json.dumps({"ok": True, "event": "ready", "replay": True, "frames": len(replay.names), "lidar_snapshot": args.lidar_snapshot}) + "\n")
    sys.stdout.flush()
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req: Any = None
            try:
                req = json.loads(line)
                res = replay.capture(req) if req.get("op", "capture") == "capture" else {"ok": False, "error": "unknown op"}
            except Exception as exc:
                res = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
            res["id"] = req.get("id") if isinstance(req, dict) else None
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
    finally:
        if lidar is not None:
            lidar.stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
