#!/usr/bin/env python3
"""Persistent RealSense capture worker: one JSON request per stdin line, one JSON reply per stdout line.
Keeps numpy/cv2 loaded and the depth scale cached so a capture costs only the socket round-trip + encode."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_u2_sense as S  # noqa: E402
import numpy as np  # noqa: E402

SOCK = os.environ.get("G1_REALSENSE_SOCKET", S.DEFAULT_SOCKET)
_scale = None

def capture(req):
    global _scale
    t0 = time.time()
    header, payload = S._request(SOCK, "capture", float(req.get("timeout", 4.0)))
    t1 = time.time()
    cm, dm = header.get("color") or {}, header.get("depth") or {}
    cb, db = int(cm.get("bytes", 0)), int(dm.get("bytes", 0))
    out = {"ok": True, "captured_at": t1}
    os.environ["G1_SENSE_FILTER"] = req.get("filter", "") or ""
    if cb:
        color = np.frombuffer(payload[:cb], dtype=np.dtype(cm["dtype"])).reshape(tuple(cm["shape"]))
        fmt = str(cm.get("format") or cm.get("encoding") or "bgr8").lower()
        # depth_only: the caller wants the live depth summary for the gate, not pixels; skip the encode.
        if not req.get("depth_only"):
            jpeg, w, h = S.encode_jpeg(color, fmt, int(req.get("max_width", 640)), int(req.get("quality", 80)))
            import base64
            out.update({"image_b64": base64.b64encode(jpeg).decode("ascii"), "width": w, "height": h, "jpeg_bytes": len(jpeg)})
        # scene signature (dhash, brightness, left/right split) for the memory; ~1 ms, off by request,
        # and never for a depth-only re-read (the caller keeps only the depth summary)
        if not req.get("no_scene") and not req.get("depth_only"):
            try:
                out["scene"] = S.scene_signature(color, fmt)
            except Exception as exc:  # a missing cv2 build or an odd frame must not cost the capture
                out["scene_error"] = f"{type(exc).__name__}: {exc}"
    if db and not req.get("no_depth"):
        depth = np.frombuffer(payload[cb:cb + db], dtype=np.dtype(dm["dtype"])).reshape(tuple(dm["shape"]))
        if _scale is None:
            _scale = S._depth_scale(header, SOCK, 2.0)
        out["depth"] = S.depth_stats(depth.astype("float32") * _scale)
    t2 = time.time()
    out["timing"] = {"socket_s": round(t1 - t0, 4), "encode_s": round(t2 - t1, 4)}
    return out

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = None
    try:
        req = json.loads(line)
        res = capture(req) if req.get("op", "capture") == "capture" else {"ok": False, "error": "unknown op"}
    except Exception as exc:  # keep the worker alive
        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    res["id"] = req.get("id") if isinstance(req, dict) else None
    sys.stdout.write(json.dumps(res) + "\n"); sys.stdout.flush()
