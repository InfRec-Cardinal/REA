#!/usr/bin/env python3
"""Score saved initial observations without reading episode outcomes."""

import argparse
import base64
from collections import Counter
import json
from pathlib import Path
from urllib.request import Request, urlopen


CAMERAS = ("head_camera", "left_camera", "right_camera")


def payload_from_initial(path):
    path = Path(path)
    initial = json.loads(path.read_text())
    images = []
    for camera, filename in zip(CAMERAS, initial["images"]):
        image_path = path.parent / filename
        images.append({"name": camera, "mimeType": "image/png",
                       "data": base64.b64encode(image_path.read_bytes()).decode("ascii")})
    return {
        "episodeId": initial["instance_id"], "step": 0,
        "observation": {"id": initial["instance_id"], "text": initial["task_text"],
                        "images": images, "state": {"state_text": initial["state_text"]}},
    }


def score_initials(run_root, instances, endpoint):
    run_root = Path(run_root).resolve()
    directories = {}
    plan_path = run_root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        for shard in plan["shards"]:
            for instance_id in shard["instanceIds"]:
                directories[instance_id] = run_root / shard["output"] / instance_id
    rows, missing = [], []
    metadata = None
    for instance in instances:
        instance_id = instance["instance_id"]
        directory = directories.get(instance_id, run_root / instance_id)
        input_path = directory / "initial-router-input.json"
        if not input_path.exists():
            missing.append({"instance_id": instance_id, "inputPath": str(input_path)})
            continue
        payload = payload_from_initial(input_path)
        request = Request(endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        metadata = result.get("metadata")
        rows.append({"instance": instance, "inputPath": str(input_path),
                     "scores": result["scores"], "selected_candidate": result["selected_candidate"],
                     "inference_seconds": result["inference_seconds"], "encoding": result.get("encoding")})
    return {"sourceRunRoot": str(run_root), "endpoint": endpoint, "requested": len(instances),
            "scored": len(rows), "missing": missing,
            "selectedCounts": dict(Counter(row["selected_candidate"] for row in rows)),
            "routerMetadata": metadata, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--instances", type=Path, help="Optional subset; defaults to run-root/instances.json")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8810/predict")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    instances = json.loads((args.instances or args.run_root / "instances.json").read_text())
    report = score_initials(args.run_root, instances, args.endpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "requested": report["requested"], "scored": report["scored"],
                      "missing": len(report["missing"]), "selectedCounts": report["selectedCounts"]}))
    return 1 if report["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
