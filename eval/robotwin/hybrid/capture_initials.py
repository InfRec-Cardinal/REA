#!/usr/bin/env python3
"""Capture exact manifest initial observations, without executing a policy or action."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from parallel import write_json
from export_labels import episode_directories


def load(path):
    return json.loads(Path(path).read_text())


def compare_initials(left_root, right_root, instances):
    left_root, right_root = Path(left_root).resolve(), Path(right_root).resolve()
    left = episode_directories(left_root, instances)
    right = episode_directories(right_root, instances)
    differences = []
    for instance in instances:
        instance_id = instance["instance_id"]
        paths = [mapping.get(instance_id, root / instance_id) / "initial-router-input.json"
                 for mapping, root in ((left, left_root), (right, right_root))]
        if any(not path.exists() for path in paths):
            differences.append({"instance_id": instance_id, "fields": ["missing_initial_observation"]})
            continue
        rows = [load(path) for path in paths]
        changed = [key for key in ("instance_id", "task", "seed", "task_text") if rows[0].get(key) != rows[1].get(key)]
        if json.loads(rows[0]["state_text"]) != json.loads(rows[1]["state_text"]):
            changed.append("state_text")
        for index, camera in enumerate(("head_camera", "left_camera", "right_camera")):
            files = [path.parent / row["images"][index] for path, row in zip(paths, rows)]
            if files[0].read_bytes() != files[1].read_bytes():
                changed.append(camera)
        if changed:
            differences.append({"instance_id": instance_id, "fields": changed})
    return {"leftRoot": str(left_root), "rightRoot": str(right_root), "requested": len(instances),
            "matched": len(instances) - len(differences), "differences": differences,
            "status": "matched" if not differences else "different"}


def capture_one(config, instance, output, log):
    output.mkdir(parents=True, exist_ok=True)
    requests = [
        {"id": 1, "op": "reset", "subset": config["subset"], "harnessRoot": config["harnessRoot"],
         "instance": instance, "output": str(output)},
        {"id": 2, "op": "close"},
    ]
    started = time.monotonic()
    process = subprocess.run([config["simulatorPython"], str(Path(__file__).with_name("environment.py"))],
                             input="".join(json.dumps(request) + "\n" for request in requests),
                             stdout=subprocess.PIPE, stderr=log, text=True,
                             env=dict(os.environ, **config.get("simulatorEnv", {})),
                             timeout=(config.get("environmentTimeoutMs", 660000) / 1000) + 60, check=True)
    replies = [json.loads(line) for line in process.stdout.splitlines() if line.startswith("{")]
    reset = next(reply for reply in replies if reply.get("id") == 1)
    if not reset["ok"]:
        raise RuntimeError(reset["error"])
    status = {"status": "captured", "instance": instance, "nativeActions": 0,
              "capturedAt": datetime.now(timezone.utc).isoformat(), "elapsedSeconds": time.monotonic() - started}
    write_json(output / "capture.json", status)
    return status


def summarize_capture(output):
    output = Path(output)
    plan = load(output / "plan.json")
    previous = load(output / "capture-summary.json")
    captured = errors = 0
    for shard in plan["shards"]:
        for instance_id in shard["instanceIds"]:
            directory = Path(shard["output"]) / instance_id
            if (directory / "capture.json").exists():
                captured += 1
            elif (directory / "capture-error.json").exists():
                errors += 1
    pending = plan["requested"] - captured - errors
    result = {"requested": plan["requested"], "captured": captured, "errors": errors, "pending": pending,
              "startedAt": previous["startedAt"],
              "elapsedSeconds": (datetime.now(timezone.utc) - datetime.fromisoformat(previous["startedAt"])).total_seconds(),
              "status": "running" if pending else ("failed" if errors else "completed")}
    write_json(output / "capture-summary.json", result)
    return result


def resume_shards(output, selected):
    """Continue unattempted scenes after an observed reset failure; do not retry that failure."""
    output = Path(output)
    for index in selected:
        root = output / f"shard-{index:02d}"
        config = load(root / "config.json")
        with (root / "capture.log").open("a") as log:
            for instance in load(root / "instances.json"):
                directory = root / instance["instance_id"]
                if (directory / "capture.json").exists() or (directory / "capture-error.json").exists():
                    continue
                try:
                    capture_one(config, instance, directory, log)
                except Exception as error:
                    write_json(directory / "capture-error.json", {"instance": instance, "error": str(error)})
    print(json.dumps(summarize_capture(output)), flush=True)


def capture(config_paths, instance_paths, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = [load(path) for path in config_paths]
    instances = [row for path in instance_paths for row in load(path)]
    if len({row["instance_id"] for row in instances}) != len(instances):
        raise ValueError("Capture manifests contain duplicate instance IDs")
    write_json(output / "instances.json", instances)
    shards = []
    for index, config in enumerate(configs):
        root = output / f"shard-{index:02d}"
        root.mkdir(exist_ok=True)
        rows = instances[index::len(configs)]
        write_json(root / "instances.json", rows)
        write_json(root / "config.json", config)
        shards.append({"index": index, "output": str(root), "requested": len(rows),
                       "instanceIds": [row["instance_id"] for row in rows]})
    plan = {"mode": "capture_initials", "workers": len(configs), "requested": len(instances),
            "output": str(output), "shards": shards}
    write_json(output / "plan.json", plan)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    states = {row["instance_id"]: "pending" for row in instances}
    lock = threading.Lock()

    def update(instance_id=None, status=None):
        with lock:
            if instance_id:
                states[instance_id] = status
            summary = {"requested": len(instances), "captured": list(states.values()).count("captured"),
                       "errors": list(states.values()).count("error"), "pending": list(states.values()).count("pending"),
                       "startedAt": started_at, "elapsedSeconds": time.monotonic() - started,
                       "status": "running" if "pending" in states.values() else ("failed" if "error" in states.values() else "completed")}
            write_json(output / "capture-summary.json", summary)

    def worker(index):
        shard = shards[index]
        root = Path(shard["output"])
        with (root / "capture.log").open("a") as log:
            for instance in instances[index::len(configs)]:
                directory = root / instance["instance_id"]
                try:
                    if not (directory / "capture.json").exists():
                        capture_one(configs[index], instance, directory, log)
                    update(instance["instance_id"], "captured")
                except Exception as error:
                    directory.mkdir(parents=True, exist_ok=True)
                    write_json(directory / "capture-error.json", {"instance": instance, "error": str(error)})
                    update(instance["instance_id"], "error")
                    continue

    update()
    with ThreadPoolExecutor(max_workers=len(configs)) as pool:
        list(pool.map(worker, range(len(configs))))
    summary = load(output / "capture-summary.json")
    summary["status"] = "completed" if summary["captured"] == len(instances) else "failed"
    write_json(output / "capture-summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--instances", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return capture(args.configs, args.instances, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
