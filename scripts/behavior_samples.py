#!/usr/bin/env python3
"""Export a fixed RGB/proprioception sample from BEHAVIOR's LeRobot v3 demos.

Reads the dataset only. Reference actions are written beside observations for
offline comparison, never inside observations sent to a policy.
Requires pyarrow and OpenCV in the selected Python environment.
"""

import argparse
import base64
from datetime import datetime, timezone
import json
import math
from pathlib import Path


CAMERAS = (
    ("head", "observation.rgb.zed_link_camera_0"),
    ("left_wrist", "observation.rgb.left_realsense_link_camera_0"),
    ("right_wrist", "observation.rgb.right_realsense_link_camera_0"),
)


def select_episodes(rows, tasks, task_ids):
    """Choose the smallest episode_index for each requested task, before reading frames."""
    selected = {}
    for row in rows:
        labels = row["tasks"]
        for task_id in task_ids:
            task = tasks[task_id]
            if task["task"] not in labels and task["task_name"] not in labels:
                continue
            if task_id not in selected or row["episode_index"] < selected[task_id]["episode_index"]:
                selected[task_id] = row
    missing = [task_id for task_id in task_ids if task_id not in selected]
    if missing:
        raise ValueError(f"No episode metadata found for task indices {missing}")
    return [(tasks[task_id], selected[task_id]) for task_id in task_ids]


def frame_indices(length, fractions):
    if length < 1:
        raise ValueError("An episode must contain at least one frame")
    if any(not math.isfinite(fraction) or not 0 <= fraction <= 1 for fraction in fractions):
        raise ValueError("Frame fractions must be between 0 and 1")
    return [math.floor((length - 1) * fraction) for fraction in fractions]


def data_path(info, episode):
    return info["data_path"].format(
        chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
    )


def video_location(info, episode, video_key, timestamp_seconds):
    """Map an episode-local frame timestamp into its concatenated video shard."""
    prefix = f"videos/{video_key}"
    relative_path = info["video_path"].format(
        video_key=video_key,
        chunk_index=episode[f"{prefix}/chunk_index"],
        file_index=episode[f"{prefix}/file_index"],
    )
    offset = float(episode[f"{prefix}/from_timestamp"])
    timestamp = offset + float(timestamp_seconds)
    fps = float(info["features"][video_key]["info"]["video.fps"])
    return {
        "videoKey": video_key,
        "file": relative_path,
        "episodeOffsetSeconds": offset,
        "timestampSeconds": timestamp,
        "frameIndex": round(timestamp * fps),
        "fps": fps,
    }


def build_sample(task, episode, fraction, row, images, camera_sources, data_file):
    state = list(row["observation.state"])
    action = list(row["action"])
    if len(state) != 61 or len(action) != 23:
        raise ValueError("Expected the challenge's raw 61D state and 23D action")
    sample_id = f"task-{task['task_index']:03d}:episode-{row['episode_index']}:frame-{row['frame_index']}"
    return {
        "id": sample_id,
        "taskIndex": task["task_index"],
        "taskName": task["task_name"],
        "episodeIndex": int(row["episode_index"]),
        "frameIndex": int(row["frame_index"]),
        "fraction": fraction,
        "timestampSeconds": float(row["timestamp"]),
        "datasetIndex": int(row["index"]),
        "source": {
            "dataFile": data_file,
            "episodeDataFromIndex": int(episode["dataset_from_index"]),
            "episodeDataToIndex": int(episode["dataset_to_index"]),
            "cameras": camera_sources,
        },
        "observation": {"id": sample_id, "text": task["task"], "state": state, "images": images},
        "referenceAction": action,
    }


def load_episode_metadata(dataset, task_ids):
    import pyarrow.parquet as pq

    tasks = {
        task["task_index"]: task
        for line in (dataset / "meta/tasks.jsonl").read_text().splitlines()
        if line.strip()
        for task in [json.loads(line)]
    }
    missing = [task_id for task_id in task_ids if task_id not in tasks]
    if missing:
        raise ValueError(f"Unknown task indices: {missing}")
    columns = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index", "dataset_from_index", "dataset_to_index"]
    for _, video_key in CAMERAS:
        columns.extend(f"videos/{video_key}/{field}" for field in ["chunk_index", "file_index", "from_timestamp", "to_timestamp"])
    rows = []
    for path in sorted((dataset / "meta/episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return select_episodes(rows, tasks, task_ids)


def read_data_frames(dataset, info, episode, indices):
    import pyarrow.parquet as pq

    columns = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]
    expected = [int(episode["dataset_from_index"]) + index for index in indices]
    table = pq.read_table(dataset / data_path(info, episode), columns=columns, filters=[("index", "in", expected)])
    rows = {int(row["frame_index"]): row for row in table.to_pylist()}
    for index in indices:
        row = rows.get(index)
        if row is None or row["episode_index"] != episode["episode_index"]:
            raise ValueError(f"Missing frame {index} of episode {episode['episode_index']} in its data shard")
    return rows


def read_image(dataset, location, name, expected_shape, jpeg_quality):
    import cv2

    capture = cv2.VideoCapture(str(dataset / location["file"]))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, location["frameIndex"])
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {location['frameIndex']} from {location['file']}")
    finally:
        capture.release()
    if list(frame.shape) != expected_shape:
        raise ValueError(f"Unexpected {name} image shape {frame.shape}; expected {expected_shape}")
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError(f"Could not encode {name} image")
    return {"name": name, "mimeType": "image/jpeg", "data": base64.b64encode(encoded).decode("ascii"), "width": frame.shape[1], "height": frame.shape[0]}


def export_samples(dataset, output, task_ids=(0, 1, 2), fractions=(0, 0.25, 0.5), jpeg_quality=95):
    dataset = Path(dataset).resolve()
    output = Path(output).resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("Choose an output directory outside the read-only dataset")
    info = json.loads((dataset / "meta/info.json").read_text())
    selected = load_episode_metadata(dataset, task_ids)
    plan = [{
        "taskIndex": task["task_index"], "taskName": task["task_name"],
        "episodeIndex": int(episode["episode_index"]), "length": int(episode["length"]),
        "frameIndices": frame_indices(int(episode["length"]), fractions),
    } for task, episode in selected]
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "formatVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset), "datasetFormat": info["codebase_version"], "robot": info["robot_type"],
        "fps": info["fps"], "stateDimensions": 61, "actionDimensions": 23,
        "selection": {"taskIndices": list(task_ids), "episodeRule": "smallest episode_index for each task", "frameFractions": list(fractions), "frameIndexRule": "floor((episode_length - 1) * fraction)", "plan": plan},
        "cameras": [{"name": name, "videoKey": key, "shape": info["features"][key]["shape"]} for name, key in CAMERAS],
        "imageEncoding": {"mimeType": "image/jpeg", "quality": jpeg_quality, "resized": False},
        "observationFields": ["id", "text", "state", "images"],
        "referenceActionLocation": "sample.referenceAction (excluded from observation)",
        "scope": "offline RGB/proprioception samples; no environment execution or task-success measurement",
        "status": "planned", "samplesFile": "samples.jsonl",
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    count = 0
    with (output / "samples.jsonl").open("x") as stream:
        for (task, episode), item in zip(selected, plan):
            rows = read_data_frames(dataset, info, episode, item["frameIndices"])
            for fraction, index in zip(fractions, item["frameIndices"]):
                row = rows[index]
                if row["task_index"] != task["task_index"]:
                    raise ValueError(f"Data task index disagrees with metadata for episode {episode['episode_index']}")
                images, sources = [], []
                for name, key in CAMERAS:
                    location = video_location(info, episode, key, row["timestamp"])
                    sources.append({"name": name, **location})
                    images.append(read_image(dataset, location, name, info["features"][key]["shape"], jpeg_quality))
                sample = build_sample(task, episode, fraction, row, images, sources, data_path(info, episode))
                stream.write(json.dumps(sample, separators=(",", ":"), allow_nan=False) + "\n")
                count += 1
    manifest.update(status="complete", samples=count)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="BEHAVIOR challenge demo directory (LeRobot v3) containing meta/info.json")
    parser.add_argument("--output", required=True, help="Independent output directory outside the dataset")
    parser.add_argument("--tasks", default="0,1,2", help="Task indices fixed before sampling (default: 0,1,2)")
    parser.add_argument("--fractions", default="0,0.25,0.5", help="Episode frame fractions (default: 0,0.25,0.5)")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    manifest = export_samples(args.dataset, args.output, [int(value) for value in args.tasks.split(",")], [float(value) for value in args.fractions.split(",")], args.jpeg_quality)
    print(json.dumps({"output": str(Path(args.output).resolve()), "samples": manifest["samples"], "plan": manifest["selection"]["plan"]}, indent=2))


if __name__ == "__main__":
    main()
