import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("behavior_samples", Path(__file__).resolve().parents[1] / "scripts/behavior_samples.py")
samples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(samples)


class BehaviorSampleTest(unittest.TestCase):
    def test_selection_is_order_independent_and_uses_episode_length(self):
        tasks = {index: {"task_index": index, "task_name": f"task-{index}", "task": f"instruction {index}"} for index in (0, 1, 2)}
        rows = [
            {"episode_index": 22, "tasks": ["instruction 2"], "length": 101},
            {"episode_index": 8, "tasks": ["instruction 0"], "length": 101},
            {"episode_index": 2, "tasks": ["instruction 0"], "length": 101},
            {"episode_index": 11, "tasks": ["instruction 1"], "length": 101},
        ]
        selected = samples.select_episodes(rows, tasks, [0, 1, 2])
        self.assertEqual([row["episode_index"] for _, row in selected], [2, 11, 22])
        self.assertEqual(samples.frame_indices(101, [0, 0.25, 0.5]), [0, 25, 50])

    def test_video_timestamp_includes_episode_offset_inside_shared_shard(self):
        key = "observation.rgb.zed_link_camera_0"
        info = {"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4", "features": {key: {"info": {"video.fps": 30}}}}
        episode = {f"videos/{key}/chunk_index": 2, f"videos/{key}/file_index": 4, f"videos/{key}/from_timestamp": 120.5}
        location = samples.video_location(info, episode, key, 1.2)
        self.assertEqual(location["timestampSeconds"], 121.7)
        self.assertEqual(location["frameIndex"], 3651)
        self.assertEqual(location["file"], f"videos/{key}/chunk-002/file-004.mp4")

    def test_observation_excludes_reference_action_and_privileged_fields(self):
        row = {"observation.state": [0.25] * 61, "action": [0.75] * 23, "episode_index": 4, "frame_index": 9, "timestamp": 0.3, "index": 109, "next.reward": 1, "next.terminated": True, "object_poses": {"target": [1, 2, 3]}}
        task = {"task_index": 0, "task_name": "turning_on_radio", "task": "Turn on the radio."}
        episode = {"dataset_from_index": 100, "dataset_to_index": 200}
        images = [{"name": "head", "mimeType": "image/jpeg", "data": "fixture"}]
        sample = samples.build_sample(task, episode, 0.25, row, images, [], "data/chunk-000/file-000.parquet")
        self.assertEqual(set(sample["observation"]), {"id", "text", "state", "images"})
        self.assertEqual(sample["observation"]["state"], [0.25] * 61)
        self.assertEqual(sample["referenceAction"], [0.75] * 23)
        self.assertNotIn("referenceAction", sample["observation"])
        self.assertEqual(sample["source"]["episodeDataFromIndex"], 100)


if __name__ == "__main__":
    unittest.main()
