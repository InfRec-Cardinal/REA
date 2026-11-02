"""Run real PushT and MetaWorld scenes with OPENRAL_SCENE_TEST_ROOT set."""

import base64
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest.mock import call, patch


ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = os.environ.get("OPENRAL_SCENE_TEST_ROOT")
sys.path.insert(0, str(ROOT / "scripts"))

from robot_drivers.openral import OpenRALDriver


@unittest.skipUnless(SDK_ROOT, "Set OPENRAL_SCENE_TEST_ROOT after installing requirements-openral-scenes.txt")
class OpenRALSceneTests(unittest.TestCase):
    def run_scene(self, robot_id, scene_yaml, values, camera_name, image_size):
        driver = OpenRALDriver({
            "openralRoot": SDK_ROOT,
            "robotId": robot_id,
            "mode": "sim",
            "simEnvYaml": str(scene_yaml),
            "executeEnabled": True,
        })
        reset_patch = patch.object(driver.scene_rollout.wrapped, "reset", wraps=driver.scene_rollout.wrapped.reset)
        native_reset = reset_patch.start()
        try:
            initial = driver.reset(seed=7)
            self.assertEqual(native_reset.call_args_list, [call(seed=7)])
            initial_images = {image["name"]: image for image in initial["images"]}
            self.assertIn(camera_name, initial_images)
            image = initial_images[camera_name]
            self.assertEqual(image["mimeType"], "image/png")
            png = base64.b64decode(image["data"])
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack("!II", png[16:24]), image_size)

            preview = driver.step({"name": "scene_action", "arguments": {"values": values}})
            self.assertTrue(preview["info"]["preview"])
            self.assertEqual(preview["info"]["controlSteps"], 0)
            self.assertEqual(preview["images"], initial["images"])

            result = driver.step({
                "name": "scene_trajectory",
                "arguments": {"commands": [values, values, values]},
            }, execute=True)
            self.assertEqual(result["info"]["controlSteps"], 3)
            self.assertTrue(result["info"]["executed"])
            self.assertNotEqual(result["state"]["observation"], initial["state"]["observation"])
            final_images = {image["name"]: image for image in result["images"]}
            self.assertNotEqual(final_images[camera_name]["data"], image["data"])
            self.assertEqual(driver.observe()["images"], result["images"])
            driver.stop()
            self.assertTrue(driver.hal.estop_latched)
            restarted = driver.reset(seed=7)
            self.assertFalse(driver.hal.estop_latched)
            self.assertEqual(restarted["info"]["controlSteps"], 0)
            self.assertEqual(restarted["info"]["totalControlSteps"], 0)
            self.assertNotEqual(restarted["images"], result["images"])
            self.assertEqual(restarted["state"]["observation"], initial["state"]["observation"])
            resumed = driver.step({"name": "scene_action", "arguments": {"values": values}}, execute=True)
            self.assertEqual(resumed["info"]["controlSteps"], 1)
            self.assertNotEqual(driver.reset(seed=8)["images"], initial["images"])
            self.assertEqual(native_reset.call_args_list, [call(seed=7), call(seed=7), call(seed=8)])
        finally:
            reset_patch.stop()
            driver.close()

    def test_pusht_native_pixel_actions_advance_physics_and_named_camera(self):
        self.run_scene(
            "pusht_2d", Path(SDK_ROOT) / "scenes/sim/pusht.yaml",
            [250.0, 250.0], "camera1", (96, 96),
        )

    def test_metaworld_native_four_value_actions_advance_sawyer_and_named_camera(self):
        self.run_scene(
            "sawyer", ROOT / "examples/robots/openral/scenes/metaworld_push.yaml",
            [0.1, 0.0, 0.0, 0.0], "front", (480, 480),
        )

    def test_both_scenes_through_node_python_and_official_sdk(self):
        script = """
import assert from 'node:assert/strict';
import { createOpenRALEnvironment } from './src/environments/openral.mjs';
import { runEpisode } from './src/framework/runtime.mjs';
const cases = JSON.parse(process.env.REA_OPENRAL_SCENE_TEST_CASES);
for (const scenario of cases) {
  const {values, camera, ...options} = scenario;
  const environment = createOpenRALEnvironment(options);
  try {
    const initial = await environment.reset();
    assert.equal(environment.timing, 'turn-based');
    assert.equal(initial.observation.state.robot.sceneActionDimension, values.length);
    const result = await environment.step({
      name: 'scene_trajectory', arguments: {commands: [values, values, values]},
    });
    assert.equal(result.info.controlSteps, 3);
    assert.equal(result.info.completion, 'simulated');
    assert(Number.isFinite(result.reward));
    assert.notDeepEqual(result.observation.state.observation, initial.observation.state.observation);
    assert.notEqual(
      result.observation.images.find(image => image.name === camera).data,
      initial.observation.images.find(image => image.name === camera).data,
    );
    const policy = {decide: async () => ({action: {name: 'scene_action', arguments: {values}}})};
    for (const seed of [7, 8]) {
      const episode = await runEpisode({environment, policy, seed, maxSteps: 1});
      assert.equal(episode.steps, 1);
      assert.equal(episode.reason, 'max_steps');
      assert.equal(episode.error, null);
    }
    console.log(JSON.stringify({robotId: options.robotId, controlSteps: result.info.controlSteps, camera}));
  } finally {
    await environment.close();
  }
}
"""
        common = {"openralRoot": SDK_ROOT, "python": sys.executable}
        scenarios = [
            {**common, "robotId": "pusht_2d", "simEnvYaml": str(Path(SDK_ROOT) / "scenes/sim/pusht.yaml"), "values": [250, 250], "camera": "camera1"},
            {**common, "robotId": "sawyer", "simEnvYaml": str(ROOT / "examples/robots/openral/scenes/metaworld_push.yaml"), "values": [.1, 0, 0, 0], "camera": "front"},
        ]
        process = subprocess.run(
            [os.environ.get("OPENRAL_NODE", "node"), "--input-type=module", "-e", script],
            cwd=ROOT, env={**os.environ, "REA_OPENRAL_SCENE_TEST_CASES": json.dumps(scenarios)},
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        results = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual([result["robotId"] for result in results], ["pusht_2d", "sawyer"])


if __name__ == "__main__":
    unittest.main()
