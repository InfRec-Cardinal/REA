import base64
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "test" / "fixtures"))
from robot_drivers.rpent import PLATFORMS, RPentDriver
from rpent_rpc_server import RPentFixture


def first_pixel(image):
    raw = base64.b64decode(image["data"])
    offset = 8
    while offset < len(raw):
        length = struct.unpack("!I", raw[offset:offset + 4])[0]
        if raw[offset + 4:offset + 8] == b"IDAT":
            return list(zlib.decompress(raw[offset + 8:offset + 8 + length])[1:4])
        offset += length + 12


class RPentDriverTests(unittest.TestCase):
    def make_driver(self, robot="libero", **options):
        fixture = RPentFixture(robot)
        with patch("robot_drivers.rpent.make_client", return_value=fixture):
            driver = RPentDriver({"robot": robot, **options})
        self.addCleanup(driver.close)
        return driver, fixture

    def test_every_implemented_platform_describes_without_implicit_reset(self):
        expected_sizes = {"libero": [7], "robocasa": [12], "robotwin": [14, 16]}
        for robot, dimensions in expected_sizes.items():
            with self.subTest(robot=robot):
                driver, fixture = self.make_driver(robot)
                self.assertEqual(fixture.reset_count, 0)
                self.assertEqual([driver.control_specs[action["name"]][0] for action in driver.actions[::2]], dimensions)
                self.assertEqual(driver.timing, "turn-based")
                snapshot = driver.reset()
                self.assertEqual(fixture.reset_count, 1)
                self.assertFalse(snapshot["success"])
                self.assertEqual(snapshot["text"], f"Complete the {robot} fixture task")
                self.assertNotIn("main_images", snapshot["state"])
                self.assertNotIn("agentview_depth", snapshot["state"])
                self.assertEqual([image["name"] for image in snapshot["images"]], driver.camera_names)
                self.assertEqual(first_pixel(snapshot["images"][0]), [255, 0, 0] if robot == "robotwin" else [0, 0, 255])
                self.assertFalse(driver.stop()["supported"])
                json.dumps(snapshot, allow_nan=False)

    def test_preview_does_not_advance_any_platform(self):
        for robot in PLATFORMS:
            with self.subTest(robot=robot):
                driver, fixture = self.make_driver(robot)
                driver.reset()
                for action in driver.actions:
                    size, _, chunk = driver.control_specs[action["name"]]
                    arguments = {"commands": [[0] * size] * 2} if chunk else {"values": [0] * size}
                    result = driver.step({"name": action["name"], "arguments": arguments})
                    self.assertTrue(result["info"]["preview"])
                self.assertEqual(fixture.steps, 0)
                self.assertEqual(fixture.reset_count, 1)

    def test_libero_native_chunk_preserves_flags_and_actual_step_count(self):
        driver, fixture = self.make_driver()
        driver.reset()
        result = driver.step({"name": "rpent_trajectory", "arguments": {"commands": [[0] * 7] * 5}}, execute=True)
        self.assertTrue(result["terminated"])
        self.assertTrue(result["success"])
        self.assertFalse(result["truncated"])
        self.assertEqual(result["reward"], 3)
        self.assertEqual(result["info"]["controlSteps"], 5)
        self.assertEqual(result["info"]["nativeSignals"][0]["terminated"], [False, False, True, True, True])
        self.assertEqual(len(result["info"]["native"]), 5)
        self.assertEqual([call[0] for call in fixture.calls].count("env.chunk_step"), 1)

    def test_robocasa_steps_four_tuple_and_stops_chunk_on_success(self):
        driver, fixture = self.make_driver("robocasa")
        driver.reset(task="OpenDrawer")
        result = driver.step({"name": "rpent_trajectory", "arguments": {"commands": [[0] * 12] * 5}}, execute=True)
        self.assertTrue(result["success"])
        self.assertTrue(result["terminated"])
        self.assertEqual(result["info"]["controlSteps"], 3)
        self.assertEqual(result["reward"], 1)
        self.assertNotIn("env.chunk_step", [call[0] for call in fixture.calls])
        self.assertEqual([call[0] for call in fixture.calls].count("env.step"), 3)

    def test_robotwin_both_representations_and_short_native_chunk(self):
        driver, fixture = self.make_driver("robotwin")
        driver.reset(task={"task_name": "adjust_bottle", "task_config": "demo_clean"})
        single = driver.step({"name": "rpent_qpos", "arguments": {"values": [0] * 14}}, execute=True)
        self.assertFalse(single["terminated"])
        result = driver.step({"name": "rpent_ee_trajectory", "arguments": {"commands": [[0] * 16] * 5}}, execute=True)
        self.assertEqual(result["info"]["controlSteps"], 2)
        self.assertEqual(result["info"]["episodeControlSteps"], 3)
        self.assertEqual(result["state"]["episode_status"]["take_action_cnt"], 3)
        self.assertIn("arm_qpos_real12", result["state"]["robot_state"])
        calls = [call for call in fixture.calls if call[0] in ("env.step", "env.chunk_step")]
        self.assertEqual(calls[0][2], {"action_type": "qpos"})
        self.assertEqual(calls[1][2], {"action_type": "ee", "return_all_frames": False})

    def test_fixed_endpoint_task_seed_and_episode_lifecycle(self):
        driver, fixture = self.make_driver("libero")
        with self.assertRaisesRegex(ValueError, "cannot change seed"):
            driver.reset(seed=1)
        with self.assertRaisesRegex(ValueError, "cannot change task"):
            driver.reset(task=4)
        self.assertEqual(fixture.reset_count, 0)
        driver.reset(task={"task": 0, "suite": "libero_spatial"})
        driver.step({"name": "rpent_trajectory", "arguments": {"commands": [[0] * 7] * 3}}, execute=True)
        with self.assertRaisesRegex(RuntimeError, "episode ended"):
            driver.step({"name": "rpent_action", "arguments": {"values": [0] * 7}}, execute=True)
        driver.close()
        self.assertTrue(fixture.closed)
        self.assertNotIn("shutdown", [call[0] for call in fixture.calls])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            driver.reset()

    def test_only_explicit_server_ownership_requests_shutdown(self):
        driver, fixture = self.make_driver(shutdownOnClose=True)
        driver.close()
        driver.close()
        self.assertEqual([call[0] for call in fixture.calls].count("shutdown"), 1)

    def test_factory_extension_and_native_action_validation(self):
        driver = RPentDriver({"robot": "robotwin", "clientFactory": "rpent_rpc_server:make_client", "factoryOptions": {"robot": "robotwin"}})
        self.addCleanup(driver.close)
        driver.reset()
        with self.assertRaisesRegex(ValueError, "16 native values"):
            driver.step({"name": "rpent_ee", "arguments": {"values": [0] * 14}}, execute=True)
        self.assertEqual(driver.client.steps, 0)
        with self.assertRaisesRegex(ValueError, "not implemented upstream"):
            RPentDriver({"robot": "franka"})


@unittest.skipUnless(os.environ.get("RPENT_ROOT"), "Set RPENT_ROOT to run real upstream HTTP/socket loopback tests")
class RPentUpstreamTransportTests(unittest.TestCase):
    def test_all_platforms_round_trip_official_http_and_socket_transports(self):
        sys.path.insert(0, os.environ["RPENT_ROOT"])
        from rpent.utils.rpc.http_rpc import HttpRpcServer
        from rpent.utils.rpc.socket_rpc import SocketRpcServer
        from rpent.robots.base import enumerate_robots
        self.assertEqual(set(enumerate_robots()), set(PLATFORMS))
        for transport, server_type in (("http", HttpRpcServer), ("socket", SocketRpcServer)):
            for robot in PLATFORMS:
                with self.subTest(transport=transport, robot=robot):
                    fixture = RPentFixture(robot, arrays=True)
                    server = server_type(("127.0.0.1", 0), lambda method, args, kwargs: fixture.call(method, args, kwargs))
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    driver = RPentDriver({"robot": robot, "endpoint": f"{transport}://127.0.0.1:{server.server_address[1]}"})
                    try:
                        self.assertEqual(fixture.reset_count, 0)
                        driver.reset()
                        action = driver.actions[-1]
                        size = driver.control_specs[action["name"]][0]
                        result = driver.step({"name": action["name"], "arguments": {"commands": [[0] * size] * 5}}, execute=True)
                        self.assertTrue(result["success"])
                        self.assertEqual(result["info"]["controlSteps"], 5 if robot == "libero" else 3)
                        self.assertNotIn("main_images", result["state"])
                        self.assertEqual(first_pixel(result["images"][0]), [255, 0, 0] if robot == "robotwin" else [0, 0, 255])
                        json.dumps(result, allow_nan=False)
                        node_script = """
import assert from 'node:assert/strict';
import {createRPentEnvironment} from './src/environments/rpent.mjs';
const options = JSON.parse(process.env.REA_RPENT_TEST_OPTIONS);
const environment = createRPentEnvironment({...options, execute: true});
try {
  const initial = await environment.reset({seed: 0});
  assert.equal(initial.success, false);
  const spec = initial.observation.actions.at(-1);
  const size = spec.parameters.properties.commands.items.maxItems;
  const result = await environment.step({name: spec.name, arguments: {commands: Array.from({length: 5}, () => Array(size).fill(0))}});
  assert.equal(result.success, true);
  assert.equal(result.terminated, true);
  assert.equal(result.info.controlSteps, options.robot === 'libero' ? 5 : 3);
  assert.equal(result.info.requestedControlSteps, 5);
  assert.equal('main_images' in result.observation.state, false);
  assert(result.observation.images.every(image => image.name && image.mimeType === 'image/png'));
} finally {
  await environment.close();
}
"""
                        node_options = {"robot": robot, "endpoint": f"{transport}://127.0.0.1:{server.server_address[1]}", "python": sys.executable, "rpentRoot": os.environ["RPENT_ROOT"]}
                        bridge = subprocess.run(
                            [os.environ.get("RPENT_NODE", "node"), "--input-type=module", "-e", node_script],
                            cwd=ROOT, env={**os.environ, "REA_RPENT_TEST_OPTIONS": json.dumps(node_options)},
                            capture_output=True, text=True, timeout=20,
                        )
                        self.assertEqual(bridge.returncode, 0, bridge.stderr)
                        self.assertEqual(fixture.reset_count, 2)
                    finally:
                        driver.close()
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)


@unittest.skipUnless(os.environ.get("REA_RPENT_PHYSICS"), "Set REA_RPENT_PHYSICS=libero or libero,robocasa with the upstream simulator dependencies/assets installed")
class RPentPhysicsTests(unittest.TestCase):
    def test_native_physics_through_node_python_and_official_rpc(self):
        for platform in os.environ["REA_RPENT_PHYSICS"].split(","):
            with self.subTest(platform=platform):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "test" / "fixtures" / "rpent_physics_smoke.py"), platform],
                    cwd=ROOT, capture_output=True, text=True, timeout=240,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertEqual(report["controlSteps"], 1)
                self.assertEqual(report["trajectoryControlSteps"], 2)
                self.assertEqual(report["bridge"], "node-python")
                self.assertTrue(report["images"])


if __name__ == "__main__":
    unittest.main()
