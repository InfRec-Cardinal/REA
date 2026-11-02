import base64
import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from robot_drivers.lerobot import LeRobotDriver


class FakeRobot:
    def __init__(self, keys=("shoulder.pan", "gripper.pos"), cameras=("front",)):
        self.action_features = {key: float for key in keys}
        self.observation_features = {**self.action_features, **{name: (1, 2, 3) for name in cameras}}
        self.values = {key: 0.0 for key in keys}
        self.cameras = {name: [[[10, 20, 30], [40, 50, 60]]] for name in cameras}
        self.is_connected = False
        self.connect_calls = []
        self.events = []

    def connect(self, calibrate=True):
        self.connect_calls.append(calibrate)
        self.is_connected = True

    def get_observation(self):
        self.events.append("observe")
        return {**self.values, **self.cameras}

    def send_action(self, action):
        self.events.append(dict(action))
        self.values = {key: min(10.0, max(-10.0, value)) for key, value in action.items()}
        return dict(self.values)

    def disconnect(self):
        self.events.append("disconnect")
        self.is_connected = False


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class TimedRobot(FakeRobot):
    def __init__(self, clock, send_durations=()):
        super().__init__()
        self.clock = clock
        self.send_durations = iter(send_durations)
        self.send_times = []
        self.observation_times = []

    def send_action(self, action):
        self.send_times.append(self.clock.now)
        self.clock.now += next(self.send_durations, 0)
        return super().send_action(action)

    def get_observation(self):
        self.observation_times.append(self.clock.now)
        return super().get_observation()


class LeRobotDriverTests(unittest.TestCase):
    def make_driver(self, robot=None, **options):
        robot = robot or FakeRobot()
        factory_module = types.ModuleType("fast_test_robot_factory")
        received = []

        def factory(factory_options):
            received.append(factory_options)
            return robot

        factory_module.create_robot = factory
        with patch.dict(sys.modules, {factory_module.__name__: factory_module}):
            driver = LeRobotDriver({
                "robotFactory": "fast_test_robot_factory:create_robot",
                "factoryOptions": {"port": "fake"},
                **options,
            })
        self.assertEqual(received, [{"port": "fake"}])
        return driver, robot

    def test_single_arm_schema_and_reset_observes_without_calibrating(self):
        driver, robot = self.make_driver(units={"shoulder.pan": "degrees"})
        self.assertEqual(robot.connect_calls, [])
        schema = driver.actions[0]["parameters"]["properties"]["values"]
        self.assertEqual(schema["required"], ["shoulder.pan", "gripper.pos"])
        self.assertEqual(schema["properties"]["shoulder.pan"]["type"], "number")
        self.assertIn("degrees", schema["properties"]["shoulder.pan"]["description"])
        self.assertIn("not inferred", schema["properties"]["gripper.pos"]["description"])
        snapshot = driver.reset(task="reach", seed=7)
        self.assertEqual(robot.connect_calls, [False])
        self.assertEqual(robot.events, ["observe"])
        self.assertEqual(snapshot["info"]["reset"], "observe_only")
        self.assertFalse(snapshot["info"]["seedApplied"])
        self.assertIsNone(snapshot["success"])
        self.assertFalse(snapshot["terminated"])
        driver.close()
        driver.close()
        self.assertEqual(robot.events[-1], "disconnect")
        self.assertEqual(robot.events.count("disconnect"), 1)

    def test_dual_arm_native_names_and_named_cameras_leave_state(self):
        keys = ("left_shoulder.pan", "left_gripper.pos", "right_shoulder.pan", "right_gripper.pos")
        driver, robot = self.make_driver(FakeRobot(keys, ("left_wrist", "right_wrist", "overhead")))
        self.assertEqual(driver.metadata["actionKeys"], list(keys))
        self.assertEqual(driver.metadata["cameraNames"], ["left_wrist", "right_wrist", "overhead"])
        snapshot = driver.observe()
        self.assertEqual(list(snapshot["state"]), list(keys))
        self.assertEqual([image["name"] for image in snapshot["images"]], driver.metadata["cameraNames"])
        for image in snapshot["images"]:
            self.assertEqual(image["mimeType"], "image/png")
            self.assertTrue(base64.b64decode(image["data"]).startswith(b"\x89PNG\r\n\x1a\n"))
        json.dumps(snapshot, allow_nan=False)
        values = {key: 1.0 for key in keys}
        result = driver.step({"name": "robot_command", "arguments": {"values": values}}, execute=True)
        self.assertEqual(result["info"]["actualActions"], [values])

    def test_preview_never_sends_or_sleeps_and_captures_once(self):
        driver, robot = self.make_driver()
        command = {"shoulder.pan": 5.0, "gripper.pos": 2.0}
        with patch("robot_drivers.lerobot.time.sleep") as sleep:
            snapshot = driver.step({"name": "robot_trajectory", "arguments": {"commands": [command, command]}})
        sleep.assert_not_called()
        self.assertEqual(robot.events, ["observe"])
        self.assertEqual(snapshot["state"], {"shoulder.pan": 0.0, "gripper.pos": 0.0})
        self.assertEqual(snapshot["info"]["requestedActions"], [command, command])
        self.assertEqual(snapshot["info"]["actualActions"], [])
        self.assertEqual(snapshot["info"]["controlSteps"], 0)
        self.assertEqual(snapshot["info"]["requestedControlSteps"], 2)
        self.assertTrue(snapshot["info"]["preview"])

    def test_depth_pixels_are_excluded_from_rgb_images_and_state(self):
        robot = FakeRobot()
        robot.observation_features["front_depth"] = (1, 2, 1)
        robot.cameras["front_depth"] = [[[420], [430]]]
        driver, robot = self.make_driver(robot)
        snapshot = driver.reset()
        self.assertEqual(driver.metadata["omittedImageFields"], ["front_depth"])
        self.assertEqual([image["name"] for image in snapshot["images"]], ["front"])
        self.assertNotIn("front_depth", snapshot["state"])

    def test_chunk_order_cadence_and_clamped_receipts(self):
        driver, robot = self.make_driver(controlHz=20)
        commands = [
            {"shoulder.pan": 2.0, "gripper.pos": 3.0},
            {"shoulder.pan": 20.0, "gripper.pos": 4.0},
            {"shoulder.pan": -20.0, "gripper.pos": 5.0},
        ]
        clock = FakeClock()
        with patch("robot_drivers.lerobot.time.monotonic", clock.monotonic), patch("robot_drivers.lerobot.time.sleep", clock.sleep):
            snapshot = driver.step({"name": "robot_trajectory", "arguments": {"commands": commands}}, execute=True)
        self.assertEqual(robot.events, [*commands, "observe"])
        self.assertEqual(len(clock.sleeps), 3)
        for duration in clock.sleeps:
            self.assertAlmostEqual(duration, 0.05)
        self.assertEqual(snapshot["info"]["actualActions"], [
            commands[0], {"shoulder.pan": 10.0, "gripper.pos": 4.0}, {"shoulder.pan": -10.0, "gripper.pos": 5.0},
        ])
        self.assertEqual(snapshot["info"]["controlSteps"], 3)
        self.assertEqual(snapshot["info"]["controlPeriodMs"], 50)
        self.assertAlmostEqual(snapshot["info"]["controlElapsedMs"], 150)
        self.assertEqual(snapshot["info"]["completion"], "submitted")
        self.assertIsNone(snapshot["success"])
        self.assertIsNone(snapshot["reward"])

    def test_single_commands_wait_a_full_period_before_observation(self):
        clock = FakeClock()
        driver, robot = self.make_driver(TimedRobot(clock), controlHz=20)
        action = {"name": "robot_command", "arguments": {"values": {"shoulder.pan": 1.0, "gripper.pos": 2.0}}}
        with patch("robot_drivers.lerobot.time.monotonic", clock.monotonic), patch("robot_drivers.lerobot.time.sleep", clock.sleep):
            driver.step(action, execute=True)
            driver.step(action, execute=True)
        self.assertEqual(robot.send_times, [0, 0.05])
        self.assertEqual(robot.observation_times, [0.05, 0.1])
        self.assertEqual(clock.sleeps, [0.05, 0.05])

    def test_slow_sdk_does_not_trigger_catch_up_commands(self):
        clock = FakeClock()
        driver, robot = self.make_driver(TimedRobot(clock, send_durations=[0.13, 0, 0]), controlHz=20)
        command = {"shoulder.pan": 1.0, "gripper.pos": 2.0}
        with patch("robot_drivers.lerobot.time.monotonic", clock.monotonic), patch("robot_drivers.lerobot.time.sleep", clock.sleep):
            snapshot = driver.step({"name": "robot_trajectory", "arguments": {"commands": [command] * 3}}, execute=True)
        self.assertEqual(len(robot.send_times), 3)
        for sent, expected in zip(robot.send_times, [0, 0.18, 0.23]):
            self.assertAlmostEqual(sent, expected)
        self.assertAlmostEqual(robot.observation_times[0], 0.28)
        self.assertAlmostEqual(snapshot["info"]["controlElapsedMs"], 280)
        self.assertEqual(snapshot["info"]["controlSteps"], 3)

    def test_stop_does_not_invent_a_motion_and_factory_hook_is_used(self):
        driver, robot = self.make_driver()
        self.assertFalse(driver.stop()["supported"])
        self.assertFalse(driver.metadata["stopSupported"])
        self.assertEqual(robot.events, [])
        robot.stop = lambda: {"native": "stopped"}
        driver, robot = self.make_driver(robot)
        self.assertEqual(driver.stop(), {"supported": True, "invoked": False})
        driver.observe()
        self.assertEqual(driver.stop(), {"supported": True, "invoked": True, "result": {"native": "stopped"}})
        self.assertEqual(robot.events, ["observe"])

    def test_commands_follow_discovered_features_and_chunk_limit(self):
        driver, robot = self.make_driver()
        with self.assertRaisesRegex(ValueError, "native action_features"):
            driver.step({"name": "robot_command", "arguments": {"values": {"unknown.pos": 1}}}, execute=True)
        with self.assertRaisesRegex(ValueError, "1 to 100"):
            driver.step({"name": "robot_trajectory", "arguments": {"commands": []}}, execute=True)
        self.assertEqual(robot.connect_calls, [])
        self.assertEqual(robot.events, [])

    def test_config_registration_decode_and_official_factory_without_sdk_install(self):
        robot = FakeRobot()
        imports = []
        decoded_config = object()
        robot_config = {"type": "so101_follower", "port": "fake", "cameras": {}}
        registry = {}

        class RobotConfig:
            @classmethod
            def get_known_choices(cls):
                return registry

        def decode(config_type, value):
            self.assertIs(config_type, RobotConfig)
            self.assertEqual(value, robot_config)
            self.assertEqual(imports, ["lerobot.cameras.opencv", "lerobot.robots.so_follower"])
            return decoded_config

        def make_robot_from_config(config):
            self.assertIs(config, decoded_config)
            return robot

        robots_module = types.ModuleType("lerobot.robots")
        robots_module.RobotConfig = RobotConfig
        robots_module.make_robot_from_config = make_robot_from_config
        draccus_module = types.ModuleType("draccus")
        draccus_module.decode = decode
        real_import = importlib.import_module

        def import_module(name, package=None):
            if name in ("lerobot.cameras.opencv", "lerobot.robots.so_follower"):
                imports.append(name)
                return types.ModuleType(name)
            return real_import(name, package)

        with patch.dict(sys.modules, {"lerobot.robots": robots_module, "draccus": draccus_module}), patch("robot_drivers.lerobot.importlib.import_module", import_module):
            driver = LeRobotDriver({"robotConfig": robot_config, "registrationModules": ["lerobot.cameras.opencv"]})
        self.assertIs(driver.robot, robot)
        self.assertEqual(robot.connect_calls, [])
        self.assertEqual(driver.metadata["robotType"], "so101_follower")


if __name__ == "__main__":
    unittest.main()
