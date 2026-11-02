import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from robot_drivers.openral import OpenRALDriver, _load_sdk, list_openral_robots


class NativeAction:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class StopRequested(Exception):
    pass


def description(name="test_robot", modes=("joint_position", "gripper_position")):
    return SimpleNamespace(name=name, joints=[SimpleNamespace(name=f"joint{i}", joint_type="revolute", position_limits=(-2, 2), role="arm") for i in range(2)],
                           capabilities=SimpleNamespace(supported_control_modes=modes), hal=SimpleNamespace(real=None))


class FakeHAL:
    def __init__(self):
        self.description = description()
        self.positions = [0, 0]
        self.sent = []
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.image_calls = 0
        self.success = None

    def connect(self):
        self.connect_calls += 1

    def disconnect(self):
        self.disconnect_calls += 1

    def read_state(self):
        return SimpleNamespace(name=["joint0", "joint1"], position=self.positions, velocity=[], effort=[], stamp_ns=100)

    def read_images(self):
        self.image_calls += 1
        return {"top": [[[10, 20, 30]]]}

    def send_action(self, action):
        self.sent.append(action)
        if hasattr(action, "joint_targets"):
            self.positions = action.joint_targets[-1]

    def task_success(self):
        return self.success

    def estop(self):
        raise StopRequested("native stop forwarded")


class SimAttachedHAL(FakeHAL):
    pass


def command(values=(.1, .2), mode="joint_position"):
    return {"controlMode": mode, "values": list(values)}


class OpenRALDriverTests(unittest.TestCase):
    def driver(self, hal=None, **options):
        hal = hal or FakeHAL()
        sdk = SimpleNamespace(Action=NativeAction, ROSEStopRequested=StopRequested)
        driver = OpenRALDriver({"_hal": hal, "_sdk": sdk, "executeEnabled": True, **options})
        self.addCleanup(driver.close)
        return driver, hal

    def test_reset_observes_without_homing_or_reconnecting(self):
        driver, hal = self.driver()
        first = driver.reset(seed=5)
        driver.reset(seed=6)
        self.assertEqual(hal.connect_calls, 1)
        self.assertEqual(hal.sent, [])
        self.assertFalse(first["info"]["seedApplied"])
        self.assertEqual(first["state"]["stampNs"], 100)

    def test_preview_does_not_submit(self):
        driver, hal = self.driver(executeEnabled=False)
        result = driver.step({"name": "hal_action", "arguments": command()})
        self.assertFalse(result["info"]["executed"])
        self.assertEqual(hal.sent, [])
        with self.assertRaisesRegex(ValueError, "preview"):
            driver.step({"name": "hal_action", "arguments": command()}, execute=True)

    def test_sequence_preserves_each_waypoint_as_one_sdk_step(self):
        driver, hal = self.driver()
        driver.reset()
        hal.image_calls = 0
        result = driver.step({"name": "hal_sequence", "arguments": {"commands": [command((.1, .2)), command((.3, .4))]}}, execute=True)
        self.assertEqual([a.joint_targets for a in hal.sent], [[[.1, .2]], [[.3, .4]]])
        self.assertEqual([a.horizon for a in hal.sent], [1, 1])
        self.assertEqual(hal.image_calls, 1)
        self.assertEqual(result["info"]["controlSteps"], 2)
        self.assertEqual(result["state"]["position"], [.3, .4])

    def test_whole_sequence_is_parsed_before_submission(self):
        driver, hal = self.driver()
        with self.assertRaisesRegex(ValueError, "finite"):
            driver.step({"name": "hal_sequence", "arguments": {"commands": [command(), command((float('nan'), 0))]}}, execute=True)
        self.assertEqual(hal.sent, [])

    def test_modes_follow_implementation_not_overbroad_manifest(self):
        driver, hal = self.driver()
        self.assertEqual(driver.control_modes, ["joint_position"])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            driver.step({"name": "hal_action", "arguments": command((.5,), "gripper_position")}, execute=True)

    def test_sdk_success_is_not_terminal_or_reward(self):
        driver, hal = self.driver()
        hal.success = True
        result = driver.observe()
        self.assertTrue(result["success"])
        self.assertFalse(result["terminated"])
        self.assertIsNone(result["reward"])

    def test_atomic_group_uses_shared_sdk_tick(self):
        driver, hal = self.driver(SimAttachedHAL())
        result = driver.step({"name": "hal_group", "arguments": {"commands": [command(), command((.5,), "gripper_position")]}}, execute=True)
        self.assertEqual([a.tick_index for a in hal.sent], [1, 1])
        self.assertEqual([a.tick_group_size for a in hal.sent], [2, 2])
        self.assertEqual(result["info"]["controlSteps"], 1)
        self.assertEqual(result["info"]["submittedActions"], 2)

    def test_expected_estop_exception_is_returned_explicitly(self):
        driver, hal = self.driver()
        driver.reset()
        result = driver.stop()
        self.assertEqual(result["sdkException"], "StopRequested")
        with self.assertRaisesRegex(RuntimeError, "estop"):
            driver.step({"name": "hal_action", "arguments": command()}, execute=True)

    def test_close_releases_scene_environment(self):
        hal = SimAttachedHAL()
        closes = []
        hal.env = SimpleNamespace(close=lambda: closes.append(True))
        driver, hal = self.driver(hal)
        driver.close()
        driver.close()
        self.assertEqual(hal.disconnect_calls, 1)
        self.assertEqual(closes, [True])

    def test_callback_real_hal_requires_explicit_transport_factory(self):
        desc = description()
        desc.hal.real = "openral_hal.franka_panda_real:FrankaPandaRealHAL"
        sdk = SimpleNamespace(RobotDescription=SimpleNamespace(from_yaml=lambda path: desc))
        with self.assertRaisesRegex(ValueError, "halFactory"):
            OpenRALDriver({"_sdk": sdk, "mode": "real", "robotYaml": "fixture.yaml", "transport": {"fci_ip": "example.invalid"}})


SDK_ROOT = os.environ.get("OPENRAL_TEST_ROOT")


@unittest.skipUnless(SDK_ROOT, "Set OPENRAL_TEST_ROOT in the isolated OpenRAL SDK environment")
class OpenRALSDKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sdk = _load_sdk(SDK_ROOT)
        cls.catalog = list_openral_robots(SDK_ROOT)

    def test_all_current_manifests_are_discovered_and_classified(self):
        self.assertEqual(len(self.catalog), 21)
        self.assertEqual(sum(item["bareSimKind"] == "mujoco" for item in self.catalog), 11)
        self.assertEqual(sum(item["bareSimKind"] == "kinematic" for item in self.catalog), 2)
        self.assertEqual(sum(bool(item["realEntrypoint"]) for item in self.catalog), 9)

    def test_all_manifests_round_trip_through_real_sim_attached_hal_with_mock_rollout(self):
        import numpy as np
        from openral_hal.sim_attached import SimAttachedHAL as SDKSimAttachedHAL

        class Rollout:
            def __init__(self, n):
                self.action_dim = n
                self.n = n
                self.steps = 0
                self.resets = 0
                self.closed = False

            def reset(self, seed=None):
                self.resets += 1
                return {"joint_positions": np.zeros(self.n), "images": {"top": np.zeros((2, 2, 3), dtype=np.uint8)}}

            def step(self, values):
                self.steps += 1
                return SimpleNamespace(observation={"joint_positions": np.full(self.n, .1), "images": {}}, reward=7, done=True)

            def task_success(self):
                return self.steps > 0

            def close(self):
                self.closed = True

        for item in self.catalog:
            with self.subTest(robot=item["robotId"]):
                desc = self.sdk.RobotDescription.from_yaml(item["robotYaml"])
                env = Rollout(len(desc.joints))
                hal = SDKSimAttachedHAL(env, desc, env_reset_seed=12)
                driver = OpenRALDriver({"_sdk": self.sdk, "_hal": hal, "robotId": item["robotId"], "executeEnabled": True, "cameraNames": []})
                try:
                    driver.reset()
                    joint_values = [0] * sum(joint.role != "gripper" for joint in desc.joints)
                    driver.step({"name": "hal_action", "arguments": command(joint_values)})
                    self.assertEqual(env.steps, 0)
                    result = driver.step({"name": "hal_action", "arguments": command(joint_values)}, execute=True)
                    self.assertEqual(env.steps, 1)
                    self.assertEqual(env.resets, 1)
                    self.assertTrue(result["success"])
                    self.assertFalse(result["terminated"])
                    self.assertIsNone(result["reward"])
                finally:
                    driver.close()
                self.assertTrue(env.closed)

    def test_real_single_controller_sdk_adapters_with_upstream_sim_transport(self):
        from openral_hal.sim_transport import SimTransport
        for robot_id in ("franka_panda", "ur5e", "ur10e", "sawyer"):
            with self.subTest(robot=robot_id):
                desc = self.sdk.RobotDescription.from_yaml(str(Path(SDK_ROOT) / "robots" / robot_id / "robot.yaml"))
                transport = SimTransport(len(desc.joints))
                hal = self.sdk.build_hal(desc, mode="real", transport={"publish_fn": transport.publish, "state_fn": transport.state})
                driver = OpenRALDriver({"_sdk": self.sdk, "_hal": hal, "mode": "real", "executeEnabled": True, "controlHz": 1000, "cameraNames": []})
                try:
                    driver.reset()
                    target = [.1] * len(desc.joints)
                    result = driver.step({"name": "hal_action", "arguments": command(target)}, execute=True)
                    self.assertEqual(result["state"]["position"], target)
                    self.assertEqual(transport.call_count, 1)
                finally:
                    driver.close()

    def test_so100_and_so101_use_actual_sdk_unit_conversion_with_digital_twin(self):
        from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig
        for robot_id in ("so100_follower", "so101_follower"):
            with self.subTest(robot=robot_id):
                desc = self.sdk.RobotDescription.from_yaml(str(Path(SDK_ROOT) / "robots" / robot_id / "robot.yaml"))
                twin = SO100DigitalTwin(SO100DigitalTwinConfig())
                hal = self.sdk.build_hal(desc, mode="real", transport={"robot": twin, "calibrate_on_connect": False})
                driver = OpenRALDriver({"_sdk": self.sdk, "_hal": hal, "robotId": robot_id, "mode": "real", "executeEnabled": True, "controlHz": 1000, "cameraNames": []})
                try:
                    driver.reset()
                    target = [.1, .2, .3, .4, .5, .6]
                    result = driver.step({"name": "hal_action", "arguments": command(target)}, execute=True)
                    for actual, expected in zip(result["state"]["position"], target):
                        self.assertAlmostEqual(actual, expected)
                    self.assertEqual(driver.metadata["robotId"], robot_id)
                finally:
                    driver.close()

    def test_real_bimanual_sdk_adapters_publish_all_four_controller_commands(self):
        for robot_id in ("aloha_bimanual", "openarm"):
            with self.subTest(robot=robot_id):
                desc = self.sdk.RobotDescription.from_yaml(str(Path(SDK_ROOT) / "robots" / robot_id / "robot.yaml"))
                n = len(desc.joints)
                calls = []
                state = {"position": [0.0] * n, "velocity": [0.0] * n, "effort": [0.0] * n}
                hal = self.sdk.build_hal(desc, mode="real", transport={"publish_fn": lambda topic, msg: calls.append((topic, msg)), "state_fn": lambda: state, "require_can_links": False})
                driver = OpenRALDriver({"_sdk": self.sdk, "_hal": hal, "mode": "real", "executeEnabled": True, "controlHz": 1000, "cameraNames": []})
                try:
                    driver.reset()
                    driver.step({"name": "hal_action", "arguments": command([0.0] * n)})
                    self.assertEqual(calls, [])
                    driver.step({"name": "hal_action", "arguments": command([0.0] * n)}, execute=True)
                    self.assertEqual(len(calls), 4)
                    self.assertEqual(len({topic for topic, _ in calls}), 4)
                finally:
                    driver.close()

    def test_galaxea_sdk_uses_mock_sidecar_for_joint_and_gripper_commands(self):
        from openral_hal.galaxea_a1 import GalaxeaA1HAL

        class Transport:
            def __init__(self):
                self.messages = []
                self.connected = False

            def connect(self, hello, timeout_s):
                self.connected = True

            def snapshot(self):
                return SimpleNamespace(position=(0.0, 1.0, -1.0, 0.0, 0.0, 0.0), velocity=(0.0,) * 6, effort=(0.0,) * 6,
                                       stamp_ns=time.time_ns(), received_monotonic=time.monotonic(), status_codes=(0,) * 7,
                                       status_received_monotonic=time.monotonic(), gripper_norm=.5)

            def send(self, message):
                self.messages.append(message)

            def close(self):
                self.connected = False

            def estop(self, timeout_s):
                self.close()

        transport = Transport()
        hal = GalaxeaA1HAL(transport=transport)
        driver = OpenRALDriver({"_sdk": self.sdk, "_hal": hal, "mode": "real", "executeEnabled": True, "controlHz": 1000, "cameraNames": []})
        try:
            first = driver.reset()
            driver.step({"name": "hal_sequence", "arguments": {"commands": [command(first["state"]["position"]), command((.5,), "gripper_position")]}}, execute=True)
            self.assertEqual(len(transport.messages), 2)
            self.assertEqual(transport.messages[0]["joint_targets"], first["state"]["position"])
            self.assertEqual(transport.messages[1]["gripper"], .5)
        finally:
            driver.close()
        self.assertFalse(transport.connected)

    def test_openarm_real_atomic_compact_slots_reach_all_controllers(self):
        desc = self.sdk.RobotDescription.from_yaml(str(Path(SDK_ROOT) / "robots" / "openarm" / "robot.yaml"))
        calls = []
        state = {"position": [0.0] * 16, "velocity": [0.0] * 16, "effort": [0.0] * 16}
        hal = self.sdk.build_hal(desc, mode="real", transport={"publish_fn": lambda topic, msg: calls.append((topic, msg)), "state_fn": lambda: state, "require_can_links": False})
        driver = OpenRALDriver({"_sdk": self.sdk, "_hal": hal, "mode": "real", "executeEnabled": True, "cameraNames": []})
        try:
            driver.reset()
            names = [joint.name for joint in desc.joints]
            group = [{**command([.1] * 7), "jointNames": names[:7]}, {**command((.2,), "gripper_position"), "eeName": names[7]},
                     {**command([.3] * 7), "jointNames": names[8:15]}, {**command((-.2,), "gripper_position"), "eeName": names[15]}]
            result = driver.step({"name": "hal_group", "arguments": {"commands": group}}, execute=True)
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[0][1]["joint_targets"], [[.1] * 7])
            self.assertEqual(calls[2][1]["joint_targets"], [[.3] * 7])
            self.assertEqual(result["info"]["controlSteps"], 1)
        finally:
            driver.close()

    def scene_driver(self, robot_id, rollout):
        with patch("openral_hal.sim_bringup.build_sim_env_from_yaml", return_value=(rollout, 9)):
            driver = OpenRALDriver({"openralRoot": SDK_ROOT, "robotId": robot_id, "simEnvYaml": "fixture.yaml", "executeEnabled": True})
        self.addCleanup(driver.close)
        return driver

    def test_gr1_official_wrapper_preserves_native_29_value_dict_order(self):
        import numpy as np
        from gymnasium.spaces import Box, Dict
        from openral_sim.backends.robocasa import _RoboCasaSim

        groups = (("waist", 3), ("right_arm", 7), ("left_arm", 7), ("right_hand", 6), ("left_hand", 6))

        class GymEnv:
            def __init__(self):
                self.action_space = Dict({"action." + name: Box(-float(i + 1), float(i + 1), (width,), dtype=np.float32) for i, (name, width) in enumerate(groups)})
                self.actions = []
                self.seeds = []

            def reset(self, seed=None):
                self.seeds.append(seed)
                return {}, {}

            def step(self, action):
                self.actions.append(action)
                return {}, 0.25, False, False, {}

            def _check_success(self):
                return False

            def close(self):
                pass

        native = GymEnv()
        rollout = _RoboCasaSim(scene=SimpleNamespace(id="robocasa/gr1/test"), task=SimpleNamespace(instruction="Put the cup in the drawer", success_key="is_success"),
                              _env=native, _camera_keys=(), _state_layout="gr1", _is_gymnasium_wrapped=True)
        rollout._wrap_obs = lambda raw: {"state": np.zeros(39), "images": {}}
        driver = self.scene_driver("gr1", rollout)
        first = driver.reset(seed=17)
        self.assertEqual(native.seeds, [17])
        self.assertEqual(first["text"], "Put the cup in the drawer")
        self.assertEqual(driver.metadata["sceneActionDimension"], 29)
        self.assertEqual([item["name"] for item in driver.metadata["sceneActionLayout"]], ["action." + name for name, _ in groups])
        bounds = driver.actions[0]["parameters"]["properties"]["values"]["prefixItems"]
        expected_bounds = [float(i + 1) for i, (_, width) in enumerate(groups) for _ in range(width)]
        self.assertEqual([item["maximum"] for item in bounds], expected_bounds)
        values = [i / 30 for i in range(29)]
        driver.step({"name": "scene_action", "arguments": {"values": values}}, execute=True)
        offset = 0
        for name, width in groups:
            np.testing.assert_allclose(native.actions[0]["action." + name], values[offset:offset + width], rtol=1e-6)
            offset += width
        driver.stop()
        reset = driver.reset(seed=18)
        self.assertEqual(native.seeds, [17, 18])
        self.assertEqual(reset["info"]["totalControlSteps"], 0)
        self.assertFalse(driver.hal._estop_latched)
        driver.step({"name": "scene_action", "arguments": {"values": values}}, execute=True)
        self.assertEqual(len(native.actions), 2)

    def test_aloha_official_wrapper_accepts_radians_outside_nominal_gym_box(self):
        import numpy as np
        from gymnasium.spaces import Box
        from openral_sim.backends.aloha import _AlohaSim

        class GymEnv:
            action_space = Box(-1, 1, (14,), dtype=np.float32)

            def __init__(self):
                self.actions = []

            def reset(self, seed=None):
                return {"agent_pos": np.zeros(14)}, {}

            def step(self, action):
                self.actions.append(action)
                return {"agent_pos": action}, 1, False, False, {"is_success": False}

            def close(self):
                pass

        native = GymEnv()
        rollout = _AlohaSim(scene=SimpleNamespace(id="aloha_transfer_cube"), task=SimpleNamespace(instruction="Transfer the cube"), _env=native)
        driver = self.scene_driver("aloha_bimanual", rollout)
        driver.reset()
        values = [0, 1.17, -1.4, 0, 0, 0, .5] * 2
        result = driver.step({"name": "scene_action", "arguments": {"values": values}}, execute=True)
        self.assertNotIn("prefixItems", driver.actions[0]["parameters"]["properties"]["values"])
        np.testing.assert_allclose(native.actions[0], values, rtol=1e-6)
        np.testing.assert_allclose(result["state"]["observation"], values, rtol=1e-6)
        self.assertEqual(result["info"]["controlSteps"], 1)

    @unittest.skipUnless(os.environ.get("OPENRAL_TEST_PHYSICS"), "Set OPENRAL_TEST_PHYSICS=1 to download upstream robot assets and run MuJoCo")
    def test_every_bare_simulator_runs_reset_preview_and_two_control_steps(self):
        for item in self.catalog:
            if not item["bareSim"]:
                continue
            with self.subTest(robot=item["robotId"]):
                driver = OpenRALDriver({"openralRoot": SDK_ROOT, "robotId": item["robotId"], "executeEnabled": True, "cameraNames": [], "transport": {"gravity_enabled": False}})
                try:
                    first = driver.reset()
                    target = command(first["state"]["position"])
                    preview = driver.step({"name": "hal_action", "arguments": target})
                    self.assertEqual(preview["info"]["controlSteps"], 0)
                    result = driver.step({"name": "hal_sequence", "arguments": {"commands": [target, target]}}, execute=True)
                    self.assertEqual(result["info"]["controlSteps"], 2)
                    self.assertEqual(len(result["state"]["position"]), item["jointCount"])
                    if item["bareSimKind"] == "mujoco":
                        self.assertGreater(result["info"]["simTimeNs"], first["info"]["simTimeNs"])
                finally:
                    driver.close()


if __name__ == "__main__":
    unittest.main()
