from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from robot_drivers.aloha import AlohaDriver


class FakeRandom:
    def __init__(self, seed):
        self.random = random.Random(seed)

    def uniform(self, low, high):
        return [self.random.uniform(a, b) for a, b in zip(low, high)]


class FakeTimeStep:
    def __init__(self, reward=None, last=False, discount=1):
        self.reward = reward
        self.discount = discount
        self._last = last

    def last(self):
        return self._last


class FakePhysics:
    def __init__(self):
        self.model = SimpleNamespace(actuator_ctrlrange=[[-2, 2]] * 16)
        self.renders = []
        self.freed = False

    def render(self, **kwargs):
        self.renders.append(kwargs)
        return [[[255, 0, 0]]]

    def free(self):
        self.freed = True


class FakeEnv:
    def __init__(self):
        self.physics = FakePhysics()
        self.task = SimpleNamespace(max_reward=4, get_qpos=lambda physics: [0] * 14, get_qvel=lambda physics: [0] * 14)
        info = SimpleNamespace(joint_lower_limits=[-2] * 6, joint_upper_limits=[2] * 6)
        self.puppet_bot_left = SimpleNamespace(arm=SimpleNamespace(group_info=info))
        self.puppet_bot_right = SimpleNamespace(arm=SimpleNamespace(group_info=info))
        self.reset_calls = []
        self.step_calls = []
        self.rewards = []
        self.last_step = None
        self.discount = 1
        self.image_calls = 0

    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        self.step_calls = []
        return FakeTimeStep()

    def step(self, target):
        self.step_calls.append(target)
        if hasattr(self.task, "get_observation"):
            self.task.get_observation(self.physics)
        else:
            self.get_observation()
        index = len(self.step_calls) - 1
        reward = self.rewards[index] if index < len(self.rewards) else 0
        return FakeTimeStep(reward, len(self.step_calls) == self.last_step, self.discount)

    def get_qpos(self):
        return [0] * 14

    def get_qvel(self):
        return [0] * 14

    def get_effort(self):
        return [0] * 14

    def get_images(self):
        self.image_calls += 1
        return {"cam_high": [[[255, 0, 0]]]}


class FakeSDK:
    def __init__(self):
        self.envs = []
        self.real_options = []
        self.constants = SimpleNamespace(DT=.02, JOINT_NAMES=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
                                         SIM_TASK_CONFIGS={"sim_transfer_cube_scripted": {"episode_len": 400}, "sim_insertion_scripted": {"episode_len": 400}})
        self.np = SimpleNamespace(asarray=lambda values, dtype: list(values), random=SimpleNamespace(RandomState=FakeRandom))
        self.module = SimpleNamespace(make_sim_env=self.make_sim_env, make_real_env=self.make_real_env, BOX_POSE=[None])

    def make_sim_env(self, task_name):
        env = FakeEnv()
        self.envs.append(env)
        return env

    def make_real_env(self, **kwargs):
        self.real_options.append(kwargs)
        return self.make_sim_env(None)


def joint_action(count=None):
    target = [0] * 6 + [.5] + [0] * 6 + [.5]
    return {"name": "joint_position" if count is None else "joint_trajectory", "arguments": {"positions": target if count is None else [target[:] for _ in range(count)]}}


class AlohaDriverTests(unittest.TestCase):
    def driver(self, **options):
        sdk = FakeSDK()
        driver = AlohaDriver({"_sdk": sdk, "executeEnabled": True, **options})
        self.addCleanup(driver.close)
        return driver, sdk

    def test_sim_reset_samples_seeded_pose_without_exposing_object_state(self):
        driver, sdk = self.driver()
        first = driver.reset(seed=21)
        pose = sdk.module.BOX_POSE[0][:]
        driver.reset(seed=21)
        self.assertEqual(sdk.module.BOX_POSE[0], pose)
        self.assertEqual(len(pose), 7)
        self.assertEqual(set(first["state"]), {"qpos", "qvel"})
        self.assertEqual(first["images"][0]["name"], "top")
        self.assertEqual(first["images"][0]["mimeType"], "image/png")
        self.assertEqual(driver.metadata["cameraNames"], ["top"])

    def test_preview_never_steps(self):
        driver, sdk = self.driver(executeEnabled=False)
        driver.reset()
        snapshot = driver.step(joint_action(3))
        self.assertEqual(sdk.envs[0].step_calls, [])
        self.assertEqual(snapshot["info"]["plannedControlSteps"], 3)
        self.assertFalse(snapshot["info"]["executed"])
        with self.assertRaisesRegex(ValueError, "preview"):
            driver.step(joint_action(), execute=True)

    def test_chunk_captures_only_after_final_step_and_latches_success(self):
        driver, sdk = self.driver()
        driver.reset()
        env = sdk.envs[0]
        env.rewards = [0, 4, 0]
        env.physics.renders.clear()
        snapshot = driver.step(joint_action(3), execute=True)
        self.assertEqual(len(env.step_calls), 3)
        self.assertEqual(len(env.physics.renders), 1)
        self.assertEqual(snapshot["reward"], 4)
        self.assertEqual(snapshot["info"]["lastReward"], 0)
        self.assertEqual(snapshot["info"]["maxObservedReward"], 4)
        self.assertEqual(snapshot["info"]["controlSteps"], 3)
        self.assertEqual(snapshot["info"]["totalControlSteps"], 3)
        self.assertTrue(snapshot["success"])
        self.assertFalse(snapshot["terminated"])
        self.assertEqual(snapshot["info"]["controlDt"], .02)

    def test_act_horizon_truncates_chunk(self):
        driver, sdk = self.driver(maxSteps=2, cameraNames=[])
        driver.reset()
        snapshot = driver.step(joint_action(4), execute=True)
        self.assertEqual(len(sdk.envs[0].step_calls), 2)
        self.assertTrue(snapshot["truncated"])
        self.assertFalse(snapshot["terminated"])
        self.assertFalse(snapshot["success"])

    def test_dm_env_terminal_and_time_limit_are_distinct(self):
        for discount, terminated, truncated in ((0, True, False), (1, False, True)):
            driver, sdk = self.driver(cameraNames=[])
            driver.reset()
            sdk.envs[0].last_step = 1
            sdk.envs[0].discount = discount
            snapshot = driver.step(joint_action(2), execute=True)
            self.assertEqual((snapshot["terminated"], snapshot["truncated"]), (terminated, truncated))
            self.assertEqual(len(sdk.envs[0].step_calls), 1)

    def test_complete_chunk_is_validated_before_any_command(self):
        driver, sdk = self.driver()
        driver.reset()
        action = joint_action(2)
        action["arguments"]["positions"][1][6] = 1.1
        with self.assertRaisesRegex(ValueError, "SDK limits"):
            driver.step(action, execute=True)
        self.assertEqual(sdk.envs[0].step_calls, [])
        spec = driver.actions[0]["parameters"]["properties"]["positions"]["prefixItems"]
        self.assertEqual(spec[0]["minimum"], -2)
        self.assertEqual(spec[6]["maximum"], 1)

    def test_switch_to_insertion_releases_physics_and_samples_two_objects(self):
        driver, sdk = self.driver(cameraNames=[])
        old_physics = sdk.envs[0].physics
        driver.reset({"taskName": "sim_insertion_scripted"}, seed=7)
        self.assertTrue(old_physics.freed)
        self.assertEqual(len(sdk.module.BOX_POSE[0]), 14)

    def test_real_initialization_and_reset_do_not_home(self):
        driver, sdk = self.driver(mode="real", cameraNames=["cam_high"])
        snapshot = driver.reset()
        self.assertEqual(sdk.real_options, [{"init_node": True, "setup_robots": False}])
        self.assertEqual(sdk.envs[0].reset_calls, [{"fake": True}])
        self.assertEqual(sdk.envs[0].step_calls, [])
        self.assertIsNone(snapshot["success"])
        self.assertEqual(driver.timing, "realtime")

    def test_real_chunk_reports_submission_and_captures_once(self):
        driver, sdk = self.driver(mode="real", cameraNames=["cam_high"])
        driver.reset()
        sdk.envs[0].image_calls = 0
        snapshot = driver.step(joint_action(3), execute=True)
        self.assertEqual(len(sdk.envs[0].step_calls), 3)
        self.assertEqual(sdk.envs[0].image_calls, 1)
        self.assertEqual(snapshot["info"]["completion"], "submitted")
        self.assertIsNone(snapshot["success"])
        self.assertIsNone(snapshot["reward"])
        self.assertEqual(snapshot["info"]["lastReward"], 0)
        self.assertFalse(driver.stop()["physicalStop"])

    def test_close_frees_simulator_and_is_idempotent(self):
        driver, sdk = self.driver()
        driver.close()
        driver.close()
        self.assertTrue(sdk.envs[0].physics.freed)


if __name__ == "__main__":
    unittest.main()
