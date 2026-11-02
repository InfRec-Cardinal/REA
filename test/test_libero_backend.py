import argparse
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


backend_path = Path(__file__).resolve().parents[1] / "scripts" / "libero_backend.py"
spec = importlib.util.spec_from_file_location("libero_backend", backend_path)
backend_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_module)


class FakeArray(list):
    """Minimal stand-in for a numpy array: iterable, indexable and ``tolist``-able."""

    def tolist(self):
        return list(self)


class LiberoBackendTests(unittest.TestCase):
    def make_backend(self):
        args = argparse.Namespace(
            camera_size=256, horizon=20, repeat_steps=5, amplitude=0.5, settle_steps=10,
        )
        backend = backend_module.LiberoBackend(args)
        backend.env = Mock()
        backend.observe = Mock(side_effect=lambda: {
            "success": backend.success, "terminated": backend.terminated, "steps": backend.steps,
        })
        return backend

    def test_gripper_state_survives_motion_and_hold(self):
        closed, gripper = backend_module.token_action("P", -1.0)
        moved, gripper = backend_module.token_action("W", gripper)
        held, gripper = backend_module.token_action("X", gripper)
        opened, gripper = backend_module.token_action("O", gripper)
        self.assertEqual(closed, [0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(moved, [0.5, 0, 0, 0, 0, 0, 1])
        self.assertEqual(held, [0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(opened, [0, 0, 0, 0, 0, 0, -1])

    def test_initial_states_preserve_one_based_seed_partition(self):
        self.assertEqual(backend_module.initial_state_index(1, 100), 0)
        self.assertEqual(backend_module.initial_state_index(50, 100), 49)
        self.assertIsNone(backend_module.initial_state_index(51, 50))
        self.assertIsNone(backend_module.initial_state_index(100, 50))
        with self.assertRaises(ValueError):
            backend_module.initial_state_index(101, 100)

    def test_proprio_filters_object_ground_truth(self):
        observation = {
            "robot0_eef_pos": FakeArray([1, 2, 3]),
            "robot0_gripper_qpos": FakeArray([0.04, -0.04]),
            "object-state": FakeArray([9, 9, 9]),
            "butter_1_pos": FakeArray([9, 9, 9]),
        }
        proprio = backend_module.proprio_observation(observation)
        self.assertEqual(proprio, {
            "eefPosition": [1, 2, 3], "gripperPositions": [0.04, -0.04],
            "eef_cm": [100.0, 200.0, 300.0], "gripper": "open", "gripper_fraction": 1.0, "gripper_command": "open",
        })
        self.assertFalse(any("object" in key or "butter" in key for key in proprio))
        self.assertEqual(sorted(backend_module.proprio_observation({"robot0_joint_pos": FakeArray([0.1])})), ["jointPositions"])

    def test_compact_proprio_labels_gripper_and_rounds_world_cm(self):
        def observation(qpos):
            return {"robot0_eef_pos": FakeArray([0.458, 0.0, 0.36849]), "robot0_gripper_qpos": FakeArray([qpos, -qpos]),
                    "butter_1_pos": FakeArray([9, 9, 9])}
        opened = backend_module.compact_proprio(observation(0.0388), False, None)
        self.assertEqual(opened, {"eef_cm": [45.8, 0.0, 36.8], "gripper": "open", "gripper_fraction": 0.97, "gripper_command": "open"})
        closed = backend_module.compact_proprio(observation(0.0008), True, None)
        self.assertEqual((closed["gripper"], closed["gripper_fraction"], closed["gripper_command"]), ("closed", 0.02, "close"))
        holding = backend_module.compact_proprio(observation(0.016), True, 0.4)
        self.assertEqual((holding["gripper"], holding["gripper_fraction"]), ("holding", 0.4))
        self.assertEqual(backend_module.compact_proprio(observation(0.016), False, 0.4)["gripper"], "partial")
        moving = backend_module.compact_proprio(observation(0.016), True, 0.97)
        self.assertEqual(moving["gripper"], "moving")
        self.assertEqual(backend_module.compact_proprio(observation(0.016), True, 0.415)["gripper"], "holding")
        self.assertEqual(backend_module.compact_proprio(observation(0.05), False, None)["gripper_fraction"], 1.0)
        self.assertEqual(backend_module.compact_proprio(observation(-0.01), True, None)["gripper_fraction"], 0.0)
        self.assertEqual(backend_module.gripper_state(True, 0.85, None), "open")
        self.assertEqual(backend_module.gripper_state(True, 0.08, None), "closed")

    def test_token_budget_arithmetic_for_pro_and_long_horizons(self):
        self.assertEqual(backend_module.token_budget(0, 600, 5), {
            "tokensTotal": 120, "tokensUsed": 0, "tokensLeft": 120, "steps": 0, "stepsLeft": 600, "horizon": 600, "repeatSteps": 5,
        })
        self.assertEqual(backend_module.token_budget(165, 600, 5), {
            "tokensTotal": 120, "tokensUsed": 33, "tokensLeft": 87, "steps": 165, "stepsLeft": 435, "horizon": 600, "repeatSteps": 5,
        })
        long = backend_module.token_budget(1000, 1000, 5)
        self.assertEqual((long["tokensTotal"], long["tokensUsed"], long["tokensLeft"], long["stepsLeft"]), (200, 200, 0, 0))
        partial = backend_module.token_budget(3, 600, 5)
        self.assertEqual((partial["tokensUsed"], partial["tokensLeft"]), (0, 119))

    def test_observe_exposes_compact_proprio_budget_and_tracks_gripper_motion(self):
        args = argparse.Namespace(camera_size=256, horizon=600, repeat_steps=5, amplitude=0.5, settle_steps=10)
        backend = backend_module.LiberoBackend(args)
        backend.env = Mock()
        backend.task = {"suite": "libero_goal_task", "index": 0, "name": "task", "language": "open the drawer", "seed": 51}
        backend.steps = 165
        backend.gripper = 1.0

        def observation(qpos):
            return {
                "agentview_image": [[1], [2]], "robot0_eye_in_hand_image": [[3], [4]],
                "robot0_eef_pos": FakeArray([0.458, 0.0, 0.368]), "robot0_gripper_qpos": FakeArray([qpos, -qpos]),
                "object-state": FakeArray([9, 9, 9]), "butter_1_pos": FakeArray([9, 9, 9]),
            }

        titles = []
        draw = Mock()
        draw.text.side_effect = lambda position, title, fill: titles.append(title)
        montage = Mock(width=512, height=280)
        image_module = Mock()
        image_module.new.return_value = montage
        robosuite = Mock()
        robosuite.macros = Mock(IMAGE_CONVENTION="opengl")
        modules = {
            "PIL": Mock(Image=image_module, ImageDraw=Mock(Draw=Mock(return_value=draw))),
            "robosuite": robosuite, "robosuite.macros": robosuite.macros,
        }
        with patch.dict(sys.modules, modules):
            backend.observation = observation(0.0388)
            first = backend.observe()
            backend.observation = observation(0.016)
            second = backend.observe()
            backend.observation = observation(0.016)
            third = backend.observe()
        libero = first["frame"]["libero"]
        self.assertEqual(sorted(libero), ["budget", "observationId", "proprio", "recentHistory", "task"])
        self.assertEqual(libero["proprio"]["eef_cm"], [45.8, 0.0, 36.8])
        self.assertEqual(libero["proprio"]["gripper"], "open")
        self.assertEqual(libero["proprio"]["gripper_command"], "close")
        self.assertEqual(libero["proprio"]["eefPosition"], [0.458, 0.0, 0.368])
        self.assertNotIn("butter_1_pos", libero["proprio"])
        self.assertEqual(libero["budget"], {
            "tokensTotal": 120, "tokensUsed": 33, "tokensLeft": 87, "steps": 165, "stepsLeft": 435, "horizon": 600, "repeatSteps": 5,
        })
        self.assertEqual(second["frame"]["libero"]["proprio"]["gripper"], "moving")
        self.assertEqual(third["frame"]["libero"]["proprio"]["gripper"], "holding")
        self.assertEqual(backend.previous_gripper_fraction, 0.4)
        self.assertEqual(titles[:2], ["External camera (front)", "Wrist camera (L/R mirrored)"])
        self.assertEqual(first["frame"]["width"], 512)
        self.assertEqual((first["steps"], first["horizon"], first["repeatSteps"]), (165, 600, 5))

    def test_reset_uses_random_development_and_official_heldout_states(self):
        for seed, source, state_index in (
            (51, "seeded_random_development", None),
            (50, "official_pruned_heldout", 49),
        ):
            with self.subTest(seed=seed):
                backend = self.make_backend()
                environment = Mock(language_instruction="actual remapped task instruction")
                environment.step.return_value = ({}, 0, False, {})
                environment.check_success.return_value = False
                suite = Mock()
                suite.get_num_tasks.return_value = 10
                suite.get_task_init_states.return_value = list(range(50))
                suite.get_task.return_value = Mock(name="filename task")
                factory = Mock(return_value=environment)
                benchmark = Mock()
                benchmark.get_benchmark.return_value = Mock(return_value=suite)
                with patch.dict(sys.modules, {
                    "numpy": Mock(asarray=lambda value: value),
                    "libero": Mock(benchmark=benchmark),
                    "libero.envs": Mock(OffScreenRenderEnv=factory),
                }):
                    backend.reset({
                        "suite": "libero_goal_task", "task": 0, "seed": seed,
                        "horizon": 60, "repeatSteps": 3, "amplitude": 0.3,
                    })
                environment.seed.assert_called_once_with(seed)
                environment.reset.assert_called_once()
                if state_index is None:
                    environment.set_init_state.assert_not_called()
                    suite.get_task_init_states.assert_not_called()
                else:
                    environment.set_init_state.assert_called_once_with(state_index)
                    suite.get_task_init_states.assert_called_once_with(0)
                self.assertEqual(backend.initial_state_source, source)
                self.assertEqual(backend.initial_state_index, state_index)
                self.assertEqual(backend.task["language"], "actual remapped task instruction")
                self.assertEqual(backend.horizon, 60)
                self.assertEqual(backend.repeat_steps, 3)
                self.assertEqual(backend.amplitude, 0.3)

    def test_horizon_done_does_not_report_success(self):
        backend = self.make_backend()
        backend.env.step.return_value = ({}, 0.0, True, {})
        backend.env.check_success.return_value = False
        with patch.dict(sys.modules, {"numpy": Mock(asarray=lambda value: value)}):
            result = backend.step({"token": "X"})
        self.assertEqual(result, {"success": False, "terminated": True, "steps": 1})
        self.assertEqual(backend.env.step.call_count, 1)

    def test_success_ends_repeat_without_extra_actions(self):
        backend = self.make_backend()
        backend.env.step.return_value = ({}, 1.0, False, {})
        backend.env.check_success.side_effect = [False, True]
        with patch.dict(sys.modules, {"numpy": Mock(asarray=lambda value: value)}):
            result = backend.step({"token": "P"})
        self.assertEqual(result, {"success": True, "terminated": True, "steps": 2})
        self.assertEqual(backend.env.step.call_count, 2)


if __name__ == "__main__":
    unittest.main()
