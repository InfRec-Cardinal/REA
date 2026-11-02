"""Online continuation uses the existing scene and the episode's remaining resources."""
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_behavior_backend import FakeCodeEnv, FakeLow

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("behavior_online_backend", ROOT / "scripts/behavior_backend.py")
backend_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_module)
agent_module = backend_module.online_agent_module()


class OnlineAgentTests(unittest.TestCase):
    def make_backend(self, runner, held=None, budget=10, horizon=20, lift_error=False):
        low = FakeLow()
        env = FakeCodeEnv(low)
        hands = []
        low.lifted_arms = []

        def functions(current):
            self.assertIs(current, env)
            def check(arm=0):
                hands.append(arm)
                return arm == held
            def lift(arm=0):
                if lift_error:
                    raise RuntimeError("lift planning failed")
                low.env.step([0])
                low.lifted_arms.append(arm)
                low.completed = True
            return {"navigate_to_pose": env._exec_globals["navigate_to_pose"],
                    "check_object_in_hand": check, "lift_arm": lift}

        backend = backend_module.BehaviorBackend(
            backend_module.default_args(frame_size=32, judge_frame_size="32x32", infra_probe_ports=""),
            env_factory=lambda task: env, agent_runner=runner, api_functions=functions)
        # These tests exercise continuation and tool execution; camera rendering has its own tests.
        backend.observe = lambda request: {"steps": backend.sim_steps, "stepsLeft": backend.horizon - backend.sim_steps}
        backend.reset({"suite": "behavior", "task": "drill", "seed": 1,
                       "phase": "heldout", "horizon": horizon, "wall_budget_s": budget})
        return backend, env, low, hands

    def test_continues_same_scene_and_scores_only_at_end(self):
        seen = {}
        def runner(**kwargs):
            seen.update(kwargs)
            kwargs["fns"]["navigate_to_pose"]([1, 2, 3])
            return {"stop_reason": "agent_finished", "final_output": "done", "tool_calls": [], "tool_results": []}

        backend, env, low, hands = self.make_backend(runner)
        backend.run_program({"code": "navigate_to_pose([0, 0, 0])", "wall_timeout_s": 3})
        with patch.dict(os.environ, {"REA_API_KEY": "test-secret", "REA_BASE_URL": "https://provider.invalid/v1", "REA_MODEL": "test-model"}):
            reply = backend.dispatch({"command": "run_agent", "images": False,
                                      "wall_timeout_s": 8, "context": {"error": "left grasp failed"}})
        self.assertEqual(env.reset_calls, [1])
        self.assertEqual((reply["steps_before"], reply["steps_after"]), (1, 2))
        self.assertEqual(hands, [0, 1])
        self.assertEqual(seen["args"].agent_context["previous_execution"], {"error": "left grasp failed"})
        self.assertLessEqual(seen["budget_s"], 8)
        self.assertEqual(reply["agent_trace"]["tool_results"][0]["result"], "false")
        self.assertNotIn("test-secret", json.dumps(reply))
        self.assertEqual(low.task_completed_calls, 0)
        backend.score()
        self.assertEqual(low.task_completed_calls, 1)
        with self.assertRaisesRegex(ValueError, "sealed"):
            backend.run_agent({"images": False})

    def test_held_object_stops_before_model_and_leaves_holding_arm_selected(self):
        def runner(**kwargs):
            self.fail("held object must not start SDK")
        backend, env, low, hands = self.make_backend(runner, held=1)
        with patch.dict(os.environ, {}, clear=True):
            reply = backend.run_agent({"images": False})
        self.assertEqual(reply["agent_trace"]["stop_reason"], "inhand_guard")
        self.assertEqual(reply["agent_trace"]["held_arm"], 1)
        self.assertEqual(reply["agent_trace"]["tool_results"][-1]["result"], "true")
        self.assertEqual(hands, [0, 1, 1])
        self.assertEqual(low.lifted_arms, [1])
        self.assertEqual(reply["steps"], 1)
        self.assertEqual(env.reset_calls, [1])
        self.assertEqual(low.task_completed_calls, 0)
        self.assertTrue(backend.score()["task_completed"])

    def test_failed_guard_lift_hands_current_state_to_sdk(self):
        seen = {}
        def runner(**kwargs):
            seen.update(kwargs)
            return {"stop_reason": "agent_finished", "tool_calls": [], "tool_results": []}
        backend, env, low, hands = self.make_backend(runner, held=0, lift_error=True)
        with patch.dict(os.environ, {"REA_API_KEY": "test-secret", "REA_BASE_URL": "https://provider.invalid/v1", "REA_MODEL": "test-model"}):
            reply = backend.run_agent({"images": False})
        self.assertEqual(reply["agent_trace"]["stop_reason"], "agent_finished")
        self.assertEqual(hands, [0, 0])
        self.assertEqual(env.reset_calls, [1])
        self.assertEqual(low.lifted_arms, [])
        self.assertEqual(seen["args"].agent_context["hand_checks"], {"0": "true"})
        self.assertIn("lift planning failed", seen["args"].agent_context["handoff_tools"][1]["result"])
        self.assertEqual(low.task_completed_calls, 0)

    def test_agent_uses_remaining_wall_budget_and_restores_step_wrapper(self):
        def runner(**kwargs):
            time.sleep(1)
            self.fail("model wait exceeded remaining budget")
        backend, env, low, _ = self.make_backend(runner, budget=1)
        backend.wall_used_s = 0.9
        original_step = low.env.step
        with patch.dict(os.environ, {"REA_API_KEY": "test-secret", "REA_BASE_URL": "https://provider.invalid/v1", "REA_MODEL": "test-model"}):
            started = time.monotonic()
            reply = backend.run_agent({"images": False, "wall_timeout_s": 4})
        self.assertLess(time.monotonic() - started, 0.8)
        self.assertEqual(reply["wall_timeout_s"], 0.1)
        self.assertEqual(reply["truncated_reason"], "wall_timeout")
        self.assertEqual(reply["wall_left_s"], 0)
        self.assertEqual(low.env.step, original_step)
        self.assertFalse(backend._running)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        self.assertEqual(low.task_completed_calls, 0)

    def test_horizon_is_shared_with_prior_program(self):
        def runner(**kwargs):
            kwargs["fns"]["navigate_to_pose"]([0, 0, 0])
            kwargs["fns"]["navigate_to_pose"]([0, 0, 0])
            self.fail("horizon should interrupt the second action")
        backend, env, low, _ = self.make_backend(runner, horizon=2)
        backend.run_program({"code": "navigate_to_pose([0, 0, 0])", "wall_timeout_s": 3})
        with patch.dict(os.environ, {"REA_API_KEY": "test-secret", "REA_BASE_URL": "https://provider.invalid/v1", "REA_MODEL": "test-model"}):
            reply = backend.run_agent({"images": False})
        self.assertEqual(reply["truncated_reason"], "horizon")
        self.assertEqual((reply["steps_before"], reply["steps_after"]), (1, 2))
        self.assertEqual(low.env.steps, 2)
        self.assertEqual(low.task_completed_calls, 0)

    def test_budget_exhausted_does_not_start_model_or_tools(self):
        def runner(**kwargs):
            self.fail("no episode time remains")
        backend, env, low, hands = self.make_backend(runner, budget=2)
        backend.wall_used_s = 2
        reply = backend.run_agent({"images": False})
        self.assertEqual(reply["truncated_reason"], "wall_budget")
        self.assertEqual(reply["wall_s"], 0)
        self.assertEqual(hands, [])
        self.assertEqual(low.task_completed_calls, 0)

    def test_catalog_and_config_cover_all_five_tasks(self):
        backend, *_ = self.make_backend(lambda **kwargs: {})
        self.assertEqual([row["name"] for row in backend.catalog()["suites"][0]["tasks"]],
                         ["trash", "radio", "teddy", "drill", "camera"])
        self.assertEqual(backend_module.TASKS["camera"]["config"],
                         "env_configs/r1pro/r1pro_pick_up_camera_oracle.yaml")

    def test_tool_results_include_exact_model_feedback(self):
        server = agent_module.ToolServer({"check_object_in_hand": lambda arm: arm == 1},
                                        time.monotonic() + 5, backend_module.ProgramInterrupt)
        self.assertEqual(server.execute("check_object_in_hand", {"arm": 1}), "true")
        self.assertEqual(server.calls[0]["result"], "true")
        self.assertEqual(server.tool_results[0]["result"], "true")
        self.assertEqual(server.tool_results[0]["args"], {"arm": 1})

    def test_relative_sdk_python_and_image_work_from_child_directory(self):
        image = b"\xff\xd8camera-frame-fixture\xff\xd9"
        with tempfile.TemporaryDirectory() as directory:
            interpreter = Path(directory) / "venv" / "bin" / "python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text(
                "#!" + sys.executable + "\n"
                "import json, sys\nfrom pathlib import Path\n"
                "cfg = json.loads(sys.argv[-1])\n"
                "photo = Path(cfg['image'])\n"
                "assert photo.is_absolute() and photo.read_bytes()[:2] == b'\\xff\\xd8'\n"
                "print(json.dumps({'type':'done','stop_reason':'agent_finished','final_output':'image readable'}), flush=True)\n")
            interpreter.chmod(0o755)
            episode = Path(directory) / "episode"
            episode.mkdir()
            args = SimpleNamespace(agent_model="fake", agent_max_turns=3, agent_llm_timeout=5,
                                   agent_images=True, sdk_python=os.path.relpath(interpreter))
            result = agent_module.run_episode({}, "pick up", "robot", image, args, 5,
                                              "test-secret", "https://unused.invalid", backend_module.ProgramInterrupt,
                                              os.path.relpath(episode), lambda **entry: None)
        self.assertEqual(result["stop_reason"], "agent_finished")
        self.assertEqual(result["final_output"], "image readable")
        self.assertTrue(result["images"])

    def test_credentials_stay_with_their_provider(self):
        cases = [
            ({"REA_API_KEY": "rea-key", "REA_BASE_URL": "https://provider.invalid/v1", "OPENAI_BASE_URL": "https://other.invalid"},
             ("rea-key", "https://provider.invalid/v1")),
            ({"OPENAI_API_KEY": "openai-key", "REA_BASE_URL": "https://other.invalid"},
             ("openai-key", "https://api.openai.com/v1")),
            ({"AGENT_API_KEY": "agent-key", "AGENT_BASE_URL": "https://agent.invalid/v1"},
             ("agent-key", "https://agent.invalid/v1")),
        ]
        for environment, expected in cases:
            with self.subTest(environment=tuple(environment)), patch.dict(os.environ, environment, clear=True):
                self.assertEqual(backend_module.online_agent_credentials(), expected)
        with patch.dict(os.environ, {"AGENT_API_KEY": "agent-key"}, clear=True):
            with self.assertRaisesRegex(ValueError, "AGENT_BASE_URL"):
                backend_module.online_agent_credentials()
        with patch.dict(os.environ, {"REA_API_KEY": "rea-key"}, clear=True):
            with self.assertRaisesRegex(ValueError, "REA_BASE_URL"):
                backend_module.online_agent_credentials()

    def test_sdk_wait_is_bounded_and_child_is_reaped(self):
        real_popen = subprocess.Popen
        children = []
        def sleeper(*args, **kwargs):
            child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
            children.append(child)
            return child
        args = SimpleNamespace(agent_model="fake", agent_max_turns=3, agent_llm_timeout=30,
                               agent_images=False, sdk_python=sys.executable)
        with tempfile.TemporaryDirectory() as directory, patch.object(agent_module.subprocess, "Popen", sleeper):
            started = time.monotonic()
            result = agent_module.run_episode({}, "pick up", "robot", None, args, 0.1,
                                              "test-secret", "https://unused.invalid", backend_module.ProgramInterrupt,
                                              directory, lambda **entry: None)
        self.assertEqual(result["stop_reason"], "program_budget")
        self.assertLess(time.monotonic() - started, 1)
        self.assertIsNotNone(children[0].poll())


if __name__ == "__main__":
    unittest.main()
