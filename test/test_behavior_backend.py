import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import time
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

backend_path = Path(__file__).resolve().parents[1] / "scripts" / "behavior_backend.py"
spec = importlib.util.spec_from_file_location("behavior_backend", backend_path)
backend_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_module)

try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover
    np = None
    Image = None

FIXED_NOW = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)


class FakeInnerEnv:
    """Stands in for og.Environment: every primitive steps through `step`."""

    def __init__(self):
        self.steps = 0

    def step(self, action):
        self.steps += 1
        return {}, 0.0, False, False, {}


class FakeLow:
    def __init__(self, robot_key="robot_abc"):
        self.env = FakeInnerEnv()
        self._sim_step_count = 0
        self.robot_key = robot_key
        self.completed = False
        self.task_completed_calls = 0
        self.perception_calls = 0
        self.reset_seeds = []

    def _frame(self, value):
        frame = np.zeros((512, 512, 4), dtype=np.uint8)
        frame[..., 0] = value
        frame[..., 3] = 255
        return frame

    def get_observation(self):
        return {"task": {"low_dim": [1.0, 2.0]}, self.robot_key: {
            f"{self.robot_key}:zed_link:Camera:0": {"rgb": self._frame(10), "depth_linear": np.zeros((512, 512))},
            f"{self.robot_key}:left_realsense_link:Camera:0": {"rgb": self._frame(20)},
            f"{self.robot_key}:right_realsense_link:Camera:0": {"rgb": self._frame(30)},
            "proprio": np.zeros(68),
        }}

    def get_perception_observation(self):
        self.perception_calls += 1
        return self.get_observation()

    def task_completed(self):
        self.task_completed_calls += 1
        return self.completed


class FakeCodeEnv:
    def __init__(self, low):
        self.low_level_env = low
        self.reset_calls = []
        self._init_exec_globals()

    def _init_exec_globals(self):
        low = self.low_level_env

        def move(n):
            for _ in range(n):
                low.env.step([0.0])
            return True

        self._exec_globals = {"__name__": "__main__", "env": low, "obs": {}, "APIS": {}, "INPUTS": {}, "RESULT": None,
                              "navigate_to_pose": lambda *a, **k: move(1), "grasp_object": lambda *a, **k: move(5), "check_object_in_hand": lambda arm=0: True}

    def reset(self, *, seed=None, options=None):
        self.reset_calls.append(seed)
        self.low_level_env.reset_seeds.append(seed)
        self._init_exec_globals()
        return {}, {}

    def _exec_user_code(self, code):
        stdout, stderr = io.StringIO(), io.StringIO()
        ok = True
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, self._exec_globals, self._exec_globals)
        except BaseException:
            ok = False
            traceback.print_exc(file=stderr)
        return {"ok": ok, "stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "result": self._exec_globals.get("RESULT")}


@unittest.skipUnless(np is not None and Image is not None, "numpy and Pillow are required")
class BehaviorBackendTests(unittest.TestCase):
    def make_backend(self, **overrides):
        options = {"frame_size": 64, "wall_timeout_s": 5.0, "judge_frame_size": "48x48", "infra_probe_ports": "", "max_steps": 1000}
        options.update(overrides)
        args = backend_module.default_args(**options)
        low = FakeLow()
        code_env = FakeCodeEnv(low)
        built = []

        def factory(task):
            built.append(task)
            return code_env

        backend = backend_module.BehaviorBackend(args, env_factory=factory, now=lambda: FIXED_NOW, service_probe=lambda: {"8114": True, "8115": True})
        return backend, code_env, low, built

    def test_catalog_lists_the_five_tasks_and_seed_ranges(self):
        backend, *_ = self.make_backend()
        catalog = backend.catalog()
        self.assertEqual([t["name"] for t in catalog["suites"][0]["tasks"]], ["trash", "radio", "teddy", "drill", "camera"])
        self.assertEqual(catalog["seeds"], {"heldout": [1, 25], "development": [26, 35]})
        self.assertEqual(catalog["backend"]["benchmark"], "behavior")

    def test_reset_loads_the_instance_and_returns_head_frames(self):
        backend, code_env, low, built = self.make_backend()
        reply = backend.reset({"suite": "behavior", "task": "trash", "seed": 26, "phase": "development", "horizon": 500, "wall_budget_s": 900})
        self.assertEqual(built, ["trash"])
        self.assertEqual(code_env.reset_calls, [26])
        self.assertEqual(reply["initialStateIndex"], 26)
        self.assertEqual(reply["resetSemantics"], "aspire")
        self.assertEqual(reply["task"]["language"], backend_module.TASKS["trash"]["language"])
        self.assertEqual(reply["frame"]["width"], 64)
        self.assertEqual(reply["frame_judge"]["width"], 48)
        self.assertEqual(reply["horizon"], 500)
        self.assertEqual(reply["wallBudgetS"], 900)
        self.assertEqual(low.task_completed_calls, 0)

    def test_reset_rejects_wrong_phase_seeds_and_other_reset_semantics(self):
        backend, *_ = self.make_backend()
        with self.assertRaises(ValueError):
            backend.reset({"suite": "behavior", "task": "trash", "seed": 3, "phase": "development"})
        with self.assertRaises(ValueError):
            backend.reset({"suite": "behavior", "task": "trash", "seed": 30, "phase": "heldout"})
        with self.assertRaises(ValueError):
            backend.reset({"suite": "behavior", "task": "trash", "seed": 40})
        with self.assertRaises(ValueError):
            backend.reset({"suite": "behavior", "task": "trash", "seed": 1, "reset": "seeded_random"})
        with self.assertRaises(ValueError):
            backend.reset({"suite": "libero_goal", "task": "trash", "seed": 1})

    def test_r4_front_view_has_three_tiles_and_restores_external_camera(self):
        try:
            from scipy.spatial.transform import Rotation
        except ImportError:
            self.skipTest("scipy is required for external camera poses")
        backend, _, low, _ = self.make_backend()
        backend.reset({"suite": "behavior", "task": "trash", "seed": 1})
        original_position = np.array([8.0, 9.0, 10.0])
        original_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        camera = SimpleNamespace(position=original_position.copy(), orientation=original_orientation.copy())
        camera.get_position_orientation = lambda: (camera.position.copy(), camera.orientation.copy())
        def move_camera(position, orientation):
            camera.position = np.asarray(position).copy()
            camera.orientation = np.asarray(orientation).copy()
        camera.set_position_orientation = move_camera
        low.env.external_sensors = {"external_camera": camera}
        low.env.robots = [SimpleNamespace(get_position_orientation=lambda: (np.zeros(3), original_orientation))]
        original_observation = low.get_observation
        renders = []
        def observation():
            obs = original_observation()
            renders.append(camera.position.copy())
            value = 40 if camera.position[2] > 1.5 else (80 if camera.position[1] > 0 else 120)
            obs["external"] = {"external_camera": {"rgb": low._frame(value)}}
            return obs
        low.get_observation = observation
        fake_torch = SimpleNamespace(tensor=lambda data, dtype: np.asarray(data, dtype=dtype), float32=np.float32,
                                     Tensor=type("FakeTensor", (), {}))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            reply = backend.observe({"view": "front"})
        self.assertEqual((reply["frame"]["width"], reply["frame"]["height"]), (192, 64))
        self.assertEqual((reply["frame_judge"]["width"], reply["frame_judge"]["height"]), (144, 48))
        self.assertEqual(reply["frame"]["behavior"]["front"]["views"], ["front_high", "front_left", "front_right"])
        import base64
        image = Image.open(io.BytesIO(base64.b64decode(reply["frame"]["imageB64"])))
        for x, value in ((32, 40), (96, 80), (160, 120)):
            self.assertAlmostEqual(image.getpixel((x, 32))[0], value, delta=2)
        np.testing.assert_array_equal(camera.position, original_position)
        np.testing.assert_array_equal(camera.orientation, original_orientation)
        np.testing.assert_array_equal(renders[-1], original_position)
        self.assertEqual(low.env.steps, 0)
        self.assertEqual(low.task_completed_calls, 0)

    def test_one_task_per_backend_process(self):
        backend, *_ = self.make_backend()
        backend.reset({"suite": "behavior", "task": "trash", "seed": 1})
        with self.assertRaises(ValueError):
            backend.reset({"suite": "behavior", "task": "radio", "seed": 1})

    def test_run_program_counts_inner_steps_and_scores_once(self):
        backend, code_env, low, _ = self.make_backend()
        backend.reset({"suite": "behavior", "task": "radio", "seed": 2, "phase": "heldout", "wall_budget_s": 100})
        code = "navigate_to_pose([0, 0, 0])\ngrasp_object(None, None, 'red radio')\nprint('done')\n"
        digest = hashlib.sha256(code.encode()).hexdigest()
        reply = backend.run_program({"which": "fix", "code": code, "allowed_sha256": [digest], "wall_timeout_s": 10})
        self.assertIsNone(reply["exception"])
        self.assertEqual(reply["steps"], 6)
        self.assertEqual(reply["sha256"], digest)
        self.assertFalse(reply["truncated"])
        self.assertEqual(low._sim_step_count, 6)
        observe = backend.observe({"view": "montage"})
        self.assertEqual(observe["steps"], 6)
        self.assertEqual(observe["frame"]["width"], 128)
        low.completed = True
        score = backend.score()
        self.assertEqual(score, {"task_completed": True, "steps": 6, "horizon": 1000, "scoredAt": FIXED_NOW.isoformat()})
        self.assertEqual(low.task_completed_calls, 1)
        with self.assertRaises(ValueError):
            backend.score()
        with self.assertRaises(ValueError):
            backend.run_program({"which": "fix", "code": code})

    def test_heldout_requires_pin_and_pin_must_match(self):
        backend, *_ = self.make_backend()
        backend.reset({"suite": "behavior", "task": "trash", "seed": 5, "phase": "heldout"})
        with self.assertRaises(ValueError):
            backend.run_program({"which": "fix", "code": "print(1)\n", "allowed_sha256": []})
        with self.assertRaises(ValueError):
            backend.run_program({"which": "fix", "code": "print(1)\n", "allowed_sha256": ["0" * 64]})

    def test_forbidden_identifiers_are_refused_but_comments_are_not(self):
        backend, *_ = self.make_backend()
        backend.reset({"suite": "behavior", "task": "trash", "seed": 26})
        for code in ("env.robot.get_joint_positions()\n", "print(obs['task'])\n", "APIS['R1ProControlApi']\n", "x = getattr(navigate_to_pose, '__self__')\n",
                     "env.controller._get_obj_in_hand()\n", "import omnigibson as og\n", "task_completed()\n"):
            with self.assertRaises(ValueError, msg=code):
                backend.run_program({"which": "fix", "code": code})
        ok = "# the env is stepped by the API; obs are perception only\nnavigate_to_pose([0, 0, 0])\n"
        reply = backend.run_program({"which": "fix", "code": ok})
        self.assertIsNone(reply["exception"])
        self.assertEqual(backend_module.scan_program("s = 'env'\n"), None)  # string literal, cannot reach the object without getattr/eval
        self.assertEqual(backend_module.scan_program("print(f'{env.robot}')\n"), "env")  # f-string expressions still count

    def test_horizon_truncates_a_program_and_reports_it(self):
        backend, code_env, low, _ = self.make_backend(max_steps=10)
        backend.reset({"suite": "behavior", "task": "trash", "seed": 26, "horizon": 4})
        reply = backend.run_program({"which": "fix", "code": "navigate_to_pose([0, 0, 0])\ngrasp_object(None, None, 'x')\nprint('unreachable')\n"})
        self.assertTrue(reply["truncated"])
        self.assertEqual(reply["truncated_reason"], "horizon")
        self.assertEqual(reply["steps_after"], 4)
        self.assertEqual(low.env.steps, 4)

    def test_program_timeout_waits_until_native_step_returns(self):
        backend, code_env, low, _ = self.make_backend()
        backend.reset({"suite": "behavior", "task": "trash", "seed": 26})
        callback_returned = []

        def native_step(action):
            self.assertTrue(backend._in_sim_step)
            backend._alarm(None, None)
            callback_returned.append(True)
            return {}, 0.0, False, False, {}

        low.env.step = native_step
        reply = backend.run_program({"code": "navigate_to_pose([0, 0, 0])\nprint('unreachable')\n"})
        self.assertEqual(callback_returned, [True])
        self.assertEqual(reply["truncated_reason"], "wall_timeout")
        self.assertEqual(reply["exception_class"], "timeout")
        self.assertEqual(reply["steps"], 1)
        self.assertNotIn("unreachable", reply["stdout_tail"])
        self.assertIs(low.env.step, native_step)
        self.assertFalse(backend._in_sim_step)
        self.assertFalse(backend._running)

    def test_agent_timeout_waits_until_native_step_returns(self):
        backend, code_env, low, _ = self.make_backend()
        backend.reset({"suite": "behavior", "task": "trash", "seed": 26})
        callback_returned = []

        def native_step(action):
            self.assertTrue(backend._in_sim_step)
            backend._alarm(None, None)
            callback_returned.append(True)
            return {}, 0.0, False, False, {}

        def agent_runner(**kwargs):
            kwargs["fns"]["navigate_to_pose"]([0, 0, 0])
            self.fail("the step boundary must interrupt before another agent action")

        low.env.step = native_step
        backend.agent_runner = agent_runner
        backend.api_functions = lambda env: {
            "check_object_in_hand": lambda arm=0: False,
            "navigate_to_pose": lambda pose: low.env.step(pose),
        }
        with patch.dict("os.environ", {"REA_API_KEY": "test-key", "REA_BASE_URL": "https://provider.invalid/v1", "REA_MODEL": "test-model"}):
            reply = backend.run_agent({"images": False})
        self.assertEqual(callback_returned, [True])
        self.assertEqual(reply["truncated_reason"], "wall_timeout")
        self.assertEqual(reply["exception_class"], "timeout")
        self.assertEqual(reply["steps"], 1)
        self.assertIs(low.env.step, native_step)
        self.assertFalse(backend._in_sim_step)
        self.assertFalse(backend._running)
        self.assertEqual(low.task_completed_calls, 0)

    def test_native_step_exception_restores_interrupt_state(self):
        backend, code_env, low, _ = self.make_backend()
        backend.reset({"suite": "behavior", "task": "trash", "seed": 26})

        def native_step(action):
            raise RuntimeError("simulator step failed")

        low.env.step = native_step
        reply = backend.run_program({"code": "navigate_to_pose([0, 0, 0])\n"})
        self.assertEqual(reply["exception"]["type"], "RuntimeError")
        self.assertIs(low.env.step, native_step)
        self.assertFalse(backend._in_sim_step)
        self.assertFalse(backend._running)

    def test_wall_timeout_interrupts_a_busy_program(self):
        backend, *_ = self.make_backend()
        backend.reset({"suite": "behavior", "task": "trash", "seed": 26})
        started = time.perf_counter()
        reply = backend.run_program({"which": "fix", "code": "import time\nwhile True:\n    time.sleep(0.01)\n", "wall_timeout_s": 0.5})
        self.assertLess(time.perf_counter() - started, 5)
        self.assertEqual(reply["truncated_reason"], "wall_timeout")
        self.assertEqual(reply["exception_class"], "timeout")

    def test_program_exception_is_classified_and_perception_render_is_used_when_patch_on(self):
        import os
        backend, code_env, low, _ = self.make_backend()
        backend.reset({"suite": "behavior", "task": "trash", "seed": 26})
        reply = backend.run_program({"which": "fix", "code": "raise ValueError('No detections; environment constraints or model mismatch')\n"})
        self.assertEqual(reply["exception"]["type"], "ValueError")
        self.assertEqual(reply["exception_class"], "program")
        os.environ["REA_BEHAVIOR_PT_OBS"] = "1"
        try:
            before = low.perception_calls
            backend.observe({})
            self.assertEqual(low.perception_calls, before + 1)
        finally:
            del os.environ["REA_BEHAVIOR_PT_OBS"]

    def test_wire_protocol_roundtrip(self):
        backend, *_ = self.make_backend()
        reply = backend_module.handle_request(backend, json.dumps({"id": 1, "command": "catalog"}))
        reply = json.loads(reply) if isinstance(reply, str) else reply
        self.assertTrue(reply["ok"])
        bad = backend_module.handle_request(backend, json.dumps({"id": 2, "command": "score"}))
        bad = json.loads(bad) if isinstance(bad, str) else bad
        self.assertFalse(bad["ok"])
        self.assertIn("Reset an episode", bad["error"])

    def test_front_view_falls_back_to_three_head_frames_without_an_external_camera(self):
        backend, _low, _code_env, _built = self.make_backend()
        backend.reset({"suite": "behavior", "task": "teddy", "seed": 26})
        observe = backend.observe({"view": "front"})
        self.assertEqual(observe["frame"]["view"], "front")
        self.assertEqual(observe["frame"]["width"], 3 * 64)
        self.assertEqual(observe["frame"]["height"], 64)
        self.assertEqual(observe["frame"]["behavior"]["front"], {"views": ["head", "head", "head"], "fallback": "no_external_camera"})
        self.assertEqual(observe["frame_judge"]["width"], 3 * 48)
        self.assertEqual(observe["frame_judge"]["height"], 48)
        self.assertEqual(observe["task"]["language"], backend_module.TASKS["teddy"]["language"])


if __name__ == "__main__":
    unittest.main()
