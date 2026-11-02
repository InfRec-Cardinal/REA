import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import signal
import time
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch


backend_path = Path(__file__).resolve().parents[1] / "scripts" / "aspire_backend.py"
spec = importlib.util.spec_from_file_location("aspire_backend", backend_path)
backend_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_module)

try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover - imaging deps are optional for the pure tests
    np = None
    Image = None

FIXED_NOW = datetime(2026, 9, 9, 8, 0, 0, tzinfo=timezone.utc)
TASK_NAMES = ["open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_stove", "turn_on_the_stove"]
SCORING_KEYS = {"success", "reward", "terminated", "task_completed", "done"}
RUN_REPLY_KEYS = {
    "which", "steps_before", "steps_after", "steps", "wall_s", "exception", "exception_class", "truncated",
    "truncated_reason", "horizon", "sha256", "stdout_tail", "stderr_tail", "wall_timeout_s", "wall_used_s",
    "wall_budget_s", "wall_left_s", "infra_probe",
}


def fake_task_lookup(suite, task):
    if isinstance(task, int):
        if not 0 <= task < len(TASK_NAMES):
            raise ValueError(f"Task index out of range: {task}")
        return task
    if task not in TASK_NAMES:
        raise ValueError(f"Task not found in {suite}: {task}")
    return TASK_NAMES.index(task)


class FakeSim:
    def __init__(self):
        self.calls = []

    def render(self, camera_name, width, height, depth=False):
        self.calls.append((camera_name, width, height, depth))
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[0, :, :] = 255  # top row of MuJoCo's bottom-up frame -> bottom row once flipped
        if camera_name == "robot0_eye_in_hand":
            frame[:, :, 2] = 200
        return frame


class FakeInnerEnv:
    def __init__(self):
        self.sim = FakeSim()
        self.closed = 0

    def close(self):
        self.closed += 1


class FakeHandle:
    """Dataclass-like handle: instance attributes can shadow the class-level step()."""

    def __init__(self, low, init_states, task_language):
        self.low = low
        self.init_states = init_states
        self.task_language = task_language
        self.env = FakeInnerEnv()
        self.step_calls = 0

    def step(self, action):
        self.step_calls += 1
        self.low._sim_step_count += 1
        return ({}, 0.0, False, {})


class FakeLow:
    def __init__(self, init_states, task_language="Open the middle drawer of the cabinet"):
        self._sim_step_count = 0
        self._gripper_fraction = 1.0
        self.max_steps = 4000
        self._current_obs = {
            "robot0_eef_pos": [0.458, 0.0, 0.368],
            "robot0_gripper_qpos": [0.0388, -0.0388],
            "akita_black_bowl_1_pos": [9.0, 9.0, 9.0],
        }
        self.handle = FakeHandle(self, init_states, task_language)
        self.task_completed_calls = 0
        self.completed = True

    def task_completed(self):
        self.task_completed_calls += 1
        return self.completed


class FakeCodeEnv:
    """Mirrors CodeExecutionEnvBase._exec_user_code / reset (spec F11) without ASPIRE."""

    def __init__(self, low):
        self.low_level_env = low
        self.reset_calls = []
        self._apis = {"api": Mock()}
        self._init_exec_globals()

    def _init_exec_globals(self):
        self._exec_globals = {"__name__": "__main__", "env": self.low_level_env, "RESULT": None}

    def reset(self, *, seed=None, options=None):
        self.reset_calls.append({"seed": seed, "init_states": self.low_level_env.handle.init_states})
        self.low_level_env._sim_step_count = 0
        for _ in range(10):
            self.low_level_env.handle.step([0.0] * 8)
        self._init_exec_globals()
        return {}, {}

    def _exec_user_code(self, code):
        self._exec_globals["env"] = self.low_level_env
        stdout, stderr = io.StringIO(), io.StringIO()
        ok = True
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, self._exec_globals, self._exec_globals)
        except BaseException:
            ok = False
            traceback.print_exc(file=stderr)
        return {"ok": ok, "stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "result": self._exec_globals.get("RESULT")}


def collect_keys(payload, found=None):
    found = set() if found is None else found
    if isinstance(payload, dict):
        for key, value in payload.items():
            found.add(key)
            collect_keys(value, found)
    elif isinstance(payload, list):
        for value in payload:
            collect_keys(value, found)
    return found


@unittest.skipUnless(np is not None and Image is not None, "numpy and Pillow are required")
class AspireBackendTests(unittest.TestCase):
    def make_backend(self, init_states=None, service_probe=None, clock=None, **overrides):
        # infra_probe_ports "" -> no TCP probes on the test machine; judge frame 16x10 keeps renders cheap.
        options = {"frame_size": 32, "wall_timeout_s": 5.0, "judge_frame_size": "16x10", "infra_probe_ports": ""}
        options.update(overrides)
        args = backend_module.default_args(**options)
        low = FakeLow(list(range(50)) if init_states is None else init_states)
        code_env = FakeCodeEnv(low)
        factory = Mock(return_value=code_env)
        extra = {} if clock is None else {"clock": clock}
        backend = backend_module.AspireBackend(
            args, env_factory=factory, task_lookup=fake_task_lookup, now=lambda: FIXED_NOW, service_probe=service_probe, **extra,
        )
        return backend, code_env, low, factory

    def reset(self, backend, seed=51, **extra):
        return backend.reset({"suite": "libero_goal_swap", "task": TASK_NAMES[0], "seed": seed, **extra})

    # 1. reset semantics
    def test_reset_aspire_semantics_and_wrap(self):
        backend, code_env, low, factory = self.make_backend()
        reply = self.reset(backend, seed=7)
        self.assertEqual(code_env.reset_calls, [{"seed": 7, "init_states": list(range(50))}])
        self.assertEqual(reply["initialStateIndex"], 6)
        self.assertEqual(reply["initialStateSource"], "aspire_modulo_init_state")
        self.assertEqual(reply["resetSemantics"], "aspire")
        self.assertEqual(reply["episode"], 1)
        self.assertEqual(reply["steps"], 10)
        self.assertEqual(reply["task"], {
            "suite": "libero_goal_swap", "index": 0, "name": TASK_NAMES[0],
            "language": "Open the middle drawer of the cabinet", "seed": 7,
        })
        factory.assert_called_once_with(backend.args.config, "libero_goal_swap", 0, 4000, backend.args.api)
        wrapped = self.reset(backend, seed=51)
        self.assertEqual(wrapped["initialStateIndex"], 0)
        self.assertEqual(wrapped["episode"], 2)
        self.assertEqual(factory.call_count, 1)  # same identity -> env reused

    def test_reset_seeded_random_hides_init_states_only_during_reset(self):
        backend, code_env, low, _ = self.make_backend()
        reply = self.reset(backend, seed=51, reset="seeded_random")
        self.assertEqual(code_env.reset_calls, [{"seed": 51, "init_states": None}])
        self.assertEqual(low.handle.init_states, list(range(50)))
        self.assertEqual(reply["initialStateSource"], "seeded_random_development")
        self.assertIsNone(reply["initialStateIndex"])
        self.assertEqual(reply["resetSemantics"], "seeded_random")
        with self.assertRaisesRegex(ValueError, "development seeds 51-100"):
            self.reset(backend, seed=50, reset="seeded_random")
        with self.assertRaisesRegex(ValueError, "Unknown reset semantics"):
            self.reset(backend, seed=51, reset="random")
        with self.assertRaisesRegex(ValueError, "seed must be an integer"):
            self.reset(backend, seed=0)
        with self.assertRaisesRegex(ValueError, "seed must be an integer"):
            self.reset(backend, seed="51")
        with self.assertRaisesRegex(ValueError, "horizon must be a positive integer"):
            self.reset(backend, seed=51, horizon=0)
        with self.assertRaisesRegex(ValueError, "Unsupported suite"):
            backend.reset({"suite": "libero_90", "task": TASK_NAMES[0], "seed": 51})
        with self.assertRaisesRegex(ValueError, "Task not found"):
            backend.reset({"suite": "libero_goal_swap", "task": "missing_task", "seed": 51})

    def test_reset_with_index_task_and_no_init_states(self):
        backend, _, _, _ = self.make_backend(init_states=None)
        backend.low_placeholder = None
        with patch.object(backend_module, "list_task_names", return_value=TASK_NAMES):
            reply = backend.reset({"suite": "libero_goal_swap", "task": 2, "seed": 3})
        self.assertEqual(reply["task"]["name"], "turn_on_the_stove")
        self.assertEqual(reply["task"]["index"], 2)
        self.assertEqual(reply["initialStateIndex"], 2)
        backend2, _, low2, _ = self.make_backend(init_states=[])
        reply2 = self.reset(backend2, seed=3)
        self.assertIsNone(reply2["initialStateIndex"])
        self.assertEqual(reply2["initialStateSource"], "aspire_modulo_init_state")

    def test_env_rebuilt_when_identity_changes_and_old_env_closed(self):
        backend, code_env, low, factory = self.make_backend()
        self.reset(backend, seed=51)
        second_low = FakeLow(list(range(50)))
        factory.return_value = FakeCodeEnv(second_low)
        reply = self.reset(backend, seed=51, horizon=600)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(low.handle.env.closed, 1)
        self.assertEqual(reply["horizon"], 600)
        self.assertEqual(reply["stepsLeft"], 590)
        self.assertIs(backend.low, second_low)

    # 2. run_program accounting
    def test_run_program_counts_steps_and_enforces_budgets(self):
        backend, code_env, low, _ = self.make_backend()
        self.reset(backend, seed=51)
        program = "for _ in range(30):\n    env.handle.step([0.0] * 8)\nprint('done stepping')\n"
        reply = backend.run_program({"which": "fix", "code": program})
        self.assertEqual(reply["which"], "fix")
        self.assertEqual(reply["steps_before"], 10)
        self.assertEqual(reply["steps_after"], 40)
        self.assertEqual(reply["steps"], 30)
        self.assertIsNone(reply["exception"])
        self.assertFalse(reply["truncated"])
        self.assertIsNone(reply["truncated_reason"])
        self.assertEqual(reply["horizon"], 4000)
        self.assertEqual(reply["stdout_tail"], "done stepping\n")
        self.assertEqual(reply["stderr_tail"], "")
        self.assertEqual(len(reply["sha256"]), 64)
        self.assertIsInstance(reply["wall_s"], float)
        self.assertEqual(set(reply), RUN_REPLY_KEYS)
        self.assertIsNone(reply["exception_class"])
        self.assertEqual(reply["wall_timeout_s"], 5.0)  # the backend default when the request carries none
        self.assertEqual(reply["wall_used_s"], reply["wall_s"])
        self.assertIsNone(reply["wall_budget_s"])
        self.assertIsNone(reply["wall_left_s"])
        self.assertIsNone(reply["infra_probe"])  # clean runs never probe the services

        limited = backend.run_program({"code": program, "max_steps": 12})
        self.assertEqual(limited["which"], "custom")
        self.assertTrue(limited["truncated"])
        self.assertEqual(limited["truncated_reason"], "max_steps")
        self.assertEqual(limited["steps"], 12)
        self.assertEqual(limited["steps_before"], 40)
        self.assertEqual(limited["steps_after"], 52)
        self.assertIsNone(limited["exception"])
        self.assertNotIn("step", vars(low.handle))  # guard removed
        self.assertEqual(low.handle.step.__func__, FakeHandle.step)

    def test_run_program_stops_at_horizon(self):
        backend, _, low, _ = self.make_backend()
        self.reset(backend, seed=51, horizon=20)
        program = "for _ in range(30):\n    env.handle.step([0.0] * 8)\n"
        reply = backend.run_program({"code": program})
        self.assertEqual(reply["steps_before"], 10)
        self.assertEqual(reply["steps_after"], 20)
        self.assertEqual(reply["truncated_reason"], "horizon")
        self.assertTrue(reply["truncated"])
        self.assertIsNone(reply["exception"])
        observed = backend.observe({})
        self.assertEqual(observed["stepsLeft"], 0)
        self.assertEqual(observed["frame"]["libero"]["budget"], {"steps": 20, "horizon": 20, "stepsLeft": 0})

    def test_program_swallowing_interrupt_is_still_truncated(self):
        backend, _, low, _ = self.make_backend()
        self.reset(backend, seed=51)
        program = (
            "for _ in range(5):\n"
            "    try:\n"
            "        env.handle.step([0.0] * 8)\n"
            "    except BaseException:\n"
            "        pass\n"
        )
        reply = backend.run_program({"code": program, "max_steps": 2})
        self.assertEqual(reply["steps"], 2)
        self.assertTrue(reply["truncated"])
        self.assertEqual(reply["truncated_reason"], "max_steps")

    def test_exec_globals_persist_across_runs_and_prelude_code(self):
        backend, code_env, _, _ = self.make_backend()
        self.reset(backend, seed=51)
        backend.run_program({"code": "counter = 7\nRESULT = 'ignored'\n"})
        reply = backend.run_program({"code": "assert counter == 7\n"})
        self.assertIsNone(reply["exception"])
        self.assertEqual(backend_module.PRELUDE_CODE, "open_gripper()\ngoto_home_joint_position()\n")
        code_env._exec_globals["open_gripper"] = Mock()
        code_env._exec_globals["goto_home_joint_position"] = Mock()
        prelude = backend.run_program({"which": "prelude", "code": backend_module.PRELUDE_CODE})
        self.assertEqual(prelude["which"], "prelude")
        self.assertIsNone(prelude["exception"])
        code_env._exec_globals["open_gripper"].assert_called_once_with()
        code_env._exec_globals["goto_home_joint_position"].assert_called_once_with()
        self.reset(backend, seed=52)
        after_reset = backend.run_program({"code": "counter\n"})
        self.assertEqual(after_reset["exception"]["type"], "NameError")

    # 3. program exceptions
    def test_program_exception_is_reported_and_backend_stays_usable(self):
        backend, _, _, _ = self.make_backend()
        self.reset(backend, seed=51)
        program = "env.handle.step([0.0] * 8)\nraise RuntimeError('SAM3 could not localize')\n"
        reply = backend.run_program({"code": program, "which": "evo"})
        self.assertEqual(reply["exception"]["type"], "RuntimeError")
        self.assertEqual(reply["exception"]["message"], "SAM3 could not localize")
        self.assertIn("Traceback", reply["exception"]["traceback_tail"])
        self.assertIn("RuntimeError: SAM3 could not localize", reply["stderr_tail"])
        self.assertFalse(reply["truncated"])
        self.assertIsNone(reply["truncated_reason"])
        self.assertEqual(reply["steps"], 1)
        observed = backend.observe({"view": "agentview"})
        self.assertEqual(observed["steps"], 11)
        self.assertFalse(observed["sealed"])

    def test_run_program_argument_errors(self):
        backend, _, _, _ = self.make_backend()
        with self.assertRaisesRegex(ValueError, "Reset an episode before run_program"):
            backend.run_program({"code": "x = 1\n"})
        self.reset(backend, seed=51)
        with self.assertRaisesRegex(ValueError, "exactly one of path or code"):
            backend.run_program({})
        with self.assertRaisesRegex(ValueError, "exactly one of path or code"):
            backend.run_program({"code": "x = 1\n", "path": "/tmp/does-not-exist.py"})
        with self.assertRaisesRegex(ValueError, "program file not found: /tmp/nope-aspire-program.py"):
            backend.run_program({"path": "/tmp/nope-aspire-program.py"})
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            backend.run_program({"path": "relative/fix.py"})
        with self.assertRaisesRegex(ValueError, "max_steps must be a positive integer"):
            backend.run_program({"code": "x = 1\n", "max_steps": 0})
        with self.assertRaisesRegex(ValueError, "wall_timeout_s must be a positive number"):
            backend.run_program({"code": "x = 1\n", "wall_timeout_s": -1})

    def test_run_program_reads_program_file(self):
        backend, _, _, _ = self.make_backend()
        self.reset(backend, seed=51)
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write("env.handle.step([0.0] * 8)\n")
            path = handle.name
        try:
            reply = backend.run_program({"which": "fix", "path": path})
        finally:
            Path(path).unlink()
        self.assertEqual(reply["steps"], 1)
        import hashlib

        self.assertEqual(reply["sha256"], hashlib.sha256(b"env.handle.step([0.0] * 8)\n").hexdigest())

    # 4. wall timeout
    def test_wall_timeout_interrupts_and_timer_is_cancelled(self):
        backend, _, low, _ = self.make_backend()
        self.reset(backend, seed=51)
        started = time.monotonic()
        reply = backend.run_program({"code": "while True:\n    pass\n", "wall_timeout_s": 0.5})
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0)
        self.assertTrue(reply["truncated"])
        self.assertEqual(reply["truncated_reason"], "wall_timeout")
        self.assertIsNone(reply["exception"])
        self.assertFalse(backend._running)
        fast = backend.run_program({"code": "env.handle.step([0.0] * 8)\n"})
        self.assertFalse(fast["truncated"])
        self.assertIsNone(fast["exception"])
        time.sleep(0.7)  # a leaked timer would fire here (default SIGALRM action kills the process)

    # 5. forbidden identifier scan
    def test_forbidden_scan_is_whole_word(self):
        scan = backend_module.scan_program
        self.assertEqual(scan("if env.task_completed():\n    pass\n"), "task_completed")
        self.assertEqual(scan("d = env.handle.env.sim.data\n"), "sim.data")
        self.assertEqual(scan("env.handle.env.check_success()"), "check_success")
        self.assertEqual(scan("states = env.handle.init_states"), "init_states")
        self.assertIsNone(scan("goal = env.handle.task_language\n"))
        self.assertIsNone(scan("my_init_states_count = 3\nsimulation.data = 1\n"))
        self.assertIsNone(scan("results = compute_rewards_table()\n"))
        backend, _, _, _ = self.make_backend()
        self.reset(backend, seed=51)
        with self.assertRaisesRegex(ValueError, "program uses forbidden identifier: task_completed"):
            backend.run_program({"code": "env.task_completed()\n"})
        with self.assertRaisesRegex(ValueError, "program uses forbidden identifier: sim.data"):
            backend.run_program({"code": "env.handle.env.sim.data\n"})

    # 6. observe payload and montage
    def test_observe_payload_has_no_scoring_keys_and_labels_gripper(self):
        backend, _, low, _ = self.make_backend()
        reply = self.reset(backend, seed=51)
        self.assertFalse(collect_keys(reply) & SCORING_KEYS)
        self.assertEqual(reply["proprio"], {
            "eef_cm": [45.8, 0.0, 36.8], "gripper": "open", "gripper_fraction": 0.97, "gripper_command": "open",
        })
        self.assertEqual(reply["frame"]["libero"]["proprio"], reply["proprio"])
        self.assertNotIn("akita_black_bowl_1_pos", json.dumps(reply))
        self.assertEqual(reply["frame"]["libero"]["budget"], {"steps": 10, "horizon": 4000, "stepsLeft": 3990})
        self.assertEqual(reply["frame"]["libero"]["observationId"], "1:1")
        self.assertEqual(reply["observationId"], "1:1")
        self.assertEqual(reply["frame"]["capturedAt"], "2026-09-09T08:00:00+00:00")
        self.assertEqual(reply["frame"]["view"], "agentview")
        self.assertEqual((reply["frame"]["width"], reply["frame"]["height"]), (32, 32))
        self.assertIsNone(reply["wrist"])

        low._current_obs["robot0_gripper_qpos"] = [0.0008, -0.0008]
        low._gripper_fraction = 0.0
        moving = backend.observe({})
        self.assertEqual(moving["proprio"]["gripper"], "moving")  # 0.97 -> 0.02 changed by > 0.02
        self.assertEqual(moving["proprio"]["gripper_command"], "close")
        closed = backend.observe({})
        self.assertEqual(closed["proprio"]["gripper"], "closed")
        self.assertEqual(closed["proprio"]["gripper_fraction"], 0.02)
        self.assertEqual(closed["observationId"], "1:3")

        low._current_obs["robot0_gripper_qpos"] = [0.016, -0.016]
        backend.observe({})  # absorbs the change
        holding = backend.observe({})
        self.assertEqual(holding["proprio"]["gripper"], "holding")
        self.assertEqual(holding["proprio"]["gripper_fraction"], 0.4)
        low._gripper_fraction = 1.0
        partial = backend.observe({})
        self.assertEqual(partial["proprio"]["gripper"], "partial")
        self.assertEqual(partial["proprio"]["gripper_command"], "open")
        with self.assertRaisesRegex(ValueError, "Unknown view"):
            backend.observe({"view": "wrist"})

    def test_observe_requires_episode(self):
        backend, _, _, _ = self.make_backend()
        with self.assertRaisesRegex(ValueError, "Reset an episode before observing"):
            backend.observe({})
        with self.assertRaisesRegex(ValueError, "Reset an episode before score"):
            backend.score()

    def test_observe_renders_upright_frames_and_montage(self):
        backend, _, low, _ = self.make_backend()
        self.reset(backend, seed=51)
        sim = low.handle.env.sim
        sim.calls.clear()
        reply = backend.observe({"view": "agentview"})
        self.assertEqual(sim.calls, [("agentview", 32, 32, False), ("agentview", 16, 10, False)])
        frame = Image.open(io.BytesIO(base64.b64decode(reply["frame"]["imageB64"])))
        self.assertEqual(frame.size, (32, 32))
        pixels = np.asarray(frame.convert("RGB"))
        self.assertGreater(int(pixels[31, 16, 0]), 200)  # white row flipped to the bottom
        self.assertLess(int(pixels[0, 16, 0]), 60)

        sim.calls.clear()
        montage = backend.observe({"view": "montage", "wrist": True})
        self.assertEqual(sim.calls, [("agentview", 32, 32, False), ("robot0_eye_in_hand", 32, 32, False), ("agentview", 16, 10, False)])
        self.assertEqual((montage["frame"]["width"], montage["frame"]["height"]), (64, 56))
        self.assertEqual(montage["frame"]["view"], "montage")
        self.assertEqual(montage["frame"]["mimeType"], "image/jpeg")
        image = Image.open(io.BytesIO(base64.b64decode(montage["frame"]["imageB64"]))).convert("RGB")
        self.assertEqual(image.size, (64, 56))
        pixels = np.asarray(image)
        self.assertGreater(int(pixels[55, 8, 0]), 200)  # agentview bottom row (white) under the title bar
        self.assertGreater(int(pixels[40, 48, 2]), 150)  # wrist half carries the blue channel
        self.assertLess(int(pixels[40, 48, 0]), 60)
        self.assertEqual(montage["wrist"]["view"], "wrist")
        self.assertEqual((montage["wrist"]["width"], montage["wrist"]["height"]), (32, 32))
        self.assertEqual(montage["wrist"]["capturedAt"], montage["frame"]["capturedAt"])
        self.assertFalse(collect_keys(montage) & SCORING_KEYS)

    # 7. score seals the episode
    def test_score_reads_task_completed_once_and_seals(self):
        backend, _, low, _ = self.make_backend()
        self.reset(backend, seed=51)
        backend.run_program({"code": "env.handle.step([0.0] * 8)\n"})
        scored = backend.score()
        self.assertEqual(scored, {
            "task_completed": True, "steps": 11, "horizon": 4000, "scoredAt": "2026-09-09T08:00:00+00:00",
        })
        self.assertEqual(low.task_completed_calls, 1)
        with self.assertRaisesRegex(ValueError, "score was already called for this episode"):
            backend.score()
        with self.assertRaisesRegex(ValueError, "Episode is sealed by score; reset to continue"):
            backend.run_program({"code": "x = 1\n"})
        self.assertEqual(low.task_completed_calls, 1)
        final = backend.observe({"view": "montage"})
        self.assertTrue(final["sealed"])
        self.assertEqual(final["steps"], 11)
        low.completed = False
        reply = self.reset(backend, seed=52)
        self.assertFalse(reply["sealed"])
        self.assertFalse(backend.sealed)
        self.assertIsNone(backend.scored)
        backend.run_program({"code": "x = 1\n"})
        self.assertEqual(backend.score()["task_completed"], False)
        self.assertEqual(low.task_completed_calls, 2)

    def test_score_is_the_only_success_read(self):
        backend, _, low, _ = self.make_backend()
        self.reset(backend, seed=51)
        backend.observe({"view": "montage", "wrist": True})
        backend.run_program({"code": "for _ in range(3):\n    env.handle.step([0.0] * 8)\n"})
        backend.observe({})
        self.assertEqual(low.task_completed_calls, 0)

    # 8. dispatch and protocol loop
    def test_dispatch_unknown_command_and_error_envelope(self):
        backend, _, _, _ = self.make_backend()
        with self.assertRaisesRegex(ValueError, "Unknown command: bogus"):
            backend.dispatch({"command": "bogus"})
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            envelope = backend_module.handle_request(backend, '{"id": 9, "command": "bogus"}')
        self.assertEqual(envelope, {"id": 9, "ok": False, "error": "ValueError: Unknown command: bogus"})
        self.assertIn("ValueError: Unknown command: bogus", stderr.getvalue())
        with patch("sys.stderr", new_callable=io.StringIO):
            bad_json = backend_module.handle_request(backend, "not json\n")
        self.assertFalse(bad_json["ok"])
        self.assertIsNone(bad_json["id"])
        self.assertTrue(bad_json["error"].startswith("JSONDecodeError"))

    def test_serve_stays_alive_after_errors_and_closes_on_eof(self):
        backend, _, low, _ = self.make_backend()
        lines = [
            '{"id": 1, "command": "bogus"}\n',
            "\n",
            '{"id": 2, "command": "observe"}\n',
            '{"id": 3, "command": "reset", "suite": "libero_goal_swap", "task": "open_the_middle_drawer_of_the_cabinet", "seed": 51}\n',
            '{"id": 4, "command": "score"}\n',
        ]
        output = io.StringIO()
        with patch("sys.stderr", new_callable=io.StringIO):
            backend_module.serve(backend, lines, output)
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([reply["id"] for reply in replies], [1, 2, 3, 4])
        self.assertEqual(replies[0], {"id": 1, "ok": False, "error": "ValueError: Unknown command: bogus"})
        self.assertEqual(replies[1], {"id": 2, "ok": False, "error": "ValueError: Reset an episode before observing"})
        self.assertTrue(replies[2]["ok"])
        self.assertEqual(replies[2]["initialStateIndex"], 0)
        self.assertEqual(replies[3], {
            "id": 4, "ok": True, "task_completed": True, "steps": 10, "horizon": 4000,
            "scoredAt": "2026-09-09T08:00:00+00:00",
        })
        self.assertEqual(low.handle.env.closed, 1)  # EOF closes the env
        self.assertIsNone(backend.code_env)
        self.assertEqual(backend.close(), {"closed": True})  # idempotent

    def test_catalog_lists_suites_and_backend_settings(self):
        backend, _, _, _ = self.make_backend(config="cfg.yaml", api="FakeApi", max_steps=4000, frame_size=32)
        with patch.object(backend_module, "list_task_names", side_effect=lambda suite: [f"{suite}_task"]):
            catalog = backend.catalog()
        self.assertEqual([suite["name"] for suite in catalog["suites"]], list(backend_module.SUITES))
        self.assertEqual(catalog["suites"][0]["family"], "pro")
        self.assertEqual(catalog["suites"][-1]["family"], "long")
        self.assertEqual(catalog["suites"][0]["tasks"], [{"index": 0, "name": "libero_goal_swap_task"}])
        self.assertEqual(catalog["backend"]["config"], "cfg.yaml")
        self.assertEqual(catalog["backend"]["api"], "FakeApi")
        self.assertEqual(catalog["backend"]["maxSteps"], 4000)
        self.assertEqual(catalog["backend"]["frameSize"], 32)
        self.assertNotIn("language", json.dumps(catalog["suites"]))

    def test_classify_result_handles_module_prefixed_interrupt_and_chained_errors(self):
        classify = backend_module.classify_result
        interrupted = classify({"ok": False, "stdout": "", "stderr": "Traceback...\naspire_backend.ProgramInterrupt: max_steps\n"}, 52, 4000, "max_steps", 2000)
        self.assertEqual(interrupted["truncated_reason"], "max_steps")
        self.assertIsNone(interrupted["exception"])
        dotted = classify({"ok": False, "stdout": "", "stderr": "Traceback...\nrequests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8114)\n"}, 20, 4000, None, 2000)
        self.assertEqual(dotted["exception"]["type"], "ConnectionError")
        self.assertEqual(dotted["exception"]["message"], "HTTPConnectionPool(host='127.0.0.1', port=8114)")
        self.assertFalse(dotted["truncated"])
        at_horizon = classify({"ok": True, "stdout": "x" * 5000, "stderr": ""}, 4000, 4000, None, 100)
        self.assertEqual(at_horizon["truncated_reason"], "horizon")
        self.assertIsNone(at_horizon["exception"])
        self.assertEqual(len(at_horizon["stdout_tail"]), 100)
        bare = classify({"ok": False, "stdout": "", "stderr": "Traceback...\nKeyboardInterrupt\n"}, 20, 4000, None, 2000)
        self.assertEqual(bare["exception"], {"type": "KeyboardInterrupt", "message": "", "traceback_tail": "Traceback...\nKeyboardInterrupt\n", "class": "program"})
        self.assertEqual(bare["exception_class"], "program")
        self.assertEqual(dotted["exception"]["class"], "infra")
        self.assertEqual(dotted["exception_class"], "infra")
        self.assertEqual(interrupted["exception_class"], None)  # step budgets are not exceptions
        wall = classify({"ok": False, "stdout": "", "stderr": "Traceback...\nProgramInterrupt: wall_timeout\n"}, 52, 4000, "wall_timeout", 2000)
        self.assertEqual((wall["truncated_reason"], wall["exception_class"], wall["exception"]), ("wall_timeout", "timeout", None))
        swallowed = classify({"ok": True, "stdout": "", "stderr": ""}, 60, 4000, "wall_timeout", 2000)
        self.assertEqual((swallowed["truncated_reason"], swallowed["exception_class"]), ("wall_timeout", "timeout"))

    def test_classify_result_treats_systemexit_zero_as_a_clean_stop(self):
        # LLM-written programs end a successful grasp with `raise SystemExit(0)`; that is the program stopping, not failing,
        # so recover-on-exception must not re-run another program over the completed state (BEHAVIOR held-out incident).
        classify = backend_module.classify_result
        for tail in ("SystemExit: 0", "SystemExit", "SystemExit: None"):
            clean = classify({"ok": False, "stdout": "grasped", "stderr": f"Traceback...\n  File \"<string>\", line 186, in <module>\n{tail}\n"}, 3801, 100000, None, 2000)
            self.assertIsNone(clean["exception"], tail)
            self.assertIsNone(clean["exception_class"], tail)
            self.assertFalse(clean["truncated"], tail)
            self.assertEqual(clean["program_exit"], "SystemExit:" + (tail.partition(":")[2].strip() or "0"), tail)
        failed = classify({"ok": False, "stdout": "", "stderr": "Traceback...\nSystemExit: 1\n"}, 20, 100000, None, 2000)
        self.assertEqual((failed["exception"]["type"], failed["exception_class"], failed["program_exit"]), ("SystemExit", "program", None))
        ordinary = classify({"ok": False, "stdout": "", "stderr": "Traceback...\nRuntimeError: Failed to grasp the red radio\n"}, 20, 100000, None, 2000)
        self.assertEqual((ordinary["exception"]["type"], ordinary["program_exit"]), ("RuntimeError", None))

    def test_compact_proprio_matches_shared_listing(self):
        proprio = backend_module.compact_proprio(
            {"robot0_eef_pos": [0.1234, -0.04, 0.215], "robot0_gripper_qpos": [0.05, -0.05], "plate_1_pos": [1, 2, 3]},
            False, None,
        )
        self.assertEqual(proprio, {"eef_cm": [12.3, -4.0, 21.5], "gripper": "open", "gripper_fraction": 1.0, "gripper_command": "open"})
        self.assertEqual(backend_module.gripper_state(True, 0.4, 0.41), "holding")
        self.assertEqual(backend_module.gripper_state(True, 0.4, 0.5), "moving")
        self.assertEqual(backend_module.gripper_state(False, 0.4, None), "partial")


    # 9. wall-timeout robustness: boundary interrupts, re-armed alarm, protocol-loop fallback
    def test_handle_request_recovers_from_a_stray_program_interrupt(self):
        backend, _, low, _ = self.make_backend()
        self.reset(backend, seed=51)
        before = signal.getsignal(signal.SIGALRM)
        # Simulate a run whose teardown never happened: guard installed, handler installed, timer live.
        low.handle.step = lambda action: None
        backend._program_handle = low.handle
        backend._previous_alarm = signal.signal(signal.SIGALRM, backend._alarm)
        backend._alarm_installed = True
        backend._running = True
        signal.setitimer(signal.ITIMER_REAL, 30.0)
        try:
            with patch.object(backend, "dispatch", side_effect=backend_module.ProgramInterrupt("wall_timeout")):
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    envelope = backend_module.handle_request(backend, '{"id": 4, "command": "observe"}')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, before)
        self.assertEqual(envelope, {"id": 4, "ok": False, "error": "ProgramInterrupt: wall_timeout"})
        self.assertIn("ProgramInterrupt: wall_timeout", stderr.getvalue())
        self.assertFalse(backend._running)
        self.assertFalse(backend._alarm_installed)
        self.assertNotIn("step", vars(low.handle))
        # The backend still serves requests afterwards.
        self.assertEqual(backend.observe({})["steps"], 10)

    def test_boundary_interrupt_during_teardown_keeps_the_result_and_the_teardown(self):
        backend, _, low, _ = self.make_backend()
        self.reset(backend, seed=51)
        before = signal.getsignal(signal.SIGALRM)
        real_setitimer = signal.setitimer
        fired = []

        def flaky_setitimer(which, seconds, interval=0.0):
            # The cancel call of the first teardown "receives" the alarm: the interrupt escapes the
            # finally block and run_program must finish the teardown a second time.
            if seconds == 0 and not fired:
                fired.append(True)
                raise backend_module.ProgramInterrupt("wall_timeout")
            return real_setitimer(which, seconds, interval)

        program = "for _ in range(3):\n    env.handle.step([0.0] * 8)\nprint('ran')\n"
        with patch.object(backend_module.signal, "setitimer", side_effect=flaky_setitimer):
            reply = backend.run_program({"code": program, "wall_timeout_s": 5.0})
        self.assertEqual(fired, [True])
        self.assertEqual(reply["steps"], 3)
        self.assertEqual(reply["stdout_tail"], "ran\n")
        self.assertIsNone(reply["exception"])
        self.assertFalse(reply["truncated"])
        self.assertFalse(backend._running)
        self.assertFalse(backend._alarm_installed)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        self.assertEqual(signal.getsignal(signal.SIGALRM), before)
        self.assertNotIn("step", vars(low.handle))
        again = backend.run_program({"code": "env.handle.step([0.0] * 8)\n"})
        self.assertEqual(again["steps"], 1)

    def test_interrupt_escaping_exec_user_code_is_reported_as_timeout(self):
        backend, code_env, _, _ = self.make_backend()
        self.reset(backend, seed=51)
        original = code_env._exec_user_code

        def exec_then_interrupt(code):
            original(code)
            raise backend_module.ProgramInterrupt("wall_timeout")

        with patch.object(code_env, "_exec_user_code", side_effect=exec_then_interrupt):
            reply = backend.run_program({"code": "env.handle.step([0.0] * 8)\n"})
        self.assertEqual(reply["steps"], 1)
        self.assertTrue(reply["truncated"])
        self.assertEqual(reply["truncated_reason"], "wall_timeout")
        self.assertEqual(reply["exception_class"], "timeout")
        self.assertIsNone(reply["exception"])
        self.assertFalse(backend._running)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_wall_timeout_is_rearmed_and_stops_a_program_that_swallows_it(self):
        backend, _, low, _ = self.make_backend()
        self.reset(backend, seed=51)
        armed = backend.run_program({"code": "import signal as s\nprint(s.getitimer(s.ITIMER_REAL)[1])\n"})
        self.assertEqual(armed["stdout_tail"], f"{backend_module.WALL_TIMEOUT_REARM_S}\n")
        program = (
            "import time\n"
            "for _ in range(3000):\n"
            "    try:\n"
            "        env.handle.step([0.0] * 8)\n"
            "        time.sleep(0.005)\n"
            "    except BaseException:\n"
            "        pass\n"
        )
        started = time.monotonic()
        reply = backend.run_program({"code": program, "wall_timeout_s": 0.3})
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 3.0)
        self.assertTrue(reply["truncated"])
        self.assertEqual(reply["truncated_reason"], "wall_timeout")
        self.assertEqual(reply["exception_class"], "timeout")
        self.assertIsNone(reply["exception"])
        self.assertLess(reply["steps"], 500)  # no sim progress once the deadline passed (guard re-raises)
        self.assertEqual(reply["wall_timeout_s"], 0.3)
        self.assertFalse(backend._running)
        self.assertFalse(backend._alarm_installed)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        self.assertNotIn("step", vars(low.handle))
        time.sleep(1.2)  # a leaked re-armed timer would fire here

    # 10. forbidden identifiers: env internals and interpreter escape hatches
    def test_forbidden_scan_covers_env_internals_and_introspection(self):
        scan = backend_module.scan_program
        self.assertEqual(scan('pos = env._current_obs["akita_black_bowl_1_pos"]\n'), "_current_obs")
        self.assertEqual(scan("info = env._current_info\n"), "_current_info")
        self.assertEqual(scan("env.handle.env._check_success()\n"), "_check_success")
        self.assertEqual(scan("inner = env.handle.env\n"), "handle.env")
        self.assertEqual(scan('getattr(env, "task_" + "completed")()\n'), "getattr")
        self.assertEqual(scan('__import__("os").system("true")\n'), "__import__")
        self.assertEqual(scan('eval("env.task_" + "completed()")\n'), "eval(")
        self.assertEqual(scan('eval ("1")\n'), "eval(")
        self.assertEqual(scan('exec("x = 1")\n'), "exec(")
        self.assertEqual(scan('globals()["env"]\n'), "globals(")
        self.assertEqual(scan("vars(env)\n"), "vars(")
        self.assertIsNone(scan("evaluate(x)\nmy_getattr_cache = {}\nexecute_plan()\nglobals_count = 1\nvars_seen = []\n"))
        self.assertIsNone(scan('rgb = obs["agentview_image"]\nlanguage = env.handle.task_language\n'))
        self.assertIsNone(scan('depth = obs["agentview_depth"]\nprint(depth.ndim if depth.ndim == 3 else 2)\n'))
        backend, _, _, _ = self.make_backend()
        self.reset(backend, seed=51)
        for code, name in (('getattr(env, "x")\n', "getattr"), ("env.handle.env\n", "handle.env"), ("env._current_obs\n", "_current_obs")):
            with self.assertRaisesRegex(ValueError, f"program uses forbidden identifier: {name}"):
                backend.run_program({"code": code})

    # 11. audited program pin (allowed_sha256) and phases
    def test_allowed_sha256_pin_and_heldout_phase(self):
        backend, _, _, _ = self.make_backend()
        program = "env.handle.step([0.0] * 8)\n"
        digest = hashlib.sha256(program.encode("utf-8")).hexdigest()
        other = hashlib.sha256(b"x = 1\n").hexdigest()
        with self.assertRaisesRegex(ValueError, "heldout phase requires seeds 1-50 with aspire reset semantics"):
            self.reset(backend, seed=51, phase="heldout")
        with self.assertRaisesRegex(ValueError, "development phase requires seeds 51-100"):
            self.reset(backend, seed=7, phase="development")
        with self.assertRaisesRegex(ValueError, "Unknown phase: staging"):
            self.reset(backend, seed=51, phase="staging")
        reply = self.reset(backend, seed=7, phase="heldout")
        self.assertEqual(reply["phase"], "heldout")
        self.assertIsNone(reply["wallBudgetS"])
        with self.assertRaisesRegex(ValueError, "heldout phase requires allowed_sha256 on every run_program"):
            backend.run_program({"code": program})
        with self.assertRaisesRegex(ValueError, "allowed_sha256 must be a non-empty list"):
            backend.run_program({"code": program, "allowed_sha256": []})
        with self.assertRaisesRegex(ValueError, "allowed_sha256 must be a non-empty list"):
            backend.run_program({"code": program, "allowed_sha256": ["deadbeef"]})
        with self.assertRaisesRegex(ValueError, f"program sha256 {digest} is not in allowed_sha256"):
            backend.run_program({"code": program, "allowed_sha256": [other]})
        pinned = backend.run_program({"code": program, "allowed_sha256": [other, digest.upper()]})
        self.assertEqual(pinned["sha256"], digest)
        self.assertEqual(pinned["steps"], 1)
        prelude = backend.run_program({"which": "prelude", "code": backend_module.PRELUDE_CODE, "allowed_sha256": [other]})
        self.assertEqual(prelude["exception"]["type"], "NameError")  # ran (fake env has no skills); the pin let it through
        # Development (no phase): the pin is enforced only when the request carries a list.
        reply = self.reset(backend, seed=51)
        self.assertIsNone(reply["phase"])
        self.assertEqual(backend.run_program({"code": program})["steps"], 1)
        with self.assertRaisesRegex(ValueError, "is not in allowed_sha256"):
            backend.run_program({"code": program, "allowed_sha256": [other]})
        self.assertEqual(self.reset(backend, seed=51, phase="development", reset="seeded_random")["phase"], "development")
        with self.assertRaisesRegex(ValueError, "development phase requires seeded_random"):
            self.reset(backend, seed=51, phase="development")

    # 12. judge frame
    def test_observe_returns_an_aspect_preserving_judge_frame(self):
        backend, _, low, _ = self.make_backend()
        reply = self.reset(backend, seed=51)
        judge = reply["frame_judge"]
        self.assertEqual((judge["width"], judge["height"], judge["view"], judge["camera"]), (16, 10, "judge", "agentview"))
        self.assertEqual(judge["mimeType"], "image/jpeg")
        self.assertEqual(judge["capturedAt"], reply["frame"]["capturedAt"])
        self.assertEqual(judge["libero"], reply["frame"]["libero"])
        self.assertIsNot(judge["libero"], reply["frame"]["libero"])
        image = Image.open(io.BytesIO(base64.b64decode(judge["imageB64"]))).convert("RGB")
        self.assertEqual(image.size, (16, 10))
        pixels = np.asarray(image)
        self.assertGreater(int(pixels[9, 8, 0]), 200)  # flipped upright like the montage frame
        self.assertLess(int(pixels[0, 8, 0]), 60)
        self.assertFalse(collect_keys(reply) & SCORING_KEYS)
        sim = low.handle.env.sim
        sim.calls.clear()
        without = backend.observe({"judge_frame": False})
        self.assertIsNone(without["frame_judge"])
        self.assertEqual(sim.calls, [("agentview", 32, 32, False)])
        self.assertEqual(backend.catalog.__self__.judge_frame_size, (16, 10))
        with patch.object(backend_module, "list_task_names", side_effect=lambda suite: [f"{suite}_task"]):
            self.assertEqual(backend.catalog()["backend"]["judgeFrameSize"], {"width": 16, "height": 10})
        parse = backend_module.parse_frame_size
        self.assertEqual(parse("512x320"), (512, 320))
        self.assertEqual(parse(" 800X512 "), (800, 512))
        self.assertEqual(parse((640, 400)), (640, 400))
        for bad in ("512", "0x10", "axb", "512x-1"):
            with self.assertRaises(ValueError):
                parse(bad)
        self.assertEqual(backend_module.default_args().judge_frame_size, "512x320")
        default = backend_module.AspireBackend(backend_module.default_args(), env_factory=Mock(), task_lookup=fake_task_lookup)
        self.assertEqual(default.judge_frame_size, (512, 320))
        with self.assertRaisesRegex(ValueError, "WxH"):
            self.make_backend(judge_frame_size="wide")

    # 13. exception classes: program vs infra vs timeout
    def test_exception_classes_and_service_probe(self):
        classify = backend_module.classify_exception
        self.assertEqual(classify("RuntimeError", "Failed to communicate with SAM3 service at http://127.0.0.1:8114: boom"), "infra")
        self.assertEqual(classify("RuntimeError", "Request to http://127.0.0.1:8116/ik failed after 5 retries / 15.00s. Last error: x"), "infra")
        self.assertEqual(classify("ConnectionError", "HTTPConnectionPool(host='127.0.0.1', port=8115)"), "infra")
        self.assertEqual(classify("ReadTimeout", ""), "infra")
        self.assertEqual(classify("RuntimeError", "CUDA out of memory. Tried to allocate 2 GiB"), "infra")
        self.assertEqual(classify("RuntimeError", "SAM3 could not localize 'bowl'"), "program")
        self.assertEqual(classify("RuntimeError", "IK failed for position [0.1, 0.2, 0.3] with all orientation fallbacks"), "program")
        self.assertEqual(classify("ValueError", "Could not get object mask for 'plate'"), "program")
        self.assertEqual(classify("KeyError", "'legless'"), "program")

        probe = Mock(return_value={"8114": True, "8115": True, "8116": True})
        backend, _, _, _ = self.make_backend(service_probe=probe)
        self.reset(backend, seed=51)
        clean = backend.run_program({"code": "env.handle.step([0.0] * 8)\n"})
        self.assertIsNone(clean["infra_probe"])
        probe.assert_not_called()
        infra = backend.run_program({"code": "raise RuntimeError('Failed to communicate with GraspNet service at http://127.0.0.1:8115: HTTPConnectionPool')\n"})
        self.assertEqual(infra["exception"]["class"], "infra")
        self.assertEqual(infra["exception_class"], "infra")
        self.assertEqual(infra["infra_probe"], {"8114": True, "8115": True, "8116": True})
        self.assertNotIn("infra_reason", infra["exception"])
        program = backend.run_program({"code": "raise RuntimeError('SAM3 could not localize')\n"})
        self.assertEqual((program["exception"]["class"], program["exception_class"]), ("program", "program"))
        self.assertEqual(probe.call_count, 2)

        probe.return_value = {"8114": True, "8115": True, "8116": False}
        ik = backend.run_program({"code": "raise RuntimeError('IK failed for position [0.1] with all orientation fallbacks')\n"})
        self.assertEqual(ik["exception"]["class"], "infra")
        self.assertEqual(ik["exception"]["infra_reason"], "service port(s) not listening: 8116")
        self.assertEqual(ik["exception_class"], "infra")
        self.assertEqual(ik["exception"]["type"], "RuntimeError")
        timeout = backend.run_program({"code": "while True:\n    pass\n", "wall_timeout_s": 0.3})
        self.assertEqual(timeout["truncated_reason"], "wall_timeout")
        self.assertEqual(timeout["exception_class"], "infra")  # a service was down while the run timed out
        self.assertIsNone(timeout["exception"])
        probe.return_value = {"8114": True, "8115": True, "8116": True}
        timeout = backend.run_program({"code": "while True:\n    pass\n", "wall_timeout_s": 0.3})
        self.assertEqual(timeout["exception_class"], "timeout")

        probe.side_effect = OSError("probe broke")
        with patch("sys.stderr", new_callable=io.StringIO):
            broken = backend.run_program({"code": "raise ValueError('nope')\n"})
        self.assertEqual(broken["infra_probe"], {})
        self.assertEqual(broken["exception_class"], "program")
        self.assertEqual(backend_module.parse_ports("8114, 8115,8116"), [8114, 8115, 8116])
        self.assertEqual(backend_module.parse_ports(""), [])
        with self.assertRaises(ValueError):
            backend_module.parse_ports("70000")
        self.assertEqual(backend_module.probe_ports([]), {})
        self.assertEqual(backend_module.default_args().infra_probe_ports, "8114,8115,8116")

    # 14. per-request wall timeout and episode wall budget
    def test_wall_budget_clamps_the_per_request_timeout_and_exhausts(self):
        ticks = iter([0.0, 0.6, 1.0, 1.6, 2.0, 2.05, 3.0, 3.5])
        backend, _, low, _ = self.make_backend(clock=lambda: next(ticks))
        with self.assertRaisesRegex(ValueError, "wall_budget_s must be a positive number"):
            self.reset(backend, seed=51, wall_budget_s=0)
        reply = self.reset(backend, seed=51, wall_budget_s=1.0)
        self.assertEqual(reply["wallBudgetS"], 1.0)
        first = backend.run_program({"code": "env.handle.step([0.0] * 8)\n", "wall_timeout_s": 5.0})
        self.assertEqual((first["wall_timeout_s"], first["wall_s"], first["wall_used_s"], first["wall_budget_s"], first["wall_left_s"]), (1.0, 0.6, 0.6, 1.0, 0.4))
        self.assertFalse(first["truncated"])
        second = backend.run_program({"code": "env.handle.step([0.0] * 8)\n", "wall_timeout_s": 5.0})
        self.assertEqual((second["wall_timeout_s"], second["wall_used_s"], second["wall_left_s"]), (0.4, 1.2, 0.0))
        self.assertEqual(low._sim_step_count, 12)
        third = backend.run_program({"code": "env.handle.step([0.0] * 8)\n", "wall_timeout_s": 5.0})
        self.assertEqual(set(third), RUN_REPLY_KEYS)
        self.assertEqual((third["steps"], third["steps_before"], third["steps_after"], third["wall_s"]), (0, 12, 12, 0.0))
        self.assertTrue(third["truncated"])
        self.assertEqual(third["truncated_reason"], "wall_budget")
        self.assertEqual(third["exception_class"], "timeout")
        self.assertIsNone(third["exception"])
        self.assertEqual((third["wall_timeout_s"], third["wall_used_s"], third["wall_left_s"]), (5.0, 1.2, 0.0))
        self.assertIsNone(third["infra_probe"])
        self.assertEqual(low._sim_step_count, 12)  # nothing ran
        self.assertNotIn("step", vars(low.handle))
        self.assertFalse(backend._running)
        # Per-request timeouts are honoured verbatim without a budget, and reset clears the accounting.
        reply = self.reset(backend, seed=52)
        self.assertIsNone(reply["wallBudgetS"])
        self.assertEqual(backend.wall_used_s, 0.0)
        plain = backend.run_program({"code": "x = 1\n", "wall_timeout_s": 7.5})
        self.assertEqual((plain["wall_timeout_s"], plain["wall_budget_s"], plain["wall_left_s"], plain["wall_used_s"]), (7.5, None, None, 0.05))


if __name__ == "__main__":
    unittest.main()
