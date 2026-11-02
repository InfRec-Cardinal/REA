"""Run with the LIBERO Python runtime and greenlet on PYTHONPATH."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

if importlib.util.find_spec("greenlet") is None:
    raise unittest.SkipTest("LIBERO continuation tests require optional greenlet")

source = Path(__file__).resolve().parents[1] / "scripts" / "async_libero_backend.py"
spec = importlib.util.spec_from_file_location("async_libero_test", source)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class ContinuationTest(unittest.TestCase):
    def test_program_stack_is_preserved_across_real_control_boundaries(self):
        low = SimpleNamespace(_sim_step_count=0, handle=SimpleNamespace())
        actions = []
        def step(action):
            actions.append(action)
            low._sim_step_count += 1
        low.handle.step = step
        def execute():
            for i in range(7):
                low.handle.step(i)
            return {"ok": True}
        continuation = backend.ChunkedProgram(low, execute, 20)
        _, done = continuation.advance(3)
        self.assertFalse(done)
        self.assertEqual(actions, [0, 1, 2])
        _, done = continuation.advance(3)
        self.assertFalse(done)
        result, done = continuation.advance(3)
        self.assertTrue(done)
        self.assertTrue(result["ok"])
        self.assertEqual(actions, list(range(7)))
        self.assertIs(low.handle.step, step)

    def test_stopping_does_not_execute_pending_control(self):
        low = SimpleNamespace(_sim_step_count=0, handle=SimpleNamespace())
        def step(_):
            low._sim_step_count += 1
        low.handle.step = step
        def execute():
            for i in range(7):
                low.handle.step(i)
        continuation = backend.ChunkedProgram(low, execute, 20)
        continuation.advance(2)
        continuation.close()
        self.assertEqual(low._sim_step_count, 2)
        self.assertIs(low.handle.step, step)


if __name__ == "__main__":
    unittest.main()
