import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("robotwin_code_worker", Path(__file__).resolve().parents[1] / "scripts/robotwin_code_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class CodeWorkerTest(unittest.TestCase):
    def test_program_computes_target_from_supplied_observation(self):
        result = worker.execute_program(
            "target = np.array(observation['state']['position']) + [0, 0, 0.05]\n"
            "move_gripper('left', target, [1, 0, 0, 0], 0)\nprint(instruction)",
            {"state": {"position": [0.1, -0.2, 0.8]}}, "lift",
        )
        self.assertAlmostEqual(result["action"]["position"][2], 0.85)
        self.assertEqual(result["stdout"], "lift\n")

    def test_program_cannot_hide_multiple_moves_in_one_decision(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            worker.execute_program(
                "for height in (0.8, 0.9):\n    move_gripper('left', [0, 0, height], [1, 0, 0, 0], 1)",
                {}, "lift",
            )


if __name__ == "__main__":
    unittest.main()
