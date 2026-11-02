import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("aloha_code_worker", Path(__file__).resolve().parents[1] / "scripts/aloha_code_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class AlohaCodeWorkerTests(unittest.TestCase):
    def test_program_changes_selected_joints_and_preserves_the_other_arm(self):
        qpos = [.1] * 6 + [.4] + [.2] * 6 + [.6]
        result = worker.execute_program(
            "target = np.array(observation['state']['qpos'])\n"
            "target[0] += 0.05\ntarget[6] = 0.8\njoint_position(target)\nprint(instruction)",
            {"state": {"qpos": qpos}}, "Open left gripper.",
        )
        self.assertEqual(result["action"]["name"], "joint_position")
        target = result["action"]["arguments"]["positions"]
        self.assertAlmostEqual(target[0], .15)
        self.assertEqual(target[6], .8)
        self.assertEqual(target[7:], qpos[7:])
        self.assertEqual(result["stdout"], "Open left gripper.\n")
        self.assertEqual(qpos[0], .1)

    def test_program_can_compute_a_bounded_chunk(self):
        result = worker.execute_program(
            "start = np.array(observation['state']['qpos'], dtype=float)\n"
            "target = start.copy()\ntarget[7] = .2\n"
            "joint_trajectory(np.linspace(start, target, 4))", {"state": {"qpos": [0] * 14}}, "Move.",
        )
        self.assertEqual(result["action"]["name"], "joint_trajectory")
        positions = result["action"]["arguments"]["positions"]
        self.assertEqual(len(positions), 4)
        self.assertEqual(positions[0], [0] * 14)
        self.assertEqual(positions[-1][7], .2)

    def test_each_decision_returns_one_command(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            worker.execute_program("joint_position([0] * 14)\njoint_trajectory([[0] * 14])", {}, "Hold.")

    def test_output_uses_the_aloha_joint14_contract(self):
        for source in ("joint_position([0] * 7)", "joint_trajectory([[0] * 14] * 101)"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                worker.execute_program(source, {}, "Hold.")


if __name__ == "__main__":
    unittest.main()
