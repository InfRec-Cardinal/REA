import unittest

from agent_vla_async.directives import (
    advance_forward_steps,
    gate_observation,
    parse_policy_action,
)


def observation(*, depth_center=1.0, side=1.0, tilt=0.0):
    return {
        "state": {"usable": True, "ready": True, "rpy": [tilt, 0.0, 0.0]},
        "depth": {"usable": True, "center_m": depth_center,
                  "left_m": side, "right_m": 1.0},
        "lidar": {"usable": True, "center_m": 1.5,
                  "left_m": 1.0, "right_m": 1.0},
    }


class DirectivePureTests(unittest.TestCase):
    def test_parses_canonical_policy_action(self):
        directive = parse_policy_action({
            "type": "policy_action", "seq": 7,
            "captured_at": 10.0, "published_at": 11.0,
            "action": "FORWARD", "duration_ms": 1000,
        }, 4.5)
        self.assertEqual((directive.seq, directive.action, directive.duration_ms),
                         (7, "FORWARD", 1000))
        self.assertEqual(directive.received_at, 4.5)

    def test_gate_hard_stops_and_narrow_only_allows_forward(self):
        self.assertTrue(gate_observation(observation(depth_center=0.4), "FORWARD").hard_stop)
        self.assertTrue(gate_observation(observation(tilt=0.4), "FORWARD").hard_stop)
        missing_sensor = observation()
        missing_sensor["depth"]["usable"] = False
        self.assertTrue(gate_observation(missing_sensor, "FORWARD").hard_stop)
        narrow_forward = gate_observation(observation(side=0.4), "FORWARD")
        self.assertTrue(narrow_forward.allowed)
        self.assertTrue(narrow_forward.narrow)
        narrow_turn = gate_observation(observation(side=0.4), "TURN_LEFT")
        self.assertFalse(narrow_turn.allowed)
        self.assertFalse(narrow_turn.hard_stop)
        self.assertTrue(gate_observation(observation(side=0.3), "FORWARD").hard_stop)

    def test_only_forward_accumulates_twenty_steps(self):
        steps, credit = 0, 0.0
        for _ in range(67):
            steps, credit = advance_forward_steps(steps, credit, "FORWARD", 0.18, 0.6)
        steps, credit = advance_forward_steps(steps, credit, "TURN_RIGHT", 1.0, 0.6)
        self.assertEqual(steps, 20)
        self.assertAlmostEqual(credit, 0.06, places=6)


if __name__ == "__main__":
    unittest.main()
