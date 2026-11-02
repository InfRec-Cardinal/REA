import argparse
import asyncio
import threading
import time
import unittest

from agent_vla_async.physical_steps_mission import (
    GaitEvent,
    GaitPhaseCounter,
    PhysicalStepsMission,
    continuous_gate_observation,
    local_avoidance_yaw,
)


class FakeSnapshot(dict):
    def to_dict(self):
        return dict(self)


class FakeNav:
    def observe(self):
        return FakeSnapshot({
            "state": {"usable": True, "ready": True, "rpy": [0.0, 0.0, 0.0]},
            "depth": {"usable": True, "center_m": 2.0, "left_m": 2.0, "right_m": 2.0},
            "lidar": {"usable": True, "center_m": 2.0, "left_m": 2.0, "right_m": 2.0},
        })


class FakeGait:
    def __init__(self):
        self.ready = threading.Event()
        self.ready.set()
        self.error = None
        self.baseline = 0.0
        self.moving = threading.Event()
        self._events = []
        self._count = 0

    @property
    def count(self):
        return self._count

    def start(self):
        return None

    def close(self):
        return None

    def push(self):
        self._count += 1
        self._events.append(GaitEvent(self._count, 1 if self._count % 2 else -1, time.monotonic(), 0.1))

    def drain(self):
        events, self._events = self._events, []
        return events


class FakeConsole:
    execute = False

    def __init__(self, gait, drives_per_step=1):
        self.gait = gait
        self.drives_per_step = drives_per_step
        self.drives = 0
        self.stops = 0

    async def drive(self, vx, yaw, duration):
        self.drives += 1
        if self.drives % self.drives_per_step == 0:
            self.gait.push()
        return {"ok": True}

    async def stop(self):
        self.stops += 1
        return {"ok": True}


class GaitPhaseCounterTests(unittest.TestCase):
    def test_counts_twenty_alternating_large_phases(self):
        counter = GaitPhaseCounter(0.0)
        now = 0.0
        events = []
        for step in range(20):
            value = 0.12 if step % 2 == 0 else -0.12
            for _ in range(6):
                now += 0.05
                event = counter.update(value, now, True)
                if event is not None:
                    events.append(event)
            for _ in range(3):
                now += 0.05
                counter.update(0.0, now, True)
        self.assertEqual(20, counter.count)
        self.assertEqual(list(range(1, 21)), [event.count for event in events])

    def test_counts_forty_alternating_large_phases(self):
        counter = GaitPhaseCounter(0.0)
        now = 0.0
        events = []
        for step in range(40):
            value = 0.12 if step % 2 == 0 else -0.12
            for _ in range(6):
                now += 0.05
                event = counter.update(value, now, True)
                if event is not None:
                    events.append(event)
            for _ in range(3):
                now += 0.05
                counter.update(0.0, now, True)
        self.assertEqual(40, counter.count)
        self.assertEqual(list(range(1, 41)), [event.count for event in events])

    def test_does_not_count_static_or_same_side_twice(self):
        counter = GaitPhaseCounter(-0.02)
        for index in range(20):
            self.assertIsNone(counter.update(-0.02, index * 0.05, False))
        for index in range(20):
            counter.update(0.10, 2.0 + index * 0.05, True)
        self.assertEqual(1, counter.count)

    def test_baseline_follows_settling_pose_before_motion(self):
        counter = GaitPhaseCounter(0.08)
        now = 0.0
        for _ in range(80):
            now += 0.05
            counter.update(-0.02, now, False)
        self.assertAlmostEqual(-0.02, counter.baseline, places=4)
        events = []
        for value in ([0.12] * 6 + [0.0] * 3 + [-0.12] * 6):
            now += 0.05
            event = counter.update(value, now, True)
            if event is not None:
                events.append(event)
        self.assertEqual(2, len(events))


class AvoidanceTests(unittest.TestCase):
    def test_transient_state_reader_failure_reuses_recent_valid_state(self):
        gait = FakeGait()
        mission = PhysicalStepsMission(
            FakeNav(), FakeConsole(gait), gait,
            argparse.Namespace(
                auto_forward=True, no_lidar=True, state_grace_s=2.2,
                sensor_grace_s=0.6,
            ),
        )
        mission.observation = FakeNav().observe()
        mission.effective_observation()
        mission.observation = dict(mission.observation)
        mission.observation["state"] = {
            "usable": False, "ready": False, "reason": "control_failed",
        }
        effective = mission.effective_observation()
        self.assertTrue(effective["state"]["usable"])
        self.assertTrue(continuous_gate_observation(effective, require_lidar=False).allowed)

    def test_optional_lidar_allows_usable_depth_only(self):
        observation = {
            "state": {"usable": True, "ready": True, "rpy": [0.0, 0.0, 0.0]},
            "depth": {"usable": True, "center_m": 2.0, "left_m": 2.0, "right_m": 2.0},
            "lidar": {"usable": False, "reason": "snapshot_missing"},
        }
        self.assertTrue(
            continuous_gate_observation(observation, require_lidar=False).allowed
        )

    def test_depth_direction_wins_when_lidar_side_labels_disagree(self):
        observation = {
            "depth": {"left_m": 0.44, "right_m": 1.8},
            "lidar": {"left_m": 1.2, "right_m": 0.35},
        }
        self.assertLess(local_avoidance_yaw(observation), 0.0)

    def test_midrange_center_steers_instead_of_stopping(self):
        observation = {
            "state": {"usable": True, "ready": True, "rpy": [0.0, 0.0, 0.0]},
            "depth": {
                "usable": True, "center_m": 0.72, "left_m": 0.44, "right_m": 1.8,
            },
            "lidar": {
                "usable": True, "center_m": 0.70, "left_m": 1.2, "right_m": 0.35,
            },
        }
        decision = continuous_gate_observation(observation)
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.hard_stop)
        self.assertTrue(decision.narrow)


class MissionIntegrationTests(unittest.TestCase):
    def test_auto_forward_finishes_twenty_physical_events_without_stdin(self):
        gait = FakeGait()
        console = FakeConsole(gait)
        args = argparse.Namespace(
            auto_forward=True, sensor_hz=30.0, tilt_limit=0.35,
            sensor_grace_s=0.2, state_grace_s=2.2,
            no_lidar=False,
            avoid_yaw=0.12, policy_turn_yaw=0.18, steps=20,
            max_runtime_s=3.0, gait_stall_s=1.0,
            forward_vx=0.15, lease_s=0.03, refresh_s=0.01,
        )
        mission = PhysicalStepsMission(FakeNav(), console, gait, args)
        rc = asyncio.run(mission.run())
        self.assertEqual(0, rc)
        self.assertEqual(20, mission.forward_steps)
        self.assertEqual(20, console.drives)
        self.assertEqual(1, console.stops)

    def test_auto_forward_finishes_forty_physical_events_without_stdin(self):
        gait = FakeGait()
        console = FakeConsole(gait, drives_per_step=4)
        args = argparse.Namespace(
            auto_forward=True, sensor_hz=30.0, tilt_limit=0.35,
            sensor_grace_s=0.2, state_grace_s=2.2,
            no_lidar=False,
            avoid_yaw=0.12, policy_turn_yaw=0.18, steps=40,
            max_runtime_s=3.0, gait_stall_s=1.0,
            forward_vx=0.15, lease_s=0.03, refresh_s=0.01,
        )
        mission = PhysicalStepsMission(FakeNav(), console, gait, args)
        rc = asyncio.run(mission.run())
        self.assertEqual(0, rc)
        self.assertEqual(40, mission.forward_steps)
        self.assertEqual(160, console.drives)
        self.assertEqual(1, console.stops)
        self.assertLess(mission.max_lease_gap_s, args.lease_s)


if __name__ == "__main__":
    unittest.main()
