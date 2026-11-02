"""End to end: agents_sdk_arm.run_episode in rea-online style drives eval/behavior/online/runtime.mjs (the online runtime)
against fake R1Pro API functions, with a scripted worker in place of the model (test/fixtures/behavior_online_fake_worker.py)."""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("behavior_agents_sdk_arm_e2e", ROOT / "eval/behavior/runner/agents.py")
arm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arm)
FAKE_WORKER = ROOT / "test" / "fixtures" / "behavior_online_fake_worker.py"
NODE = shutil.which("node")
SECRET = "test-only-fake-credential-0123456789"


class Budget(BaseException):
    pass


class FakeRobot:
    """Budget-free stand-ins for the R1Pro API functions the ToolServer calls."""

    def __init__(self, navigate_s=0.0):
        self.held = False; self.calls = []; self.navigate_s = navigate_s

    def fns(self):
        import numpy as np

        def record(name, value):
            self.calls.append(name)
            return value

        def check_object_in_hand(arm=0):
            self.calls.append(("check_object_in_hand", arm))
            return np.bool_(self.held and arm == 0)

        def navigate_to_pose(pose):
            time.sleep(self.navigate_s)
            return record("navigate_to_pose", True)

        def lift_arm(arm=0):
            self.held = True
            return record("lift_arm", True)
        return {"get_object_pose": lambda name, return_bbox_extent=True: record("get_object_pose", (np.array([1.0, 2.0, 0.8]), np.array([1.0, 0, 0, 0]), np.array([0.2, 0.1, 0.1]))),
                "navigate_to_pose": navigate_to_pose, "check_object_in_hand": check_object_in_hand, "lift_arm": lift_arm}


def args_for(**over):
    values = dict(agent_model="gpt-6-astra", agent_max_turns=60, agent_llm_timeout=180, agent_images=True, sdk_python=sys.executable,
                  agent_style="rea-online", agent_node=NODE, agent_max_decisions=40, agent_effort="medium", agent_max_tokens=2048,
                  online_worker=str(FAKE_WORKER))
    values.update(over)
    return SimpleNamespace(**values)


@unittest.skipUnless(NODE, "node is required for the online runtime")
class OnlineRuntimeEndToEnd(unittest.TestCase):
    def run_script(self, directory, script, robot, budget_s=60.0, observe=True, **over):
        script_path = Path(directory) / "script.json"
        script_path.write_text(json.dumps(script))
        episode = Path(directory) / "episode"; episode.mkdir(exist_ok=True)
        observed = []

        def render(path):
            observed.append(path); Path(path).write_bytes(b"\xff\xd8fake-head-camera\xff\xd9")
        logs = []
        with patch.dict(os.environ, {"REA_FAKE_SCRIPT": str(script_path)}):
            result = arm.run_episode(robot.fns(), "Pick up the red radio from the table and hold it in the gripper.",
                                     "You are controlling a R1Pro robot with API described below.\nGoal: pick up the red radio on the table.",
                                     b"\xff\xd8initial\xff\xd9", args_for(**over), budget_s, SECRET, "https://relay.invalid/v1", Budget, episode,
                                     lambda **entry: logs.append(entry), observe=render if observe else None)
        return result, episode, observed

    def test_helper_relay_refused_and_accepted_finish_and_records(self):
        robot = FakeRobot()
        script = {
            "1": {"helpers": [["get_object_pose", {"object_name": "red radio"}]], "action": {"skill": "navigate_to_pose", "arguments": {"x": 0.4, "y": 1.6, "yaw": 0.8}}},
            "2": {"action": {"skill": "finish", "arguments": {"arm": 0}}},
            "3": {"helpers": [["check_object_in_hand", {"arm": 0}]], "action": {"skill": "lift_arm", "arguments": {"arm": 0}}},
            "4": {"action": {"skill": "finish", "arguments": {"arm": 0}}},
            "5": {"action": {"skill": "lift_arm", "arguments": {"arm": 1}}},
        }
        with tempfile.TemporaryDirectory() as directory:
            result, episode, observed = self.run_script(directory, script, robot)
            self.assertEqual(result["stop_reason"], "agent_finished", result.get("error"))
            self.assertEqual(result["agent_style"], "rea-online")
            self.assertEqual(result["turns"], 4)
            self.assertEqual([f["accepted"] for f in result["finishes"]], [False, True])
            # the finish gate checks only the named arm, once per finish; the helper check also ran on arm 0
            self.assertEqual([c for c in robot.calls if isinstance(c, tuple)], [("check_object_in_hand", 0)] * 3)
            self.assertEqual([r["tool"] for r in result["tool_results"]],
                             ["get_object_pose", "navigate_to_pose", "check_object_in_hand", "check_object_in_hand", "lift_arm", "check_object_in_hand"])
            self.assertEqual(len(observed), 4)
            self.assertTrue(all(Path(p).exists() for p in observed))
            self.assertEqual(result["usage"]["requests"], 6)
            records = [json.loads(line) for line in (episode / "online-decisions.jsonl").read_text().splitlines()]
            self.assertEqual([r["action"]["skill"] for r in records], ["navigate_to_pose", "finish", "lift_arm", "finish"])
            self.assertEqual(records[0]["helper_calls"][0]["tool"], "get_object_pose")
            self.assertIn("position_xyz", records[0]["helper_calls"][0]["result"])
            self.assertTrue(records[1]["result"].startswith("REFUSED"))
            self.assertTrue(records[3]["result"].startswith("ACCEPTED"))
            self.assertEqual(records[2]["route"], "deliberate")
            self.assertRegex(records[2]["guidance"], r"^finish was refused: check_object_in_hand\(0\) was False")
            self.assertEqual(records[2]["helper_calls"][0]["result"], "false")
            seen = json.loads((episode / "online" / "decision-003" / "seen.json").read_text())
            self.assertIn("Decision 3 of at most 40; about", seen["texts"][0])
            self.assertIn("Supervisor guidance: finish was refused", seen["texts"][0])
            self.assertEqual(seen["texts"][-1], "Image: head camera now")
            self.assertEqual(seen["images"], 1)
            self.assertTrue(seen["key_seen"])
            self.assertIn("Call take_action with skill finish and the arm that holds the object", seen["texts"][0])
            self.assertNotIn("Stop (reply with a one-line summary)", seen["texts"][0])
            events = [json.loads(line) for line in (episode / "online-events.jsonl").read_text().splitlines()]
            self.assertEqual([e["type"] for e in events], ["reset", "step", "step", "step", "step", "episode_end"])
            config = json.loads((episode / "online" / "config.json").read_text())
            self.assertEqual(config["max_decisions"], 40)
            self.assertEqual(config["file_sha256"], arm.online_file_hashes())
            for path in episode.rglob("*"):
                if path.is_file():
                    self.assertNotIn(SECRET.encode(), path.read_bytes(), path)

    def test_budget_exhaustion_ends_the_episode(self):
        robot = FakeRobot(navigate_s=2.5)
        script = {str(k): {"action": {"skill": "navigate_to_pose", "arguments": {"x": k, "y": 0, "yaw": 0}}} for k in range(1, 5)}
        with tempfile.TemporaryDirectory() as directory:
            started = time.monotonic()
            result, episode, _ = self.run_script(directory, script, robot, budget_s=2.0)
            self.assertEqual(result["stop_reason"], "program_budget")
            self.assertLess(time.monotonic() - started, 10)
            self.assertEqual(robot.calls.count("navigate_to_pose"), 1)
            # the records survive the budget ending: one decision ran (the next observe answered BUDGET EXHAUSTED)
            self.assertEqual(result["turns"], 1)
            self.assertEqual(result["usage"]["decisions"], 1)
            self.assertEqual(result["usage"]["requests"], 1)
            events = [json.loads(line) for line in (episode / "online-events.jsonl").read_text().splitlines()]
            self.assertEqual([e["type"] for e in events], ["reset", "step", "episode_end"])
            self.assertEqual(json.loads((episode / "online-runtime-result.json").read_text())["stop_reason"], "program_budget")

    def test_a_decision_that_ends_inside_the_grace_keeps_its_record_and_runs_nothing(self):
        robot = FakeRobot()
        script = {"1": {"think_s": 3.0, "action": {"skill": "navigate_to_pose", "arguments": {"x": 1, "y": 0, "yaw": 0}}}}
        with tempfile.TemporaryDirectory() as directory:
            result, episode, _ = self.run_script(directory, script, robot, budget_s=2.0)
            self.assertEqual(result["stop_reason"], "program_budget")
            self.assertEqual(robot.calls.count("navigate_to_pose"), 0)  # past the budget nothing runs
            records = [json.loads(line) for line in (episode / "online-decisions.jsonl").read_text().splitlines()]
            self.assertEqual([r["decision"] for r in records], [1])
            self.assertTrue(records[0]["result"].startswith("BUDGET EXHAUSTED"))
            self.assertEqual(result["turns"], 1)
            self.assertEqual(result["usage"]["requests"], 1)
            self.assertGreater(result["past_deadline_s"], 0.5)

    def test_budget_end_mid_decision_kills_the_worker_and_keeps_the_records(self):
        script = {"1": {"think_s": 8.0, "action": {"skill": "lift_arm", "arguments": {"arm": 0}}}}
        with tempfile.TemporaryDirectory() as directory, patch.object(arm, "BUDGET_GRACE_S", 1.0):
            result, episode, _ = self.run_script(directory, script, FakeRobot(), budget_s=2.0)
            self.assertEqual(result["stop_reason"], "program_budget")
            decision = episode / "online" / "decision-001"
            pid = int((decision / "pid").read_text())
            time.sleep(0.5)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            runtime = json.loads((episode / "online-runtime-result.json").read_text())
            self.assertEqual((runtime["reason"], runtime["stop_reason"], runtime["decisions"]), ("stopped_by_runner", "program_budget", 1))
            events = [json.loads(line) for line in (episode / "online-events.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["type"], "episode_end")
            self.assertEqual(result["turns"], 1)
            time.sleep(6.5)
            self.assertFalse((decision / "decision.json").exists())  # no late write into the finished episode

    def test_records_name_the_episode_task_and_seed(self):
        script = {"1": {"action": {"skill": "lift_arm", "arguments": {"arm": 0}}}, "2": {"action": {"skill": "finish", "arguments": {"arm": 0}}}}
        with tempfile.TemporaryDirectory() as directory:
            _, episode, _ = self.run_script(directory, script, FakeRobot(), task="radio", seed=3)
            events = [json.loads(line) for line in (episode / "online-events.jsonl").read_text().splitlines()]
            self.assertEqual(events[0], {"type": "reset", "task": "radio", "seed": 3})
            self.assertEqual((events[-1]["summary"]["task"], events[-1]["summary"]["seed"]), ("radio", 3))

    def test_malformed_helper_arguments_answer_an_error_instead_of_crashing_the_runner(self):
        script = {"1": {"helpers": [["get_object_pose", {"name": "red radio"}]], "action": {"skill": "finish", "arguments": {"arm": 0}}},
                  "2": {"action": None, "status": "provider_error", "provider_error": "APIConnectionError"}}
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self.run_script(directory, script, FakeRobot())
            self.assertTrue(result["tool_results"][0]["result"].startswith("ERROR KeyError"))
            self.assertEqual(result["stop_reason"], "provider_error")

    def test_decision_timeout_during_a_helper_keeps_the_runtime_stop_reason(self):
        robot = FakeRobot(); fns = robot.fns()
        fns["sample_grasp_pose"] = lambda name: (time.sleep(6.0), (None, None))[1]
        robot.fns = lambda: fns
        script = {"1": {"helpers": [["sample_grasp_pose", {"object_name": "red radio"}]], "action": {"skill": "lift_arm", "arguments": {"arm": 0}}}}
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads((ROOT / "eval/behavior/online/config.example.json").read_text()); config["decision_timeout_ms"] = 3000
            config_path = Path(directory) / "online-config.json"; config_path.write_text(json.dumps(config))
            result, episode, _ = self.run_script(directory, script, robot, online_config=str(config_path))
            self.assertEqual(json.loads((episode / "online-runtime-result.json").read_text())["stop_reason"], "decision_timeout")
            self.assertEqual(result["stop_reason"], "decision_timeout", result.get("error"))
            self.assertEqual(result["turns"], 1)

    def test_no_decision_timeout_of_its_own(self):
        config = json.loads((ROOT / "eval/behavior/online/config.example.json").read_text())
        self.assertNotIn("decision_timeout_ms", config)  # the runtime's limit is budget_s + 600 s: the runner's budget ends first
        source = (ROOT / "eval/behavior/online/runtime.mjs").read_text()
        self.assertIn("decisionTimeoutMs: config.decision_timeout_ms ?? Math.max(1000, (config.budget_s + 600) * 1000)", source)

    def test_provider_failure_stops_unscored_and_rerun_sets_records_aside(self):
        script = {"1": {"action": None, "status": "provider_error", "provider_error": "APIConnectionError"}}
        with tempfile.TemporaryDirectory() as directory:
            result, episode, _ = self.run_script(directory, script, FakeRobot())
            self.assertEqual(result["stop_reason"], "provider_error")
            self.assertEqual(result["provider_error"], "APIConnectionError")
            fields = arm.outcome_fields(result["stop_reason"], False, 0.0)
            self.assertIsNone(fields["success"]); self.assertTrue(fields["unscored"])
            again, episode, _ = self.run_script(directory, {"1": {"action": {"skill": "finish", "arguments": {"arm": 1}}}, "2": {"action": None, "status": "provider_error", "provider_error": "RateLimitError"}}, FakeRobot())
            self.assertEqual(again["stop_reason"], "provider_error")
            self.assertTrue((episode / "online.prev-1" / "online" / "decision-001" / "decision.json").exists())
            self.assertEqual(again["previous_attempt"], str((episode / "online.prev-1").resolve()))
            self.assertEqual(len((episode / "online-decisions.jsonl").read_text().splitlines()), 1)

    def test_invalid_decisions_count_and_missing_camera_is_reported(self):
        script = {"1": {"action": None, "status": "model_invalid_action"},
                  "2": {"action": {"skill": "finish", "arguments": {"arm": 0}}},
                  "3": {"action": {"skill": "finish", "arguments": {"arm": 0}}}}
        with tempfile.TemporaryDirectory() as directory:
            result, episode, _ = self.run_script(directory, script, FakeRobot(), observe=False, agent_max_decisions=5)
            self.assertEqual(result["stop_reason"], "max_decisions")
            self.assertEqual(result["turns"], 5)
            records = [json.loads(line) for line in (episode / "online-decisions.jsonl").read_text().splitlines()]
            self.assertTrue(records[0]["result"].startswith("NO ACTION"))
            seen = json.loads((episode / "online" / "decision-001" / "seen.json").read_text())
            self.assertIn("unavailable", seen["texts"][-1])


if __name__ == "__main__":
    unittest.main()
