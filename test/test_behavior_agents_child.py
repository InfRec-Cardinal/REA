"""End to end: agents_sdk_arm.run_episode in sdk style (--agent-api responses) runs the real Agents SDK child against a fake
Responses relay (test/fixtures/fake_responses_relay.py). Needs an interpreter with openai-agents: set BEHAVIOR_SDK_PYTHON to it
(e.g. a venv with openai-agents 0.22.3); skipped otherwise."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("behavior_agents_sdk_arm_child", ROOT / "eval/behavior/runner/agents.py")
arm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arm)
SDK_PYTHON = os.environ.get("BEHAVIOR_SDK_PYTHON")


def sdk_available():
    if not SDK_PYTHON:
        return False
    return subprocess.run([SDK_PYTHON, "-c", "import agents, agents.retry"], capture_output=True).returncode == 0


class Budget(BaseException):
    pass


@unittest.skipUnless(sdk_available(), "BEHAVIOR_SDK_PYTHON with openai-agents is required")
class SdkChildEndToEnd(unittest.TestCase):
    def run_relay(self, modes, budget_s=60.0):
        directory = Path(tempfile.mkdtemp(prefix="sdk-child-")); episode = directory / "episode"; episode.mkdir()
        bodies = directory / "bodies.jsonl"
        relay = subprocess.Popen([sys.executable, str(ROOT / "test/fixtures/fake_responses_relay.py")], stdout=subprocess.PIPE, text=True,
                                 env=dict(os.environ, MODE=",".join(modes), BODIES=str(bodies)))
        port = int(relay.stdout.readline().split()[1])
        fns = {"get_robot_position": lambda: ([1.0, 2.0, 0.0], [1.0, 0, 0, 0], 0.5)}
        args = SimpleNamespace(agent_model="gpt-6-astra", agent_max_turns=60, agent_llm_timeout=20, agent_images=0, sdk_python=SDK_PYTHON,
                               agent_style="sdk", agent_api="responses", agent_effort="medium", agent_max_tokens=2048)
        try:
            result = arm.run_episode(fns, "Pick up the radio.", "Goal: pick up the radio.", None, args, budget_s, "sk-test",
                                     f"http://127.0.0.1:{port}/v1", Budget, episode, lambda **k: None)
        finally:
            relay.kill(); relay.wait(); relay.stdout.close()
        requests = [json.loads(line)["body"] for line in bodies.read_text().splitlines()] if bodies.exists() else []
        return result, requests

    def test_one_cut_reply_is_re_requested_and_tool_results_carry_the_clock(self):
        result, requests = self.run_relay(["tool", "cut", "final"])
        self.assertEqual(result["stop_reason"], "agent_finished", result.get("error"))
        self.assertEqual(result["truncated_retries"], 1)
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[1]["input"], requests[2]["input"])  # the identical turn again
        self.assertEqual(result["turns"], 2)
        self.assertEqual(result["tool_results"][0]["result"][:1], "{")  # the runner records the bare value
        outputs = [item for item in requests[1]["input"] if item.get("type") == "function_call_output"]
        self.assertRegex(outputs[0]["output"], r"\[took 0 s; about \d+ s of the 60 s budget remain\]$")
        self.assertEqual((requests[0]["max_output_tokens"], requests[0]["reasoning"], requests[0]["store"]), (2048, {"effort": "medium"}, False))
        self.assertNotIn("tool_choice", requests[0])

    def test_a_reply_still_cut_after_the_retries_is_a_scored_max_output_tokens(self):
        result, requests = self.run_relay(["tool", "cut"])
        self.assertEqual(result["stop_reason"], "max_output_tokens")
        self.assertEqual(len(requests), 2 + arm.CUT_RETRIES)
        self.assertEqual(result["turns"], 1)
        self.assertNotIn("unscored", arm.outcome_fields(result["stop_reason"], False, 0.0))

    def test_http_400_is_a_scored_agent_error_with_its_status(self):
        result, _ = self.run_relay(["tool", "http400"])
        self.assertEqual((result["stop_reason"], result["http_status"], result["turns"]), ("agent_error", 400, 1))

    def test_http_408_and_a_null_body_are_provider_errors(self):
        for mode, label in (("http408", "HTTP 408"), ("null", "malformed_response")):
            with self.subTest(mode=mode):
                result, _ = self.run_relay(["tool", mode])
                self.assertEqual((result["stop_reason"], result.get("provider_error")), ("provider_error", label), result.get("error"))
                self.assertTrue(arm.outcome_fields(result["stop_reason"], True, 1.0)["unscored"])

    def test_a_budget_ending_keeps_turns_and_usage(self):
        started = time.monotonic()
        result, requests = self.run_relay(["tool", "tool", "slow"], budget_s=3.0)
        self.assertEqual(result["stop_reason"], "program_budget")
        self.assertLess(time.monotonic() - started, 3.0 + 6)
        self.assertEqual(result["turns"], 2)  # from the child's progress messages: it was stopped without a done
        self.assertEqual(result["usage"]["requests"], 2)
        self.assertTrue(all(r["result"].startswith("{") for r in result["tool_results"]))  # nothing ran past the budget
        self.assertEqual(len(result["tool_results"]), 2)


if __name__ == "__main__":
    unittest.main()
