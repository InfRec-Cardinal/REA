"""BEHAVIOR agent arms (behavior_arm_runner.py --mode agents-sdk): model settings, provider-failure rule, finish gate, credentials."""
import importlib.util
import os
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arm = load("behavior_agents_sdk_arm_unit", ROOT / "eval/behavior/runner/agents.py")
with patch.dict(os.environ, {}):  # the runner sets two Kit defaults at import; keep them out of this process
    runner = load("behavior_arm_runner_unit", ROOT / "eval/behavior/runner/run_episode.py")


# Stand-ins with the openai exception class names (the SDK is not installed in the test interpreter).
class APIError(Exception):
    pass


class APIConnectionError(APIError):
    pass


class APITimeoutError(APIConnectionError):
    pass


class APIStatusError(APIError):
    def __init__(self, message, status_code):
        super().__init__(message); self.status_code = status_code


class RateLimitError(APIStatusError):
    pass


class AuthenticationError(APIStatusError):
    pass


class PermissionDeniedError(APIStatusError):
    pass


class InternalServerError(APIStatusError):
    pass


class BadRequestError(APIStatusError):
    pass


class MaxTurnsExceeded(Exception):
    pass


class ModelBehaviorError(Exception):
    pass


def response_failed(code):  # the literal openai-agents 0.22.3 message
    return ModelBehaviorError("Responses stream ended with terminal event `response.failed`. status=failed; "
                              f"error=ResponseError(code='{code}', message='m', misalignment=None).")


CUT = ModelBehaviorError("Responses stream ended with terminal event `response.incomplete`. status=incomplete; "
                         "incomplete_details=IncompleteDetails(reason='max_output_tokens')")


def attribute_error(filename, response):
    """An AttributeError raised from a frame in `filename` (as the SDK's model adapter raises it on a `null` body)."""
    namespace = {}
    exec(compile("def get_response(response):\n    return response.usage\n", filename, "exec"), namespace)
    try:
        namespace["get_response"](response)
    except AttributeError as exc:
        return exc


class ModelSettingsTest(unittest.TestCase):
    def test_chat_completions_keeps_the_sdk_defaults(self):
        self.assertIsNone(arm.agent_model_settings("chat_completions"))
        self.assertIsNone(arm.agent_model_settings(None))

    def test_responses_sets_effort_tokens_and_no_storage_only(self):
        self.assertEqual(arm.agent_model_settings("responses", "medium", 2048),
                         {"reasoning": {"effort": "medium"}, "max_tokens": 2048, "store": False})
        self.assertNotIn("tool_choice", arm.agent_model_settings("responses"))
        self.assertNotIn("parallel_tool_calls", arm.agent_model_settings("responses"))
        with self.assertRaises(ValueError):
            arm.agent_model_settings("completions")


class ProviderFailureTest(unittest.TestCase):
    def test_transport_auth_rate_and_server_errors_are_infrastructure(self):
        for exc, label in [(APIConnectionError("Connection error."), "APIConnectionError"),
                           (APITimeoutError("Request timed out."), "APIConnectionError"),
                           (RateLimitError("rate limited", 429), "RateLimitError"),
                           (AuthenticationError("bad key", 401), "AuthenticationError"),
                           (PermissionDeniedError("forbidden", 403), "PermissionDeniedError"),
                           (InternalServerError("boom", 500), "InternalServerError"),
                           (APIStatusError("bad gateway", 502), "HTTP 502"),
                           (BadRequestError("Error code: 400 - insufficient balance for this request", 400), "insufficient_balance_or_quota"),
                           (APIStatusError("You exceeded your current quota", 429), "insufficient_balance_or_quota")]:
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(arm.provider_failure(exc), label)

    def test_wrapped_provider_errors_are_found_through_the_cause_chain(self):
        try:
            try:
                raise APITimeoutError("Request timed out.")
            except APITimeoutError as inner:
                raise RuntimeError("agent run failed") from inner
        except RuntimeError as outer:
            self.assertEqual(arm.provider_failure(outer), "APIConnectionError")

    def test_task_outcomes_are_not_provider_errors(self):
        for exc in (MaxTurnsExceeded("Max turns (60) exceeded"), BadRequestError("invalid tool schema", 400),
                    ValueError("bad arguments"), RuntimeError("runner closed the tool channel")):
            with self.subTest(exc=type(exc).__name__):
                self.assertIsNone(arm.provider_failure(exc))

    def test_classification_by_names(self):
        self.assertEqual(arm.classify_provider_failure(["APITimeoutError", "APIConnectionError", "APIError"]), "APIConnectionError")
        self.assertEqual(arm.classify_provider_failure(["APIStatusError"], 503, ""), "HTTP 503")
        self.assertIsNone(arm.classify_provider_failure(["APIStatusError"], 404, "model not found"))

    def test_request_timeout_status_is_infrastructure(self):
        self.assertEqual(arm.provider_failure(APIStatusError("Error code: 408 - upstream request timeout", 408)), "HTTP 408")
        self.assertEqual(arm.classify_provider_failure(["APIStatusError"], 408, ""), "HTTP 408")

    def test_null_or_non_json_response_body_is_infrastructure(self):
        sdk = "/venv/lib/python3.10/site-packages/agents/models/"
        self.assertEqual(arm.provider_failure(attribute_error(sdk + "openai_responses.py", None)), "malformed_response")
        self.assertEqual(arm.provider_failure(attribute_error(sdk + "openai_responses.py", "<html>502</html>")), "malformed_response")
        self.assertEqual(arm.provider_failure(attribute_error(sdk + "openai_chatcompletions.py", (None, None))), "malformed_response")
        try:
            raise RuntimeError("agent run failed") from attribute_error(sdk + "openai_responses.py", None)
        except RuntimeError as outer:
            self.assertEqual(arm.provider_failure(outer), "malformed_response")
        self.assertIsNone(arm.provider_failure(attribute_error(str(ROOT / "eval/behavior/runner/agents.py"), None)))
        self.assertIsNone(arm.provider_failure(AttributeError("'NoneType' object has no attribute 'usage'")))

    def test_provider_side_response_failed_is_infrastructure(self):
        for code in ("server_error", "rate_limit_exceeded"):
            with self.subTest(code=code):
                self.assertEqual(arm.provider_failure(response_failed(code)), "response_failed")
        for exc in (response_failed("invalid_image"), response_failed("invalid_prompt"), CUT):
            with self.subTest(exc=str(exc)[-60:]):
                self.assertIsNone(arm.provider_failure(exc))

    def test_sdk_child_stop_reasons(self):
        self.assertEqual(arm.agent_failure(CUT), {"stop_reason": "max_output_tokens"})
        self.assertTrue(arm.is_cut_reply(CUT)); self.assertFalse(arm.is_cut_reply(response_failed("server_error")))
        self.assertEqual(arm.agent_failure(BadRequestError("content refused", 400)), {"stop_reason": "agent_error", "http_status": 400})
        self.assertEqual(arm.agent_failure(APIStatusError("model not found", 404)), {"stop_reason": "agent_error", "http_status": 404})
        self.assertEqual(arm.agent_failure(InternalServerError("boom", 503)),
                         {"stop_reason": "provider_error", "provider_error": "InternalServerError", "http_status": 503})
        self.assertEqual(arm.agent_failure(MaxTurnsExceeded("Max turns (60) exceeded")), {"stop_reason": "max_turns"})
        self.assertEqual(arm.agent_failure(response_failed("server_error")), {"stop_reason": "provider_error", "provider_error": "response_failed"})
        for stop in ("max_output_tokens", "agent_error", "max_turns"):
            self.assertNotIn("unscored", arm.outcome_fields(stop, False, 0.0))

    def test_provider_error_leaves_the_episode_unscored(self):
        self.assertEqual(arm.outcome_fields("provider_error", True, 1.0),
                         {"success": None, "reward": None, "unscored": True, "stop_reason": "provider_error", "unused_task_completed": True})
        for stop in ("agent_finished", "program_budget", "max_turns", "agent_error", "max_decisions"):
            with self.subTest(stop=stop):
                self.assertEqual(arm.outcome_fields(stop, False, 0.0), {"success": False, "reward": 0.0, "stop_reason": stop})


class FinishGateTest(unittest.TestCase):
    def test_gate_reads_the_single_check(self):
        self.assertEqual(arm.finish_gate("true", 1)[0], True)
        self.assertTrue(arm.finish_gate("true", 1)[1].startswith("ACCEPTED"))
        for value in ("false", "ERROR RuntimeError: x", "null"):
            accepted, message = arm.finish_gate(value, 0)
            self.assertFalse(accepted); self.assertTrue(message.startswith("REFUSED"))
        self.assertEqual(arm.finish_gate("BUDGET EXHAUSTED: stop now.", 0), (False, "BUDGET EXHAUSTED: stop now."))

    def test_finish_checks_only_the_named_arm_once(self):
        calls = []
        def check(arm=0):
            calls.append(arm)
            return arm == 1
        server = arm.ToolServer({"check_object_in_hand": check}, None, KeyboardInterrupt)
        self.assertTrue(server.finish({"arm": 0}).startswith("REFUSED"))
        self.assertFalse(server.finished)
        self.assertTrue(server.finish({"arm": 1}).startswith("ACCEPTED"))
        self.assertTrue(server.finished)
        self.assertEqual(calls, [0, 1])
        self.assertTrue(server.finish({"arm": "left"}).startswith("REFUSED"))
        self.assertEqual(calls, [0, 1])
        self.assertEqual([f["accepted"] for f in server.finishes], [False, True, False])


class ToolServerTest(unittest.TestCase):
    def test_malformed_arguments_answer_an_error(self):
        server = arm.ToolServer({"get_object_pose": lambda *a, **k: None}, None, KeyboardInterrupt)
        self.assertTrue(server.execute("get_object_pose", {"name": "red radio"}).startswith("ERROR KeyError"))
        self.assertTrue(server.execute("compute_approach_pose", {"robot_x": "near"}).startswith("ERROR"))
        self.assertTrue(server.execute("navigate_to_pose", ["x"]).startswith("ERROR"))
        self.assertEqual(len(server.tool_results), 3)


class ChildLinesTest(unittest.TestCase):
    """The runner's reader of the agent's JSON lines: every received line is returned, also when several arrive at once."""

    def setUp(self):
        r, self.w = os.pipe()
        self.stream = os.fdopen(r, "r")
        self.lines = arm.ChildLines(self.stream)

    def tearDown(self):
        self.stream.close()
        try: os.close(self.w)
        except OSError: pass

    def test_a_progress_line_and_the_call_after_it_in_one_write_are_both_read(self):
        # The smoke deadlock: select() + a buffered readline() returned the progress line and hid the call behind it.
        os.write(self.w, b'{"type": "progress", "turns": 5}\n{"type": "call", "id": 7, "tool": "navigate_to_pose"}\n')
        self.assertEqual(self.lines.get(1.0), '{"type": "progress", "turns": 5}\n')
        self.assertEqual(self.lines.get(0.0), '{"type": "call", "id": 7, "tool": "navigate_to_pose"}\n')
        self.assertIsNone(self.lines.get(0.05))

    def test_a_partial_line_does_not_extend_the_timeout(self):
        # Review of PR #20: each chunk without a newline restarted the wait, so a trickle could hold the runner past its budget.
        def trickle():
            for _ in range(8):
                time.sleep(0.2)
                os.write(self.w, b"x")
        writer = threading.Thread(target=trickle); writer.start()
        started = time.monotonic()
        self.assertIsNone(self.lines.get(0.5))
        self.assertLess(time.monotonic() - started, 0.9)
        writer.join()
        os.write(self.w, b'yz"}\n')
        self.assertEqual(self.lines.get(1.0), 'x' * 8 + 'yz"}\n')  # the buffered bytes are kept and the line is whole

    def test_get_zero_is_one_poll(self):
        self.assertIsNone(self.lines.get(0.0))
        os.write(self.w, b'{"type": "call", "id": 1}\n{"type": "ca')
        self.assertEqual(self.lines.get(0.0), '{"type": "call", "id": 1}\n')
        self.assertIsNone(self.lines.get(0.0))

    def test_a_line_split_across_writes_waits_for_its_end_and_a_partial_tail_is_kept_at_eof(self):
        os.write(self.w, b'{"type": "do')
        self.assertIsNone(self.lines.get(0.05))
        os.write(self.w, 'ne", "note": "é"}\ntail'.encode())
        self.assertEqual(self.lines.get(1.0), '{"type": "done", "note": "é"}\n')
        os.close(self.w)
        self.assertEqual(self.lines.get(1.0), "tail\n")
        self.assertEqual(self.lines.get(1.0), "")
        self.assertEqual(self.lines.get(1.0), "")


class ClockTest(unittest.TestCase):
    def test_sdk_tool_results_carry_duration_and_remaining_budget(self):
        self.assertRegex(arm.with_clock("true", time.monotonic() + 100, 1500, 8.2), r"^true \[took 8 s; about (99|100) s of the 1500 s budget remain\]$")
        self.assertEqual(arm.with_clock("BUDGET EXHAUSTED: stop now.", time.monotonic(), 1500, 0.0), "BUDGET EXHAUSTED: stop now.")
        self.assertRegex(arm.with_clock("false", time.monotonic() - 5, 1500), r"^false \[about 0 s of the 1500 s budget remain\]$")


class ProvenanceTest(unittest.TestCase):
    def test_code_hashes_cover_both_arms_and_the_framework(self):
        hashes = arm.code_file_hashes([ROOT / "test/fixtures/behavior_task_prompt.txt", None])
        for key in ("eval/behavior/runner/agents.py", "eval/behavior/runner/run_episode.py",
                    "eval/behavior/online/runtime.mjs", "eval/behavior/online/worker.py", "test/fixtures/behavior_task_prompt.txt", "src/"):
            with self.subTest(key=key):
                self.assertRegex(hashes[key] or "", "^[0-9a-f]{64}$")
        self.assertEqual(hashes, arm.code_file_hashes([ROOT / "test/fixtures/behavior_task_prompt.txt"]))

    def test_repo_commit_prefers_the_deployed_pin(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory, patch.object(arm, "REPO_ROOT", Path(directory)):
            self.assertIsNone(arm.repo_commit())  # neither COMMIT nor .git
            Path(directory, "COMMIT").write_text("0123abc\n")
            self.assertEqual(arm.repo_commit(), "0123abc")


class TaskTextTest(unittest.TestCase):
    PROMPT = (ROOT / "test/fixtures/behavior_task_prompt.txt").read_text()

    def test_online_gets_the_sdk_text_with_the_finish_wording(self):
        sdk = arm.sdk_instructions(self.PROMPT, 1500)
        online = arm.online_task_context(self.PROMPT, 1500)
        self.assertIn(arm.SDK_STOP_SENTENCE, sdk)
        self.assertEqual(online, sdk.replace(arm.SDK_STOP_SENTENCE, arm.REA_FINISH_SENTENCE))
        self.assertIn("1500 s wall-clock budget", online)

    def test_tool_specs_come_from_the_sdk_schemas(self):
        skills, helpers = arm.online_tool_specs()
        described = {name: desc for name, desc, _ in arm.TOOL_SCHEMAS}
        self.assertEqual([s["name"] for s in skills][-1], "finish")
        for spec in skills[:-1] + helpers:
            self.assertEqual(spec["description"], described[spec["name"]])
        self.assertEqual(skills[-1]["parameters"], {"arm": "integer"})
        self.assertFalse(set(arm.REA_SKILLS) & set(arm.REA_HELPERS))


class CredentialsTest(unittest.TestCase):
    def test_agent_key_takes_precedence_and_needs_its_base(self):
        environ = {"AGENT_API_KEY": "agent", "AGENT_BASE_URL": "https://relay.invalid/v1", "REA_API_KEY": "rea"}
        self.assertEqual(runner.agent_credentials(environ), ("agent", "https://relay.invalid/v1"))
        with self.assertRaisesRegex(ValueError, "AGENT_BASE_URL"):
            runner.agent_credentials({"AGENT_API_KEY": "agent"})
        with patch.object(runner, "credentials", return_value=("rea", "https://provider.invalid/v1")):
            self.assertEqual(runner.agent_credentials({"REA_API_KEY": "rea"}), ("rea", "https://provider.invalid/v1"))

    def test_program_labels(self):
        from types import SimpleNamespace
        self.assertEqual(runner.agent_program_label(SimpleNamespace(agent_style="sdk", agent_model="gpt-6-astra")), "agents-sdk:gpt-6-astra")
        self.assertEqual(runner.agent_program_label(SimpleNamespace(agent_style="rea-online", agent_model="gpt-6-astra")), "rea-online:gpt-6-astra")


if __name__ == "__main__":
    unittest.main()
