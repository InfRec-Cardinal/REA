import io
import unittest
from unittest.mock import patch

from agent_vla_async.brain import RemoteBrainClient, main, terminal_event_for_intent


class FakeProcess:
    def __init__(self):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO('{"ok":true,"n":1}\n{"ok":true,"n":2}\n')
        self.stderr = io.StringIO("")
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class PersistentBrainClientTests(unittest.TestCase):
    def test_multiple_calls_share_one_ssh_process(self):
        process = FakeProcess()
        with patch("agent_vla_async.brain.subprocess.Popen", return_value=process) as popen:
            client = RemoteBrainClient()
            self.assertEqual(1, client.call({"op": "ping"})["n"])
            self.assertEqual(2, client.call({"op": "status"})["n"])
            self.assertEqual(1, popen.call_count)
            sent = process.stdin.getvalue().splitlines()
            self.assertEqual(2, len(sent))
            client.close()

    def test_submit_forwards_an_explicit_policy_timeout(self):
        class RecordingClient:
            def __init__(self):
                self.requests = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def call(self, request):
                self.requests.append(request)
                return {"ok": True}

        client = RecordingClient()
        with (
            patch("agent_vla_async.brain.RemoteBrainClient", return_value=client),
            patch("agent_vla_async.brain._print"),
        ):
            rc = main(["submit", "inspect door", "--policy-timeout", "27.5"])
        self.assertEqual(0, rc)
        self.assertEqual(27.5, client.requests[0]["policy_wait_timeout_s"])

    def test_terminal_event_must_belong_to_current_intent(self):
        events = [
            {"kind": "PREVIEW_COMPLETE", "intent_id": "old"},
            {"kind": "ACTION_READY", "intent_id": "current"},
            {"kind": "EXECUTION_COMPLETE", "intent_id": "current"},
        ]
        terminal = terminal_event_for_intent(
            events,
            "current",
            {"PREVIEW_COMPLETE", "EXECUTION_COMPLETE"},
        )
        self.assertEqual("EXECUTION_COMPLETE", terminal["kind"])

    def test_controller_commands_use_the_persistent_rpc_channel(self):
        class RecordingClient:
            def __init__(self):
                self.requests = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def call(self, request):
                self.requests.append(request)
                return {"ok": True}

        cases = (
            (
                ["sonic-validate", "touch the door"],
                {"op": "sonic_validate", "task": "touch the door"},
            ),
            (["controller-status"], {"op": "controller_status"}),
            (
                ["controller-prepare", "SONIC_WBC"],
                {"op": "controller_prepare", "controller_mode": "SONIC_WBC"},
            ),
            (
                ["controller-stop", "SONIC_WBC"],
                {"op": "controller_stop", "controller_mode": "SONIC_WBC"},
            ),
        )
        for argv, expected in cases:
            with self.subTest(argv=argv):
                client = RecordingClient()
                with (
                    patch("agent_vla_async.brain.RemoteBrainClient", return_value=client),
                    patch("agent_vla_async.brain._print"),
                ):
                    rc = main(argv)
                self.assertEqual(0, rc)
                self.assertEqual(expected, client.requests[0])

    def test_submit_forwards_controller_mode(self):
        class RecordingClient:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def call(self, request):
                self.request = request
                return {"ok": False}

        client = RecordingClient()
        with (
            patch("agent_vla_async.brain.RemoteBrainClient", return_value=client),
            patch("agent_vla_async.brain._print"),
        ):
            rc = main(["submit", "inspect", "--controller-mode", "SONIC_WBC"])
        self.assertEqual(1, rc)
        self.assertEqual("SONIC_WBC", client.request["controller_mode"])

    def test_stale_terminal_event_is_ignored(self):
        terminal = terminal_event_for_intent(
            [{"kind": "PREVIEW_COMPLETE", "intent_id": "old"}],
            "current",
            {"PREVIEW_COMPLETE", "EXECUTION_COMPLETE"},
        )
        self.assertIsNone(terminal)


if __name__ == "__main__":
    unittest.main()
