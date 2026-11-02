import json
import unittest

from agent_vla_async.arm import (
    ArmActionNotFound,
    ArmActionTimeout,
    HighLevelArmActionBackend,
    parse_action_state,
)


ACTION_LIST = [
    [
        {"id": 99, "name": "release_arm"},
        {"id": 36, "mode_machine": [5, 6], "name": "forward_push"},
    ],
    [
        {"name": "Scratch_head", "time": 8.1},
    ],
]


class FakeSubscriber:
    def __init__(self):
        self.callback = None
        self.queue_len = None
        self.closed = False

    def Init(self, callback, queue_len):
        self.callback = callback
        self.queue_len = queue_len

    def Close(self):
        self.closed = True

    def publish(self, payload):
        self.callback(payload)


class FakeClient:
    def __init__(self):
        self.timeout = None
        self.initialized = False
        self.registered = []
        self.executed = []
        self.raw_calls = []
        self.on_execute = None
        self.on_custom = None

    def SetTimeout(self, timeout):
        self.timeout = timeout

    def Init(self):
        self.initialized = True

    def _RegistApi(self, api_id, priority):
        self.registered.append((api_id, priority))

    def GetActionList(self):
        return 0, ACTION_LIST

    def ExecuteAction(self, action_id):
        self.executed.append(action_id)
        if self.on_execute is not None:
            self.on_execute(action_id)
        return 0

    def _Call(self, api_id, parameter):
        self.raw_calls.append((api_id, parameter))
        if api_id == 7108 and self.on_custom is not None:
            self.on_custom(json.loads(parameter)["action_name"])
        return 0, ""


class Message:
    def __init__(self, data):
        self.data = data


class FakeCompletedProcess:
    def __init__(self, response, returncode=0):
        self.stdout = json.dumps(response)
        self.stderr = ""
        self.returncode = returncode


class FakeSubprocessRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, command, **kwargs):
        request = json.loads(kwargs["input"])
        self.calls.append({"command": command, "request": request, "kwargs": kwargs})
        operation = request["op"]
        if operation == "list":
            return FakeCompletedProcess({
                "ok": True, "op": "list", "rc": 0, "actions": ACTION_LIST,
            })
        if operation == "execute":
            return FakeCompletedProcess({
                "ok": True,
                "op": "execute",
                "executed": True,
                "rc": 0,
                "state": {"holding": True, "id": 36, "name": request["name"]},
            })
        if operation == "release":
            return FakeCompletedProcess({
                "ok": True, "op": "release", "executed": True, "rc": 0, "state": None,
            })
        if operation == "stop_custom":
            return FakeCompletedProcess({
                "ok": True, "op": "stop_custom", "stopped": True, "rc": 0,
            })
        raise AssertionError("unexpected operation")


class ArmActionBackendTests(unittest.TestCase):
    def make_backend(self, execute=False):
        client = FakeClient()
        subscriber = FakeSubscriber()
        backend = HighLevelArmActionBackend(
            execute_enabled=execute,
            client=client,
            state_subscriber=subscriber,
            completion_timeout_s=0.05,
        )
        return backend, client, subscriber

    def test_real_firmware_list_shape_is_resolved_by_name(self):
        backend, client, subscriber = self.make_backend()
        actions = backend.list_actions()
        push = backend.resolve_action("Forward Push")
        custom = backend.resolve_action("scratch-head")

        self.assertEqual(push.action_id, 36)
        self.assertFalse(push.custom)
        self.assertIsNone(custom.action_id)
        self.assertTrue(custom.custom)
        self.assertEqual(custom.duration_s, 8.1)
        self.assertTrue(client.initialized)
        self.assertEqual(subscriber.queue_len, 10)
        self.assertIn("forward_push", [action.name for action in actions.values()])

    def test_default_dry_run_resolves_but_sends_no_action(self):
        backend, client, _ = self.make_backend()
        result = backend.execute("forward_push")

        self.assertTrue(result["ok"])
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["executed"])
        self.assertEqual(result["action"]["id"], 36)
        self.assertEqual(client.executed, [])
        self.assertEqual(client.raw_calls, [])

    def test_preset_execute_waits_for_matching_state(self):
        backend, client, subscriber = self.make_backend(execute=True)
        backend.connect()
        client.on_execute = lambda action_id: subscriber.publish(
            Message('{"holding":true,"id":36,"name":"forward_push"}')
        )

        result = backend.execute("forward_push", wait=True)

        self.assertEqual(client.executed, [36])
        self.assertTrue(result["executed"])
        self.assertEqual(result["state"]["name"], "forward_push")
        self.assertTrue(result["state"]["holding"])

    def test_custom_execute_and_stop_use_high_level_rpc_ids(self):
        backend, client, subscriber = self.make_backend(execute=True)
        backend.connect()
        client.on_custom = lambda name: subscriber.publish(
            {"holding": False, "id": 100, "name": name}
        )

        result = backend.execute("Scratch_head")
        stopped = backend.stop_custom_action()

        self.assertEqual(result["action"]["custom"], True)
        self.assertEqual(client.raw_calls[0][0], 7108)
        self.assertEqual(json.loads(client.raw_calls[0][1]), {"action_name": "Scratch_head"})
        self.assertEqual(client.raw_calls[1], (7113, ""))
        self.assertTrue(stopped["stopped"])

    def test_state_parser_accepts_sdk_string_message_and_status(self):
        state = parse_action_state(
            Message('{"holding":false,"action_id":100,"action_name":"Scratch_head",'
                    '"status":"completed"}')
        )
        self.assertEqual(state.name, "Scratch_head")
        self.assertEqual(state.action_id, 100)
        self.assertFalse(state.holding)
        self.assertEqual(state.status, "completed")

    def test_wait_ignores_old_or_running_state_then_times_out(self):
        backend, _, _ = self.make_backend(execute=True)
        backend.update_action_state({
            "holding": False, "id": 36, "name": "forward_push", "status": "running",
        })
        with self.assertRaises(ArmActionTimeout):
            backend.wait_for_completion(
                "forward_push", after_version=0, timeout_s=0.005
            )

    def test_unknown_action_is_not_replaced_by_a_static_id(self):
        backend, _, _ = self.make_backend()
        with self.assertRaises(ArmActionNotFound):
            backend.execute("not_installed_here")

    def test_close_releases_state_subscriber(self):
        backend, _, subscriber = self.make_backend()
        backend.connect()
        backend.close()
        self.assertTrue(subscriber.closed)

    def test_default_transport_uses_system_python_bridge_for_dry_run_list(self):
        runner = FakeSubprocessRunner()
        backend = HighLevelArmActionBackend(subprocess_runner=runner)

        result = backend.execute("forward_push")

        self.assertEqual(backend.transport, "bridge")
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["transport"], "bridge")
        self.assertEqual([call["request"]["op"] for call in runner.calls], ["list"])
        self.assertEqual(runner.calls[0]["command"][0], "/usr/bin/python3")
        self.assertTrue(runner.calls[0]["command"][1].endswith("arm_bridge_worker.py"))

    def test_bridge_executes_and_releases_without_importing_sdk_in_daemon(self):
        runner = FakeSubprocessRunner()
        backend = HighLevelArmActionBackend(
            execute_enabled=True,
            subprocess_runner=runner,
        )

        pushed = backend.execute("forward_push", wait=True, timeout_s=2.0)
        released = backend.release(wait=False)
        stopped = backend.stop_custom_action()

        self.assertEqual(pushed["state"]["name"], "forward_push")
        self.assertEqual(released["action"]["name"], "release_arm")
        self.assertTrue(stopped["stopped"])
        self.assertEqual(
            [call["request"]["op"] for call in runner.calls],
            ["list", "execute", "release", "stop_custom"],
        )
        self.assertTrue(runner.calls[1]["request"]["wait"])
        self.assertEqual(runner.calls[1]["request"]["completion_timeout_s"], 2.0)

    def test_bridge_timeout_error_keeps_existing_exception_type(self):
        runner = FakeSubprocessRunner()

        def timeout_after_list(command, **kwargs):
            request = json.loads(kwargs["input"])
            if request["op"] == "list":
                return FakeCompletedProcess({
                    "ok": True, "op": "list", "rc": 0, "actions": ACTION_LIST,
                })
            return FakeCompletedProcess({
                "ok": False,
                "error_type": "ArmActionTimeout",
                "error": "state wait timed out",
            }, returncode=1)

        backend = HighLevelArmActionBackend(
            execute_enabled=True,
            subprocess_runner=timeout_after_list,
        )
        with self.assertRaisesRegex(ArmActionTimeout, "state wait timed out"):
            backend.execute("forward_push", timeout_s=0.01)


if __name__ == "__main__":
    unittest.main()
