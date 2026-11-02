import io
import json
import unittest

from agent_vla_async.arm_bridge_worker import handle_request, main


ACTION_LIST = [
    [{"id": 99, "name": "release_arm"}, {"id": 36, "name": "forward_push"}],
    [{"name": "Scratch_head", "time": 8.1}],
]


class FakeRuntime:
    instances = []

    def __init__(self, request):
        self.request = request
        self.executed = []
        self.stops = 0
        self.__class__.instances.append(self)

    def action_list(self):
        return ACTION_LIST

    def execute(self, name, wait, timeout_s):
        self.executed.append((name, wait, timeout_s))
        return {
            "ok": True,
            "executed": True,
            "rc": 0,
            "action": {"name": name},
            "state": None,
        }

    def stop_custom(self):
        self.stops += 1
        return {"ok": True, "stopped": True, "rc": 0}


class ArmBridgeWorkerTests(unittest.TestCase):
    def setUp(self):
        FakeRuntime.instances.clear()

    def test_handle_request_maps_release_to_advertised_release_name(self):
        result = handle_request(
            {"op": "release", "wait": False, "completion_timeout_s": 3.0},
            runtime_type=FakeRuntime,
        )
        runtime = FakeRuntime.instances[-1]
        self.assertEqual(result["op"], "release")
        self.assertEqual(runtime.executed, [("release_arm", False, 3.0)])

    def test_handle_request_lists_and_stops_custom_via_high_level_runtime(self):
        listed = handle_request({"op": "list"}, runtime_type=FakeRuntime)
        stopped = handle_request({"op": "stop_custom"}, runtime_type=FakeRuntime)
        self.assertEqual(listed["actions"], ACTION_LIST)
        self.assertTrue(stopped["stopped"])
        self.assertEqual(FakeRuntime.instances[-1].stops, 1)

    def test_main_emits_exactly_one_json_even_if_handler_prints(self):
        source = io.StringIO('{"op":"list"}')
        target = io.StringIO()

        def noisy_handler(request):
            print("vendor SDK diagnostic that must not escape")
            return {"ok": True, "op": request["op"], "actions": []}

        rc = main(stdin=source, stdout=target, handler=noisy_handler)
        lines = target.getvalue().splitlines()
        self.assertEqual(rc, 0)
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), {"ok": True, "op": "list", "actions": []})

    def test_main_error_is_also_one_json_object(self):
        source = io.StringIO("not-json")
        target = io.StringIO()
        rc = main(stdin=source, stdout=target)
        lines = target.getvalue().splitlines()
        self.assertEqual(rc, 1)
        self.assertEqual(len(lines), 1)
        self.assertFalse(json.loads(lines[0])["ok"])


if __name__ == "__main__":
    unittest.main()
