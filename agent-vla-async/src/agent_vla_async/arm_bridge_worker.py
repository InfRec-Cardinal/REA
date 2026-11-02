"""One-shot system-Python bridge for Unitree's high-level G1 arm service.

The Python 3.11 cerebellum invokes this file with ``/usr/bin/python3`` because
the deployed CycloneDDS extension is built for the robot's system Python 3.8.
One JSON request is read from stdin and exactly one JSON response is emitted.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Optional


EXECUTE_CUSTOM_ACTION_API_ID = 7108
STOP_CUSTOM_ACTION_API_ID = 7113
STATE_TOPIC = "rt/arm/action/state"
# Extra sys.path entries for the Unitree SDK (unitree_sdk2_python and its dependencies), os.pathsep-separated in
# UNITREE_SDK2_PYTHONPATH; empty when the SDK is installed in the interpreter itself.
DEFAULT_SDK_PATHS = tuple(path for path in os.environ.get("UNITREE_SDK2_PYTHONPATH", "").split(os.pathsep) if path)
ALLOWED_OPERATIONS = {"list", "execute", "release", "stop_custom"}


class ArmActionNotFound(RuntimeError):
    pass


class ArmActionTimeout(RuntimeError):
    pass


def _normalise_name(name: str) -> str:
    return "_".join(str(name).strip().replace("-", " ").replace("_", " ").split()).casefold()


def _iter_records(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if isinstance(payload.get("name"), str):
            yield payload
            return
        for name, value in payload.items():
            if isinstance(value, int) and not isinstance(value, bool):
                yield {"name": str(name), "id": value}
            elif isinstance(value, (Mapping, list, tuple)):
                yield from _iter_records(value)
        return
    if isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _iter_records(item)


def _find_action(payload: Any, name: str) -> Dict[str, Any]:
    expected = _normalise_name(name)
    for record in _iter_records(payload):
        if _normalise_name(str(record.get("name", ""))) == expected:
            return dict(record)
    raise ArmActionNotFound("arm action %r was not returned by GetActionList" % name)


def _rpc_code(result: Any) -> int:
    return int(result[0] if isinstance(result, tuple) else result)


def _state_payload(message: Any) -> Optional[Dict[str, Any]]:
    if message is None:
        return None
    data = getattr(message, "data", message)
    if callable(data):
        data = data()
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    if isinstance(data, str):
        data = json.loads(data)
    return dict(data) if isinstance(data, Mapping) else None


def _terminal_state(state: Mapping[str, Any]) -> bool:
    value = state.get("status", state.get("state"))
    if value is None:
        return True
    return _normalise_name(str(value)) not in {
        "pending", "running", "executing", "in_progress"
    }


class SdkArmRuntime:
    """System-Python owner of one high-level SDK2 arm RPC session."""

    def __init__(self, request: Mapping[str, Any]) -> None:
        sdk_paths = request.get("sdk_paths", DEFAULT_SDK_PATHS)
        if not isinstance(sdk_paths, (list, tuple)):
            sdk_paths = DEFAULT_SDK_PATHS
        for path in reversed(tuple(str(item) for item in sdk_paths)):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

        from unitree_sdk2py.core.channel import (  # type: ignore
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.g1.arm.g1_arm_action_client import (  # type: ignore
            G1ArmActionClient,
        )
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_  # type: ignore

        interface = str(request.get("network_interface", "eth0"))
        ChannelFactoryInitialize(0, interface)
        self._subscriber_type = ChannelSubscriber
        self._string_type = String_
        self.client = G1ArmActionClient()
        self.client.SetTimeout(float(request.get("rpc_timeout_s", 5.0)))
        self.client.Init()
        register = getattr(self.client, "_RegistApi", None)
        if callable(register):
            if not hasattr(self.client, "ExecuteCustomAction"):
                register(EXECUTE_CUSTOM_ACTION_API_ID, 0)
            if not hasattr(self.client, "StopCustomAction"):
                register(STOP_CUSTOM_ACTION_API_ID, 0)

    def action_list(self) -> Any:
        result = self.client.GetActionList()
        if isinstance(result, tuple) and len(result) == 2:
            code, payload = result
        else:
            code, payload = 0, result
        if int(code) != 0:
            raise RuntimeError("GetActionList returned %s" % code)
        return payload

    def _execute_custom(self, name: str) -> int:
        method = getattr(self.client, "ExecuteCustomAction", None)
        if callable(method):
            return int(method(name))
        result = self.client._Call(
            EXECUTE_CUSTOM_ACTION_API_ID,
            json.dumps({"action_name": name}, separators=(",", ":")),
        )
        return _rpc_code(result)

    def _wait_for_state(
        self,
        subscriber: Any,
        name: str,
        timeout_s: float,
    ) -> Dict[str, Any]:
        expected = _normalise_name(name)
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ArmActionTimeout("timed out waiting for arm action %r" % name)
            state = _state_payload(subscriber.Read(remaining))
            if (
                state is not None
                and _normalise_name(str(state.get("name", state.get("action_name", ""))))
                == expected
                and _terminal_state(state)
            ):
                return state

    def execute(self, name: str, wait: bool, timeout_s: float) -> Dict[str, Any]:
        actions = self.action_list()
        action = _find_action(actions, name)
        action_id = action.get("id", action.get("action_id"))
        custom = not isinstance(action_id, int) or isinstance(action_id, bool)
        subscriber = None
        if wait:
            subscriber = self._subscriber_type(STATE_TOPIC, self._string_type)
            subscriber.Init()
        try:
            rc = (
                self._execute_custom(str(action["name"]))
                if custom
                else int(self.client.ExecuteAction(int(action_id)))
            )
            if rc != 0:
                raise RuntimeError("arm action %r returned %d" % (name, rc))
            state = (
                self._wait_for_state(subscriber, str(action["name"]), timeout_s)
                if subscriber is not None
                else None
            )
            return {
                "ok": True,
                "executed": True,
                "rc": rc,
                "action": {
                    "name": str(action["name"]),
                    "id": int(action_id) if not custom else None,
                    "custom": custom,
                },
                "state": state,
            }
        finally:
            if subscriber is not None:
                subscriber.Close()

    def stop_custom(self) -> Dict[str, Any]:
        method = getattr(self.client, "StopCustomAction", None)
        if callable(method):
            rc = int(method())
        else:
            rc = _rpc_code(self.client._Call(STOP_CUSTOM_ACTION_API_ID, ""))
        if rc != 0:
            raise RuntimeError("StopCustomAction returned %d" % rc)
        return {"ok": True, "stopped": True, "rc": rc}


def handle_request(
    request: Mapping[str, Any],
    runtime_type: Any = SdkArmRuntime,
) -> Dict[str, Any]:
    operation = request.get("op")
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError("unsupported arm bridge operation")
    runtime = runtime_type(request)
    if operation == "list":
        return {"ok": True, "op": operation, "rc": 0, "actions": runtime.action_list()}
    if operation == "stop_custom":
        result = runtime.stop_custom()
        result["op"] = operation
        return result

    name = "release_arm" if operation == "release" else request.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("execute operation requires an action name")
    result = runtime.execute(
        name,
        bool(request.get("wait", True)),
        float(request.get("completion_timeout_s", 15.0)),
    )
    result["op"] = operation
    return result


def main(
    stdin: Any = None,
    stdout: Any = None,
    handler: Any = handle_request,
) -> int:
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    try:
        request = json.loads(input_stream.read())
        if not isinstance(request, dict):
            raise ValueError("arm bridge request must be a JSON object")
        diagnostics = io.StringIO()
        with contextlib.redirect_stdout(diagnostics), contextlib.redirect_stderr(diagnostics):
            response = handler(request)
        code = 0 if response.get("ok") is True else 1
    except Exception as exc:
        response = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        code = 1
    output_stream.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
    output_stream.flush()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
