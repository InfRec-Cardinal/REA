"""High-level Unitree G1 arm-action backend.

Only the SDK2 ``arm`` RPC service and its read-only action-state topic are
used here.  This module never publishes lowcmd, joint targets, or torque.
The SDK is imported lazily so policy and unit-test processes can import this
module on machines where Unitree's runtime is not installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence


ARM_ACTION_STATE_TOPIC = "rt/arm/action/state"
EXECUTE_CUSTOM_ACTION_API_ID = 7108
STOP_CUSTOM_ACTION_API_ID = 7113

# Extra sys.path entries for the Unitree SDK (unitree_sdk2_python and its dependencies), os.pathsep-separated in
# UNITREE_SDK2_PYTHONPATH; empty when the SDK is installed in the interpreter itself.
DEFAULT_SDK_PATHS = tuple(path for path in os.environ.get("UNITREE_SDK2_PYTHONPATH", "").split(os.pathsep) if path)
DEFAULT_BRIDGE_PYTHON = "/usr/bin/python3"
TRANSPORTS = {"auto", "bridge", "direct"}


class ArmActionError(RuntimeError):
    """Base error raised by the high-level arm-action backend."""


class ArmActionNotFound(ArmActionError):
    """The requested action name was not returned by GetActionList."""


class ArmActionTimeout(ArmActionError):
    """No matching terminal action-state update arrived in time."""


@dataclass(frozen=True)
class ArmAction:
    """One action advertised by the robot's arm service."""

    name: str
    action_id: Optional[int]
    custom: bool
    duration_s: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "id": self.action_id,
            "custom": self.custom,
            "duration_s": self.duration_s,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ArmActionState:
    """Parsed ``rt/arm/action/state`` payload."""

    name: str
    action_id: Optional[int]
    holding: bool
    status: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "id": self.action_id,
            "holding": self.holding,
            "status": self.status,
        }


def _normalise_name(name: str) -> str:
    return "_".join(str(name).strip().replace("-", " ").replace("_", " ").split()).casefold()


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    return int(value) if isinstance(value, int) else None


def _duration(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _iter_action_records(payload: Any) -> Iterable[Mapping[str, Any]]:
    """Yield records from the nested response used by current G1 firmware."""
    if isinstance(payload, Mapping):
        if isinstance(payload.get("name"), str):
            yield payload
            return
        for name, value in payload.items():
            if isinstance(value, int) and not isinstance(value, bool):
                yield {"name": str(name), "id": value}
            elif isinstance(value, (Mapping, list, tuple)):
                yield from _iter_action_records(value)
        return
    if isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _iter_action_records(item)


def parse_action_list(payload: Any) -> Dict[str, ArmAction]:
    """Parse GetActionList data into a normalized-name lookup.

    G1 firmware currently returns ``[preset_actions, custom_actions]``.  A
    preset record has an integer ``id``; a custom teach-action record has a
    ``name`` and usually a ``time`` but no id.
    """
    actions: Dict[str, ArmAction] = {}
    for record in _iter_action_records(payload):
        name = str(record["name"]).strip()
        action_id = _integer(record.get("id", record.get("action_id")))
        duration_s = _duration(record.get("time", record.get("duration_s")))
        action = ArmAction(
            name=name,
            action_id=action_id,
            custom=action_id is None,
            duration_s=duration_s,
            metadata=dict(record),
        )
        actions[_normalise_name(name)] = action
    if not actions:
        raise ArmActionError("GetActionList returned no named actions")
    return actions


def _message_data(payload: Any) -> Any:
    if isinstance(payload, (str, bytes, bytearray, Mapping)):
        return payload
    data = getattr(payload, "data", None)
    return data() if callable(data) else data


def parse_action_state(payload: Any) -> ArmActionState:
    """Parse a JSON string, mapping, or SDK ``String_`` state message."""
    data = _message_data(payload)
    if isinstance(data, (bytes, bytearray)):
        data = bytes(data).decode("utf-8")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError("arm action state is not valid JSON") from exc
    if not isinstance(data, Mapping):
        raise ValueError("arm action state must be a JSON object")

    name = data.get("name", data.get("action_name"))
    if not isinstance(name, str) or not name.strip():
        raise ValueError("arm action state has no action name")
    action_id = _integer(data.get("id", data.get("action_id")))
    holding_value = data.get("holding", False)
    if isinstance(holding_value, str):
        holding = holding_value.strip().casefold() == "true"
    else:
        holding = bool(holding_value)
    status_value = data.get("status", data.get("state"))
    status = str(status_value) if status_value is not None else None
    return ArmActionState(
        name=name.strip(),
        action_id=action_id,
        holding=holding,
        status=status,
        raw=dict(data),
    )


class HighLevelArmActionBackend:
    """Long-lived wrapper around Unitree's high-level G1 arm action service.

    ``execute_enabled`` defaults to false.  In that mode ``execute`` still
    resolves the requested name against GetActionList but sends no command.
    A fake client and subscriber may be injected without importing the SDK.
    """

    def __init__(
        self,
        *,
        execute_enabled: bool = False,
        client: Any = None,
        state_subscriber: Any = None,
        network_interface: str = "eth0",
        rpc_timeout_s: float = 5.0,
        completion_timeout_s: float = 15.0,
        sdk_paths: Sequence[str] = DEFAULT_SDK_PATHS,
        transport: str = "auto",
        bridge_python: str = DEFAULT_BRIDGE_PYTHON,
        bridge_worker_path: Optional[str] = None,
        subprocess_runner: Optional[Callable[..., Any]] = None,
    ) -> None:
        if transport not in TRANSPORTS:
            raise ValueError("transport must be auto, bridge, or direct")
        self.execute_enabled = bool(execute_enabled)
        self.network_interface = network_interface
        self.rpc_timeout_s = float(rpc_timeout_s)
        self.completion_timeout_s = float(completion_timeout_s)
        self.sdk_paths = tuple(sdk_paths)
        self.transport = (
            "bridge"
            if transport == "bridge"
            or (transport == "auto" and client is None and state_subscriber is None)
            else "direct"
        )
        self.bridge_python = bridge_python
        self.bridge_worker_path = bridge_worker_path or os.path.join(
            os.path.dirname(__file__), "arm_bridge_worker.py"
        )
        self._subprocess_runner = subprocess_runner or subprocess.run
        self._client = client
        self._subscriber = state_subscriber
        self._owns_client = client is None
        self._connected = False
        self._actions: Optional[Dict[str, ArmAction]] = None
        self._condition = threading.Condition()
        self._latest_state: Optional[ArmActionState] = None
        self._state_version = 0
        self._last_state_error: Optional[str] = None

    @property
    def latest_state(self) -> Optional[ArmActionState]:
        with self._condition:
            return self._latest_state

    @property
    def state_version(self) -> int:
        with self._condition:
            return self._state_version

    def _load_sdk(self) -> Any:
        for path in reversed(self.sdk_paths):
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

        return ChannelFactoryInitialize, ChannelSubscriber, G1ArmActionClient, String_

    def connect(self) -> "HighLevelArmActionBackend":
        if self._connected:
            return self

        if self.transport == "bridge":
            if not os.path.isfile(self.bridge_worker_path):
                raise ArmActionError(
                    "arm bridge worker is missing: %s" % self.bridge_worker_path
                )
            self._connected = True
            return self

        if self._owns_client:
            factory_init, subscriber_type, client_type, string_type = self._load_sdk()
            factory_init(0, self.network_interface)
            self._client = client_type()
            self._subscriber = subscriber_type(ARM_ACTION_STATE_TOPIC, string_type)

        if self._client is None:
            raise ArmActionError("arm action client is unavailable")
        if hasattr(self._client, "SetTimeout"):
            self._client.SetTimeout(self.rpc_timeout_s)
        if hasattr(self._client, "Init"):
            self._client.Init()

        # Python SDK releases may lag the C++ high-level arm API.  Register the
        # two official arm-service RPC ids when the public methods are absent.
        register = getattr(self._client, "_RegistApi", None)
        if callable(register):
            if not hasattr(self._client, "ExecuteCustomAction"):
                register(EXECUTE_CUSTOM_ACTION_API_ID, 0)
            if not hasattr(self._client, "StopCustomAction"):
                register(STOP_CUSTOM_ACTION_API_ID, 0)

        if self._subscriber is not None and hasattr(self._subscriber, "Init"):
            self._subscriber.Init(self._state_callback, 10)
        self._connected = True
        return self

    def close(self) -> None:
        if (
            self.transport == "direct"
            and self._subscriber is not None
            and hasattr(self._subscriber, "Close")
        ):
            self._subscriber.Close()
        self._connected = False

    def __enter__(self) -> "HighLevelArmActionBackend":
        return self.connect()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _bridge_call(
        self,
        request: Mapping[str, Any],
        *,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.connect()
        payload = dict(request)
        payload.update(
            {
                "network_interface": self.network_interface,
                "rpc_timeout_s": self.rpc_timeout_s,
                "sdk_paths": list(self.sdk_paths),
            }
        )
        process_timeout = (
            float(timeout_s) + self.rpc_timeout_s + 5.0
            if timeout_s is not None
            else self.rpc_timeout_s + 5.0
        )
        command = [self.bridge_python, self.bridge_worker_path]
        try:
            completed = self._subprocess_runner(
                command,
                input=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                text=True,
                capture_output=True,
                timeout=process_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ArmActionTimeout("arm bridge worker timed out") from exc
        except OSError as exc:
            raise ArmActionError("arm bridge worker could not start: %s" % exc) from exc

        stdout = str(getattr(completed, "stdout", "")).strip()
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as exc:
            stderr = str(getattr(completed, "stderr", "")).strip()
            detail = stderr or stdout or "no output"
            raise ArmActionError("arm bridge returned invalid JSON: %s" % detail) from exc
        if not isinstance(response, dict):
            raise ArmActionError("arm bridge response must be a JSON object")
        if response.get("ok") is not True:
            error = str(response.get("error", "arm bridge operation failed"))
            error_type = response.get("error_type")
            if error_type == "ArmActionTimeout":
                raise ArmActionTimeout(error)
            if error_type == "ArmActionNotFound":
                raise ArmActionNotFound(error)
            raise ArmActionError(error)
        if int(getattr(completed, "returncode", 0)) != 0:
            raise ArmActionError(
                "arm bridge exited with %s" % getattr(completed, "returncode", None)
            )
        return response

    def _get_action_list_payload(self) -> Any:
        if self.transport == "bridge":
            return self._bridge_call({"op": "list"}).get("actions")
        self.connect()
        result = self._client.GetActionList()
        if isinstance(result, tuple) and len(result) == 2:
            code, payload = result
        else:
            code, payload = 0, result
        if int(code) != 0:
            raise ArmActionError("GetActionList returned %s" % code)
        return payload

    def list_actions(self, *, refresh: bool = False) -> Dict[str, ArmAction]:
        if refresh or self._actions is None:
            self._actions = parse_action_list(self._get_action_list_payload())
        return dict(self._actions)

    def resolve_action(self, name: str, *, refresh: bool = False) -> ArmAction:
        key = _normalise_name(name)
        action = self.list_actions(refresh=refresh).get(key)
        if action is None:
            available = ", ".join(sorted(item.name for item in self._actions.values()))
            raise ArmActionNotFound("unknown arm action %r; available: %s" % (name, available))
        return action

    def update_action_state(self, payload: Any) -> ArmActionState:
        state = parse_action_state(payload)
        with self._condition:
            self._latest_state = state
            self._state_version += 1
            self._last_state_error = None
            self._condition.notify_all()
        return state

    def _state_callback(self, payload: Any) -> None:
        try:
            self.update_action_state(payload)
        except ValueError as exc:
            with self._condition:
                self._last_state_error = str(exc)
                self._condition.notify_all()

    @staticmethod
    def _state_is_terminal(state: ArmActionState) -> bool:
        if state.status is None:
            # The documented G1 topic is an action-state event containing only
            # id/name/holding; the arrival of that matching event is terminal.
            return True
        status = _normalise_name(state.status)
        return status not in {"pending", "running", "executing", "in_progress"}

    def wait_for_completion(
        self,
        name: str,
        *,
        timeout_s: Optional[float] = None,
        after_version: Optional[int] = None,
        require_released: bool = False,
    ) -> ArmActionState:
        timeout = self.completion_timeout_s if timeout_s is None else float(timeout_s)
        expected = _normalise_name(name)
        deadline = time.monotonic() + timeout
        with self._condition:
            baseline = self._state_version if after_version is None else int(after_version)
            while True:
                state = self._latest_state
                if (
                    self._state_version > baseline
                    and state is not None
                    and _normalise_name(state.name) == expected
                    and self._state_is_terminal(state)
                    and (not require_released or not state.holding)
                ):
                    return state
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = ": %s" % self._last_state_error if self._last_state_error else ""
                    raise ArmActionTimeout("timed out waiting for arm action %r%s" % (name, detail))
                self._condition.wait(remaining)

    def _execute_custom(self, name: str) -> int:
        method = getattr(self._client, "ExecuteCustomAction", None)
        if callable(method):
            return int(method(name))
        raw_call = getattr(self._client, "_Call", None)
        if not callable(raw_call):
            raise ArmActionError("SDK has no high-level custom-action RPC")
        result = raw_call(
            EXECUTE_CUSTOM_ACTION_API_ID,
            json.dumps({"action_name": name}, separators=(",", ":")),
        )
        return int(result[0] if isinstance(result, tuple) else result)

    def execute(
        self,
        name: str,
        *,
        wait: bool = True,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        action = self.resolve_action(name)
        if not self.execute_enabled:
            return {
                "ok": True,
                "dry_run": True,
                "executed": False,
                "action": action.to_dict(),
                "rc": None,
                "state": None,
                "transport": self.transport,
            }

        if self.transport == "bridge":
            completion_timeout = (
                self.completion_timeout_s if timeout_s is None else float(timeout_s)
            )
            op = "release" if _normalise_name(action.name) == "release_arm" else "execute"
            response = self._bridge_call(
                {
                    "op": op,
                    "name": action.name,
                    "wait": bool(wait),
                    "completion_timeout_s": completion_timeout,
                },
                timeout_s=completion_timeout if wait else None,
            )
            state_payload = response.get("state")
            state = (
                self.update_action_state(state_payload)
                if isinstance(state_payload, Mapping)
                else None
            )
            return {
                "ok": True,
                "dry_run": False,
                "executed": bool(response.get("executed", True)),
                "action": action.to_dict(),
                "rc": int(response.get("rc", 0)),
                "state": state.to_dict() if state is not None else None,
                "transport": self.transport,
            }

        baseline = self.state_version
        if action.custom:
            rc = self._execute_custom(action.name)
        else:
            rc = int(self._client.ExecuteAction(action.action_id))
        if rc != 0:
            raise ArmActionError("arm action %r returned %d" % (action.name, rc))

        state = None
        if wait:
            state = self.wait_for_completion(
                action.name,
                timeout_s=timeout_s,
                after_version=baseline,
            )
        return {
            "ok": True,
            "dry_run": False,
            "executed": True,
            "action": action.to_dict(),
            "rc": rc,
            "state": state.to_dict() if state is not None else None,
            "transport": self.transport,
        }

    def stop_custom_action(self) -> Dict[str, Any]:
        if not self.execute_enabled:
            return {
                "ok": True,
                "dry_run": True,
                "stopped": False,
                "rc": None,
                "transport": self.transport,
            }
        if self.transport == "bridge":
            response = self._bridge_call({"op": "stop_custom"})
            return {
                "ok": True,
                "dry_run": False,
                "stopped": bool(response.get("stopped", True)),
                "rc": int(response.get("rc", 0)),
                "transport": self.transport,
            }
        self.connect()
        method = getattr(self._client, "StopCustomAction", None)
        if callable(method):
            rc = int(method())
        else:
            raw_call = getattr(self._client, "_Call", None)
            if not callable(raw_call):
                raise ArmActionError("SDK has no high-level stop-custom-action RPC")
            result = raw_call(STOP_CUSTOM_ACTION_API_ID, "")
            rc = int(result[0] if isinstance(result, tuple) else result)
        if rc != 0:
            raise ArmActionError("StopCustomAction returned %d" % rc)
        return {
            "ok": True,
            "dry_run": False,
            "stopped": True,
            "rc": rc,
            "transport": self.transport,
        }

    def release(
        self,
        *,
        wait: bool = False,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Release a held preset arm action through the advertised id 99 action."""
        return self.execute("release_arm", wait=wait, timeout_s=timeout_s)


G1ArmActionBackend = HighLevelArmActionBackend


__all__ = [
    "ARM_ACTION_STATE_TOPIC",
    "ArmAction",
    "ArmActionError",
    "ArmActionNotFound",
    "ArmActionState",
    "ArmActionTimeout",
    "G1ArmActionBackend",
    "HighLevelArmActionBackend",
    "parse_action_list",
    "parse_action_state",
]
