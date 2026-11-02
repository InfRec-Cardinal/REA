"""Workstation-side Agent brain client. Transport is an SSH connection to the robot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set

from .protocol import CONTROLLER_MODES, HYBRID_HIGH_LEVEL

# The robot's SSH destination (an alias from ~/.ssh/config works) and the g1vla entry point on it.
DEFAULT_HOST = os.environ.get("G1_SSH_HOST", "unitree@g1")
DEFAULT_REMOTE_CLI = os.environ.get("G1VLA_REMOTE_CLI", "g1vla")


class RemoteBrainClient:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        remote_cli: str = DEFAULT_REMOTE_CLI,
    ) -> None:
        self.host = host
        self.remote_cli = remote_cli
        self._process: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()

    def open(self) -> "RemoteBrainClient":
        if self._process is not None and self._process.poll() is None:
            return self
        self._process = subprocess.Popen(
            ["ssh", "-T", "-o", "BatchMode=yes", self.host, self.remote_cli, "rpc-stream"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        return self

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

    def __enter__(self) -> "RemoteBrainClient":
        return self.open()

    def __exit__(self, *args: Any) -> None:
        self.close()

    def call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            process = self.open()._process
            assert process is not None and process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                stderr = process.stderr.read().strip() if process.stderr is not None else ""
                raise RuntimeError(stderr or "persistent SSH RPC stream closed")
            return json.loads(line)


def _print(payload: Dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def _load_plan(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("plan file must contain one JSON object")
    return value


def terminal_event_for_intent(
    events: List[Dict[str, Any]],
    intent_id: str,
    terminal_kinds: Set[str],
) -> Optional[Dict[str, Any]]:
    """Return a terminal event only when it belongs to the active intent."""
    return next(
        (
            event
            for event in events
            if event.get("intent_id") == intent_id
            and event.get("kind") in terminal_kinds
        ),
        None,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="SSH destination of the robot (default G1_SSH_HOST or unitree@g1)")
    parser.add_argument("--remote-cli", default=DEFAULT_REMOTE_CLI, help="g1vla on the robot (default G1VLA_REMOTE_CLI or g1vla on PATH)")
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit")
    submit.add_argument("task")
    submit.add_argument("--ttl", type=float, default=120.0)
    submit.add_argument("--heartbeat-timeout", type=float, default=8.0)
    submit.add_argument("--inference-period", type=float, default=0.5)
    submit.add_argument("--policy-timeout", type=float)
    submit.add_argument("--execute", action="store_true")
    submit.add_argument("--plan-file")
    submit.add_argument(
        "--controller-mode", choices=CONTROLLER_MODES, default=HYBRID_HIGH_LEVEL
    )

    heartbeat = sub.add_parser("heartbeat")
    heartbeat.add_argument("intent_id")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("intent_id", nargs="?")
    sub.add_parser("status")
    sub.add_parser("capabilities")
    sub.add_parser("sonic-status")
    sonic_validate = sub.add_parser("sonic-validate")
    sonic_validate.add_argument("task")
    sub.add_parser("controller-status")
    controller_prepare = sub.add_parser("controller-prepare")
    controller_prepare.add_argument("controller_mode", choices=CONTROLLER_MODES)
    controller_stop = sub.add_parser("controller-stop")
    controller_stop.add_argument("controller_mode", nargs="?", choices=CONTROLLER_MODES)
    events = sub.add_parser("events")
    events.add_argument("--after", type=int, default=0)

    run = sub.add_parser("run")
    run.add_argument("task")
    run.add_argument("--seconds", type=float, default=15.0)
    run.add_argument("--heartbeat-every", type=float, default=2.0)
    run.add_argument("--inference-period", type=float, default=0.5)
    run.add_argument("--policy-timeout", type=float)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--plan-file")
    run.add_argument(
        "--controller-mode", choices=CONTROLLER_MODES, default=HYBRID_HIGH_LEVEL
    )

    args = parser.parse_args(argv)
    with RemoteBrainClient(host=args.host) as client:
        if args.command == "submit":
            plan = _load_plan(args.plan_file)
            request = {
                "op": "submit",
                "task": args.task,
                "ttl_s": args.ttl,
                "heartbeat_timeout_s": args.heartbeat_timeout,
                "inference_period_s": args.inference_period,
                "execute": args.execute,
                "controller_mode": args.controller_mode,
                "plan": plan,
            }
            if args.policy_timeout is not None:
                request["policy_wait_timeout_s"] = args.policy_timeout
            response = client.call(request)
        elif args.command == "heartbeat":
            response = client.call({"op": "heartbeat", "intent_id": args.intent_id})
        elif args.command == "cancel":
            response = client.call({"op": "cancel", "intent_id": args.intent_id})
        elif args.command == "status":
            response = client.call({"op": "status"})
        elif args.command == "capabilities":
            response = client.call({"op": "capabilities"})
        elif args.command == "sonic-status":
            response = client.call({"op": "sonic_status"})
        elif args.command == "sonic-validate":
            response = client.call({"op": "sonic_validate", "task": args.task})
        elif args.command == "controller-status":
            response = client.call({"op": "controller_status"})
        elif args.command == "controller-prepare":
            response = client.call(
                {
                    "op": "controller_prepare",
                    "controller_mode": args.controller_mode,
                }
            )
        elif args.command == "controller-stop":
            response = client.call(
                {
                    "op": "controller_stop",
                    "controller_mode": args.controller_mode,
                }
            )
        elif args.command == "events":
            response = client.call({"op": "events", "after": args.after})
        else:
            plan = _load_plan(args.plan_file)
            capabilities = client.call({"op": "capabilities"})
            baseline = client.call({"op": "status"})
            request = {
                "op": "submit",
                "task": args.task,
                "ttl_s": args.seconds + 10.0,
                "heartbeat_timeout_s": args.heartbeat_every * 2.5,
                "inference_period_s": args.inference_period,
                "execute": args.execute,
                "controller_mode": args.controller_mode,
                "plan": plan,
            }
            if args.policy_timeout is not None:
                request["policy_wait_timeout_s"] = args.policy_timeout
            response = client.call(request)
            if not response.get("ok"):
                response["capabilities"] = capabilities
                _print(response)
                return 1
            intent_id = response["intent"]["intent_id"]
            deadline = time.monotonic() + args.seconds
            last_sequence = int(
                baseline.get("status", {}).get("last_sequence", 0)
            )
            captured = []
            terminal_event = None
            terminal_kinds = {
                "EXECUTION_COMPLETE",
                "EXECUTION_FAILED",
                "PREVIEW_COMPLETE",
                "PREVIEW_FAILED",
                "EXPIRED",
                "PAUSED",
                "CANCELLED",
            }
            try:
                while time.monotonic() < deadline:
                    time.sleep(min(args.heartbeat_every, 0.5))
                    batch = client.call({"op": "events", "after": last_sequence})
                    captured.extend(batch.get("events", []))
                    last_sequence = batch.get("last_sequence", last_sequence)
                    terminal_event = terminal_event_for_intent(
                        captured, intent_id, terminal_kinds
                    )
                    if terminal_event is not None:
                        break
                    heartbeat = client.call({"op": "heartbeat", "intent_id": intent_id})
                    if not heartbeat.get("ok"):
                        break
            finally:
                status = client.call({"op": "status"})["status"]
                active = status.get("active")
                if active and active.get("intent_id") == intent_id:
                    client.call({"op": "cancel", "intent_id": intent_id})
            terminal_result = (
                terminal_event.get("detail", {}).get("result", {})
                if terminal_event is not None
                else {}
            )
            response = {
                "ok": bool(
                    terminal_event is not None
                    and terminal_event.get("kind") in {
                        "EXECUTION_COMPLETE", "PREVIEW_COMPLETE"
                    }
                    and terminal_result.get("ok", True)
                ),
                "kind": (
                    terminal_event.get("kind")
                    if terminal_event is not None
                    else "SESSION_TIMEOUT"
                ),
                "intent_id": intent_id,
                "capabilities": capabilities,
                "events": captured,
                "result": terminal_result,
                "status": client.call({"op": "status"})["status"],
            }
    _print(response)
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
