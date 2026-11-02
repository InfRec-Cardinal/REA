"""JSON-lines host for optional robot SDK and environment-service drivers.

A driver kind is one of the built-in names below or "package.module:ClassOrFactory" for your own driver: an object with
id, timing, actions (JSON-Schema action specs) and metadata, plus reset(task, seed), observe(), step(action, execute),
stop() and close(); see docs/extending/robots.md. The module must be importable by this interpreter (PYTHONPATH).
"""

import importlib
import json
import os
import signal
import sys


def make_driver(kind, options):
    if ":" in kind:
        module_name, _, attribute = kind.partition(":")
        factory = getattr(importlib.import_module(module_name), attribute, None)
        if not callable(factory):
            raise ValueError(f"{kind} does not name a driver class or factory")
        return factory(options)
    if kind == "aloha":
        from robot_drivers.aloha import AlohaDriver
        return AlohaDriver(options)
    if kind == "lerobot":
        from robot_drivers.lerobot import LeRobotDriver
        return LeRobotDriver(options)
    if kind == "openral":
        from robot_drivers.openral import OpenRALDriver
        return OpenRALDriver(options)
    if kind == "rpent":
        from robot_drivers.rpent import RPentDriver
        return RPentDriver(options)
    raise ValueError(f"Unknown robot driver: {kind}")


class RobotBackend:
    def __init__(self, driver_factory=make_driver):
        self.driver_factory = driver_factory
        self.driver = None
        self.execute_enabled = False

    def snapshot(self, value):
        return {**value, "actions": self.driver.actions, "metadata": self.driver.metadata}

    def dispatch(self, request):
        operation = request["op"]
        if operation == "configure":
            if self.driver is not None:
                raise ValueError("Robot backend is already configured")
            options = request.get("options", {})
            self.execute_enabled = options.get("executeEnabled", False) is True
            self.driver = self.driver_factory(request["driver"], options)
            return {"id": self.driver.id, "timing": self.driver.timing, "actions": self.driver.actions, "metadata": self.driver.metadata}
        if operation == "close":
            self.close()
            return {"closed": True}
        if self.driver is None:
            raise ValueError("Configure the robot backend first")
        if operation == "reset":
            return self.snapshot(self.driver.reset(task=request.get("task"), seed=request.get("seed", 0)))
        if operation == "observe":
            return self.snapshot(self.driver.observe())
        if operation == "step":
            execute = request.get("execute", False) is True
            if execute and not self.execute_enabled:
                raise ValueError("Robot backend was configured for preview")
            return self.snapshot(self.driver.step(request["action"], execute=execute))
        if operation == "stop":
            return self.driver.stop() if self.execute_enabled else {"supported": False, "preview": True}
        raise ValueError(f"Unknown robot operation: {operation}")

    def close(self):
        if self.driver is not None:
            driver, self.driver = self.driver, None
            driver.close()


def main():
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    backend = RobotBackend()

    def terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    protocol_output.write(json.dumps({"event": "ready"}) + "\n")
    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                result = backend.dispatch(request)
                encoded = json.dumps({"id": request.get("id"), "ok": True, "result": result}, allow_nan=False)
            except Exception as error:
                print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
                encoded = json.dumps({"id": request.get("id"), "ok": False, "error": str(error)})
            protocol_output.write(encoded + "\n")
            if request.get("op") == "close":
                break
    finally:
        backend.close()


if __name__ == "__main__":
    main()
