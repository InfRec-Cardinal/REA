#!/usr/bin/env python3
"""Cooperatively pause frozen LIBERO programs at real control-step boundaries.

greenlet preserves the program's Python stack, including a partially executed SDK
primitive. Continuing does not replay code. Selecting another candidate abandons
that stack and starts its frozen program from the current physical state.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal

from greenlet import greenlet, GreenletExit

source = Path(__file__).with_name("libero_program_backend.py")
spec = importlib.util.spec_from_file_location("async_libero_base", source)
programs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(programs)


class ChunkedProgram:
    """A same-thread continuation; no simulator work happens between requests."""
    def __init__(self, low, execute, horizon):
        self.low = low
        self.horizon = horizon
        self.stop_at = 0
        self.original_step = low.handle.step
        self.fiber = greenlet(execute)
        self.low.handle.step = self.step

    def step(self, action):
        if self.low._sim_step_count >= self.horizon:
            raise programs.base.ProgramInterrupt("horizon")
        if self.low._sim_step_count >= self.stop_at:
            self.fiber.parent.switch({"boundary": "control_steps"})
        return self.original_step(action)

    def advance(self, control_steps):
        self.stop_at = min(self.horizon, self.low._sim_step_count + control_steps)
        result = self.fiber.switch()
        if self.fiber.dead:
            self.low.handle.step = self.original_step
        return result, self.fiber.dead

    def close(self):
        if not self.fiber.dead:
            self.fiber.throw(GreenletExit)
        self.low.handle.step = self.original_step


class AsyncLiberoBackend(programs.base.AspireBackend):
    def __init__(self, args, **kwargs):
        root = Path(os.environ["REA_PROGRAM_BUNDLE"])
        bundle = json.loads((root / "bundle.json").read_text())
        self.programs = {(task["suite"], task["task"]): {
            name: (root / candidate["path"]).read_text()
            for name, candidate in task["candidates"].items()
        } for task in bundle["tasks"]}
        self.continuation = None
        self.candidate = None
        super().__init__(args, **kwargs)
        self.gpu_assignment = programs.configure_egl_cuda_lane() if kwargs.get("env_factory") is None else None
        if kwargs.get("env_factory") is None:
            from aspire.sim.cap.integrations.vision import sam3
            from aspire.sim.cap.integrations.motion import pyroki
            sam3.SERVICE_URL = os.environ.get("ASYNC_SAM3_URL", sam3.SERVICE_URL)
            if os.environ.get("ASYNC_PYROKI_URL"):
                pyroki.init_pyroki.__defaults__ = (os.environ["ASYNC_PYROKI_URL"],)

    def stop_program(self):
        if self.continuation is not None:
            self.continuation.close()
            self.continuation = None
        self.candidate = None
        return {"stopped": True}

    def reset(self, request):
        self.stop_program()
        reply = super().reset(request)
        reply["gpuAssignment"] = self.gpu_assignment
        return reply

    def program_chunk(self, request):
        self._require_episode("program_chunk")
        if self.sealed:
            raise ValueError("Episode is sealed by final score")
        candidate = request["candidateId"]
        code = self.programs[(self.task["suite"], self.task["name"])][candidate]
        chunk_steps = int(request.get("controlSteps", 100))
        if chunk_steps < 1:
            raise ValueError("controlSteps must be positive")
        changed = self.candidate is not None and self.candidate != candidate
        if self.continuation is None or changed:
            self.stop_program()
            self.code_env._init_exec_globals()
            self.candidate = candidate
            self.continuation = ChunkedProgram(self.low, lambda: self.code_env._exec_user_code(code), self.horizon)
        before = int(self.low._sim_step_count)
        started = self.clock()
        previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(programs.base.ProgramInterrupt("wall_timeout")))
        signal.setitimer(signal.ITIMER_REAL, float(request.get("wall_timeout_s", 120)))
        try:
            result, done = self.continuation.advance(chunk_steps)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        elapsed = self.clock() - started
        self.wall_used_s += elapsed
        after = int(self.low._sim_step_count)
        reply = {"candidateId": candidate, "candidateChanged": changed, "programDone": done,
                 "steps_before": before, "steps_after": after, "steps": after - before,
                 "wall_s": elapsed, "boundary": "program_end" if done else "control_steps",
                 "exception": None, "truncated": after >= self.horizon}
        if done:
            reply.update({"stdout_tail": result.get("stdout", "")[-2000:],
                          "stderr_tail": result.get("stderr", "")[-2000:]})
            if not result.get("ok", False):
                reply["exception"] = reply["stderr_tail"].splitlines()[-1] if reply["stderr_tail"] else "program failed"
                reply["exception_class"] = "program" if all(self._probe_services().values()) else "infra"
            self.continuation = None
        return reply

    def dispatch(self, request):
        if request.get("command") == "program_chunk":
            return self.program_chunk(request)
        if request.get("command") == "stop_program":
            return self.stop_program()
        return super().dispatch(request)

    def close(self):
        self.stop_program()
        return super().close()


if __name__ == "__main__":
    programs.base.AspireBackend = AsyncLiberoBackend
    programs.base.main()
