"""JSON-lines program backend embedding ASPIRE's BEHAVIOR-1K (R1Pro, OmniGibson) code-execution env.

Same wire contract as scripts/aspire_backend.py (docs/extending/program-backends.md): one JSON object
per stdin line `{id, command, ...}`, one reply per line. Commands: catalog, reset, run_program, run_agent, observe,
score, close.

Protocol invariant (identical to the LIBERO backend): nothing on the policy/judge/recovery path reads the
task predicate, the reward or object poses. The only success read is `score` (ASPIRE's own
`task_completed()`), called once per episode; it seals the episode. `run_program` executes through
`CodeExecutionEnvBase._exec_user_code`, never `step` (which computes reward and task_completed), and
programs are refused when they name the low-level env, the observation dict or the interpreter (see
FORBIDDEN_IDENTIFIERS): a frozen program may only call ASPIRE's R1ProControlApi functions.

Benchmark tasks: five R1Pro navigate-and-pick-up tasks from eval/behavior/runner/behavior_tasks.py
(trash, radio, teddy, drill and camera);
task instances are the 2025-challenge instances, `seed` = instance id; held-out seeds 1-25, development
seeds 26-35. Success = ASPIRE's task_completed() for the task (object in hand and lifted).

ASPIRE is imported lazily inside the env recipe so the module is unit-testable without OmniGibson (a fake
code env is injected through `env_factory`).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
import re
import signal
import sys
import time
import tokenize
import tempfile
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("aspire_backend_shared", os.path.join(_HERE, "aspire_backend.py"))
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
ProgramInterrupt = _shared.ProgramInterrupt
classify_result = _shared.classify_result
compose_montage = _shared.compose_montage
parse_frame_size = _shared.parse_frame_size
parse_ports = _shared.parse_ports
probe_ports = _shared.probe_ports
handle_request = _shared.handle_request
serve = _shared.serve
_positive_int = _shared._positive_int
_positive_number = _shared._positive_number
_forbidden_pattern = _shared._forbidden_pattern
WALL_TIMEOUT_REARM_S = _shared.WALL_TIMEOUT_REARM_S
SHA256_HEX = _shared.SHA256_HEX

SUITE = "behavior"
# task -> ASPIRE config (relative to --aspire-root), OmniGibson activity, and the instruction the judge sees.
_task_spec = importlib.util.spec_from_file_location("behavior_tasks", os.path.join(_HERE, "..", "eval", "behavior", "runner", "behavior_tasks.py"))
_task_module = importlib.util.module_from_spec(_task_spec)
_task_spec.loader.exec_module(_task_module)
TASKS = {name: {key: _task_module.TASKS5[name][key] for key in ("config", "activity", "language")}
         for name in _task_module.TASKS5_ACTIVE}
TASK_NAMES = tuple(TASKS)
HELDOUT_SEEDS = range(1, 26)
DEVELOPMENT_SEEDS = range(26, 36)
RESET_SEMANTICS = ("aspire",)
PHASES = ("development", "heldout")
VIEWS = ("agentview", "montage", "front")
# The ASPIRE checkout with BEHAVIOR-1K support (see scripts/aspire_backend.py).
DEFAULT_ASPIRE_ROOT = os.environ.get("ASPIRE_ROOT")
DEFAULT_PYTHON_ROOT = os.environ.get("ASPIRE_PYTHON_ROOT")
DEFAULT_API = "R1ProControlApi"
DEFAULT_JUDGE_FRAME_SIZE = "512x512"
DEFAULT_INFRA_PROBE_PORTS = "8114,8115"
HEAD_CAMERA = "zed_link:Camera:0"
LEFT_WRIST_CAMERA = "left_realsense_link:Camera:0"
MONTAGE_TITLES = ("Head camera", "Left wrist camera")

# Whole-identifier matches on comment-stripped code. `env`, `obs`, `INPUTS` and `APIS` are the exec
# globals through which a program could reach the OmniGibson env, the task's low_dim object states or
# the API objects' env handle; the rest are the privileged reward/pose helpers of R1ProBehaviourLowLevel
# and the interpreter escape hatches. ASPIRE's own API functions (find_object_*, get_object_pose,
# navigate_to_pose, sample_grasp_pose, grasp_object, check_object_in_hand, ...) stay callable.
FORBIDDEN_IDENTIFIERS = (
    "env", "obs", "INPUTS", "APIS", "RESULT",
    "task_completed", "compute_reward", "pick_up_radio_reward", "pick_up_trash_reward", "pick_up_target_reward", "get_object_state",
    "PICKUP_TARGETS", "initial_pickup_poses", "_pickup_target_objects",
    "object_registry", "object_scope", "task_relevant_obj", "_get_obj_in_hand", "_grasp_obj", "_navigate_to_pose",
    "initial_radio_pose", "initial_trash_pose", "load_task_instance", "low_level_env", "low_dim",
    "omnigibson", "og.sim", "get_position_orientation", "controller",
    "getattr", "setattr", "__import__", "importlib", "eval(", "exec(", "globals(", "locals(", "vars(", "__dict__", "__class__",
)
FORBIDDEN_PATTERNS = tuple((name, _forbidden_pattern(name)) for name in FORBIDDEN_IDENTIFIERS)


FSTRING_PREFIX = re.compile(r"^[rRbBuU]*[fF]")


def strip_comments(code):
    """`code` without comments and plain string literals: neither can reach an object (getattr/eval/exec are
    forbidden), so a comment or a message mentioning `env` is not a privileged read. f-strings are kept
    because on Python < 3.12 their embedded expressions are part of the STRING token."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return code
    out = []
    for kind, text, _start, _end, _line in tokens:
        if kind == tokenize.COMMENT:
            continue
        if kind == tokenize.STRING and not FSTRING_PREFIX.match(text):
            out.append('""')
            continue
        out.append(text if kind not in (tokenize.NEWLINE, tokenize.NL) else "\n")
    return " ".join(out)


def scan_program(code):
    """First forbidden identifier used by `code` (comments ignored) or None."""
    stripped = strip_comments(code)
    for name, pattern in FORBIDDEN_PATTERNS:
        if pattern.search(stripped):
            return name
    return None


# >>> ASPIRE ENV RECIPE (the only place that knows how ASPIRE's BEHAVIOR env is built) >>>
def build_code_env(aspire_root, config_relpath, api_name):
    """Instantiate the env block of an ASPIRE r1pro YAML exactly like ASPIRE's runner (_worker_setup)."""
    import copy

    import yaml

    from aspire.sim.cap.envs.configs.instantiate import instantiate

    with open(os.path.join(aspire_root, config_relpath), encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    env_factory = copy.deepcopy(config["env"])
    env_factory["cfg"]["apis"] = [api_name]
    env_factory["cfg"]["privileged"] = False
    return instantiate(env_factory)
# <<< ASPIRE ENV RECIPE <<<


def _to_uint8_rgb(tensor):
    import numpy as np

    array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    return np.ascontiguousarray(array[..., :3].astype(np.uint8))


_AGENT_MODULE = None


def online_agent_module():
    global _AGENT_MODULE
    if _AGENT_MODULE is None:
        spec = importlib.util.spec_from_file_location("behavior_online_agent", os.path.join(_HERE, "..", "eval", "behavior", "runner", "agents.py"))
        _AGENT_MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_AGENT_MODULE)
    return _AGENT_MODULE


def online_api_functions(code_env):
    from aspire.sim.cap.envs.launch_b1k import _find_apis

    functions = {}
    for api in _find_apis(code_env).values():
        functions.update(api.functions())
    return functions


def online_agent_credentials():
    """Keep each credential paired with its provider's endpoint."""
    if os.environ.get("REA_API_KEY"):
        base = os.environ.get("REA_BASE_URL")
        if not base:
            raise ValueError("REA_BASE_URL is required with REA_API_KEY")
        return os.environ["REA_API_KEY"], base
    if os.environ.get("AGENT_API_KEY"):
        base = os.environ.get("AGENT_BASE_URL")
        if not base:
            raise ValueError("AGENT_BASE_URL is required with AGENT_API_KEY")
        return os.environ["AGENT_API_KEY"], base
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"], os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    raise ValueError("Set REA_API_KEY, AGENT_API_KEY or OPENAI_API_KEY for the online agent")


def run_online_agent(**kwargs):
    """Separate adapter so backend tests can exercise tool execution without model requests."""
    return online_agent_module().run_episode(**kwargs)


class BehaviorBackend:
    def __init__(self, args, env_factory=None, clock=time.perf_counter, now=None, service_probe=None, agent_runner=None, api_functions=None):
        self.args = args
        self.agent_runner = agent_runner or run_online_agent
        self.api_functions = api_functions or online_api_functions
        self.env_factory = env_factory or (lambda task: build_code_env(args.aspire_root, TASKS[task]["config"], args.api))
        self.clock = clock
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.service_probe = service_probe or (lambda: probe_ports(parse_ports(args.infra_probe_ports)))
        self.judge_frame_size = parse_frame_size(args.judge_frame_size)
        self.code_env = None
        self.low = None
        self.identity = None
        self.task = None
        self.episode = 0
        self.observation_number = 0
        self.horizon = int(args.max_steps)
        self.sim_steps = 0
        self.sealed = False
        self.scored = None
        self.phase = None
        self.reset_semantics = None
        self.wall_budget_s = None
        self.wall_used_s = 0.0
        self._running = False
        self._in_sim_step = False
        self._interrupt = None
        self._guarded_env = None
        self._original_step = None
        self._alarm_installed = False
        self._previous_alarm = None

    # ---- catalog ----
    def catalog(self):
        return {
            "suites": [{"name": SUITE, "family": "behavior", "tasks": [{"index": i, "name": name} for i, name in enumerate(TASK_NAMES)]}],
            "seeds": {"heldout": [HELDOUT_SEEDS.start, HELDOUT_SEEDS.stop - 1], "development": [DEVELOPMENT_SEEDS.start, DEVELOPMENT_SEEDS.stop - 1]},
            "backend": {
                "benchmark": SUITE, "api": self.args.api, "maxSteps": int(self.args.max_steps), "frameSize": int(self.args.frame_size),
                "judgeFrameSize": list(self.judge_frame_size), "aspireRoot": os.path.abspath(self.args.aspire_root) if self.args.aspire_root else None,
                "pythonRoot": os.path.abspath(self.args.python_root) if self.args.python_root else None, "configs": {name: spec["config"] for name, spec in TASKS.items()},
                "perceptionRender": "pathtracing" if os.environ.get("REA_BEHAVIOR_PT_OBS") else "default",
            },
        }

    # ---- reset ----
    def reset(self, request):
        suite = request.get("suite")
        if suite != SUITE:
            raise ValueError(f"Unsupported suite: {suite} (this backend serves '{SUITE}')")
        task = request.get("task")
        if isinstance(task, int) and not isinstance(task, bool):
            if not 0 <= task < len(TASK_NAMES):
                raise ValueError(f"Task index out of range: {task}")
            task = TASK_NAMES[task]
        if task not in TASKS:
            raise ValueError(f"Task not found in {SUITE}: {task} (expected one of {', '.join(TASK_NAMES)})")
        seed = request.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not (seed in HELDOUT_SEEDS or seed in DEVELOPMENT_SEEDS):
            raise ValueError(f"seed must be a task-instance id in {HELDOUT_SEEDS.start}-{HELDOUT_SEEDS.stop - 1} (held-out) or {DEVELOPMENT_SEEDS.start}-{DEVELOPMENT_SEEDS.stop - 1} (development)")
        horizon = _positive_int(request.get("horizon", int(self.args.max_steps)), "horizon")
        reset_semantics = request.get("reset", "aspire")
        if reset_semantics not in RESET_SEMANTICS:
            raise ValueError(f"Unknown reset semantics: {reset_semantics} (BEHAVIOR instances are fixed; only 'aspire' exists)")
        phase = request.get("phase")
        if phase is not None:
            if phase not in PHASES:
                raise ValueError(f"Unknown phase: {phase}")
            if phase == "heldout" and seed not in HELDOUT_SEEDS:
                raise ValueError(f"heldout phase requires seeds {HELDOUT_SEEDS.start}-{HELDOUT_SEEDS.stop - 1}")
            if phase == "development" and seed not in DEVELOPMENT_SEEDS:
                raise ValueError(f"development phase requires seeds {DEVELOPMENT_SEEDS.start}-{DEVELOPMENT_SEEDS.stop - 1}")
        wall_budget_s = request.get("wall_budget_s")
        if wall_budget_s is not None:
            wall_budget_s = _positive_number(wall_budget_s, "wall_budget_s")

        identity = (task, self.args.api)
        if self.code_env is None or identity != self.identity:
            if self.code_env is not None:
                # OmniGibson hosts one scene per process; a lane serves one task (the runner's tasks files are per task).
                raise ValueError(f"this backend process already hosts task {self.identity[0]}; start another backend for {task}")
            self.code_env = self.env_factory(task)
            self.low = self.code_env.low_level_env
            self.identity = identity
        self.task = None
        self.sealed = False
        self.scored = None
        self.sim_steps = 0
        self.code_env.reset(seed=seed)
        if hasattr(self.low, "_sim_step_count"):
            self.low._sim_step_count = 0
        self.reset_semantics = reset_semantics
        self.phase = phase
        self.wall_budget_s = wall_budget_s
        self.wall_used_s = 0.0
        self.horizon = horizon
        self.episode += 1
        self.observation_number = 0
        self.task = {
            "suite": SUITE, "index": TASK_NAMES.index(task), "name": task, "activity": TASKS[task]["activity"],
            "language": TASKS[task]["language"], "seed": seed, "instance_id": seed,
        }
        reply = self.observe({})
        reply.update({
            "initialStateSource": "behavior_2025_task_instance",
            "initialStateIndex": seed,
            "resetSemantics": reset_semantics,
            "phase": phase,
            "wallBudgetS": wall_budget_s,
            "episode": self.episode,
        })
        return reply

    # ---- run_program ----
    def _require_episode(self, verb):
        if self.code_env is None or self.task is None:
            raise ValueError(f"Reset an episode before {verb}")

    def _alarm(self, signum, frame):
        if self._running:
            self._interrupt = "wall_timeout"
            if not self._in_sim_step:
                raise ProgramInterrupt("wall_timeout")

    def _call_sim_step(self, original_step, action):
        # Raising from a PhysX contact callback crosses pybind11 and aborts the process.
        # Defer that interruption until the native simulator step returns to Python.
        self._in_sim_step = True
        try:
            result = original_step(action)
        finally:
            self._in_sim_step = False
        if self._interrupt == "wall_timeout":
            raise ProgramInterrupt("wall_timeout")
        return result

    def _teardown_program(self):
        self._running = False
        self._in_sim_step = False
        if self._alarm_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._previous_alarm if self._previous_alarm is not None else signal.SIG_DFL)
            self._alarm_installed = False
            self._previous_alarm = None
        if self._guarded_env is not None:
            if self._original_step is not None:
                self._guarded_env.step = self._original_step
            self._guarded_env = None
            self._original_step = None

    def _check_program_pin(self, allowed, code, digest):
        if allowed is None:
            return
        if not isinstance(allowed, list) or not all(isinstance(item, str) and SHA256_HEX.match(item) for item in allowed):
            raise ValueError("allowed_sha256 must be a list of sha256 hex digests")
        if self.phase == "heldout" and not allowed:
            raise ValueError("heldout phase requires a non-empty allowed_sha256 pin")
        if allowed and digest not in allowed:
            raise ValueError("program sha256 is not in allowed_sha256 (audited-program pin)")

    def _effective_wall_timeout(self, requested):
        if self.wall_budget_s is None:
            return requested, None
        left = max(0.0, round(self.wall_budget_s - self.wall_used_s, 3))
        return (min(requested, left) if left > 0 else requested), left

    def _probe_services(self):
        try:
            probe = self.service_probe() or {}
        except Exception as error:  # a broken probe must not turn into an episode error
            print(f"service probe failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            return {}
        return {str(port): bool(up) for port, up in probe.items()}

    def _run_reply(self, which, digest, steps_before, steps_after, wall_s, classified, wall_timeout_s, probe):
        wall_left = None if self.wall_budget_s is None else max(0.0, round(self.wall_budget_s - self.wall_used_s, 3))
        return {
            "which": which, "steps_before": steps_before, "steps_after": steps_after, "steps": steps_after - steps_before,
            "wall_s": wall_s, "exception": classified["exception"], "exception_class": classified["exception_class"],
            "truncated": classified["truncated"], "truncated_reason": classified["truncated_reason"], "horizon": self.horizon,
            "sha256": digest, "stdout_tail": classified["stdout_tail"], "stderr_tail": classified["stderr_tail"],
            "wall_timeout_s": wall_timeout_s, "wall_used_s": self.wall_used_s, "wall_budget_s": self.wall_budget_s,
            "wall_left_s": wall_left, "infra_probe": probe,
        }

    def run_program(self, request):
        self._require_episode("run_program")
        if self.sealed:
            raise ValueError("Episode is sealed by score; reset to continue")
        which = request.get("which", "custom")
        if not isinstance(which, str) or not which:
            raise ValueError("which must be a non-empty string")
        path = request.get("path")
        code = request.get("code")
        if (path is None) == (code is None):
            raise ValueError("exactly one of path or code is required")
        if path is not None:
            if not isinstance(path, str) or not os.path.isabs(path):
                raise ValueError(f"program path must be absolute: {path}")
            if not os.path.isfile(path):
                raise ValueError(f"program file not found: {path}")
            with open(path, encoding="utf-8") as handle:
                code = handle.read()
        if not isinstance(code, str):
            raise ValueError("code must be a string")
        forbidden = scan_program(code)
        if forbidden is not None:
            raise ValueError(f"program uses forbidden identifier: {forbidden}")
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        self._check_program_pin(request.get("allowed_sha256"), code, digest)
        max_steps = request.get("max_steps")
        if max_steps is not None:
            max_steps = _positive_int(max_steps, "max_steps")
        requested_timeout = request.get("wall_timeout_s")
        if requested_timeout is None:
            requested_timeout = self.args.wall_timeout_s
        requested_timeout = _positive_number(requested_timeout, "wall_timeout_s")
        wall_timeout_s, wall_left = self._effective_wall_timeout(requested_timeout)

        horizon = self.horizon
        steps_before = int(self.sim_steps)
        if wall_left is not None and wall_left <= 0:
            classified = {"exception": None, "exception_class": "timeout", "truncated": True, "truncated_reason": "wall_budget", "stdout_tail": "", "stderr_tail": ""}
            return self._run_reply(which, digest, steps_before, steps_before, 0.0, classified, wall_timeout_s, None)
        stop_at = horizon if max_steps is None else min(horizon, steps_before + max_steps)
        inner = self.low.env  # the OmniGibson environment every primitive steps through
        original_step = inner.step
        backend = self

        def guarded_step(action):
            if backend._interrupt == "wall_timeout":
                raise ProgramInterrupt("wall_timeout")
            if backend.sim_steps >= stop_at:
                reason = "horizon" if stop_at == horizon else "max_steps"
                backend._interrupt = reason
                raise ProgramInterrupt(reason)
            backend.sim_steps += 1
            if hasattr(backend.low, "_sim_step_count"):
                backend.low._sim_step_count = backend.sim_steps
            return backend._call_sim_step(original_step, action)

        self._interrupt = None
        result = None
        late_interrupt = None
        started = self.clock()
        try:
            try:
                self._previous_alarm = signal.signal(signal.SIGALRM, self._alarm)
                self._alarm_installed = True
                self._guarded_env = inner
                self._original_step = original_step
                inner.step = guarded_step
                self._running = True
                signal.setitimer(signal.ITIMER_REAL, float(wall_timeout_s), WALL_TIMEOUT_REARM_S)
                try:
                    result = self.code_env._exec_user_code(code)
                except ProgramInterrupt as interrupt:
                    result = {"ok": False, "stdout": "", "stderr": f"ProgramInterrupt: {interrupt.reason}\n", "result": None}
            finally:
                self._teardown_program()
        except ProgramInterrupt as interrupt:
            late_interrupt = interrupt
            self._teardown_program()
        if result is None:
            reason = late_interrupt.reason if late_interrupt is not None else "wall_timeout"
            result = {"ok": False, "stdout": "", "stderr": f"ProgramInterrupt: {reason}\n", "result": None}
        wall_s = round(self.clock() - started, 3)
        self.wall_used_s = round(self.wall_used_s + wall_s, 3)
        steps_after = int(self.sim_steps)
        classified = classify_result(result, steps_after, horizon, self._interrupt, int(self.args.tail_chars))
        probe = None
        if classified["exception"] is not None or classified["truncated_reason"] == "wall_timeout":
            probe = self._probe_services()
            down = sorted(port for port, up in probe.items() if not up)
            if down:
                classified["exception_class"] = "infra"
                if classified["exception"] is not None:
                    classified["exception"]["class"] = "infra"
                    classified["exception"]["infra_reason"] = f"service port(s) not listening: {', '.join(down)}"
        return self._run_reply(which, digest, steps_before, steps_after, wall_s, classified, wall_timeout_s, probe)

    # ---- online Agents SDK continuation ----
    def run_agent(self, request):
        self._require_episode("run_agent")
        if self.sealed:
            raise ValueError("Episode is sealed by score; reset to continue")
        requested = _positive_number(request.get("wall_timeout_s", self.args.wall_timeout_s), "wall_timeout_s")
        timeout, wall_left = self._effective_wall_timeout(requested)
        steps_before = int(self.sim_steps)
        reason = "wall_budget" if wall_left is not None and wall_left <= 0 else None
        if self.sim_steps >= self.horizon:
            reason = "horizon"
        model = request.get("model") or os.environ.get("REA_MODEL")
        max_turns = _positive_int(request.get("max_turns", 60), "max_turns")
        trace = {"model": model, "stop_reason": reason, "tool_calls": [], "tool_results": [],
                 "tool_calls_count": 0, "tool_time_s": 0.0, "llm_time_s": 0.0, "usage": None}
        started = self.clock()
        self._interrupt = reason
        if reason is None:
            deadline = time.monotonic() + timeout
            inner = self.low.env
            original_step = inner.step

            def check_budget():
                if self._interrupt is not None:
                    raise ProgramInterrupt(self._interrupt)
                if time.monotonic() >= deadline:
                    self._interrupt = "wall_timeout"
                    raise ProgramInterrupt("wall_timeout")
                if self.sim_steps >= self.horizon:
                    self._interrupt = "horizon"
                    raise ProgramInterrupt("horizon")

            def guarded_step(action):
                check_budget()
                self.sim_steps += 1
                if hasattr(self.low, "_sim_step_count"):
                    self.low._sim_step_count = self.sim_steps
                return self._call_sim_step(original_step, action)

            def bounded(fn):
                def call(*args, **kwargs):
                    check_budget()
                    return fn(*args, **kwargs)
                return call

            guard = None
            try:
                self._previous_alarm = signal.signal(signal.SIGALRM, self._alarm)
                self._alarm_installed = True
                self._guarded_env = inner
                self._original_step = original_step
                inner.step = guarded_step
                self._running = True
                # The SDK loop also enforces this deadline while waiting for model output.
                signal.setitimer(signal.ITIMER_REAL, timeout)
                fns = {name: bounded(fn) for name, fn in self.api_functions(self.code_env).items()}
                guard = online_agent_module().ToolServer(fns, deadline, ProgramInterrupt)
                hands = {}
                held_and_lifted = False
                for arm in (0, 1):
                    answer = guard.execute("check_object_in_hand", {"arm": arm})
                    hands[str(arm)] = answer
                    if self._interrupt is not None:
                        raise ProgramInterrupt(self._interrupt)
                    if answer == "true":
                        lift_result = guard.execute("lift_arm", {"arm": arm})
                        check_budget()
                        lift_ok = guard.calls[-1].get("ok") and lift_result != "false"
                        hands[str(arm)] = guard.execute("check_object_in_hand", {"arm": arm})
                        check_budget()
                        held_and_lifted = lift_ok and hands[str(arm)] == "true"
                        if held_and_lifted:
                            trace.update({"stop_reason": "inhand_guard", "held_arm": arm, "hand_checks": hands})
                        break
                if not held_and_lifted:
                    key, base = online_agent_credentials()
                    initial = None
                    if request.get("images", True):
                        head, _ = self._camera_frames()
                        initial = base64.b64decode(self._encode(head))
                    check_budget()
                    prompt_path = Path(os.environ.get("REA_BEHAVIOR_PROMPTS") or Path(_HERE).parent / "eval" / "behavior" / "prompts") / (self.task["name"] + ".txt")
                    prompt = prompt_path.read_text() if prompt_path.exists() else "You control a R1Pro robot."
                    prompt = "\n".join(line for line in prompt.splitlines() if line.strip() and not line.strip().startswith("prompt: |"))
                    context = {"previous_execution": request.get("context"), "hand_checks": hands, "handoff_tools": guard.tool_results}
                    if not model:
                        raise ValueError("run_agent needs a model (request.model or REA_MODEL)")
                    args = SimpleNamespace(agent_model=model, agent_max_turns=max_turns,
                                           agent_images=bool(request.get("images", True)), agent_llm_timeout=120.0,
                                           sdk_python=request.get("sdk_python"), agent_context=context)
                    with tempfile.TemporaryDirectory(prefix="rea-behavior-agent-") as directory:
                        trace = self.agent_runner(fns=fns, instruction=self.task["language"], task_prompt=prompt,
                                                  initial_jpg=initial, args=args, budget_s=max(0.0, deadline - time.monotonic()),
                                                  api_key=key, base_url=base, budget_exc=ProgramInterrupt,
                                                  ep_dir=directory, log=lambda **entry: print(json.dumps(entry), file=sys.stderr, flush=True))
                        stderr_path = Path(directory) / "agent_child.stderr"
                        if trace.get("error") and stderr_path.exists():
                            trace["stderr_tail"] = stderr_path.read_text()[-int(self.args.tail_chars):]
                    trace["hand_checks"] = hands
            except ProgramInterrupt as interrupt:
                self._interrupt = interrupt.reason
                trace["stop_reason"] = interrupt.reason
            except Exception as error:
                trace.update({"stop_reason": "agent_error", "error": f"{type(error).__name__}: {error}"})
            finally:
                self._teardown_program()
                if guard is not None:
                    trace["tool_calls"] = guard.calls + trace.get("tool_calls", [])
                    trace["tool_results"] = guard.tool_results + trace.get("tool_results", [])
                    trace["tool_calls_count"] = len(trace["tool_results"])
                    trace["tool_time_s"] = round(sum(item.get("s", 0) for item in trace["tool_results"]), 3)
        wall_s = round(self.clock() - started, 3) if reason is None else 0.0
        self.wall_used_s = round(self.wall_used_s + wall_s, 3)
        reason = self._interrupt
        if reason is None and trace.get("stop_reason") == "program_budget":
            reason = "wall_timeout"
        if reason is not None:
            trace["stop_reason"] = reason
        trace["wall_s"] = wall_s
        trace["llm_time_s"] = round(max(0.0, wall_s - trace.get("tool_time_s", 0.0)), 3)
        error = trace.get("error")
        exception = {"type": "AgentError", "message": str(error), "class": "agent"} if error and reason is None else None
        return {
            "which": "agents-sdk", "steps_before": steps_before, "steps_after": int(self.sim_steps),
            "steps": int(self.sim_steps) - steps_before, "horizon": self.horizon, "wall_s": wall_s,
            "wall_timeout_s": timeout, "wall_used_s": self.wall_used_s, "wall_budget_s": self.wall_budget_s,
            "wall_left_s": None if self.wall_budget_s is None else max(0.0, round(self.wall_budget_s - self.wall_used_s, 3)),
            "exception": exception, "exception_class": "timeout" if reason in ("wall_timeout", "wall_budget") else ("agent" if exception else None),
            "truncated": reason is not None, "truncated_reason": reason,
            "stdout_tail": str(trace.get("final_output") or "")[-int(self.args.tail_chars):],
            "stderr_tail": str(error or "")[-int(self.args.tail_chars):], "infra_probe": None, "agent_trace": trace,
        }

    # ---- observe ----
    def _camera_frames(self):
        """Head and left-wrist RGB frames as PIL images (a perception-quality render when REA_BEHAVIOR_PT_OBS is set)."""
        from PIL import Image

        low = self.low
        getter = getattr(low, "get_perception_observation", None) if os.environ.get("REA_BEHAVIOR_PT_OBS") else None
        obs = getter() if callable(getter) else low.get_observation()
        robot_key = next(key for key in obs if str(key).startswith("robot"))
        cameras = obs[robot_key]
        head = Image.fromarray(_to_uint8_rgb(cameras[f"{robot_key}:{HEAD_CAMERA}"]["rgb"]))
        wrist_entry = cameras.get(f"{robot_key}:{LEFT_WRIST_CAMERA}")
        wrist = Image.fromarray(_to_uint8_rgb(wrist_entry["rgb"])) if wrist_entry is not None else None
        return head, wrist

    def _front_views(self):
        """R4 BEHAVIOR view: three PIL frames from the external VisionSensor moved in front of the robot (restored afterwards), plus metadata.
        Poses: front_high = 1.4 m ahead, 1.8 m up, looking at the head/raised-arm region; front_left / front_right = torso-height
        overviews from +-35 degrees. Each pose is rendered twice (the first render after a camera move can return the previous frame)
        through the perception path (PathTracing when REA_BEHAVIOR_PT_OBS is set). Falls back to three head-camera copies when the low-level
        env has no external camera (unit tests, other simulators), recorded in the metadata."""
        import math

        import numpy as np
        from PIL import Image

        low = self.low
        og_env = getattr(low, "env", None)
        sensors = getattr(og_env, "external_sensors", None) or {}
        cam = sensors.get("external_camera") if isinstance(sensors, dict) else None
        if cam is None:
            head, _wrist = self._camera_frames()
            return [head, head, head], {"views": ["head", "head", "head"], "fallback": "no_external_camera"}
        from scipy.spatial.transform import Rotation as R
        import torch as th

        to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=float)
        cpos0, cquat0 = cam.get_position_orientation()
        rpos_t, rq = og_env.robots[0].get_position_orientation()
        rpos = to_np(rpos_t); yaw = R.from_quat(to_np(rq)).as_euler("xyz")[2]
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        dist, height, angle = 1.4, 1.3, 35.0

        def lookat(cp, tgt):
            f = np.asarray(tgt, float) - np.asarray(cp, float); f /= max(np.linalg.norm(f), 1e-6)
            up = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.98 else np.array([0.0, 1.0, 0.0])
            r = np.cross(f, up); r /= max(np.linalg.norm(r), 1e-6); u = np.cross(r, f)
            return R.from_matrix(np.stack([r, u, -f], axis=1)).as_quat()

        poses = {"front_high": (rpos + fwd * dist + np.array([0.0, 0.0, height + 0.5]), rpos + np.array([0.0, 0.0, 1.45]))}
        for sign, tag in ((+1, "front_left"), (-1, "front_right")):
            d = np.array([math.cos(yaw + math.radians(sign * angle)), math.sin(yaw + math.radians(sign * angle)), 0.0])
            poses[tag] = (rpos + d * dist * 0.95 + np.array([0.0, 0.0, height]), rpos + np.array([0.0, 0.0, 1.1]))
        getter = getattr(low, "get_perception_observation", None) if os.environ.get("REA_BEHAVIOR_PT_OBS") else None
        render = getter if callable(getter) else low.get_observation
        frames = []
        try:
            for tag, (cp, tgt) in poses.items():
                cam.set_position_orientation(position=th.tensor(cp, dtype=th.float32), orientation=th.tensor(lookat(cp, tgt), dtype=th.float32))
                obs = None
                for _ in range(2):
                    obs = render()
                frames.append(Image.fromarray(_to_uint8_rgb(obs["external"]["external_camera"]["rgb"])))
        finally:
            cam.set_position_orientation(position=cpos0, orientation=cquat0)
            try:
                render()  # flush so the program's next perception frame is not the moved view
            except Exception:
                pass
        return frames, {"views": list(poses), "camera": "external_camera_moved", "dist": dist, "height": height, "angle_deg": angle}

    def _encode(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(self.args.jpeg_quality))
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def observe(self, request):
        from PIL import Image

        if self.code_env is None or self.task is None:
            raise ValueError("Reset an episode before observing")
        view = request.get("view", "agentview")
        if view not in VIEWS:
            raise ValueError(f"Unknown view: {view}")
        wrist = bool(request.get("wrist", False))
        want_judge = bool(request.get("judge_frame", True))
        size = int(self.args.frame_size)
        head, wrist_image = self._camera_frames()
        agentview = head.resize((size, size), Image.BILINEAR) if head.size != (size, size) else head
        judge_image = None
        front_meta = None
        if want_judge:
            judge_image = head.resize(self.judge_frame_size, Image.BILINEAR) if head.size != tuple(self.judge_frame_size) else head
        if view == "montage":
            wrist_view = (wrist_image.resize((size, size), Image.BILINEAR) if wrist_image is not None else Image.new("RGB", (size, size), "black"))
            image = compose_montage(agentview, wrist_view, size, titles=MONTAGE_TITLES)
        elif view == "front":
            # Judge view for pick-up tasks: three renders from the task's external camera moved in front of the robot (front_high,
            # front_left, front_right), side by side. The head camera looks down the raised arm after a grasp and cannot see the
            # held object, so a head-frame judge answers not_done after a successful grasp.
            views, front_meta = self._front_views()
            tiles = [v.resize((size, size), Image.BILINEAR) if v.size != (size, size) else v for v in views]
            image = Image.new("RGB", (size * len(tiles), size))
            for i, tile in enumerate(tiles):
                image.paste(tile, (i * size, 0))
            if want_judge:
                jw, jh = self.judge_frame_size
                jt = [v.resize((jh, jh), Image.BILINEAR) for v in views]
                judge_image = Image.new("RGB", (jh * len(jt), jh))
                for i, tile in enumerate(jt):
                    judge_image.paste(tile, (i * jh, 0))
        else:
            image = agentview
        captured_at = self.now().isoformat()
        steps = int(self.sim_steps)
        steps_left = self.horizon - steps
        self.observation_number += 1
        observation_id = f"{self.episode}:{self.observation_number}"
        task = dict(self.task)
        meta = {"task": task, "proprio": None, "observationId": observation_id, "budget": {"steps": steps, "horizon": self.horizon, "stepsLeft": steps_left}}
        if front_meta is not None:
            meta["front"] = front_meta
        frame = {"imageB64": self._encode(image), "mimeType": "image/jpeg", "width": image.width, "height": image.height, "view": view, "capturedAt": captured_at, "behavior": meta, "libero": meta}
        judge_payload = None
        if judge_image is not None:
            judge_payload = {"imageB64": self._encode(judge_image), "mimeType": "image/jpeg", "width": judge_image.width, "height": judge_image.height,
                             "view": "judge", "camera": "head", "capturedAt": captured_at, "behavior": json.loads(json.dumps(meta)), "libero": json.loads(json.dumps(meta))}
        wrist_payload = None
        if wrist and wrist_image is not None:
            wrist_payload = {"imageB64": self._encode(wrist_image), "mimeType": "image/jpeg", "width": wrist_image.width, "height": wrist_image.height, "view": "wrist", "capturedAt": captured_at}
        return {
            "frame": frame, "frame_judge": judge_payload, "wrist": wrist_payload, "proprio": None,
            "steps": steps, "horizon": self.horizon, "stepsLeft": steps_left, "task": task, "episode": self.episode,
            "observationId": observation_id, "sealed": self.sealed,
        }

    # ---- score (the single success read; seals the episode) ----
    def score(self):
        self._require_episode("score")
        if self.sealed:
            raise ValueError("score was already called for this episode")
        completed = bool(self.low.task_completed())
        self.scored = {"task_completed": completed, "steps": int(self.sim_steps), "horizon": self.horizon, "scoredAt": self.now().isoformat()}
        self.sealed = True
        return dict(self.scored)

    # ---- close ----
    def close(self):
        # OmniGibson/Isaac teardown segfaults after the fact upstream; the reply is flushed first and the
        # process exits right after (serve() returns on EOF), so nothing is torn down here.
        self.task = None
        return {"closed": True}

    def dispatch(self, request):
        command = request.get("command")
        if command == "catalog":
            return self.catalog()
        if command == "reset":
            return self.reset(request)
        if command == "run_program":
            return self.run_program(request)
        if command == "run_agent":
            return self.run_agent(request)
        if command == "observe":
            return self.observe(request)
        if command == "score":
            return self.score()
        if command == "close":
            return self.close()
        raise ValueError(f"Unknown command: {command}")


def build_parser():
    parser = argparse.ArgumentParser(description="ASPIRE BEHAVIOR-1K (R1Pro) program backend")
    parser.add_argument("--aspire-root", default=DEFAULT_ASPIRE_ROOT, help="ASPIRE's aspire/sim directory (default ASPIRE_ROOT)")
    parser.add_argument("--python-root", default=DEFAULT_PYTHON_ROOT, help="directory that makes `aspire` importable (default ASPIRE_PYTHON_ROOT)")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--judge-frame-size", default=DEFAULT_JUDGE_FRAME_SIZE)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--wall-timeout-s", type=float, default=2000.0)
    parser.add_argument("--tail-chars", type=int, default=2000)
    parser.add_argument("--infra-probe-ports", default=DEFAULT_INFRA_PROBE_PORTS)
    return parser


def default_args(**overrides):
    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.aspire_root or not args.python_root:
        parser.error("--aspire-root and --python-root (or ASPIRE_ROOT and ASPIRE_PYTHON_ROOT) are required")
    parse_frame_size(args.judge_frame_size)
    parse_ports(args.infra_probe_ports)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    args.aspire_root = os.path.abspath(args.aspire_root)
    args.python_root = os.path.abspath(args.python_root)
    if args.python_root not in sys.path:
        sys.path.insert(0, args.python_root)
    os.chdir(args.aspire_root)
    # Isaac Sim / OmniGibson write to fd 1 natively: keep a private copy of stdout for the protocol and
    # send everything else (including native writes) to stderr.
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    backend = BehaviorBackend(args)
    serve(backend, sys.stdin, protocol_output)
    protocol_output.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
