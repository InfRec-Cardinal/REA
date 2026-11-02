"""JSON-lines program backend embedding ASPIRE's LIBERO code-execution env (frozen programs as System 0).

Protocol: docs/extending/program-backends.md. One JSON object per stdin line `{id, command, ...}`; one
reply per line `{id, ok: true, ...}` or `{id, ok: false, error}`. Commands: catalog, reset, run_program,
observe, score, close.

Protocol invariant: nothing on the policy/judge/recovery path reads check_success, reward,
task_completed, object poses or sim.data. The only success read is the `score` command, which
seals the episode. `run_program` executes through `CodeExecutionEnvBase._exec_user_code`, never
`step` (which would compute reward and task_completed).

ASPIRE is imported lazily inside the recipe block so the module is unit-testable on a machine
without ASPIRE, LIBERO or MuJoCo (a fake env is injected through `env_factory`/`task_lookup`).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import signal
import socket
import sys
import time
from datetime import datetime, timezone

SUITES = (
    "libero_goal_swap",
    "libero_goal_task",
    "libero_object_swap",
    "libero_object_task",
    "libero_spatial_swap",
    "libero_spatial_task",
    "libero_10_swap",
    "libero_10_task",
    # unperturbed LIBERO suites (canonical layouts and instructions)
    "libero_goal",
    "libero_object",
    "libero_spatial",
    "libero_10",
)
# The ASPIRE checkout: ASPIRE_ROOT is its aspire/sim directory, ASPIRE_PYTHON_ROOT the directory that makes `aspire` importable.
DEFAULT_ASPIRE_ROOT = os.environ.get("ASPIRE_ROOT")
DEFAULT_PYTHON_ROOT = os.environ.get("ASPIRE_PYTHON_ROOT")
DEFAULT_CONFIG = "env_configs/libero/franka_libero_traced.yaml"
DEFAULT_API = "FrankaLiberoApiReducedSkillLibrary"
# Judge frames keep the 800x512 agentview aspect of ASPIRE's recorded frames;
# MuJoCo keeps the camera's vertical field of view fixed, so a WxH render shows the same vertical
# extent as 800x512 and a horizontal extent scaled by (W/H)/(800/512) (1.024 for 512x320).
DEFAULT_JUDGE_FRAME_SIZE = "512x320"
# Perception/IK service ports probed after a failed program run (SAM3, GraspNet, PyRoKi).
DEFAULT_INFRA_PROBE_PORTS = "8114,8115,8116"
RESET_SEMANTICS = ("aspire", "seeded_random")
PHASES = ("development", "heldout")
VIEWS = ("agentview", "montage")
WRIST_CAMERA = "robot0_eye_in_hand"
TITLE_HEIGHT = 24
MONTAGE_TITLES = ("External camera (front)", "Wrist camera (L/R mirrored)")
PRELUDE_CODE = "open_gripper()\ngoto_home_joint_position()\n"
# After the wall deadline the timer re-fires every WALL_TIMEOUT_REARM_S seconds while a program is
# running, so a program that swallows BaseException cannot outlive its deadline by busy-waiting.
WALL_TIMEOUT_REARM_S = 1.0
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
FRAME_SIZE_SPEC = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")
# Whole-identifier matches; names ending in "(" match the call form with optional whitespace
# (`eval (x)`), so a program cannot reach the env internals, reward or success checks, or the
# interpreter (getattr/__import__/eval/exec/globals/vars) to rebuild a forbidden name at run time.
FORBIDDEN_IDENTIFIERS = (
    "task_completed",
    "check_success",
    "_check_success",
    "compute_reward",
    "_current_reward",
    "_current_done",
    "_current_obs",
    "_current_info",
    "_get_object_pose",
    "_get_all_object_poses",
    "sim.data",
    "sim.model",
    "body_name2id",
    "init_states",
    "set_init_state",
    "handle.env",
    "getattr",
    "__import__",
    "eval(",
    "exec(",
    "globals(",
    "vars(",
)


def _forbidden_pattern(name):
    if name.endswith("("):
        return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name[:-1]) + r"\s*\(")
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")


FORBIDDEN_PATTERNS = tuple((name, _forbidden_pattern(name)) for name in FORBIDDEN_IDENTIFIERS)
# Last traceback line of an interrupted program; the class may carry a module prefix when the
# module is imported under a name other than __main__ (e.g. in the unit tests).
INTERRUPT_LINE = re.compile(r"^(?:[\w.]+\.)?ProgramInterrupt(?::\s*(\w+))?\s*$")

# ---- exception classes: "program" (the program or perception result), "infra" (perception/IK
# service or connection failure, GPU/render failure), "timeout" (wall deadline or episode budget).
EXCEPTION_CLASSES = ("program", "infra", "timeout")
# Exception class names (module prefix stripped) raised by requests/urllib3/http.client/socket,
# torch (OutOfMemoryError) and MuJoCo (FatalError) when a service or the renderer fails.
INFRA_EXCEPTION_TYPES = frozenset({
    "ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout", "TimeoutError", "ConnectTimeoutError",
    "ReadTimeoutError", "MaxRetryError", "NewConnectionError", "ProtocolError", "RemoteDisconnected",
    "ChunkedEncodingError", "HTTPError", "SSLError", "ProxyError", "IncompleteRead", "BrokenPipeError",
    "ConnectionResetError", "ConnectionRefusedError", "ConnectionAbortedError", "RequestException",
    "URLError", "OutOfMemoryError", "CudaError", "FatalError", "GLError",
})
# Messages of ASPIRE's service wrappers (sam3.py "Failed to communicate with SAM3 service at",
# graspnet.py "... GraspNet service at", serve_utils.post_with_retries "Request to <url> failed after
# N retries"), of requests/urllib3, of HTTP 5xx via raise_for_status, and of CUDA/EGL failures.
INFRA_MESSAGE = re.compile(
    r"Failed to communicate with|service at http|Request to \S+ failed after|HTTPConnectionPool|"
    r"Max retries exceeded|Connection refused|Connection reset|Connection aborted|Remote end closed|"
    r"Read timed out|Failed to establish a new connection|Name or service not known|"
    r"Temporary failure in name resolution|\b5\d\d Server Error|Service Unavailable|Bad Gateway|"
    r"Gateway Time-?out|CUDA out of memory|CUDA error|CUDA_ERROR|\bEGL\b|OpenGL|GLContext|"
    r"MuJoCo Error|mujoco\.FatalError",
    re.IGNORECASE,
)


def scan_program(code):
    """Return the first forbidden identifier used by `code` (whole-word match) or None."""
    for name, pattern in FORBIDDEN_PATTERNS:
        if pattern.search(code):
            return name
    return None


def classify_exception(type_name, message):
    """'infra' for perception/IK service, connection, GPU and renderer failures; 'program' otherwise."""
    if type_name in INFRA_EXCEPTION_TYPES or INFRA_MESSAGE.search(message or ""):
        return "infra"
    return "program"


def parse_frame_size(value):
    """'WxH' (or a 2-sequence) -> (width, height) of positive ints."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        width, height = value
    else:
        match = FRAME_SIZE_SPEC.match(str(value))
        if not match:
            raise ValueError(f"frame size must look like WxH (e.g. 512x320): {value!r}")
        width, height = match.group(1), match.group(2)
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("frame size dimensions must be positive integers")
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError("frame size dimensions must be positive integers")
    return width, height


def parse_ports(value):
    """'8114,8115' -> [8114, 8115]; '' -> [] (probe disabled)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [item.strip() for item in str(value).split(",") if item.strip()]
    ports = []
    for item in items:
        port = int(item)
        if not 1 <= port <= 65535:
            raise ValueError(f"invalid service port: {item}")
        ports.append(port)
    return ports


def probe_ports(ports, host="127.0.0.1", timeout_s=0.5):
    """TCP-connect probe of the perception/IK service ports -> {"<port>": listening?}."""
    status = {}
    for port in ports:
        try:
            with socket.create_connection((host, int(port)), timeout=timeout_s):
                status[str(port)] = True
        except OSError:
            status[str(port)] = False
    return status


# ---- compact proprio (shared listing, spec section 6.3) ----
GRIPPER_METRIC_LENGTH = 0.04


def gripper_state(command_closed, fraction, previous_fraction):
    if previous_fraction is not None and abs(fraction - previous_fraction) > 0.02:
        return "moving"
    if fraction >= 0.85:
        return "open"
    if fraction <= 0.08:
        return "closed"
    return "holding" if command_closed else "partial"


def compact_proprio(observation, command_closed, previous_fraction):
    fraction = min(max(float(observation["robot0_gripper_qpos"][0]) / GRIPPER_METRIC_LENGTH, 0.0), 1.0)
    return {
        "eef_cm": [round(float(value) * 100.0, 1) for value in observation["robot0_eef_pos"]],
        "gripper": gripper_state(command_closed, fraction, previous_fraction),
        "gripper_fraction": round(fraction, 3),
        "gripper_command": "close" if command_closed else "open",
    }


class ProgramInterrupt(BaseException):
    """Raised inside a running program to stop it (BaseException so `except Exception:` cannot swallow it)."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


# >>> ASPIRE ENV RECIPE (the only place that knows how ASPIRE/LIBERO are imported and built) >>>
def _benchmark_module():
    """LIBERO benchmark module. ASPIRE's venv exposes the LIBERO-PRO checkout as top-level
    `libero` (replay_trial.py:121 `from libero import benchmark`); `libero.libero.benchmark` is
    the pip layout and is tried second."""
    try:
        from libero import benchmark  # type: ignore[import-not-found]
    except ImportError:
        from libero.libero import benchmark  # type: ignore[import-not-found,no-redef]
    return benchmark


def list_task_names(suite):
    """Task names of a LIBERO suite in index order (replay_trial.py:123-134)."""
    benchmark = _benchmark_module()
    suites = benchmark.get_benchmark_dict()
    if suite not in suites:
        raise ValueError(f"Unsupported suite: {suite}")
    task_suite = suites[suite]()
    count = getattr(task_suite, "n_tasks", None)
    if count is None:
        count = task_suite.get_num_tasks()
    return [task_suite.get_task(index).name for index in range(int(count))]


def find_task_index(suite: str, task) -> int:
    """int passthrough (range-checked) or name lookup; replay_trial.py:123-134."""
    names = list_task_names(suite)
    if isinstance(task, bool) or not isinstance(task, (int, str)):
        raise ValueError("task must be a task name or an integer index")
    if isinstance(task, int):
        if not 0 <= task < len(names):
            raise ValueError(f"Task index out of range: {task}")
        return task
    if task not in names:
        raise ValueError(f"Task not found in {suite}: {task}")
    return names.index(task)


def build_code_env(config_path: str, suite: str, task_index: int, max_steps: int, api_name: str):
    """Return a CodeExecutionEnvBase for one LIBERO task; verified default = replay_trial.py:355-389.

    instantiate() constructs FrankaLiberoEnv (which calls reset() once with seed=None,
    libero.py:153) and the API, whose __init__ opens the SAM3/GraspNet/PyRoKi HTTP clients and
    runs a PyRoKi JIT warm-up IK call (libero_reduced.py:63-96).
    """
    import copy

    import yaml
    from aspire.sim.cap.envs.configs.instantiate import instantiate

    with open(config_path) as handle:
        config = copy.deepcopy(yaml.safe_load(handle))
    config["env"]["cfg"]["low_level"] = {
        "_target_": "aspire.sim.cap.envs.simulators.libero.FrankaLiberoEnv",
        "suite_name": suite,
        "task_id": task_index,
        "privileged": False,
        "max_steps": max_steps,
        "seed": None,
        "enable_render": False,
        "viser_debug": False,
    }
    config["env"]["cfg"]["apis"] = [api_name]
    return instantiate(config["env"])


# <<< ASPIRE ENV RECIPE <<<


def find_apis(obj, depth=0):
    """Walk the wrapper chain to the `_apis` dict (replay_trial.py:249-261)."""
    if obj is None or depth > 5:
        return {}
    apis = getattr(obj, "_apis", None)
    if isinstance(apis, dict):
        return apis
    for attr in ("env", "low_level_env", "_executor", "executor"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            found = find_apis(child, depth + 1)
            if found:
                return found
    return {}


def classify_result(result, steps_after, horizon, interrupt, tail_chars):
    """Turn an `_exec_user_code` result into the run_program exception/truncation fields.

    `interrupt` is the reason of the last ProgramInterrupt the backend raised (or None).
    `exception_class` is "timeout" for a wall deadline / episode wall budget, otherwise the class
    of the exception ("program" | "infra") or None when the run ended cleanly or at a step budget.
    """
    stdout = result.get("stdout") or ""
    stderr = result.get("stderr") or ""
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    last = lines[-1].strip() if lines else ""
    match = INTERRUPT_LINE.match(last)
    exception = None
    program_exit = None
    truncated_reason = None
    if not result.get("ok"):
        if match:
            truncated_reason = match.group(1) or interrupt or "wall_timeout"
        else:
            type_name, _, message = last.partition(":")
            type_name = type_name.strip().split(".")[-1] or "Unknown"
            message = message.strip()
            if type_name == "SystemExit" and message in ("", "0", "None"):
                # `raise SystemExit(0)` / `sys.exit()` is the program's own "stop here" (LLM-written BEHAVIOR programs end a
                # successful grasp that way): a clean end, not a failure, so it must not trigger exception recovery.
                # A non-zero code stays a program exception.
                program_exit = f"SystemExit:{message or '0'}"
            else:
                exception = {
                    "type": type_name,
                    "message": message,
                    "traceback_tail": stderr[-tail_chars:],
                    "class": classify_exception(type_name, message),
                }
    if truncated_reason is None and interrupt is not None:
        truncated_reason = interrupt
    if steps_after >= horizon:
        truncated_reason = "horizon"
    if truncated_reason in ("wall_timeout", "wall_budget"):
        exception_class = "timeout"
    elif exception is not None:
        exception_class = exception["class"]
    else:
        exception_class = None
    return {
        "exception": exception,
        "exception_class": exception_class,
        "program_exit": program_exit,
        "truncated": truncated_reason is not None,
        "truncated_reason": truncated_reason,
        "stdout_tail": stdout[-tail_chars:],
        "stderr_tail": stderr[-tail_chars:],
    }


def compose_montage(agentview, wrist, size, titles=MONTAGE_TITLES):
    """Agentview + wrist side by side under a 24 px title bar (libero_backend.py montage layout)."""
    from PIL import Image, ImageDraw

    montage = Image.new("RGB", (size * 2, size + TITLE_HEIGHT), "#161616")
    draw = ImageDraw.Draw(montage)
    for index, (image, title) in enumerate(((agentview, titles[0]), (wrist, titles[1]))):
        montage.paste(image, (index * size, TITLE_HEIGHT))
        draw.text((index * size + 6, 6), title, fill="white")
    return montage


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


class AspireBackend:
    def __init__(self, args, env_factory=None, task_lookup=None, clock=time.perf_counter, now=None, service_probe=None):
        self.args = args
        self.env_factory = env_factory or build_code_env
        self.task_lookup = task_lookup or find_task_index
        self.clock = clock
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.judge_frame_size = parse_frame_size(getattr(args, "judge_frame_size", DEFAULT_JUDGE_FRAME_SIZE))
        self.infra_probe_ports = parse_ports(getattr(args, "infra_probe_ports", DEFAULT_INFRA_PROBE_PORTS))
        self.service_probe = service_probe or (lambda: probe_ports(self.infra_probe_ports))
        self.code_env = None
        self.low = None
        self.identity = None
        self.task = None
        self.episode = 0
        self.observation_number = 0
        self.horizon = int(args.max_steps)
        self.sealed = False
        self.scored = None
        self.previous_gripper_fraction = None
        self.initial_state_source = None
        self.initial_state_index = None
        self.reset_semantics = None
        self.phase = None
        self.wall_budget_s = None
        self.wall_used_s = 0.0
        self._running = False
        self._interrupt = None
        self._program_handle = None
        self._alarm_installed = False
        self._previous_alarm = None

    # ---- catalog ----
    def catalog(self):
        suites = []
        for suite_name in SUITES:
            try:
                names = list_task_names(suite_name)
            except ValueError as error:
                print(f"catalog: skipping {suite_name}: {error}", file=sys.stderr, flush=True)
                continue
            suites.append({
                "name": suite_name,
                "family": "long" if suite_name.startswith("libero_10") else ("original" if suite_name in ("libero_goal", "libero_object", "libero_spatial") else "pro"),
                "tasks": [{"index": index, "name": name} for index, name in enumerate(names)],
            })
        return {
            "suites": suites,
            "backend": {
                "config": self.args.config,
                "api": self.args.api,
                "maxSteps": int(self.args.max_steps),
                "frameSize": int(self.args.frame_size),
                "judgeFrameSize": {"width": self.judge_frame_size[0], "height": self.judge_frame_size[1]},
                "wallTimeoutS": float(self.args.wall_timeout_s),
                "infraProbePorts": list(self.infra_probe_ports),
                "aspireRoot": os.path.abspath(self.args.aspire_root) if self.args.aspire_root else None,
                "pythonRoot": os.path.abspath(self.args.python_root) if self.args.python_root else None,
            },
        }

    # ---- reset ----
    def reset(self, request):
        suite = request.get("suite")
        if suite not in SUITES:
            raise ValueError(f"Unsupported suite: {suite}")
        task = request.get("task")
        seed = request.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 1 <= seed <= 100:
            raise ValueError("seed must be an integer between 1 and 100")
        horizon = _positive_int(request.get("horizon", self.args.max_steps), "horizon")
        reset_semantics = request.get("reset", "aspire")
        if reset_semantics not in RESET_SEMANTICS:
            raise ValueError(f"Unknown reset semantics: {reset_semantics}")
        if reset_semantics == "seeded_random" and seed <= 50:
            raise ValueError("seeded_random reset is only allowed for development seeds 51-100")
        phase = request.get("phase")
        if phase is not None:
            if phase not in PHASES:
                raise ValueError(f"Unknown phase: {phase}")
            if phase == "heldout" and (seed > 50 or reset_semantics != "aspire"):
                raise ValueError("heldout phase requires seeds 1-50 with aspire reset semantics")
            if phase == "development" and seed <= 50:
                raise ValueError("development phase requires seeds 51-100")
            # Under the aspire rule seed s replays official state (s - 1) mod 50: a development seed would reuse the
            # initial state of an evaluation seed.
            if phase == "development" and reset_semantics != "seeded_random":
                raise ValueError("development phase requires seeded_random reset semantics")
        wall_budget_s = request.get("wall_budget_s")
        if wall_budget_s is not None:
            wall_budget_s = _positive_number(wall_budget_s, "wall_budget_s")
        task_index = self.task_lookup(suite, task)
        task_name = task if isinstance(task, str) else list_task_names(suite)[task_index]

        identity = (suite, task_index, horizon, self.args.api)
        if self.code_env is None or identity != self.identity:
            self._close_env()
            self.code_env = self.env_factory(self.args.config, suite, task_index, horizon, self.args.api)
            self.low = self.code_env.low_level_env
            self.identity = identity
        low = self.low
        self.task = None
        self.sealed = False
        self.scored = None
        if reset_semantics == "seeded_random":
            saved = low.handle.init_states
            low.handle.init_states = None
            try:
                self.code_env.reset(seed=seed)
            finally:
                low.handle.init_states = saved
            self.initial_state_source = "seeded_random_development"
            self.initial_state_index = None
        else:
            self.code_env.reset(seed=seed)
            states = low.handle.init_states
            self.initial_state_source = "aspire_modulo_init_state"
            self.initial_state_index = (seed - 1) % len(states) if states is not None and len(states) > 0 else None
        self.reset_semantics = reset_semantics
        self.phase = phase
        self.wall_budget_s = wall_budget_s
        self.wall_used_s = 0.0
        self.horizon = horizon
        self.episode += 1
        self.observation_number = 0
        self.previous_gripper_fraction = None
        self.task = {
            "suite": suite,
            "index": int(task_index),
            "name": task_name,
            "language": low.handle.task_language,
            "seed": seed,
        }
        self._reset_trace_loggers()
        reply = self.observe({})
        reply.update({
            "initialStateSource": self.initial_state_source,
            "initialStateIndex": self.initial_state_index,
            "resetSemantics": self.reset_semantics,
            "phase": self.phase,
            "wallBudgetS": self.wall_budget_s,
            "episode": self.episode,
        })
        return reply

    def _reset_trace_loggers(self):
        for api in find_apis(self.code_env).values():
            get_logger = getattr(api, "get_trace_logger", None)
            if get_logger is None:
                continue
            try:
                get_logger().reset()
            except Exception as error:
                print(f"trace logger reset failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)

    def _close_env(self):
        if self.code_env is None:
            return
        handle = getattr(self.low, "handle", None)
        inner = getattr(handle, "env", None)
        close = getattr(inner, "close", None)
        if callable(close):
            try:
                close()
            except Exception as error:
                print(f"env close failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        self.code_env = None
        self.low = None
        self.identity = None

    # ---- run_program ----
    def _require_episode(self, verb):
        if self.code_env is None or self.task is None:
            raise ValueError(f"Reset an episode before {verb}")

    def _alarm(self, signum, frame):
        if self._running:
            self._interrupt = "wall_timeout"
            raise ProgramInterrupt("wall_timeout")

    def _teardown_program(self):
        """Idempotent end-of-run teardown; safe to call twice and outside a run.

        `_running = False` comes first so a pending SIGALRM becomes a no-op before the timer is
        cancelled; the handler is restored only after the timer is off, so a restored default
        action can never see a live timer; the step guard is removed last.
        """
        self._running = False
        signal.setitimer(signal.ITIMER_REAL, 0)
        if self._alarm_installed:
            self._alarm_installed = False
            previous = self._previous_alarm
            self._previous_alarm = None
            signal.signal(signal.SIGALRM, signal.SIG_DFL if previous is None else previous)
        handle = self._program_handle
        if handle is not None:
            self._program_handle = None
            try:
                del handle.step
            except AttributeError:
                pass

    def abort_program(self):
        """Clear the run state after a ProgramInterrupt escaped run_program (protocol-loop fallback)."""
        self._teardown_program()

    def _check_program_pin(self, allowed, code, digest):
        if allowed is None:
            if self.phase == "heldout":
                raise ValueError("heldout phase requires allowed_sha256 on every run_program")
            return
        if (not isinstance(allowed, list) or not allowed
                or not all(isinstance(item, str) and SHA256_HEX.match(item.lower()) for item in allowed)):
            raise ValueError("allowed_sha256 must be a non-empty list of 64-hex sha256 strings")
        if digest in {item.lower() for item in allowed} or code == PRELUDE_CODE:
            return
        raise ValueError(f"program sha256 {digest} is not in allowed_sha256 (audited program pin)")

    def _effective_wall_timeout(self, requested):
        """Per-request deadline clamped to the episode's remaining wall budget (None = no budget)."""
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
            "which": which,
            "steps_before": steps_before,
            "steps_after": steps_after,
            "steps": steps_after - steps_before,
            "wall_s": wall_s,
            "exception": classified["exception"],
            "exception_class": classified["exception_class"],
            "truncated": classified["truncated"],
            "truncated_reason": classified["truncated_reason"],
            "horizon": self.horizon,
            "sha256": digest,
            "stdout_tail": classified["stdout_tail"],
            "stderr_tail": classified["stderr_tail"],
            "wall_timeout_s": wall_timeout_s,
            "wall_used_s": self.wall_used_s,
            "wall_budget_s": self.wall_budget_s,
            "wall_left_s": wall_left,
            "infra_probe": probe,
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

        low = self.low
        horizon = self.horizon
        steps_before = int(low._sim_step_count)
        if wall_left is not None and wall_left <= 0:
            # Episode wall budget exhausted: report a truncated no-op run instead of executing.
            classified = {
                "exception": None, "exception_class": "timeout", "truncated": True,
                "truncated_reason": "wall_budget", "stdout_tail": "", "stderr_tail": "",
            }
            return self._run_reply(which, digest, steps_before, steps_before, 0.0, classified, wall_timeout_s, None)
        stop_at = horizon if max_steps is None else min(horizon, steps_before + max_steps)
        handle = low.handle
        original_step = handle.step

        def guarded_step(action):
            if self._interrupt == "wall_timeout":
                # The one alarm was swallowed by the program (except BaseException); keep stopping it.
                raise ProgramInterrupt("wall_timeout")
            if low._sim_step_count >= stop_at:
                reason = "horizon" if stop_at == horizon else "max_steps"
                self._interrupt = reason
                raise ProgramInterrupt(reason)
            return original_step(action)

        self._interrupt = None
        result = None
        late_interrupt = None
        started = self.clock()
        try:
            try:
                self._previous_alarm = signal.signal(signal.SIGALRM, self._alarm)
                self._alarm_installed = True
                self._program_handle = handle
                handle.step = guarded_step
                self._running = True
                signal.setitimer(signal.ITIMER_REAL, float(wall_timeout_s), WALL_TIMEOUT_REARM_S)
                try:
                    result = self.code_env._exec_user_code(code)
                except ProgramInterrupt as interrupt:
                    # The timer fired after the program's own BaseException capture returned.
                    result = {"ok": False, "stdout": "", "stderr": f"ProgramInterrupt: {interrupt.reason}\n", "result": None}
            finally:
                self._teardown_program()
        except ProgramInterrupt as interrupt:
            # The timer fired between the program returning and the teardown's first statement (or
            # inside the teardown itself): finish the idempotent teardown and keep the program's result.
            late_interrupt = interrupt
            self._teardown_program()
        if result is None:
            reason = late_interrupt.reason if late_interrupt is not None else "wall_timeout"
            result = {"ok": False, "stdout": "", "stderr": f"ProgramInterrupt: {reason}\n", "result": None}
        wall_s = round(self.clock() - started, 3)
        self.wall_used_s = round(self.wall_used_s + wall_s, 3)
        steps_after = int(low._sim_step_count)
        classified = classify_result(result, steps_after, horizon, self._interrupt, int(self.args.tail_chars))
        probe = None
        if classified["exception"] is not None or classified["truncated_reason"] == "wall_timeout":
            probe = self._probe_services()
            down = sorted(port for port, up in probe.items() if not up)
            if down:
                # A service is gone: whatever the program reported, the run failed on infrastructure.
                classified["exception_class"] = "infra"
                if classified["exception"] is not None:
                    classified["exception"]["class"] = "infra"
                    classified["exception"]["infra_reason"] = f"service port(s) not listening: {', '.join(down)}"
        return self._run_reply(which, digest, steps_before, steps_after, wall_s, classified, wall_timeout_s, probe)

    # ---- observe ----
    def _render(self, camera, width=None, height=None):
        import numpy as np
        from PIL import Image

        size = int(self.args.frame_size)
        width = size if width is None else int(width)
        height = size if height is None else int(height)
        frame = self.low.handle.env.sim.render(camera_name=camera, width=width, height=height, depth=False)[::-1]
        return Image.fromarray(np.ascontiguousarray(frame))

    def _encode(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(self.args.jpeg_quality))
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def observe(self, request):
        if self.code_env is None or self.task is None:
            raise ValueError("Reset an episode before observing")
        view = request.get("view", "agentview")
        if view not in VIEWS:
            raise ValueError(f"Unknown view: {view}")
        wrist = bool(request.get("wrist", False))
        want_judge = bool(request.get("judge_frame", True))
        low = self.low
        size = int(self.args.frame_size)
        agentview = self._render("agentview")
        wrist_image = self._render(WRIST_CAMERA) if view == "montage" or wrist else None
        judge_image = self._render("agentview", *self.judge_frame_size) if want_judge else None
        image = compose_montage(agentview, wrist_image, size) if view == "montage" else agentview
        proprio = compact_proprio(low._current_obs, low._gripper_fraction < 0.5, self.previous_gripper_fraction)
        self.previous_gripper_fraction = proprio["gripper_fraction"]
        captured_at = self.now().isoformat()
        steps = int(low._sim_step_count)
        steps_left = self.horizon - steps
        self.observation_number += 1
        observation_id = f"{self.episode}:{self.observation_number}"
        task = dict(self.task)
        libero = {
            "task": task,
            "proprio": proprio,
            "observationId": observation_id,
            "budget": {"steps": steps, "horizon": self.horizon, "stepsLeft": steps_left},
        }
        frame = {
            "imageB64": self._encode(image),
            "mimeType": "image/jpeg",
            "width": image.width,
            "height": image.height,
            "view": view,
            "capturedAt": captured_at,
            "libero": libero,
        }
        judge_payload = None
        if judge_image is not None:
            judge_payload = {
                "imageB64": self._encode(judge_image),
                "mimeType": "image/jpeg",
                "width": judge_image.width,
                "height": judge_image.height,
                "view": "judge",
                "camera": "agentview",
                "capturedAt": captured_at,
                "libero": json.loads(json.dumps(libero)),
            }
        wrist_payload = None
        if wrist:
            wrist_payload = {
                "imageB64": self._encode(wrist_image),
                "mimeType": "image/jpeg",
                "width": wrist_image.width,
                "height": wrist_image.height,
                "view": "wrist",
                "capturedAt": captured_at,
            }
        return {
            "frame": frame,
            "frame_judge": judge_payload,
            "wrist": wrist_payload,
            "proprio": proprio,
            "steps": steps,
            "horizon": self.horizon,
            "stepsLeft": steps_left,
            "task": task,
            "episode": self.episode,
            "observationId": observation_id,
            "sealed": self.sealed,
        }

    # ---- score (the single success read; seals the episode) ----
    def score(self):
        self._require_episode("score")
        if self.sealed:
            raise ValueError("score was already called for this episode")
        completed = bool(self.low.task_completed())
        self.scored = {
            "task_completed": completed,
            "steps": int(self.low._sim_step_count),
            "horizon": self.horizon,
            "scoredAt": self.now().isoformat(),
        }
        self.sealed = True
        return dict(self.scored)

    # ---- close ----
    def close(self):
        self._close_env()
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
        if command == "observe":
            return self.observe(request)
        if command == "score":
            return self.score()
        if command == "close":
            return self.close()
        raise ValueError(f"Unknown command: {command}")


def handle_request(backend, line):
    """One protocol line -> one reply envelope; errors (including a stray ProgramInterrupt) never leave the loop."""
    request = {}
    try:
        request = json.loads(line)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        return {"id": request.get("id"), "ok": True, **backend.dispatch(request)}
    except (Exception, ProgramInterrupt) as error:
        if isinstance(error, ProgramInterrupt):
            backend.abort_program()
        message = f"{type(error).__name__}: {error}"
        print(message, file=sys.stderr, flush=True)
        request_id = request.get("id") if isinstance(request, dict) else None
        return {"id": request_id, "ok": False, "error": message}


def serve(backend, lines, output):
    try:
        for line in lines:
            if not line.strip():
                continue
            response = handle_request(backend, line)
            output.write(json.dumps(response, separators=(",", ":")) + "\n")
            output.flush()
    finally:
        backend.close()


def build_parser():
    parser = argparse.ArgumentParser(description="ASPIRE LIBERO code-execution backend (JSON lines over stdio)")
    parser.add_argument("--aspire-root", default=DEFAULT_ASPIRE_ROOT, help="ASPIRE's aspire/sim directory (default ASPIRE_ROOT)")
    parser.add_argument("--python-root", default=DEFAULT_PYTHON_ROOT, help="directory that makes `aspire` importable (default ASPIRE_PYTHON_ROOT)")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--judge-frame-size", default=DEFAULT_JUDGE_FRAME_SIZE,
                        help="WxH of the aspect-preserving agentview judge frame (frame_judge), default 512x320")
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--wall-timeout-s", type=float, default=240.0)
    parser.add_argument("--tail-chars", type=int, default=2000)
    parser.add_argument("--infra-probe-ports", default=DEFAULT_INFRA_PROBE_PORTS,
                        help="comma-separated localhost ports probed after a failed run to classify infra failures ('' disables)")
    return parser


def default_args(**overrides):
    """Parser defaults as a Namespace (test helper; keeps tests aligned with the CLI)."""
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
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if os.environ["MUJOCO_GL"] != "egl":
        print(f"warning: MUJOCO_GL={os.environ['MUJOCO_GL']} (ASPIRE offscreen rendering expects egl)", file=sys.stderr, flush=True)
    args.aspire_root = os.path.abspath(args.aspire_root)
    args.python_root = os.path.abspath(args.python_root)
    if args.python_root not in sys.path:
        sys.path.insert(0, args.python_root)
    os.chdir(args.aspire_root)
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    backend = AspireBackend(args)
    serve(backend, sys.stdin, protocol_output)


if __name__ == "__main__":
    main()
