"""OpenAI Agents SDK baseline for the BEHAVIOR arms3 comparison (behavior_arm_runner.py --mode agents-sdk).

Two processes. The RUNNER (Isaac Sim process) owns the simulator and executes tools; it spawns the AGENT process (the same venv
interpreter, but without Kit on sys.path: Kit's pip archive shadows aiohttp/anyio/httpx and breaks the OpenAI SDK) which runs the
openai-agents loop against the OpenAI-compatible relay. Tools are FunctionTools whose invocation is forwarded over a JSON-lines pipe:
agent -> runner {"type":"call","id":n,"tool":name,"args":{...}} ; runner -> agent {"type":"result","id":n,"value":"..."}.
The whole episode (tool execution AND model latency) shares the program budget; past the deadline every tool returns
"BUDGET EXHAUSTED" and the runner ends the loop. Nothing here reads simulator success, ground truth or the task predicate.

Agent styles (args.agent_style). "sdk" (default) is the child above: one Runner.run over all TOOL_SCHEMAS tools, with
args.agent_api "chat_completions" (default) or "responses" (OpenAIResponsesModel with reasoning
effort, max_output_tokens and store=False; a reply cut at max_output_tokens is re-requested up to 3 times, and every tool
result carries the clock the rea-online decision message states). "rea-online" spawns REA's episode runtime (eval/behavior/online/runtime.mjs) instead: the same
JSON-lines channel plus {"type":"observe","id"} (the runner renders the current head-camera JPEG and answers its path) and the
"finish" call (one check_object_in_hand(arm) on the named arm: True ends the episode, False refuses and the episode continues).
In both styles a provider that keeps failing after retries ends the episode with stop_reason "provider_error", which the
runner records as unscored (infrastructure, rerun), never as a task outcome.
"""
import asyncio, base64, json, math, os, re, select, subprocess, sys, time, traceback
from pathlib import Path

TOOL_SCHEMAS = [
    ("find_object_base_rotate", "Rotate the robot base in place to look for the named object with the head camera (SAM3). Returns True when the object is in view.", {"object_name": "string"}),
    ("find_object_torso_rotate", "Rotate the torso/head to look for the named object without moving the base. Returns True when the object is in view.", {"object_name": "string"}),
    ("get_object_pose", "Estimate the named object's pose from the current head-camera view (SAM3 + depth): position xyz in metres (world frame), quaternion wxyz and bounding-box extent; 'not found' when it is not detected.", {"object_name": "string"}),
    ("get_robot_position", "Current robot base position and yaw in the world frame.", {}),
    ("compute_approach_pose", "Pure geometry (no motion): the base pose [x, y, yaw] that faces the object from standoff_m metres away along the robot->object line.", {"robot_x": "number", "robot_y": "number", "object_x": "number", "object_y": "number", "standoff_m": "number"}),
    ("navigate_to_pose", "Drive the base to a 2D world pose (x, y, yaw radians); interpolates toward the goal when the exact pose is unreachable. Can take minutes.", {"x": "number", "y": "number", "yaw": "number"}),
    ("sample_grasp_pose", "Sample grasp candidates for the named object from the current view (simple top-down grasp + GraspNet). Returns candidate ids with pre-grasp/grasp positions; 'none' when no grasp is found (approach closer, centre the object, retry).", {"object_name": "string"}),
    ("grasp_object", "Execute grasp candidate candidate_id (from the latest sample_grasp_pose of that object) with arm 0 (left) or 1 (right): pre-grasp approach, grasp, close. Verify with check_object_in_hand.", {"object_name": "string", "candidate_id": "integer", "arm": "integer"}),
    ("check_object_in_hand", "True when the gripper of the given arm holds an object.", {"arm": "integer"}),
    ("lift_arm", "Lift the given arm straight up (after a grasp).", {"arm": "integer"}),
    ("open_gripper", "Open the gripper of the given arm.", {"arm": "integer"}),
    ("close_gripper", "Close the gripper of the given arm.", {"arm": "integer"}),
    ("reset_torso", "Return the torso to its neutral pose.", {}),
    ("get_current_eef_pose", "Current end-effector pose of the given arm (position xyz, quaternion).", {"arm": "integer"}),
    ("get_sam3_mask", "Pixel count of the named object's SAM3 mask in the current head-camera view (0 = not visible).", {"object_name": "string"}),
    ("detect_objects", "Detect objects matching a text prompt in the current head-camera view (SAM3): boxes [x1,y1,x2,y2] in a 512x512 image with scores and mask pixel counts.", {"text_prompt": "string"}),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
REA_DIR = REPO_ROOT / "eval" / "behavior" / "online"
# Files whose bytes define the rea-online arm's prompts, routing and decision loop (their sha256 goes into every record).
REA_FILES = ("prompting.py", "behavior_routing.mjs", "runtime.mjs", "worker.py", "config.example.json")
# rea-online: skills that run as the decision's one action, and read-only helpers the model may call inside a decision.
REA_SKILLS = ("find_object_base_rotate", "find_object_torso_rotate", "navigate_to_pose", "grasp_object", "lift_arm",
               "open_gripper", "close_gripper", "reset_torso", "finish")
REA_HELPERS = ("get_object_pose", "get_robot_position", "compute_approach_pose", "sample_grasp_pose", "get_current_eef_pose",
                "get_sam3_mask", "detect_objects", "check_object_in_hand")
FINISH_SCHEMA = ("finish", "End the episode holding the target object. Accepted only when check_object_in_hand is True for the arm you "
                 "name (the runner checks it once, on that arm); otherwise it is refused and the episode continues.", {"arm": "integer"})

SDK_STOP_SENTENCE = ("Stop (reply with a one-line summary) as soon as check_object_in_hand is True and the arm is lifted, "
                     "or when a tool answers BUDGET EXHAUSTED. ")
REA_FINISH_SENTENCE = ("Call take_action with skill finish and the arm that holds the object as soon as check_object_in_hand is "
                        "True for that arm and it is lifted. ")


def sdk_instructions(task_prompt, budget_s, context=None):
    """The Agents SDK arm's instructions: the ASPIRE task prompt plus its typical-loop and budget paragraph."""
    instructions = (f"{task_prompt.strip()}\n\n"
                    "You control the robot ONLY through the tools. Continue from the current physical state; the scene has not been reset. "
                    "Preserve any object already held. Use the previous execution feedback to change approach, grasp candidate or arm when needed. Typical loop: find_object_base_rotate / find_object_torso_rotate to locate the object; get_object_pose and get_robot_position; "
                    "compute_approach_pose (standoff about 0.6-0.8 m) then navigate_to_pose; re-detect (find_object_torso_rotate, get_object_pose); sample_grasp_pose; grasp_object with a candidate id; "
                    "check_object_in_hand; lift_arm when held. If a grasp fails, open the gripper, back up or re-centre the object and try other candidates. "
                    f"The whole episode has a {budget_s:.0f} s wall-clock budget including your own thinking time; navigation and grasping take minutes each, so avoid needless calls. "
                    + SDK_STOP_SENTENCE +
                    "The task is judged by whether the object is held and lifted at the end; do not place it anywhere.")
    if context:
        instructions += "\nPrevious execution and current hand checks: " + json.dumps(context, ensure_ascii=False)
    return instructions


def online_task_context(task_prompt, budget_s):
    """The same domain text for the rea-online arm, with the SDK's stop sentence replaced by the finish wording."""
    text = sdk_instructions(task_prompt, budget_s)
    assert SDK_STOP_SENTENCE in text
    return text.replace(SDK_STOP_SENTENCE, REA_FINISH_SENTENCE)


def agent_model_settings(api, effort="medium", max_tokens=2048):
    """ModelSettings keyword arguments for the SDK arm's model, or None (chat_completions: the SDK defaults).
    tool_choice and parallel_tool_calls stay at the SDK defaults so the agent design is the same under both APIs."""
    if api in (None, "", "chat_completions"):
        return None
    if api != "responses":
        raise ValueError(f"unknown agent api {api!r}")
    return {"reasoning": {"effort": str(effort)}, "max_tokens": int(max_tokens), "store": False}


# Provider failures: infrastructure, not task outcomes. An episode that ends on one is recorded unscored and rerun.
PROVIDER_ERROR_CLASSES = ("APIConnectionError", "APITimeoutError", "RateLimitError", "AuthenticationError",
                          "PermissionDeniedError", "InternalServerError")
PROVIDER_ERROR_TEXT = re.compile(r"insufficient[ _]balance|insufficient[ _]quota|exceeded your current quota|quota exceeded|"
                                 r"quota[ _]exhausted|余额不足", re.I)
# An HTTP-200 Responses reply with status failed for a provider-side reason (openai-agents raises ModelBehaviorError
# "... terminal event `response.failed`. status=failed; error=ResponseError(code='server_error', ...)").
RESPONSE_FAILED_TEXT = re.compile(r"status=failed; error=ResponseError\(code='(server_error|rate_limit_exceeded)'")
# A 200 response whose body is `null` or not JSON reaches the Agents SDK's model adapter as None or a str (the chat
# completions adapter wraps it in a tuple) and fails on its first attribute access there: a relay failure, not model output.
MALFORMED_RESPONSE_TEXT = re.compile(r"'(NoneType|str|tuple)' object has no attribute '\w+'")
MODEL_ADAPTER_FILES = ("openai_responses.py", "openai_chatcompletions.py")


def classify_provider_failure(class_names, status_code=None, message=""):
    """Label of a provider failure from an exception's class names (MRO), HTTP status and message, or None."""
    names = set(class_names or ())
    for name in PROVIDER_ERROR_CLASSES:
        if name in names:
            return name
    if isinstance(status_code, int) and (status_code >= 500 or status_code == 408):  # 408: request timeout
        return f"HTTP {status_code}"
    if PROVIDER_ERROR_TEXT.search(message or ""):
        return "insufficient_balance_or_quota"
    if RESPONSE_FAILED_TEXT.search(message or ""):
        return "response_failed"
    return None


def provider_failure(exc):
    """classify_provider_failure for an exception, following its __cause__ / __context__ chain."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        label = classify_provider_failure([c.__name__ for c in type(exc).__mro__], getattr(exc, "status_code", None), str(exc))
        if label:
            return label
        if isinstance(exc, AttributeError) and MALFORMED_RESPONSE_TEXT.fullmatch(str(exc)):
            frames = traceback.extract_tb(exc.__traceback__)
            if frames and Path(frames[-1].filename).name in MODEL_ADAPTER_FILES:  # raised in the SDK's model adapter itself
                return "malformed_response"
        exc = exc.__cause__ or exc.__context__
    return None


# A Responses reply cut at max_output_tokens (openai-agents raises ModelBehaviorError "... `response.incomplete`. status=incomplete;
# incomplete_details=IncompleteDetails(reason='max_output_tokens')"). The SDK arm re-requests it at most CUT_RETRIES times.
CUT_RETRIES = 3


def is_cut_reply(exc):
    return type(exc).__name__ == "ModelBehaviorError" and "max_output_tokens" in str(exc)


def agent_failure(exc):
    """The SDK child's done fields for an exception that ended Runner.run: stop_reason (provider_error is unscored; max_turns,
    max_output_tokens and agent_error are task outcomes), the provider label, the HTTP status when there is one."""
    fields = {}
    provider = provider_failure(exc)
    if provider:
        fields.update(stop_reason="provider_error", provider_error=provider)
    elif "MaxTurns" in type(exc).__name__:
        fields["stop_reason"] = "max_turns"
    elif is_cut_reply(exc):
        fields["stop_reason"] = "max_output_tokens"  # still cut after CUT_RETRIES re-requests: a scored task outcome
    else:
        fields["stop_reason"] = "agent_error"
    if isinstance(getattr(exc, "status_code", None), int):
        fields["http_status"] = exc.status_code
    return fields


def outcome_fields(stop_reason, success, reward):
    """Score fields for result.json: an episode that ended on a provider failure is unscored (success None), so the queue
    driver reruns it; every other stop keeps the predicate's score."""
    if stop_reason == "provider_error":
        return {"success": None, "reward": None, "unscored": True, "stop_reason": "provider_error",
                "unused_task_completed": success}
    return {"success": success, "reward": reward, "stop_reason": stop_reason}


def finish_gate(check_value, arm):
    """finish(arm) after the runner's single check_object_in_hand(arm): (accepted, message for the agent)."""
    value = str(check_value).strip()
    if value.startswith("BUDGET EXHAUSTED"):
        return False, value
    if value.lower() == "true":
        return True, f"ACCEPTED: check_object_in_hand(arm={arm}) is True; the episode ends now."
    return False, (f"REFUSED: check_object_in_hand(arm={arm}) returned {value}, so arm {arm} does not hold the object. "
                   "The episode continues.")


def online_tool_specs():
    """(skills, helpers) of the rea-online arm as [{name, description, parameters}], taken from TOOL_SCHEMAS (+ finish)."""
    schemas = {name: (desc, params) for name, desc, params in TOOL_SCHEMAS + [FINISH_SCHEMA]}
    spec = lambda n: {"name": n, "description": schemas[n][0], "parameters": dict(schemas[n][1])}
    return [spec(n) for n in REA_SKILLS], [spec(n) for n in REA_HELPERS]


def file_sha256(paths):
    import hashlib
    return {Path(p).name: (hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None) for p in paths}


def online_file_hashes():
    return file_sha256([REA_DIR / name for name in REA_FILES])


# Files whose bytes define either arm (task text, tool descriptions, finish gate, runner loop, the online runtime and the
# framework it runs on). Their sha256 goes into every agent_trace.json / result.json, next to the deployed commit.
CODE_FILES = ("eval/behavior/runner/agents.py", "eval/behavior/runner/run_episode.py",
              "eval/behavior/runner/behavior_tasks.py", *(f"eval/behavior/online/{name}" for name in REA_FILES))


def code_file_hashes(extra=()):
    """sha256 of CODE_FILES (+ extra paths) keyed by repo-relative path (basenames collide), plus "src/" = one digest over
    every file of src/ (sorted path and sha256 lines), which the online runtime imports through src/framework/index.mjs."""
    import hashlib
    out = {}
    for path in [REPO_ROOT / rel for rel in CODE_FILES] + [Path(p) for p in extra if p]:
        path = Path(path).resolve()
        try:
            key = str(path.relative_to(REPO_ROOT))
        except ValueError:
            key = str(path)
        out[key] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    src = REPO_ROOT / "src"
    lines = [f"{f.relative_to(REPO_ROOT)} {hashlib.sha256(f.read_bytes()).hexdigest()}"
             for f in sorted(src.rglob("*")) if f.is_file() and "__pycache__" not in f.parts]
    out["src/"] = hashlib.sha256("\n".join(lines).encode()).hexdigest() if lines else None
    return out


def repo_commit():
    """The deployed commit (<repo>/COMMIT, written by the deployment), or the git HEAD of a checkout, or None."""
    pin = REPO_ROOT / "COMMIT"
    if pin.is_file():
        return pin.read_text().strip() or None
    if not (REPO_ROOT / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def with_clock(value, deadline, budget_s, seconds=None):
    """SDK arm (Responses API): a tool result plus how long the call took and the remaining program budget, the clock the online agent's
    decision message states. The runner records the bare value; only the agent sees the suffix."""
    if str(value).startswith("BUDGET EXHAUSTED"):
        return value
    took = f"took {round(seconds)} s; " if seconds is not None else ""
    return f"{value} [{took}about {max(0, round(deadline - time.monotonic()))} s of the {budget_s:.0f} s budget remain]"


def _np():
    import numpy as np
    return np


def _tolist(x):
    np = _np()
    if hasattr(x, "detach"): x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.size > 64: return f"<array shape={list(x.shape)} dtype={x.dtype}>"  # images / point clouds are not serialised into traces
        return [round(float(v), 4) for v in x.reshape(-1).tolist()]
    if isinstance(x, (list, tuple)): return [(_tolist(v) if isinstance(v, (list, tuple)) or hasattr(v, "shape") else v) for v in x]
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, (np.floating, np.integer)): return round(float(x), 4)
    return x


def _text(x):
    if isinstance(x, str): return x
    try: return json.dumps(_tolist(x))
    except Exception: return str(x)[:1000]


# =============================== runner side (Isaac process) ===============================
class ToolServer:
    """Executes the agent's tool calls on the (budget-wrapped) API functions and records them."""

    def __init__(self, fns, deadline, budget_exc):
        self.fns, self.deadline, self.budget_exc = fns, deadline, budget_exc
        self.calls = []; self.tool_results = []; self.grasp_cache = {}; self.stop = None; self.t0 = time.monotonic()
        self.finishes = []; self.finished = False; self.observations = []

    def _call(self, name, *a, **k):
        t = time.monotonic(); rec = {"tool": name, "args": _tolist(list(a)) if a else [], "kwargs": {kk: _tolist(v) for kk, v in k.items()}, "t": round(t - self.t0, 1)}
        try:
            out = self.fns[name](*a, **k); rec["ok"] = True; rec["s"] = round(time.monotonic() - t, 2); rec["result"] = _text(out); return out
        except self.budget_exc:
            rec["ok"] = False; rec["error"] = "program budget exhausted"; rec["s"] = round(time.monotonic() - t, 2); self.stop = "program_budget"; return "BUDGET EXHAUSTED: stop now."
        except Exception as e:
            rec["ok"] = False; rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"; rec["s"] = round(time.monotonic() - t, 2); return f"ERROR {type(e).__name__}: {str(e)[:300]}"
        finally:
            self.calls.append(rec)

    def execute(self, name, args):
        started = time.monotonic()
        try:
            value = self._execute(name, args)
        except Exception as e:  # malformed model arguments (missing key, wrong type, non-object args): an error the agent sees
            value = f"ERROR {type(e).__name__}: {str(e)[:300]}"  # (WallBudgetExceeded is a BaseException: _call handles it)
        self.tool_results.append({"tool": name, "args": args or {}, "result": value,
                                  "t": round(started - self.t0, 3), "s": round(time.monotonic() - started, 3)})
        return value

    def finish(self, args):
        """rea-online finish(arm): ONE check_object_in_hand(arm) on the named arm (which also leaves the controller on that arm,
        the arm the predicate reads). True ends the episode; anything else refuses it. The other arm is never probed."""
        started = time.monotonic(); raw = (args or {}).get("arm")
        try:
            arm = int(raw)
        except (TypeError, ValueError):
            arm = None
        if arm not in (0, 1):
            accepted, message, value = False, f"REFUSED: finish needs arm 0 (left) or 1 (right), got {raw!r}. The episode continues.", None
        else:
            value = self.execute("check_object_in_hand", {"arm": arm})
            accepted, message = finish_gate(value, arm)
        self.finished = self.finished or accepted
        self.finishes.append({"arm": raw, "check": value, "accepted": accepted, "t": round(started - self.t0, 3),
                              "s": round(time.monotonic() - started, 3)})
        return message

    def _execute(self, name, args):
        np = _np(); a = args or {}
        if self.deadline and time.monotonic() > self.deadline:
            self.stop = "program_budget"; return "BUDGET EXHAUSTED: stop now."
        if name in ("find_object_base_rotate", "find_object_torso_rotate", "get_sam3_mask"): return _text(self._call(name, a["object_name"]))
        if name == "get_object_pose":
            out = self._call("get_object_pose", a["object_name"], return_bbox_extent=True)
            if isinstance(out, str): return out
            try:
                pos, quat = out[0], out[1]; extent = out[2] if len(out) > 2 else None
                if pos is None: return "not found"
                return json.dumps({"position_xyz": _tolist(pos), "quaternion_wxyz": _tolist(quat), "bbox_extent": _tolist(extent) if extent is not None else None})
            except Exception: return _text(out)
        if name == "get_robot_position":
            out = self._call("get_robot_position")
            if isinstance(out, str): return out
            try:
                pos = _tolist(out[0]); yaw = float(out[2]) if len(out) > 2 else None
                return json.dumps({"x": pos[0], "y": pos[1], "yaw": round(yaw, 4) if yaw is not None else None})
            except Exception: return _text(out)
        if name == "compute_approach_pose":
            rx, ry, ox, oy, st = (float(a[k]) for k in ("robot_x", "robot_y", "object_x", "object_y", "standoff_m"))
            dx, dy = ox - rx, oy - ry; dist = math.hypot(dx, dy)
            if dist < 1e-6: return json.dumps({"x": rx, "y": ry, "yaw": 0.0, "distance_m": 0.0})
            t = max(0.0, dist - st) / dist
            return json.dumps({"x": round(rx + dx * t, 3), "y": round(ry + dy * t, 3), "yaw": round(math.atan2(dy, dx), 4), "distance_m": round(dist, 3)})
        if name == "navigate_to_pose": return _text(self._call("navigate_to_pose", np.array([float(a["x"]), float(a["y"]), float(a["yaw"])], dtype=float)))
        if name == "sample_grasp_pose":
            out = self._call("sample_grasp_pose", a["object_name"])
            if isinstance(out, str): return out
            try:
                pre, gr = out
                if pre is None or gr is None or not len(pre): return "none"
                self.grasp_cache[a["object_name"]] = (list(pre), list(gr))
                rows = [{"id": i, "pregrasp_position": _tolist(p[0]), "grasp_position": _tolist(g[0])} for i, (p, g) in enumerate(zip(pre, gr))]
                return json.dumps({"object": a["object_name"], "candidates": rows, "note": "id 0 = simple top-down grasp when available; later ids = GraspNet candidates"})
            except Exception: return _text(out)
        if name == "grasp_object":
            cache = self.grasp_cache.get(a["object_name"])
            if not cache: return "ERROR: call sample_grasp_pose for this object first"
            pre, gr = cache; cid = int(a.get("candidate_id", 0)); arm = int(a.get("arm", 0))
            if not (0 <= cid < len(pre)): return f"ERROR: candidate_id out of range (0..{len(pre) - 1})"
            return _text(self._call("grasp_object", pre[cid], gr[cid], a["object_name"], arm=arm))
        if name in ("check_object_in_hand", "lift_arm", "open_gripper", "close_gripper", "get_current_eef_pose"): return _text(self._call(name, arm=int(a.get("arm", 0))))
        if name == "reset_torso": return _text(self._call("reset_torso"))
        if name == "detect_objects":
            rgb = self._call("get_env_observation")
            if isinstance(rgb, str): return rgb
            try:
                img = rgb[0]; masks = self._call("segment_sam3_text_prompt", img, a["text_prompt"])
                if isinstance(masks, str): return masks
                out = [{"box": _tolist(m.get("box")) if m.get("box") is not None else None, "score": round(float(m.get("score", 0)), 3), "pixels": int(np.asarray(m["mask"]).sum()) if m.get("mask") is not None else None} for m in (masks or [])]
                return json.dumps({"prompt": a["text_prompt"], "detections": out[:10], "image_size": list(np.asarray(img).shape[:2])})
            except Exception as e: return f"ERROR {type(e).__name__}: {str(e)[:200]}"
        return f"ERROR: unknown tool {name}"


def _agent_python(args):
    sdk_python = str(getattr(args, "sdk_python", None) or sys.executable)
    if os.sep in sdk_python or (os.altsep and os.altsep in sdk_python):
        sdk_python = str(Path(sdk_python).expanduser().absolute())
    return sdk_python


# The agent's done message: the fields the runner keeps.
DONE_KEYS = ("final_output", "turns", "error", "usage", "stop_reason", "provider_error", "http_status", "truncated_retries")
# rea-online: past the budget nothing runs; calls answer BUDGET EXHAUSTED for this long while the runtime ends and writes its
# records (outcome-neutral: no skill or check executes in it).
BUDGET_GRACE_S = 10.0


class ChildLines:
    """Complete lines from the agent process's stdout.

    select() sees only the pipe. A buffered readline() after it can read several lines at once and return the first, and
    the next select() then waits on an empty pipe while the rest sit unseen in Python's buffer: a progress line read
    together with the tool call written right after it would leave that call unanswered until the budget ended. Reading the
    descriptor itself keeps every received line visible."""

    def __init__(self, stream):
        self.fd = stream.fileno(); self.pending = []; self.partial = b""; self.eof = False

    def get(self, timeout):
        """The next line (str, with its newline), None if none arrives within timeout s, or "" at end of file.

        The timeout bounds the whole call: bytes without a newline do not restart it (they stay buffered for the next call)."""
        end = time.monotonic() + max(0.0, timeout)
        while not self.pending:
            if self.eof:
                return ""
            ready, _, _ = select.select([self.fd], [], [], max(0.0, end - time.monotonic()))
            if not ready:
                return None
            chunk = os.read(self.fd, 65536)
            if not chunk:
                self.eof = True
                if self.partial:
                    self.pending.append(self.partial); self.partial = b""
                continue
            *lines, self.partial = (self.partial + chunk).split(b"\n")
            self.pending.extend(lines)
            if not self.pending and time.monotonic() >= end:
                return None
        return self.pending.pop(0).decode("utf-8", "replace") + "\n"


REA_RECORDS = ("online", "online-events.jsonl", "online-decisions.jsonl", "online-runtime-result.json", "agent_runtime.stderr")


def _set_aside_previous_attempt(ep_dir):
    """A rerun of an unscored episode keeps the previous attempt's online records under online.prev-<k>/ instead of mixing them."""
    present = [name for name in REA_RECORDS if (ep_dir / name).exists()]
    if not present:
        return None
    k = 1
    while (ep_dir / f"online.prev-{k}").exists():
        k += 1
    target = ep_dir / f"online.prev-{k}"; target.mkdir()
    for name in present:
        (ep_dir / name).rename(target / name)
    return str(target)


def online_config(args, instruction, task_prompt, budget_s, base_url, agent_python, ep_dir, deadline_epoch):
    """The rea-online runtime's configuration: the behavioural keys of config.example.json (or --online-config) plus this episode's
    model, budget, task text and tool schemas. Holds no credentials (the key travels in the AGENT_API_KEY environment variable)."""
    source = Path(getattr(args, "online_config", None) or (REA_DIR / "config.example.json"))
    config = json.loads(source.read_text())
    skills, helpers = online_tool_specs()
    config.update({
        "model": args.agent_model,
        "reasoning_effort": str(getattr(args, "agent_effort", None) or config.get("reasoning_effort", "medium")),
        "max_output_tokens": int(getattr(args, "agent_max_tokens", None) or config.get("max_output_tokens", 2048)),
        "max_decisions": int(getattr(args, "agent_max_decisions", None) or config.get("max_decisions", 40)),
        "base_url": base_url, "agent_python": agent_python, "budget_s": float(budget_s), "deadline_epoch": deadline_epoch,
        "task": getattr(args, "task", None), "seed": getattr(args, "seed", None),
        "instruction": instruction, "task_context": online_task_context(task_prompt, budget_s),
        "episode_dir": str(ep_dir), "online_dir": str(ep_dir / "online"),
        "skills": skills, "helper_specs": helpers,
        "agents_sdk_arm": str(Path(__file__).resolve()),
        "worker": str(getattr(args, "online_worker", None) or os.environ.get("REA_BEHAVIOR_WORKER") or (REA_DIR / "worker.py")),
        "source_config": str(source.resolve()), "file_sha256": online_file_hashes(),
    })
    return config


def run_episode(fns, instruction, task_prompt, initial_jpg, args, budget_s, api_key, base_url, budget_exc, ep_dir, log, observe=None):
    """Serve the agent's tool calls in the current scene; model waits and actions share one deadline.

    observe(path): renders the current head-camera JPEG to path (rea-online "observe" requests); it runs inside the budget."""
    t0 = time.monotonic(); deadline = t0 + budget_s
    ep_dir = Path(ep_dir).resolve()
    style = getattr(args, "agent_style", None) or "sdk"
    if style not in ("sdk", "rea-online"):
        raise ValueError(f"unknown agent style {style!r}")
    api = getattr(args, "agent_api", None) or "chat_completions"
    effort = getattr(args, "agent_effort", None) or "medium"
    max_tokens = int(getattr(args, "agent_max_tokens", None) or 2048)
    sdk_python = _agent_python(args)
    server = ToolServer(fns, deadline, budget_exc)
    img_path = None; previous = None
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CARB_", "EXP_PATH", "ISAAC_", "OMNI_", "PYTHONPATH"))}
    env["AGENT_API_KEY"] = api_key; env["PYTHONUNBUFFERED"] = "1"
    if style == "rea-online":
        previous = _set_aside_previous_attempt(ep_dir)
        (ep_dir / "online").mkdir(parents=True)
        config = online_config(args, instruction, task_prompt, budget_s, base_url, sdk_python, ep_dir, time.time() + budget_s)
        config_path = ep_dir / "online" / "config.json"; config_path.write_text(json.dumps(config, indent=1))
        command = [str(getattr(args, "agent_node", None) or "node"), str(REA_DIR / "runtime.mjs"), str(config_path)]
        stderr_name = "agent_runtime.stderr"
    else:
        if initial_jpg and args.agent_images:
            img_path = str(Path(ep_dir) / "agent_initial.jpg"); Path(img_path).write_bytes(initial_jpg)
        cfg = {"model": args.agent_model, "max_turns": int(args.agent_max_turns),
               "llm_timeout": min(float(args.agent_llm_timeout), budget_s), "instruction": instruction,
               "task_prompt": task_prompt, "budget_s": budget_s, "image": img_path,
               "base_url": base_url, "context": getattr(args, "agent_context", None),
               "api": api, "effort": effort, "max_tokens": max_tokens}
        command = [sdk_python, str(Path(__file__).resolve()), "--agent-child", json.dumps(cfg)]
        stderr_name = "agent_child.stderr"
    result = {"final_output": None, "turns": None, "error": None, "usage": None}
    child = None; end_by = deadline; grace = {"on": False}
    clock = style == "sdk" and api == "responses"
    grace_s = BUDGET_GRACE_S if style == "rea-online" else 0.0  # the SDK child reports its progress itself and is stopped at once

    def reply(rid, value):
        child.stdin.write(json.dumps({"type": "result", "id": rid, "value": value}) + "\n")
        child.stdin.flush()

    def budget_over():
        """The budget has ended: nothing runs any more. rea-online: every further call answers BUDGET EXHAUSTED for up to
        BUDGET_GRACE_S while the runtime ends and writes its records (its last decision); then it is terminated."""
        nonlocal end_by
        server.stop = server.stop or "program_budget"
        if not grace["on"]:
            grace["on"] = True; end_by = time.monotonic() + grace_s

    def take_done(msg):
        result.update({k: msg.get(k) for k in DONE_KEYS if k in msg and not (k in ("turns", "usage") and msg.get(k) is None)})

    with open(str(Path(ep_dir) / stderr_name), "w") as child_stderr:
        try:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=child_stderr,
                                     text=True, env=env, cwd=str(ep_dir))
            lines = ChildLines(child.stdout)
            while True:
                remaining = end_by - time.monotonic()
                if remaining <= 0:
                    if server.finished or grace["on"]:
                        break
                    if not server.stop:
                        result["error"] = "episode wall budget exhausted (agent loop)"
                    budget_over(); continue
                line = lines.get(min(remaining, 1.0))
                if line is None:
                    if child.poll() is not None:
                        if not server.finished and not grace["on"]: result["error"] = f"agent process exited rc={child.returncode}"
                        break
                    continue
                if not line:
                    if not server.finished and not grace["on"]: result["error"] = f"agent process closed the tool channel (rc={child.poll()})"
                    break
                try: msg = json.loads(line)
                except ValueError: continue
                kind = msg.get("type")
                if kind in ("call", "observe"):
                    tool, args_in = msg.get("tool"), msg.get("args") or {}
                    if server.finished:
                        value = "ERROR: the episode has finished; no further calls run."
                    elif server.stop:
                        value = "BUDGET EXHAUSTED: stop now."
                    elif kind == "observe":
                        value = _observe(server, observe, ep_dir, budget_exc) if style == "rea-online" else "ERROR: unknown request observe"
                    elif style == "rea-online" and tool == "finish":
                        value = server.finish(args_in)
                        if server.finished: end_by = min(deadline, time.monotonic() + 30.0)  # wait briefly for the runtime's records
                    elif style == "rea-online" and tool not in REA_SKILLS + REA_HELPERS:
                        value = f"ERROR: unknown tool {tool}"
                    else:
                        value = server.execute(tool, args_in)
                        if clock:
                            value = with_clock(value, deadline, budget_s, server.tool_results[-1]["s"])
                    try:
                        reply(msg.get("id"), value)
                    except BrokenPipeError:
                        # The child closed the channel while this call ran (rea-online: a decision timeout ends runtime.mjs, which
                        # sends done and exits before this reply). Read that done instead of dropping it.
                        tail = None
                        try: child.wait(timeout=5)
                        except subprocess.TimeoutExpired: pass
                        if child.poll() is not None:
                            while (rest := lines.get(5.0)):
                                try: m = json.loads(rest)
                                except ValueError: continue
                                if m.get("type") == "progress": take_done(m)
                                if m.get("type") == "done": tail = m; break
                        if tail:
                            take_done(tail)
                        elif not server.finished and not server.stop:
                            result["error"] = "agent process closed the tool channel"
                        break
                    if server.stop: budget_over()
                elif kind == "done":
                    take_done(msg); break
                elif kind == "progress":  # running turns / usage, so a budget ending keeps them
                    take_done({k: msg.get(k) for k in ("turns", "usage")})
                elif kind == "log":
                    log(stage="agent", **{k: v for k, v in msg.items() if k != "type"})
        except budget_exc:
            server.stop = "program_budget"
            result["error"] = "episode execution budget exhausted"
        finally:
            if child is not None:
                try:
                    if child.poll() is None:
                        child.terminate()  # the online runtime kills its worker and writes its records on SIGTERM
                        try: child.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=5)
                except budget_exc:
                    server.stop = "program_budget"
                    child.kill()
                    child.wait(timeout=5)
                finally:
                    try: child.stdin.close()
                    except BrokenPipeError: pass
                    child.stdout.close()
    if style == "rea-online" and result.get("turns") is None:  # the runtime was stopped before it reported
        try: runtime_result = json.loads((ep_dir / "online-runtime-result.json").read_text())
        except (OSError, ValueError): runtime_result = {}
        if runtime_result.get("decisions") is not None:
            result["turns"] = runtime_result["decisions"]
            if runtime_result.get("usage_total") is not None:
                result["usage"] = {**runtime_result["usage_total"], "decisions": runtime_result["decisions"]}
    if server.finished:
        stop = "agent_finished"
    else:
        stop = server.stop or result.get("stop_reason") or ("agent_error" if result.get("error") else "agent_finished")
    observe_s = round(sum(o.get("s", 0) for o in server.observations), 3)
    result.update({"stop_reason": stop, "tool_calls": server.calls, "tool_results": server.tool_results,
                   "tool_calls_count": len(server.tool_results), "tool_time_s": round(sum(c.get("s", 0) for c in server.tool_results), 3),
                   "wall_s": round(time.monotonic() - t0, 3), "past_deadline_s": round(max(0.0, time.monotonic() - deadline), 3),
                   "model": args.agent_model,
                   "max_turns": int(args.agent_max_turns), "images": bool(img_path) or style == "rea-online",
                   "agent_style": style, "agent_api": "responses" if style == "rea-online" else api, "effort": effort,
                   "max_output_tokens": max_tokens if (style == "rea-online" or api == "responses") else None})
    if style == "rea-online":
        result.update({"finishes": server.finishes, "observations": server.observations, "observe_time_s": observe_s,
                       "online_file_sha256": online_file_hashes(), "previous_attempt": previous,
                       "max_decisions": int(getattr(args, "agent_max_decisions", None) or 40)})
    result["llm_time_s"] = round(max(0.0, result["wall_s"] - result["tool_time_s"] - observe_s), 3)
    return result


def _observe(server, observe, ep_dir, budget_exc):
    """Render the current head-camera JPEG for a rea-online decision; the render runs inside the episode budget."""
    started = time.monotonic(); n = len(server.observations) + 1
    path = Path(ep_dir) / "online" / "observations" / f"obs-{n:03d}.jpg"
    if server.deadline and started > server.deadline:
        server.stop = "program_budget"; value = "BUDGET EXHAUSTED: stop now."
    elif observe is None:
        value = "ERROR: no camera available"
    else:
        try:
            path.parent.mkdir(parents=True, exist_ok=True); observe(str(path)); value = str(path)
        except budget_exc:
            server.stop = "program_budget"; value = "BUDGET EXHAUSTED: stop now."
        except Exception as e:
            value = f"ERROR {type(e).__name__}: {str(e)[:300]}"
    server.observations.append({"value": value, "t": round(started - server.t0, 3), "s": round(time.monotonic() - started, 3)})
    return value


# =============================== agent side (clean venv process) ===============================
def _agent_child(cfg):
    from agents import Agent, Runner, RunConfig, RunHooks, FunctionTool, set_tracing_disabled
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from openai import AsyncOpenAI
    set_tracing_disabled(True)
    out = sys.stdout; inp = sys.stdin; counter = {"n": 0}

    def rpc(tool, args):
        counter["n"] += 1; rid = counter["n"]
        out.write(json.dumps({"type": "call", "id": rid, "tool": tool, "args": args}) + "\n"); out.flush()
        while True:
            line = inp.readline()
            if not line: raise RuntimeError("runner closed the tool channel")
            msg = json.loads(line)
            if msg.get("type") == "result" and msg.get("id") == rid: return str(msg.get("value"))

    tools = []
    for name, desc, params in TOOL_SCHEMAS:
        schema = {"type": "object", "properties": {k: {"type": t} for k, t in params.items()}, "required": list(params), "additionalProperties": False}
        async def invoke(ctx, args_json, _name=name):
            try: args = json.loads(args_json) if args_json else {}
            except Exception as e: return f"ERROR: bad arguments ({e})"
            return rpc(_name, args)
        tools.append(FunctionTool(name=name, description=desc, params_json_schema=schema, on_invoke_tool=invoke, strict_json_schema=True))
    client = AsyncOpenAI(api_key=os.environ.get("AGENT_API_KEY"), base_url=cfg["base_url"], timeout=cfg["llm_timeout"], max_retries=2)
    instructions = sdk_instructions(cfg["task_prompt"], cfg["budget_s"], cfg.get("context"))
    settings = agent_model_settings(cfg.get("api"), cfg.get("effort", "medium"), cfg.get("max_tokens", 2048))
    cut = {"n": 0}
    if settings is None:  # chat completions: the SDK defaults
        model = OpenAIChatCompletionsModel(model=cfg["model"], openai_client=client)
        agent = Agent(name="R1Pro operator", instructions=instructions, tools=tools, model=model)
    else:
        from agents import ModelSettings, OpenAIResponsesModel
        from agents.retry import ModelRetrySettings
        model = OpenAIResponsesModel(cfg["model"], client)

        def retry_cut(ctx):
            """A reply cut at max_output_tokens is re-requested (at most CUT_RETRIES times per turn), so one cut reply does not
            end the episode (online loses only that decision). Every other error is left to the client's own retries."""
            hit = is_cut_reply(ctx.error)
            cut["n"] += int(hit)
            return hit
        agent = Agent(name="R1Pro operator", instructions=instructions, tools=tools, model=model,
                      model_settings=ModelSettings(**settings, retry=ModelRetrySettings(max_retries=CUT_RETRIES, policy=retry_cut)))
    content = [{"type": "input_text", "text": f"Task: {cfg['instruction']} " + ("The attached image is the robot's current head camera when you take over (the object may be out of view). " if cfg.get("image") else "") + "Begin."}]
    if cfg.get("image"):
        content.append({"type": "input_image", "detail": "high", "image_url": "data:image/jpeg;base64," + base64.b64encode(Path(cfg["image"]).read_bytes()).decode()})
    done = {"type": "done", "final_output": None, "turns": None, "error": None, "usage": None, "stop_reason": None}

    class Progress(RunHooks):
        """After every model response: the running turn count and usage to the runner, so a budget ending keeps them."""
        def __init__(self):
            self.turns, self.usage = 0, {"input_tokens": 0, "output_tokens": 0, "requests": 0}

        async def on_llm_end(self, context, agent, response):
            u = getattr(response, "usage", None); self.turns += 1
            self.usage["input_tokens"] += int(getattr(u, "input_tokens", 0) or 0)
            self.usage["output_tokens"] += int(getattr(u, "output_tokens", 0) or 0)
            self.usage["requests"] += 1
            out.write(json.dumps({"type": "progress", "turns": self.turns, "usage": dict(self.usage)}) + "\n"); out.flush()
    progress = Progress()

    async def go():
        return await Runner.run(agent, input=[{"role": "user", "content": content}], max_turns=cfg["max_turns"], hooks=progress,
                                run_config=RunConfig(model=model, tracing_disabled=True))
    try:
        res = asyncio.run(go())
        done["final_output"] = str(res.final_output)[:2000]; done["stop_reason"] = "agent_finished"
        try:
            done["turns"] = len(res.raw_responses)
            u = [r.usage for r in res.raw_responses if getattr(r, "usage", None)]
            done["usage"] = {"input_tokens": sum(getattr(x, "input_tokens", 0) or 0 for x in u), "output_tokens": sum(getattr(x, "output_tokens", 0) or 0 for x in u), "requests": len(u)}
        except Exception: pass
    except Exception as e:
        key = os.environ.get("AGENT_API_KEY") or ""
        msg = f"{type(e).__name__}: {str(e)[:400]}"; done["error"] = msg.replace(key, "[REDACTED]") if key else msg
        done.update(agent_failure(e))
        tb = traceback.format_exc()[-3000:]
        sys.stderr.write(tb.replace(key, "[REDACTED]") if key else tb)
    if done["turns"] is None:
        done.update(turns=progress.turns, usage=dict(progress.usage))
    done["truncated_retries"] = cut["n"]
    out.write(json.dumps(done) + "\n"); out.flush()


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--agent-child":
    _agent_child(json.loads(sys.argv[2]))
