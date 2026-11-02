#!/usr/bin/env python3
"""One BEHAVIOR-1K episode inside ASPIRE's environment: frozen programs (single or routed) or an online agent.

Builds ASPIRE's R1Pro code-execution environment from an env config, resets the requested task instance ("seed"), then
runs a LADDER of frozen programs (REA's escalation ladder on BEHAVIOR):

  mode single             run programs[0] once (the zero-shot / fixed-program arms)
  mode routed-exception   run programs[0]; run the next program only if the previous one raised (sandbox_rc != 0)
  mode routed-judge       run programs[0]; ask the visual judge (initial head frame vs. current head frame + instruction);
                          run the next program only while the judge says "not done", up to --max-recoveries

Success is the benchmark predicate (low-level env task_completed()) read ONCE after the last program run; the runner
never uses it for control flow. All program runs of an episode share one program budget (--program-budget, default
1500 s, clock starts at the first program statement). The budget is enforced at the API boundary: once it is exhausted,
every exposed API function raises WallBudgetExceeded (a BaseException, so a program's `except Exception` cannot swallow
it; ASPIRE's sandbox catches it and returns sandbox_rc=1). Interrupting Isaac from a thread shuts Kit down without
artifacts, so no interrupt_main watchdog is used; a last-resort watchdog fires at budget + 600 s. Artifacts:
  <out>/episodes/behavior/<task>/seed_<N>/{result.json, runs/<i>.json, initial.jpg, final_<i>.jpg, final.jpg}

Judge and recovery options:
  --judge-view front      judge frames from an external camera moved in front of the robot (two overviews + one close-up per
                          gripper), rendered through the PT perception path and restored afterwards; head frames stay on disk
  --judge-prompt v3       pick-up criterion (held in a gripper, lifted; no placement needed) and the multi-view image list
  --recovery-guard inhand recovery rungs get a prelude that lifts and stops when check_object_in_hand() is already true
  --judge-log             single / routed-exception: call the judge after every run for calibration only (never decides)
  mode routed-inhand      alternative to the visual judge: recover only while check_object_in_hand() is false for both arms (the
                          same proprioceptive signal the recovery guard uses); every judge record also logs this signal
  A program's own `raise SystemExit(0)` / `sys.exit()` is a clean stop (sandbox_rc 0, recorded as program_exit), not an exception.

Online agents (mode agents-sdk): --agent-style sdk (one OpenAI Agents SDK run over the R1Pro tools; --agent-api
chat_completions or responses) or rea-online (REA's episode runtime, eval/behavior/online). Both share this reset, budget clock,
ToolServer and score. The agent's credentials are AGENT_API_KEY / AGENT_BASE_URL when AGENT_API_KEY is set, else REA_API_KEY /
REA_BASE_URL, which the judge also uses. An agent that stops on a provider failure (after retries) leaves the episode unscored (success None, unscored,
stop_reason provider_error) so the queue driver reruns it from scratch; task outcomes are never rerun.
"""
import argparse, base64, io, json, os, re, subprocess, sys, threading, time, traceback, _thread, urllib.request, urllib.error
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

# task table: five navigate-and-pick-up tasks (behavior_tasks.py next to this file)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from behavior_tasks import TASKS5, CONFIGS, INSTRUCTIONS, CRITERION  # noqa: E402

# ---- visual judge (the Python port of src/infer-judge.mjs: default prompt, agentview note, initial frame on) ----
JUDGE_SYSTEM = ("You are a strict visual inspector for a robot manipulation benchmark. Report only what is visible in the image. "
                "Answer with one JSON object and nothing else.")
AGENTVIEW_NOTE = "External front camera."


def judge_prompt(instruction):
    return "\n".join([
        f'Instruction given to the robot: "{instruction}"',
        f"Two images are attached: the first shows the robot workspace at the start of the episode, before the robot moved; the second shows the workspace now. {AGENTVIEW_NOTE}",
        "Question: judging only from what is visible in the second image, has the instruction been completed? Consider the end state the instruction asks for (object on/in the target, drawer open, switch on, ...), not what the robot seems to be doing.",
        "If the target state is hidden or ambiguous, answer done=false or lower your confidence.",
        'Answer exactly: {"done": true|false, "confidence": <number 0.0-1.0>, "why": "<at most 12 words>"}',
    ])




def judge_prompt_v3(task, instruction, tags):
    views = ", ".join({"front_high": "a view from in front of the robot, above head height, looking at the head and any raised arm",
                       "front_left": "an overview from the robot's front-left at torso height", "front_right": "an overview from the robot's front-right at torso height",
                       "overview_left": "an overview from the robot's front-left", "overview_right": "an overview from the robot's front-right",
                       "gripper0": "a close-up of the LEFT gripper", "gripper1": "a close-up of the RIGHT gripper", "head": "the robot's head camera"}.get(t, t) for t in tags)
    return "\n".join([
        f'Instruction given to the robot: "{instruction}"',
        CRITERION[task],
        f"Images: image 1 is the robot's head camera at the start of the episode, before the robot moved (it shows where the object was). The following {len(tags)} images were taken NOW, at the end, by an external camera placed in front of the robot: {views}.",
        "After a successful pick-up this robot usually raises the holding arm straight up, so the held object often appears high above the robot's head, near the top of the image, and may look stretched by the wide-angle lens.",
        "An object resting on the robot's wheeled base or platform, on the floor, on a table or on any furniture is NOT held, even if it touches the robot or a gripper.",
        "Question: has the instruction been completed as defined above? Answer done=true only if the target object is clearly inside the closed fingers of one gripper at the end of an arm, raised off the floor/table, in at least one of the current images. If the object lies on any surface (including the robot's base) or is not visible in any current image, answer done=false.",
        'Answer exactly: {"done": true|false, "confidence": <number 0.0-1.0>, "why": "<at most 12 words>"}'])


GUARD_PRELUDE = '''# REA recovery guard (harness-added prelude): a previous attempt may have left the target object in a gripper; a recovery program must not undo it.
# (check_object_in_hand(arm) switches the controller's active arm; the loop ends on the holding arm when one holds, and the
#  recovery program's own API calls select arms afterwards.)
for _guard_arm in (0, 1):
    try:
        _guard_held = bool(check_object_in_hand(arm=_guard_arm))
    except Exception:
        _guard_held = False
    if _guard_held:
        print("recovery guard: object already in hand (arm %d); lifting and stopping" % _guard_arm)
        try:
            lift_arm(arm=_guard_arm)
        except Exception:
            pass
        try:
            check_object_in_hand(arm=_guard_arm)  # leave the controller's active arm on the holding arm (the benchmark predicate reads it)
        except Exception:
            pass
        raise SystemExit(0)

'''

SYSTEM_EXIT_CLEAN = re.compile(r"^SystemExit(?::\s*(0|None))?\s*$")


def clean_exit_line(stderr):
    lines = [l.strip() for l in (stderr or "").splitlines() if l.strip()]
    return lines[-1] if lines and SYSTEM_EXIT_CLEAN.match(lines[-1]) else None


def parse_judge(text):
    for m in re.finditer(r"\{[^{}]*\}", text or ""):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(obj, dict) and "done" in obj:
            done = obj["done"]
            if isinstance(done, str):
                done = {"true": True, "yes": True, "false": False, "no": False}.get(done.strip().lower())
            if not isinstance(done, bool):
                return None
            conf = obj.get("confidence", 0.5)
            try:
                conf = min(1.0, max(0.0, float(conf)))
            except Exception:
                return None
            return {"done": done, "confidence": conf, "why": str(obj.get("why", ""))[:200]}
    return None


def call_judge(initial_jpg, final_jpg, instruction, model, api_key, base_url, max_tokens=200, deadline_s=60, prompt_text=None, extra_images=()):
    """v1: two images (initial, final) with the default judge prompt. v3: prompt_text + [initial] + final + extra_images."""
    url = base_url.rstrip("/") + "/chat/completions"
    part = lambda b: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode(), "detail": "high"}}
    images = [initial_jpg, final_jpg] + list(extra_images)
    body = {"model": model, "stream": False, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": [{"type": "text", "text": prompt_text or judge_prompt(instruction)}] + [part(b) for b in images]}]}
    if "deepseek" in model.lower():
        body["thinking"] = {"type": "disabled"}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json", "authorization": "Bearer " + api_key})
    t0 = time.time(); last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=deadline_s) as r:
                payload = json.load(r)
            text = payload["choices"][0]["message"]["content"]
            parsed = parse_judge(text)
            return {**(parsed or {"done": None, "confidence": None, "why": "unparseable"}), "raw": (text or "")[:300], "latency_s": round(time.time() - t0, 2),
                    "usage": payload.get("usage"), "model": model, "error": None if parsed else "invalid_judge_output"}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:200]}"
        time.sleep(3 * (attempt + 1))
    return {"done": None, "confidence": None, "why": "", "raw": "", "latency_s": round(time.time() - t0, 2), "model": model, "error": last}


def agent_credentials(environ=None):
    """The agent's provider: AGENT_API_KEY / AGENT_BASE_URL when AGENT_API_KEY is set, else REA_API_KEY / REA_BASE_URL."""
    environ = os.environ if environ is None else environ
    if environ.get("AGENT_API_KEY"):
        base = environ.get("AGENT_BASE_URL")
        if not base:
            raise ValueError("AGENT_BASE_URL is required with AGENT_API_KEY")
        return environ["AGENT_API_KEY"], base
    return credentials()


def agent_program_label(args):
    style = getattr(args, "agent_style", "sdk")
    return {"rea-online": "rea-online:"}.get(style, "agents-sdk:") + args.agent_model


def credentials():
    """The judge's and agents' OpenAI-compatible provider: REA_API_KEY and REA_BASE_URL."""
    return os.environ.get("REA_API_KEY"), os.environ.get("REA_BASE_URL")


# ---- ASPIRE env ----
def head_frame_jpeg(low):
    """Head-camera RGB via the perception path (a PathTracing frame when REA_BEHAVIOR_PT_OBS=1, for GPUs without RT cores), as JPEG bytes."""
    import numpy as np
    from PIL import Image
    from aspire.sim.cap.integrations.r1pro.utils import obs_get_rgb_depth
    obs = low.get_perception_observation() if hasattr(low, "get_perception_observation") else low.get_observation()
    rgb, _ = obs_get_rgb_depth(obs)
    rgb = rgb.cpu().numpy()[..., :3] if hasattr(rgb, "cpu") else np.asarray(rgb)[..., :3]
    buf = io.BytesIO(); Image.fromarray(rgb.astype(np.uint8)).save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _np(x):
    import numpy as np
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=float)


def front_views_jpeg(low, dist=1.4, height=1.3, angle=35.0, close_dist=0.55, captures=2):
    """Judge views from the task's external VisionSensor moved in front of the robot: front_high (above head height, sees a raised
    arm and what it holds) and two torso-height overviews (front-left / front-right). Each pose is rendered `captures` times and
    the last frame kept (the first render after a camera move can return the previous frame); the camera pose is restored.
    Returns an ordered dict tag -> JPEG bytes. Uses RAW_FNS (not the budget-wrapped API)."""
    import math, numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation as R
    og_env = low.env; cam = og_env.external_sensors["external_camera"]
    import torch as th
    cpos0, cquat0 = cam.get_position_orientation()
    rp = RAW_FNS["get_robot_position"]()
    rpos = _np(rp[0]) if isinstance(rp, (tuple, list)) else _np(rp)
    if isinstance(rp, (tuple, list)) and len(rp) > 2: yaw = float(rp[2])
    else:
        _, rq = og_env.robots[0].get_position_orientation(); yaw = R.from_quat(_np(rq)).as_euler("xyz")[2]
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0]); left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    def lookat(cp, tgt):
        f = np.asarray(tgt, float) - np.asarray(cp, float); f /= max(np.linalg.norm(f), 1e-6)
        up = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.98 else np.array([0.0, 1.0, 0.0])
        r = np.cross(f, up); r /= max(np.linalg.norm(r), 1e-6); u = np.cross(r, f)
        return R.from_matrix(np.stack([r, u, -f], axis=1)).as_quat()
    poses = {}
    # front_high: in front of the robot above head height, looking at the head/raised-arm region (ASPIRE's lift_arm raises the
    # holding arm straight up, so a held object ends ~1.8-2.0 m high); front_left/right: torso-height overviews from +-angle.
    poses["front_high"] = (rpos + fwd * dist + np.array([0, 0, height + 0.5]), rpos + np.array([0, 0, 1.45]))
    for sign, tag in ((+1, "front_left"), (-1, "front_right")):
        d = np.array([math.cos(yaw + math.radians(sign * angle)), math.sin(yaw + math.radians(sign * angle)), 0.0])
        poses[tag] = (rpos + d * dist * 0.95 + np.array([0, 0, height]), rpos + np.array([0, 0, 1.1]))
    out = {}
    try:
        for tag, (cp, tgt) in poses.items():
            cam.set_position_orientation(position=th.tensor(cp, dtype=th.float32), orientation=th.tensor(lookat(cp, tgt), dtype=th.float32))
            arr = None
            for _ in range(captures):
                obs = low.get_perception_observation(); arr = _np(obs["external"]["external_camera"]["rgb"])[..., :3].astype(np.uint8)
            buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="JPEG", quality=92); out[tag] = buf.getvalue()
    finally:
        cam.set_position_orientation(position=cpos0, orientation=cquat0)
        try: low.get_perception_observation()  # flush so the next program-side perception frame is not the moved view
        except Exception: pass
    return out


def parse_exception(stderr):
    lines = [l for l in (stderr or "").strip().splitlines() if l.strip()]
    for l in reversed(lines):
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Exit|Warning)?):\s*(.*)$", l)
        if m and ("Error" in m.group(1) or "Exception" in m.group(1) or "Interrupt" in m.group(1)):
            return {"type": m.group(1), "message": m.group(2)[:300]}
    return None


class WallBudgetExceeded(BaseException):
    """Raised inside API calls once the episode's program budget is spent (BaseException: survives `except Exception`)."""


BUDGET = {"deadline": None, "exceeded": False, "calls": 0, "trace": []}
RAW_FNS = {}  # unwrapped API functions for runner-side use (camera placement); never counted as program API calls
VIDEO = {"stride": 0, "size": 384, "fps": 20, "cams": [], "render": "rt", "dir": None, "writers": {}, "frames": 0, "steps": 0, "errors": 0, "overhead_s": 0.0,
         "low": None, "start_step": None}


def record_video_step():
    """Every VIDEO['stride']-th simulator step: append one frame per camera to a streaming mp4 (imageio -> ffmpeg pipe, no RAM
    growth). rt = the frames the simulator already rendered this step (external VisionSensor + head camera, RayTracedLighting,
    speckled on GPUs without RT cores); pt = one PathTracing perception render per frame (clean, ~1 s). Capture time is excluded from the program
    budget (the deadline is shifted) and reported separately."""
    v = VIDEO
    if not v["stride"] or v["low"] is None: return
    v["steps"] += 1
    if v["steps"] % v["stride"]: return
    t0 = time.time()
    try:
        import numpy as np
        from PIL import Image
        low = v["low"]; frames = {}
        if v["render"] == "pt":
            saved = {k: os.environ.get(k) for k in ("REA_BEHAVIOR_PT_SPP", "REA_BEHAVIOR_PT_RENDERS")}
            if v.get("pt_spp"): os.environ["REA_BEHAVIOR_PT_SPP"] = str(v["pt_spp"])
            if v.get("pt_renders"): os.environ["REA_BEHAVIOR_PT_RENDERS"] = str(v["pt_renders"])
            cam = cpose = None
            try:
                if "follow" in v["cams"]:
                    # third-person chase camera: the task's external VisionSensor moved behind-left of the robot for this render only
                    import math, torch as th
                    from scipy.spatial.transform import Rotation as R
                    og_env = low.env; cam = og_env.external_sensors["external_camera"]; cpose = cam.get_position_orientation()
                    rpos_t, rq = og_env.robots[0].get_position_orientation(); rpos = _np(rpos_t); yaw = R.from_quat(_np(rq)).as_euler("xyz")[2]
                    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0]); left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
                    cp = rpos - fwd * v.get("follow_dist", 1.6) + left * v.get("follow_side", 0.6) + np.array([0.0, 0.0, v.get("follow_height", 1.5)])
                    tgt = rpos + fwd * 0.8 + np.array([0.0, 0.0, 0.8])
                    f = tgt - cp; f /= max(np.linalg.norm(f), 1e-6); up = np.array([0.0, 0.0, 1.0]); r = np.cross(f, up); r /= max(np.linalg.norm(r), 1e-6); u = np.cross(r, f)
                    q = R.from_matrix(np.stack([r, u, -f], axis=1)).as_quat()
                    cam.set_position_orientation(position=th.tensor(cp, dtype=th.float32), orientation=th.tensor(q, dtype=th.float32))
                obs = low.get_perception_observation()
            finally:
                for k, val in saved.items():
                    if val is None: os.environ.pop(k, None)
                    else: os.environ[k] = val
                if cam is not None and cpose is not None: cam.set_position_orientation(position=cpose[0], orientation=cpose[1])
            ext = _np(obs["external"]["external_camera"]["rgb"])[..., :3]
            if "follow" in v["cams"]: frames["follow"] = ext
            elif "external" in v["cams"]: frames["external"] = ext
            if "head" in v["cams"]:
                rk = [k for k in obs if "robot" in k][0]; frames["head"] = _np(obs[rk][f"{rk}:zed_link:Camera:0"]["rgb"])[..., :3]
        else:
            if "external" in v["cams"]: frames["external"] = _np(low.env.external_sensors["external_camera"].get_obs()[0]["rgb"][:, :, :3])
            if "head" in v["cams"]:
                obs = getattr(low, "cur_observation", None)
                if obs:
                    rk = [k for k in obs if "robot" in k][0]; frames["head"] = _np(obs[rk][f"{rk}:zed_link:Camera:0"]["rgb"][:, :, :3])
        for cam, arr in frames.items():
            arr = np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
            if v["size"] and arr.shape[0] != v["size"]: arr = np.asarray(Image.fromarray(arr).resize((v["size"], v["size"]), Image.BILINEAR))
            w = v["writers"].get(cam)
            if w is None:
                import imageio
                w = imageio.get_writer(str(Path(v["dir"]) / f"video_{cam}.mp4"), fps=v["fps"], format="FFMPEG", codec="libx264", macro_block_size=None,
                                       ffmpeg_params=["-crf", "23", "-pix_fmt", "yuv420p"])
                v["writers"][cam] = w
            w.append_data(arr)
        v["frames"] += 1
    except Exception as e:
        v["errors"] += 1
        if v["errors"] <= 3: print(json.dumps({"stage": "video_error", "error": str(e)[:200]}), flush=True)
    finally:
        dt = time.time() - t0; v["overhead_s"] += dt
        if BUDGET["deadline"] is not None: BUDGET["deadline"] += dt  # recording is not program time


def close_video():
    v = VIDEO
    for cam, w in list(v["writers"].items()):
        try: w.close()
        except Exception: pass
    v["writers"].clear()
    if not v["stride"]: return None
    return {"stride": v["stride"], "fps": v["fps"], "size": v["size"], "render": v["render"], "pt_spp": v.get("pt_spp"), "pt_renders": v.get("pt_renders"), "cams": v["cams"], "frames": v["frames"], "sim_steps_seen": v["steps"],
            "start_step": v["start_step"], "overhead_s": round(v["overhead_s"], 1), "errors": v["errors"], "files": [f"video_{c}.mp4" for c in v["cams"]]}


def wrap_api(fn, name):
    def wrapped(*a, **k):
        BUDGET["calls"] += 1
        if BUDGET["deadline"] is not None and time.time() > BUDGET["deadline"]:
            BUDGET["exceeded"] = True
            raise WallBudgetExceeded(f"program budget exhausted before {name}()")
        t0 = time.time()
        try:
            return fn(*a, **k)
        finally:
            if len(BUDGET["trace"]) < 2000:
                BUDGET["trace"].append((name, round(time.time() - t0, 2)))
    wrapped.__name__ = name; wrapped.__doc__ = fn.__doc__
    return wrapped


def install_budget_wrappers(env):
    """Wrap the 26 exposed API functions (per-call check) AND the low-level env.step (per-simulator-step check).
    Compound primitives such as find_object_base_rotate run for minutes inside one API call, so the step-level check is
    what actually bounds the overshoot to one simulator step."""
    from aspire.sim.cap.envs.launch_b1k import _find_apis
    n = 0
    for api in _find_apis(env).values():
        orig = api.functions
        RAW_FNS.update(orig())
        def functions(orig=orig):
            return {name: wrap_api(fn, name) for name, fn in orig().items()}
        api.functions = functions; n += len(orig())
    low = env.low_level_env; orig_step = low.step
    def step(action, _orig=orig_step):
        if BUDGET["deadline"] is not None and time.time() > BUDGET["deadline"]:
            BUDGET["exceeded"] = True
            raise WallBudgetExceeded("program budget exhausted (simulator step)")
        res = _orig(action); record_video_step(); return res
    low.step = step
    return n


class LastResortWatchdog:
    """Safety net for programs that spin without simulator steps or API calls. Interrupting the main thread shuts Kit
    down without artifacts, so instead it writes a failure result.json for the episode and exits the process."""
    def __init__(self, seconds, ep_dir, base_result, runs):
        self.seconds = max(1, int(seconds)); self.cancel = threading.Event(); self.fired = threading.Event()
        self.ep_dir, self.base_result, self.runs = ep_dir, base_result, runs
        self.t = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        if not self.cancel.wait(self.seconds):
            self.fired.set()
            # agents-sdk: only runner / ASPIRE code runs on the blocked thread, so a hang is infrastructure (protocol: died before
            # scoring -> success null, rerun); the ladder modes keep their historical failure record.
            infra = self.base_result.get("mode") == "agents-sdk"
            result = {**self.base_result, "success": None if infra else False, "reward": None if infra else 0.0,
                      "stop_reason": "last_resort_watchdog", "runs": list(self.runs), "recoveries": max(0, len(self.runs) - 1), "unscored": True}
            try:
                (self.ep_dir / "result.json").write_text(json.dumps(result, indent=1, default=str))
                print(json.dumps({"stage": "LAST_RESORT_WATCHDOG", "note": "budget+grace exceeded without a step; episode "
                                  + ("left unscored (rerun)" if infra else "recorded as failure")}), flush=True)
                close_video()
            finally:
                os._exit(4)
    def start(self): self.t.start(); return self
    def stop(self): self.cancel.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(CONFIGS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--programs", default="", help="comma-separated program files (ladder order); not used by --mode agents-sdk")
    ap.add_argument("--mode", choices=["single", "routed-exception", "routed-judge", "routed-inhand", "agents-sdk"], default="single",
                    help="agents-sdk: OpenAI Agents SDK tool-calling agent over the R1Pro API (agents_sdk_arm.py); --programs is ignored")
    ap.add_argument("--agent-model", default=os.environ.get("REA_MODEL"), help="agent model (default REA_MODEL)"); ap.add_argument("--agent-max-turns", type=int, default=60); ap.add_argument("--agent-llm-timeout", type=float, default=180.0)
    ap.add_argument("--agent-images", type=int, default=1, help="1: attach the initial head frame to the first user message");
    ap.add_argument("--agent-style", choices=["sdk", "rea-online"], default="sdk", help="sdk: one Agents SDK run over the R1Pro tools; rea-online: REA's episode runtime (eval/behavior/online)")
    ap.add_argument("--agent-api", choices=["chat_completions", "responses"], default="chat_completions", help="sdk style: model API")
    ap.add_argument("--agent-effort", default="medium", help="reasoning effort (responses API and rea-online)"); ap.add_argument("--agent-max-tokens", type=int, default=2048, help="max output tokens per request (responses API and rea-online)")
    ap.add_argument("--agent-node", default="node", help="rea-online: node binary"); ap.add_argument("--agent-max-decisions", type=int, default=40, help="rea-online: decisions per episode")
    ap.add_argument("--agent-python", dest="sdk_python", default=None, help="interpreter for the agent process and the rea-online worker (default: this one)")
    ap.add_argument("--online-config", default=None, help="rea-online: behavioural config (default eval/behavior/online/config.example.json)")
    ap.add_argument("--task-prompt-dir", default=os.environ.get("REA_BEHAVIOR_PROMPTS") or str(Path(__file__).resolve().parents[1] / "prompts"), help="directory with <task>.txt task prompts for the agent instructions (see eval/behavior/prompts/README.md)")
    ap.add_argument("--max-recoveries", type=int, default=2)
    ap.add_argument("--program-budget", "--wall-budget", dest="program_budget", type=float, default=1500.0, help="seconds of program execution shared by all runs of the episode (clock starts at the first program statement)")
    ap.add_argument("--judge-model", default=os.environ.get("REA_JUDGE_MODEL") or os.environ.get("REA_MODEL"), help="judge model (default REA_JUDGE_MODEL, then REA_MODEL)"); ap.add_argument("--judge-threshold", type=float, default=0.6)
    ap.add_argument("--judge-view", choices=["head", "front"], default="head"); ap.add_argument("--judge-prompt", choices=["v1", "v3"], default="v1")
    ap.add_argument("--recovery-guard", choices=["none", "inhand"], default="none")
    ap.add_argument("--judge-log", action="store_true", help="single / routed-exception: advisory judge call after every run (calibration only)")
    ap.add_argument("--view-dist", type=float, default=1.4); ap.add_argument("--view-height", type=float, default=1.3); ap.add_argument("--view-angle", type=float, default=35.0); ap.add_argument("--close-dist", type=float, default=0.55)
    ap.add_argument("--record-video", type=int, default=0, metavar="STRIDE", help="record an mp4 per camera, one frame every STRIDE simulator steps (0 = off)")
    # (chase-camera defaults: the external VisionSensor is wide-angle; 2.4 m back made the robot tiny, 1.6 m / 1.5 m high frames it)
    ap.add_argument("--video-cams", default="external,head", help="comma list of external (fixed room camera), head (robot head camera), follow (third-person chase camera; pt only; replaces external in the same render)");
    ap.add_argument("--follow-dist", type=float, default=1.6); ap.add_argument("--follow-side", type=float, default=0.6); ap.add_argument("--follow-height", type=float, default=1.5); ap.add_argument("--video-size", type=int, default=384); ap.add_argument("--video-fps", type=int, default=20)
    ap.add_argument("--video-render", choices=["rt", "pt"], default="rt", help="rt: frames already rendered by the step (fast; speckled on GPUs without RT cores); pt: PathTracing render per frame (clean, ~1 s each)")
    ap.add_argument("--video-pt-spp", type=int, default=0, help="pt: PathTracing samples per pixel for video frames (0 = the perception default REA_BEHAVIOR_PT_SPP)")
    ap.add_argument("--video-pt-renders", type=int, default=0, help="pt: renders per video frame (0 = the perception default REA_BEHAVIOR_PT_RENDERS; the first render after a mode switch can be stale)")
    ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sim-root", default=os.environ.get("ASPIRE_SIM_ROOT"), required=not os.environ.get("ASPIRE_SIM_ROOT"), help="ASPIRE's aspire/sim directory (default ASPIRE_SIM_ROOT)")
    args = ap.parse_args()

    programs = [Path(p) for p in (args.programs or "").split(",") if p]
    if args.mode == "single":
        programs = programs[:1]
    ladder = programs[: 1 + args.max_recoveries] if args.mode != "agents-sdk" else []
    ep_dir = Path(args.out) / "episodes" / "behavior" / args.task / f"seed_{args.seed}"
    (ep_dir / "runs").mkdir(parents=True, exist_ok=True)
    os.chdir(args.sim_root); sys.path.insert(0, str(Path(args.sim_root).parents[1]))
    t_start = time.time()
    def log(**kw): print(json.dumps({"t": round(time.time() - t_start, 1), **kw}, default=str), flush=True)
    log(stage="start", task=args.task, seed=args.seed, mode=args.mode, ladder=[str(p) for p in ladder], arm=args.arm)

    from aspire.sim.cap.envs.launch_b1k import B1KLaunchArgs, _instantiate_env, _reset_env, _parse_code_blocks
    from aspire.sim.cap.utils.launch_utils import _load_config
    la = B1KLaunchArgs(config_path=CONFIGS[args.task], trial_ids=[args.seed], record_video=False, output_dir=str(ep_dir / "aspire_out"))
    if hasattr(la, "model"): la.model = "arms3"
    env_factory, config, _servers = _load_config(la)
    env = _instantiate_env(env_factory); low = env.low_level_env
    n_wrapped = install_budget_wrappers(env)
    log(stage="env_built", api_functions_wrapped=n_wrapped)
    _reset_env(env, args.seed); log(stage="reset_done")
    initial = head_frame_jpeg(low); (ep_dir / "initial.jpg").write_bytes(initial)
    if args.record_video > 0:
        VIDEO.update(stride=args.record_video, size=args.video_size, fps=args.video_fps, cams=[c for c in args.video_cams.split(",") if c], render=args.video_render,
                     dir=str(ep_dir), low=low, start_step=getattr(low, "_step_count", None), pt_spp=args.video_pt_spp, pt_renders=args.video_pt_renders, follow_dist=args.follow_dist, follow_side=args.follow_side, follow_height=args.follow_height)
        log(stage="video_on", **{k: VIDEO[k] for k in ("stride", "size", "fps", "cams", "render", "start_step")})
    placeholder = {"suite": "behavior", "task": args.task, "seed": args.seed, "arm": args.arm, "mode": args.mode, "success": None, "reward": None,
                   "stop_reason": "process_died_before_scoring", "runs": [], "recoveries": 0, "wall_s": None, "program_budget_s": args.program_budget}
    (ep_dir / "result.json").write_text(json.dumps(placeholder, indent=1))
    key, base = credentials() if args.mode != "agents-sdk" else (None, None)  # the judge's provider: unused by agents-sdk
    instruction = INSTRUCTIONS[args.task]
    runs = []; final = None; stop_reason = None; t_prog = None
    def inhand_now():
        """Per-arm 'object in hand' read straight from the robot's assisted-grasp state, WITHOUT touching the controller's active
        arm or stepping the simulator. (check_object_in_hand(arm) calls controller.overwrite_arm(arm) and settles the robot; the
        sealed predicate task_completed() reads _get_obj_in_hand() of the controller's CURRENT arm, so a runner-side probe that
        switched arms would corrupt the score.)"""
        try:
            robot = low.controller.robot; ag = getattr(robot, "_ag_obj_in_hand", {})
            out = {str(i): (ag.get(name) is not None) for i, name in enumerate(robot.arm_names)}
            out["controller_arm"] = str(low.controller.arm)
            return out
        except Exception as e:
            return {"error": str(e)[:200]}

    def judge_now(i):
        """Render the judge's frames for run i (head always; front views when requested) and call the judge. Returns the judge record."""
        nonlocal final
        saved_deadline = BUDGET["deadline"]; BUDGET["deadline"] = None  # runner-side rendering is not program time
        views = {}
        try:
            if args.judge_view == "front":
                try:
                    views = front_views_jpeg(low, args.view_dist, args.view_height, args.view_angle, args.close_dist)
                    for tag, b in views.items(): (ep_dir / f"final_{i}_{tag}.jpg").write_bytes(b)
                except Exception as e:
                    log(stage="front_view_error", error=str(e)[:300]); views = {}
        finally:
            BUDGET["deadline"] = saved_deadline
        if args.judge_prompt == "v3" and views:
            tags = list(views); prompt_text = judge_prompt_v3(args.task, instruction, tags)
            j = call_judge(initial, views[tags[0]], instruction, args.judge_model, key, base, prompt_text=prompt_text, extra_images=[views[t] for t in tags[1:]])
            j["images"] = ["initial_head"] + tags; j["prompt"] = "v3"
        else:
            j = call_judge(initial, final or initial, instruction, args.judge_model, key, base)
            j["images"] = ["initial_head", "final_head"]; j["prompt"] = "v1" if args.judge_prompt == "v1" else "v1_fallback"
        j["verdict"] = "done" if (j.get("done") is True and (j.get("confidence") or 0) >= args.judge_threshold) else "not_done"
        j["inhand"] = inhand_now()  # calibration only: the proprioceptive signal next to the visual verdict
        return j
    budget_left = lambda: args.program_budget - (time.time() - t_prog) if t_prog else args.program_budget
    guard = LastResortWatchdog(args.program_budget + 900, ep_dir, placeholder, runs).start()
    if args.mode == "agents-sdk":
        import agents_sdk_arm
        from aspire.sim.cap.envs.launch_b1k import _find_apis
        fns = {}
        for api in _find_apis(env).values(): fns.update(api.functions())  # budget-wrapped API functions (install_budget_wrappers)
        prompt_file = Path(args.task_prompt_dir) / f"{args.task}.txt"
        task_prompt = prompt_file.read_text() if prompt_file.exists() else f"You are controlling a R1Pro robot. Goal: {instruction}"
        task_prompt = "\n".join(l for l in task_prompt.splitlines() if l.strip() and not l.strip().startswith("prompt: |"))
        agent_key, agent_base = agent_credentials()
        label = agent_program_label(args)
        agent_meta = {"agent_style": args.agent_style, "agent_api": "responses" if args.agent_style == "rea-online" else args.agent_api,
                      "agent_effort": args.agent_effort, "agent_max_tokens": args.agent_max_tokens,
                      "agent_max_decisions": args.agent_max_decisions if args.agent_style == "rea-online" else None,
                      "agent_base_host": re.sub(r"^https?://([^/]+).*$", r"\1", agent_base or ""),
                      "online_file_sha256": agents_sdk_arm.online_file_hashes() if args.agent_style == "rea-online" else None,
                      "code_sha256": agents_sdk_arm.code_file_hashes([prompt_file, args.online_config,
                                                                      os.environ.get("REA_BEHAVIOR_WORKER") if args.agent_style == "rea-online" else None]),
                      "repo_commit": agents_sdk_arm.repo_commit()}

        def observe(path):
            """rea-online: the current head-camera JPEG at path (rendered inside the program budget)."""
            Path(path).write_bytes(head_frame_jpeg(low))
        t_prog = time.time(); BUDGET["deadline"] = t_prog + args.program_budget
        log(stage="run_start", index=0, program=label, budget_left=round(budget_left()), **{k: v for k, v in agent_meta.items() if k not in ("online_file_sha256", "code_sha256")})
        agent_result = agents_sdk_arm.run_episode(fns, instruction, task_prompt, initial, args, args.program_budget, agent_key, agent_base, WallBudgetExceeded, ep_dir, log, observe=observe)
        BUDGET["deadline"] = None
        agent_result.update(agent_meta)
        (ep_dir / "agent_trace.json").write_text(json.dumps(agent_result, indent=1, default=str))
        rec = {"index": 0, "program": label, "blocks": None, "sandbox_rc": 0 if agent_result.get("error") is None else 1, "truncated_reason": "program_budget" if agent_result["stop_reason"] == "program_budget" else None,
               "wall_s": agent_result["wall_s"], "sim_steps_before": 0, "sim_steps_after": getattr(low, "_step_count", None), "api_calls": BUDGET["calls"], "exception": None if agent_result.get("error") is None else {"type": "AgentError", "message": agent_result["error"][:300]},
               "agent": {k: agent_result.get(k) for k in ("stop_reason", "turns", "usage", "tool_calls_count", "tool_time_s", "llm_time_s", "final_output", "model", "images",
                                                          "provider_error", "http_status", "truncated_retries", "past_deadline_s",
                                                          "finishes", "observe_time_s", *agent_meta)}, "decision": "stop",
               "reply": {"exception": None, "stdout_tail": (agent_result.get("final_output") or "")[-1500:], "stderr_tail": (agent_result.get("error") or "")[-2000:], "truncated_reason": None, "sandbox_rc": 0}}
        try:
            final = head_frame_jpeg(low); (ep_dir / "final_0.jpg").write_bytes(final)
        except Exception as e:
            log(stage="final_frame_error", error=str(e)[:200])
        (ep_dir / "runs" / "0.json").write_text(json.dumps(rec, indent=1, default=str)); runs.append(rec)
        stop_reason = agent_result["stop_reason"]
        log(stage="run_end", index=0, rc=rec["sandbox_rc"], wall_s=rec["wall_s"], api_calls=rec["api_calls"], stop=stop_reason, turns=agent_result.get("turns"), tool_calls=agent_result.get("tool_calls_count"))
    for i, prog in enumerate(ladder):
        if budget_left() <= 30:
            stop_reason = "program_budget_exhausted_before_run"; break
        run_path = prog
        if i > 0 and args.recovery_guard == "inhand":
            run_path = ep_dir / "runs" / f"recovery_{i}.py"; run_path.write_text(GUARD_PRELUDE + Path(prog).read_text())
        blocks = _parse_code_blocks(str(run_path))
        rec = {"index": i, "program": str(prog), "blocks": len(blocks), "sandbox_rc": 0, "stdout": "", "stderr": "", "truncated_reason": None, "wall_s": None,
               "sim_steps_before": getattr(low, "_step_count", None), "api_calls_before": BUDGET["calls"], "guard": args.recovery_guard if i > 0 else "none",
               "executed_file": str(run_path), "program_exit": None}
        if t_prog is None:
            t_prog = time.time(); BUDGET["deadline"] = t_prog + args.program_budget
        t0 = time.time(); log(stage="run_start", index=i, program=str(prog), budget_left=round(budget_left()))
        trace_start = len(BUDGET["trace"])
        try:
            for b, code in enumerate(blocks):
                _, reward, terminated, truncated, info = env.step(code)
                block_rc = int(info.get("sandbox_rc", 0)); block_err = info.get("stderr", "")
                rec["stdout"] += info.get("stdout", ""); rec["stderr"] += block_err
                if BUDGET["exceeded"]:
                    rec["sandbox_rc"] = max(rec["sandbox_rc"], block_rc); rec["truncated_reason"] = "program_budget"; break
                if block_rc != 0 and clean_exit_line(block_err):  # the program stopped itself (raise SystemExit(0) / sys.exit()): a clean end
                    rec["program_exit"] = clean_exit_line(block_err); break
                rec["sandbox_rc"] = max(rec["sandbox_rc"], block_rc)
                if truncated:
                    rec["truncated_reason"] = "sim_step_budget"; break
        except KeyboardInterrupt:
            rec["truncated_reason"] = "last_resort_watchdog"; rec["sandbox_rc"] = max(rec["sandbox_rc"], 1); rec["stderr"] += "\nKeyboardInterrupt: last-resort watchdog"
        except Exception:
            rec["truncated_reason"] = "runner_error"; rec["sandbox_rc"] = max(rec["sandbox_rc"], 1); rec["stderr"] += "\n" + traceback.format_exc()[-1500:]
        rec["wall_s"] = round(time.time() - t0, 1); rec["sim_steps_after"] = getattr(low, "_step_count", None); rec["api_calls"] = BUDGET["calls"] - rec["api_calls_before"]
        rec["api_trace"] = BUDGET["trace"][trace_start:trace_start + 400]
        rec["exception"] = parse_exception(rec["stderr"])
        saved_deadline = BUDGET["deadline"]; BUDGET["deadline"] = None  # runner-side rendering is not program time
        try:
            final = head_frame_jpeg(low); (ep_dir / f"final_{i}.jpg").write_bytes(final)
        except Exception as e:
            log(stage="final_frame_error", error=str(e)[:200])
        finally:
            BUDGET["deadline"] = saved_deadline
        log(stage="run_end", index=i, rc=rec["sandbox_rc"], wall_s=rec["wall_s"], api_calls=rec["api_calls"], truncated=rec["truncated_reason"], exception=rec["exception"])
        # ---- recovery decision (never from the predicate) ----
        decision = "stop"
        if i < len(ladder) - 1 and rec["truncated_reason"] not in ("program_budget", "last_resort_watchdog") and budget_left() > 30:
            if args.mode == "routed-exception":
                decision = "recover" if rec["sandbox_rc"] != 0 else "stop"
            elif args.mode == "routed-judge":
                if not key:
                    rec["judge"] = {"error": "no REA_API_KEY"}; decision = "stop"
                else:
                    j = judge_now(i); rec["judge"] = j; decision = "stop" if j["verdict"] == "done" else "recover"
                    log(stage="judge", **{k: j.get(k) for k in ("done", "confidence", "why", "verdict", "latency_s", "error", "images", "inhand")})
            elif args.mode == "routed-inhand":
                ih = inhand_now(); rec["inhand"] = ih; rec["inhand_held"] = any(v is True for v in ih.values())
                decision = "stop" if rec["inhand_held"] else "recover"; log(stage="inhand", inhand=ih, decision=decision)
        elif args.mode == "routed-inhand":
            ih = inhand_now(); rec["inhand"] = ih; rec["inhand_held"] = any(v is True for v in ih.values()); log(stage="inhand_final", inhand=ih)
        elif args.mode == "routed-judge" and key:  # last run or no budget: still record the judge's opinion for calibration
            j = judge_now(i); rec["judge"] = j; log(stage="judge_final", **{k: j.get(k) for k in ("done", "confidence", "why", "verdict", "latency_s", "error", "images")})
        if args.judge_log and args.mode != "routed-judge" and key:  # (routed-inhand included: its judge verdicts are advisory)  # advisory only: calibration data for the judge on this arm's episodes
            j = judge_now(i); j["advisory"] = True; rec["judge"] = j
            log(stage="judge_advisory", **{k: j.get(k) for k in ("done", "confidence", "why", "verdict", "latency_s", "error", "images")})
        rec["decision"] = decision
        rec["reply"] = {"exception": rec["exception"], "stdout_tail": rec["stdout"][-1500:], "stderr_tail": rec["stderr"][-2000:], "truncated_reason": rec["truncated_reason"], "sandbox_rc": rec["sandbox_rc"]}
        slim = {k: v for k, v in rec.items() if k not in ("stdout", "stderr")}
        (ep_dir / "runs" / f"{i}.json").write_text(json.dumps(slim, indent=1, default=str))
        runs.append(slim)
        if decision == "stop":
            stop_reason = stop_reason or ("judge_done" if args.mode == "routed-judge" and rec.get("judge", {}).get("verdict") == "done" else
                                          "inhand_done" if args.mode == "routed-inhand" and rec.get("inhand_held") else
                                          "program_budget" if rec["truncated_reason"] in ("program_budget", "last_resort_watchdog") else
                                          "ladder_end" if i == len(ladder) - 1 else "no_recovery_needed"); break
    guard.stop()
    video_info = close_video()
    if video_info: log(stage="video_done", **video_info)
    if final is not None: (ep_dir / "final.jpg").write_bytes(final)
    # ---- sealed score: the benchmark predicate, read once (budget check off: scoring is not program time) ----
    BUDGET["deadline"] = None
    score_ctx = inhand_now()  # diagnostic: per-arm in-hand state and the controller's active arm at scoring time
    try:
        success = bool(low.task_completed()); reward = float(env.compute_reward())
    except Exception as e:
        success, reward = None, None; log(stage="score_error", error=str(e)[:200])
    result = {"suite": "behavior", "task": args.task, "seed": args.seed, "arm": args.arm, "mode": args.mode, "ladder": [str(p) for p in ladder],
              "success": success, "reward": reward, "steps": getattr(low, "_step_count", None), "runs_executed": len(runs), "recoveries": max(0, len(runs) - 1),
              "stop_reason": stop_reason, "wall_s": round(time.time() - t_start, 1), "program_wall_s": round(time.time() - t_prog - VIDEO["overhead_s"], 1) if t_prog else 0.0, "video": video_info, "program_budget_s": args.program_budget, "api_calls": BUDGET["calls"],
              "judge_model": args.judge_model if (args.mode == "routed-judge" or args.judge_log) else None, "judge_threshold": args.judge_threshold if args.mode == "routed-judge" else None,
              "judge_view": args.judge_view, "judge_prompt": args.judge_prompt, "recovery_guard": args.recovery_guard, "judge_log": bool(args.judge_log), "score_context": score_ctx,
              "pt_perception": bool(os.environ.get("REA_BEHAVIOR_PT_OBS")), "runs": runs}
    if args.mode == "agents-sdk":
        result.update({**agent_meta, "program": label})
        result.update(agents_sdk_arm.outcome_fields(stop_reason, success, reward))  # provider failure -> unscored, rerun
    (ep_dir / "result.json").write_text(json.dumps(result, indent=1, default=str))
    log(stage="DONE", success=result["success"], reward=result["reward"], recoveries=result["recoveries"], wall_s=result["wall_s"], stop=result["stop_reason"])
    sys.stdout.flush(); os._exit(0)  # Isaac teardown segfaults are known; artifacts are already written


if __name__ == "__main__":
    main()
