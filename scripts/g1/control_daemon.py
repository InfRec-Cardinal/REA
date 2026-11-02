#!/usr/bin/env python3
"""Persistent G1 locomotion daemon. Reuses a robot-side helper module's SDK loading, preflight checks and
motion envelope (G1_CONTROL_SCRIPT, see docs/robots/unitree-g1.md), but keeps DDS + LocoClient + state
subscribers alive so a command costs ~one RPC.
Protocol: one JSON per stdin line -> one JSON per stdout line.
  {"op":"status"}                                   -> readiness snapshot
  {"op":"move","vx":0.3,"vy":0,"yaw":0,"duration":1.0} -> preflight, SetVelocity, returns immediately;
                                                       a zero-velocity stop is sent automatically after duration
  {"op":"stop"}                                     -> zero velocity now
Never sends lowcmd; only the sport/loco SetVelocity RPC, exactly like g1_control.py."""
import importlib.util, json, os, sys, threading, time

CTRL = os.environ.get("G1_CONTROL_SCRIPT")
if not CTRL:
    sys.exit("control_daemon: set G1_CONTROL_SCRIPT to the robot-side g1_control.py helper (see docs/robots/unitree-g1.md)")
spec = importlib.util.spec_from_file_location("g1_control", CTRL)
g1 = importlib.util.module_from_spec(spec); sys.modules["g1_control"] = g1; spec.loader.exec_module(g1)

settings = g1.Settings(
    iface=os.environ.get("G1_IFACE", g1.DEFAULT_IFACE), sdk_path=g1.DEFAULT_SDK_PATH, sdk_deps_path=g1.DEFAULT_SDK_DEPS_PATH,
    timeout=3.0, min_soc=20, required_mode_machine=5, allowed_fsm_ids=tuple(g1.DEFAULT_ALLOWED_FSM_IDS), max_tilt_rad=0.35,
    max_abs_vx=0.5, max_abs_vy=0.3, max_abs_yaw=1.0, max_duration=3.0)

sdk, client = g1._initialize(settings)
_arm = None

def arm_client():
    """Lazy G1ArmActionClient on the same DDS domain; predefined arm actions only (no lowcmd)."""
    global _arm
    if _arm is None:
        for path in (g1.DEFAULT_SDK_DEPS_PATH, g1.DEFAULT_SDK_PATH):
            if path and path not in sys.path:
                sys.path.insert(0, path)
        from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient  # type: ignore
        _arm = G1ArmActionClient(); _arm.SetTimeout(10.0); _arm.Init()
    return _arm
low_sub = sdk.channel_subscriber(g1.LOWSTATE_TOPIC, sdk.low_state_type); low_sub.Init()
bms_sub = sdk.channel_subscriber(g1.BMS_TOPIC, sdk.bms_state_type); bms_sub.Init()
lock = threading.Lock()
_stop_timer = None

def snapshot():
    low = g1._read_sample(low_sub, settings.timeout, "low-state")
    bms = g1._read_sample(bms_sub, settings.timeout, "battery")
    fsm_code, fsm_id = g1._get_fsm_id(client)
    rpy = [float(v) for v in low.imu_state.rpy]
    reasons = []
    if fsm_code != 0: reasons.append(f"FSM query returned {fsm_code}")
    elif fsm_id not in settings.allowed_fsm_ids: reasons.append(f"FSM {fsm_id} is not in allowed set {list(settings.allowed_fsm_ids)}")
    if int(low.mode_machine) != settings.required_mode_machine: reasons.append(f"mode_machine is {int(low.mode_machine)}, expected {settings.required_mode_machine}")
    if int(bms.soc) < settings.min_soc: reasons.append(f"battery SOC is {int(bms.soc)}, minimum is {settings.min_soc}")
    if abs(rpy[0]) > settings.max_tilt_rad or abs(rpy[1]) > settings.max_tilt_rad: reasons.append(f"roll/pitch exceed {settings.max_tilt_rad:.3f} rad")
    return {"ready": not reasons, "reasons": reasons, "fsm_id": fsm_id, "mode_machine": int(low.mode_machine), "soc": int(bms.soc),
            "rpy": rpy, "gyro": [float(v) for v in low.imu_state.gyroscope]}

def send_stop():
    with lock:
        return int(client.SetVelocity(0.0, 0.0, 0.0, 1.0))

def handle(req):
    global _stop_timer
    op = req.get("op")
    t0 = time.time()
    if op == "status":
        s = snapshot(); s.update({"ok": True, "timing": {"status_s": round(time.time() - t0, 4)}}); return s
    if op == "stop":
        if _stop_timer: _stop_timer.cancel()
        rc = send_stop(); return {"ok": rc == 0, "stop_rc": rc, "timing": {"rpc_s": round(time.time() - t0, 4)}}
    if op == "arm":
        aid = int(req.get("action_id"))
        code = int(arm_client().ExecuteAction(aid))
        return {"ok": code == 0, "code": code, "action_id": aid, "timing": {"rpc_s": round(time.time() - t0, 4)}}
    if op == "arm_list":
        code, data = arm_client().GetActionList()
        return {"ok": int(code) == 0, "code": int(code), "actions": data}
    if op == "move":
        vx, vy, yaw, dur = float(req.get("vx", 0)), float(req.get("vy", 0)), float(req.get("yaw", 0)), float(req.get("duration", 0))
        g1._validate_motion(vx, vy, yaw, dur, settings)
        pre = snapshot(); t1 = time.time()
        if not pre["ready"]:
            return {"ok": False, "error": "preflight rejected motion: " + "; ".join(pre["reasons"]), "preflight": pre}
        if _stop_timer: _stop_timer.cancel()
        with lock:
            rc = int(client.SetVelocity(vx, vy, yaw, dur))
        t2 = time.time()
        _stop_timer = threading.Timer(dur + 0.05, send_stop); _stop_timer.daemon = True; _stop_timer.start()
        return {"ok": rc == 0, "move_rc": rc, "accepted_at": t2, "preflight": pre,
                "command": g1._command_payload("move", vx, vy, yaw, dur),
                "timing": {"preflight_s": round(t1 - t0, 4), "rpc_s": round(t2 - t1, 4), "total_s": round(t2 - t0, 4)}}
    return {"ok": False, "error": f"unknown op {op!r}"}

sys.stdout.write(json.dumps({"ok": True, "event": "ready", "status": snapshot()}) + "\n"); sys.stdout.flush()
try:
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        req = None
        try:
            req = json.loads(line); res = handle(req)
        except Exception as exc:
            res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        res["id"] = req.get("id") if isinstance(req, dict) else None
        sys.stdout.write(json.dumps(res) + "\n"); sys.stdout.flush()
finally:
    # The parent is gone (stdin EOF) or we are dying: never leave a pulse running behind us. Best
    # effort — the sport service's own watchdog is the last line, this is the second-to-last.
    try:
        if _stop_timer is not None: _stop_timer.cancel()
        send_stop()
    except Exception:
        pass
