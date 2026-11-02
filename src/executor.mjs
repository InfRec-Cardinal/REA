import { performance } from "node:perf_hooks";

import { depthGate, nextArmHeld, verdict } from "./gate.mjs";
import { G1_PROFILE } from "./profiles/g1.mjs";
import { createDisabledObstacleSource, createSnapshotObstacleSource } from "./sensors/obstacles.mjs";

function seconds(ms) {
  return Number((ms / 1000).toFixed(4));
}

/**
 * Persistent-daemon actuator. Talks to control_daemon.py, which owns DDS and the LocoClient and only
 * ever issues the sport-service SetVelocity / arm-action RPCs. It never touches lowcmd: a lowcmd
 * takeover on this robot means commanding all 17 joints correctly on the first packet or it falls.
 */
export function createDaemonActuator({ worker, config }) {
  return {
    kind: "daemon",
    async status() {
      return worker.request({ op: "status" }, 5000);
    },
    async stop() {
      return worker.request({ op: "stop" }, 5000);
    },
    async move(motion) {
      return worker.request(
        { op: "move", vx: motion.vx, vy: motion.vy, yaw: motion.yaw, duration: motion.duration },
        config.controlTimeoutMs,
      );
    },
    async arm(actionId) {
      return worker.request({ op: "arm", action_id: actionId }, config.armTimeoutMs);
    },
  };
}

/**
 * HTTP console actuator. Useful when the G1 console already owns the
 * SDK connection and a second DDS participant would be one too many.
 */
export function createConsoleActuator({ baseUrl, fetchFn = globalThis.fetch, config }) {
  const root = String(baseUrl).replace(/\/+$/u, "");
  async function post(path, body, timeoutMs) {
    const response = await fetchFn(`${root}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok || payload?.ok === false) {
      const error = new Error(payload?.error ?? `G1 console returned HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return payload ?? { ok: true };
  }
  return {
    kind: "console",
    async status() {
      try {
        const response = await fetchFn(`${root}/api/status`, { signal: AbortSignal.timeout(5000) });
        const payload = await response.json().catch(() => null);
        return { ok: true, ready: payload?.ready ?? true, reasons: payload?.reasons ?? [], ...payload };
      } catch (error) {
        return { ok: false, ready: false, reasons: [`console:${error.message}`] };
      }
    },
    stop() {
      return post("/api/stop", {}, 5000);
    },
    move(motion) {
      return post("/api/drive", { vx: motion.vx, vy: motion.vy, yaw: motion.yaw, duration: motion.duration }, config.controlTimeoutMs);
    },
    arm(actionId, primitive) {
      return post("/api/arm", { name: primitive, action_id: actionId }, config.armTimeoutMs);
    },
  };
}

/**
 * Evaluation actuator: a robot that is always standing, always ready, and never moves. It exists so
 * the gate can be exercised on depth and lidar alone offline (the null actuator refuses everything
 * with "no actuator attached", which makes every velocity token a refusal). It records what it was
 * asked to do. G1_ACTUATOR=mock; never for a deployment.
 */
export function createMockActuator({ now = () => Date.now() } = {}) {
  const dispatched = [];
  // The yaw the mock "turned" so far: reported as the IMU yaw, so the spatial memory's heading snap
  // agrees with the pulses it integrated instead of resetting every turn to zero.
  let yaw = 0;
  return {
    kind: "mock",
    dispatched,
    async status() {
      return { ok: true, ready: true, reasons: [], fsm_id: 801, mode_machine: 5, soc: 100, rpy: [0, 0, Number(yaw.toFixed(4))], mock: true };
    },
    async stop() {
      dispatched.push({ op: "stop", at: now() });
      return { ok: true, executed: true, mock: true };
    },
    async move(motion) {
      dispatched.push({ op: "move", motion, at: now() });
      yaw += (Number(motion?.yaw) || 0) * (Number(motion?.duration) || 0);
      if (yaw > Math.PI) yaw -= 2 * Math.PI;
      if (yaw <= -Math.PI) yaw += 2 * Math.PI;
      return { ok: true, move_rc: 0, mock: true, motion };
    },
    async arm(actionId, primitive) {
      dispatched.push({ op: "arm", actionId, primitive, at: now() });
      return { ok: true, code: 0, mock: true };
    },
  };
}

/** Dry actuator: reports readiness as unknown and refuses to move. Used when G1_ACTUATOR=none. */
export function createNullActuator() {
  return {
    kind: "none",
    async status() {
      return { ok: true, ready: false, reasons: ["no actuator attached"], fsm_id: null, soc: null };
    },
    async stop() {
      return { ok: true, executed: false, note: "no actuator attached" };
    },
    async move() {
      return { ok: false, error: "no actuator attached" };
    },
    async arm() {
      return { ok: false, error: "no actuator attached" };
    },
  };
}

/**
 * Gate a decision against live evidence and, if it survives, hand it to the actuator.
 *
 * Two safety properties inherited from the ancestors are preserved exactly:
 *   1. execution is opt-in twice over: the service needs G1_EXECUTE_ENABLED and the request needs
 *      `execute: true` (or exec_mode "real"). Neither alone moves the robot.
 *   2. anything that fails the gate is not merely skipped, it is answered with a zero-velocity stop,
 *      so a refused decision actively halts residual motion instead of letting a previous pulse run on.
 */
export function createExecutor({ actuator, config, logger = null, now = () => performance.now(), profile = G1_PROFILE, obstacles = null }) {
  const state = { armHeld: null };
  const stop = profile.roles.stop;
  const lidarConfigured = Boolean(profile.sensing?.lidar);
  const obstacleSource = obstacles ?? (lidarConfigured
    ? createSnapshotObstacleSource({ path: config.lidarSnapshot ?? profile.sensing.lidar.snapshot, profile })
    : createDisabledObstacleSource());
  const disabled = createDisabledObstacleSource();

  async function evidence({ frame, useLidar, nearCenterM, nearSideM }) {
    const [status, lidar] = await Promise.all([
      actuator.status().catch((error) => ({ ok: false, ready: false, reasons: [`status:${error.message}`] })),
      useLidar && lidarConfigured ? obstacleSource.read() : disabled.read(),
    ]);
    const depth = depthGate(frame?.depth, {
      nearCenterM: nearCenterM ?? config.depthNearCenterM,
      nearSideM: nearSideM ?? config.depthNearSideM,
      profile,
    });
    return { status, lidar, depth };
  }

  async function execute({
    token,
    durationMs = null,
    frame,
    execute: shouldExecute = false,
    useLidar = config.useLidar,
    nearCenterM = null,
    nearSideM = null,
  }) {
    const started = now();
    const record = profile.actionOrStop(token);
    const { status, lidar, depth } = await evidence({ frame, useLidar, nearCenterM, nearSideM });
    const gate = verdict({ token: record.token, state: status, depth, lidar, useLidar, armHeld: state.armHeld, profile });
    // An actuator that cannot express this token (the cerebellum has no strafe) refuses it at the
    // gate, so the refusal issues a stop and teaches a lesson like any other veto.
    if (gate.allowed && record.token !== stop && typeof actuator.supports === "function" && !actuator.supports(record)) {
      gate.allowed = false;
      gate.reasons = [...gate.reasons, "actuator:unsupported_token"];
    }
    const gateS = seconds(now() - started);

    const base = {
      token: record.token,
      action: record.name,
      kind: record.kind,
      gate: {
        ...gate,
        use_lidar: useLidar,
        state: { ready: status?.ready ?? false, fsm_id: status?.fsm_id ?? null, soc: status?.soc ?? null, reasons: status?.reasons ?? [], rpy: Array.isArray(status?.rpy) ? status.rpy : null },
        depth,
        lidar,
      },
      arm_held: state.armHeld,
      executed: false,
      preview: !shouldExecute,
      dispatch: null,
      timing: { gate_s: gateS, exec_s: null, total_s: gateS },
    };

    // A stop is always honoured, gate or no gate: refusing to stop is never the safe answer.
    if (record.token === stop) {
      if (shouldExecute && config.executeEnabled) {
        const dispatch = await actuator.stop().catch((error) => ({ ok: false, error: error.message }));
        base.dispatch = dispatch;
        base.executed = dispatch?.ok === true;
      }
      base.timing.total_s = seconds(now() - started);
      return base;
    }

    if (!gate.allowed) {
      // A real execution request actively zeros residual velocity. A preview must never touch the
      // actuator, even when the service itself has execution enabled.
      if (shouldExecute && config.executeEnabled) {
        base.dispatch = await actuator.stop().catch((error) => ({ ok: false, error: error.message }));
      }
      base.error_type = "gate_refused";
      base.error = gate.reasons.join("; ");
      base.timing.total_s = seconds(now() - started);
      logger?.info?.({ event: "gate_refused", token: record.token, reasons: gate.reasons });
      return base;
    }

    if (!shouldExecute) {
      base.motion = record.kind === "velocity" ? profile.motionFor?.(record.token, durationMs, { maxDurationS: config.maxPulseS }) ?? null : null;
      base.timing.total_s = seconds(now() - started);
      return base;
    }
    if (!config.executeEnabled) {
      base.error_type = "execute_disabled";
      base.error = "execution is disabled (set G1_EXECUTE_ENABLED=1)";
      base.timing.total_s = seconds(now() - started);
      return base;
    }

    const dispatchStarted = now();
    let dispatch;
    if (record.kind === "arm_primitive") {
      dispatch = await actuator.arm(record.actionId, record.primitive).catch((error) => ({ ok: false, error: error.message }));
      if (dispatch?.ok) state.armHeld = nextArmHeld(state.armHeld, record.token, profile);
    } else {
      const motion = profile.motionFor?.(record.token, durationMs, { maxDurationS: config.maxPulseS }) ?? null;
      base.motion = motion;
      // A profile with no motion envelope cannot express a velocity pulse: report it as a failed
      // dispatch rather than handing the actuator a null motion.
      dispatch = motion === null
        ? { ok: false, error: "profile has no motion envelope" }
        : await actuator.move(motion).catch((error) => ({ ok: false, error: error.message }));
    }
    base.dispatch = dispatch;
    base.executed = dispatch?.ok === true;
    base.preview = false;
    if (!base.executed) {
      // A latest-wins actuator reports a pulse a newer decision cut short as superseded, not failed.
      base.error_type = base.error_type ?? (dispatch?.superseded ? "superseded" : "dispatch_failed");
      base.error = dispatch?.error ?? "actuator rejected the command";
    }
    base.timing.exec_s = seconds(now() - dispatchStarted);
    base.timing.total_s = seconds(now() - started);
    return base;
  }

  /**
   * Raw arm dispatch by firmware action id, for ids outside the three-token set (the G1 exposes
   * dozens). Still state-gated and still subject to the holding interlock, because the failure mode
   * being prevented — a second holding pose erroring 7401, or an arm action on a robot that is not
   * standing — does not care which id was used.
   *
   * `force` overrides the holding interlock only (the caller knows the arm state better than the
   * bookkeeping does, e.g. after a daemon restart). It never overrides robot readiness: an arm action
   * on a tilted or wrong-FSM robot is exactly what the state gate exists to refuse.
   */
  async function armRaw(actionId, { force = false, holding = null } = {}) {
    const started = now();
    if (!config.executeEnabled) {
      return { ok: false, error_type: "execute_disabled", error: "execution is disabled (set G1_EXECUTE_ENABLED=1)" };
    }
    const status = await actuator.status().catch((error) => ({ ready: false, reasons: [`status:${error.message}`] }));
    if (!status?.ready) {
      return {
        ok: false,
        error_type: "gate_refused",
        error: `state:${(status?.reasons ?? ["unknown"]).join(",")}`,
        state: { ready: false, reasons: status?.reasons ?? [] },
      };
    }
    if (holding === true && state.armHeld !== null && !force) {
      return { ok: false, error_type: "gate_refused", error: `arm:already_holding_${state.armHeld}` };
    }
    const releaseId = profile.arms ? profile.ACTION_TABLE[profile.arms.release]?.actionId ?? null : null;
    const dispatch = await actuator.arm(actionId, null).catch((error) => ({ ok: false, error: error.message }));
    if (dispatch?.ok) {
      if (actionId === releaseId) state.armHeld = null;
      else if (holding === true) state.armHeld = `id:${actionId}`;
    }
    return {
      ok: dispatch?.ok === true,
      action_id: actionId,
      executed: dispatch?.ok === true,
      dispatch,
      arm_held: state.armHeld,
      timing: { total_s: seconds(now() - started) },
      error_type: dispatch?.ok === true ? null : "dispatch_failed",
      error: dispatch?.ok === true ? null : (dispatch?.error ?? "arm service rejected the action"),
    };
  }

  return {
    execute,
    armRaw,
    evidence,
    obstacles: obstacleSource,
    profile,
    stop: () => actuator.stop(),
    status: () => actuator.status(),
    armState: () => ({ held: state.armHeld, holding_tokens: [...profile.HOLDING_TOKENS] }),
    actuatorInfo: () => ({ kind: actuator.kind ?? null, serialized: actuator.serialized ?? true, latestWins: actuator.latestWins ?? false }),
    releaseArmState: () => {
      state.armHeld = null;
    },
  };
}
