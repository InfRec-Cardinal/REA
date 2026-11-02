// Bridge to the agent-vla-async cerebellum: the agent's tokens become the cerebellum's bounded plans.
//
// The cerebellum (agent-vla-async/src/agent_vla_async/runtime.py) is the robot-side asynchronous
// runtime: it owns perception, an advisory VLA loop, a latest-wins mailbox, a heartbeat/TTL watchdog
// and a structured ExecutionPlan executor behind a Unix-socket JSON RPC. It re-gates every pulse
// through its own navigation adapter (depth, lidar, robot state) before the sport service sees it.
//
// This module is an *actuator* in the agent's sense: the executor has already applied the agent's fused gate and
// the double opt-in when it calls move()/arm()/stop() here. The bridge then
//   - clamps the pulse into the cerebellum envelope (|vx| <= 0.20 m/s, |yaw| <= 0.35 rad/s) and splits
//     a long pulse into <= 0.60 s sub-pulses, because that is what the cerebellum's NavigationPlan
//     accepts (protocol.py) — speed is clamped, duration is kept, so a bridged W covers less ground
//     than a direct one, never more;
//   - submits one intent with execute:true, keeps its heartbeat alive while it runs, and returns only
//     when the cerebellum reports a terminal event — so a dead agent process means an expired intent
//     and a stopped robot, by the cerebellum's watchdog;
//   - maps stop() onto cancel + controller_stop, which is the cerebellum's zero-velocity path.
// Strafes (A/D) have no cerebellum pulse (vy is implicitly 0); supports() lets the executor refuse them
// at the gate instead of discovering it at dispatch.

import { createConnection } from "node:net";
import { performance } from "node:perf_hooks";

/** Mirrors MAX_SPORT_VX / MAX_SPORT_YAW / MAX_SPORT_PULSE_S in agent_vla_async/protocol.py. */
export const CEREBELLUM_ENVELOPE = Object.freeze({ maxVx: 0.2, maxYaw: 0.35, maxPulseS: 0.6 });

export const CEREBELLUM_TERMINAL_KINDS = Object.freeze(
  new Set(["EXECUTION_COMPLETE", "EXECUTION_FAILED", "PREVIEW_COMPLETE", "PREVIEW_FAILED", "EXPIRED", "CANCELLED", "SUPERSEDED", "CONTROLLER_STOPPED", "SHUTDOWN"]),
);

/**
 * The agent's arm primitive -> cerebellum named arm action. Only the two the cerebellum advertises
 * (runtime._availability: forward_push, release_arm) exist; `U` (both_hands_up) is therefore refused at
 * the agent's gate as unsupported rather than failing at dispatch.
 */
export const CEREBELLUM_ARM_ACTIONS = Object.freeze({ forward_push: "forward_push", release_arm: "release_arm" });

const DEFAULT_SOCKET = "/run/user/1000/g1-vla-async/control.sock";

function clampAbs(value, limit) {
  const number = Number(value) || 0;
  return Math.sign(number) * Math.min(Math.abs(number), limit);
}

/**
 * Split one agent motion {vx, vy, yaw, duration} into cerebellum pulses. Returns { pulses, clamped }
 * or { error } when the motion cannot be expressed (a lateral component).
 */
export function pulsesFor(motion, envelope = CEREBELLUM_ENVELOPE) {
  if (!motion || typeof motion !== "object") return { error: "motion is required" };
  if (Number(motion.vy) !== 0 && Number.isFinite(Number(motion.vy))) return { error: "cerebellum has no lateral pulse (vy must be 0)" };
  const duration = Number(motion.duration);
  if (!Number.isFinite(duration) || duration <= 0) return { error: "motion duration must be positive" };
  const vx = clampAbs(motion.vx, envelope.maxVx);
  const yaw = clampAbs(motion.yaw, envelope.maxYaw);
  const count = Math.max(1, Math.ceil(duration / envelope.maxPulseS - 1e-9));
  const each = Number((duration / count).toFixed(3));
  return {
    pulses: Array.from({ length: count }, () => ({ vx: Number(vx.toFixed(4)), yaw: Number(yaw.toFixed(4)), duration_s: each })),
    clamped: { vx: Math.abs(vx - Number(motion.vx || 0)) > 1e-9, yaw: Math.abs(yaw - Number(motion.yaw || 0)) > 1e-9 },
  };
}

/** One request, one JSON line back, over the robot-local Unix socket — the same shape as rpc.call() in Python. */
export function createCerebellumClient({ socketPath = DEFAULT_SOCKET, timeoutMs = 5000 } = {}) {
  function call(request) {
    return new Promise((resolve, reject) => {
      let settled = false;
      let buffer = "";
      const socket = createConnection(socketPath);
      const done = (fn, value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        fn(value);
      };
      const timer = setTimeout(() => {
        socket.destroy();
        done(reject, new Error(`cerebellum did not answer within ${timeoutMs} ms`));
      }, timeoutMs);
      socket.setEncoding("utf8");
      socket.once("connect", () => socket.write(`${JSON.stringify(request)}\n`));
      socket.on("data", chunk => {
        buffer += chunk;
        const newline = buffer.indexOf("\n");
        if (newline === -1) return;
        const line = buffer.slice(0, newline);
        socket.end();
        try {
          done(resolve, JSON.parse(line));
        } catch {
          done(reject, new Error("cerebellum returned malformed JSON"));
        }
      });
      socket.once("error", error => done(reject, error));
      socket.once("close", () => done(reject, new Error("cerebellum closed the RPC connection")));
    });
  }
  return { call, socketPath };
}

/**
 * The cerebellum as an agent actuator. `client.call(request)` is the only transport dependency, so tests
 * drive it with a scripted client and the integration test with the real daemon on a socket.
 */
export function createCerebellumActuator({
  client,
  config = {},
  logger = null,
  now = () => performance.now(),
  sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
}) {
  if (!client || typeof client.call !== "function") throw new TypeError("cerebellum actuator needs a client with call()");
  const heartbeatMs = Math.max(50, Math.round((config.cerebellumHeartbeatS ?? 0.5) * 1000));
  const policyWaitS = config.cerebellumPolicyWaitS ?? 3;
  const waitForPolicy = config.cerebellumWaitForPolicy ?? false;
  const task = config.cerebellumTask ?? "rea-agent bounded pulse";
  const controlTimeoutMs = config.controlTimeoutMs ?? 8000;
  const armTimeoutMs = config.armTimeoutMs ?? 30000;

  function supports(record) {
    if (!record) return false;
    if (record.kind === "velocity") return Number(record.vy ?? 0) === 0;
    if (record.kind === "arm_primitive") return Boolean(CEREBELLUM_ARM_ACTIONS[record.primitive]);
    return true;
  }

  async function status() {
    try {
      const caps = await client.call({ op: "capabilities" });
      const statuses = Array.isArray(caps.capability_status) ? caps.capability_status : [];
      const sport = statuses.find(item => item.name === "sport_navigation") ?? null;
      const state = sport?.detail?.state ?? {};
      const verdict = sport?.detail?.verdict ?? {};
      // `ready` is the robot's state only (usable, standing, not tilted). The cerebellum folds its
      // forward path-clear verdict into `sport_navigation.available`, but the agent's own gate already
      // decides per token whether something ahead matters — an arm gesture with a wall in front is
      // fine, a forward pulse is not — so the path verdict is surfaced separately, not as unreadiness.
      const reasons = [];
      if (!sport) reasons.push("cerebellum:no_sport_navigation");
      else if (sport.detail?.error) reasons.push(`cerebellum:${sport.detail.error}`);
      else if (state.usable === false || state.ready === false) reasons.push(`cerebellum:${state.reason ?? "robot_not_ready"}`);
      if (caps.dry_run === true || caps.execution_enabled === false) reasons.push("cerebellum:dry_run_mode");
      const blockedAhead = verdict.allowed === false;
      return {
        ok: true,
        ready: Boolean(sport) && reasons.length === 0,
        reasons,
        blocked_ahead: blockedAhead,
        blocked_reasons: blockedAhead ? (Array.isArray(verdict.reasons) ? verdict.reasons : []) : [],
        sport_available: Boolean(sport?.available),
        fsm_id: state.fsm_id ?? null,
        soc: state.soc ?? null,
        mode_machine: state.mode_machine ?? null,
        rpy: state.rpy ?? null,
        execute_enabled: caps.execution_enabled ?? null,
        dry_run: caps.dry_run ?? null,
        controller_mode: caps.controller_mode ?? null,
        capabilities: Array.isArray(caps.capabilities) ? caps.capabilities : [],
        kind: caps.kind ?? null,
      };
    } catch (error) {
      return { ok: false, ready: false, reasons: [`cerebellum:${error.message}`] };
    }
  }

  function terminal(event, events, intentId) {
    const result = event.detail?.result ?? {};
    const ok = event.kind === "EXECUTION_COMPLETE" && result.ok !== false;
    // SUPERSEDED / CANCELLED / EXPIRED carry no result; count the stages the cerebellum did complete.
    const stages = kind => events.filter(item => item.kind === "STAGE_COMPLETE" && item.detail?.stage === kind).length;
    return {
      ok,
      kind: event.kind,
      superseded: event.kind === "SUPERSEDED",
      intent_id: intentId,
      motion_commands_sent: Number(result.motion_commands_sent ?? stages("sport_pulse")),
      arm_actions_sent: Number(result.arm_actions_sent ?? stages("arm_action")),
      reason: result.reason ?? event.detail?.reason ?? null,
      events: events.map(item => item.kind),
      result,
      ...(ok ? {} : { error: result.reason ?? event.detail?.reason ?? event.kind }),
    };
  }

  // Latest wins, like the cerebellum itself: a second real dispatch supersedes the one in flight (the
  // cerebellum stops the first intent, then runs the second), and the first is reported as SUPERSEDED
  // with the pulses it did complete. Nothing is ever queued behind a running intent — a queued pulse
  // would execute on stale gate evidence, and a stop() must leave nothing waiting to move afterwards.
  /** Submit one execute intent and stay with it — heartbeats every `heartbeatMs` — until it ends. */
  async function runIntent(plan, { ttlS, deadlineMs }) {
    const before = await client.call({ op: "status" });
    let lastSequence = Number(before?.status?.last_sequence ?? 0);
    const submitted = await client.call({
      op: "submit",
      task,
      execute: true,
      plan,
      ttl_s: ttlS,
      heartbeat_timeout_s: Math.max(2, (heartbeatMs / 1000) * 4),
      inference_period_s: 0.5,
      policy_wait_timeout_s: policyWaitS,
      wait_for_policy: waitForPolicy,
      controller_mode: "HYBRID_HIGH_LEVEL",
    });
    if (!submitted?.ok) {
      const reasons = Array.isArray(submitted?.reasons) ? submitted.reasons : [];
      return { ok: false, kind: submitted?.kind ?? "EXECUTION_UNAVAILABLE", reasons, error: reasons.join("; ") || submitted?.error || "cerebellum refused the intent", submitted };
    }
    const intentId = submitted.intent?.intent_id;
    const started = now();
    const events = [];
    let heartbeatLost = false;
    try {
      while (now() - started < deadlineMs) {
        await sleep(heartbeatMs);
        const batch = await client.call({ op: "events", after: lastSequence });
        lastSequence = Number(batch?.last_sequence ?? lastSequence);
        for (const event of batch?.events ?? []) {
          if (event.intent_id !== intentId) continue;
          events.push(event);
          if (CEREBELLUM_TERMINAL_KINDS.has(event.kind)) return terminal(event, events, intentId);
        }
        if (heartbeatLost) break; // one extra events sweep after the intent went away, then give up
        const heartbeat = await client.call({ op: "heartbeat", intent_id: intentId });
        if (!heartbeat?.ok) heartbeatLost = true;
      }
    } catch (error) {
      // The transport failed while the cerebellum may still be walking. Cancel (its _deactivate
      // path issues the stop); if even that fails, its heartbeat watchdog expires the intent and
      // stops the robot within heartbeat_timeout_s. Either way this is reported as a failure.
      const cancelled = await client.call({ op: "cancel", intent_id: intentId }).then(() => true).catch(() => false);
      logger?.warn?.({ event: "cerebellum_transport_error", intent_id: intentId, cancelled, error: error?.message ?? String(error) });
      return { ok: false, kind: "TRANSPORT_ERROR", intent_id: intentId, cancelled, error: `cerebellum transport failed mid-intent: ${error?.message ?? String(error)}`, events: events.map(item => item.kind) };
    }
    await client.call({ op: "cancel", intent_id: intentId }).catch(() => {});
    logger?.warn?.({ event: "cerebellum_intent_timeout", intent_id: intentId, events: events.map(item => item.kind) });
    return { ok: false, kind: heartbeatLost ? "INTENT_LOST" : "TIMEOUT", intent_id: intentId, error: heartbeatLost ? "the cerebellum no longer holds the intent" : "the cerebellum reported no terminal event before the deadline", events: events.map(item => item.kind) };
  }

  async function move(motion) {
    const split = pulsesFor(motion);
    if (split.error) return { ok: false, error: split.error };
    const durationS = split.pulses.reduce((sum, pulse) => sum + pulse.duration_s, 0);
    const plan = { source: "rea-agent", task, navigation: { target: "token", pulses: split.pulses } };
    const out = await runIntent(plan, { ttlS: durationS + 10, deadlineMs: Math.max(controlTimeoutMs, (durationS + policyWaitS + 3) * 1000) });
    // What actually went to the sport service, for dead reckoning: the clamped speed over the pulses
    // the cerebellum confirmed (a superseded plan walked fewer than it was asked to).
    const completed = Number.isFinite(out.motion_commands_sent) ? Math.min(out.motion_commands_sent, split.pulses.length) : 0;
    const executedMotion = completed > 0
      ? { vx: split.pulses[0].vx, vy: 0, yaw: split.pulses[0].yaw, duration: Number(split.pulses.slice(0, completed).reduce((sum, pulse) => sum + pulse.duration_s, 0).toFixed(3)) }
      : null;
    return { ...out, pulses: split.pulses.length, clamped: split.clamped, executed_motion: executedMotion };
  }

  async function stop() {
    const cancelled = await client.call({ op: "cancel" }).catch(error => ({ ok: false, error: error.message }));
    const stopped = await client.call({ op: "controller_stop", controller_mode: "HYBRID_HIGH_LEVEL" }).catch(error => ({ ok: false, error: error.message }));
    const ok = stopped?.ok === true;
    return { ok, executed: ok, cancelled: cancelled?.cancelled === true, stopped, ...(ok ? {} : { error: stopped?.error ?? "cerebellum controller_stop failed" }) };
  }

  async function arm(actionId, primitive) {
    const action = CEREBELLUM_ARM_ACTIONS[primitive] ?? null;
    if (!action) return { ok: false, error: `no cerebellum arm action for ${primitive ?? `id ${actionId}`}` };
    const plan = { source: "rea-agent", task, arm_steps: [{ action }] };
    const out = await runIntent(plan, { ttlS: armTimeoutMs / 1000 + 10, deadlineMs: armTimeoutMs + policyWaitS * 1000 });
    return { ...out, action, action_id: actionId };
  }

  return { kind: "cerebellum", status, stop, move, arm, supports, envelope: CEREBELLUM_ENVELOPE, serialized: false, latestWins: true };
}
