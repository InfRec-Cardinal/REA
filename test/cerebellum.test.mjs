import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { test } from "node:test";

import { ACTION_TABLE, motionFor } from "../src/actions.mjs";
import { CEREBELLUM_ENVELOPE, createCerebellumActuator, createCerebellumClient, pulsesFor } from "../src/cerebellum.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const FAST = { cerebellumHeartbeatS: 0.1, cerebellumPolicyWaitS: 1, controlTimeoutMs: 3000, armTimeoutMs: 3000 };

/** A scripted cerebellum: the RPC surface of runtime.py, completing every plan on the first events sweep. */
function fakeCerebellum({ executeEnabled = true, sportAvailable = true, failPlan = false, sweepsBeforeComplete = 1 } = {}) {
  const calls = [];
  const events = [];
  const moved = [];
  let seq = 0;
  let generation = 0;
  let active = null;
  let pending = null;
  let heartbeats = 0;
  let stops = 0;
  let sweeps = 0;
  const emit = (kind, intentId, detail = {}) => events.push({ sequence: ++seq, kind, at: 0, intent_id: intentId, detail });
  emit("READY", null, {});
  const client = {
    async call(request) {
      calls.push(request);
      switch (request.op) {
        case "status":
          return { ok: true, status: { active, last_sequence: seq, execute_enabled: executeEnabled } };
        case "capabilities":
          return {
            ok: true,
            kind: executeEnabled && sportAvailable ? "EXECUTION_AVAILABLE" : "EXECUTION_UNAVAILABLE",
            execution_enabled: executeEnabled,
            dry_run: !executeEnabled,
            controller_mode: "HYBRID_HIGH_LEVEL",
            capabilities: sportAvailable ? ["sport_navigation", "arm_action:forward_push", "arm_action:release_arm"] : ["arm_action:forward_push", "arm_action:release_arm"],
            capability_status: [
              {
                name: "sport_navigation",
                available: sportAvailable,
                detail: { verdict: { allowed: sportAvailable, reasons: sportAvailable ? [] : ["depth_center_near"] }, state: { fsm_id: 801, soc: 77, mode_machine: 5, usable: true, ready: true } },
              },
              { name: "arm_action:forward_push", available: true },
              { name: "arm_action:release_arm", available: true },
            ],
          };
        case "submit": {
          if (request.execute && !executeEnabled) {
            return { ok: false, kind: "EXECUTION_UNAVAILABLE", reasons: ["cerebellum_service_is_running_in_dry_run_mode"], motion_commands_sent: 0 };
          }
          const intentId = `intent-${++generation}`;
          if (active) {
            // like runtime.py: the previous intent is stopped and superseded; its pending plan is dropped
            emit("SUPERSEDE_STOPPED", active.intent_id, {});
            emit("SUPERSEDED", active.intent_id, { by: intentId });
            stops += 1;
          }
          active = { intent_id: intentId, execute: request.execute };
          pending = { intentId, plan: request.plan };
          sweeps = 0;
          emit("ACCEPTED", intentId, { generation });
          return { ok: true, kind: "EXECUTION_AVAILABLE", intent: { intent_id: intentId, ...request }, generation };
        }
        case "events": {
          sweeps += 1;
          if (pending && sweeps >= sweepsBeforeComplete) {
            const { intentId, plan } = pending;
            pending = null;
            sweeps = 0;
            const pulses = plan.navigation?.pulses ?? [];
            pulses.forEach((pulse, index) => {
              moved.push(pulse);
              emit("STAGE_COMPLETE", intentId, { stage: "sport_pulse", index });
            });
            if (failPlan) {
              emit("EXECUTION_FAILED", intentId, { result: { ok: false, kind: "EXECUTION_FAILED", reason: "navigation_refused:depth_center_near", motion_commands_sent: 1, arm_actions_sent: 0 } });
            } else {
              emit("EXECUTION_COMPLETE", intentId, { result: { ok: true, kind: "EXECUTION_COMPLETE", motion_commands_sent: pulses.length, arm_actions_sent: (plan.arm_steps ?? []).length } });
            }
            active = null;
          }
          return { ok: true, events: events.filter(event => event.sequence > (request.after ?? 0)), last_sequence: seq };
        }
        case "heartbeat":
          heartbeats += 1;
          return active && active.intent_id === request.intent_id ? { ok: true, intent_id: request.intent_id } : { ok: false, error: "intent is not active" };
        case "cancel": {
          const had = active !== null;
          if (had) emit("CANCELLED", active.intent_id);
          active = null;
          return { ok: true, cancelled: had };
        }
        case "controller_stop":
          stops += 1;
          return { ok: true, kind: "CONTROLLER_STOPPED", controller_mode: "HYBRID_HIGH_LEVEL", result: { ok: true } };
        default:
          return { ok: false, error: "unknown operation" };
      }
    },
  };
  return { client, calls, moved, stats: () => ({ heartbeats, stops }) };
}

const instant = () => Promise.resolve();

test("agent pulses are clamped into the cerebellum envelope and split into <= 0.6 s sub-pulses", () => {
  const forward = pulsesFor(motionFor("W", null, { maxDurationS: 2 }));
  assert.equal(forward.pulses.length, 4, "2.0 s => four 0.5 s pulses");
  assert.deepEqual(forward.pulses[0], { vx: 0.2, yaw: -0.075, duration_s: 0.5 });
  assert.deepEqual(forward.clamped, { vx: true, yaw: false }, "0.30 m/s is clamped to the cerebellum's 0.20; the yaw trim is inside its envelope");
  assert.equal(Number(forward.pulses.reduce((sum, pulse) => sum + pulse.duration_s, 0).toFixed(3)), 2, "duration is kept: a bridged W covers less ground, never more");

  const turn = pulsesFor(motionFor("Q"));
  assert.deepEqual(turn.pulses, [{ vx: 0, yaw: 0.35, duration_s: 0.6 }]);
  assert.deepEqual(turn.clamped, { vx: false, yaw: false });

  assert.deepEqual(pulsesFor(motionFor("S", 700)), { pulses: [{ vx: -0.2, yaw: 0, duration_s: 0.35 }, { vx: -0.2, yaw: 0, duration_s: 0.35 }], clamped: { vx: true, yaw: false } });
  assert.match(pulsesFor(motionFor("A")).error, /no lateral pulse/u);
  assert.match(pulsesFor({ vx: 0.1, vy: 0, yaw: 0, duration: 0 }).error, /duration/u);
  assert.match(pulsesFor(null).error, /required/u);
  assert.equal(CEREBELLUM_ENVELOPE.maxVx, 0.2);
});

test("a forward token becomes one heartbeat-kept execute intent that ends on the cerebellum's terminal event", async () => {
  const fake = fakeCerebellum();
  const actuator = createCerebellumActuator({ client: fake.client, config: FAST, sleep: instant });
  assert.equal(actuator.kind, "cerebellum");
  const out = await actuator.move(motionFor("W", null, { maxDurationS: 2 }));
  assert.equal(out.ok, true);
  assert.equal(out.kind, "EXECUTION_COMPLETE");
  assert.equal(out.motion_commands_sent, 4);
  assert.equal(out.pulses, 4);
  assert.deepEqual(out.executed_motion, { vx: 0.2, vy: 0, yaw: -0.075, duration: 2 }, "what actually went out, for dead reckoning");
  assert.equal(out.intent_id, "intent-1");
  assert.deepEqual(out.events, ["ACCEPTED", "STAGE_COMPLETE", "STAGE_COMPLETE", "STAGE_COMPLETE", "STAGE_COMPLETE", "EXECUTION_COMPLETE"]);
  assert.equal(fake.moved.length, 4);
  assert.equal(fake.moved[0].vx, 0.2);

  const submit = fake.calls.find(call => call.op === "submit");
  assert.equal(submit.execute, true);
  assert.equal(submit.wait_for_policy, false, "the agent already decided; the pulse does not wait for the advisory VLA");
  assert.equal(submit.policy_wait_timeout_s, 1);
  assert.equal(submit.controller_mode, "HYBRID_HIGH_LEVEL");
  assert.equal(submit.plan.source, "rea-agent");
  assert.equal(submit.plan.navigation.pulses.length, 4);
  assert.ok(submit.ttl_s > 2);
  assert.ok(submit.heartbeat_timeout_s >= 2);
  assert.deepEqual(fake.calls.map(call => call.op).slice(0, 3), ["status", "submit", "events"]);
});

test("a dry-run cerebellum, a refused plan and a lost intent are reported, never mistaken for motion", async () => {
  const dry = fakeCerebellum({ executeEnabled: false });
  const dryOut = await createCerebellumActuator({ client: dry.client, config: FAST, sleep: instant }).move(motionFor("W"));
  assert.equal(dryOut.ok, false);
  assert.equal(dryOut.kind, "EXECUTION_UNAVAILABLE");
  assert.match(dryOut.error, /dry_run_mode/u);
  assert.equal(dry.moved.length, 0);

  const refused = fakeCerebellum({ failPlan: true });
  const refusedOut = await createCerebellumActuator({ client: refused.client, config: FAST, sleep: instant }).move(motionFor("W"));
  assert.equal(refusedOut.ok, false);
  assert.equal(refusedOut.kind, "EXECUTION_FAILED");
  assert.match(refusedOut.error, /navigation_refused/u);
  assert.equal(refusedOut.motion_commands_sent, 1);

  // an intent the cerebellum no longer holds (expired) yields INTENT_LOST after one extra events sweep
  const lost = fakeCerebellum();
  const original = lost.client.call.bind(lost.client);
  lost.client.call = async request => {
    if (request.op === "events") return { ok: true, events: [], last_sequence: 0 };
    if (request.op === "heartbeat") return { ok: false, error: "intent is not active" };
    return original(request);
  };
  const lostOut = await createCerebellumActuator({ client: lost.client, config: FAST, sleep: instant }).move(motionFor("Q"));
  assert.equal(lostOut.ok, false);
  assert.equal(lostOut.kind, "INTENT_LOST");
  assert.equal(lost.calls.filter(call => call.op === "cancel").length, 1, "the bridge cancels what it cannot follow");

  const lateral = await createCerebellumActuator({ client: fakeCerebellum().client, config: FAST, sleep: instant }).move(motionFor("D"));
  assert.equal(lateral.ok, false);
  assert.match(lateral.error, /no lateral pulse/u);
});

test("stop is cancel plus controller_stop, status maps the capability handshake, arm uses named actions", async () => {
  const fake = fakeCerebellum();
  const actuator = createCerebellumActuator({ client: fake.client, config: FAST, sleep: instant });
  const stopped = await actuator.stop();
  assert.equal(stopped.ok, true);
  assert.equal(stopped.executed, true);
  assert.equal(stopped.cancelled, false, "nothing was active");
  assert.deepEqual(fake.calls.map(call => call.op), ["cancel", "controller_stop"]);
  assert.equal(fake.stats().stops, 1);

  const status = await actuator.status();
  assert.equal(status.ready, true);
  assert.deepEqual(status.reasons, []);
  assert.equal(status.fsm_id, 801);
  assert.equal(status.soc, 77);
  assert.equal(status.execute_enabled, true);
  assert.deepEqual(status.capabilities, ["sport_navigation", "arm_action:forward_push", "arm_action:release_arm"]);

  // something close ahead is not unreadiness: the agent's gate decides per token what a near obstacle means
  const blocked = await createCerebellumActuator({ client: fakeCerebellum({ sportAvailable: false }).client, config: FAST, sleep: instant }).status();
  assert.equal(blocked.ready, true, "the robot is standing and usable");
  assert.deepEqual(blocked.reasons, []);
  assert.equal(blocked.blocked_ahead, true);
  assert.deepEqual(blocked.blocked_reasons, ["depth_center_near"]);
  assert.equal(blocked.sport_available, false);
  // a dry-run cerebellum cannot execute anything, so it is not ready for a real dispatch
  const dry = await createCerebellumActuator({ client: fakeCerebellum({ executeEnabled: false }).client, config: FAST, sleep: instant }).status();
  assert.equal(dry.ready, false);
  assert.deepEqual(dry.reasons, ["cerebellum:dry_run_mode"]);
  const down = await createCerebellumActuator({ client: { call: async () => { throw new Error("ECONNREFUSED"); } }, config: FAST, sleep: instant }).status();
  assert.equal(down.ready, false);
  assert.deepEqual(down.reasons, ["cerebellum:ECONNREFUSED"]);

  const arm = await actuator.arm(36, "forward_push");
  assert.equal(arm.ok, true);
  assert.equal(arm.action, "forward_push");
  assert.equal(arm.arm_actions_sent, 1);
  const submit = fake.calls.filter(call => call.op === "submit").at(-1);
  assert.deepEqual(submit.plan.arm_steps, [{ action: "forward_push" }]);
  assert.equal(submit.plan.navigation, undefined);
  const unknown = await actuator.arm(7, "wave");
  assert.equal(unknown.ok, false);
  assert.match(unknown.error, /no cerebellum arm action/u);

  assert.equal(actuator.supports(ACTION_TABLE.W), true);
  assert.equal(actuator.supports(ACTION_TABLE.Q), true);
  assert.equal(actuator.supports(ACTION_TABLE.A), false, "the cerebellum has no strafe");
  assert.equal(actuator.supports(ACTION_TABLE.D), false);
  assert.equal(actuator.supports(ACTION_TABLE.P), true);
  assert.equal(actuator.supports(ACTION_TABLE.R), true);
  assert.equal(actuator.supports(ACTION_TABLE.U), false, "the cerebellum advertises no both_hands_up action, so U is refused at the gate");
  assert.equal(actuator.supports(ACTION_TABLE.X), true);
  const up = await actuator.arm(18, "both_hands_up");
  assert.equal(up.ok, false);
  assert.match(up.error, /no cerebellum arm action/u);
  assert.throws(() => createCerebellumActuator({ client: {} }), /needs a client/u);
});

test("the bridge keeps the heartbeat alive while a plan runs, and reports TIMEOUT with a cancel when no terminal event comes", async () => {
  const slow = fakeCerebellum({ sweepsBeforeComplete: 4 });
  const actuator = createCerebellumActuator({ client: slow.client, config: FAST, sleep: instant });
  const out = await actuator.move(motionFor("Q"));
  assert.equal(out.ok, true);
  assert.ok(slow.stats().heartbeats >= 3, `heartbeats were sent while waiting (${slow.stats().heartbeats})`);
  assert.deepEqual(slow.calls.filter(call => call.op === "heartbeat").map(call => call.intent_id), Array(slow.stats().heartbeats).fill("intent-1"));

  const never = fakeCerebellum({ sweepsBeforeComplete: Number.POSITIVE_INFINITY });
  let clock = 0;
  const stuck = createCerebellumActuator({ client: never.client, config: FAST, sleep: async () => { clock += 1000; }, now: () => clock });
  const timedOut = await stuck.move(motionFor("Q"));
  assert.equal(timedOut.ok, false);
  assert.equal(timedOut.kind, "TIMEOUT");
  assert.match(timedOut.error, /no terminal event/u);
  assert.equal(never.calls.filter(call => call.op === "cancel").length, 1, "the bridge cancels the intent it gave up on");
  assert.ok(never.stats().heartbeats > 0);
});

test("a transport failure mid-intent cancels the intent and is reported as TRANSPORT_ERROR, never as motion", async () => {
  const fake = fakeCerebellum({ sweepsBeforeComplete: 3 });
  const original = fake.client.call.bind(fake.client);
  let sweeps = 0;
  fake.client.call = async request => {
    if (request.op === "events" && ++sweeps === 2) throw new Error("ECONNRESET");
    return original(request);
  };
  const warnings = [];
  const actuator = createCerebellumActuator({ client: fake.client, config: FAST, sleep: instant, logger: { warn: record => warnings.push(record) } });
  const out = await actuator.move(motionFor("W"));
  assert.equal(out.ok, false);
  assert.equal(out.kind, "TRANSPORT_ERROR");
  assert.equal(out.cancelled, true);
  assert.match(out.error, /ECONNRESET/u);
  assert.equal(fake.calls.filter(call => call.op === "cancel").length, 1, "the still-running plan is cancelled (the cerebellum's _deactivate stops the robot)");
  assert.equal(warnings[0].event, "cerebellum_transport_error");

  // if even the cancel fails, the cerebellum's heartbeat watchdog is what stops the robot; report it honestly
  const dead = fakeCerebellum({ sweepsBeforeComplete: 3 });
  const orig2 = dead.client.call.bind(dead.client);
  let n = 0;
  dead.client.call = async request => {
    if (request.op === "submit") return orig2(request);
    if (request.op === "status") return orig2(request);
    if (++n >= 1) throw new Error("socket closed");
    return orig2(request);
  };
  const lost = await createCerebellumActuator({ client: dead.client, config: FAST, sleep: instant }).move(motionFor("W"));
  assert.equal(lost.kind, "TRANSPORT_ERROR");
  assert.equal(lost.cancelled, false);
});

test("latest wins: a second move supersedes the one in flight, which is reported as SUPERSEDED with the pulses it did complete", async () => {
  const fake = fakeCerebellum({ sweepsBeforeComplete: 3 });
  const actuator = createCerebellumActuator({ client: fake.client, config: FAST, sleep: instant });
  assert.equal(actuator.serialized, false);
  assert.equal(actuator.latestWins, true);
  const first = actuator.move(motionFor("W", null, { maxDurationS: 2 }));
  await Promise.resolve();
  const second = await actuator.move(motionFor("E"));
  const superseded = await first;
  assert.equal(second.ok, true);
  assert.equal(second.intent_id, "intent-2");
  assert.equal(superseded.ok, false);
  assert.equal(superseded.kind, "SUPERSEDED");
  assert.equal(superseded.superseded, true);
  assert.equal(superseded.intent_id, "intent-1");
  assert.equal(superseded.motion_commands_sent, 0, "no stage of the first plan had completed when it was cut short");
  assert.equal(superseded.executed_motion, null);
  assert.ok(fake.stats().stops >= 1, "the cerebellum stopped the first intent before running the second");
  assert.equal(fake.calls.filter(call => call.op === "submit").length, 2);
});

test("stop() leaves nothing waiting to move: an intent in flight is cancelled and no plan is submitted afterwards", async () => {
  const fake = fakeCerebellum({ sweepsBeforeComplete: Number.POSITIVE_INFINITY });
  const actuator = createCerebellumActuator({ client: fake.client, config: FAST, sleep: instant });
  const inflight = actuator.move(motionFor("W"));
  await Promise.resolve();
  const stopped = await actuator.stop();
  assert.equal(stopped.ok, true);
  assert.equal(stopped.cancelled, true, "the running intent was cancelled");
  const out = await inflight;
  assert.equal(out.ok, false);
  assert.equal(out.kind, "CANCELLED");
  const ops = fake.calls.map(call => call.op);
  const stopAt = ops.lastIndexOf("controller_stop");
  assert.equal(ops.slice(stopAt + 1).includes("submit"), false, "nothing is submitted after a stop");
  assert.equal(fake.calls.filter(call => call.op === "submit").length, 1);
});

// ---- integration: the real cerebellum runtime (fake adapters) on a Unix socket, driven by the real client
const PY_ENV = { ...process.env, PYTHONPATH: `${join(ROOT, "agent-vla-async", "src")}${process.env.PYTHONPATH ? `:${process.env.PYTHONPATH}` : ""}` };
const pythonReady = (() => {
  const probe = spawnSync("python3", ["-c", "import agent_vla_async.runtime"], { env: PY_ENV, encoding: "utf8" });
  return probe.status === 0;
})();

test("integration: the real cerebellum runtime executes a bridged W as four 0.2 m/s pulses and stops on request", { skip: pythonReady ? false : "python3 with agent_vla_async importable is required" }, async () => {
  const socketPath = `/tmp/nea-cb-${process.pid}.sock`;
  const daemon = spawn("python3", [join(ROOT, "agent-vla-async", "scripts", "fake_cerebellum.py"), "--socket", socketPath, "--execute", "--policy-delay", "0.5"], {
    env: PY_ENV,
    stdio: ["pipe", "pipe", "pipe"],
  });
  const lines = [];
  let stderr = "";
  daemon.stderr.on("data", chunk => { stderr += chunk; });
  const ready = new Promise((resolve, reject) => {
    let buffer = "";
    daemon.stdout.on("data", chunk => {
      buffer += chunk;
      let newline;
      while ((newline = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        if (!line.trim()) continue;
        const message = JSON.parse(line);
        lines.push(message);
        if (message.event === "ready") resolve(message);
      }
    });
    daemon.once("exit", code => reject(new Error(`fake cerebellum exited early (${code}): ${stderr}`)));
    setTimeout(() => reject(new Error(`fake cerebellum did not become ready: ${stderr}`)), 15000).unref();
  });
  try {
    await ready;
    assert.equal(existsSync(socketPath), true);
    const client = createCerebellumClient({ socketPath, timeoutMs: 3000 });
    const ping = await client.call({ op: "ping" });
    assert.equal(ping.service, "agent-vla-cerebellum");

    const actuator = createCerebellumActuator({ client, config: { ...FAST, cerebellumPolicyWaitS: 0.2 } });
    const status = await actuator.status();
    assert.equal(status.ready, true, JSON.stringify(status));
    assert.equal(status.execute_enabled, true);

    const started = performance.now();
    const out = await actuator.move(motionFor("W", null, { maxDurationS: 2 }));
    const elapsedS = (performance.now() - started) / 1000;
    assert.equal(out.ok, true, JSON.stringify(out));
    assert.equal(out.kind, "EXECUTION_COMPLETE");
    assert.equal(out.motion_commands_sent, 4);
    assert.ok(out.events.includes("POLICY_ADVISORY_SKIPPED"), "the bounded pulse did not wait for the 0.5 s mock policy");
    assert.ok(elapsedS < 3, `bridged pulse took ${elapsedS.toFixed(2)} s`);

    const stopped = await actuator.stop();
    assert.equal(stopped.ok, true);

    const strafe = await actuator.move(motionFor("A"));
    assert.equal(strafe.ok, false);
  } finally {
    daemon.stdin.end();
    const exit = await new Promise(resolve => {
      const timer = setTimeout(() => { daemon.kill("SIGKILL"); resolve(null); }, 5000);
      daemon.once("exit", () => { clearTimeout(timer); resolve(lines.find(line => line.event === "exit") ?? null); });
    });
    for (const leftover of [socketPath, `${socketPath}.lock`]) rmSync(leftover, { force: true });
    if (exit) {
      assert.equal(exit.steps.length, 4, "the fake navigation adapter received exactly the four sub-pulses");
      assert.deepEqual(exit.steps[0], { vx: 0.2, yaw: -0.075, duration_s: 0.5 });
      assert.ok(exit.stops >= 1, "controller_stop reached navigation.stop()");
      assert.equal(exit.status.motion_pulse_count, 4);
    }
  }
});
