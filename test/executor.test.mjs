import assert from "node:assert/strict";
import { test } from "node:test";

import { createExecutor } from "../src/executor.mjs";
import HOLONOMIC from "./fixtures/profiles/holonomic.mjs";

const CONFIG = {
  executeEnabled: true,
  useLidar: false,
  lidarSnapshot: "/nonexistent/obstacles.json",
  depthNearCenterM: 0.55,
  depthNearSideM: 0.35,
  controlTimeoutMs: 1000,
  armTimeoutMs: 1000,
  maxPulseS: 2,
};

const CLEAR_FRAME = { depth: { center_m: 2, left_m: 2, right_m: 2 } };
const BLOCKED_FRAME = { depth: { center_m: 0.2, left_m: 2, right_m: 2 } };

function recordingActuator({ ready = true } = {}) {
  const calls = [];
  return {
    calls,
    kind: "test",
    async status() {
      return { ok: true, ready, reasons: ready ? [] : ["not ready"], fsm_id: ready ? 801 : 4, soc: 90 };
    },
    async stop() {
      calls.push({ op: "stop" });
      return { ok: true };
    },
    async move(motion) {
      calls.push({ op: "move", motion });
      return { ok: true, move_rc: 0 };
    },
    async arm(actionId) {
      calls.push({ op: "arm", actionId });
      return { ok: true, code: 0 };
    },
  };
}

test("a clear forward decision reaches the actuator", async () => {
  const actuator = recordingActuator();
  const executor = createExecutor({ actuator, config: CONFIG });
  const result = await executor.execute({ token: "W", durationMs: 1000, frame: CLEAR_FRAME, execute: true });
  assert.equal(result.executed, true);
  assert.equal(actuator.calls.at(-1).op, "move");
  assert.ok(actuator.calls.at(-1).motion.vx > 0);
});

test("a refused decision issues a stop rather than doing nothing", async () => {
  // Silently skipping would let a previous pulse keep running; the refusal must actively halt.
  const actuator = recordingActuator();
  const executor = createExecutor({ actuator, config: CONFIG });
  const result = await executor.execute({ token: "W", durationMs: 1000, frame: BLOCKED_FRAME, execute: true });
  assert.equal(result.executed, false);
  assert.equal(result.error_type, "gate_refused");
  assert.ok(result.gate.reasons.some((reason) => reason.startsWith("depth:")));
  assert.deepEqual(actuator.calls.at(-1), { op: "stop" });
});

test("execution requires the request flag as well as the service flag", async () => {
  const actuator = recordingActuator();
  const executor = createExecutor({ actuator, config: CONFIG });
  const preview = await executor.execute({ token: "W", durationMs: 1000, frame: CLEAR_FRAME, execute: false });
  assert.equal(preview.executed, false);
  assert.equal(preview.preview, true);
  assert.ok(preview.motion, "a preview still reports what it would have sent");
  assert.equal(actuator.calls.length, 0, "a preview must not touch the actuator");
});

test("a gate-refused preview never issues stop when service execution is enabled", async () => {
  const actuator = recordingActuator();
  const executor = createExecutor({ actuator, config: CONFIG });
  const preview = await executor.execute({
    token: "W",
    durationMs: 1000,
    frame: BLOCKED_FRAME,
    execute: false,
  });
  assert.equal(preview.error_type, "gate_refused");
  assert.equal(preview.preview, true);
  assert.equal(preview.executed, false);
  assert.equal(preview.dispatch, null);
  assert.equal(actuator.calls.length, 0, "preview-only website traffic must issue no SDK command");
});

test("execution requires the service flag as well as the request flag", async () => {
  const actuator = recordingActuator();
  const executor = createExecutor({ actuator, config: { ...CONFIG, executeEnabled: false } });
  const result = await executor.execute({ token: "W", durationMs: 1000, frame: CLEAR_FRAME, execute: true });
  assert.equal(result.executed, false);
  assert.equal(result.error_type, "execute_disabled");
  assert.equal(actuator.calls.length, 0);
});

test("a stop is honoured even when the robot is not walk-ready", async () => {
  const actuator = recordingActuator({ ready: false });
  const executor = createExecutor({ actuator, config: CONFIG });
  const result = await executor.execute({ token: "X", frame: CLEAR_FRAME, execute: true });
  assert.equal(result.executed, true);
  assert.deepEqual(actuator.calls.at(-1), { op: "stop" });
});

test("a robot that is not walk-ready cannot be driven", async () => {
  const actuator = recordingActuator({ ready: false });
  const executor = createExecutor({ actuator, config: CONFIG });
  const result = await executor.execute({ token: "W", durationMs: 1000, frame: CLEAR_FRAME, execute: true });
  assert.equal(result.executed, false);
  assert.ok(result.gate.reasons.some((reason) => reason.startsWith("state:")));
});

test("arm poses interlock: hold, refuse a second hold, release, hold again", async () => {
  const actuator = recordingActuator();
  const executor = createExecutor({ actuator, config: CONFIG });

  const push = await executor.execute({ token: "P", frame: null, execute: true, useLidar: false });
  assert.equal(push.executed, true);
  assert.equal(executor.armState().held, "P");

  const second = await executor.execute({ token: "U", frame: null, execute: true, useLidar: false });
  assert.equal(second.executed, false, "the arm service errors 7401 on a second holding pose");
  assert.equal(second.error_type, "gate_refused");

  const release = await executor.execute({ token: "R", frame: null, execute: true, useLidar: false });
  assert.equal(release.executed, true);
  assert.equal(executor.armState().held, null);

  const again = await executor.execute({ token: "U", frame: null, execute: true, useLidar: false });
  assert.equal(again.executed, true);
});

test("a raw arm id is refused when the robot is not ready, force or not; force only overrides the holding interlock", async () => {
  const notReady = recordingActuator({ ready: false });
  const refused = createExecutor({ actuator: notReady, config: CONFIG });
  assert.equal((await refused.armRaw(36)).ok, false);
  const forced = await refused.armRaw(36, { force: true });
  assert.equal(forced.ok, false, "a tilted or wrong-FSM robot is exactly what the state gate refuses");
  assert.equal(forced.error_type, "gate_refused");
  assert.equal(notReady.calls.some(call => call.op === "arm"), false);

  const ready = recordingActuator();
  const executor = createExecutor({ actuator: ready, config: CONFIG });
  assert.equal((await executor.execute({ token: "U", frame: CLEAR_FRAME, execute: true })).executed, true);
  assert.equal((await executor.armRaw(36, { holding: true })).error_type, "gate_refused", "a second holding pose is refused");
  assert.equal((await executor.armRaw(36, { holding: true, force: true })).ok, true, "force overrides the interlock");
});

test("a missing lidar snapshot blocks motion when lidar is required", async () => {
  const actuator = recordingActuator();
  const executor = createExecutor({ actuator, config: { ...CONFIG, useLidar: true } });
  const result = await executor.execute({ token: "W", durationMs: 1000, frame: CLEAR_FRAME, execute: true, useLidar: true });
  assert.equal(result.executed, false);
  assert.ok(result.gate.reasons.some((reason) => reason.startsWith("lidar:")));
});

test("a token the actuator cannot express is refused at the gate, with a stop, before dispatch", async () => {
  const actuator = recordingActuator();
  actuator.supports = record => record.token !== "D";
  const executor = createExecutor({ actuator, config: CONFIG });
  const real = await executor.execute({ token: "D", durationMs: 600, frame: CLEAR_FRAME, execute: true });
  assert.equal(real.executed, false);
  assert.equal(real.error_type, "gate_refused");
  assert.deepEqual(real.gate.reasons, ["actuator:unsupported_token"]);
  assert.deepEqual(actuator.calls.at(-1), { op: "stop" }, "a real request that is refused still zeros velocity");
  assert.equal(actuator.calls.some(call => call.op === "move"), false);

  const preview = await executor.execute({ token: "D", durationMs: 600, frame: CLEAR_FRAME, execute: false });
  assert.equal(preview.gate.allowed, false);
  assert.equal(preview.preview, true);
  assert.equal(actuator.calls.length, 1, "a preview never touches the actuator");

  const allowed = await executor.execute({ token: "A", durationMs: 600, frame: CLEAR_FRAME, execute: true });
  assert.equal(allowed.executed, true);
  const stop = await executor.execute({ token: "X", frame: CLEAR_FRAME, execute: true });
  assert.equal(stop.gate.allowed, true, "a stop is never refused, whatever the actuator supports");
});

test("a dispatch cut short by a newer decision is reported as superseded, not as an actuator failure", async () => {
  const actuator = recordingActuator();
  actuator.move = async () => ({ ok: false, kind: "SUPERSEDED", superseded: true, motion_commands_sent: 1, error: "SUPERSEDED" });
  const executor = createExecutor({ actuator, config: CONFIG });
  const result = await executor.execute({ token: "W", durationMs: 1000, frame: CLEAR_FRAME, execute: true });
  assert.equal(result.executed, false);
  assert.equal(result.error_type, "superseded");
  assert.equal(result.dispatch.motion_commands_sent, 1);
  assert.deepEqual(executor.actuatorInfo(), { kind: "test", serialized: true, latestWins: false });
});

test("the mock actuator is always ready and records what it was asked, and still obeys the double opt-in", async () => {
  const { createMockActuator } = await import("../src/executor.mjs");
  const mock = createMockActuator({ now: () => 5 });
  const executor = createExecutor({ actuator: mock, config: CONFIG });
  const preview = await executor.execute({ token: "W", durationMs: 1000, frame: CLEAR_FRAME, execute: false });
  assert.equal(preview.gate.allowed, true, "a ready robot with clear depth lets W through");
  assert.equal(preview.executed, false);
  assert.equal(mock.dispatched.length, 0, "a preview never reaches even a mock actuator");
  const real = await executor.execute({ token: "W", durationMs: 1000, frame: CLEAR_FRAME, execute: true });
  assert.equal(real.executed, true);
  assert.deepEqual(mock.dispatched.at(-1).op, "move");
  assert.equal(real.gate.state.rpy.length, 3, "the mock reports an IMU pose for the spatial memory");
  const blocked = await executor.execute({ token: "W", durationMs: 1000, frame: BLOCKED_FRAME, execute: true });
  assert.equal(blocked.error_type, "gate_refused", "depth still gates a mock robot");
  assert.equal(mock.dispatched.at(-1).op, "stop");
});

test("the mock actuator reports the yaw it pretended to turn, so the spatial heading snap agrees with the pulses", async () => {
  const { createMockActuator } = await import("../src/executor.mjs");
  const mock = createMockActuator();
  const executor = createExecutor({ actuator: mock, config: CONFIG });
  assert.equal((await mock.status()).rpy[2], 0);
  const turned = await executor.execute({ token: "Q", frame: CLEAR_FRAME, execute: true });
  assert.equal(turned.executed, true);
  const yaw = (await mock.status()).rpy[2];
  assert.ok(Math.abs(yaw - 0.35 * 0.6) < 1e-3, `one Q pulse is 0.21 rad, got ${yaw}`);
  await executor.execute({ token: "E", frame: CLEAR_FRAME, execute: true });
  assert.ok(Math.abs((await mock.status()).rpy[2]) < 1e-3, "an E pulse turns it back");
});

test("a refused decision under another profile reports that profile's stop token and never mentions lidar", async () => {
  const actuator = recordingActuator();
  const executor = createExecutor({ actuator, config: { ...CONFIG, useLidar: true }, profile: HOLONOMIC });
  const refused = await executor.execute({ token: "F", durationMs: 1000, frame: { depth: { center_m: 0.2, left_m: 2, right_m: 2 } }, execute: true });
  assert.equal(refused.executed, false);
  assert.deepEqual(refused.gate.reasons, ["depth:depth_center_near"]);
  assert.equal(refused.gate.lidar.reason, "lidar_disabled", "no lidar in this profile");
  assert.deepEqual(actuator.calls.at(-1), { op: "stop" });
  const unknown = await executor.execute({ token: "W", frame: CLEAR_FRAME, execute: true });
  assert.equal(unknown.token, "H", "an unknown token is this profile's stop");
  assert.equal(unknown.executed, true, "and a stop is honoured");
  assert.deepEqual(executor.armState(), { held: null, holding_tokens: [] });
  const moved = await executor.execute({ token: "F", durationMs: 1000, frame: CLEAR_FRAME, execute: true });
  assert.equal(moved.executed, true);
  assert.equal(actuator.calls.at(-1).motion.vx, 0.2);
});
