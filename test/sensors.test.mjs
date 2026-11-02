import assert from "node:assert/strict";
import { test } from "node:test";

import { createDisabledObstacleSource, createFixtureObstacleSource, createSnapshotObstacleSource, evaluateLidar, parseFixtureLidar, sectorVerdict } from "../src/sensors/obstacles.mjs";
import { createFixtureCamera, parseFixtureDepth } from "../src/camera.mjs";
import { createExecutor } from "../src/executor.mjs";
import { lidarGate } from "../src/gate.mjs";

const NOW = 1_000;
function snapshot({ center = 3, left = 3, right = 3, ageS = 0.1, pointsPerS = 9000 } = {}) {
  // 36 sectors of 10 degrees, index 0 at -180; centre window [-12,12] is index 17/18, left [20,60] 20..23, right [-40,-15] 14..16
  const ring = new Array(36).fill(5);
  ring[17] = center; ring[18] = center;
  for (const i of [20, 21, 22, 23]) ring[i] = left;
  for (const i of [14, 15, 16]) ring[i] = right;
  return { unix_s: NOW - ageS, ring_m: ring, points_per_s: pointsPerS };
}

test("evaluateLidar reproduces the gate's snapshot rules from a parsed snapshot", () => {
  assert.equal(evaluateLidar(snapshot(), { now: NOW }).reason, "clear");
  assert.equal(evaluateLidar(snapshot({ center: 0.5 }), { now: NOW }).reason, "lidar_center_near");
  assert.equal(evaluateLidar(snapshot({ left: 0.3 }), { now: NOW }).reason, "lidar_left_near");
  assert.equal(evaluateLidar(snapshot({ right: 0.3 }), { now: NOW }).reason, "lidar_right_near");
  assert.equal(evaluateLidar(snapshot({ ageS: 2 }), { now: NOW }).reason, "lidar_stale");
  assert.equal(evaluateLidar(snapshot({ pointsPerS: 10 }), { now: NOW }).reason, "lidar_stream_starved");
  assert.equal(evaluateLidar({ unix_s: NOW, ring_m: [] }, { now: NOW }).reason, "lidar_ring_empty");
  assert.equal(evaluateLidar(null, { now: NOW }).reason, "lidar_snapshot_unreadable");
  assert.equal(evaluateLidar(snapshot({ center: 0.95 }), { now: NOW, nearCenterM: 1.2 }).reason, "lidar_center_near", "overrides still apply");
});

test("the snapshot source maps a missing and a malformed file to the two existing reasons", async () => {
  const missing = createSnapshotObstacleSource({ path: "/nonexistent/obstacles.json" });
  assert.equal((await missing.read()).reason, "lidar_snapshot_missing");
  const malformed = createSnapshotObstacleSource({ path: "/any", readFileFn: async () => "not json" });
  assert.equal((await malformed.read()).reason, "lidar_snapshot_unreadable");
  const clear = createSnapshotObstacleSource({ path: "/any", now: () => NOW, readFileFn: async () => JSON.stringify(snapshot()) });
  assert.equal((await clear.read()).reason, "clear");
  assert.equal(clear.kind, "snapshot");
  assert.equal((await lidarGate({ snapshot: "/nonexistent/obstacles.json" })).reason, "lidar_snapshot_missing", "the old entry point still works");
});

test("fixture and disabled sources need no filesystem", async () => {
  assert.deepEqual(sectorVerdict({ center: 5, left: 5, right: 5 }).reason, "clear");
  const near = createFixtureObstacleSource({ center: 0.4, left: 5, right: 5 });
  assert.equal(near.kind, "fixture");
  assert.equal((await near.read()).reason, "lidar_center_near");
  let distance = 5;
  const live = createFixtureObstacleSource(() => ({ center: distance, left: 5, right: 5 }));
  assert.equal((await live.read()).blocked, false);
  distance = 0.2;
  assert.equal((await live.read()).blocked, true);
  assert.deepEqual(await createDisabledObstacleSource().read(), { usable: true, blocked: false, reason: "lidar_disabled" });
  assert.deepEqual(parseFixtureLidar("5,4,3"), { center: 5, left: 4, right: 3 });
  assert.throws(() => parseFixtureLidar("5,4"), /center,left,right/u);
});

test("the fixture camera cycles frames and answers depth-only re-reads with the frame last served", async () => {
  const frames = parseFixtureDepth("2.5,2.5,2.5;0.3,2.5,2.5");
  assert.deepEqual(frames.map(f => f.center_m), [2.5, 0.3]);
  assert.equal(frames[1].front_min_m, 0.3);
  const camera = createFixtureCamera({ frames, now: () => 7000 });
  assert.equal(camera.kind, "fixture");
  const first = await camera.capture({ maxWidth: 320 });
  assert.equal(typeof first.imageB64, "string");
  assert.equal(first.mimeType, "image/png");
  assert.equal(first.depth.center_m, 2.5);
  assert.equal(first.capturedAt, 7);
  const reread = await camera.capture({ depthOnly: true });
  assert.equal(reread.imageB64, undefined);
  assert.equal(reread.depth.center_m, 2.5, "a live re-read sees the frame the model saw");
  assert.equal((await camera.capture({})).depth.center_m, 0.3);
  assert.equal((await camera.capture({})).depth.center_m, 2.5, "frames cycle");
  assert.equal((await camera.capture({ includeDepth: false, needDepth: false })).depth, null);
  assert.throws(() => parseFixtureDepth("1,2"), /center,left,right/u);
});

test("the executor consumes an obstacle source and refuses on lidar without touching the filesystem", async () => {
  const calls = [];
  const actuator = { kind: "test", async status() { return { ok: true, ready: true, reasons: [] }; }, async stop() { calls.push("stop"); return { ok: true }; }, async move() { calls.push("move"); return { ok: true }; }, async arm() { return { ok: true }; } };
  const config = { executeEnabled: true, useLidar: true, depthNearCenterM: 0.55, depthNearSideM: 0.35, maxPulseS: 2 };
  const blocked = createExecutor({ actuator, config, obstacles: createFixtureObstacleSource({ center: 0.4, left: 5, right: 5 }) });
  const refused = await blocked.execute({ token: "W", frame: { depth: { center_m: 2, left_m: 2, right_m: 2 } }, execute: true });
  assert.equal(refused.executed, false);
  assert.deepEqual(refused.gate.reasons, ["lidar:lidar_center_near"]);
  assert.deepEqual(calls, ["stop"]);
  assert.equal(blocked.obstacles.kind, "fixture");
  const clear = createExecutor({ actuator, config, obstacles: createFixtureObstacleSource({ center: 5, left: 5, right: 5 }) });
  assert.equal((await clear.execute({ token: "W", frame: { depth: { center_m: 2, left_m: 2, right_m: 2 } }, execute: true })).executed, true);
});
