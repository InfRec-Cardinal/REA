// Obstacle sources: where the gate's lidar summary comes from.
//   snapshot  the Livox producer's JSON file (today's behaviour)
//   fixture   fixed or computed sector distances, for tests and the offline service demo
//   disabled  the summary the gate sees when lidar is switched off
// Every source answers read() with the same shape the gate consumed before: { usable, blocked,
// reason, age, center, front, left, right }.
import { readFile } from "node:fs/promises";

import { G1_PROFILE } from "../profiles/g1.mjs";

function ringMin(ring, lo, hi) {
  let best = -1;
  const n = ring.length;
  for (let i = 0; i < n; i += 1) {
    const value = ring[i];
    if (value == null || value < 0) continue;
    const centre = -180 + (i + 0.5) * (360 / n);
    if (centre >= lo && centre <= hi && (best < 0 || value < best)) best = value;
  }
  return best;
}

/** Threshold rules over sector distances (metres; -1 = no return). */
export function sectorVerdict({ center, front = center, left, right }, { profile = G1_PROFILE, age = 0, nearCenterM, nearSideM } = {}) {
  const cfg = profile.sensing?.lidar ?? {};
  const nc = nearCenterM ?? cfg.nearCenterM;
  const ns = nearSideM ?? cfg.nearSideM;
  let reason = "clear";
  if (center >= 0 && center < nc) reason = "lidar_center_near";
  else if (left >= 0 && left < ns) reason = "lidar_left_near";
  else if (right >= 0 && right < ns) reason = "lidar_right_near";
  return { usable: true, blocked: reason !== "clear", reason, age, center, front, left, right };
}

/** The gate's lidar rules over one parsed snapshot: staleness, starvation, then sector distances. */
export function evaluateLidar(snap, { now = Date.now() / 1000, profile = G1_PROFILE, ...overrides } = {}) {
  if (!snap || typeof snap !== "object") return { usable: false, blocked: true, reason: "lidar_snapshot_unreadable" };
  const cfg = { ...profile.sensing.lidar, ...overrides };
  const age = now - Number(snap.unix_s ?? 0);
  const ring = (snap.ring_m ?? []).map(Number);
  if (!ring.length) return { usable: false, blocked: true, reason: "lidar_ring_empty", age };
  if (age > cfg.maxAgeS) return { usable: false, blocked: true, reason: "lidar_stale", age };
  if (Number(snap.points_per_s ?? 0) < cfg.minPointsPerS) return { usable: false, blocked: true, reason: "lidar_stream_starved", age };
  const sectors = { center: ringMin(ring, ...cfg.center), front: ringMin(ring, ...cfg.front), left: ringMin(ring, ...cfg.left), right: ringMin(ring, ...cfg.right) };
  return sectorVerdict(sectors, { profile, age, nearCenterM: cfg.nearCenterM, nearSideM: cfg.nearSideM });
}

export function createSnapshotObstacleSource({ path, profile = G1_PROFILE, now = () => Date.now() / 1000, readFileFn = readFile } = {}) {
  return {
    kind: "snapshot",
    path,
    async read(overrides = {}) {
      let snap;
      try {
        snap = JSON.parse(await readFileFn(path, "utf8"));
      } catch (error) {
        return { usable: false, blocked: true, reason: `lidar_snapshot_${error?.code === "ENOENT" ? "missing" : "unreadable"}` };
      }
      return evaluateLidar(snap, { now: now(), profile, ...overrides });
    },
  };
}

export function createFixtureObstacleSource(sectors, { profile = G1_PROFILE } = {}) {
  const current = typeof sectors === "function" ? sectors : () => sectors;
  return { kind: "fixture", async read(overrides = {}) { return sectorVerdict(current(), { profile, ...overrides }); } };
}

export function createDisabledObstacleSource() {
  return { kind: "disabled", async read() { return { usable: true, blocked: false, reason: "lidar_disabled" }; } };
}

/** `G1_FIXTURE_LIDAR="center,left,right"` in metres. */
export function parseFixtureLidar(text) {
  const parts = String(text ?? "").split(",").map(part => Number(part.trim()));
  if (parts.length !== 3 || parts.some(value => !Number.isFinite(value))) throw new Error("G1_FIXTURE_LIDAR wants center,left,right in metres");
  const [center, left, right] = parts;
  return { center, left, right };
}
