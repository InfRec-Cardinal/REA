import { G1_PROFILE } from "./profiles/g1.mjs";
import { createSnapshotObstacleSource } from "./sensors/obstacles.mjs";

// The G1 thresholds, kept as named exports for existing callers; the gate itself reads them from the
// profile it is handed. Sector windows are azimuth windows, not radii (the MID-360 self-occludes).
export const LIDAR_DEFAULTS = G1_PROFILE.sensing.lidar;
export const DEPTH_DEFAULTS = Object.freeze({ nearCenterM: G1_PROFILE.sensing.depth.nearCenterM, nearSideM: G1_PROFILE.sensing.depth.nearSideM });
export const DEPTH_FLOOR_CENTER_M = G1_PROFILE.sensing.depth.floorCenterM;
export const DEPTH_FLOOR_SIDE_M = G1_PROFILE.sensing.depth.floorSideM;

/** Compatibility entry point: read the lidar snapshot file once and evaluate it. */
export function lidarGate({ snapshot = LIDAR_DEFAULTS.snapshot, now = Date.now() / 1000, ...overrides } = {}) {
  return createSnapshotObstacleSource({ path: snapshot, now: () => now }).read(overrides);
}

/**
 * Depth veto from the RealSense summary. Thresholds may be lowered by an explicit opt-in on the
 * request (contact tasks), never below the profile's hard floors.
 */
export function depthGate(depth, { nearCenterM, nearSideM, profile = G1_PROFILE } = {}) {
  const limitsOf = profile.sensing?.depth;
  if (!limitsOf) return { usable: false, blocked: true, reason: "depth_missing", near_center_m: null, near_side_m: null };
  const nc = Math.max(limitsOf.floorCenterM, Number(nearCenterM) || limitsOf.nearCenterM);
  const ns = Math.max(limitsOf.floorSideM, Number(nearSideM) || limitsOf.nearSideM);
  const limits = { near_center_m: nc, near_side_m: ns };
  if (!depth || depth.center_m == null) return { usable: false, blocked: true, reason: "depth_missing", ...limits };
  if (depth.center_m < nc) return { usable: true, blocked: true, reason: "depth_center_near", ...limits, ...depth };
  if (depth.left_m != null && depth.left_m < ns) return { usable: true, blocked: true, reason: "depth_left_near", ...limits, ...depth };
  if (depth.right_m != null && depth.right_m < ns) return { usable: true, blocked: true, reason: "depth_right_near", ...limits, ...depth };
  return { usable: true, blocked: false, reason: "clear", ...limits, ...depth };
}

/**
 * Fused verdict. Which vetoes apply is the token's `veto` record in the profile:
 *   veto.depth  center | left | right | null   the depth window that must be clear
 *   veto.lidar  center | left | right | usable | null   the lidar sector that must be clear, or
 *               merely a usable scan
 * The stop token is always allowed; arm gestures check state and the holding interlock only.
 * `depth` must be a depthGate() result: anything else fails closed as depth_missing.
 */
export function verdict({ token, state, depth, lidar, useLidar = true, armHeld = null, profile = G1_PROFILE }) {
  const record = profile.actionOrStop(token);
  const stop = profile.roles.stop;
  if (record.token === stop) return { allowed: true, reasons: [], token: stop, kind: record.kind };
  const reasons = [];
  if (!state?.ready) reasons.push(`state:${(state?.reasons ?? ["unknown"]).join(",")}`);

  // Arm interlock: the arm service rejects a second grab while one is held (error 7401), so the gate refuses it first.
  if (record.kind === "arm_primitive") {
    const holding = profile.arms?.holding ?? [];
    if (holding.includes(record.token) && armHeld !== null && armHeld !== record.token) reasons.push(`arm:already_holding_${armHeld}`);
    return { allowed: reasons.length === 0, reasons, token: record.token, kind: record.kind };
  }

  const veto = record.veto ?? { depth: null, lidar: null };
  const lidarConfigured = Boolean(profile.sensing?.lidar);
  const needLidar = useLidar && lidarConfigured && veto.lidar !== null;
  if (needLidar && lidar?.usable !== true) reasons.push(`lidar:${lidar?.reason ?? "unavailable"}`);
  const depthGated = Boolean(depth) && typeof depth.usable === "boolean";

  if (veto.depth === "center") {
    if (!depthGated) reasons.push("depth:depth_missing");
    else if (depth.blocked) reasons.push(`depth:${depth.reason}`);
  } else if (veto.depth === "left" || veto.depth === "right") {
    const side = veto.depth;
    const depthSide = depth?.[`${side}_m`];
    if (!depthGated) reasons.push("depth:depth_missing");
    else if (depth.blocked && depth.reason === `depth_${side}_near`) reasons.push(`depth:${depth.reason}`);
    else if (!depth.usable) reasons.push(`depth:${depth.reason}`);
    else if (depthSide != null && depthSide < depth.near_side_m) reasons.push(`depth:depth_${side}_near`);
  }

  // In-process lidar gate: reading the snapshot here costs 4-10 ms; shelling out cost 2.3 s per decision.
  if (needLidar) {
    if (veto.lidar === "center" && lidar?.blocked) reasons.push(`lidar:${lidar.reason}`);
    else if ((veto.lidar === "left" || veto.lidar === "right") && lidar?.usable
      && lidar[veto.lidar] >= 0 && lidar[veto.lidar] < profile.sensing.lidar.nearSideM) reasons.push(`lidar:lidar_${veto.lidar}_near`);
  }
  return { allowed: reasons.length === 0, reasons, token: record.token, kind: record.kind };
}

/** Tracks which arm pose is held so the interlock has something to check. */
export function nextArmHeld(current, token, profile = G1_PROFILE) {
  if (!profile.arms) return current;
  if (token === profile.arms.release) return null;
  if (profile.arms.holding.includes(token)) return token;
  return current;
}
