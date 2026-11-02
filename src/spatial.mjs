// Spatial memory: where the robot has been, by dead reckoning, and what happened there.
//
// This robot exposes no odometry to the harness, so the pose is integrated from the pulses that were
// actually dispatched (vx, vy, yaw, duration in the body frame) and snapped to the IMU yaw when the
// actuator reports one. It is an estimate that drifts: the walking controller under-delivers small
// pulses, the yaw trim on W is a correction not a measurement, and nothing closes the loop. So every
// consumer is told the confidence, the frame is the session's start pose, and nothing here is used
// as safety evidence — only as memory: "this spot, facing this way, has been visited; W was refused
// here", which is what a loop looks like before the token history shows it.
//
// Cells are 0.35 m squares x 45° heading sectors. A *stay* is the run of visits between departures,
// where a departure is a dispatched pulse that moved the pose into another cell: the G1's real Q (12°)
// or a 500 ms W (0.15 m) stays in the cell, so "refused W, then one turn" is one stay, not a revisit.
// The heading sector has hysteresis so IMU jitter across a border never opens a new stay, and the
// sectors wrap so +180° and -180° are one cell.
//
// Nothing in this module can move the robot.

const TWO_PI = Math.PI * 2;

export function wrapAngle(radians) {
  let value = radians % TWO_PI;
  if (value > Math.PI) value -= TWO_PI;
  if (value <= -Math.PI) value += TWO_PI;
  return value;
}

function round(value, digits = 3) {
  return Number(value.toFixed(digits));
}

/** Integrate one body-frame pulse from a pose. Midpoint heading keeps a turning pulse honest. */
export function integratePulse(pose, motion) {
  const vx = Number(motion?.vx) || 0;
  const vy = Number(motion?.vy) || 0;
  const yawRate = Number(motion?.yaw) || 0;
  const duration = Number(motion?.duration) || 0;
  if (duration <= 0) return { ...pose };
  const mid = pose.heading + (yawRate * duration) / 2;
  return {
    x: pose.x + (vx * Math.cos(mid) - vy * Math.sin(mid)) * duration,
    y: pose.y + (vx * Math.sin(mid) + vy * Math.cos(mid)) * duration,
    heading: wrapAngle(pose.heading + yawRate * duration),
  };
}

export function createSpatialMemory({ cellM = 0.35, sectorDeg = 45, sectorHysteresisDeg = 10, maxVisits = 256, now = () => Date.now() } = {}) {
  const pose = { x: 0, y: 0, heading: 0 };
  const visits = []; // { seq, cell, stayId, departures, x, y, heading, token, executed, refused, reasons, outcome, centerM, at }
  const sectors = Math.max(1, Math.round(360 / sectorDeg));
  let travelled = 0;
  let turned = 0;
  let moves = 0; // dispatched pulses that changed the pose
  let departures = 0; // dispatched pulses that moved the pose into another cell
  let stays = 0; // a stay is one or more consecutive visits with no departure in between
  let lastSector = null; // the sector of the last visit: the hysteresis reference
  let imuOffset = null; // imu yaw at session start, so the session frame starts at heading 0
  let imuSnaps = 0;

  const toDeg = radians => (radians * 180) / Math.PI;
  const foldSector = raw => ((raw % sectors) + sectors) % sectors;

  /**
   * Heading sector with hysteresis: the previous visit's sector is kept until the heading is more than
   * half a sector plus the margin away from that sector's centre. Jitter on a border is one sector.
   */
  function sectorOf(heading, reference = lastSector) {
    const raw = foldSector(Math.round(toDeg(heading) / sectorDeg));
    if (reference === null || raw === reference) return raw;
    const centre = (reference * sectorDeg * Math.PI) / 180;
    const away = Math.abs(toDeg(wrapAngle(heading - centre)));
    return away <= sectorDeg / 2 + sectorHysteresisDeg ? reference : raw;
  }

  function cellOf(p, reference = lastSector) {
    const cx = Math.round(p.x / cellM);
    const cy = Math.round(p.y / cellM);
    return `${cx},${cy},${sectorOf(p.heading, reference)}`;
  }

  /** Snap the heading to the IMU when the actuator reports one; drift in x/y remains. */
  function observeHeading(imuYaw) {
    if (!Number.isFinite(imuYaw)) return false;
    if (imuOffset === null) {
      imuOffset = imuYaw - pose.heading;
      return false;
    }
    pose.heading = wrapAngle(imuYaw - imuOffset);
    imuSnaps += 1;
    return true;
  }

  /** Advance the pose by a pulse that was actually dispatched. */
  function advance(motion) {
    const before = { ...pose };
    const after = integratePulse(pose, motion);
    const dx = after.x - before.x;
    const dy = after.y - before.y;
    const moved = Math.hypot(dx, dy) > 1e-6 || Math.abs(wrapAngle(after.heading - before.heading)) > 1e-6;
    travelled += Math.hypot(dx, dy);
    turned += Math.abs(wrapAngle(after.heading - before.heading));
    const cellBefore = cellOf(before);
    pose.x = after.x;
    pose.y = after.y;
    pose.heading = after.heading;
    if (moved) moves += 1;
    if (cellOf(pose) !== cellBefore) departures += 1;
    return { ...pose };
  }

  /** The stay a mark right now would belong to: the last visit's, if no pulse has left its cell since. */
  function currentStay() {
    const last = visits.at(-1);
    return last && last.departures === departures ? last.stayId : null;
  }

  /**
   * Earlier *stays* at the current cell (same place, roughly the same way round). Steps decided from
   * one spot, including in-place turns and short pulses that stay in the cell, are one stay, not a
   * revisit: a refused W followed by a turn is normal; coming back after walking away is what counts.
   */
  function revisit(p = pose, { within = null } = {}) {
    const cell = cellOf(p);
    const stay = currentStay();
    const pool = within ? visits.slice(-within) : visits;
    const earlier = pool.filter(visit => visit.cell === cell && visit.stayId !== stay);
    return {
      cell,
      count: new Set(earlier.map(visit => visit.stayId)).size,
      refused: earlier.filter(visit => visit.refused).length,
      firstSeq: earlier[0]?.seq ?? null,
      lastSeq: earlier.at(-1)?.seq ?? null,
      tokens: earlier.map(visit => visit.token),
    };
  }

  /** Remember what happened at the current pose (call before advance() for the step's own pulse). */
  function mark({ seq, token, executed = false, refused = false, reasons = [], outcome = null, centerM = null } = {}) {
    const here = revisit();
    const stayId = currentStay() ?? ++stays;
    lastSector = Number(here.cell.split(",")[2]);
    const visit = {
      seq,
      cell: here.cell,
      stayId,
      departures,
      x: round(pose.x),
      y: round(pose.y),
      heading: round(pose.heading),
      token,
      executed,
      refused,
      reasons: [...reasons],
      outcome,
      centerM: Number.isFinite(centerM) ? round(centerM, 2) : null,
      at: now(),
    };
    visits.push(visit);
    while (visits.length > maxVisits) visits.shift();
    return { visit, revisits: here.count, refusedHere: here.refused };
  }

  /**
   * Confidence in the estimate: 1 at the start, falling with distance and turning, recovering a
   * little each time the IMU snaps the heading. Purely indicative.
   */
  function confidence() {
    const drift = 0.12 * travelled + 0.08 * turned;
    const value = 1 / (1 + drift) + Math.min(0.2, 0.02 * imuSnaps);
    return round(Math.max(0, Math.min(1, value)), 2);
  }

  function signals() {
    const here = revisit();
    return {
      revisits: here.count,
      recentRevisits: revisit(pose, { within: 12 }).count,
      refusedHere: here.refused,
      travelledM: round(travelled, 2),
      turnedRad: round(turned, 2),
      confidence: confidence(),
    };
  }

  /** One line for a prompt: where the robot thinks it is, and whether it has been here before. */
  function summary({ maxChars = 200 } = {}) {
    const deg = Math.round(toDeg(pose.heading));
    const parts = [`est. pose ${pose.x >= 0 ? "+" : ""}${pose.x.toFixed(1)} m fwd, ${pose.y >= 0 ? "+" : ""}${pose.y.toFixed(1)} m left, heading ${deg >= 0 ? "+" : ""}${deg}°`];
    parts.push(`travelled ${travelled.toFixed(1)} m (dead reckoning, confidence ${confidence()})`);
    const here = revisit();
    if (here.count > 0) {
      const refused = here.refused > 0 ? `, ${here.refused} refused here` : "";
      parts.push(`this spot seen ${here.count}x before (#${here.firstSeq}${here.count > 1 ? `..#${here.lastSeq}` : ""}${refused})`);
    }
    return parts.join("; ").slice(0, maxChars);
  }

  function snapshot() {
    return {
      pose: { x: round(pose.x), y: round(pose.y), heading_rad: round(pose.heading), heading_deg: Math.round(toDeg(pose.heading)) },
      travelled_m: round(travelled, 2),
      turned_rad: round(turned, 2),
      confidence: confidence(),
      imu_snaps: imuSnaps,
      visits: visits.length,
      moves,
      departures,
      stays,
      here: revisit(),
      cell_m: cellM,
      sector_deg: sectorDeg,
    };
  }

  return {
    advance,
    observeHeading,
    mark,
    revisit,
    signals,
    summary,
    snapshot,
    cellOf: p => cellOf(p ?? pose),
    pose: () => ({ ...pose }),
    trail: ({ max = 12 } = {}) => visits.slice(-max).map(visit => ({ seq: visit.seq, x: visit.x, y: visit.y, heading: visit.heading, token: visit.token, refused: visit.refused, outcome: visit.outcome })),
  };
}
