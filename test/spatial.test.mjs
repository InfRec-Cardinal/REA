import assert from "node:assert/strict";
import { test } from "node:test";

import { motionFor } from "../src/actions.mjs";
import { createSpatialMemory, integratePulse, wrapAngle } from "../src/spatial.mjs";

const close = (a, b, eps = 1e-3) => Math.abs(a - b) < eps;

test("pulses integrate in the body frame with a midpoint heading", () => {
  const start = { x: 0, y: 0, heading: 0 };
  const forward = integratePulse(start, { vx: 0.3, vy: 0, yaw: 0, duration: 2 });
  assert.ok(close(forward.x, 0.6) && close(forward.y, 0) && close(forward.heading, 0));
  const turned = integratePulse(start, { vx: 0, vy: 0, yaw: Math.PI / 2, duration: 1 });
  assert.ok(close(turned.heading, Math.PI / 2));
  const afterTurn = integratePulse(turned, { vx: 0.3, vy: 0, yaw: 0, duration: 1 });
  assert.ok(close(afterTurn.x, 0) && close(afterTurn.y, 0.3), "facing +90°, forward is +y (left)");
  const strafe = integratePulse(start, { vx: 0, vy: 0.3, yaw: 0, duration: 1 });
  assert.ok(close(strafe.x, 0) && close(strafe.y, 0.3), "vy is left");
  assert.deepEqual(integratePulse(start, { vx: 1, vy: 1, yaw: 1, duration: 0 }), start, "no duration, no motion");
  assert.ok(close(wrapAngle(3 * Math.PI), Math.PI));
  assert.ok(close(wrapAngle(-3 * Math.PI), Math.PI));
});

test("the session trail remembers what happened where and detects a revisit", () => {
  let clock = 0;
  const spatial = createSpatialMemory({ cellM: 0.35, sectorDeg: 45, now: () => (clock += 1) });
  assert.deepEqual(spatial.revisit().count, 0);
  // step 1: W refused here
  let marked = spatial.mark({ seq: 1, token: "W", refused: true, reasons: ["depth:depth_center_near"], outcome: "refused", centerM: 0.4 });
  assert.equal(marked.revisits, 0);
  // step 2: turn left 90° (executed)
  spatial.mark({ seq: 2, token: "Q", executed: true, outcome: "progress" });
  spatial.advance({ vx: 0, vy: 0, yaw: Math.PI / 2, duration: 1 });
  // step 3: forward 1 m
  spatial.mark({ seq: 3, token: "W", executed: true, outcome: "progress", centerM: 2.0 });
  spatial.advance({ vx: 0.5, vy: 0, yaw: 0, duration: 2 });
  // step 4: turn right 90°, back to the original heading but 1 m to the left
  spatial.mark({ seq: 4, token: "E", executed: true });
  spatial.advance({ vx: 0, vy: 0, yaw: -Math.PI / 2, duration: 1 });
  assert.equal(spatial.revisit().count, 0, "1 m left of the start is a different cell");
  // step 5: strafe right 1 m => back where we started, same heading
  spatial.mark({ seq: 5, token: "D", executed: true });
  spatial.advance({ vx: 0, vy: -0.5, yaw: 0, duration: 2 });
  const here = spatial.revisit();
  assert.equal(here.count, 1, "back at the start cell, facing the same way");
  assert.equal(here.refused, 1);
  assert.equal(here.firstSeq, 1);
  const signals = spatial.signals();
  assert.equal(signals.revisits, 1);
  assert.equal(signals.refusedHere, 1);
  assert.ok(signals.travelledM > 1.9 && signals.travelledM < 2.1);
  assert.ok(signals.confidence < 1 && signals.confidence > 0.5);
  const summary = spatial.summary();
  assert.match(summary, /est\. pose/u);
  assert.match(summary, /this spot seen 1x before \(#1, 1 refused here\)/u);
  assert.ok(summary.length <= 200);
  marked = spatial.mark({ seq: 6, token: "W", refused: true, outcome: "refused" });
  assert.equal(marked.revisits, 1);
  assert.equal(marked.refusedHere, 1);
  assert.equal(spatial.snapshot().visits, 6);
  assert.equal(spatial.trail({ max: 2 }).length, 2);
  assert.equal(spatial.trail({ max: 2 })[1].seq, 6);
});

test("the IMU yaw snaps the heading in the session frame; the first reading defines the frame", () => {
  const spatial = createSpatialMemory();
  assert.equal(spatial.observeHeading(1.0), false, "the first reading only sets the offset");
  assert.ok(close(spatial.pose().heading, 0));
  spatial.advance({ vx: 0, vy: 0, yaw: 0.35, duration: 0.6 }); // dead reckoning says +0.21 rad
  assert.ok(close(spatial.pose().heading, 0.21));
  assert.equal(spatial.observeHeading(1.0 + 0.15), true, "the IMU says the turn under-delivered");
  assert.ok(close(spatial.pose().heading, 0.15));
  assert.equal(spatial.observeHeading(Number.NaN), false);
  assert.equal(spatial.snapshot().imu_snaps, 1);
});

test("the agent's own motion records integrate as expected", () => {
  const spatial = createSpatialMemory();
  spatial.mark({ seq: 1, token: "W", executed: true });
  spatial.advance(motionFor("W", null, { maxDurationS: 2 }));
  const pose = spatial.pose();
  assert.ok(pose.x > 0.55 && pose.x < 0.61, `W walks ~0.6 m (${pose.x})`);
  assert.ok(pose.heading < 0, "the yaw trim turns slightly right");
});

test("in-place turns and short pulses that stay in the cell are one stay: a refused W then one real Q is not a revisit", () => {
  const spatial = createSpatialMemory({ cellM: 0.35, sectorDeg: 45 });
  spatial.mark({ seq: 1, token: "W", refused: true, reasons: ["depth:depth_center_near"], outcome: "refused", centerM: 0.4 });
  // the G1's real Q: 0.35 rad/s for 0.6 s = 12 degrees, well inside a 45 degree sector
  spatial.mark({ seq: 2, token: "Q", executed: true, outcome: "progress" });
  spatial.advance({ vx: 0, vy: 0, yaw: 0.35, duration: 0.6 });
  assert.deepEqual([spatial.signals().revisits, spatial.signals().refusedHere], [0, 0], "one turn in place does not leave the cell");
  // a 500 ms W: 0.3 m/s * 0.5 s = 0.15 m, inside a 0.35 m cell
  const marked = spatial.mark({ seq: 3, token: "W", executed: true, outcome: "progress" });
  assert.equal(marked.revisits, 0);
  spatial.advance({ vx: 0.3, vy: 0, yaw: 0, duration: 0.5 });
  assert.equal(spatial.signals().revisits, 0, "a short pulse does not leave the cell either");
  assert.equal(spatial.snapshot().stays, 1);
  assert.equal(spatial.snapshot().departures, 0);
  // two more W pulses leave the cell; walking back is the revisit
  spatial.mark({ seq: 4, token: "W", executed: true });
  spatial.advance({ vx: 0.3, vy: 0, yaw: 0, duration: 1.0 });
  assert.equal(spatial.snapshot().departures, 1);
  spatial.mark({ seq: 5, token: "S", executed: true });
  spatial.advance({ vx: -0.3, vy: 0, yaw: 0, duration: 1.5 });
  const back = spatial.revisit();
  assert.equal(back.count, 1, "back at the start cell after walking away");
  assert.equal(back.refused, 1);
});

test("the heading sector has hysteresis and wraps: IMU jitter on a border and +-180 degrees are one cell", () => {
  const spatial = createSpatialMemory({ cellM: 0.35, sectorDeg: 45 });
  spatial.observeHeading(0); // defines the frame
  spatial.mark({ seq: 1, token: "W", refused: true });
  // the IMU jitters across the 22.5 degree border with no pulse
  spatial.observeHeading((22.6 * Math.PI) / 180);
  const jitter = spatial.mark({ seq: 2, token: "Q" });
  assert.equal(jitter.revisits, 0);
  assert.equal(jitter.visit.cell, "0,0,0", "22.6 degrees stays in the sector the last visit used");
  spatial.observeHeading((22.4 * Math.PI) / 180);
  assert.equal(spatial.mark({ seq: 3, token: "Q" }).revisits, 0);
  assert.equal(spatial.snapshot().stays, 1);
  // a real 40 degree heading is the next sector
  spatial.observeHeading((40 * Math.PI) / 180);
  assert.equal(spatial.cellOf(), "0,0,1");
  const wrap = createSpatialMemory({ cellM: 0.35, sectorDeg: 45 });
  assert.equal(wrap.cellOf({ x: 0, y: 0, heading: Math.PI - 0.01 }), wrap.cellOf({ x: 0, y: 0, heading: -Math.PI + 0.01 }), "+180 and -180 are the same sector");
  assert.equal(wrap.cellOf({ x: 0, y: 0, heading: Math.PI - 0.01 }), "0,0,4");
});
