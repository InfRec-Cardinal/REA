import assert from "node:assert/strict";
import { test } from "node:test";

import { createWorkerCamera } from "../src/camera.mjs";

test("a depth-only capture asks the worker to skip the JPEG and returns the depth summary alone", async () => {
  const requests = [];
  const worker = { request: async payload => { requests.push(payload); return payload.depth_only ? { ok: true, depth: { center_m: 1.2 }, captured_at: 42 } : { ok: true, image_b64: "aGk=", depth: { center_m: 1.2 }, scene: { dhash: "a1b2c3d4e5f60718", brightness: 0.4, left_right: 0.05 } }; } };
  const camera = createWorkerCamera({ worker });
  const live = await camera.capture({ depthOnly: true, maxWidth: 320 });
  assert.deepEqual(requests[0], { op: "capture", max_width: 320, quality: 40, filter: "", no_depth: false, depth_only: true, no_scene: true }, "a live re-read wants no scene hash");
  assert.deepEqual(live.depth, { center_m: 1.2 });
  assert.equal(live.capturedAt, 42);
  assert.equal(live.imageB64, undefined);
  const full = await camera.capture({});
  assert.equal(requests[1].depth_only, false);
  assert.equal(requests[1].no_scene, false);
  assert.equal(full.imageB64, "aGk=");
  assert.deepEqual(full.scene, { dhash: "a1b2c3d4e5f60718", brightness: 0.4, left_right: 0.05 }, "the scene signature rides along with the frame");
  assert.equal(live.scene, null);
  const empty = createWorkerCamera({ worker: { request: async () => ({ ok: true }) } });
  await assert.rejects(empty.capture({ depthOnly: true }), /no depth summary/u);
  await assert.rejects(empty.capture({}), /no JPEG image/u);
});
