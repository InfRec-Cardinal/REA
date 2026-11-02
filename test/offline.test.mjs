import assert from "node:assert/strict";
import test from "node:test";

import { runOfflineDemo } from "../examples/offline.mjs";

test("offline quickstart carries streamed tokens through sessions and the real gate", async () => {
  const results = await runOfflineDemo();
  assert.deepEqual(results.map(result => result.seq), [1, 2]);
  assert.deepEqual(results.map(result => result.token), ["W", "W"]);
  assert.deepEqual(results.map(result => result.gate_allowed), [true, false]);
  assert.ok(results[1].reasons.some(reason => reason.includes("depth_center_near")));
  assert.ok(results.every(result => result.preview && !result.executed));
});
