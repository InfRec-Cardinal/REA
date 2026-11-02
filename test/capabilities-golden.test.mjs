import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import { goldenAgent } from "./fixtures/golden-agent.mjs";

test("the default profile's capabilities blocks match the recorded golden", async () => {
  const golden = JSON.parse(await readFile(new URL("./fixtures/capabilities-golden.json", import.meta.url), "utf8"));
  const actual = goldenAgent().capabilities();
  assert.deepEqual(actual.tokens, golden.tokens);
  assert.deepEqual(actual.token_map, golden.token_map);
  assert.deepEqual(actual.manipulation, golden.manipulation);
  assert.deepEqual(actual.gate, golden.gate);
  for (const name of Object.keys(golden.engines)) assert.deepEqual(actual.engines[name], golden.engines[name], name);
});
