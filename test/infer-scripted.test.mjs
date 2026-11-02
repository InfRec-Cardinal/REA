import assert from "node:assert/strict";
import { test } from "node:test";

import { createScriptedInfer, defaultScriptedTokens } from "../src/infer-scripted.mjs";
import HOLONOMIC from "./fixtures/profiles/holonomic.mjs";
import { G1_PROFILE } from "../src/profiles/g1.mjs";

const FRAME = { imageB64: "aGk=" };

test("the scripted engine replays its tokens, repeats the last one and never needs a provider", async () => {
  const infer = createScriptedInfer({ tokens: ["W", "Q"] });
  assert.equal(infer.engine, "scripted");
  assert.equal((await infer("any-model", FRAME, {})).token, "W");
  assert.equal((await infer("any-model", FRAME, {})).token, "Q");
  const third = await infer("any-model", FRAME, {});
  assert.equal(third.token, "Q", "the last token repeats");
  assert.equal(third.source, "scripted");
  assert.equal(typeof third.tokenReadyS, "number");
  infer.reset();
  assert.equal((await infer("any-model", FRAME, {})).token, "W");
  await assert.rejects(infer("any-model", { imageB64: "" }, {}), error => error.errorType === "config_error");
  assert.throws(() => createScriptedInfer({ tokens: ["Z"] }), /scripted tokens/u);
  assert.equal((await createScriptedInfer({ tokens: ["f"], profile: HOLONOMIC })("m", FRAME, {})).token, "F", "tokens are validated against the profile");
});

test("the default scripted token follows the profile's forward role, so a profile without W still starts", () => {
  assert.deepEqual(defaultScriptedTokens(G1_PROFILE), ["W"]);
  assert.deepEqual(defaultScriptedTokens(HOLONOMIC), ["F"], "the holonomic base has no W at all");
  // The point of the default: constructing the engine must not throw under a foreign profile.
  assert.equal(createScriptedInfer({ tokens: defaultScriptedTokens(HOLONOMIC), profile: HOLONOMIC }).engine, "scripted");
  const stopOnly = { roles: { stop: "H" }, TOKENS: ["H"] };
  assert.deepEqual(defaultScriptedTokens(stopOnly), ["H"], "a profile with no forward role falls back to its stop");
});
