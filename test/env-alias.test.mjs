import assert from "node:assert/strict";
import { test } from "node:test";

import { ENV_ALIASES, deprecatedEnvNotices, readEnv } from "../src/env.mjs";

test("REA_* variables win, FAST_* are read as aliases, and the notices name each deprecated variable once", () => {
  assert.equal(readEnv({ REA_MODEL: "rea", FAST_MODEL: "fast" }, "REA_MODEL"), "rea");
  assert.equal(readEnv({ FAST_MODEL: "fast" }, "REA_MODEL"), "fast");
  assert.equal(readEnv({ REA_MODEL: "", FAST_MODEL: "fast" }, "REA_MODEL"), "fast", "an empty REA_* is unset");
  assert.equal(readEnv({}, "REA_MODEL"), undefined);
  assert.equal(readEnv({ FAST_SOMETHING: "x" }, "REA_SOMETHING"), undefined, "only the five public names have aliases");
  assert.deepEqual(Object.keys(ENV_ALIASES), ["REA_MODEL", "REA_BASE_URL", "REA_API_KEY", "REA_ENV_FILE", "REA_POLICY_MODE"]);
  assert.deepEqual(deprecatedEnvNotices({ FAST_MODEL: "m", FAST_BASE_URL: "u", REA_API_KEY: "k", FAST_API_KEY: "old" }), [
    "REA_MODEL is not set; using the deprecated FAST_MODEL (rename it in .env)",
    "REA_BASE_URL is not set; using the deprecated FAST_BASE_URL (rename it in .env)",
  ]);
  assert.deepEqual(deprecatedEnvNotices({}), []);
});
