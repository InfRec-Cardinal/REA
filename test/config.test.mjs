import assert from "node:assert/strict";
import test from "node:test";

import { loadConfig, senseKindOf } from "../src/config.mjs";
import { DEMO_ENV } from "../examples/service.mjs";

test("Prime Agent RPC is the default harness with a two-process persistent pool", () => {
  const config = loadConfig({ REA_API_KEY: "configured" });
  assert.equal(config.engine, "prime");
  assert.equal(config.agentHarness, "prime-agent-rpc");
  assert.equal(config.primeAgentVersion, "0.8.0");
  assert.equal(config.primeAgentProvider, "openai-compatible");
  assert.equal(config.primeAgentPoolSize, 2);
  assert.equal(config.primeWarmup, true);
  // No fallback by default: max_tokens=1 is a hard contract that many vision models cannot honour,
  // and a fallback that cannot answer would only fail the readiness probe.
  assert.equal(config.fallbackModel, "");
  assert.equal(config.actuator, "none");
  assert.equal(config.persistentControl, false);
  assert.equal(config.executeEnabled, false);
  assert.match(config.primeAgentBin, /runtime\/prime-agent\/node_modules\/\.bin\/prime-agent$/u);
  assert.match(config.primeAgentConfigDir, /prime-agent-config\/?$/u);
});

test("the token and tool engines remain explicit opt-in alternatives", () => {
  for (const engine of ["token", "tool"]) {
    assert.equal(loadConfig({ G1_ENGINE: engine }).engine, engine);
  }
  assert.throws(() => loadConfig({ G1_ENGINE: "unknown" }), /must be one of prime, token, tool/u);
});

test("no engine gets a fallback model unless one is configured", () => {
  for (const engine of ["prime", "token", "tool"]) {
    assert.equal(loadConfig({ G1_ENGINE: engine }).fallbackModel, "");
  }
});

test("the primary model comes from G1_PRIMARY_MODEL, else REA_MODEL", () => {
  assert.equal(loadConfig({ REA_MODEL: "shared" }).primaryModel, "shared");
  assert.equal(loadConfig({ REA_MODEL: "shared", G1_PRIMARY_MODEL: "g1" }).primaryModel, "g1");
  assert.equal(loadConfig({}).primaryModel, "");
});

test("an explicit fallback is always honoured, whatever the engine", () => {
  assert.equal(loadConfig({ G1_ENGINE: "prime", G1_FALLBACK_MODEL: "custom" }).fallbackModel, "custom");
});

test("the legacy execute alias is parsed as a boolean: a falsy string can never enable actuation", () => {
  for (const value of ["0", "false", "off", "no"]) {
    assert.equal(loadConfig({ G1_DIRECT_EXECUTE_ENABLED: value }).executeEnabled, false, value);
  }
  assert.equal(loadConfig({ G1_DIRECT_EXECUTE_ENABLED: "1" }).executeEnabled, true);
  assert.equal(loadConfig({ G1_DIRECT_EXECUTE_ENABLED: "1", G1_EXECUTE_ENABLED: "0" }).executeEnabled, false, "the current name wins");
  assert.throws(() => loadConfig({ G1_DIRECT_EXECUTE_ENABLED: "maybe" }), /boolean/u);
});

test("safety knobs default to tight, and the version comes from package.json", () => {
  const config = loadConfig({ G1_ENGINE: "token" });
  assert.equal(config.allowRequestLidarOff, false);
  assert.equal(config.gateLiveDepth, true);
  assert.equal(config.agentToken, "");
  assert.match(config.version, /^\d+\.\d+\.\d+$/u);
  assert.equal(config.harnessVersion, config.version, "a non-Prime engine reports this service's version, not Prime's");
  assert.equal(loadConfig({}).harnessVersion, "0.8.0", "the Prime engine reports Prime's version");
  assert.equal(loadConfig({ G1_ALLOW_REQUEST_LIDAR_OFF: "1", G1_GATE_LIVE_DEPTH: "0", G1_AGENT_TOKEN: "s3cret" }).allowRequestLidarOff, true);
  assert.equal(loadConfig({ G1_GATE_LIVE_DEPTH: "0" }).gateLiveDepth, false);
  assert.equal(loadConfig({ G1_AGENT_TOKEN: "s3cret" }).agentToken, "s3cret");
});

test("engine meta drives the provider requirement, output tokens and the scripted defaults; port 0 is legal", () => {
  const scripted = loadConfig({ G1_ENGINE: "scripted", G1_AGENT_PORT: "0" });
  assert.equal(scripted.providerRequired, false);
  assert.equal(scripted.modelOutputTokens, 1);
  assert.equal(scripted.fallbackModel, "", "one-token engines get no fallback");
  assert.deepEqual(scripted.scriptedTokens, ["W"]);
  assert.equal(scripted.port, 0);
  assert.deepEqual(loadConfig({ G1_ENGINE: "scripted", G1_SCRIPTED_TOKENS: "W, Q ,X" }).scriptedTokens, ["W", "Q", "X"]);
  assert.equal(loadConfig({ G1_ENGINE: "token" }).providerRequired, true);
  assert.equal(loadConfig({ G1_ENGINE: "tool" }).modelOutputTokens, 64);
  assert.equal(loadConfig({ G1_ENGINE: "prime" }).providerRequired, false);
});

test("sensing selection: G1_SENSE and G1_OBSTACLES with fixtures, and G1_PERSISTENT_CAPTURE still works", () => {
  assert.equal(senseKindOf(loadConfig({})), "worker");
  assert.equal(senseKindOf(loadConfig({ G1_PERSISTENT_CAPTURE: "0" })), "subprocess");
  assert.equal(senseKindOf(loadConfig({ G1_SENSE: "fixture", G1_PERSISTENT_CAPTURE: "1" })), "fixture", "an explicit G1_SENSE wins");
  assert.throws(() => loadConfig({ G1_SENSE: "camera" }), /G1_SENSE must be one of worker, subprocess, fixture/u);
  const fixture = loadConfig({ G1_SENSE: "fixture", G1_FIXTURE_DEPTH: "2.5,2.5,2.5;0.3,2.5,2.5", G1_OBSTACLES: "fixture", G1_FIXTURE_LIDAR: "4,3,2" });
  assert.deepEqual(fixture.fixtureDepth.map(frame => frame.center_m), [2.5, 0.3]);
  assert.deepEqual(fixture.fixtureLidar, { center: 4, left: 3, right: 2 });
  assert.equal(fixture.obstacles, "fixture");
  assert.equal(loadConfig({}).obstacles, "snapshot");
  assert.deepEqual(loadConfig({}).fixtureLidar, { center: 5, left: 5, right: 5 });
  assert.throws(() => loadConfig({ G1_OBSTACLES: "lidar" }), /G1_OBSTACLES must be one of snapshot, fixture/u);
});

test("fixture sensing is refused with a real actuator and execution enabled", () => {
  // The fixtures answer the gate with constants (2.5 m, 5 m) while a real actuator can move the robot.
  const live = { G1_EXECUTE_ENABLED: "1", G1_ACTUATOR: "daemon", G1_SENSE: "fixture", G1_FIXTURE_DEPTH: "2.5,2.5,2.5" };
  assert.throws(() => loadConfig(live), /synthetic evidence/u);
  assert.throws(() => loadConfig({ ...live, G1_SENSE: undefined, G1_OBSTACLES: "fixture" }), /synthetic evidence/u);
  for (const actuator of ["console", "cerebellum"]) {
    assert.throws(() => loadConfig({ ...live, G1_ACTUATOR: actuator }), /synthetic evidence/u, actuator);
  }
  // A mock actuator cannot move anything, so fixtures are exactly what it is for.
  assert.equal(loadConfig({ ...live, G1_ACTUATOR: "mock" }).executeEnabled, true);
  assert.equal(senseKindOf(loadConfig({ ...live, G1_ACTUATOR: "none" })), "fixture", "no actuator at all is fine too");
  // Preview-only is fine with any actuator: nothing is sent.
  assert.equal(loadConfig({ ...live, G1_EXECUTE_ENABLED: "0" }).actuator, "daemon");
  // And the shipped demo environment must keep loading.
  assert.equal(loadConfig({ ...DEMO_ENV }).engine, "scripted");
});
