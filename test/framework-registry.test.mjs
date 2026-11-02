import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { createEnvironment, createPolicy, createRegistry, listEnvironments, listPolicies } from "../src/framework/registry.mjs";
import { runEpisode } from "../src/framework/runtime.mjs";

const ACTION = { name: "wait", parameters: { type: "object", properties: {}, additionalProperties: false } };
const SNAPSHOT = { state: { ready: true }, actions: [ACTION] };

function workerFixture() {
  const requests = [];
  return {
    requests,
    health: () => ({ running: true }),
    stop() {},
    async request(payload) {
      requests.push(payload);
      return { ok: true, result: payload.op === "configure" ? { actions: [ACTION], metadata: {}, timing: "turn-based" } : { ...SNAPSHOT, observation: SNAPSHOT } };
    },
  };
}

test("registry discovers built-in adapters without starting their runtimes", async () => {
  assert.deepEqual(listEnvironments().map(item => item.type), ["demo", "g1", "libero", "arc-agi3", "gymnasium", "aloha", "lerobot", "openral", "rpent", "robot", "python-robot", "module"]);
  assert.deepEqual(listPolicies().map(item => item.type), ["demo", "scripted", "chat", "routed", "progress", "module", "lingbot-vla", "pi05", "aloha-code"]);
  for (const { type } of listEnvironments()) {
    const environment = await createEnvironment({ type, options: { python: "/not/an/interpreter" } });
    if (type === "arc-agi3") assert.equal(await environment.scorecard({ close: true }), null);
    await environment.close();
  }
});

test("one demo policy runs grid and joint episodes through the shared runtime", async () => {
  const environment = await createEnvironment("demo");
  const policy = await createPolicy("demo");
  for (const task of ["grid", "joints"]) {
    const result = await runEpisode({ environment, policy, task, maxSteps: 3 });
    assert.equal(result.success, true);
    assert.equal(result.steps, 1);
  }
  await environment.close();
});

test("registries isolate extensions and pass the episode context to lazy factories", async () => {
  const registry = createRegistry({ builtins: false });
  let created = 0;
  let closed = 0;
  registry.registerEnvironment("custom", (options, context) => {
    created += 1;
    assert.equal(options.label, "robot");
    assert.equal(context.episode.id, "episode-a");
    assert.equal(context.registry, registry);
    return { id: "custom:robot", timing: "realtime", reset: async () => ({ observation: SNAPSHOT }), step: async () => ({ observation: SNAPSHOT }), close: async () => { closed += 1; } };
  }, { description: "Custom test robot", timing: "turn-based" });
  const spec = { type: "custom", options: { label: "robot" } };
  const unused = await registry.createEnvironment(spec);
  await unused.close();
  assert.equal(created, 0);
  const environment = await registry.createEnvironment(spec, { episode: { id: "episode-a" } });
  assert.equal(created, 0);
  await environment.reset();
  assert.equal(created, 1);
  assert.equal(environment.id, "custom:robot");
  assert.equal(environment.timing, "realtime");
  await environment.close();
  await environment.close();
  assert.equal(closed, 1);
  await assert.rejects(createEnvironment(spec), /Unknown environment type: custom/);
});

test("robot bridges apply source env defaults and explicit options win", async () => {
  const profiles = [
    ["aloha", { ACT_ROOT: "/env/act", ALOHA_ROOT: "/env/aloha" }, { actRoot: "/env/act", alohaRoot: "/env/aloha" }],
    ["openral", { OPENRAL_ROOT: "/env/openral" }, { openralRoot: "/env/openral" }],
    ["rpent", { RPENT_ROOT: "/env/rpent" }, { rpentRoot: "/env/rpent" }],
    ["lerobot", {}, {}],
  ];
  for (const [type, env, expected] of profiles) {
    const worker = workerFixture();
    const environment = await createEnvironment({ type, options: { worker, ...(type === "openral" ? { openralRoot: "/explicit/openral" } : {}) } }, { env });
    assert.equal(worker.requests.length, 0);
    await environment.reset({ task: "task", seed: 9 });
    const configure = worker.requests[0];
    assert.equal(configure.driver, type);
    for (const [key, value] of Object.entries(expected)) assert.equal(configure.options[key], type === "openral" ? "/explicit/openral" : value);
    assert.equal(worker.requests[1].seed, 9);
    assert.equal(configure.options.executeEnabled, ["aloha", "openral", "rpent"].includes(type));
    await environment.close();
  }
});

test("ARC and Gymnasium registry adapters defer configuration until reset", async () => {
  for (const type of ["arc-agi3", "gymnasium"]) {
    const worker = workerFixture();
    const environment = await createEnvironment({ type, options: { worker, environmentId: "CartPole-v1" } }, { env: { ARC_ENVIRONMENTS_DIR: "/games" } });
    assert.equal(worker.requests.length, 0);
    await environment.reset({ task: "game", seed: 4 });
    assert.equal(worker.requests[0].op, "configure");
    assert.equal(worker.requests[1].op, "reset");
    if (type === "arc-agi3") assert.equal(worker.requests[0].options.environmentsDir, "/games");
    await environment.close();
  }
});

test("source paths follow the recipe directory and OpenRAL manifests follow its source root", async () => {
  const worker = workerFixture();
  const environment = await createEnvironment({ type: "openral", options: { worker, openralRoot: "../openral", robotYaml: "./robots/franka_panda/robot.yaml", simEnvYaml: "scenes/benchmark/libero.yaml" } }, { baseDir: "/project/configs", env: {} });
  await environment.reset();
  const options = worker.requests[0].options;
  assert.equal(options.openralRoot, "/project/openral");
  assert.equal(options.robotYaml, "/project/openral/robots/franka_panda/robot.yaml");
  assert.equal(options.simEnvYaml, "/project/openral/scenes/benchmark/libero.yaml");
  await environment.close();
});

test("LIBERO can reuse a supplied client and keeps reset identity", async () => {
  const requests = [];
  const client = {
    async request(command, args) {
      requests.push({ command, args });
      return { frame: { capturedAt: "2026-01-01T00:00:00Z", mimeType: "image/png", imageB64: "", libero: { observationId: 1, task: { language: "pick up" }, proprio: {} } }, steps: 0, horizon: 30, terminated: false, success: false };
    },
    async close() { requests.push({ command: "close" }); },
  };
  const environment = await createEnvironment({ type: "libero", options: { client } });
  assert.equal(requests.length, 0);
  await environment.reset({ task: { suite: "libero_10_task", task: 2 }, seed: 8 });
  assert.deepEqual(requests[0], { command: "reset", args: { suite: "libero_10_task", task: 2, seed: 8 } });
  await environment.close();
  assert.equal(requests.at(-1).command, "close");
});

test("G1 registry constructs the existing camera and executor path without moving by default", async () => {
  const captures = [];
  const network = [];
  const environment = await createEnvironment({
    type: "g1",
    options: {
      camera: { async capture(options) { captures.push(options); return { imageB64: "aW1hZ2U=", capturedAt: Date.now() / 1000, depth: { center_m: 3, left_m: 3, right_m: 3 } }; } },
      config: { actuator: "console", useLidar: false, executeEnabled: true, consoleBaseUrl: "http://g1-gateway" },
    },
  }, {
    env: {},
    fetch: async (url, options) => { network.push({ url, options }); return { ok: true, json: async () => ({ ready: true }) }; },
  });
  assert.equal(captures.length, 0);
  assert.equal(network.length, 0);
  const initial = await environment.reset({ task: "look ahead" });
  assert.equal(initial.info.physicalReset, false);
  const next = await environment.step({ name: "W", arguments: {} });
  assert.equal(next.info.execution.preview, true);
  assert.equal(next.info.execution.executed, false);
  assert.equal(network.every(request => request.url === "http://g1-gateway/api/status"), true);
  await environment.close();
});

test("chat policy reads provider defaults while explicit options take precedence", async () => {
  const env = { REA_MODEL: "default-vision", REA_BASE_URL: "http://127.0.0.1:1/v1", REA_API_KEY: "local-test", REA_POLICY_MODE: "json" };
  const defaults = await createPolicy("chat", { env });
  assert.equal(defaults.id, "progress:chat:default-vision:json");
  const explicit = await createPolicy({ type: "chat", options: { model: "custom-vision", mode: "tool" } }, { env });
  assert.equal(explicit.id, "progress:chat:custom-vision:tool");
  // The names the variables had before the project was renamed still configure the policy.
  assert.equal((await createPolicy("chat", { env: { REA_MODEL: "legacy-vision", REA_BASE_URL: "http://127.0.0.1:1/v1" } })).id, "progress:chat:legacy-vision:tool");
  await assert.rejects(createPolicy("chat", { env: {} }), /requires baseUrl and model/);
});

test("G1 mock configured through the environment runs reset and step with the real executor", async context => {
  let captures = 0;
  let completions = 0;
  const environment = await createEnvironment({ type: "g1", options: {
    camera: { async capture() {
      captures += 1;
      return { imageB64: "aW1hZ2U=", depth: { center_m: 3, left_m: 3, right_m: 3 } };
    } },
    execute: true,
    waitForCompletion: async () => { completions += 1; },
  } }, { env: { G1_ACTUATOR: "mock", G1_EXECUTE_ENABLED: "1", G1_USE_LIDAR: "0" } });
  context.after(() => environment.close());
  assert.equal(captures, 0);
  const initial = await environment.reset({ task: "exercise offline navigation" });
  assert.equal(initial.observation.state.robot.mock, true);
  assert.equal(initial.observation.state.robot.ready, true);
  const next = await environment.step({ name: "W", arguments: { durationMs: 400 } });
  assert.equal(next.info.execution.gate.allowed, true);
  assert.equal(next.info.execution.executed, true);
  assert.equal(next.info.execution.dispatch.mock, true);
  assert.ok(next.info.execution.dispatch.motion.vx > 0);
  assert.equal(next.info.completion, "completion_hook");
  assert.equal(completions, 1);
  assert.equal(next.success, null);
});

test("G1 accepts the same Python and process environment overrides as other adapters", async context => {
  const baseDir = await mkdtemp(join(tmpdir(), "fast-g1-python-"));
  context.after(() => rm(baseDir, { recursive: true, force: true }));
  await writeFile(join(baseDir, "camera.mjs"), 'process.stdout.write(JSON.stringify({ok:true,image_b64:process.env.REA_G1_IMAGE,depth:{center_m:3,left_m:3,right_m:3}}));');
  const environment = await createEnvironment({ type: "g1", options: {
    python: process.execPath,
    config: { python: "/not/the/chosen/interpreter", senseScript: "./camera.mjs", persistentCapture: false, useLidar: false },
    executor: { evidence: async () => ({ status: { ready: true } }) },
  } }, { baseDir, env: { REA_G1_IMAGE: "aW1hZ2U=" } });
  context.after(() => environment.close());
  const initial = await environment.reset();
  assert.equal(initial.observation.images[0].data, "aW1hZ2U=");
});

test("scripted and routed policies use the same registry and reset their sequence", async () => {
  const action = { name: "wait", arguments: {} };
  const scripted = await createPolicy({ type: "scripted", options: { actions: [action] } });
  assert.deepEqual((await scripted.decide()).action, action);
  await assert.rejects(scripted.decide(), /exhausted/);
  await scripted.reset();
  assert.deepEqual((await scripted.decide()).action, action);
  const registry = createRegistry();
  const closed = [];
  registry.registerPolicy("custom", ({ label }) => ({ id: label, decide: async () => ({ action }), close: async () => { closed.push(label); } }));
  const routed = await registry.createPolicy({ type: "routed", options: { reflex: { type: "custom", options: { label: "fast" } }, deliberate: { type: "custom", options: { label: "careful" } } } });
  assert.equal((await routed.decide({ history: [] })).model, "fast");
  await routed.close();
  assert.deepEqual(closed, ["fast", "careful"]);
  await assert.rejects(registry.createPolicy({ type: "routed", options: { reflex: { type: "custom", options: { label: "allocated" } }, deliberate: "missing" } }), /Unknown policy/);
  assert.equal(closed.at(-1), "allocated");
});

test("module factories resolve relative to configuration and support named exports", async context => {
  const baseDir = await mkdtemp(join(tmpdir(), "fast-registry-"));
  context.after(() => rm(baseDir, { recursive: true, force: true }));
  await writeFile(join(baseDir, "extension.mjs"), `
    export function robot(options) { return { id: options.label, reset: async () => ({ observation: { actions: [] } }), step: async () => ({}), scorecard: async () => ({ score: 1 }) }; }
    export function policy(options) { return { id: options.label, decide: async () => ({ action: { name: "wait", arguments: {} } }) }; }
  `);
  const environment = await createEnvironment({ type: "module", options: { module: "./extension.mjs", export: "robot", label: "custom-arm" } }, { baseDir });
  await environment.reset();
  assert.equal(environment.id, "custom-arm");
  assert.deepEqual(await environment.scorecard({ close: true }), { score: 1 });
  const policy = await createPolicy({ type: "module", options: { module: "./extension.mjs", export: "policy", label: "custom-policy" } }, { baseDir });
  assert.equal(policy.id, "custom-policy");
  await environment.close();
});

test("generic robot reports the missing runtime driver instead of pretending support", async () => {
  const environment = await createEnvironment("robot");
  await assert.rejects(environment.reset(), /JavaScript driver/);
  await environment.close();
});

test("the G1 environment resolves G1_ROBOT_PROFILE from the process environment", async context => {
  const environment = await createEnvironment({ type: "g1", options: {
    camera: { async capture() { return { imageB64: "aW1hZ2U=", depth: { center_m: 3, left_m: 3, right_m: 3 } }; } },
    execute: true,
  } }, { env: { G1_ACTUATOR: "mock", G1_EXECUTE_ENABLED: "1", G1_USE_LIDAR: "0", G1_ROBOT_PROFILE: "./fixtures/profiles/holonomic.mjs" }, baseDir: fileURLToPath(new URL(".", import.meta.url)) });
  context.after(() => environment.close());
  const initial = await environment.reset({ task: "cross the room" });
  assert.deepEqual(initial.observation.actions.map(action => action.name), ["H", "F", "L", "R"]);
  const next = await environment.step({ name: "F", arguments: { durationMs: 400 } });
  assert.equal(next.info.execution.executed, true);
  assert.equal(next.info.execution.dispatch.motion.vx, 0.2);
});

test("G1 through the registry runs on fixture sensing with no Python and no lidar file", async context => {
  const environment = await createEnvironment({ type: "g1", options: { execute: true } }, { env: {
    G1_ACTUATOR: "mock", G1_EXECUTE_ENABLED: "1", G1_SENSE: "fixture", G1_FIXTURE_DEPTH: "2.5,2.5,2.5;0.3,2.5,2.5", G1_OBSTACLES: "fixture", G1_FIXTURE_LIDAR: "5,5,5",
  } });
  context.after(() => environment.close());
  const initial = await environment.reset({ task: "approach" });
  assert.equal(initial.observation.state.depth.center_m, 2.5);
  assert.equal(initial.observation.state.lidar.reason, "clear");
  const first = await environment.step({ name: "W", arguments: { durationMs: 400 } });
  assert.equal(first.info.execution.gate.allowed, true, "the live re-read sees the 2.5 m frame that was served");
  assert.equal(first.observation.state.depth.center_m, 0.3, "the next frame is the near one");
  const second = await environment.step({ name: "W", arguments: { durationMs: 400 } });
  assert.equal(second.info.execution.gate.allowed, false);
  assert.ok(second.info.execution.gate.reasons.includes("depth:depth_center_near"));
});
