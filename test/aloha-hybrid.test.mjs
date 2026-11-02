import assert from "node:assert/strict";
import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import test from "node:test";

import configuration from "../examples/aloha-pi05-lingbot-code.config.mjs";
import pi05CodeConfiguration from "../examples/aloha-pi05-code.config.mjs";
import { createEmbodiedAgent, createRegistry } from "../src/framework/index.mjs";

const examplesDirectory = fileURLToPath(new URL("../examples/", import.meta.url));
const cameras = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"];
const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6lGQAAAAASUVORK5CYII=";
const initialPositions = () => Array.from({ length: 14 }, (_, index) => index === 6 || index === 13 ? 0.5 : 0);
const jointVector = { type: "array", items: { type: "number" }, minItems: 14, maxItems: 14 };
const actions = [
  { name: "joint_position", parameters: { type: "object", properties: { positions: jointVector }, required: ["positions"], additionalProperties: false } },
  { name: "joint_trajectory", parameters: { type: "object", properties: { positions: { type: "array", items: jointVector, minItems: 1, maxItems: 100 } }, required: ["positions"], additionalProperties: false } },
];

// Hardware and VLA transports use fixtures; configuration, registry, HTTP routing,
// policy adapters, Python code execution and environment dispatch run normally.
function hardwareFixture({ holdSteps = false, stepLimit = 4 } = {}) {
  const requests = [];
  const arrivals = Array.from({ length: stepLimit }, () => Promise.withResolvers());
  const releases = Array.from({ length: stepLimit }, () => Promise.withResolvers());
  let activeSteps = 0;
  let maxActiveSteps = 0;
  let qpos = initialPositions();
  let steps = 0;
  let running = true;
  const snapshot = () => ({
    state: { qpos: [...qpos], qvel: Array(14).fill(0), effort: Array(14).fill(0) },
    images: cameras.map(name => ({ name, mimeType: "image/png", data: png })),
    terminated: steps === stepLimit,
    info: { execution: { status: "submitted" }, controlSteps: steps },
  });
  return {
    requests, arrivals, releases,
    get maxActiveSteps() { return maxActiveSteps; },
    health: () => ({ running }),
    stop() { running = false; },
    async request(payload) {
      requests.push(structuredClone(payload));
      let result;
      if (payload.op === "configure") {
        running = true;
        result = { actions, timing: "realtime", metadata: { cameraNames: cameras, actionDimension: 14 } };
      } else if (payload.op === "reset") {
        qpos = initialPositions();
        steps = 0;
        result = snapshot();
      } else if (payload.op === "step") {
        activeSteps += 1;
        maxActiveSteps = Math.max(maxActiveSteps, activeSteps);
        arrivals[steps].resolve();
        if (holdSteps) await releases[steps].promise;
        qpos = payload.action.name === "joint_position" ? payload.action.arguments.positions : payload.action.arguments.positions.at(-1);
        steps += 1;
        activeSteps -= 1;
        result = snapshot();
      } else {
        result = snapshot();
      }
      return { ok: true, result };
    },
  };
}

function vlaFixture(kind) {
  const requests = [];
  let stops = 0;
  return {
    requests,
    get stops() { return stops; },
    stop() { stops += 1; },
    async request(payload) {
      requests.push(structuredClone(payload));
      if (payload.op !== "infer") return { ok: true, result: {} };
      const qpos = [...(kind === "pi05" ? payload.state : payload.states["observation.state"])];
      qpos[0] += 0.1;
      return { ok: true, result: { [kind === "pi05" ? "actions" : "action"]: [qpos] } };
    },
  };
}

async function modelEndpoint(context, choose = value => value < 0.05 || value > 0.24 ? "pi05" : value < 0.15 ? "lingbot-vla" : "code") {
  const requests = [];
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks));
    requests.push(body);
    const prompt = body.messages[1].content.find(item => item.type === "text").text;
    const current = JSON.parse(prompt.split("\n\n").find(part => part.startsWith("Current observation: ")).slice("Current observation: ".length));
    const value = current.state.qpos[0];
    const candidateId = body.model === "router-fixture" ? await choose(value) : null;
    const decision = body.model === "router-fixture" ? {
      name: "select_controller", arguments: { candidateId }, thought: `Use ${candidateId} from the current joint state.`,
    } : {
      name: "run_python",
      arguments: { code: "positions = np.asarray(observation['state']['qpos'], dtype=float).copy()\npositions[0] += 0.05\njoint_position(positions)" },
      thought: "Calculate one absolute target from the latest measured joints.",
    };
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.write(`data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, function: { name: "take_action", arguments: JSON.stringify(decision) } }] }, finish_reason: "tool_calls" }] })}\n\n`);
    response.write(`data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 5 } })}\n\n`);
    response.end("data: [DONE]\n\n");
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  context.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  return { requests, url: `http://127.0.0.1:${server.address().port}/v1` };
}

test("published ALOHA hybrid configuration routes pi05, LingBot and real Python code at action boundaries", async context => {
  const endpoint = await modelEndpoint(context);
  const env = {
    ...process.env,
    REA_MODEL: "code-fixture", REA_ROUTER_MODEL: "router-fixture",
    REA_BASE_URL: endpoint.url, REA_API_KEY: "fixture-key", LINGBOT_VLA_ROBOT: "fixture-aloha",
    REA_INSTRUCTION: "Move the object to the tray.", REA_MAX_STEPS: "4", ROBOT_EXECUTE: "true",
    ALOHA_ROUTING_MODE: "sync",
  };
  const config = configuration({ env });
  assert.equal(config.policy, undefined);
  assert.deepEqual(config.systems.system0.map(candidate => candidate.id), ["pi05", "lingbot-vla", "code"]);
  assert.deepEqual(config.systems.system0.map(candidate => candidate.kind), ["vla", "vla", "code"]);
  const hardware = hardwareFixture();
  const pi05 = vlaFixture("pi05");
  const lingbot = vlaFixture("lingbot-vla");
  config.environment.options.worker = hardware;
  config.systems.system0.find(candidate => candidate.id === "pi05").policy.options.worker = pi05;
  config.systems.system0.find(candidate => candidate.id === "lingbot-vla").policy.options.worker = lingbot;
  const agent = await createEmbodiedAgent(config, { baseDir: examplesDirectory, env });
  context.after(() => agent.close());

  // Run twice through the public API to ensure episode observations and routes
  // restart cleanly, without retaining a previous controller or stale actions.
  for (let episode = 0; episode < 2; episode += 1) {
    const transitions = [];
    const result = await agent.run({ onTransition: transition => transitions.push(transition) });
    assert.equal(result.error, null, result.error?.message);
    assert.equal(result.steps, 4);
    assert.equal(result.reason, "terminated");
    assert.equal(result.success, null);
    assert.equal(result.asyncMetrics.routeChanges, 3);
    assert.equal(result.asyncMetrics.system1Calls, 4);
    assert.equal(result.asyncMetrics.system2Calls, 4);
    const steps = transitions.filter(transition => transition.type === "step");
    assert.deepEqual(steps.map(transition => transition.route.candidateId), ["pi05", "lingbot-vla", "code", "pi05"]);
    assert.deepEqual(steps.map(transition => transition.action.name), ["joint_trajectory", "joint_trajectory", "joint_position", "joint_trajectory"]);
    assert.deepEqual(steps.map(transition => transition.observation.state.qpos[0]), [0, 0.1, 0.2, 0.25]);
    assert.deepEqual(steps.map(transition => transition.nextObservation.state.qpos[0]), [0.1, 0.2, 0.25, 0.35]);
    assert.ok(steps.every(transition => transition.route.reason.includes(transition.route.candidateId)));
  }

  const piCalls = pi05.requests.filter(request => request.op === "infer");
  const lingbotCalls = lingbot.requests.filter(request => request.op === "infer");
  assert.equal(piCalls.length, 4);
  assert.equal(lingbotCalls.length, 2);
  assert.equal(lingbot.requests.filter(request => request.op === "reset").length, 2);
  assert.deepEqual(piCalls.map(request => request.state[0]), [0, 0.25, 0, 0.25]);
  assert.ok(piCalls.every(request => request.prompt.includes(env.REA_INSTRUCTION)));
  assert.deepEqual(Object.keys(piCalls[0].images), ["cam_high", "cam_left_wrist", "cam_right_wrist"]);
  assert.deepEqual(Object.keys(lingbotCalls[0].images), ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]);
  assert.deepEqual(lingbotCalls.map(request => request.states["observation.state"][0]), [0.1, 0.1]);
  assert.ok(lingbotCalls.every(request => request.task.includes(env.REA_INSTRUCTION)));
  assert.equal(hardware.requests.filter(request => request.op === "step").length, 8);
  assert.ok(hardware.requests.filter(request => request.op === "step").every(request => request.execute === true));
  assert.equal(hardware.requests.filter(request => request.op === "reset").length, 2);
  assert.equal(hardware.requests.filter(request => request.op === "close").length, 2);
  assert.equal(pi05.stops, 2);
  assert.equal(lingbot.stops, 2);
  const routerRequests = endpoint.requests.filter(request => request.model === "router-fixture");
  const codeRequests = endpoint.requests.filter(request => request.model === "code-fixture");
  assert.equal(routerRequests.length, 8);
  assert.equal(codeRequests.length, 2);
  for (const request of endpoint.requests) {
    assert.equal(request.messages[1].content.filter(item => item.type === "image_url").length, 4);
  }
});

test("dedicated pi05 plus code configuration routes both controllers without constructing LingBot", async context => {
  const endpoint = await modelEndpoint(context, value => value < 0.05 || (value > 0.14 && value < 0.2) ? "pi05" : "code");
  const env = {
    ...process.env,
    REA_MODEL: "code-fixture", REA_ROUTER_MODEL: "router-fixture",
    REA_BASE_URL: endpoint.url, REA_API_KEY: "fixture-key",
    REA_INSTRUCTION: "Move the object to the tray.", REA_MAX_STEPS: "4", ROBOT_EXECUTE: "true",
    ALOHA_ROUTING_MODE: "sync",
  };
  delete env.LINGBOT_VLA_ROBOT;
  const config = pi05CodeConfiguration({ env });
  assert.equal(config.policy, undefined);
  assert.deepEqual(config.systems.system0.map(candidate => candidate.id), ["pi05", "code"]);
  assert.deepEqual(config.systems.system0.map(candidate => candidate.kind), ["vla", "code"]);
  const hardware = hardwareFixture();
  const pi05 = vlaFixture("pi05");
  config.environment.options.worker = hardware;
  config.systems.system0.find(candidate => candidate.id === "pi05").policy.options.worker = pi05;
  const registry = createRegistry();
  let lingbotConstructions = 0;
  registry.registerPolicy("lingbot-vla", () => {
    lingbotConstructions += 1;
    throw new Error("The two-controller configuration must not construct LingBot");
  });
  const agent = await createEmbodiedAgent(config, { baseDir: examplesDirectory, env, registry });
  context.after(() => agent.close());
  const transitions = [];
  const result = await agent.run({ onTransition: transition => transitions.push(transition) });
  assert.equal(result.error, null, result.error?.message);
  assert.equal(result.steps, 4);
  assert.equal(result.reason, "terminated");
  assert.equal(result.success, null);
  assert.equal(result.asyncMetrics.routeChanges, 3);
  assert.equal(lingbotConstructions, 0);
  const steps = transitions.filter(transition => transition.type === "step");
  assert.deepEqual(steps.map(transition => transition.route.candidateId), ["pi05", "code", "pi05", "code"]);
  assert.deepEqual(steps.map(transition => transition.action.name), ["joint_trajectory", "joint_position", "joint_trajectory", "joint_position"]);
  assert.deepEqual(steps.map(transition => Number(transition.observation.state.qpos[0].toFixed(2))), [0, 0.1, 0.15, 0.25]);
  assert.deepEqual(steps.map(transition => Number(transition.nextObservation.state.qpos[0].toFixed(2))), [0.1, 0.15, 0.25, 0.3]);
  const inference = pi05.requests.filter(request => request.op === "infer");
  assert.equal(inference.length, 2);
  assert.deepEqual(inference.map(request => Number(request.state[0].toFixed(2))), [0, 0.15]);
  const routerRequests = endpoint.requests.filter(request => request.model === "router-fixture");
  assert.equal(routerRequests.length, 4);
  assert.equal(endpoint.requests.filter(request => request.model === "code-fixture").length, 2);
  for (const request of routerRequests) {
    const prompt = request.messages[1].content.find(item => item.type === "text").text;
    const schemas = JSON.parse(prompt.split("\n\n").find(part => part.startsWith("Available actions and parameter schemas: ")).slice("Available actions and parameter schemas: ".length));
    assert.deepEqual(schemas[0].parameters.properties.candidateId.enum, ["pi05", "code"]);
  }
  assert.equal(hardware.requests.filter(request => request.op === "step").length, 4);
  assert.equal(pi05.stops, 1);
});

for (const [name, configure, lastCandidate] of [
  ["pi05 plus code", pi05CodeConfiguration, "pi05"],
  ["pi05 plus LingBot plus code", configuration, "lingbot-vla"],
]) {
  test(`${name} dispatches System0 first and keeps each controller for four completed action blocks`, { timeout: 20000 }, async context => {
    const routeRequests = [];
    const routeArrivals = Array.from({ length: 3 }, () => Promise.withResolvers());
    const routeReplies = Array.from({ length: 3 }, () => Promise.withResolvers());
    const completed = Array.from({ length: 2 }, () => Promise.withResolvers());
    const hardware = hardwareFixture({ holdSteps: true, stepLimit: 9 });
    context.after(() => {
      for (const release of hardware.releases) release.resolve();
      for (const reply of routeReplies) reply.resolve("pi05");
    });
    const endpoint = await modelEndpoint(context, value => {
      const index = routeRequests.length;
      routeRequests.push(value);
      assert.equal(hardware.requests.filter(request => request.op === "step").length, index * 4 + 1,
        "background selection starts only after the scheduled System0 action dispatch");
      routeArrivals[index].resolve();
      return routeReplies[index].promise;
    });
    const env = {
      ...process.env,
      REA_MODEL: "code-fixture", REA_ROUTER_MODEL: "router-fixture",
      REA_BASE_URL: endpoint.url, REA_API_KEY: "fixture-key",
      REA_INSTRUCTION: "Move the object to the tray.", REA_MAX_STEPS: "9", ROBOT_EXECUTE: "true",
      ...(lastCandidate === "lingbot-vla" ? { LINGBOT_VLA_ROBOT: "fixture-aloha" } : {}),
    };
    delete env.ALOHA_ROUTING_MODE;
    delete env.ALOHA_ROUTE_EVERY_STEPS;
    delete env.ALOHA_MIN_CONTROLLER_STEPS;
    const config = configure({ env });
    assert.equal(config.systems.mode, "async");
    assert.equal(config.systems.thinkingSchedule, "after_dispatch");
    assert.equal(config.systems.thinkEverySteps, 4);
    assert.equal(config.systems.minControllerSteps, 4);
    assert.equal(config.systems.system1, undefined);
    assert.equal(config.systems.system2.options.exportName, "createAlohaRouterThinker");
    const pi05 = vlaFixture("pi05");
    const lingbot = vlaFixture("lingbot-vla");
    config.environment.options.worker = hardware;
    config.systems.system0.find(candidate => candidate.id === "pi05").policy.options.worker = pi05;
    if (lastCandidate === "lingbot-vla") {
      config.systems.system0.find(candidate => candidate.id === "lingbot-vla").policy.options.worker = lingbot;
    }
    const agent = await createEmbodiedAgent(config, { baseDir: examplesDirectory, env });
    context.after(() => agent.close());
    const events = [];
    const transitions = [];
    const dispatchCountsAtThinkingStart = [];
    const running = agent.run({
      onTransition: transition => transitions.push(transition),
      onEvent: event => {
        events.push(event);
        if (event.type === "thinking" && event.status === "started") {
          dispatchCountsAtThinkingStart.push(hardware.requests.filter(request => request.op === "step").length);
        }
        if (event.type === "thinking" && event.status === "completed") {
          completed[Number(event.id.split("-")[1]) - 1]?.resolve();
        }
      },
    });

    await Promise.all([routeArrivals[0].promise, hardware.arrivals[0].promise]);
    assert.equal(pi05.requests.filter(request => request.op === "infer").length, 1);
    assert.equal(endpoint.requests.filter(request => request.model === "code-fixture").length, 0);
    assert.equal(events.filter(event => event.type === "thinking" && event.status === "completed").length, 0,
      "Pi0.5 starts executing before the slow route request completes");

    routeReplies[0].resolve("code");
    await completed[0].promise;
    assert.equal(hardware.requests.filter(request => request.op === "step").length, 1,
      "a completed recommendation cannot interrupt the active action block");
    for (let index = 0; index < 3; index += 1) {
      hardware.releases[index].resolve();
      await hardware.arrivals[index + 1].promise;
      assert.equal(pi05.requests.filter(request => request.op === "infer").length, index + 2,
        "Pi0.5 keeps control until four action blocks have completed");
      assert.equal(endpoint.requests.filter(request => request.model === "code-fixture").length, 0);
      assert.equal(routeRequests.length, 1, "completing a route request does not immediately launch another");
    }
    hardware.releases[3].resolve();
    await Promise.all([hardware.arrivals[4].promise, routeArrivals[1].promise]);
    assert.equal(hardware.requests.filter(request => request.op === "step")[4].action.name, "joint_position");

    routeReplies[1].resolve(lastCandidate);
    await completed[1].promise;
    assert.equal(hardware.requests.filter(request => request.op === "step").length, 5);
    for (let index = 4; index < 7; index += 1) {
      hardware.releases[index].resolve();
      await hardware.arrivals[index + 1].promise;
      assert.equal(endpoint.requests.filter(request => request.model === "code-fixture").length, index - 2,
        "Code also keeps control until four action blocks have completed");
      assert.equal(routeRequests.length, 2);
    }
    hardware.releases[7].resolve();
    await Promise.all([hardware.arrivals[8].promise, routeArrivals[2].promise]);
    hardware.releases[8].resolve();
    const result = await running;

    assert.equal(result.error, null, result.error?.message);
    assert.equal(result.reason, "terminated");
    assert.equal(result.steps, 9);
    assert.equal(result.asyncMetrics.blockedThinkingMs, 0);
    assert.ok(result.asyncMetrics.overlapMs > 0);
    assert.equal(result.asyncMetrics.routeChanges, 2);
    assert.equal(result.asyncMetrics.system2Calls, 3);
    assert.equal(hardware.maxActiveSteps, 1);
    assert.deepEqual(dispatchCountsAtThinkingStart, [1, 5, 9],
      "the hardware action has already been dispatched when background thinking begins");
    const steps = transitions.filter(transition => transition.type === "step");
    const rounded = value => Number(value.toFixed(2));
    assert.deepEqual(steps.map(transition => transition.route.candidateId), ["pi05", "pi05", "pi05", "pi05", "code", "code", "code", "code", lastCandidate]);
    assert.deepEqual(steps.map(transition => rounded(transition.observation.state.qpos[0])), [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6]);
    assert.deepEqual(steps.map(transition => rounded(transition.nextObservation.state.qpos[0])), [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]);
    assert.deepEqual(routeRequests.map(rounded), [0, 0.4, 0.6]);
    assert.equal(endpoint.requests.filter(request => request.model === "router-fixture").length, 3);
    assert.equal(steps[4].route.adviceId, "think-1");
    assert.equal(steps[8].route.adviceId, "think-2");
    const finalInference = (lastCandidate === "pi05" ? pi05 : lingbot).requests.filter(request => request.op === "infer").at(-1);
    assert.equal(rounded(lastCandidate === "pi05" ? finalInference.state[0] : finalInference.states["observation.state"][0]), 0.6);
    assert.ok(events.some(event => event.type === "thinking" && event.status === "cancelled" && event.id === "think-3"));
  });
}
