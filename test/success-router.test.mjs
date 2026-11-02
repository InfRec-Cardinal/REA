import assert from "node:assert/strict";
import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import test from "node:test";

import { createConfiguredSystems } from "../src/framework/systems.mjs";
import { runAsyncEpisode } from "../src/framework/async-runtime.mjs";
import { createSuccessRouter } from "../src/policies/success-router.mjs";

const candidates = ["pi0", "pi05", "rdt", "dp", "patchwam", "medex", "gigabrain", "code"].map(id => ({ id, kind: id === "code" ? "program" : "vla" }));

async function predictionServer(context, predict) {
  const requests = [];
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const input = JSON.parse(Buffer.concat(chunks));
    requests.push({ url: request.url, method: request.method, input });
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify(await predict(input)));
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  context.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  return { endpoint: `http://127.0.0.1:${server.address().port}/predict`, requests };
}

function prediction(winner, pool = candidates) {
  return {
    scores: Object.fromEntries(pool.map(({ id }) => [id, id === winner ? 0.9 : 0.2])),
    trained_candidates: pool.map(({ id }) => id),
    model: "fixture-success-bce",
  };
}

test("routes either VLA or code by trained success probabilities and holds the initial choice", async context => {
  const { endpoint, requests } = await predictionServer(context, input => prediction(input.seed === 1 ? "code" : "medex"));
  const router = createSuccessRouter({ endpoint }, { episode: { id: "scene:0" } });
  const observation = { id: "initial", text: "Pick up the cup", images: [{ name: "head", data: "image" }], state: { joints: [0, 1] } };
  const initial = await router.route({ observation, instruction: "Pick up the cup", candidates, seed: 1, step: 0 });
  assert.equal(initial.candidateId, "code");
  assert.equal(initial.meta.scope, "episode");
  assert.equal(initial.meta.scores.code, 0.9);
  const continued = await router.route({ observation: { id: "later" }, candidates, seed: 2, step: 1, advice: { candidateId: "medex" } });
  assert.equal(continued.candidateId, "code");
  assert.equal(continued.meta.initialObservationId, "initial");
  assert.equal(continued.meta.reused, true);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "/predict");
  assert.equal(requests[0].method, "POST");
  assert.deepEqual(requests[0].input.observation, observation);
  assert.equal(requests[0].input.episodeId, "scene:0");
  assert.deepEqual(requests[0].input.candidates, candidates);
  const next = await router.route({ observation: { id: "new-episode" }, candidates, seed: 2, step: 0 });
  assert.equal(next.candidateId, "medex");
  assert.equal(next.meta.reused, false);
  assert.equal(requests.length, 2);
});

test("a seven-model checkpoint cannot silently supply a code policy score", async context => {
  const { endpoint } = await predictionServer(context, () => ({ ...prediction("medex", candidates.slice(0, 7)), scores: { ...prediction("medex").scores, code: 0.99 } }));
  const router = createSuccessRouter({ endpoint });
  await assert.rejects(router.route({ observation: {}, candidates, step: 0 }), /no trained success probability for candidate code/);
  await assert.rejects(router.route({ observation: {}, candidates, step: 1 }), /requires the initial observation/);
});

test("Python float state text survives JavaScript transport and reaches the router encoder unchanged", async context => {
  const stateText = '{"endpose": {"left_gripper": 1.0, "right_gripper": 0.0}, "joint": [1.0, 0.0]}';
  const observation = JSON.parse('{"id":"initial","state":{"vla":{"endpose":{"left_gripper":1.0,"right_gripper":0.0},"joint_action":{"vector":[1.0,0.0]}}}}');
  observation.state.vla.state_text = stateText;
  const { endpoint, requests } = await predictionServer(context, () => prediction("medex"));
  await createSuccessRouter({ endpoint }).route({ observation, candidates, step: 0 });
  const transported = requests[0].input;
  assert.equal(JSON.stringify(transported.observation.state.vla.joint_action.vector), "[1,0]");
  assert.equal(transported.observation.state.vla.state_text, stateText);
  const scripts = fileURLToPath(new URL("../scripts", import.meta.url));
  const python = spawnSync("python3", ["-c",
    "import json,sys;sys.path.insert(0,sys.argv[1]);from success_router_server import robot_state_text;sys.stdout.write(robot_state_text(json.load(sys.stdin)['observation']))", scripts],
  { input: JSON.stringify(transported), encoding: "utf8" });
  assert.equal(python.status, 0, python.stderr);
  assert.equal(python.stdout, stateText);
});

test("configured System1 dispatches the selected policy and records scores through the REA runtime", async context => {
  const { endpoint, requests } = await predictionServer(context, () => prediction("code"));
  const dispatched = [];
  const records = [];
  let count = 0;
  const observation = () => ({ id: String(count), state: { count }, actions: [{ name: "move", parameters: { type: "object" } }] });
  const systems = await createConfiguredSystems({
    system0: candidates.map(candidate => ({ ...candidate, policy: { type: "fixture", options: { id: candidate.id } } })),
    system1: { type: "module", options: { module: "./src/policies/success-router.mjs", endpoint } },
  }, {
    baseDir: fileURLToPath(new URL("../", import.meta.url)),
    episode: { id: "test:42" },
    registry: {
      createPolicy: async ({ options }) => ({ decide: async () => { dispatched.push(options.id); return { action: { name: "move", arguments: {} } }; } }),
    },
  });
  const result = await runAsyncEpisode({
    ...systems,
    environment: {
      reset: async () => ({ observation: observation() }),
      step: async () => { count++; return { observation: observation(), success: count === 3, terminated: count === 3 }; },
    },
    maxSteps: 3,
    onTransition: record => records.push(record),
  });
  await systems.close();
  assert.equal(result.success, true);
  assert.deepEqual(dispatched, ["code", "code", "code"]);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].input.episodeId, "test:42");
  const steps = records.filter(record => record.type === "step");
  assert.deepEqual(steps.map(record => record.route.meta.reused), [false, true, true]);
  assert.equal(steps[2].route.meta.scores.medex, 0.2);
  assert.equal(steps[0].route.model, "fixture-success-bce");
});

test("prediction requests follow episode cancellation", async () => {
  const controller = new AbortController();
  const router = createSuccessRouter({
    fetchImpl: async (_, { signal }) => new Promise((resolve, reject) => {
      signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      controller.abort(new Error("episode cancelled"));
    }),
  });
  await assert.rejects(router.route({ observation: {}, candidates, step: 0, signal: controller.signal }), /episode cancelled/);
});
