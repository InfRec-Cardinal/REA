import assert from "node:assert/strict";
import test from "node:test";

import {
  ROUTE_TABLE_SCHEMA, buildRouteTable, candidateEvidence, createRouteMemoryRouter, rankCandidates, routeLadder,
  routeTableSha256, taskKey, verifyRouteTable,
} from "../src/framework/route-memory.mjs";

const episodes = (task, candidateId, outcomes, steps = 10) => outcomes.map(success => ({ task, candidateId, success, steps }));
const candidates = [{ id: "fix" }, { id: "evo" }, { id: "vla" }];

test("task keys name LIBERO-style tasks, strings and ids", () => {
  assert.equal(taskKey({ suite: "libero_goal_task", task: "open_drawer" }), "libero_goal_task/open_drawer");
  assert.equal(taskKey("trash"), "trash");
  assert.equal(taskKey({ id: "scene-3" }), "scene-3");
  assert.equal(taskKey(null), "default");
});

test("evidence counts only scored episodes and steps per success covers failures too", () => {
  const evidence = candidateEvidence([{ success: true, steps: 10 }, { success: false, steps: 30 }, { success: null, steps: 5 }]);
  assert.deepEqual(evidence, { n: 2, successes: 1, unscored: 1, rate: 0.5, steps: 40, stepsPerSuccess: 40 });
});

test("ranking prefers the higher success rate, then fewer steps per success, then candidate order", () => {
  const evidence = {
    fix: { n: 10, successes: 8, stepsPerSuccess: 50 },
    evo: { n: 5, successes: 4, stepsPerSuccess: 40 },
    vla: { n: 10, successes: 9, stepsPerSuccess: 90 },
  };
  assert.deepEqual(rankCandidates(evidence, ["fix", "evo", "vla"]), ["vla", "evo", "fix"]);
  const tied = { fix: { n: 2, successes: 1, stepsPerSuccess: 30 }, evo: { n: 2, successes: 1, stepsPerSuccess: 30 } };
  assert.deepEqual(rankCandidates(tied, ["fix", "evo"]), ["fix", "evo"]);
  assert.deepEqual(rankCandidates({ fix: { n: 3, successes: 0, stepsPerSuccess: null }, evo: { n: 0 } }, ["fix", "evo"]), ["fix"], "no evidence, no rung");
});

test("the ladder returns to the first capability: [first, other, first]", () => {
  assert.deepEqual(routeLadder(["evo", "fix"]), ["evo", "fix", "evo"]);
  assert.deepEqual(routeLadder(["fix"]), ["fix", "fix"]);
  assert.deepEqual(routeLadder(["a", "b", "c"], 5), ["a", "b", "c", "a"]);
  assert.deepEqual(routeLadder(["a", "b", "c"], 1), ["a"]);
});

test("a route table is built per task from development episodes and verified by its sha256", () => {
  const results = [
    ...episodes("drawer", "fix", [true, false, false]),
    ...episodes("drawer", "evo", [true, true, false], 20),
    ...episodes("bowl", "fix", [true, true]),
  ];
  const table = buildRouteTable(results, { candidates: ["fix", "evo"], createdAt: "2026-01-01T00:00:00Z" });
  assert.equal(table.schema, ROUTE_TABLE_SCHEMA);
  assert.deepEqual(Object.keys(table.tasks), ["bowl", "drawer"]);
  assert.equal(table.tasks.drawer.choice, "evo");
  assert.deepEqual(table.tasks.drawer.ladder, ["evo", "fix", "evo"]);
  assert.deepEqual(table.tasks.bowl.ladder, ["fix", "fix"]);
  assert.equal(verifyRouteTable(table), table);
  assert.equal(table.sha256, routeTableSha256(table));
  assert.throws(() => verifyRouteTable({ ...table, tasks: { ...table.tasks, bowl: { ...table.tasks.bowl, ladder: ["evo"] } } }), /sha256/u);
  assert.throws(() => buildRouteTable([{ task: "x", candidateId: "other", success: true, steps: 1 }], { candidates: ["fix"] }), /unknown candidate/u);
});

test("the router runs the remembered capability first and falls back to candidate order", async () => {
  const table = buildRouteTable(episodes("drawer", "evo", [true]), { candidates: ["fix", "evo"] });
  const router = createRouteMemoryRouter({ table, recoverOn: "exception" });
  assert.deepEqual((await router.route({ task: "drawer", candidates, runs: [] })).candidateId, "evo");
  const unknown = await router.route({ task: "bowl", candidates, runs: [] });
  assert.equal(unknown.candidateId, "fix");
  assert.match(unknown.reason, /no route evidence for bowl/u);
});

test("recoverOn judge asks System 2 after every run and escalates unless it says done", async () => {
  const table = buildRouteTable([...episodes("drawer", "fix", [true]), ...episodes("drawer", "evo", [false])], { candidates: ["fix", "evo"] });
  const router = createRouteMemoryRouter({ table, recoverOn: "judge" });
  let calls = 0;
  const judge = verdict => async () => { calls += 1; return { verdict }; };
  const after = status => [{ candidateId: "fix", status, steps: 3 }];
  assert.deepEqual(await router.route({ task: "drawer", candidates, runs: after("returned"), judge: judge("done") }), { stop: true, reason: "judge_done" });
  assert.equal((await router.route({ task: "drawer", candidates, runs: after("returned"), judge: judge("not_done") })).candidateId, "evo");
  assert.equal((await router.route({ task: "drawer", candidates, runs: after("failed"), judge: judge("not_done") })).candidateId, "evo");
  assert.equal((await router.route({ task: "drawer", candidates, runs: after("returned"), judge: judge("error") })).candidateId, "evo", "a judge error escalates by default");
  assert.equal(calls, 4);
  const stopping = createRouteMemoryRouter({ table, recoverOn: "judge", onJudgeError: "stop" });
  assert.deepEqual(await stopping.route({ task: "drawer", candidates, runs: after("returned"), judge: judge("error") }), { stop: true, reason: "judge_error" });
  await assert.rejects(router.route({ task: "drawer", candidates, runs: after("returned"), judge: null }), /needs a System 2 judge/u);
});

test("recoverOn exception, always and never decide without a judge", async () => {
  const table = buildRouteTable([...episodes("t", "fix", [true]), ...episodes("t", "evo", [false])], { candidates: ["fix", "evo"] });
  const ran = status => [{ candidateId: "fix", status, steps: 1 }];
  const exception = createRouteMemoryRouter({ table, recoverOn: "exception" });
  assert.deepEqual(await exception.route({ task: "t", candidates, runs: ran("returned") }), { stop: true, reason: "run_returned" });
  assert.equal((await exception.route({ task: "t", candidates, runs: ran("failed") })).reason, "run_failed");
  assert.equal((await exception.route({ task: "t", candidates, runs: ran("budget") })).reason, "run_budget");
  const always = createRouteMemoryRouter({ table, recoverOn: "always" });
  assert.equal((await always.route({ task: "t", candidates, runs: ran("returned") })).candidateId, "evo");
  const never = createRouteMemoryRouter({ table, recoverOn: "never" });
  assert.deepEqual(await never.route({ task: "t", candidates, runs: ran("failed") }), { stop: true, reason: "single_run" });
});

test("the ladder ends the episode when every rung has run", async () => {
  const table = buildRouteTable([...episodes("t", "fix", [true]), ...episodes("t", "evo", [false])], { candidates: ["fix", "evo"] });
  const router = createRouteMemoryRouter({ table, recoverOn: "always" });
  const runs = ["fix", "evo", "fix"].map(candidateId => ({ candidateId, status: "returned", steps: 1 }));
  const decision = await router.route({ task: "t", candidates, runs });
  assert.equal(decision.stop, true);
  assert.match(decision.reason, /^ladder_exhausted/u);
  const short = createRouteMemoryRouter({ table, recoverOn: "always", maxRuns: 1 });
  assert.equal((await short.route({ task: "t", candidates, runs: runs.slice(0, 1) })).stop, true);
});
