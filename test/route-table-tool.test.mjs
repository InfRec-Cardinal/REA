import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { createConfiguredSystems } from "../src/framework/systems.mjs";
import { createRegistry } from "../src/framework/registry.mjs";
import { verifyRouteTable } from "../src/framework/route-memory.mjs";
import { buildFromRuns, main, parseRuns } from "../tools/route-table.mjs";

async function evaluation(directory, episodes) {
  await mkdir(directory, { recursive: true });
  const results = episodes.map((episode, index) => ({ id: `e${index}`, file: `${String(index + 1).padStart(4, "0")}.json`, reason: "router_stop", success: episode.success }));
  for (const [index, episode] of episodes.entries()) await writeFile(join(directory, results[index].file), JSON.stringify({ id: `e${index}`, seed: index, ...episode }));
  await writeFile(join(directory, "summary.json"), JSON.stringify({ episodesRun: episodes.length, results }));
}

test("the route-table tool builds a verified table from one development evaluation per capability", async context => {
  const root = await mkdtemp(join(tmpdir(), "rea-route-table-"));
  context.after(() => rm(root, { recursive: true, force: true }));
  await evaluation(join(root, "fix"), [
    { task: "drawer", success: true, steps: 40 }, { task: "drawer", success: false, steps: 90 }, { task: "bowl", success: true, steps: 30 },
  ]);
  await evaluation(join(root, "evo"), [
    { task: "drawer", success: true, steps: 50 }, { task: "drawer", success: true, steps: 60 },
    { task: "bowl", success: null, steps: 0, error: { code: "setup_error", message: "simulator crashed" } },
  ]);
  const table = await buildFromRuns(parseRuns([`fix=${join(root, "fix")}`, `evo=${join(root, "evo")}`]), { createdAt: "2026-01-01T00:00:00Z" });
  verifyRouteTable(table);
  assert.deepEqual(table.tasks.drawer.ladder, ["evo", "fix", "evo"]);
  assert.deepEqual(table.tasks.bowl.ladder, ["fix", "fix"], "an errored episode is no evidence");
  assert.deepEqual(table.sources.map(source => source.candidate), ["fix", "evo"]);
  assert.match(table.sources[0].summarySha256, /^[0-9a-f]{64}$/u);
  assert.throws(() => parseRuns(["fix"]), /<capability>=<evaluation directory>/u);
  assert.throws(() => parseRuns(["fix=a", "fix=b"]), /one --run/u);

  const output = join(root, "route-table.json");
  const write = process.stdout.write;
  process.stdout.write = () => true;
  try {
    assert.equal(await main(["--run", `fix=${join(root, "fix")}`, "--run", `evo=${join(root, "evo")}`, "--output", output]), 0);
    await assert.rejects(main(["--run", `fix=${join(root, "fix")}`, "--output", output]), /EEXIST/u, "an existing table is never overwritten without --force");
  } finally {
    process.stdout.write = write;
  }
  verifyRouteTable(JSON.parse(await readFile(output, "utf8")));

  // A configuration names the table by path; System 1 loads and verifies it, System 2 is the boundary judge.
  const systems = await createConfiguredSystems({
    mode: "routed",
    system0: [{ id: "fix", policy: { type: "demo" } }, { id: "evo", policy: { type: "demo" } }],
    system1: { type: "route-memory", options: { table: "route-table.json", recoverOn: "judge" } },
    system2: { type: "judge", options: { model: "judge-model" } },
  }, { registry: createRegistry(), baseDir: root, env: {} });
  try {
    assert.equal(systems.mode, "routed");
    assert.equal(typeof systems.system2.judge, "function");
    const first = await systems.system1({ task: "drawer", candidates: [{ id: "fix" }, { id: "evo" }], runs: [] });
    assert.equal(first.candidateId, "evo");
  } finally {
    await systems.close();
  }
});
