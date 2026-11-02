#!/usr/bin/env node
// Build a route table (System 1's route memory) from development runs, one evaluation per System 0 capability:
//
//   node tools/route-table.mjs --run fix=outputs/dev-fix --run evo=outputs/dev-evo [--max-runs 3] --output route-table.json
//
// Each --run names a capability id and the output directory of `rea run` / agent.evaluate() for a configuration that ran
// only that capability on development episodes. Episodes that errored or have no boolean success carry no evidence.
// Build tables from development episodes only; evaluate them on different episodes.
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

import { buildRouteTable } from "../src/framework/route-memory.mjs";

/** The episode results of one evaluation directory: [{ task, seed, success, steps }]. */
export async function readEvaluation(directory) {
  const summaryText = await readFile(join(directory, "summary.json"), "utf8");
  const summary = JSON.parse(summaryText);
  const results = [];
  for (const entry of summary.results ?? []) {
    const episode = JSON.parse(await readFile(join(directory, entry.file), "utf8"));
    if (episode.error) continue;
    results.push({ task: episode.task ?? null, seed: episode.seed ?? null, success: episode.success, steps: episode.steps });
  }
  return { results, summarySha256: createHash("sha256").update(summaryText).digest("hex"), episodes: summary.episodesRun ?? results.length };
}

/** Parse ["fix=dir", ...] into [{ candidate, directory }] with unique candidate ids. */
export function parseRuns(values, cwd = process.cwd()) {
  const runs = values.map(value => {
    const index = value.indexOf("=");
    if (index < 1 || index === value.length - 1) throw new Error(`--run takes <capability>=<evaluation directory>, got ${JSON.stringify(value)}`);
    return { candidate: value.slice(0, index), directory: resolve(cwd, value.slice(index + 1)) };
  });
  if (new Set(runs.map(run => run.candidate)).size !== runs.length) throw new Error("each capability may appear in one --run");
  return runs;
}

export async function buildFromRuns(runs, { maxRuns = 3, createdAt } = {}) {
  const results = [];
  const sources = [];
  for (const run of runs) {
    const evaluation = await readEvaluation(run.directory);
    results.push(...evaluation.results.map(result => ({ ...result, candidateId: run.candidate })));
    sources.push({ candidate: run.candidate, directory: run.directory, summarySha256: evaluation.summarySha256, episodes: evaluation.episodes });
  }
  return buildRouteTable(results, { candidates: runs.map(run => run.candidate), maxRuns, sources, ...(createdAt ? { createdAt } : {}) });
}

export async function main(argv = process.argv.slice(2)) {
  const { values } = parseArgs({ args: argv, options: {
    run: { type: "string", multiple: true }, output: { type: "string" }, "max-runs": { type: "string", default: "3" },
    force: { type: "boolean", default: false }, help: { type: "boolean", short: "h", default: false },
  } });
  if (values.help || !values.run?.length || !values.output) {
    process.stdout.write("Usage: node tools/route-table.mjs --run <capability>=<evaluation dir> [--run ...] [--max-runs 3] --output route-table.json [--force]\n");
    return values.help ? 0 : 1;
  }
  const table = await buildFromRuns(parseRuns(values.run), { maxRuns: Number(values["max-runs"]) });
  await writeFile(values.output, `${JSON.stringify(table, null, 2)}\n`, { flag: values.force ? "w" : "wx" });
  for (const [task, entry] of Object.entries(table.tasks)) process.stdout.write(`${task}: ${entry.ladder.join(" -> ") || "(no evidence)"}\n`);
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    process.exitCode = await main();
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
