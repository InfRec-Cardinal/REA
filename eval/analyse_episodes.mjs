#!/usr/bin/env node
// Offline: does the controller get better across sessions on the same task? Reads the episode store
// (runtime/memory/episodes.jsonl by default) and prints, per task, one line per session in time order
// plus a first-half / second-half comparison of refusal rate, escalations and judge score.
//
//   node eval/analyse_episodes.mjs [path/to/episodes.jsonl] [--task reach-doorway] [--json]
//
// Evidence, not proof: sessions differ in where the robot started and what the operator did. Use it
// to see whether the trend is there before designing the controlled comparison (eval/README.md).

import { createEpisodeStore } from "../src/episodes.mjs";

function arg(name, fallback = null) {
  const index = process.argv.indexOf(`--${name}`);
  if (index === -1) return fallback;
  return process.argv[index + 1] ?? fallback;
}

function mean(values) {
  const clean = values.filter(Number.isFinite);
  return clean.length ? Number((clean.reduce((a, b) => a + b, 0) / clean.length).toFixed(3)) : null;
}

async function main() {
  const VALUED = new Set(["--task"]); // flags that take a value; --json is a switch
  const positional = process.argv.slice(2).filter((value, index, all) => !value.startsWith("--") && !(index > 0 && VALUED.has(all[index - 1])));
  const path = positional[0] ?? "runtime/memory/episodes.jsonl";
  const store = createEpisodeStore({ path, maxSessions: 100000 });
  const loaded = await store.load();
  if (!loaded.sessions) {
    console.error(`no sessions in ${path}`);
    process.exit(2);
  }
  const wanted = arg("task", null);
  const digests = store.sessions().map(id => store.digest(id)).filter(d => d && d.steps > 0 && (!wanted || d.task_key === wanted));
  const byTask = new Map();
  for (const d of digests) {
    if (!byTask.has(d.task_key)) byTask.set(d.task_key, []);
    byTask.get(d.task_key).push(d);
  }
  const report = {};
  for (const [task, list] of byTask) {
    list.sort((a, b) => (a.started_at ?? 0) - (b.started_at ?? 0));
    const half = Math.floor(list.length / 2);
    const early = list.slice(0, half);
    const late = list.slice(half);
    const agg = rows => ({
      sessions: rows.length,
      refusal_rate: mean(rows.map(r => r.refusal_rate)),
      escalations_per_session: mean(rows.map(r => r.escalations)),
      deliberate_per_session: mean(rows.map(r => r.deliberate)),
      mean_judge_score: mean(rows.map(r => r.mean_score)),
      steps_per_session: mean(rows.map(r => r.steps)),
      reviewer_stop_or_correct: rows.reduce((sum, r) => sum + (r.reviews?.STOP ?? 0) + (r.reviews?.CORRECT ?? 0), 0),
    });
    report[task] = { sessions: list.map(d => ({ session_id: d.session_id, started_at: d.started_at, steps: d.steps, refusal_rate: d.refusal_rate, escalations: d.escalations, deliberate: d.deliberate, mean_score: d.mean_score, last_outcome: d.last_outcome, reviews: d.reviews, max_revisits: d.max_revisits })), early: agg(early), late: agg(late) };
  }
  if (process.argv.includes("--json")) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }
  for (const [task, r] of Object.entries(report)) {
    console.log(`\n== ${task} (${r.sessions.length} sessions) ==`);
    console.log("started (UTC)      steps refusal escal delib judge   ended        revisits reviews");
    for (const s of r.sessions) {
      const when = s.started_at ? new Date(s.started_at).toISOString().slice(0, 16).replace("T", " ") : "?";
      console.log(`${when.padEnd(18)} ${String(s.steps).padEnd(5)} ${String(s.refusal_rate ?? "—").padEnd(7)} ${String(s.escalations).padEnd(5)} ${String(s.deliberate).padEnd(5)} ${String(s.mean_score ?? "—").padEnd(7)} ${String(s.last_outcome ?? "?").padEnd(12)} ${String(s.max_revisits).padEnd(8)} S${s.reviews?.STOP ?? 0}/C${s.reviews?.CORRECT ?? 0}/K${s.reviews?.KEEP ?? 0}`);
    }
    console.log(`first half : ${JSON.stringify(r.early)}`);
    console.log(`second half: ${JSON.stringify(r.late)}`);
  }
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
