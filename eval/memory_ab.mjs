#!/usr/bin/env node
// Cold vs hot memory, session by session, against a running service. Preview-only: every step is sent
// with execute:false, so this never moves a robot. What it measures is how the brain decides — token
// distribution, refusal rate, escalations, deliberate fraction, judge scores, lessons/episodes used,
// decision latency — with the memory cleared before each "cold" session and left to accumulate for
// each "hot" one. Arms alternate so provider drift lands on both.
//
//   node eval/memory_ab.mjs [--base http://127.0.0.1:8091] [--token BEARER] [--sessions 6] [--steps 12]
//                           [--instruction "..."] [--arms cold,hot] [--engine token] [--out eval/out/memory_ab.jsonl]
//                           [--yes] [--keep-episodes]
//
// The cold arm needs POST /internal/brain/reset, which is loopback-only: run this on the same host as
// the service. That reset WIPES the service's persisted memory (experience, routing scores and, unless
// --keep-episodes, the episode store); the files are rotated to *.bak-<unix_ms> first, and this script
// refuses to start against a store that already holds lessons unless --yes is given. Outcomes (did the robot get through the door) need execution and the robot; with the
// replay worker the environment does not change between steps, so read the numbers as "what the
// controller would have asked for", not as task success. >= 6 sessions per arm before quoting anything.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

function arg(name, fallback = null) {
  const index = process.argv.indexOf(`--${name}`);
  if (index === -1) return fallback;
  return process.argv[index + 1] ?? fallback;
}

function quantile(values, q) {
  const clean = values.filter(Number.isFinite);
  if (!clean.length) return null;
  const sorted = [...clean].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return Number((sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)).toFixed(4));
}

async function main() {
  const base = String(arg("base", "http://127.0.0.1:8091")).replace(/\/+$/u, "");
  const token = arg("token", process.env.G1_AGENT_TOKEN ?? "");
  const sessionsPerArm = Number(arg("sessions", 6));
  const steps = Number(arg("steps", 12));
  const instruction = arg("instruction", "Move toward the nearest passable doorway or opening while avoiding collision.");
  const arms = String(arg("arms", "cold,hot")).split(",");
  const engine = arg("engine", null);
  const out = arg("out", "eval/out/memory_ab.jsonl");
  const yes = process.argv.includes("--yes");
  const keepEpisodes = process.argv.includes("--keep-episodes");
  const headers = { "content-type": "application/json", ...(token ? { authorization: `Bearer ${token}` } : {}) };
  const post = async (path, body) => {
    const response = await fetch(`${base}${path}`, { method: "POST", headers, body: JSON.stringify(body) });
    return { status: response.status, body: await response.json().catch(() => ({})) };
  };

  const health = await fetch(`${base}/health`).then(r => r.json());
  if (!health.brain?.enabled) {
    console.error("the service's brain is disabled (G1_BRAIN_ENABLED=0); nothing to compare");
    process.exit(2);
  }
  console.error(`service ${base} engine=${health.engine} ready=${health.ready} lessons=${health.brain.lessons} episodes=${health.brain.episodes ?? "?"}`);
  if (arms.includes("cold") && !yes && ((health.brain.lessons ?? 0) > 0 || (health.brain.episodes ?? 0) > 0)) {
    console.error("the service already holds learned memory; the cold arm would wipe it (rotated to *.bak first). Re-run with --yes to proceed, or --arms hot.");
    process.exit(2);
  }
  await mkdir(dirname(out), { recursive: true });
  const records = [];
  const digests = [];

  for (let round = 0; round < sessionsPerArm; round += 1) {
    for (const arm of round % 2 === 0 ? arms : [...arms].reverse()) {
      if (arm === "cold") {
        const reset = await post("/internal/brain/reset", { confirm: true, experience: true, router: true, episodes: !keepEpisodes });
        if (reset.status !== 202) {
          console.error(`cold arm: POST /internal/brain/reset returned ${reset.status} ${JSON.stringify(reset.body)} — run this on the service host`);
          process.exit(2);
        }
      }
      let sessionId = null;
      for (let step = 1; step <= steps; step += 1) {
        const started = Date.now();
        const body = { instruction, execute: false, ...(sessionId ? { session_id: sessionId } : {}), ...(engine ? { cfg: { engine } } : {}) };
        const { status, body: res } = await post("/v1/step", body);
        sessionId = res.session_id ?? sessionId;
        const record = {
          arm,
          round,
          session_id: sessionId,
          step,
          http: status,
          ok: res.ok ?? null,
          token: res.token ?? null,
          engine: res.engine ?? null,
          route: res.brain?.route ?? null,
          escalated: res.brain?.escalated ?? null,
          inline: res.brain?.inline?.ok ?? null,
          allowed: res.gate?.allowed ?? null,
          reasons: res.gate?.reasons ?? [],
          judge_score: res.brain?.judge?.score ?? null,
          judge_outcome: res.brain?.judge?.outcome ?? null,
          lessons_used: res.brain?.lessons_used ?? null,
          episodes_used: res.brain?.episodes_used ?? null,
          revisits: res.brain?.spatial?.revisits ?? null,
          token_ready_s: res.timing?.token_ready_s ?? null,
          total_s: res.timing?.total_s ?? null,
          wall_s: (Date.now() - started) / 1000,
          error_type: res.error_type ?? null,
          at: started,
        };
        records.push(record);
        process.stderr.write(`\r${arm} round ${round + 1}/${sessionsPerArm} step ${step}/${steps}: ${record.token ?? "?"} ${record.engine ?? ""} ${record.allowed === false ? "REFUSED" : ""} ${record.total_s ?? ""}s      `);
      }
      const digest = sessionId ? (await post("/v1/brain/session", { session_id: sessionId })).body : null;
      if (digest?.digest) digests.push({ arm, round, ...digest.digest, spatial: digest.spatial });
      await writeFile(out, `${records.map(r => JSON.stringify(r)).join("\n")}\n`, "utf8");
    }
  }
  process.stderr.write("\n");

  const summary = {};
  for (const arm of arms) {
    const rows = records.filter(r => r.arm === arm && r.http === 200);
    const decided = rows.filter(r => r.ok !== null);
    summary[arm] = {
      sessions: sessionsPerArm,
      steps: rows.length,
      refusal_rate: decided.length ? Number((decided.filter(r => r.allowed === false).length / decided.length).toFixed(3)) : null,
      escalation_rate: decided.length ? Number((decided.filter(r => r.escalated === true).length / decided.length).toFixed(3)) : null,
      deliberate_fraction: decided.length ? Number((decided.filter(r => r.engine === "react").length / decided.length).toFixed(3)) : null,
      inline_rescues: decided.filter(r => r.inline === true).length,
      mean_judge_score: decided.length ? Number((decided.map(r => r.judge_score ?? 0).reduce((a, b) => a + b, 0) / decided.length).toFixed(3)) : null,
      lessons_used_mean: decided.length ? Number((decided.map(r => r.lessons_used ?? 0).reduce((a, b) => a + b, 0) / decided.length).toFixed(2)) : null,
      episodes_used_mean: decided.length ? Number((decided.map(r => r.episodes_used ?? 0).reduce((a, b) => a + b, 0) / decided.length).toFixed(2)) : null,
      token_ready_median_s: quantile(decided.map(r => r.token_ready_s), 0.5),
      token_ready_p90_s: quantile(decided.map(r => r.token_ready_s), 0.9),
      total_median_s: quantile(decided.map(r => r.total_s), 0.5),
      tokens: decided.reduce((acc, r) => ({ ...acc, [r.token ?? "?"]: (acc[r.token ?? "?"] ?? 0) + 1 }), {}),
      errors: rows.filter(r => r.error_type).length,
    };
  }
  const report = { base, instruction, engine, sessions_per_arm: sessionsPerArm, steps_per_session: steps, arms, cold_keeps_episodes: keepEpisodes, summary, digests, at: new Date().toISOString() };
  await writeFile(out.replace(/\.jsonl$/u, "") + ".summary.json", `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log("\narm   steps refusal escal. delib. inline judge  lessons episodes ready med  total med  tokens");
  for (const arm of arms) {
    const s = summary[arm];
    console.log(`${arm.padEnd(5)} ${String(s.steps).padEnd(5)} ${String(s.refusal_rate).padEnd(7)} ${String(s.escalation_rate).padEnd(6)} ${String(s.deliberate_fraction).padEnd(6)} ${String(s.inline_rescues).padEnd(6)} ${String(s.mean_judge_score).padEnd(6)} ${String(s.lessons_used_mean).padEnd(7)} ${String(s.episodes_used_mean).padEnd(8)} ${String(s.token_ready_median_s).padEnd(10)} ${String(s.total_median_s).padEnd(10)} ${JSON.stringify(s.tokens)}`);
  }
  console.log(`\nwrote ${out} and its .summary.json`);
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
