#!/usr/bin/env node
// Does the Experience block cost latency? Paired, interleaved trials of the one-token engine with
// 0 / 120 / 240-character experience blocks on the same frame, same model, same session.
//
//   node eval/prompt_latency.mjs --frame path.jpg [--model M] [--trials 30] [--blocks 0,120,240]
//                                [--variant short] [--depth 2.5,2.5,2.5] [--out eval/out/prompt_latency.jsonl]
//
// Needs REA_API_KEY (and REA_BASE_URL) in the environment or in --env .env. Talks to the provider
// only; no robot, no service. Reports median / p90 ttfb_s and token_ready_s per arm and the paired
// median difference against the 0-character arm. >= 30 trials per arm is the minimum worth quoting.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

import { createTokenInfer } from "../src/infer-token.mjs";
import { createDispatcher } from "../src/provider.mjs";

const LESSONS = [
  "[-] centre near, left far: W was vetoed (depth_center_near); prefer Q",
  "[-] centre near, right near: alternating Q/E without progress; commit to one side or stop",
  "[+] centre mid: W advanced 0.42 m safely",
  "[-] centre near: reviewer replaced W with Q — the door is on the left",
  "[-] centre mid, lidar lidar_left_near: A was vetoed (lidar_left_near); prefer E",
];

function arg(name, fallback = null) {
  const index = process.argv.indexOf(`--${name}`);
  if (index === -1) return fallback;
  return process.argv[index + 1] ?? fallback;
}

async function loadEnvFile(path) {
  try {
    const text = await readFile(path, "utf8");
    // The subset of POSIX `. ./.env` used here: optional `export`, KEY=value, quoted values kept whole, an unquoted
    // trailing `# comment` dropped.
    for (const line of text.split(/\r?\n/u)) {
      const match = /^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/u.exec(line);
      if (!match || process.env[match[1]] !== undefined) continue;
      let value = match[2];
      const quoted = /^(["'])(.*)\1\s*(#.*)?$/u.exec(value);
      if (quoted) value = quoted[2];
      else value = value.replace(/\s+#.*$/u, "");
      process.env[match[1]] = value;
    }
  } catch {
    /* optional */
  }
}

function block(chars) {
  if (chars <= 0) return null;
  let text = "";
  for (let i = 0; text.length < chars; i += 1) text += (text ? "\n" : "") + LESSONS[i % LESSONS.length];
  return text.slice(0, chars);
}

function quantile(values, q) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return Number((sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)).toFixed(4));
}

async function main() {
  await loadEnvFile(arg("env", ".env"));
  const apiKey = process.env.REA_API_KEY ?? "";
  const baseUrl = process.env.REA_BASE_URL ?? "https://api.openai.com/v1";
  const model = arg("model", process.env.G1_PRIMARY_MODEL ?? process.env.REA_MODEL);
  const framePath = arg("frame");
  const trials = Number(arg("trials", 30));
  const blocks = String(arg("blocks", "0,120,240")).split(",").map(Number);
  const variant = arg("variant", "short");
  const depth = String(arg("depth", "2.5,2.5,2.5")).split(",").map(Number);
  const out = arg("out", "eval/out/prompt_latency.jsonl");
  if (!apiKey) {
    console.error("REA_API_KEY is not set (pass --env .env or export it). This script talks to the provider; nothing runs without a key.");
    process.exit(2);
  }
  if (!framePath) {
    console.error("--frame path.jpg is required (any JPEG the robot might see)");
    process.exit(2);
  }
  const frame = {
    imageB64: (await readFile(framePath)).toString("base64"),
    mimeType: "image/jpeg",
    depth: { center_m: depth[0], left_m: depth[1], right_m: depth[2], front_min_m: Math.min(...depth) * 0.9 },
  };
  const infer = createTokenInfer({ apiKey, baseUrl, dispatcher: await createDispatcher() });
  await mkdir(dirname(out), { recursive: true });
  const records = [];
  const instruction = arg("instruction", "Move toward the nearest passable doorway or opening while avoiding collision.");

  console.error(`model=${model} trials=${trials} blocks=${blocks.join(",")} variant=${variant} frame=${framePath}`);
  for (let trial = 0; trial < trials; trial += 1) {
    // interleave the arms within every trial so provider drift lands on all of them alike
    const order = trial % 2 === 0 ? blocks : [...blocks].reverse();
    for (const chars of order) {
      const experience = block(chars);
      const started = Date.now();
      let record;
      try {
        const result = await infer(model, frame, { instruction, experience, deadlineMs: 15000, cfg: { promptVariant: variant } });
        record = { trial, chars, prompt_chars: experience?.length ?? 0, ok: true, token: result.token, ttfb_s: result.timings?.ttfb_s ?? null, token_ready_s: result.tokenReadyS, source: result.source, at: started };
      } catch (error) {
        record = { trial, chars, prompt_chars: experience?.length ?? 0, ok: false, error_type: error?.errorType ?? "error", error: error?.message ?? String(error), at: started };
      }
      records.push(record);
      await writeFile(out, `${records.map(r => JSON.stringify(r)).join("\n")}\n`, "utf8");
      process.stderr.write(`\rtrial ${trial + 1}/${trials} block ${chars}: ${record.ok ? `${record.token} ${record.token_ready_s}s` : record.error_type}          `);
    }
  }
  process.stderr.write("\n");

  const summary = {};
  for (const chars of blocks) {
    const arm = records.filter(r => r.chars === chars && r.ok);
    summary[chars] = {
      n: arm.length,
      failed: records.filter(r => r.chars === chars && !r.ok).length,
      ttfb_median_s: quantile(arm.map(r => r.ttfb_s).filter(Number.isFinite), 0.5),
      ttfb_p90_s: quantile(arm.map(r => r.ttfb_s).filter(Number.isFinite), 0.9),
      token_ready_median_s: quantile(arm.map(r => r.token_ready_s), 0.5),
      token_ready_p90_s: quantile(arm.map(r => r.token_ready_s), 0.9),
      tokens: arm.reduce((acc, r) => ({ ...acc, [r.token]: (acc[r.token] ?? 0) + 1 }), {}),
    };
  }
  const base = blocks[0];
  for (const chars of blocks.slice(1)) {
    const diffs = [];
    for (let trial = 0; trial < trials; trial += 1) {
      const a = records.find(r => r.trial === trial && r.chars === base && r.ok);
      const b = records.find(r => r.trial === trial && r.chars === chars && r.ok);
      if (a && b) diffs.push(b.token_ready_s - a.token_ready_s);
    }
    summary[chars].paired_vs_base_median_s = quantile(diffs, 0.5);
    summary[chars].paired_vs_base_p90_s = quantile(diffs, 0.9);
    summary[chars].paired_n = diffs.length;
  }
  const report = { model, variant, frame: framePath, trials, blocks, summary, at: new Date().toISOString() };
  await writeFile(out.replace(/\.jsonl$/u, "") + ".summary.json", `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log("\nblock  n   ttfb med  ttfb p90  ready med  ready p90  paired Δ med   tokens");
  for (const chars of blocks) {
    const s = summary[chars];
    console.log(`${String(chars).padEnd(6)} ${String(s.n).padEnd(3)} ${String(s.ttfb_median_s).padEnd(9)} ${String(s.ttfb_p90_s).padEnd(9)} ${String(s.token_ready_median_s).padEnd(10)} ${String(s.token_ready_p90_s).padEnd(10)} ${String(s.paired_vs_base_median_s ?? "—").padEnd(14)} ${JSON.stringify(s.tokens)}`);
  }
  console.log(`\nwrote ${out} and its .summary.json`);
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
