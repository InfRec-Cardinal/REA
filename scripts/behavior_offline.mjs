#!/usr/bin/env node
import { readFile, mkdir, writeFile, appendFile } from "node:fs/promises";
import { resolve, join } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

import { createPolicy } from "../src/framework/registry.mjs";
import { validateAction } from "../src/framework/actions.mjs";
import { R1PRO_ACTION_SPEC, describeR1ProState, createR1ProHoldAction } from "../src/environments/behavior-actions.mjs";
import { loadCliEnvironment } from "./rea.mjs";
import { readEnv } from "../src/env.mjs";

// Group errors retain their native units; averaging radians with velocity commands
// into a single number would hide what an offline imitation comparison measures.
export const ERROR_GROUPS = {
  baseCommand: { indices: [0, 1, 2], units: "normalized command" },
  trunk: { indices: [3, 4, 5, 6], units: "radians" },
  leftArm: { indices: [7, 8, 9, 10, 11, 12, 13], units: "radians" },
  leftGripper: { indices: [14], units: "normalized command" },
  rightArm: { indices: [15, 16, 17, 18, 19, 20, 21], units: "radians" },
  rightGripper: { indices: [22], units: "normalized command" },
};

export function actionErrors(values, reference) {
  if (reference?.length !== 23 || !reference.every(Number.isFinite)) throw new Error("Reference action must contain 23 finite numbers");
  return Object.fromEntries(Object.entries(ERROR_GROUPS).map(([name, group]) => [name,
    group.indices.reduce((sum, index) => sum + Math.abs(values[index] - reference[index]), 0) / group.indices.length,
  ]));
}

export function policyObservation(sample) {
  const observation = sample.observation;
  return {
    id: sample.id,
    text: observation.text,
    state: describeR1ProState(observation.state),
    images: observation.images,
    actions: [R1PRO_ACTION_SPEC],
  };
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

export async function evaluateOffline({ samples, policy, outputDir, decisionMs = 60000, onResult = () => {}, metadata = {} }) {
  await mkdir(outputDir, { recursive: true });
  const startedAt = new Date().toISOString();
  const manifest = {
    kind: "behavior-demonstration-offline-actions", startedAt,
    policy: policy.id, samples: samples.map(({ id, taskName, episodeIndex, frameIndex }) => ({ id, taskName, episodeIndex, frameIndex })),
    observationModalities: ["RGB", "proprioception"], independentFrames: true,
    decisionMs, actionErrorGroups: ERROR_GROUPS,
    limitations: "Actions are compared with demonstration labels. No action is executed in a simulator. This measures neither task success nor the BEHAVIOR Q score; demonstrations may admit other valid actions.",
    ...metadata,
  };
  await writeFile(join(outputDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
  await writeFile(join(outputDir, "decisions.jsonl"), "");
  const records = [];
  try {
    for (const sample of samples) {
      const observation = policyObservation(sample);
      const hold = createR1ProHoldAction(sample.observation.state);
      const record = {
        id: sample.id, taskName: sample.taskName, episodeIndex: sample.episodeIndex, frameIndex: sample.frameIndex,
        referenceAction: sample.referenceAction,
        holdAction: hold, holdMAE: actionErrors(hold.arguments.values, sample.referenceAction),
        action: null, mae: null, usage: null, error: null,
      };
      const started = performance.now();
      try {
        await policy.reset?.({ task: sample.taskName });
        const result = await policy.decide({
          observation, actions: observation.actions, instruction: observation.text,
          history: [], notes: "", feedback: null,
          budget: { decisionMs }, signal: AbortSignal.timeout(decisionMs),
        });
        record.thought = result.thought ?? null;
        record.usage = result.usage ?? null;
        record.action = validateAction(result.action, observation.actions);
        record.mae = actionErrors(record.action.arguments.values, sample.referenceAction);
      } catch (error) {
        record.usage ??= error.usage ?? null;
        record.error = { code: error.code ?? error.errorType ?? "policy_error", message: error.message };
      }
      record.durationMs = performance.now() - started;
      records.push(record);
      await appendFile(join(outputDir, "decisions.jsonl"), JSON.stringify(record) + "\n");
      await onResult(record);
    }
  } finally {
    await policy.close?.();
  }
  const valid = records.filter(record => record.mae);
  const summary = {
    kind: manifest.kind, startedAt, completedAt: new Date().toISOString(), policy: policy.id,
    samples: records.length, validActions: valid.length, errors: records.length - valid.length,
    actionValidityRate: records.length ? valid.length / records.length : null,
    meanLatencyMs: mean(records.map(record => record.durationMs)),
    perGroupMAE: Object.fromEntries(Object.entries(ERROR_GROUPS).map(([name, group]) => [name, {
      units: group.units,
      agent: mean(valid.map(record => record.mae[name])),
      holdOnSameSamples: mean(valid.map(record => record.holdMAE[name])),
    }])),
    usage: records.some(record => record.usage) ? records.reduce((total, record) => {
      for (const [key, value] of Object.entries(record.usage ?? {})) if (Number.isFinite(value)) total[key] = (total[key] ?? 0) + value;
      return total;
    }, {}) : null,
    usageComplete: records.length > 0 && records.every(record => record.usage),
    simulationSteps: 0, taskSuccessRate: null, qScore: null, limitations: manifest.limitations,
  };
  await writeFile(join(outputDir, "summary.json"), JSON.stringify(summary, null, 2) + "\n");
  return summary;
}

export async function main(argv = process.argv.slice(2)) {
  const { values } = parseArgs({ args: argv, options: {
    samples: { type: "string" }, output: { type: "string" },
    model: { type: "string" }, mode: { type: "string", default: "json" },
    "decision-ms": { type: "string", default: "60000" }, help: { type: "boolean" },
  } });
  if (values.help) {
    process.stdout.write("Usage: node scripts/behavior_offline.mjs --samples samples.jsonl --output outputs/behavior-offline [--model NAME] [--mode json|tool] [--decision-ms 60000]\nUses REA_MODEL/REA_BASE_URL/REA_API_KEY (or REA_BASE_URL/REA_API_KEY) and the unified framework chat policy.\nOffline demonstration action comparison only; no simulation success score.\n");
    return;
  }
  if (!values.samples || !values.output) throw new Error("--samples and --output are required");
  const decisionMs = Number(values["decision-ms"]);
  if (!Number.isFinite(decisionMs) || decisionMs <= 0) throw new Error("--decision-ms must be positive");
  loadCliEnvironment();
  const samples = (await readFile(resolve(values.samples), "utf8")).trim().split("\n").filter(Boolean).map(line => JSON.parse(line));
  const policy = await createPolicy({ type: "chat", options: {
    ...(values.model ? { model: values.model } : {}), mode: values.mode,
    historyLimit: 0,
  } });
  const summary = await evaluateOffline({ samples, policy, outputDir: resolve(values.output), decisionMs,
    metadata: { samplesFile: resolve(values.samples), model: values.model ?? readEnv(process.env, "REA_MODEL"), mode: values.mode },
    onResult: record => process.stdout.write(JSON.stringify({ id: record.id, valid: !record.error, durationMs: record.durationMs, error: record.error }) + "\n"),
  });
  process.stdout.write(JSON.stringify(summary, null, 2) + "\n");
  if (summary.errors) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => { process.stderr.write(error.message + "\n"); process.exitCode = 1; });
}
