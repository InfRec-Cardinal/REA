import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { createInterface } from "node:readline";

import { createEmbodiedAgent } from "./agent.mjs";
import { createBridge } from "./bridge.mjs";
import { BRIDGE_DESCRIPTION, createRegistry } from "./registry.mjs";

// The evaluator owns the episode length; REA's own budget should not cut an episode short unless a configuration sets it.
const SERVE_BUDGET = Object.freeze({ maxSteps: 100000, maxDurationMs: 24 * 3600 * 1000 });

function summaryOf(value) {
  if (!value || typeof value !== "object") return value ?? null;
  const { finalObservation: _final, ...rest } = value;
  return rest;
}

/**
 * Serve a configured REA agent (one policy, or async / routed systems) to an external evaluator over JSON lines:
 * one request per input line, { id, op, ... }, one reply per output line, { id, ok: true, result } or { id, ok: false, error }.
 *
 *   configure  { actions, timing?, name?, resumable? }      the action specifications the evaluator accepts
 *   reset      { task?, seed?, instruction? }               start an episode (ends the previous one)
 *   step       { observation, terminated?, truncated?,      deliver an observation and wait for the next action:
 *                success?, reward?, info? }                 { action } or { done: true, reason, summary }
 *   end        { result? }                                  end the episode; result.success is recorded as its outcome
 *   close                                                   end, close the agent and stop serving
 *
 * Observations use REA's format: { text, state, images: [{ name, mimeType, data }], ... }. Nothing but protocol lines
 * is written to `output`.
 */
export async function servePolicy({
  config, baseDir, env = process.env, registry = createRegistry(), input = process.stdin, output = process.stdout,
  outputDir = null,
} = {}) {
  const bridge = createBridge();
  registry.registerEnvironment("bridge", () => bridge.environment, BRIDGE_DESCRIPTION);
  // The configured environment is replaced by the bridge; JavaScript configurations keep their policy or systems factories.
  const factory = ["createEnvironment", "createPolicy", "createSystems"].some(key => typeof config[key] === "function");
  const served = factory
    ? { ...config, createEnvironment: () => bridge.environment, budget: { ...SERVE_BUDGET, ...config.budget } }
    : { ...config, environment: { type: "bridge" }, budget: { ...SERVE_BUDGET, ...config.budget } };
  const agent = await createEmbodiedAgent(served, { baseDir, env, registry });
  if (outputDir) await mkdir(outputDir, { recursive: true });
  let current = null;
  let episode = 0;
  let closed = false;

  const write = message => output.write(`${JSON.stringify(message)}\n`);
  const prefix = () => String(episode).padStart(4, "0");

  async function settle(externalResult) {
    if (!current) return null;
    bridge.end(externalResult);
    const outcome = await current.done;
    current = null;
    if (outputDir) await writeFile(join(outputDir, `${prefix()}.json`), `${JSON.stringify({ ...outcome, external: externalResult ?? null }, null, 2)}\n`);
    return outcome;
  }

  async function start(request) {
    await settle(null);
    episode = bridge.begin();
    const tracePath = outputDir ? join(outputDir, `${prefix()}.jsonl`) : null;
    if (tracePath) await writeFile(tracePath, "");
    const run = agent.run({
      ...(request.task !== undefined ? { task: request.task } : {}),
      ...(request.seed !== undefined ? { seed: request.seed } : {}),
      ...(request.instruction !== undefined ? { instruction: request.instruction } : {}),
      onTransition: tracePath ? record => appendFile(tracePath, `${JSON.stringify(record, (key, value) => key === "images" ? undefined : value)}\n`) : undefined,
    });
    const done = run.then(summary => {
      const outcome = { episode, reason: summary.reason, summary: summaryOf(summary) };
      bridge.finish(outcome);
      return outcome;
    }, error => {
      const outcome = { episode, reason: "error", error: error?.message ?? String(error), summary: null };
      bridge.finish(outcome);
      return outcome;
    });
    current = { done };
    return { episode };
  }

  async function handle(request) {
    switch (request.op) {
      case "configure": {
        const { actions, timing, name, resumable } = request;
        return { ...bridge.configure({ actions, timing, name, resumable }), mode: config.systems?.mode ?? "policy" };
      }
      case "reset": return start(request);
      case "step":
        if (!current) throw new Error("send reset before the first step");
        return bridge.step(request);
      case "end": {
        const outcome = await settle(request.result ?? null);
        return outcome ?? { episode, reason: "no_episode", summary: null };
      }
      case "close":
        await settle(null);
        closed = true;
        return { closed: true };
      default:
        throw new Error(`unknown op ${JSON.stringify(request.op)}`);
    }
  }

  const lines = createInterface({ input, crlfDelay: Infinity });
  try {
    for await (const line of lines) {
      if (!line.trim()) continue;
      let request;
      try {
        request = JSON.parse(line);
      } catch {
        write({ id: null, ok: false, error: "request is not JSON" });
        continue;
      }
      try {
        write({ id: request.id ?? null, ok: true, result: await handle(request) });
      } catch (error) {
        write({ id: request.id ?? null, ok: false, error: error?.message ?? String(error) });
      }
      if (closed) break;
    }
  } finally {
    lines.close();
    await settle(null).catch(() => {});
    await agent.close();
  }
  return { episodes: episode };
}
