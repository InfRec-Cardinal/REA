import { appendFile, mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { LineWorker } from "../../../src/workers.mjs";
import { runAsyncEpisode } from "../../../src/framework/async-runtime.mjs";
import { createSuccessRouter } from "../../../src/policies/success-router.mjs";
import { createRobotwinCodePolicy } from "../../../src/policies/robotwin-code.mjs";
import { createRobotwinVlaPolicy } from "../../../src/policies/robotwin-vla.mjs";

const here = dirname(fileURLToPath(import.meta.url));
// VLA candidates are the keys of config.vlaEndpoints; VLA_IDS are the services of the reference setup.
export const VLA_IDS = ["pi05", "gigabrain", "spatial_forcing", "xvla", "xwam", "patchwam", "medex"];
const terminalReasons = ["success", "max_steps", "environment_limit", "terminated", "policy_output_limit", "policy_refusal"];

export function withCodeDecisionBudget(policy, decisionLimit) {
  return {
    ...policy,
    decide(input) {
      return policy.decide({ ...input, budget: { ...input.budget,
        stepsRemaining: Math.min(input.budget.stepsRemaining, decisionLimit - input.step),
      } });
    },
  };
}

export function withCodeReplay(policy, instanceId, replay) {
  if (replay?.instanceId !== instanceId) return policy;
  return {
    ...policy,
    async decide(input) {
      if (input.step !== 0) return policy.decide(input);
      const saved = JSON.parse((await readFile(replay.decisionFile, "utf8")).trim().split("\n")[0]);
      if (saved.step !== 0 || saved.observationId !== `${instanceId}:0` || saved.candidate !== "code") {
        throw new Error("Code replay must reference this instance's recorded first code decision");
      }
      const { step, candidate, observationId, ...decision } = saved;
      return { ...decision, replay: {
        decisionFile: replay.decisionFile, instanceId, sourceStep: step,
        apiCalled: false, pythonExecuted: false,
      } };
    },
  };
}

export function classifyPolicyOutcome(result) {
  if (result.reason === "error" && result.error?.code === "codex_error"
      && /\b403 Forbidden:\s*This request contains prohibited content\b/i.test(result.error.message ?? "")) {
    return { ...result, reason: "policy_refusal", success: false,
      outcomeClassification: "The provider explicitly refused the request as prohibited content; episode failed without retry." };
  }
  if (result.reason === "error" && result.error?.code === "output_limit") {
    const tokenLimit = result.codeProtocol ? result.codeProtocol.maxOutputTokens : 2048;
    return { ...result, reason: "policy_output_limit", success: false,
      outcomeClassification: tokenLimit === null
        ? "The provider exhausted its output budget before completing an action; episode failed without retry."
        : `No complete action within the fixed ${tokenLimit}-token policy output budget; episode failed without retry.` };
  }
  return result;
}

export function createHybridEnvironment({ config, instance, output, worker = null }) {
  const transport = worker ?? new LineWorker("robotwin-environment", config.simulatorPython, [join(here, "environment.py")], {
    env: { ...process.env, ...config.simulatorEnv }, autoRestart: false,
  });
  let current;
  let codeDecisions = 0;
  let closed = false;
  const request = async (payload, signal) => {
    signal?.throwIfAborted();
    const abort = () => transport.stop();
    signal?.addEventListener("abort", abort, { once: true });
    try {
      const reply = await transport.request(payload, config.environmentTimeoutMs ?? 660000);
      signal?.throwIfAborted();
      if (!reply.ok) throw new Error(reply.error);
      return reply.result;
    } finally {
      signal?.removeEventListener("abort", abort);
    }
  };
  return {
    id: "robotwin-frozen-hybrid", timing: "turn-based",
    async reset({ signal } = {}) {
      codeDecisions = 0;
      current = await request({ op: "reset", subset: config.subset, harnessRoot: config.harnessRoot, instance, output }, signal);
      return current;
    },
    async step(action, { signal } = {}) {
      current = await request({ op: "step", action }, signal);
      if (action.name !== "robotwin_vla_chunk") codeDecisions += 1;
      if (!current.success && codeDecisions >= (config.codeDecisions ?? 40)) {
        current = { ...current, truncated: true, info: { ...current.info, codeDecisionLimit: codeDecisions } };
      }
      return current;
    },
    async stop() {
      if (closed) return;
      closed = true;
      try { await request({ op: "close" }); } finally { transport.stop(); }
    },
  };
}

export async function runInstance({ config, instance, output, mode = "hybrid", env = process.env }) {
  await mkdir(output, { recursive: true });
  const vlaIds = Object.keys(config.vlaEndpoints ?? {});
  const ids = mode === "hybrid" ? [...vlaIds, "code"] : mode === "vla" ? vlaIds : [mode];
  if (ids.some(id => ![...vlaIds, "code"].includes(id))) throw new Error(`Unknown policy mode ${mode}`);
  const policies = [];
  let environment;
  try {
    if (mode === "hybrid" || mode === "vla") {
      const health = await fetch(new URL("/health", config.routerEndpoint), { signal: AbortSignal.timeout(10000) });
      if (!health.ok) throw new Error(`Router health failed: HTTP ${health.status}`);
      const report = await health.json();
      const missing = ids.filter(id => !report.trained_candidates.includes(id));
      if (missing.length) throw new Error(`Router checkpoint lacks trained candidates: ${missing.join(", ")}`);
      await writeFile(join(output, "router.json"), JSON.stringify(report, null, 2));
    }
    const system0 = ids.map(id => {
      const policy = id === "code"
        ? withCodeDecisionBudget(withCodeReplay(createRobotwinCodePolicy({
          python: config.clientPython, model: config.codeModel ?? env.REA_CODE_MODEL ?? env.REA_MODEL,
          apiKey: env.REA_API_KEY, baseUrl: env.REA_BASE_URL,
          backend: config.codeBackend ?? "chat", codexPath: config.codeCodexPath,
          apiRetries: config.codeApiRetries,
          provider: config.codeProvider,
          providerBaseUrl: config.codeProviderBaseUrl, apiKeyFile: config.codeApiKeyFile,
          workRoot: config.codeWorkRoot ? join(config.codeWorkRoot, instance.instance_id) : undefined,
        }), instance.instance_id, config.codeReplay), config.codeDecisions ?? 40)
        : createRobotwinVlaPolicy({ model: id, url: config.vlaEndpoints[id], python: config.clientPython });
      policies.push(policy);
      return { id, kind: id === "code" ? "code" : "vla", policy: {
        ...policy,
        async decide(input) {
          const decision = await policy.decide(input);
          await appendFile(join(output, "decisions.jsonl"), JSON.stringify({
            step: input.step, candidate: id, observationId: input.observation.id, ...decision,
          }) + "\n");
          return decision;
        },
      } };
    });
    environment = createHybridEnvironment({ config, instance, output });
    const router = createSuccessRouter({ endpoint: config.routerEndpoint, requestTimeoutMs: 120000 }, { episode: { id: instance.instance_id } });
    const route = ids.length === 1
      ? async () => ({ candidateId: ids[0], reason: "Fixed-policy comparison" })
      : input => router.route(input);
    const result = await runAsyncEpisode({
      environment, system0, system1: route, task: instance.task, seed: instance.seed, instruction: instance.instruction,
      maxSteps: config.maxDecisions ?? 1500, maxDurationMs: config.episodeTimeoutMs ?? 7200000,
      decisionTimeoutMs: config.decisionTimeoutMs ?? 240000,
      onEvent: event => {
        if (event.type === "routing" && event.status === "completed") process.stdout.write(`${JSON.stringify({ instance: instance.instance_id, ...event })}\n`);
      },
      onTransition: transition => appendFile(join(output, "transitions.jsonl"), JSON.stringify(transition, (key, value) => key === "images" ? undefined : value) + "\n"),
    });
    const record = classifyPolicyOutcome({ ...result, instance, mode,
      protocol: ids.includes("code") ? "rea-bce-vla-code-v2" : "rea-bce-vla-code-v1",
      codeModel: config.codeModel ?? env.REA_CODE_MODEL ?? env.REA_MODEL ?? null,
      ...(ids.includes("code") ? {
        ...(config.codeReplay?.instanceId === instance.instance_id ? { codeReplay: config.codeReplay } : {}),
        codeProtocol: {
        version: 2,
        backend: config.codeBackend ?? "chat",
        decisionLimit: config.codeDecisions ?? 40,
        promptBudget: "Remaining code decisions, capped by the framework episode budget",
        countsInvalidPrograms: true,
        decisionTimeoutMs: config.decisionTimeoutMs ?? 240000,
        maxOutputTokens: config.codeBackend === "codex" ? null : 2048,
        reasoning: config.codeBackend === "codex" ? "cli_default" : "provider_default",
        temperature: config.codeBackend === "codex" ? "cli_default" : 0,
      }, codeProvider: {
        name: config.codeProvider ?? env.REA_CODE_PROVIDER ?? null,
        transport: config.codeBackend === "codex" ? "codex_cli_responses" : env.REA_CODE_TRANSPORT ?? "chat_completions",
        endpoint: config.codeProviderBaseUrl ?? env.REA_CODE_UPSTREAM ?? env.REA_BASE_URL,
      } } : {}),
    });
    await writeFile(join(output, "result.json"), JSON.stringify(record, null, 2));
    return record;
  } finally {
    await environment?.stop();
    await Promise.all(policies.map(policy => policy.close?.()));
  }
}

async function main() {
  const { values } = parseArgs({ options: {
    config: { type: "string" }, instances: { type: "string" }, output: { type: "string" },
    mode: { type: "string", default: "hybrid" }, limit: { type: "string" },
    "skip-errors": { type: "boolean", default: false },
  } });
  if (!values.config || !values.instances || !values.output) throw new Error("Required: --config FILE --instances FILE --output DIR [--mode hybrid|vla|code|MODEL]");
  const config = JSON.parse(await readFile(values.config, "utf8"));
  const rows = JSON.parse(await readFile(values.instances, "utf8"));
  const instances = values.limit ? rows.slice(0, Number(values.limit)) : rows;
  const output = resolve(values.output);
  await mkdir(output, { recursive: true });
  await writeFile(join(output, "instances.json"), JSON.stringify(instances, null, 2));
  const results = [];
  for (const instance of instances) {
    const directory = join(output, instance.instance_id);
    let result;
    try {
      result = JSON.parse(await readFile(join(directory, "result.json"), "utf8"));
      if (result.mode !== values.mode) throw new Error("Existing run uses a different policy mode");
      const classified = classifyPolicyOutcome(result);
      if (classified !== result) {
        result = classified;
        await writeFile(join(directory, "result.json"), JSON.stringify(result, null, 2));
      }
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
    if (!result || (!terminalReasons.includes(result.reason) && !values["skip-errors"])) {
      if (existsSync(directory)) {
        const attempts = join(output, "attempts");
        await mkdir(attempts, { recursive: true });
        await rename(directory, join(attempts, `${instance.instance_id}-${Date.now()}`));
      }
      try { result = await runInstance({ config, instance, output: directory, mode: values.mode }); }
      catch (failure) {
        result = { instance, mode: values.mode, success: null, reason: "infrastructure_error", error: failure.message };
        await mkdir(directory, { recursive: true });
        await writeFile(join(directory, "result.json"), JSON.stringify(result, null, 2));
      }
    }
    results.push(result);
    const groups = ["clean", "randomized", "all"].map(setting => {
      const selected = results.filter(item => setting === "all" || item.instance.setting === setting);
      const valid = selected.filter(item => terminalReasons.includes(item.reason));
      const successes = valid.filter(item => item.success === true).length;
      return { setting, total: selected.length, completed: valid.length, errors: selected.length - valid.length, successes,
        successRate: valid.length ? successes / valid.length : null,
        fullPanelSuccessRate: valid.length === selected.length && selected.length ? successes / selected.length : null };
    });
    const skippedErrors = values["skip-errors"]
      ? results.filter(item => !terminalReasons.includes(item.reason)).map(item => ({ instance_id: item.instance.instance_id, reason: item.reason })) : [];
    await writeFile(join(output, "summary.json"), JSON.stringify({ requested: instances.length, processed: results.length, groups,
      queueExhausted: results.length === instances.length, skippedErrors }, null, 2));
    process.stdout.write(`${JSON.stringify({ instance: instance.instance_id, success: result.success, reason: result.reason })}\n`);
    if (!terminalReasons.includes(result.reason) && !values["skip-errors"]) throw new Error(typeof result.error === "string" ? result.error : JSON.stringify(result.error ?? result.reason));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => { process.stderr.write(`${error.stack}\n`); process.exitCode = 1; });
}
