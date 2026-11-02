import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { buildDecisionPrompt, parseDecision } from "../framework/policy.mjs";
import { InferenceError, redact } from "../provider.mjs";

export const CODEX_DECISION_MS = 240000;
const KEY_ENV = "REA_CODEX_PROVIDER_API_KEY";
const FORMAT = 'Return one JSON object: {"name":"run_python","arguments":{"code":"Python source"},"thought":"brief explanation","notes":"complete replacement notes","claim":{"status":"working or complete","evidence":"brief observed evidence"}}. Generate the Python program without executing it. Use only the supplied observations.';

function decisionSchema(action) {
  return {
    type: "object",
    properties: {
      name: { type: "string", enum: [action.name] }, arguments: action.parameters,
      thought: { type: "string" }, notes: { type: "string" },
      claim: { type: "object", properties: {
        status: { type: "string", enum: ["working", "complete"] }, evidence: { type: "string" },
      }, required: ["status", "evidence"], additionalProperties: false },
    },
    required: ["name", "arguments", "thought", "notes", "claim"], additionalProperties: false,
  };
}

export function createRobotwinCodexGenerator({
  codexPath = "codex", model, provider = "rea",
  providerBaseUrl, apiKeyFile, workRoot = join(tmpdir(), "rea-robotwin-codex"), systemPrompt,
  apiRetries = process.env.REA_CODE_API_RETRIES !== "false",
} = {}) {
  if (!model || !providerBaseUrl || !apiKeyFile || !systemPrompt) {
    throw new TypeError("Codex generator requires model, providerBaseUrl, apiKeyFile and systemPrompt");
  }
  if (!/^[a-zA-Z0-9_-]+$/u.test(provider)) throw new TypeError("invalid Codex provider name");
  const active = new Set();
  let version = null;
  let apiKey = null;

  function run(args, { cwd, env, prompt = "", signal }) {
    signal?.throwIfAborted();
    return new Promise((resolveRun, reject) => {
      const child = spawn(codexPath, args, { cwd, env, detached: process.platform !== "win32", stdio: ["pipe", "pipe", "pipe"] });
      let stdout = "", stderr = "", killTimer;
      const kill = how => {
        if (child.exitCode !== null || !child.pid) return;
        try {
          if (process.platform === "win32") child.kill(how);
          else process.kill(-child.pid, how);
        } catch (error) { if (error.code !== "ESRCH") throw error; }
      };
      const stop = () => {
        kill("SIGTERM");
        killTimer ??= setTimeout(() => kill("SIGKILL"), 500);
      };
      active.add(stop);
      signal?.addEventListener("abort", stop, { once: true });
      const finish = () => {
        clearTimeout(killTimer);
        signal?.removeEventListener("abort", stop);
        active.delete(stop);
      };
      child.stdout.on("data", chunk => { stdout += chunk; });
      child.stderr.on("data", chunk => { stderr += chunk; });
      child.stdin.on("error", error => { if (error.code !== "EPIPE") reject(error); });
      child.once("error", error => { finish(); reject(error); });
      child.once("close", (code, terminationSignal) => {
        finish();
        resolveRun({ code, terminationSignal, stdout, stderr });
      });
      child.stdin.end(prompt);
      if (signal?.aborted) stop();
    });
  }

  return {
    id: `codex:${model}:robotwin-v2`,
    async decide(input) {
      input.signal?.throwIfAborted();
      const deadline = AbortSignal.timeout(CODEX_DECISION_MS);
      const signal = input.signal ? AbortSignal.any([input.signal, deadline]) : deadline;
      const actions = input.actions ?? input.observation?.actions;
      if (actions?.length !== 1 || actions[0].name !== "run_python") throw new TypeError("Codex generator requires the run_python action");
      const images = input.observation?.images ?? [];
      if (images.length !== 4) throw new TypeError("Codex RoboTwin decisions require all four camera images");
      await mkdir(workRoot, { recursive: true, mode: 0o700 });
      const workDir = await mkdtemp(join(resolve(workRoot), "decision-"));
      apiKey ??= (await readFile(apiKeyFile, "utf8")).trim();
      if (!apiKey) throw new TypeError("empty Codex provider key file");
      const env = Object.fromEntries(["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "TZ"].filter(key => process.env[key]).map(key => [key, process.env[key]]));
      env[KEY_ENV] = apiKey;
      const started = performance.now();
      const codex = { protocol: "robotwin-code-codex-v2", version, provider, baseUrl: providerBaseUrl,
        decisionMs: CODEX_DECISION_MS, outputTokenLimit: null, reasoning: "cli_default", temperature: "cli_default", workDir,
        apiRetries, attempts: [], capacityRetries: 0 };
      let usage = null;
      const failedUsage = [];
      const saveMetadata = () => writeFile(join(workDir, "metadata.json"),
        JSON.stringify({ model, codex, usage }, null, 2), { mode: 0o600 });
      try {
        if (!version) {
          const result = await run(["--version"], { cwd: workDir, env, signal });
          signal.throwIfAborted();
          if (result.code !== 0) throw new Error(`Codex version check failed: ${result.stderr}`);
          version = result.stdout.trim();
        }
        codex.version = version;
        const imagePaths = await Promise.all(images.map(async (image, index) => {
          const path = join(workDir, `camera-${index + 1}.${image.mimeType === "image/jpeg" ? "jpg" : "png"}`);
          await writeFile(path, Buffer.from(image.data, "base64"), { mode: 0o600 });
          return path;
        }));
        const schemaPath = join(workDir, "decision.schema.json"), instructionsPath = join(workDir, "instructions.txt");
        const finalPath = join(workDir, "decision.json");
        const prompt = buildDecisionPrompt({ ...input, actions, deadlineMs: CODEX_DECISION_MS, format: FORMAT })
          + "\n\n" + images.map((image, index) => `Attached image ${index + 1}: Camera: ${image.name ?? index + 1}`).join("\n");
        await Promise.all([
          writeFile(schemaPath, JSON.stringify(decisionSchema(actions[0])), { mode: 0o600 }),
          writeFile(instructionsPath, systemPrompt, { mode: 0o600 }),
          writeFile(join(workDir, "prompt.txt"), prompt, { mode: 0o600 }),
        ]);
        const config = {
          model_provider: provider,
          [`model_providers.${provider}.name`]: provider,
          [`model_providers.${provider}.base_url`]: providerBaseUrl,
          [`model_providers.${provider}.wire_api`]: "responses",
          [`model_providers.${provider}.env_key`]: KEY_ENV,
          ...(!apiRetries ? {
            [`model_providers.${provider}.request_max_retries`]: 0,
            [`model_providers.${provider}.stream_max_retries`]: 0,
          } : {}),
          model_instructions_file: instructionsPath,
          approval_policy: "never", web_search: "disabled", project_doc_max_bytes: 0,
          "features.shell_tool": false, "features.unified_exec": false, "features.view_image": false,
          "features.multi_agent": false, "features.apps": false,
          "features.plugins": false, "features.remote_plugin": false, "features.tool_suggest": false,
          "features.skill_search": false, "features.skip_host_skill_discovery": true,
          "features.browser_use": false, "features.computer_use": false, "features.image_generation": false,
          "features.goals": false, "features.hooks": false,
          "features.workspace_dependencies": false, "features.shell_snapshot": false,
        };
        const args = ["exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
          "--json", "--strict-config", "--model", model, "--output-schema", schemaPath, "--output-last-message", finalPath,
          ...imagePaths.flatMap(path => ["--image", path]),
          ...Object.entries(config).flatMap(([key, value]) => ["-c", `${key}=${JSON.stringify(value)}`]), "-"];
        for (let attempt = 1; ; attempt += 1) {
          const attemptStarted = performance.now();
          const result = await run(args, { cwd: workDir, env, prompt, signal });
          const stdout = redact(result.stdout, apiKey), stderr = redact(result.stderr, apiKey);
          const suffix = String(attempt).padStart(2, "0");
          await Promise.all([
            writeFile(join(workDir, `events.attempt-${suffix}.jsonl`), stdout, { mode: 0o600 }),
            writeFile(join(workDir, `stderr.attempt-${suffix}.log`), stderr, { mode: 0o600 }),
            writeFile(join(workDir, "events.jsonl"), stdout, { mode: 0o600 }),
            writeFile(join(workDir, "stderr.log"), stderr, { mode: 0o600 }),
          ]);
          // An interrupted process may end mid-line; its full bytes remain in the attempt log.
          const events = stdout.split("\n").slice(0, stdout.endsWith("\n") ? -1 : undefined)
            .filter(Boolean).flatMap(line => { try { return [JSON.parse(line)]; } catch { if (signal.aborted) return []; throw new Error("Invalid Codex event JSON"); } });
          const rawUsage = events.findLast(event => event.usage)?.usage;
          if (result.code !== 0 && rawUsage) failedUsage.push({ attempt, usage: rawUsage });
          if (rawUsage || failedUsage.length) {
            const counted = [...failedUsage.map(item => item.usage), ...(result.code === 0 && rawUsage ? [rawUsage] : [])];
            const sum = key => counted.reduce((total, item) => total + (item[key] ?? 0), 0);
            usage = { prompt_tokens: sum("input_tokens"), completion_tokens: sum("output_tokens"),
              total_tokens: sum("input_tokens") + sum("output_tokens"),
              prompt_tokens_details: { cached_tokens: sum("cached_input_tokens") },
              ...(counted.every(item => item.reasoning_output_tokens !== undefined)
                ? { completion_tokens_details: { reasoning_tokens: sum("reasoning_output_tokens") } } : {}),
              ...(result.code === 0 && rawUsage ? { raw_usage: rawUsage } : {}),
              ...(failedUsage.length ? { failed_attempts_usage: failedUsage } : {}) };
          }
          const record = { attempt, exitCode: result.code, terminationSignal: result.terminationSignal,
            elapsedMs: performance.now() - attemptStarted, usage: rawUsage ?? null };
          codex.attempts.push(record);
          signal.throwIfAborted();
          if (result.code === 0) break;
          const failure = events.findLast(event => event.type === "turn.failed" || event.type === "error");
          const message = failure?.error?.message ?? failure?.message ?? stderr;
          const capacity = /rate limit exceeded: Temporary capacity limit\. Please try again in (\d+(?:\.\d+)?)s\./iu.exec(message);
          if (!capacity || !apiRetries) {
            throw new InferenceError(`Codex exited ${result.code}: ${JSON.stringify(failure ?? stderr).slice(0, 1500)}`, { errorType: "codex_error" });
          }
          record.errorType = "provider_capacity";
          record.retryAfterMs = Number(capacity[1]) * 1000;
          record.retryWaitMs = record.retryAfterMs + Math.floor(Math.random() * 251);
          codex.capacityRetries += 1;
          codex.elapsedMs = performance.now() - started;
          await saveMetadata();
          // Retry the identical decision, without spending a robot action or resetting its deadline.
          await delay(record.retryWaitMs, undefined, { signal });
        }
        const decision = parseDecision(await readFile(finalPath, "utf8"), actions);
        if (typeof decision.action.arguments.code !== "string" || !decision.action.arguments.code.trim()) {
          throw new InferenceError("Codex returned no Python program", { errorType: "invalid_action" });
        }
        codex.elapsedMs = performance.now() - started;
        await saveMetadata();
        return { ...decision, model, usage, codex };
      } catch (error) {
        codex.elapsedMs = performance.now() - started;
        codex.error = input.signal?.aborted || deadline.aborted ? "deadline_or_abort" : redact(error, apiKey);
        await saveMetadata();
        if (input.signal?.aborted) throw input.signal.reason;
        const classified = deadline.aborted
          ? new InferenceError(`decision exceeded ${CODEX_DECISION_MS} ms`, { errorType: "deadline", retryable: false })
          : error instanceof InferenceError ? error : new InferenceError(redact(error, apiKey), { errorType: "codex_error" });
        classified.code ??= classified.errorType;
        classified.usage = usage;
        classified.codex = codex;
        throw classified;
      }
    },
    async close() { for (const stop of active) stop(); },
  };
}

export default createRobotwinCodexGenerator;
