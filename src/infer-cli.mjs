// CLI-backed engines: models that are reachable only through their official command-line clients.
//
// Some deployments can reach Claude or Codex models only through the official Claude Code and Codex
// CLIs. The CLIs accept an image and answer a one-token prompt in 5 to 15 s, so they can serve
// the asynchronous roles (deliberate escalation when the robot is stopped, reviewer, router model), never
// the reflex path. Each call writes the frame to a temp file, spawns the CLI with a minimal system
// prompt, parses the answer, and cleans up. Nothing here reaches the actuator: the executor and the gate
// treat the answer like any other engine's token.

import { spawn } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { performance } from "node:perf_hooks";

import { G1_PROFILE } from "./profiles/g1.mjs";
import { InferenceError } from "./provider.mjs";
import { SYSTEM_PROMPTS, buildTokenPrompt } from "./infer-token.mjs";

export function cliKindFor(model) {
  if (/^claude/iu.test(String(model))) return "claude";
  if (/^(gpt|codex|o[0-9])/iu.test(String(model))) return "codex";
  return null;
}

/** Claude's --output-format json, or Codex's --json event lines, to { text, usage, costUsd }. */
export function parseCliResult(kind, stdout) {
  const raw = String(stdout ?? "");
  if (kind === "claude") {
    try {
      const data = JSON.parse(raw.trim());
      if (data && typeof data === "object" && !Array.isArray(data)) {
        return { text: String(data.result ?? ""), usage: data.usage ?? null, costUsd: Number.isFinite(data.total_cost_usd) ? data.total_cost_usd : null, error: data.is_error ? String(data.result ?? "error") : null };
      }
    } catch {
      /* fall through: a bare line */
    }
    return { text: raw.trim(), usage: null, costUsd: null, error: null };
  }
  let text = null;
  let usage = null;
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) continue;
    let event;
    try {
      event = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (event.type === "item.completed" && event.item?.type === "agent_message" && typeof event.item.text === "string") text = event.item.text;
    if (event.type === "turn.completed" && event.usage) usage = event.usage;
  }
  if (text === null) {
    const lines = raw.trim().split("\n").map(l => l.trim()).filter(Boolean);
    text = lines.length ? lines[lines.length - 1] : "";
  }
  return { text, usage, costUsd: null, error: null };
}

export function cliArgs(kind, { model, prompt, imagePath, system, bin, effort = "low" }) {
  if (kind === "claude") {
    return { command: bin, args: ["-p", prompt, "--model", model, "--output-format", "json", "--bare", "--no-session-persistence", "--system-prompt", system, "--tools", "Read"] };
  }
  return { command: bin, args: ["exec", "-i", imagePath, "-m", model, "-c", `model_reasoning_effort=${effort}`, "--skip-git-repo-check", "--json", `${system}\n\n${prompt}`] };
}

export function createCliInfer({ claudeBin = "claude", codexBin = "codex", tmpDir = tmpdir(), spawnImpl = spawn, now = () => performance.now(), effort = "low", profile = G1_PROFILE } = {}) {
  let counter = 0;

  async function infer(model, frame, { signal, instruction, advice = null, deadlineMs = 60000, cfg = {}, experience = null, directive = null } = {}) {
    const kind = cliKindFor(model);
    if (!kind) throw new InferenceError(`no CLI serves model ${model}`, { errorType: "config_error", retryable: false });
    if (!frame?.imageB64) throw new InferenceError("current frame is missing", { errorType: "config_error", retryable: false });
    const started = now();
    const dir = await mkdtemp(join(tmpDir, "rea-cli-"));
    const imagePath = join(dir, `frame-${++counter}.jpg`);
    await writeFile(imagePath, Buffer.from(frame.imageB64, "base64"));
    const text = buildTokenPrompt({ instruction, depth: cfg.includeDepth === false ? null : (frame.depth ?? null), advice, variant: "full", experience, directive, profile });
    const prompt = kind === "claude" ? `${text}\n\nThe forward camera frame is the image file at ${imagePath}; read it before answering.` : text;
    const { command, args } = cliArgs(kind, { model, prompt, imagePath, system: SYSTEM_PROMPTS.full, bin: kind === "claude" ? claudeBin : codexBin, effort });
    const child = spawnImpl(command, args, { stdio: ["ignore", "pipe", "pipe"], env: process.env });
    let stdout = "";
    let stderr = "";
    child.stdout?.on("data", chunk => { stdout += chunk; });
    child.stderr?.on("data", chunk => { stderr += chunk; });
    const exit = new Promise(resolve => child.on("close", code => resolve(code)));
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; child.kill("SIGKILL"); }, deadlineMs);
    const onAbort = () => child.kill("SIGKILL");
    signal?.addEventListener("abort", onAbort, { once: true });
    let code;
    try {
      code = await exit;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      rm(dir, { recursive: true, force: true }).catch(() => {});
    }
    const elapsedS = Number(((now() - started) / 1000).toFixed(6));
    if (timedOut) throw new InferenceError(`${kind} CLI exceeded ${deadlineMs} ms`, { errorType: "deadline", retryable: true });
    if (code !== 0) throw new InferenceError(`${kind} CLI exited ${code}: ${stderr.slice(0, 200) || stdout.slice(0, 200)}`, { errorType: "provider_error", retryable: true });
    const parsed = parseCliResult(kind, stdout);
    if (parsed.error) throw new InferenceError(`${kind} CLI error: ${parsed.error.slice(0, 200)}`, { errorType: "provider_error", retryable: true });
    const token = profile.parseToken(parsed.text.trim());
    if (token === null) {
      throw new InferenceError(`${kind} CLI answered without a valid token: ${JSON.stringify(parsed.text.slice(0, 40))}`, { errorType: "invalid_token", retryable: true });
    }
    return {
      token,
      tokenReadyS: elapsedS,
      actionReadyS: elapsedS,
      totalS: elapsedS,
      finishReason: "stop",
      responseStatus: 200,
      source: `cli:${kind}`,
      thought: parsed.text.trim().slice(0, 200),
      usage: parsed.usage,
      costUsd: parsed.costUsd,
      timings: { ttfb_s: elapsedS, token_ready_s: elapsedS, out_chars: parsed.text.length },
    };
  }

  return Object.assign(infer, { engine: "cli", warm: async () => ({ ok: true, skipped: true }) });
}
