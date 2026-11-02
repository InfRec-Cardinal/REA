import { spawn as spawnChild } from "node:child_process";
import { delimiter as pathDelimiter, dirname, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { StringDecoder } from "node:string_decoder";

import { G1_PROFILE } from "./profiles/g1.mjs";
import { buildTokenPrompt } from "./infer-token.mjs";
import { InferenceError } from "./provider.mjs";

/** The harness system prompt for a profile: the token alphabet is the profile's, not the G1's. */
export function systemPromptFor(profile = G1_PROFILE) {
  return "You are a bounded one-step RGB-D robot action classifier. " +
    "You never execute motion or claim that motion was executed; an external controller is the only actuator. " +
    "Reasoning and tools are disabled. Return exactly one uppercase ASCII character from " +
    `${profile.TOKENS.join(",")}, with no whitespace, punctuation, Markdown, prose, or thinking.`;
}

function seconds(timestamp, started) {
  return Number(((timestamp - started) / 1000).toFixed(6));
}

function inferenceError(message, {
  errorType,
  status = null,
  finishReason = null,
  retryable = true,
} = {}) {
  return new InferenceError(message, {
    errorType,
    status,
    finishReason,
    retryable,
  });
}

function strictToken(text, profile = G1_PROFILE) {
  if (typeof text !== "string" || text.length !== 1) return null;
  return profile.parseToken(text);
}

function messageText(message) {
  if (!message || message.role !== "assistant") return null;
  if (typeof message.content === "string") return message.content;
  if (!Array.isArray(message.content)) return null;
  return message.content
    .filter(block => block?.type === "text" && typeof block.text === "string")
    .map(block => block.text)
    .join("");
}

function lastAssistant(messages) {
  if (!Array.isArray(messages)) return null;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "assistant") return messages[index];
  }
  return null;
}

function statusFrom(value) {
  if (Number.isInteger(value?.status)) return value.status;
  if (Number.isInteger(value?.error?.status)) return value.error.status;
  const message =
    value?.error?.errorMessage ??
    value?.errorMessage ??
    value?.error?.message ??
    value?.message ??
    "";
  const match = String(message).match(/\b([45]\d\d)\b/u);
  return match ? Number(match[1]) : null;
}

function streamFailure(event) {
  const detail = event?.assistantMessageEvent ?? event;
  const status = statusFrom(detail);
  const finishReason = detail?.reason ?? "error";
  const message =
    detail?.error?.errorMessage ??
    detail?.errorMessage ??
    detail?.error?.message ??
    "Prime Agent inference stream failed";
  return inferenceError(message, {
    errorType:
      status === 401 || status === 403
        ? "auth_error"
        : status !== null
          ? `upstream_${status}`
          : "stream_error",
    status,
    finishReason,
    retryable: status === null || status === 429 || status >= 500,
  });
}

function modelId(provider, model) {
  const value = String(model ?? "").trim();
  const prefix = `${provider}/`;
  return value.startsWith(prefix) ? value.slice(prefix.length) : value;
}

function argvFor({ provider, model, cwd, daemonSocket, profile = G1_PROFILE }) {
  return [
    "--mode",
    "rpc",
    "--daemon-socket",
    daemonSocket,
    "--provider",
    provider,
    "--model",
    model,
    "--thinking",
    "off",
    "--no-session",
    "--no-tools",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--no-context-files",
    "--offline",
    "--cwd",
    cwd,
    "--system-prompt",
    systemPromptFor(profile),
  ];
}

/**
 * The Prime Agent CLI is a `#!/usr/bin/env node` script, so it only starts if `node` is on PATH of
 * the spawned process. Relying on the caller for that is fragile: the systemd unit happens to
 * prepend the bundled runtime, but a test, a plain shell, or any other launcher does not, and the
 * failure surfaces far from its cause as `write EPIPE` when the dead child's stdin is written to.
 * Prepending the directory of the Node that is running us makes the spawn self-sufficient, and
 * guarantees the child runs the same runtime as the parent.
 */
export function withNodeOnPath(env = process.env, execPath = process.execPath) {
  const nodeDir = dirname(execPath);
  const current = env.PATH ?? env.Path ?? "";
  const entries = current.split(pathDelimiter).filter(Boolean);
  if (entries.includes(nodeDir)) return env;
  return { ...env, PATH: [nodeDir, ...entries].join(pathDelimiter) };
}

export function createPrimeAgentInfer({
  bin,
  configDir,
  provider,
  model,
  cwd = process.cwd(),
  spawnFn = spawnChild,
  controlSpawnFn = spawnChild,
  killFn = process.kill,
  daemonStatusFn = null,
  daemonSocket = null,
  now = () => performance.now(),
  env = process.env,
  // A cold private daemon takes about 6 s to initialise on the G1's PC2 computer. Once resident, this timeout
  // is off the decision path; keep enough headroom for first boot without weakening request limits.
  startupTimeoutMs = 15_000,
  defaultDeadlineMs = 5000,
  terminationEofWaitMs = 250,
  terminationGraceMs = 250,
  terminationKillWaitMs = 750,
  daemonTerminationGraceMs = 1500,
  profile = G1_PROFILE,
} = {}) {
  const daemonSocketPath = resolve(daemonSocket ?? configDir ?? cwd, daemonSocket ? "" : "prime-agent-daemon.sock");
  let child = null;
  let ready = false;
  let closed = false;
  let generation = 0;
  let requestSequence = 0;
  let startPromise = null;
  let queue = Promise.resolve();
  let active = null;
  let activeModel = null;
  let remoteState = null;
  let lastError = null;
  let retirement = Promise.resolve();
  let retirementPending = false;
  const pending = new Map();
  const exitPromises = new WeakMap();
  const exited = new WeakSet();

  function state() {
    return {
      status: closed
        ? "closed"
        : ready && child
          ? "ready"
          : startPromise
            ? "starting"
            : "stopped",
      running: child !== null,
      ready: ready && child !== null,
      busy: active !== null,
      pid: child?.pid ?? null,
      generation,
      model: activeModel,
      pendingRequests: pending.size,
      lastError,
    };
  }

  function rejectPending(error) {
    for (const waiter of pending.values()) waiter.reject(error);
    pending.clear();
  }

  function rejectActive(error) {
    if (!active) return;
    const current = active;
    active = null;
    current.reject(error);
  }

  function safeProcessGroupPid(current) {
    const pid = current?.pid;
    if (
      process.platform === "win32" ||
      !Number.isInteger(pid) ||
      pid <= 1 ||
      pid === process.pid
    ) {
      return null;
    }
    return pid;
  }

  function groupAlive(current) {
    const pid = safeProcessGroupPid(current);
    if (pid === null) return null;
    try {
      killFn(-pid, 0);
      return true;
    } catch (error) {
      if (error?.code === "ESRCH") return false;
      return true;
    }
  }

  function signalProcessGroup(current, signal) {
    const pid = safeProcessGroupPid(current);
    if (pid !== null) {
      try {
        killFn(-pid, signal);
        return;
      } catch (error) {
        if (error?.code !== "ESRCH") {
          // Fall through to the direct child handle when group signalling is unavailable.
        }
      }
    }
    if (Number.isInteger(current?.pid) && (current.pid <= 1 || current.pid === process.pid)) return;
    if (exited.has(current)) return;
    try {
      current.kill(signal);
    } catch {
      // The direct child has already exited.
    }
  }

  async function waitForProcessGroupExit(current, timeoutMs) {
    const deadline = Date.now() + Math.max(0, timeoutMs);
    while (true) {
      const groupRunning = groupAlive(current);
      if (exited.has(current) && groupRunning !== true) return true;
      const remaining = deadline - Date.now();
      if (remaining <= 0) return false;
      const exitPromise = exitPromises.get(current) ?? Promise.resolve();
      await Promise.race([
        exitPromise,
        new Promise(resolve => setTimeout(resolve, Math.min(20, remaining))),
      ]);
    }
  }

  async function terminateProcessGroup(current) {
    try {
      current.stdin?.end?.();
    } catch {
      // The RPC pipe has already closed.
    }
    if (await waitForProcessGroupExit(current, terminationEofWaitMs)) return;
    signalProcessGroup(current, "SIGTERM");
    if (await waitForProcessGroupExit(current, terminationGraceMs)) return;
    signalProcessGroup(current, "SIGKILL");
    await waitForProcessGroupExit(current, terminationKillWaitMs);
  }

  async function listPrimeDaemons() {
    if (daemonStatusFn) return daemonStatusFn();
    return new Promise((resolveStatus, rejectStatus) => {
      let statusChild;
      try {
        statusChild = controlSpawnFn(bin, ["status", "--json"], {
          cwd,
          env: { ...withNodeOnPath(env), PRIME_AGENT_CODING_AGENT_DIR: configDir },
          stdio: ["ignore", "pipe", "pipe"],
          windowsHide: true,
        });
      } catch (cause) {
        rejectStatus(cause);
        return;
      }
      let stdout = "";
      statusChild.stdout?.setEncoding?.("utf8");
      statusChild.stdout?.on?.("data", chunk => {
        stdout += chunk;
      });
      statusChild.stderr?.resume?.();
      const timer = setTimeout(() => {
        try {
          statusChild.kill("SIGKILL");
        } catch {
          // The status command already exited.
        }
        rejectStatus(new Error("Prime Agent status timed out"));
      }, startupTimeoutMs);
      statusChild.once("error", cause => {
        clearTimeout(timer);
        rejectStatus(cause);
      });
      statusChild.once("close", code => {
        clearTimeout(timer);
        if (code !== 0) {
          rejectStatus(new Error(`Prime Agent status exited with code ${code ?? "null"}`));
          return;
        }
        try {
          const records = JSON.parse(stdout);
          resolveStatus(Array.isArray(records) ? records : []);
        } catch (cause) {
          rejectStatus(cause);
        }
      });
    });
  }

  function daemonProcessAlive(pid) {
    if (!Number.isInteger(pid) || pid <= 1 || pid === process.pid) return false;
    for (const target of process.platform === "win32" ? [pid] : [-pid, pid]) {
      try {
        killFn(target, 0);
        return true;
      } catch (error) {
        if (error?.code !== "ESRCH") return true;
      }
    }
    return false;
  }

  function signalDaemon(pid, signal) {
    if (!Number.isInteger(pid) || pid <= 1 || pid === process.pid) return;
    for (const target of process.platform === "win32" ? [pid] : [-pid, pid]) {
      try {
        killFn(target, signal);
        return;
      } catch (error) {
        if (error?.code !== "ESRCH") return;
      }
    }
  }

  async function waitForDaemonExit(pid, timeoutMs) {
    const deadline = Date.now() + Math.max(0, timeoutMs);
    while (daemonProcessAlive(pid)) {
      const remaining = deadline - Date.now();
      if (remaining <= 0) return false;
      await new Promise(resolveWait => setTimeout(resolveWait, Math.min(20, remaining)));
    }
    return true;
  }

  async function shutdownPrivateDaemon() {
    const daemons = await listPrimeDaemons();
    const daemon = daemons.find(record => record?.socketPath === daemonSocketPath);
    const pid = daemon?.pid;
    if (!Number.isInteger(pid) || pid <= 1 || pid === process.pid) return;
    signalDaemon(pid, "SIGTERM");
    if (await waitForDaemonExit(pid, daemonTerminationGraceMs)) return;
    signalDaemon(pid, "SIGKILL");
    await waitForDaemonExit(pid, terminationKillWaitMs);
  }

  function detachCurrent(error, { kill = true } = {}) {
    const current = child;
    child = null;
    ready = false;
    activeModel = null;
    remoteState = null;
    lastError = error.message;
    rejectPending(error);
    rejectActive(error);
    if (!kill || !current) return retirement;
    const currentRetirement = terminateProcessGroup(current).catch(() => {});
    retirement = currentRetirement;
    retirementPending = true;
    void currentRetirement.finally(() => {
      if (retirement === currentRetirement) retirementPending = false;
    });
    return retirement;
  }

  function processFailure(current, error) {
    if (current !== child) return;
    detachCurrent(error);
  }

  function send(command) {
    if (!child?.stdin || child.stdin.destroyed || child.stdin.writable === false) {
      throw inferenceError("Prime Agent RPC stdin is unavailable", {
        errorType: "agent_exit",
      });
    }
    child.stdin.write(`${JSON.stringify(command)}\n`);
  }

  function request(command) {
    const id = `g1-prime-${generation}-${++requestSequence}`;
    return new Promise((resolve, reject) => {
      pending.set(id, { command: command.type, resolve, reject });
      try {
        send({ id, ...command });
      } catch (error) {
        pending.delete(id);
        reject(error);
        if (child) detachCurrent(error);
      }
    });
  }

  function sendAbort() {
    if (!child) return;
    try {
      send({ id: `g1-prime-${generation}-${++requestSequence}`, type: "abort" });
    } catch {
      // Recycling below is the fallback when stdin is already gone.
    }
  }

  function completeActive(text, finishReason, observedAt) {
    if (!active) return;
    const current = active;
    const token = strictToken(text, profile);
    active = null;
    if (token === null) {
      const observed = JSON.stringify(String(text ?? "").slice(0, 80));
      const error = inferenceError(`Prime Agent ended without exactly one valid direct token (observed ${observed})`, {
        errorType: "invalid_direct_token",
        finishReason,
      });
      current.reject(error);
      detachCurrent(error);
      return;
    }
    const tokenReadyAt = current.tokenReadyAt ?? observedAt;
    current.resolve({
      token,
      tokenReadyS: seconds(tokenReadyAt, current.started),
      totalS: seconds(observedAt, current.started),
      finishReason,
      responseStatus: null,
    });
  }

  function handleEvent(event) {
    if (!active) return;
    const observedAt = now();

    if (event.type === "auto_retry_start") {
      const error = inferenceError("Prime Agent attempted an automatic retry in bounded mode", {
        errorType: "retry_protocol_violation",
        retryable: false,
      });
      rejectActive(error);
      sendAbort();
      detachCurrent(error);
      return;
    }

    if (event.type === "message_update") {
      const update = event.assistantMessageEvent ?? {};
      if (update.type === "thinking_start" || update.type === "thinking_delta") {
        const error = inferenceError("Prime Agent emitted reasoning in direct mode", {
          errorType: "reasoning_protocol_violation",
          finishReason: "error",
          retryable: false,
        });
        rejectActive(error);
        sendAbort();
        detachCurrent(error);
        return;
      }
      if (update.type?.startsWith("toolcall_")) {
        const error = inferenceError("Prime Agent emitted a tool call in no-tools direct mode", {
          errorType: "tool_protocol_violation",
          finishReason: "toolUse",
          retryable: false,
        });
        rejectActive(error);
        sendAbort();
        detachCurrent(error);
        return;
      }
      if (update.type === "error") {
        const error = streamFailure(event);
        rejectActive(error);
        sendAbort();
        detachCurrent(error);
        return;
      }
      if (update.type === "text_delta" && typeof update.delta === "string") {
        if (active.tokenReadyAt === null && update.delta.length > 0) {
          active.tokenReadyAt = observedAt;
        }
        active.text += update.delta;
        return;
      }
      if (update.type === "done") {
        const finalMessage = event.message ?? update.partial;
        active.text = messageText(finalMessage) ?? active.text;
        active.finishReason = update.reason ?? finalMessage?.stopReason ?? active.finishReason;
      }
      return;
    }

    if (event.type === "message_end" && event.message?.role === "assistant") {
      active.text = messageText(event.message) ?? active.text;
      active.finishReason = event.message.stopReason ?? active.finishReason;
      return;
    }

    if (event.type === "agent_end") {
      const message = lastAssistant(event.messages);
      completeActive(
        messageText(message) ?? active.text,
        message?.stopReason ?? active.finishReason ?? null,
        observedAt,
      );
      return;
    }

    if (event.type === "extension_error") {
      const error = inferenceError(event.error ?? "Prime Agent extension failed", {
        errorType: "stream_error",
      });
      rejectActive(error);
      sendAbort();
      detachCurrent(error);
    }
  }

  function handleRecord(record) {
    if (record?.type === "response" && typeof record.id === "string") {
      const waiter = pending.get(record.id);
      if (!waiter) return;
      pending.delete(record.id);
      if (record.success === true) {
        waiter.resolve(record);
      } else {
        const error = inferenceError(
          `Prime Agent ${waiter.command} failed: ${record.error ?? "unknown RPC error"}`,
          {
            errorType: "rpc_error",
            retryable: waiter.command !== "set_model",
          },
        );
        waiter.reject(error);
        if (child) detachCurrent(error);
      }
      return;
    }
    handleEvent(record);
  }

  function attachJsonl(current, currentGeneration) {
    const decoder = new StringDecoder("utf8");
    let buffer = "";
    const onLine = rawLine => {
      const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
      if (line.length === 0) return;
      try {
        handleRecord(JSON.parse(line));
      } catch (cause) {
        const error = inferenceError(`Prime Agent emitted invalid JSONL: ${cause.message}`, {
          errorType: "agent_protocol_error",
        });
        if (current === child && currentGeneration === generation) detachCurrent(error);
      }
    };

    current.stdout.on("data", chunk => {
      if (current !== child || currentGeneration !== generation) return;
      buffer += typeof chunk === "string" ? chunk : decoder.write(chunk);
      while (true) {
        const newline = buffer.indexOf("\n");
        if (newline === -1) break;
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        onLine(line);
      }
    });
    current.stdout.on("end", () => {
      if (current !== child || currentGeneration !== generation) return;
      buffer += decoder.end();
      if (buffer.length > 0) onLine(buffer);
      if (!closed && current === child) {
        processFailure(current, inferenceError("Prime Agent RPC stdout ended", {
          errorType: "agent_exit",
        }));
      }
    });
  }

  async function start() {
    if (closed) {
      throw inferenceError("Prime Agent adapter is closed", {
        errorType: "config_error",
        retryable: false,
      });
    }
    if (ready && child) return state();
    if (startPromise) return startPromise;

    const starting = (async () => {
      if (![bin, configDir, provider, model, cwd].every(value => typeof value === "string" && value.length > 0)) {
        throw inferenceError("Prime Agent bin, configDir, provider, model, and cwd are required", {
          errorType: "config_error",
          retryable: false,
        });
      }

      generation += 1;
      const currentGeneration = generation;
      let current;
      try {
        if (retirementPending) await retirement;
        if (closed) {
          throw inferenceError("Prime Agent adapter is closed", {
            errorType: "config_error",
            retryable: false,
          });
        }
        current = spawnFn(bin, argvFor({ provider, model, cwd, daemonSocket: daemonSocketPath, profile }), {
          cwd,
          env: { ...withNodeOnPath(env), PRIME_AGENT_CODING_AGENT_DIR: configDir },
          stdio: ["pipe", "pipe", "pipe"],
          detached: process.platform !== "win32",
          windowsHide: true,
        });
      } catch (cause) {
        throw inferenceError(`Failed to start Prime Agent: ${cause.message}`, {
          errorType: "agent_start_error",
        });
      }
      child = current;
      ready = false;
      activeModel = null;
      remoteState = null;
      lastError = null;
      let resolveExit;
      exitPromises.set(current, new Promise(resolve => {
        resolveExit = resolve;
      }));
      attachJsonl(current, currentGeneration);
      current.stderr?.resume?.();

      current.once("error", cause => {
        if (!Number.isInteger(current.pid)) {
          exited.add(current);
          resolveExit();
        }
        processFailure(current, inferenceError(`Prime Agent process error: ${cause.message}`, {
          errorType: "agent_start_error",
        }));
      });
      current.stdin.once("error", cause => {
        processFailure(current, inferenceError(`Prime Agent RPC stdin failed: ${cause.message}`, {
          errorType: "agent_exit",
        }));
      });
      current.once("exit", (code, signal) => {
        exited.add(current);
        resolveExit();
        processFailure(current, inferenceError(
          `Prime Agent exited (code=${code ?? "null"}, signal=${signal ?? "null"})`,
          { errorType: "agent_exit" },
        ));
      });

      let timer;
      try {
        const initialise = async () => {
          const response = await request({ type: "get_state" });
          await request({ type: "set_auto_compaction", enabled: false });
          await request({ type: "set_auto_retry", enabled: false });
          return response;
        };
        const timeout = new Promise((_, reject) => {
          timer = setTimeout(() => reject(inferenceError(
            `Prime Agent get_state handshake exceeded ${startupTimeoutMs} ms`,
            { errorType: "agent_start_timeout" },
          )), startupTimeoutMs);
        });
        const response = await Promise.race([initialise(), timeout]);
        remoteState = response.data ?? null;
        const remoteModel = remoteState?.model;
        if (
          remoteModel?.provider !== provider ||
          remoteModel?.id !== model ||
          remoteModel?.maxTokens !== 1 ||
          !Array.isArray(remoteModel?.input) ||
          !remoteModel.input.includes("image") ||
          remoteState?.thinkingLevel !== "off" ||
          remoteState?.isStreaming !== false ||
          remoteState?.isCompacting === true
        ) {
          throw inferenceError("Prime Agent RPC state does not match the bounded one-token configuration", {
            errorType: "config_error",
            retryable: false,
          });
        }
        activeModel = remoteState?.model?.id ?? model;
        ready = true;
        return state();
      } catch (error) {
        if (current === child) await detachCurrent(error);
        else await retirement;
        throw error;
      } finally {
        clearTimeout(timer);
      }
    })();

    startPromise = starting;
    try {
      return await starting;
    } finally {
      if (startPromise === starting) startPromise = null;
    }
  }

  async function run(modelName, frame, options) {
    const requestedModel = modelId(provider, modelName);
    const started = now();
    await start();

    const fresh = await request({ type: "new_session" });
    if (fresh.data?.cancelled !== false) {
      const error = inferenceError("Prime Agent refused to create a fresh inference session", {
        errorType: "session_error",
        retryable: false,
      });
      detachCurrent(error);
      throw error;
    }
    // Prime rebuilds the runtime from its original CLI session config on every new_session, which
    // resets the remote model to the startup primary even if the previous request used fallback.
    activeModel = model;

    if (activeModel !== requestedModel) {
      const switched = await request({ type: "set_model", provider, modelId: requestedModel });
      const switchedModel = switched.data;
      if (
        switchedModel?.provider !== provider ||
        switchedModel?.id !== requestedModel ||
        switchedModel?.maxTokens !== 1 ||
        !Array.isArray(switchedModel?.input) ||
        !switchedModel.input.includes("image")
      ) {
        const error = inferenceError("Prime Agent switched to a model outside the bounded vision configuration", {
          errorType: "config_error",
          retryable: false,
        });
        detachCurrent(error);
        throw error;
      }
      await request({ type: "set_thinking_level", level: "off" });
      activeModel = requestedModel;
    }

    const resultPromise = new Promise((resolve, reject) => {
      active = {
        resolve,
        reject,
        started,
        text: "",
        tokenReadyAt: null,
        finishReason: null,
      };
    });
    resultPromise.catch(() => {});

    try {
      await request({
        type: "prompt",
        message: buildTokenPrompt({
          instruction: options.instruction ?? options.task,
          depth: frame.depth ?? null,
          advice: options.advice ?? null,
          variant: options.cfg?.promptVariant ?? "short",
          experience: options.experience ?? null,
          directive: options.directive ?? null,
          profile,
        }),
        images: [
          {
            type: "image",
            data: frame.imageB64,
            mimeType: frame.mimeType ?? "image/jpeg",
          },
        ],
      });
      return await resultPromise;
    } catch (error) {
      if (active?.reject) {
        const current = active;
        active = null;
        current.reject(error);
      }
      throw error;
    }
  }

  async function runWithDeadline(modelName, frame, options) {
    if (options.signal?.aborted) {
      throw inferenceError("Prime Agent inference was aborted", {
        errorType: "aborted",
        finishReason: "aborted",
      });
    }

    const deadlineMs = Number.isFinite(options.deadlineMs)
      ? Math.max(0, options.deadlineMs)
      : defaultDeadlineMs;
    let timer;
    let abortListener;
    const interrupted = new Promise((_, reject) => {
      const interrupt = error => {
        reject(error);
        sendAbort();
        if (child) detachCurrent(error);
      };
      timer = setTimeout(() => interrupt(inferenceError(
        `${modelName} Prime Agent direct deadline exceeded ${deadlineMs} ms`,
        {
          errorType: "deadline",
          finishReason: "aborted",
        },
      )), deadlineMs);
      if (options.signal) {
        abortListener = () => interrupt(inferenceError("Prime Agent inference was aborted", {
          errorType: "aborted",
          finishReason: "aborted",
        }));
        options.signal.addEventListener("abort", abortListener, { once: true });
      }
    });

    try {
      return await Promise.race([run(modelName, frame, options), interrupted]);
    } finally {
      clearTimeout(timer);
      if (abortListener) options.signal.removeEventListener("abort", abortListener);
    }
  }

  function infer(modelName, frame, options = {}) {
    if (!frame?.imageB64) {
      return Promise.reject(inferenceError("current direct frame is missing", {
        errorType: "config_error",
        retryable: false,
      }));
    }
    if (typeof modelName !== "string" || modelName.trim().length === 0) {
      return Promise.reject(inferenceError("direct inference model is missing", {
        errorType: "config_error",
        retryable: false,
      }));
    }

    const execute = async () => {
      try {
        return await runWithDeadline(modelName, frame, options);
      } catch (error) {
        await retirement;
        throw error;
      }
    };
    const result = queue.then(execute, execute);
    queue = result.catch(() => {});
    return result;
  }

  async function close({ shutdownDaemon = true } = {}) {
    if (closed) return;
    closed = true;
    const error = inferenceError("Prime Agent adapter closed", {
      errorType: "aborted",
      finishReason: "aborted",
      retryable: false,
    });
    sendAbort();
    if (child) await detachCurrent(error);
    else {
      rejectPending(error);
      rejectActive(error);
      await retirement;
    }
    if (shutdownDaemon) {
      try {
        await shutdownPrivateDaemon();
      } catch (cause) {
        lastError = `Prime Agent private daemon cleanup failed: ${cause.message}`;
      }
    }
  }

  infer.start = start;
  infer.close = close;
  infer.state = state;
  infer.engine = "prime";
  infer.harness = "prime-agent-rpc";
  infer.daemonSocket = daemonSocketPath;
  return infer;
}
