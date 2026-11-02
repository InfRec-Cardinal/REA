import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import test from "node:test";

import { createPrimeAgentInfer, systemPromptFor, withNodeOnPath } from "../src/infer-prime.mjs";
import { G1_PROFILE } from "../src/profiles/g1.mjs";
import { LIBERO_PROFILE } from "../src/profiles/libero.mjs";

const MODEL = "deepseek-v4-flash-vision-exp";
const FRAME = {
  imageB64: Buffer.from("camera-frame").toString("base64"),
  mimeType: "image/jpeg",
  depth: { center_m: 1.2 },
};

function emitRecord(child, record, { split = false, crlf = false } = {}) {
  const line = `${JSON.stringify(record)}${crlf ? "\r\n" : "\n"}`;
  if (!split) {
    child.stdout.write(line);
    return;
  }
  const middle = Math.floor(line.length / 2);
  child.stdout.write(line.slice(0, middle));
  child.stdout.write(line.slice(middle));
}

function respond(child, command, data = undefined, options = undefined) {
  emitRecord(child, {
    id: command.id,
    type: "response",
    command: command.type,
    success: true,
    ...(data === undefined ? {} : { data }),
  }, options);
}

function fakeChild(onCommand) {
  const child = new EventEmitter();
  child.pid = 4100;
  child.stdin = new PassThrough();
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  child.commands = [];
  child.kills = [];
  child.kill = signal => {
    child.kills.push(signal);
    queueMicrotask(() => child.emit("exit", null, signal));
    return true;
  };

  let input = "";
  child.stdin.setEncoding("utf8");
  child.stdin.on("data", chunk => {
    input += chunk;
    while (true) {
      const newline = input.indexOf("\n");
      if (newline === -1) break;
      const line = input.slice(0, newline);
      input = input.slice(newline + 1);
      const command = JSON.parse(line);
      child.commands.push(command);
      onCommand(command, child);
    }
  });
  return child;
}

function harness({
  onSpawn,
  onCommand,
  onKill,
  now,
  daemonStatusFn = async () => [],
  terminationEofWaitMs = 0,
  terminationGraceMs = 20,
  terminationKillWaitMs = 20,
  daemonTerminationGraceMs = 20,
} = {}) {
  const spawns = [];
  const groupSignals = [];
  const spawnFn = (bin, argv, options) => {
    const child = fakeChild((command, current) => onCommand(command, current, spawns.length));
    child.pid += spawns.length;
    const call = { bin, argv, options, child };
    spawns.push(call);
    onSpawn?.(call, spawns.length);
    return child;
  };
  const killFn = (pid, signal) => {
    groupSignals.push({ pid, signal });
    if (onKill) return onKill(pid, signal, spawns);
    const error = new Error("fake process group is absent");
    error.code = "ESRCH";
    throw error;
  };
  const infer = createPrimeAgentInfer({
    bin: "/opt/prime-agent",
    configDir: "/opt/prime-config",
    provider: "openai-compatible",
    model: MODEL,
    cwd: "/opt/g1",
    spawnFn,
    killFn,
    daemonStatusFn,
    now,
    startupTimeoutMs: 500,
    terminationEofWaitMs,
    terminationGraceMs,
    terminationKillWaitMs,
    daemonTerminationGraceMs,
  });
  return { infer, spawns, groupSignals };
}

function standardHandshake(command, child, { split = false } = {}) {
  if (command.type === "get_state") {
    respond(child, command, {
      model: { id: MODEL, provider: "openai-compatible", maxTokens: 1, input: ["text", "image"] },
      thinkingLevel: "off",
      isStreaming: false,
      isCompacting: false,
      sessionId: "initial",
    }, { split, crlf: split });
    return true;
  }
  if (["set_auto_compaction", "set_auto_retry"].includes(command.type)) {
    respond(child, command);
    return true;
  }
  return false;
}

test("starts one persistent isolated RPC process and sends a fresh base64-image session", async () => {
  let clock = 0;
  const { infer, spawns } = harness({
    now: () => (clock += 100),
    onCommand(command, child) {
      if (standardHandshake(command, child, { split: true })) return;
      if (command.type === "new_session") {
        respond(child, command, { cancelled: false });
        return;
      }
      if (command.type === "prompt") {
        respond(child, command);
        emitRecord(child, {
          type: "message_update",
          message: { role: "assistant", content: [{ type: "text", text: "W" }] },
          assistantMessageEvent: { type: "text_delta", delta: "W" },
        });
        emitRecord(child, {
          type: "message_end",
          message: {
            role: "assistant",
            content: [{ type: "text", text: "W" }],
            stopReason: "length",
          },
        });
        emitRecord(child, {
          type: "agent_end",
          messages: [{ role: "assistant", content: [{ type: "text", text: "W" }], stopReason: "length" }],
        });
      }
    },
  });

  const started = await infer.start();
  assert.equal(started.status, "ready");
  const result = await infer(MODEL, FRAME, {
    signal: new AbortController().signal,
    deadlineMs: 500,
    task: "move toward the doorway",
  });

  assert.equal(result.token, "W");
  assert.equal(result.tokenReadyS, 0.1);
  assert.equal(result.totalS, 0.3);
  assert.equal(result.finishReason, "length");
  assert.equal(spawns.length, 1);
  assert.equal(spawns[0].bin, "/opt/prime-agent");
  assert.equal(spawns[0].options.cwd, "/opt/g1");
  assert.equal(spawns[0].options.detached, process.platform !== "win32");
  assert.equal(spawns[0].options.env.PRIME_AGENT_CODING_AGENT_DIR, "/opt/prime-config");
  for (const required of [
    "rpc",
    "--daemon-socket",
    "openai-compatible",
    MODEL,
    "--no-session",
    "--no-tools",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--no-context-files",
    "--offline",
    "/opt/g1",
    "--system-prompt",
  ]) {
    assert.ok(spawns[0].argv.includes(required), `argv is missing ${required}`);
  }
  const systemPrompt = spawns[0].argv[spawns[0].argv.indexOf("--system-prompt") + 1];
  assert.match(systemPrompt, /exactly one uppercase ASCII character/u);
  assert.match(systemPrompt, /an external controller is the only actuator/u);
  assert.equal(
    spawns[0].argv[spawns[0].argv.indexOf("--daemon-socket") + 1],
    "/opt/prime-config/prime-agent-daemon.sock",
  );

  const commands = spawns[0].child.commands;
  assert.deepEqual(commands.map(command => command.type), ["get_state", "set_auto_compaction", "set_auto_retry", "new_session", "prompt"]);
  assert.deepEqual(commands[4].images, [{
    type: "image",
    data: FRAME.imageB64,
    mimeType: "image/jpeg",
  }]);
  assert.match(commands[4].message, /move toward the doorway/u);
  assert.match(commands[4].message, /"center_m":1.2/u);

  await infer(MODEL, FRAME, { deadlineMs: 500 });
  assert.equal(spawns.length, 1);
  assert.equal(commands.filter(command => command.type === "new_session").length, 2);
  assert.equal(commands.filter(command => command.type === "set_auto_compaction").length, 1);
  assert.equal(commands.filter(command => command.type === "set_auto_retry").length, 1);
  await infer.close();
  assert.deepEqual(spawns[0].child.kills, ["SIGTERM"]);
  assert.equal(infer.state().status, "closed");
});

test("retires the detached RPC process group and escalates when a descendant survives SIGTERM", async () => {
  let groupAlive = true;
  const { infer, spawns, groupSignals } = harness({
    terminationGraceMs: 5,
    onKill(pid, signal) {
      assert.equal(pid, -4100);
      if (signal === 0) {
        if (groupAlive) return true;
        const error = new Error("group exited");
        error.code = "ESRCH";
        throw error;
      }
      if (signal === "SIGTERM") {
        queueMicrotask(() => spawns[0].child.emit("exit", null, "SIGTERM"));
        return true;
      }
      if (signal === "SIGKILL") {
        groupAlive = false;
        return true;
      }
      return true;
    },
    onCommand(command, child) {
      standardHandshake(command, child);
    },
  });

  await infer.start();
  await infer.close();
  assert.deepEqual(
    groupSignals.filter(entry => entry.signal !== 0),
    [
      { pid: -4100, signal: "SIGTERM" },
      { pid: -4100, signal: "SIGKILL" },
    ],
  );
  assert.equal(groupAlive, false);
});

test("never sends a process-group signal when a child handle reports the current process pid", async () => {
  const { infer, spawns, groupSignals } = harness({
    onSpawn({ child }) {
      child.pid = process.pid;
    },
    onCommand(command, child) {
      standardHandshake(command, child);
    },
  });

  await infer.start();
  await infer.close();
  assert.deepEqual(groupSignals, []);
  assert.deepEqual(spawns[0].child.kills, []);
});

test("standalone close terminates only the daemon on its private socket", async () => {
  let daemonAlive = true;
  const daemonPid = 9200;
  const { infer, groupSignals } = harness({
    daemonStatusFn: async () => [
      { socketPath: "/tmp/unrelated-prime.sock", pid: 9300 },
      { socketPath: "/opt/prime-config/prime-agent-daemon.sock", pid: daemonPid },
    ],
    onKill(pid, signal) {
      if (Math.abs(pid) !== daemonPid) {
        const error = new Error("fake RPC group is absent");
        error.code = "ESRCH";
        throw error;
      }
      if (signal === 0) {
        if (daemonAlive) return true;
        const error = new Error("daemon exited");
        error.code = "ESRCH";
        throw error;
      }
      if (signal === "SIGTERM") {
        daemonAlive = false;
        return true;
      }
      return true;
    },
    onCommand(command, child) {
      standardHandshake(command, child);
    },
  });

  await infer.start();
  await infer.close();
  assert.equal(daemonAlive, false);
  assert.ok(groupSignals.some(entry => entry.pid === -daemonPid && entry.signal === "SIGTERM"));
  assert.equal(groupSignals.some(entry => Math.abs(entry.pid) === 9300), false);
});

test("uses completed message_update and agent_end text, and switches models with thinking off", async () => {
  const outputs = ["A", "D"];
  const { infer, spawns } = harness({
    onCommand(command, child) {
      if (standardHandshake(command, child)) return;
      if (["new_session", "set_model", "set_thinking_level"].includes(command.type)) {
        const data = command.type === "new_session"
          ? { cancelled: false }
          : command.type === "set_model"
            ? { id: command.modelId, provider: command.provider, maxTokens: 1, input: ["text", "image"] }
            : undefined;
        respond(child, command, data);
        return;
      }
      if (command.type !== "prompt") return;
      respond(child, command);
      const token = outputs.shift();
      if (token === "A") {
        emitRecord(child, {
          type: "message_update",
          message: {
            role: "assistant",
            content: [{ type: "text", text: token }],
            stopReason: "stop",
          },
          assistantMessageEvent: { type: "done", reason: "stop" },
        });
        emitRecord(child, {
          type: "agent_end",
          messages: [{ role: "assistant", content: [{ type: "text", text: token }], stopReason: "stop" }],
        });
      } else {
        emitRecord(child, {
          type: "agent_end",
          messages: [{
            role: "assistant",
            content: [{ type: "text", text: token }],
            stopReason: "length",
          }],
        });
      }
    },
  });

  assert.equal((await infer(MODEL, FRAME, { deadlineMs: 500 })).token, "A");
  assert.equal((await infer("gemini-3.5-flash-lite", FRAME, { deadlineMs: 500 })).token, "D");
  outputs.push("Q");
  assert.equal((await infer("gemini-3.5-flash-lite", FRAME, { deadlineMs: 500 })).token, "Q");
  const commands = spawns[0].child.commands;
  const switches = commands.filter(command => command.type === "set_model");
  const switched = switches[0];
  assert.deepEqual(switched, {
    id: switched.id,
    type: "set_model",
    provider: "openai-compatible",
    modelId: "gemini-3.5-flash-lite",
  });
  assert.equal(switches.length, 2, "new_session resets Prime to the CLI primary before each fallback request");
  assert.ok(commands.some(command => command.type === "set_thinking_level" && command.level === "off"));
  await infer.close();
});

test("keeps the RPC slot occupied until agent_end, not merely message_end", async () => {
  let promptChild = null;
  const { infer } = harness({
    onCommand(command, child) {
      if (standardHandshake(command, child)) return;
      if (command.type === "new_session") {
        respond(child, command, { cancelled: false });
        return;
      }
      if (command.type !== "prompt") return;
      respond(child, command);
      promptChild = child;
      emitRecord(child, {
        type: "message_update",
        assistantMessageEvent: { type: "text_delta", delta: "E" },
      });
      emitRecord(child, {
        type: "message_end",
        message: { role: "assistant", content: [{ type: "text", text: "E" }], stopReason: "stop" },
      });
    },
  });

  let settled = false;
  const pending = infer(MODEL, FRAME, { deadlineMs: 500 }).finally(() => {
    settled = true;
  });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(settled, false);
  emitRecord(promptChild, {
    type: "agent_end",
    messages: [{ role: "assistant", content: [{ type: "text", text: "E" }], stopReason: "stop" }],
  });
  assert.equal((await pending).token, "E");
  await infer.close();
});

test("rejects a valid prefix followed by prose and rejects reasoning", async () => {
  let mode = "prose";
  const { infer } = harness({
    onCommand(command, child) {
      if (standardHandshake(command, child)) return;
      if (command.type === "new_session") {
        respond(child, command, { cancelled: false });
        return;
      }
      if (command.type !== "prompt") return;
      respond(child, command);
      if (mode === "prose") {
        emitRecord(child, {
          type: "message_update",
          assistantMessageEvent: { type: "text_delta", delta: "W" },
        });
        emitRecord(child, {
          type: "message_end",
          message: {
            role: "assistant",
            content: [{ type: "text", text: "Walk forward" }],
            stopReason: "stop",
          },
        });
        emitRecord(child, {
          type: "agent_end",
          messages: [{ role: "assistant", content: [{ type: "text", text: "Walk forward" }], stopReason: "stop" }],
        });
      } else {
        emitRecord(child, {
          type: "message_update",
          assistantMessageEvent: { type: "thinking_delta", delta: "I should inspect" },
        });
      }
    },
  });

  await assert.rejects(
    infer(MODEL, FRAME, { deadlineMs: 500 }),
    error => error.errorType === "invalid_direct_token" && error.retryable === true,
  );
  mode = "thinking";
  await assert.rejects(
    infer(MODEL, FRAME, { deadlineMs: 500 }),
    error => error.errorType === "reasoning_protocol_violation" && error.retryable === false,
  );
  await infer.close();
});

test("sends abort at the deadline, retires the child, and restarts on the next inference", async () => {
  const { infer, spawns } = harness({
    onCommand(command, child, spawnNumber) {
      if (standardHandshake(command, child)) return;
      if (command.type === "new_session") {
        respond(child, command, { cancelled: false });
        return;
      }
      if (command.type !== "prompt") return;
      respond(child, command);
      if (spawnNumber === 2) {
        emitRecord(child, {
          type: "agent_end",
          messages: [{
            role: "assistant",
            content: [{ type: "text", text: "X" }],
            stopReason: "stop",
          }],
        });
      }
    },
  });

  await assert.rejects(
    infer(MODEL, FRAME, { deadlineMs: 15 }),
    error => error.errorType === "deadline" && error.finishReason === "aborted",
  );
  assert.ok(spawns[0].child.commands.some(command => command.type === "abort"));
  assert.deepEqual(spawns[0].child.kills, ["SIGTERM"]);
  assert.equal(infer.state().status, "stopped");

  const result = await infer(MODEL, FRAME, { deadlineMs: 500 });
  assert.equal(result.token, "X");
  assert.equal(spawns.length, 2);
  assert.equal(infer.state().generation, 2);
  await infer.close();
});

test("propagates caller abort and unexpected child exit, then remains restartable", async () => {
  let exitOnPrompt = false;
  const { infer, spawns } = harness({
    onCommand(command, child, spawnNumber) {
      if (standardHandshake(command, child)) return;
      if (command.type === "new_session") {
        respond(child, command, { cancelled: false });
        return;
      }
      if (command.type !== "prompt") return;
      respond(child, command);
      if (exitOnPrompt) {
        queueMicrotask(() => child.emit("exit", 7, null));
      } else if (spawnNumber >= 2) {
        emitRecord(child, {
          type: "message_end",
          message: {
            role: "assistant",
            content: [{ type: "text", text: "Q" }],
            stopReason: "stop",
          },
        });
        emitRecord(child, {
          type: "agent_end",
          messages: [{ role: "assistant", content: [{ type: "text", text: "Q" }], stopReason: "stop" }],
        });
      }
    },
  });

  const controller = new AbortController();
  const aborted = infer(MODEL, FRAME, { signal: controller.signal, deadlineMs: 500 });
  queueMicrotask(() => controller.abort());
  await assert.rejects(aborted, error => error.errorType === "aborted");
  assert.ok(spawns[0].child.commands.some(command => command.type === "abort"));

  assert.equal((await infer(MODEL, FRAME, { deadlineMs: 500 })).token, "Q");
  exitOnPrompt = true;
  await assert.rejects(
    infer(MODEL, FRAME, { deadlineMs: 500 }),
    error => error.errorType === "agent_exit" && /code=7/u.test(error.message),
  );
  exitOnPrompt = false;
  assert.equal((await infer(MODEL, FRAME, { deadlineMs: 500 })).token, "Q");
  assert.equal(spawns.length, 3);
  await infer.close();
});

test("keeps the existing missing-frame InferenceError contract", async () => {
  const { infer, spawns } = harness({ onCommand() {} });
  await assert.rejects(
    infer(MODEL, null, { deadlineMs: 500 }),
    error => {
      assert.equal(error.name, "InferenceError");
      assert.equal(error.errorType, "config_error");
      assert.equal(error.status, null);
      assert.equal(error.upstreamStatus, null);
      assert.equal(error.finishReason, null);
      assert.equal(error.retryable, false);
      return true;
    },
  );
  assert.equal(spawns.length, 0);
  await infer.close();
});

test("classifies provider authentication errors and retires the RPC child", async () => {
  const { infer, spawns } = harness({
    onCommand(command, child) {
      if (standardHandshake(command, child)) return;
      if (command.type === "new_session") {
        respond(child, command, { cancelled: false });
        return;
      }
      if (command.type === "prompt") {
        respond(child, command);
        emitRecord(child, {
          type: "message_update",
          assistantMessageEvent: {
            type: "error",
            error: {
              role: "assistant",
              content: [],
              stopReason: "error",
              errorMessage: "HTTP 401 invalid token",
            },
          },
        });
      }
    },
  });

  await assert.rejects(
    infer(MODEL, FRAME, { deadlineMs: 500 }),
    error => error.errorType === "auth_error" && error.status === 401 && error.retryable === false,
  );
  assert.deepEqual(spawns[0].child.kills, ["SIGTERM"]);
  assert.equal(infer.state().status, "stopped");
  await infer.close();
});

test("refuses Prime automatic retry events and retires the RPC child", async () => {
  const { infer, spawns } = harness({
    onCommand(command, child) {
      if (standardHandshake(command, child)) return;
      if (command.type === "new_session") {
        respond(child, command, { cancelled: false });
        return;
      }
      if (command.type === "prompt") {
        respond(child, command);
        emitRecord(child, { type: "auto_retry_start", attempt: 1 });
      }
    },
  });

  await assert.rejects(
    infer(MODEL, FRAME, { deadlineMs: 500 }),
    error => error.errorType === "retry_protocol_violation" && error.retryable === false,
  );
  assert.deepEqual(spawns[0].child.kills, ["SIGTERM"]);
  assert.equal(infer.state().status, "stopped");
  await infer.close();
});

test("requires an explicit successful new_session response", async () => {
  const { infer } = harness({
    onCommand(command, child) {
      if (standardHandshake(command, child)) return;
      if (command.type === "new_session") respond(child, command, {});
    },
  });

  await assert.rejects(
    infer(MODEL, FRAME, { deadlineMs: 500 }),
    error => error.errorType === "session_error" && error.retryable === false,
  );
  await infer.close();
});

// The Prime CLI is a `#!/usr/bin/env node` script. When `node` is absent from the spawned
// process's PATH it dies before reading stdin, and the symptom surfaces as `write EPIPE` from a
// completely unrelated line. These lock the guarantee that the spawn supplies its own runtime.
test("withNodeOnPath prepends the running Node's directory when it is absent", () => {
  const result = withNodeOnPath({ PATH: "/usr/bin:/bin" }, "/opt/node/bin/node");
  assert.equal(result.PATH, `/opt/node/bin:/usr/bin:/bin`);
});

test("withNodeOnPath is a no-op when the directory is already present", () => {
  const env = { PATH: "/opt/node/bin:/usr/bin" };
  assert.equal(withNodeOnPath(env, "/opt/node/bin/node"), env, "must not clone when nothing changes");
});

test("withNodeOnPath copes with an empty or missing PATH", () => {
  assert.equal(withNodeOnPath({}, "/opt/node/bin/node").PATH, "/opt/node/bin");
  assert.equal(withNodeOnPath({ PATH: "" }, "/opt/node/bin/node").PATH, "/opt/node/bin");
});

test("withNodeOnPath preserves every other environment variable", () => {
  const result = withNodeOnPath({ PATH: "/usr/bin", REA_API_KEY: "k" }, "/opt/node/bin/node");
  assert.equal(result.REA_API_KEY, "k");
});

test("the system prompt names the profile's own tokens, byte-for-byte unchanged for the G1", () => {
  // The literal this replaced, pasted verbatim: the G1 prompt must not move by one character.
  const before = "You are a bounded one-step RGB-D robot action classifier. "
    + "You never execute motion or claim that motion was executed; an external controller is the only actuator. "
    + "Reasoning and tools are disabled. Return exactly one uppercase ASCII character from "
    + "X,W,S,A,D,Q,E,U,P,R, with no whitespace, punctuation, Markdown, prose, or thinking.";
  assert.equal(systemPromptFor(G1_PROFILE), before);
  const libero = systemPromptFor(LIBERO_PROFILE);
  assert.match(libero, new RegExp(`from ${LIBERO_PROFILE.TOKENS.join(",")}, with no whitespace`, "u"));
  assert.notEqual(libero, before, "a different profile gets a different token list");
});
