import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { test } from "node:test";

import { cliArgs, cliKindFor, createCliInfer, parseCliResult } from "../src/infer-cli.mjs";

test("the CLI kind follows the model name and the arguments keep the calls minimal", () => {
  assert.equal(cliKindFor("claude-fable-5-1"), "claude");
  assert.equal(cliKindFor("gpt-6-astra"), "codex");
  assert.equal(cliKindFor("deepseek-v4-flash"), null);
  const claude = cliArgs("claude", { model: "claude-fable-5-1", prompt: "P", imagePath: "/tmp/f.jpg", system: "S", bin: "claude" });
  assert.ok(claude.args.includes("--bare") && claude.args.includes("--no-session-persistence") && claude.args.includes("--system-prompt"));
  assert.deepEqual(claude.args.slice(0, 4), ["-p", "P", "--model", "claude-fable-5-1"]);
  const codex = cliArgs("codex", { model: "gpt-6-astra", prompt: "P", imagePath: "/tmp/f.jpg", system: "S", bin: "codex", effort: "low" });
  assert.deepEqual(codex.args.slice(0, 5), ["exec", "-i", "/tmp/f.jpg", "-m", "gpt-6-astra"]);
  assert.ok(codex.args.includes("--json"));
});

test("Claude's JSON envelope and Codex's event lines both yield the answer, usage and cost", () => {
  const claude = parseCliResult("claude", JSON.stringify({ result: "Q", total_cost_usd: 0.0192, usage: { input_tokens: 37, output_tokens: 108 }, is_error: false }));
  assert.deepEqual({ text: claude.text, costUsd: claude.costUsd, tokens: claude.usage.output_tokens }, { text: "Q", costUsd: 0.0192, tokens: 108 });
  assert.equal(parseCliResult("claude", JSON.stringify({ result: "rate limited", is_error: true })).error, "rate limited");
  const codex = parseCliResult("codex", '{"type":"thread.started"}\n{"type":"item.completed","item":{"type":"agent_message","text":"W"}}\n{"type":"turn.completed","usage":{"input_tokens":18646,"output_tokens":43}}\n');
  assert.deepEqual({ text: codex.text, tokens: codex.usage.input_tokens }, { text: "W", tokens: 18646 });
  assert.equal(parseCliResult("codex", "junk\nE\n").text, "E");
});

function fakeSpawn(script) {
  return (command, args) => {
    const child = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.kill = () => { child.killed = true; child.emit("close", 137); };
    setTimeout(() => {
      const out = script({ command, args });
      if (out === null) return; // never answers: the deadline must fire
      child.stdout.emit("data", out.stdout ?? "");
      child.stderr.emit("data", out.stderr ?? "");
      child.emit("close", out.code ?? 0);
    }, 5);
    return child;
  };
}

test("the CLI engine writes the frame, spawns the right client and returns a token with cost", async () => {
  let seen = null;
  const infer = createCliInfer({ spawnImpl: fakeSpawn(({ command, args }) => { seen = { command, args }; return { stdout: JSON.stringify({ result: "Q", total_cost_usd: 0.02, usage: { input_tokens: 40, output_tokens: 5 } }) }; }) });
  const out = await infer("claude-fable-5-1", { imageB64: Buffer.from("jpg").toString("base64"), depth: { center_m: 1.2 } }, { instruction: "reach the door", deadlineMs: 5000 });
  assert.equal(out.token, "Q");
  assert.equal(out.costUsd, 0.02);
  assert.equal(out.source, "cli:claude");
  assert.equal(seen.command, "claude");
  assert.ok(seen.args[1].includes("reach the door") && seen.args[1].includes("image file at"));
  assert.ok(seen.args[1].includes("center_m"), "the depth line rides along in the prompt");
  const timeout = createCliInfer({ spawnImpl: fakeSpawn(() => null) });
  await assert.rejects(timeout("gpt-6-astra", { imageB64: "aGk=" }, { instruction: "go", deadlineMs: 30 }), /exceeded 30 ms/u);
  const junk = createCliInfer({ spawnImpl: fakeSpawn(() => ({ stdout: '{"type":"item.completed","item":{"type":"agent_message","text":"I cannot see"}}' })) });
  await assert.rejects(junk("gpt-6-astra", { imageB64: "aGk=" }, { instruction: "go" }), /without a valid token/u);
  await assert.rejects(infer("deepseek-v4-flash", { imageB64: "aGk=" }, { instruction: "go" }), /no CLI serves/u);
});
