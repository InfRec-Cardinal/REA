import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { setTimeout as delay } from "node:timers/promises";
import { createRobotwinCodePolicy, ROBOTWIN_CODE_SYSTEM_PROMPT } from "../src/policies/robotwin-code.mjs";
import { createRobotwinCodexGenerator } from "../src/policies/robotwin-codex.mjs";

async function fixture(t, { hang = false, capacityOnce = false, capacityUsage = null } = {}) {
  const root = await mkdtemp(join(tmpdir(), "rea-codex-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const cli = join(root, "codex"), key = join(root, "key"), workRoot = join(root, "work");
  await writeFile(key, "test-key");
  await writeFile(cli, `#!${process.execPath}
import fs from 'node:fs';
import path from 'node:path';
const args=process.argv.slice(2);
if(args[0]==='--version'){console.log('codex-cli 0.test');process.exit(0)}
const flag=name=>args[args.indexOf(name)+1];
let prompt='';for await(const chunk of process.stdin)prompt+=chunk;
fs.writeFileSync('capture.json',JSON.stringify({args,prompt,keyPresent:!!process.env.REA_CODEX_PROVIDER_API_KEY}));
fs.appendFileSync('captures.jsonl',JSON.stringify({args,prompt})+'\\n');
const attempt=fs.readFileSync('captures.jsonl','utf8').trim().split('\\n').length;
if(${capacityOnce} && attempt===1){
 console.log(JSON.stringify({type:'turn.failed',error:{message:'rate limit exceeded: Temporary capacity limit. Please try again in 2s.'},
  ...(${JSON.stringify(capacityUsage)} ? {usage:${JSON.stringify(capacityUsage)}} : {})}));
 process.exit(1);
}
${hang ? `process.on('SIGTERM',()=>{fs.writeFileSync('stopped','yes');process.exit(0)});
fs.writeFileSync('started',String(process.pid));setInterval(()=>{},1000);` : `
fs.writeFileSync(flag('--output-last-message'),JSON.stringify({name:'run_python',arguments:{code:"move_gripper('left', [0,0,1], [1,0,0,0], 1)"},thought:'Use measured pose',notes:'next grasp',claim:{status:'working',evidence:'gripper open'}}));
console.log(JSON.stringify({type:'turn.completed',usage:{input_tokens:120,output_tokens:35,cached_input_tokens:20,reasoning_output_tokens:5}}));`}
`, { mode: 0o700 });
  const options = { codexPath: cli, model: "test-model", providerBaseUrl: "http://provider.invalid/v1", apiKeyFile: key, workRoot };
  const observation = { id: "observed-scene", state: { robot_state: { left: [0, 0, 1] } }, images:
    ["head_camera", "front_camera", "left_camera", "right_camera"].map((name, index) => ({ name, mimeType: "image/png", data: Buffer.from(`pixels-${index}`).toString("base64") })) };
  return { root, options, observation, async decisionDir() { return join(workRoot, (await readdir(workRoot))[0]); } };
}

test("Codex backend passes four ordered cameras and unchanged observation/history/feedback to structured Python output", async t => {
  const f = await fixture(t);
  const requests = [];
  const policy = createRobotwinCodePolicy({ ...f.options, backend: "codex", worker: {
    async request(request) { requests.push(request); return { ok: true, result: { action: { arm: "left" } } }; }, stop() {},
  } });
  t.after(() => policy.close());
  const result = await policy.decide({ observation: f.observation, instruction: "lift the can",
    history: [{ step: 2, action: { name: "move_gripper" }, observation: { images: [{ data: "oldpixels" }], text: "prior frame" } }],
    notes: "remember the can", feedback: { moved: true }, budget: { stepsRemaining: 1, remainingMs: 500000 } });
  const dir = await f.decisionDir();
  const capture = JSON.parse(await readFile(join(dir, "capture.json"), "utf8"));
  const imageArgs = capture.args.flatMap((value, index) => value === "--image" ? [capture.args[index + 1]] : []);
  assert.equal(imageArgs.length, 4);
  assert.deepEqual(await Promise.all(imageArgs.map(file => readFile(file, "utf8"))), ["pixels-0", "pixels-1", "pixels-2", "pixels-3"]);
  assert.match(capture.prompt, /lift the can/);
  assert.match(capture.prompt, /"stepsRemaining":1/);
  assert.match(capture.prompt, /remember the can/);
  assert.match(capture.prompt, /prior frame/);
  assert.doesNotMatch(capture.prompt, /oldpixels/);
  assert.match(capture.prompt, /"execution":\{"moved":true\}/);
  assert.match(capture.prompt, /Attached image 4: Camera: right_camera/);
  assert.equal(await readFile(join(dir, "instructions.txt"), "utf8"), ROBOTWIN_CODE_SYSTEM_PROMPT);
  assert.ok(capture.args.includes("features.shell_tool=false"));
  assert.ok(capture.args.includes('web_search="disabled"'));
  assert.equal(capture.keyPresent, true);
  assert.ok(!capture.args.some(arg => /test-key|model_reasoning_effort|max_output_tokens/.test(arg)));
  assert.deepEqual(result.action, { name: "move_gripper", arguments: { arm: "left" } });
  assert.equal(requests[0].observation, f.observation);
  assert.equal(result.notes, "next grasp");
  assert.equal(result.claim.status, "working");
  assert.equal(result.usage.total_tokens, 155);
  assert.equal(result.codex.version, "codex-cli 0.test");
  assert.equal(result.codex.outputTokenLimit, null);
  assert.equal(result.codex.apiRetries, true);
  assert.ok(!capture.args.some(arg => /request_max_retries|stream_max_retries/.test(arg)));
});

test("aborting a Codex decision waits for its child to exit and never executes Python", async t => {
  const f = await fixture(t, { hang: true });
  const controller = new AbortController();
  const generator = createRobotwinCodexGenerator({ ...f.options, systemPrompt: ROBOTWIN_CODE_SYSTEM_PROMPT });
  t.after(() => generator.close());
  const pending = generator.decide({ observation: f.observation, signal: controller.signal,
    actions: [{ name: "run_python", parameters: { type: "object", properties: { code: { type: "string" } }, required: ["code"], additionalProperties: false } }] });
  const rejected = assert.rejects(pending, { name: "AbortError" });
  let dir, pid;
  // Up to 10 s for the fake CLI to start: a loaded machine (the whole suite runs in parallel) can take seconds.
  for (let i = 0; i < 1000 && !pid; i += 1) {
    await delay(10);
    try { dir = await f.decisionDir(); pid = Number(await readFile(join(dir, "started"), "utf8")); } catch { /* wait for child readiness */ }
  }
  assert.ok(pid, "child reached inference");
  controller.abort();
  await rejected;
  assert.equal(await readFile(join(dir, "stopped"), "utf8"), "yes");
  assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
});

test("capacity failure retries the identical decision and executes Python only after success", async t => {
  const f = await fixture(t, { capacityOnce: true,
    capacityUsage: { input_tokens: 7, output_tokens: 3, cached_input_tokens: 0, reasoning_output_tokens: 1 } });
  let executions = 0;
  const policy = createRobotwinCodePolicy({ ...f.options, backend: "codex", worker: {
    async request() { executions += 1; return { ok: true, result: { action: { arm: "left" } } }; }, stop() {},
  } });
  t.after(() => policy.close());
  const result = await policy.decide({ observation: f.observation, instruction: "lift", budget: { stepsRemaining: 4 } });
  const dir = await f.decisionDir();
  const captures = (await readFile(join(dir, "captures.jsonl"), "utf8")).trim().split("\n").map(JSON.parse);
  assert.equal(captures.length, 2);
  assert.deepEqual(captures[0], captures[1]);
  assert.equal(executions, 1);
  assert.equal(result.codex.capacityRetries, 1);
  assert.ok(result.codex.attempts[0].retryWaitMs >= 2000 && result.codex.attempts[0].retryWaitMs <= 2250);
  assert.match(await readFile(join(dir, "events.attempt-01.jsonl"), "utf8"), /Temporary capacity limit/);
  assert.match(await readFile(join(dir, "events.attempt-02.jsonl"), "utf8"), /turn.completed/);
  assert.equal(result.usage.total_tokens, 165);
  assert.equal(result.usage.failed_attempts_usage[0].usage.input_tokens, 7);
  assert.equal(result.codex.decisionMs, 240000);
});

test("disabling API retries sets both CLI retry counts to zero and returns the first capacity failure", async t => {
  const f = await fixture(t, { capacityOnce: true });
  let executions = 0;
  const policy = createRobotwinCodePolicy({ ...f.options, backend: "codex", provider: "openrouter", apiRetries: false,
    worker: { async request() { executions += 1; }, stop() {} },
  });
  t.after(() => policy.close());
  await assert.rejects(policy.decide({ observation: f.observation, instruction: "lift", budget: { stepsRemaining: 4 } }),
    error => error.code === "codex_error" && /Temporary capacity limit/.test(error.message));
  const dir = await f.decisionDir();
  const captures = (await readFile(join(dir, "captures.jsonl"), "utf8")).trim().split("\n").map(JSON.parse);
  assert.equal(captures.length, 1);
  assert.equal(executions, 0);
  assert.ok(captures[0].args.includes("model_providers.openrouter.request_max_retries=0"));
  assert.ok(captures[0].args.includes("model_providers.openrouter.stream_max_retries=0"));
  const metadata = JSON.parse(await readFile(join(dir, "metadata.json"), "utf8"));
  assert.equal(metadata.codex.apiRetries, false);
  assert.equal(metadata.codex.attempts.length, 1);
  assert.equal(metadata.codex.capacityRetries, 0);
});

test("a capacity retry shares the original deadline and terminates its running CLI child", async t => {
  const f = await fixture(t, { capacityOnce: true, hang: true });
  const generator = createRobotwinCodexGenerator({ ...f.options, systemPrompt: ROBOTWIN_CODE_SYSTEM_PROMPT });
  t.after(() => generator.close());
  const started = performance.now();
  await assert.rejects(generator.decide({ observation: f.observation, signal: AbortSignal.timeout(4000),
    actions: [{ name: "run_python", parameters: { type: "object", properties: { code: { type: "string" } }, required: ["code"], additionalProperties: false } }] }),
  { name: "TimeoutError" });
  // A fresh 4 s deadline for the retry would end after at least 8 s; 7 s leaves headroom for a loaded machine.
  assert.ok(performance.now() - started < 7000, "retry did not receive a fresh deadline");
  const dir = await f.decisionDir();
  const pid = Number(await readFile(join(dir, "started"), "utf8"));
  assert.equal(await readFile(join(dir, "stopped"), "utf8"), "yes");
  assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
  const metadata = JSON.parse(await readFile(join(dir, "metadata.json"), "utf8"));
  assert.equal(metadata.codex.attempts.length, 2);
  assert.equal(metadata.codex.error, "deadline_or_abort");
});
