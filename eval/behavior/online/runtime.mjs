// REA's online agent on BEHAVIOR: the framework episode runtime (src/framework runEpisode + createRoutedPolicy) driving the R1Pro skills of
// eval/behavior/runner/run_episode.py. Spawned by agents_sdk_arm.run_episode as `node runtime.mjs <config.json>`;
// stdin/stdout are the runner's JSON-lines channel:
//   runtime -> runner {"type":"call","id","tool","args"} | {"type":"observe","id"} | {"type":"done",...} | {"type":"log",...}
//   runner -> runtime {"type":"result","id","value"}
// One decision = one worker.py process (one Agents SDK run); its helper calls are relayed to the runner here. The chosen skill
// runs as the environment step. stdout carries only the protocol (plus {"type":"progress","turns","usage"} after every decision);
// diagnostics go to stderr. The runner ends the runtime with SIGTERM (budget end): the running worker is killed with it and the
// records are written before exit.
import { stallReason } from './behavior_routing.mjs';
import { readFile, writeFile, rename, appendFile, mkdir } from 'node:fs/promises';
import { createWriteStream, existsSync, writeFileSync, appendFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

console.log = (...args) => console.error(...args);
const here = dirname(fileURLToPath(import.meta.url));
const config = JSON.parse(await readFile(process.argv[2], 'utf8'));
const frameworkRoot = resolve(here, '../../..');
const { runEpisode, createRoutedPolicy, validateAction } = await import(pathToFileURL(join(frameworkRoot, 'src/framework/index.mjs')));
const onlineDir = config.online_dir;
const episodeDir = config.episode_dir;
await mkdir(onlineDir, { recursive: true });
const pad = n => String(n).padStart(3, '0');
const writeJson = async (path, value) => { await writeFile(path + '.tmp', JSON.stringify(value, null, 1)); await rename(path + '.tmp', path); };

// ---- runner channel ----
let nextId = 0;
const pending = new Map();
const channel = createInterface({ input: process.stdin });
channel.on('line', line => {
  let msg;
  try { msg = JSON.parse(line); } catch { return; }
  if (msg.type === 'result' && pending.has(msg.id)) { const { resolve: done } = pending.get(msg.id); pending.delete(msg.id); done(msg.value); }
});
channel.on('close', () => { for (const { reject } of pending.values()) reject(new Error('runner closed the channel')); pending.clear(); });
function send(message) { return new Promise(done => process.stdout.write(JSON.stringify(message) + '\n', done)); }
function request(type, payload = {}) {
  const id = ++nextId;
  return new Promise((done, reject) => { pending.set(id, { resolve: done, reject }); send({ type, id, ...payload }); });
}

// ---- environment: one step = one skill executed by the runner ----
const actionSpecs = config.skills.map(skill => ({
  name: skill.name,
  parameters: {
    type: 'object', additionalProperties: false, required: Object.keys(skill.parameters),
    properties: Object.fromEntries(Object.entries(skill.parameters).map(([name, type]) => [name, { type }])),
  },
}));
// Accounting-only action for a decision without a valid take_action. Never exposed to the model; nothing runs.
const invalid = { name: 'invalid_output', parameters: { type: 'object', properties: {}, additionalProperties: false } };
const actions = [...actionSpecs, invalid];
let finished = false;
let budgetExhausted = false;
let executed = 0;
const observation = () => ({ id: `decision-${pad(executed + 1)}`, timestampMs: Date.now(), state: null, text: config.instruction, actions });
const environment = {
  id: 'rea-online-behavior',
  timing: 'turn-based',
  async reset() { return { observation: observation(), success: false }; },
  async step(action) {
    const started = Date.now();
    const result = action.name === 'invalid_output'
      ? 'NO ACTION: the decision produced no valid take_action call; nothing ran.'
      : String(await request('call', { tool: action.name, args: action.arguments }));
    executed += 1;
    const info = { result, seconds: (Date.now() - started) / 1000 };
    if (action.name === 'finish') { info.finish = { accepted: /^ACCEPTED/.test(result) }; finished = info.finish.accepted; }
    budgetExhausted = result.startsWith('BUDGET EXHAUSTED');
    return { observation: observation(), success: null, terminated: finished, truncated: budgetExhausted, info };
  },
  async stop() {},
};

// ---- policy: one worker process per decision ----
const helperNames = new Set(config.helper_specs.map(helper => helper.name));
const decisions = new Map();
let decisionIndex = 0;
let workerErrors = 0;
const usageTotal = { input_tokens: 0, output_tokens: 0, requests: 0 };

// The runner ends this runtime with SIGTERM (SIGKILL 5 s later): take the running decision's worker with it (no relay request
// or late decision.json outlives the episode) and leave the records the episode needs.
let activeWorker = null;
let resultWritten = false;
let episodeEnded = false;
function stopByRunner(signalName) {
  activeWorker?.kill('SIGKILL');
  if (!resultWritten) {
    resultWritten = true;
    try {
      writeFileSync(join(episodeDir, 'online-runtime-result.json'), JSON.stringify({ reason: 'stopped_by_runner', signal: signalName,
        stop_reason: finished ? 'agent_finished' : 'program_budget', steps: executed, decisions: decisionIndex, usage_total: usageTotal }, null, 1));
    } catch {}
    if (!episodeEnded) {
      try { appendFileSync(join(episodeDir, 'online-events.jsonl'), JSON.stringify({ type: 'episode_end', summary: { reason: 'stopped_by_runner', steps: executed } }) + '\n'); } catch {}
    }
  }
  process.exit(143);
}
for (const name of ['SIGTERM', 'SIGINT', 'SIGHUP']) process.on(name, () => stopByRunner(name));

function compactHistory(history) {
  return history.map(record => ({ step: record.step, skill: record.action?.name, arguments: record.action?.arguments ?? {},
    result: record.info?.result ?? null, seconds: record.info?.seconds ?? null, thought: record.thought ?? null }));
}

function runWorker(dir, signal, helperCalls) {
  return new Promise((done, reject) => {
    const child = spawn(config.agent_python, [config.worker ?? join(here, 'worker.py'), dir], { stdio: ['pipe', 'pipe', 'pipe'], env: process.env });
    activeWorker = child;
    const log = createWriteStream(join(dir, 'worker.stderr'));
    child.stderr.pipe(log);
    const lines = createInterface({ input: child.stdout });
    let chain = Promise.resolve();
    lines.on('line', line => {
      let msg;
      try { msg = JSON.parse(line); } catch { return; }
      if (msg.type !== 'call') return;
      chain = chain.then(async () => {
        let value;
        if (!helperNames.has(msg.tool)) value = `ERROR: ${msg.tool} is not a helper; run skills through take_action.`;
        else if (helperCalls.length >= (config.max_helper_calls ?? 3)) value = 'Helper budget used up; call take_action now.';
        else {
          const started = Date.now();
          value = String(await request('call', { tool: msg.tool, args: msg.args ?? {} }));
          helperCalls.push({ tool: msg.tool, args: msg.args ?? {}, result: value, seconds: (Date.now() - started) / 1000 });
        }
        if (child.exitCode === null && !child.stdin.destroyed) child.stdin.write(JSON.stringify({ type: 'result', id: msg.id, value }) + '\n');
      }).catch(error => console.error('helper relay failed:', error.message));
    });
    const abort = () => child.kill('SIGKILL');
    signal?.addEventListener('abort', abort, { once: true });
    child.on('error', error => { if (activeWorker === child) activeWorker = null; signal?.removeEventListener('abort', abort); reject(error); });
    child.on('close', code => { if (activeWorker === child) activeWorker = null; signal?.removeEventListener('abort', abort); log.end(); chain.then(() => done(code)); });
  });
}

function policy(route) {
  return {
    id: 'sdk-' + route,
    async decide(ctx) {
      const decisionDeadline = (Date.now() + (ctx.budget?.decisionMs ?? config.decision_timeout_ms ?? 600000)) / 1000;
      const image = String(await request('observe'));
      // the budget ended before this decision: no model call starts (and no decision is counted)
      if (image.startsWith('BUDGET EXHAUSTED')) throw Object.assign(new Error('program budget exhausted'), { code: 'program_budget' });
      const k = ++decisionIndex;
      const guidance = ctx.guidance ?? (route === 'deliberate' ? stallReason(ctx, config.patience ?? 3) : null);
      const dir = join(onlineDir, `decision-${pad(k)}`);
      await mkdir(dir, { recursive: true });
      const imageOk = !/^(ERROR|BUDGET EXHAUSTED)/.test(image) && existsSync(image);
      const context = {
        decision: k, max_decisions: config.max_decisions, budget_s: config.budget_s,
        remaining_s: config.deadline_epoch - Date.now() / 1000, route, guidance, notes: ctx.notes,
        instruction: config.instruction, history: compactHistory(ctx.history), image: imageOk ? image : null,
        image_status: imageOk ? 'ok' : image, config_path: process.argv[2], decision_deadline_epoch: decisionDeadline,
      };
      await writeJson(join(dir, 'context.json'), context);
      const helperCalls = [];
      const started = Date.now();
      let code = null;
      let decision;
      try {
        code = await runWorker(dir, ctx.signal, helperCalls);
        decision = JSON.parse(await readFile(join(dir, 'decision.json'), 'utf8'));
      } catch (error) {
        if (ctx.signal?.aborted) throw error;
        decision = { status: 'error', action: null, error: `worker failed (exit ${code}): ${error.message}` };
      }
      const record = {
        decision: k, route, guidance, image: context.image, remaining_s: Math.round(context.remaining_s), status: decision.status,
        helper_calls: helperCalls, action: decision.action ?? null, notes: decision.notes ?? ctx.notes, thought: decision.thought ?? null,
        usage: decision.usage ?? null, model_seconds: decision.model_seconds ?? null, decision_seconds: (Date.now() - started) / 1000,
        attempts: decision.attempts ?? null, error: decision.error ?? null, provider_error: decision.provider_error ?? null,
      };
      decisions.set(k, record);
      await writeJson(join(dir, 'record.json'), record);
      if (decision.usage) for (const name of Object.keys(usageTotal)) usageTotal[name] += decision.usage[name] ?? 0;
      await send({ type: 'progress', turns: k, usage: { ...usageTotal, decisions: k } });
      if (decision.status === 'provider_error') {
        throw Object.assign(new Error(`provider failed after retries: ${decision.provider_error} (${decision.error ?? ''})`), { code: 'provider_error' });
      }
      workerErrors = decision.status === 'error' ? workerErrors + 1 : 0;
      if (workerErrors >= (config.max_consecutive_worker_errors ?? 3)) {
        throw Object.assign(new Error(`${workerErrors} consecutive worker errors; last: ${decision.error}`), { code: 'worker_error' });
      }
      let action = { name: 'invalid_output', arguments: {} };
      if (decision.status === 'ok' && decision.action) {
        try { action = validateAction({ name: decision.action.skill, arguments: decision.action.arguments }, actions); }
        catch (error) { record.invalid = error.message; }
      }
      record.executed = action;
      return { action, notes: decision.notes ?? ctx.notes, thought: decision.thought ?? null, model: config.model,
        usage: decision.usage ? { input_tokens: decision.usage.input_tokens, output_tokens: decision.usage.output_tokens } : null };
    },
  };
}

function eventView(event) {
  const strip = observation => observation ? { id: observation.id, timestampMs: observation.timestampMs } : observation;
  return event.type === 'step' ? { ...event, observation: strip(event.observation), nextObservation: strip(event.nextObservation) }
    : event.type === 'reset' ? { type: 'reset', task: event.task, seed: event.seed } : event;
}

let summary = null;
let routed = null;
try {
  routed = createRoutedPolicy({
    reflex: policy('reflex'), deliberate: policy('deliberate'), patience: config.patience ?? 3,
    maxDeliberateFraction: config.max_deliberate_fraction ?? 0.5,
    shouldDeliberate: ctx => Boolean(stallReason(ctx, config.patience ?? 3)),
  });
  summary = await runEpisode({
    environment, policy: routed, task: config.task ?? null, seed: config.seed ?? 0, instruction: config.instruction,
    maxSteps: config.max_decisions, maxDurationMs: Math.max(1000, (config.budget_s + 600) * 1000),
    // No decision timeout of its own (as on RoboTwin): the runner ends the episode at the budget first.
    decisionTimeoutMs: config.decision_timeout_ms ?? Math.max(1000, (config.budget_s + 600) * 1000), historyLimit: config.history_limit ?? 12,
    maxNotesChars: config.max_notes_chars ?? 4000,
    onTransition: async event => {
      if (event.type === 'step') {
        const record = decisions.get(event.step);
        if (record) {
          Object.assign(record, { result: event.info?.result ?? null, result_seconds: event.info?.seconds ?? null, finish: event.info?.finish ?? null });
          await writeJson(join(onlineDir, `decision-${pad(event.step)}`, 'record.json'), record);
          await appendFile(join(episodeDir, 'online-decisions.jsonl'), JSON.stringify(record) + '\n');
        }
      }
      await appendFile(join(episodeDir, 'online-events.jsonl'), JSON.stringify(eventView(event)) + '\n');
      if (event.type === 'episode_end') episodeEnded = true;
    },
  });
} catch (error) {
  summary = { reason: 'error', steps: executed, notes: null, error: { code: error.code ?? 'runtime_error', message: error.message } };
}
await routed?.close?.().catch(() => {});
const code = summary.error?.code;
const stopReason = finished ? 'agent_finished'
  : code === 'provider_error' ? 'provider_error'
    : budgetExhausted || code === 'program_budget' ? 'program_budget'
      : summary.reason === 'max_steps' ? 'max_decisions'
        : summary.reason === 'deadline' ? 'decision_timeout'
          : summary.error ? 'agent_error' : summary.reason;
if (!resultWritten) {
  await writeJson(join(episodeDir, 'online-runtime-result.json'), { ...summary, stop_reason: stopReason, decisions: decisionIndex, usage_total: usageTotal });
  resultWritten = true;
}
const providerRecord = [...decisions.values()].reverse().find(record => record.provider_error);
await send({
  type: 'done', final_output: summary.notes ? String(summary.notes).slice(0, 2000) : null, turns: decisionIndex,
  usage: { ...usageTotal, decisions: decisionIndex }, stop_reason: stopReason,
  error: summary.error ? `${summary.error.code}: ${summary.error.message}`.slice(0, 1000) : null,
  ...(stopReason === 'provider_error' ? { provider_error: providerRecord?.provider_error ?? 'provider_error' } : {}),
});
process.exit(0);
