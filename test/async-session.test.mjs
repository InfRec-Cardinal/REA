import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createAsyncSession } from "../src/framework/async-session.mjs";
import { createAsyncSystemDemo } from "../examples/async-system-demo.mjs";
import { createAsyncConsole } from "../scripts/async_console.mjs";

test("sessions submit immediately, replay from a cursor, and preserve runner callbacks", async () => {
  const wait = Promise.withResolvers();
  const callbacks = [];
  const session = createAsyncSession({ mode: "async", onEvent: event => callbacks.push(event) }, { run: async ({ onEvent, onTransition }) => {
    onEvent({ type: "thinking", status: "started", id: "think-1", atMs: 0 });
    await wait.promise;
    await onTransition({ type: "step", step: 1, observation: { id: "0", state: { x: 0 }, images: ["large image"] }, nextObservation: { id: "1", state: { x: 1 } } });
    onEvent({ type: "thinking", status: "completed", id: "think-1", atMs: 10 });
    return { reason: "success", success: true };
  } });
  assert.equal(session.status().state, "queued");
  const first = [];
  const unsubscribe = session.subscribe(event => first.push(event));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(session.status().state, "running");
  assert.equal(first.at(-1).type, "thinking");
  const cursor = first.at(-1).sequence;
  unsubscribe();
  wait.resolve();
  assert.equal((await session.done).success, true);
  const replay = [];
  const stopReplay = session.subscribe(event => replay.push(event), { after: cursor });
  assert.equal(replay[0].sequence, cursor + 1);
  assert.equal(replay.at(-1).status, "completed");
  assert.equal(replay.find(event => event.type === "transition").transition.observation.images, undefined);
  assert.equal(callbacks.length, 2);
  assert.deepEqual(session.history().map(event => event.sequence), Array.from({ length: session.history().length }, (_, index) => index + 1));
  assert.equal(session.cancel(), false);
  stopReplay();
});

test("cancelling a live brain and executor completes the session without later actions", async () => {
  const session = createAsyncSession(createAsyncSystemDemo({ brainMs: 1000, actionMs: 500, policyMs: 1 }));
  session.subscribe(event => { if (event.type === "action" && event.status === "started") session.cancel("test stop"); });
  const summary = await session.done;
  assert.equal(summary.reason, "cancelled");
  assert.equal(session.status().state, "cancelled");
  assert.ok(session.history().some(event => event.type === "thinking" && event.status === "cancelled"));
  assert.equal(session.history().filter(event => event.type === "action" && event.status === "started").length, 1);
  assert.equal(summary.steps, 0);
});

async function openConsole(t, options) {
  const { server, sessions } = await createAsyncConsole(options);
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  t.after(async () => {
    await new Promise(resolve => { server.close(resolve); server.closeAllConnections(); });
    await Promise.allSettled([...sessions.values()].map(session => session.done));
  });
  return { url: `http://127.0.0.1:${server.address().port}`, sessions };
}

test("console runs the actual runtime and SSE resumes after Last-Event-ID", async t => {
  const { url, sessions } = await openConsole(t, { demoOptions: { brainMs: 35, actionMs: 3, policyMs: 1 } });
  const started = await fetch(`${url}/api/runs`, { method: "POST", body: JSON.stringify({ mode: "async" }) });
  assert.equal(started.status, 202);
  const run = await started.json();
  const summary = await sessions.get(run.id).done;
  assert.equal(summary.success, true);
  const history = sessions.get(run.id).history();
  const after = history.at(-3).sequence;
  const controller = new AbortController();
  const response = await fetch(`${url}/api/runs/${run.id}/events`, { headers: { "last-event-id": String(after) }, signal: controller.signal });
  assert.match(response.headers.get("content-type"), /text\/event-stream/);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let text = "";
  while (!text.includes(`id: ${history.at(-1).sequence}\n`)) text += decoder.decode((await reader.read()).value);
  controller.abort();
  assert.match(text, /event: snapshot/);
  assert.ok(text.includes(`id: ${after + 1}\n`));
  assert.ok(!text.includes(`id: ${after}\n`));
  const status = await (await fetch(`${url}/api/runs/${run.id}`)).json();
  assert.equal(status.state, "completed");
  assert.ok(status.summary.asyncMetrics.routedWhileThinking > 0);
  assert.match(await (await fetch(url)).text(), /One shared timeline/);
});

test("benchmark replay retains original timing and cannot be cancelled", async t => {
  const directory = await mkdtemp(join(tmpdir(), "async-replay-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const file = join(directory, "events.jsonl");
  await writeFile(file, [
    { type: "episode", status: "started", mode: "async", atMs: 0 },
    { type: "action", status: "started", id: "action-1", atMs: 52 },
    { type: "action", status: "completed", id: "action-1", atMs: 93 },
    { type: "episode", status: "completed", atMs: 94, summary: { mode: "async", reason: "success", durationMs: 94, success: true } },
  ].map(event => JSON.stringify(event)).join("\n") + "\n");
  const { url } = await openConsole(t, { replayPath: file });
  const replay = await (await fetch(`${url}/api/runs/benchmark-replay`)).json();
  assert.equal(replay.readOnly, true);
  assert.equal(replay.source, "benchmark");
  assert.equal(replay.events[1].atMs, 52);
  assert.equal(replay.summary.durationMs, 94);
  assert.equal((await fetch(`${url}/api/runs/benchmark-replay/cancel`, { method: "POST" })).status, 409);
});
