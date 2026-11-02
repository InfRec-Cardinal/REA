import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { createEpisodeStore, taskSimilarity } from "../src/episodes.mjs";

function fill(store, id, { taskKey = "reach-doorway", task = "reach the doorway", steps = 4, refuseFirst = 2, at = 1_700_000_000_000 } = {}) {
  for (let seq = 1; seq <= steps; seq += 1) {
    const refused = seq <= refuseFirst;
    store.record(id, {
      taskKey,
      task,
      seq,
      token: refused ? "W" : "Q",
      engine: seq === steps ? "react" : "token",
      route: seq === steps ? "refusal_streak" : "default",
      allowed: !refused,
      reasons: refused ? ["depth:depth_center_near"] : [],
      executed: !refused,
      outcome: refused ? "refused" : "progress",
      score: refused ? -0.6 : 0.2,
      situation: refused ? "c:near|l:far|r:near|lidar:clear" : "c:far|l:far|r:far|lidar:clear",
      centerM: refused ? 0.4 : 2.5,
      pose: { x: 0, y: 0, heading: 0 },
      revisits: 0,
      at: at + seq * 1000,
    });
  }
}

test("a session is digested into a few honest numbers and one prompt line", () => {
  const store = createEpisodeStore({ now: () => 1_700_000_000_000 });
  fill(store, "s1");
  store.review("s1", { seq: 2, verdict: "STOP", next: "X" });
  store.review("s1", { seq: 3, verdict: "KEEP", next: "Q" });
  store.judgement("s1", { seq: 2, outcome: "unsafe" });
  const d = store.digest("s1");
  assert.equal(d.steps, 4);
  assert.equal(d.refusals, 2);
  assert.equal(d.refusal_rate, 0.5);
  assert.equal(d.executed, 2);
  assert.equal(d.escalations, 1);
  assert.equal(d.deliberate, 1);
  assert.deepEqual(d.tokens, { W: 2, Q: 2 });
  assert.deepEqual(d.refused_where, { "W at centre near": 2 });
  assert.equal(d.mean_score, -0.2);
  assert.equal(d.last_outcome, "progress");
  assert.equal(d.last_token, "Q");
  assert.deepEqual(d.reviews, { KEEP: 1, CORRECT: 0, STOP: 1 });
  assert.deepEqual(d.judgements, { unsafe: 1 });
  assert.equal(d.duration_s, 3);
  const line = store.describe(d);
  assert.match(line, /^Past episode \(11-14 22:13\): 4 steps; 2 refused \(2x W at centre near\); 1 escalations; reviewer STOP x1, CORRECT x0; ended progress on Q\.$/u);
  assert.equal(store.digest("nope"), null);
  assert.equal(store.record("", { seq: 1 }), null, "a step needs a session id");
  assert.equal(store.record("s1", { seq: 1.5 }), null, "and an integer seq");
});

test("recall prefers the same task, then a similar one (named as such), newest first, never the current session", () => {
  const store = createEpisodeStore();
  fill(store, "old", { at: 1_000_000 });
  fill(store, "newer", { at: 2_000_000, refuseFirst: 0 });
  fill(store, "other-task", { taskKey: "kitchen-table", task: "go to the kitchen table", at: 3_000_000 });
  fill(store, "similar", { taskKey: "doorway", task: "the doorway", at: 4_000_000 });
  fill(store, "half", { taskKey: "doorway-left", task: "the doorway on the left", at: 4_500_000 });
  fill(store, "me", { at: 5_000_000 });
  const recalled = store.recall({ taskKey: "reach-doorway", exclude: "me", k: 3 });
  assert.deepEqual(recalled.map(d => d.session_id), ["newer", "old", "similar"], "exact matches newest first, then the similar task; one shared word in three is not similar");
  assert.deepEqual(recalled.map(d => d.match), ["same", "same", "similar"]);
  assert.deepEqual(store.recall({ taskKey: "kitchen-table", k: 5 }).map(d => d.session_id), ["other-task"], "no other session is like the kitchen task");
  fill(store, "left", { taskKey: "turn-left", task: "turn left", at: 6_000_000 });
  fill(store, "right", { taskKey: "turn-right", task: "turn right", at: 6_100_000 });
  assert.deepEqual(store.recall({ taskKey: "turn-left", exclude: "left", k: 3 }), [], "the opposite task is not this task");
  assert.equal(taskSimilarity("reach-doorway", "doorway"), 1);
  assert.equal(taskSimilarity("turn-left", "turn-right"), 0);
  assert.equal(taskSimilarity("generic", "generic"), 0);
  const rendered = store.render(recalled, { maxChars: 150 });
  assert.equal(rendered.split("\n").length, 1, "the cap keeps one line");
  assert.ok(rendered.startsWith("Past episode ("));
  assert.match(store.describe(recalled[2]), /^Past episode \(similar task "doorway", /u, "a similar task is named, never passed off as this one");
  assert.equal(store.render([]), "");
  assert.equal(store.recall({ taskKey: "generic", k: 2 }).length, 0, "the generic key is no task and recalls nothing");
  assert.match(store.describe({ ...recalled[0], ended_at: 1e300 }), /^Past episode \(earlier\)/u, "an out-of-range timestamp never throws");
});

test("the digest counts escalations by the router's verdict, and a runaway session keeps only its newest steps once", () => {
  const store = createEpisodeStore({ maxStepsPerSession: 5 });
  const base = { taskKey: "reach-doorway", token: "W", allowed: true, outcome: "progress", score: 0.2 };
  store.record("s", { ...base, seq: 1, engine: "token", route: "default", escalated: false, at: 1 });
  store.record("s", { ...base, seq: 2, engine: "token", route: "budget_exhausted", escalated: false, at: 2 });
  store.record("s", { ...base, seq: 3, engine: "react", route: "memory", escalated: true, at: 3 });
  store.record("s", { ...base, seq: 4, engine: "react", route: "refusal_streak", escalated: true, at: 4 });
  store.record("s", { ...base, seq: 4, engine: "react", route: "refusal_streak", escalated: true, at: 4 });
  assert.equal(store.digest("s").steps, 4, "the same seq twice is one step");
  assert.equal(store.digest("s").escalations, 2, "budget_exhausted ran the cheap engine; a learned preference for react is an escalation");
  const legacy = createEpisodeStore();
  legacy.record("l", { ...base, seq: 1, route: "budget_exhausted", at: 1 });
  legacy.record("l", { ...base, seq: 2, route: "refusal_streak", at: 2 });
  assert.equal(legacy.digest("l").escalations, 1, "lines written before the flag fall back to the reasons that always meant react ran");
  for (let seq = 5; seq <= 20; seq += 1) store.record("s", { ...base, seq, at: seq });
  assert.equal(store.digest("s").steps, 5, "capped to the newest steps");
  assert.deepEqual(store.digest("s").tokens, { W: 5 });
  assert.equal(store.stats().steps, 5);
});

test("episodes round-trip through the JSONL file and the file is compacted to the newest sessions", async () => {
  const dir = await mkdtemp(join(tmpdir(), "nea-episodes-"));
  const path = join(dir, "nested", "episodes.jsonl");
  try {
    const store = createEpisodeStore({ path, maxSessions: 2 });
    await store.load();
    fill(store, "a", { at: 1_000_000 });
    fill(store, "b", { at: 2_000_000 });
    store.review("b", { seq: 1, verdict: "CORRECT", next: "Q" });
    fill(store, "c", { at: 3_000_000, steps: 2 });
    await store.flush();
    const raw = (await readFile(path, "utf8")).trim().split("\n");
    assert.equal(raw.length, 4 + 4 + 1 + 2, "every record is one appended line");
    assert.deepEqual(store.sessions(), ["b", "c"], "the oldest session was evicted from memory");

    const reloaded = createEpisodeStore({ path, maxSessions: 2 });
    const loaded = await reloaded.load();
    assert.equal(loaded.sessions, 2);
    assert.deepEqual(reloaded.sessions().sort(), ["b", "c"]);
    assert.deepEqual(reloaded.digest("b").reviews, { KEEP: 0, CORRECT: 1, STOP: 0 }, "reviews are re-attached on load");
    assert.equal(reloaded.digest("c").steps, 2);
    await reloaded.flush();
    const compacted = (await readFile(path, "utf8")).trim().split("\n");
    assert.equal(compacted.length, 4 + 1 + 2, "the evicted session's lines were dropped from the file");
    assert.equal(reloaded.stats().lines, 7);
    assert.equal(reloaded.stats().loaded, true);

    // a record() that lands while the file is being read is appended after the compaction, not lost
    const racing = createEpisodeStore({ path, maxSessions: 1 });
    const loading = racing.load();
    racing.record("live", { taskKey: "reach-doorway", seq: 1, token: "W", allowed: true, outcome: "preview", at: 9_000_000 });
    await loading;
    await racing.flush();
    const after = (await readFile(path, "utf8")).trim().split("\n").map(line => JSON.parse(line).s);
    assert.ok(after.includes("live"), "the racing record survived the compaction");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("a corrupt line is skipped and a missing file is an empty store", async () => {
  const dir = await mkdtemp(join(tmpdir(), "nea-episodes-"));
  try {
    const path = join(dir, "episodes.jsonl");
    const missing = createEpisodeStore({ path });
    assert.deepEqual(await missing.load(), { loaded: 0, sessions: 0 });
    const { writeFile } = await import("node:fs/promises");
    await writeFile(path, `{"k":"step","s":"x","t":"go","seq":1,"tok":"W","ok":true,"out":"preview","at":5}\nnot json\n{"k":"weird"}\n`, "utf8");
    const store = createEpisodeStore({ path });
    const loaded = await store.load();
    assert.equal(loaded.loaded, 1);
    assert.equal(store.digest("x").steps, 1);
    await store.flush();
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
