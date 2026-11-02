// Two-tier memory for the agent brain.
//
// The split follows the ReAct-framework convention of keeping a sequential "chat" memory apart from a
// retrievable "knowledge" memory:
//
//   EpisodicMemory    per-session ring buffer of step records — what was proposed, what the gate
//                     said, what happened. Cheap derived signals (refusal streak, oscillation, stall)
//                     are what the router escalates on.
//   ExperienceMemory  cross-session lessons, persisted as JSONL, keyed by a coarse situation
//                     signature and a task key, consolidated by hits, bounded in size, and rendered
//                     into a short prompt block. Corrections are the most valuable lessons: a vetoed
//                     or reviewer-corrected token in a recognisable situation becomes advice for the
//                     next time that situation is seen.
//
// Nothing in this module can move the robot. Memory only changes what is *asked* of the model; every
// answer still goes through the same fused gate, and the prompt block is size-capped so it cannot
// erode the one-token latency contract.

import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

import { G1_PROFILE } from "./profiles/g1.mjs";
import { flattenLine, promptTextError } from "./sanitise.mjs";

/** The G1 opposite pairs and motion set, kept as exports for existing callers. */
export const OPPOSITE_TOKENS = G1_PROFILE.opposite;
const VELOCITY_TOKENS = new Set(G1_PROFILE.motion);

/**
 * How much a lesson's origin is trusted when ranking recall. The reviewer and an external judge saw
 * the outcome (before/after frames, a replay, a human); a reflection reasoned over several steps; the
 * heuristic judge only saw the gate verdict. Corrections therefore outrank same-scored heuristics.
 */
export const SOURCE_WEIGHT = Object.freeze({ reviewer: 1.6, external: 1.5, reflection: 1.3, heuristic: 1.0 });
const ALTERNATIVE_WEIGHT = 1.15;

const STOPWORDS = new Set([
  "the", "a", "an", "to", "and", "or", "of", "in", "on", "at", "for", "with", "from", "toward", "towards",
  "into", "then", "until", "while", "is", "are", "be", "it", "its", "that", "this", "very", "as", "by",
  "please", "robot", "move", "go", "walk", "choose", "one", "immediate", "action", "current", "frame",
  "nearest", "safest", "safe", "avoid", "avoiding", "collision", "stop", "when", "path", "unclear",
]);

// ---------------------------------------------------------------------------------------------------
// Situation signature
// ---------------------------------------------------------------------------------------------------

/** Which side has more room, coarsely: two doorways with the same centre depth often differ here. */
export function asymmetryBucket(leftM, rightM) {
  const left = Number(leftM);
  const right = Number(rightM);
  if (!Number.isFinite(left) || !Number.isFinite(right) || left <= 0 || right <= 0) return "unknown";
  if (left > right * 1.4) return "left";
  if (right > left * 1.4) return "right";
  return "even";
}

/** Bits that differ between two 64-bit hex hashes; null when either is missing or malformed. */
export function hammingHex(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || !/^[0-9a-f]{16}$/iu.test(a) || !/^[0-9a-f]{16}$/iu.test(b)) return null;
  let x = BigInt(`0x${a}`) ^ BigInt(`0x${b}`);
  let count = 0;
  while (x) {
    count += Number(x & 1n);
    x >>= 1n;
  }
  return count;
}

/**
 * Scene bands, measured on recorded G1 frames (9x8 dHash of the greyscale frame): frames of one
 * doorway approach differ by 0..7 bits, re-encoding at 320 px / quality 40 by 0, exposure by <= 2, a
 * 5 % shift by ~7; frames of different rooms by 11..33 (median 24); two random hashes by 32 +- 4. So
 * <= SAME is the same view, > DIFFERENT is a different place, and the band between says nothing.
 */
export const SCENE_SAME_BITS = 10;
export const SCENE_DIFFERENT_BITS = 20;

export function depthBucket(metres, profile = G1_PROFILE) {
  if (metres === null || metres === undefined || !Number.isFinite(Number(metres))) return "unknown";
  const [near, far] = profile.sensing?.buckets?.depth ?? [0.6, 1.5];
  const value = Number(metres);
  if (value < near) return "near";
  if (value < far) return "mid";
  return "far";
}

/**
 * Coarse, comparable description of the moment a decision was made. Buckets are deliberately wide so
 * that lessons generalise: "centre near, lidar clear" recurs; "centre 0.57 m" never does.
 */
export function situationOf({ depth = null, lidar = null, state = null, lastToken = null, profile = G1_PROFILE } = {}) {
  // "clear" covers both a clear scan and a disabled lidar (the gate reports usable, not blocked,
  // reason "lidar_disabled"); only an actual veto reason or an unusable sensor changes the bucket.
  let lidarBucket = "unknown";
  if (lidar) {
    if (lidar.usable === false) lidarBucket = "unusable";
    else if (lidar.blocked === true) lidarBucket = typeof lidar.reason === "string" ? lidar.reason : "blocked";
    else if (lidar.blocked === undefined && typeof lidar.reason === "string" && /near/u.test(lidar.reason)) lidarBucket = lidar.reason;
    else lidarBucket = "clear";
  }
  return {
    center: depthBucket(depth?.center_m, profile),
    left: depthBucket(depth?.left_m, profile),
    right: depthBucket(depth?.right_m, profile),
    asym: asymmetryBucket(depth?.left_m, depth?.right_m),
    lidar: lidarBucket,
    ready: state ? state.ready === true : null,
    lastToken: profile.tokenSet.has(lastToken) ? lastToken : null,
  };
}

export function situationKey(situation) {
  const s = situation ?? {};
  const asym = s.asym && s.asym !== "unknown" ? `|a:${s.asym}` : "";
  return `c:${s.center ?? "unknown"}|l:${s.left ?? "unknown"}|r:${s.right ?? "unknown"}|lidar:${s.lidar ?? "unknown"}${asym}`;
}

/**
 * Two situation keys describe the same moment when their depth and lidar buckets agree and the
 * asymmetry, where both have one, agrees too. A key written before the asymmetry existed matches any.
 */
export function sameSituationKey(a, b) {
  if (a === b) return true;
  if (typeof a !== "string" || typeof b !== "string") return false;
  const sa = parseSituationKey(a);
  const sb = parseSituationKey(b);
  if (sa.center !== sb.center || sa.left !== sb.left || sa.right !== sb.right || sa.lidar !== sb.lidar) return false;
  const asymA = sa.asym && sa.asym !== "unknown" ? sa.asym : null;
  const asymB = sb.asym && sb.asym !== "unknown" ? sb.asym : null;
  return asymA === null || asymB === null || asymA === asymB;
}

/** The router's "dimension" (ACRouter vocabulary): what kind of moment is this, coarsely. */
export function dimensionOf(situation) {
  const s = situation ?? {};
  const lidar = s.lidar === "clear" || s.lidar === "unknown" ? "clear" : s.lidar === "unusable" ? "nolidar" : "lidar_near";
  return `${s.center ?? "unknown"}/${lidar}`;
}

export function situationPhrase(situation) {
  const s = situation ?? {};
  const parts = [`centre ${s.center ?? "unknown"}`];
  if (s.left && s.left !== "unknown") parts.push(`left ${s.left}`);
  if (s.right && s.right !== "unknown") parts.push(`right ${s.right}`);
  if (s.lidar && s.lidar !== "unknown") parts.push(`lidar ${s.lidar}`);
  if (s.asym === "left" || s.asym === "right") parts.push(`wider ${s.asym}`);
  return parts.join(", ");
}

export function taskKeywords(instruction, { max = 6 } = {}) {
  const words = String(instruction ?? "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/u)
    .filter(word => word.length > 1 && !STOPWORDS.has(word));
  const seen = new Set();
  const out = [];
  for (const word of words) {
    if (seen.has(word)) continue;
    seen.add(word);
    out.push(word);
    if (out.length >= max) break;
  }
  return out;
}

export function taskKey(instruction) {
  const words = taskKeywords(instruction);
  return words.length ? words.join("-") : "generic";
}

// ---------------------------------------------------------------------------------------------------
// Episodic memory
// ---------------------------------------------------------------------------------------------------

function isOpposite(a, b, profile) {
  return profile.opposite[a] === b;
}

/**
 * Per-session ring buffer of step records. A record is whatever the caller stores, but the signals
 * below expect: `seq`, `token`, `engine`, `gate: {allowed, reasons}`, `executed`, `depth`, `situation`,
 * and optionally `judge: {score, outcome}` and `review: {verdict, next_token}`.
 */
export function createEpisodicMemory({ capacity = 32, profile = G1_PROFILE } = {}) {
  const steps = [];

  function append(record) {
    const entry = { ...record };
    steps.push(entry);
    while (steps.length > capacity) steps.shift();
    return entry;
  }

  function get(seq) {
    for (let index = steps.length - 1; index >= 0; index -= 1) {
      if (steps[index].seq === seq) return steps[index];
    }
    return null;
  }

  function update(seq, patch) {
    const entry = get(seq);
    if (!entry) return null;
    Object.assign(entry, patch);
    return entry;
  }

  function last(count = 1) {
    return steps.slice(-count);
  }

  function signals() {
    let refusalStreak = 0;
    for (let index = steps.length - 1; index >= 0; index -= 1) {
      const step = steps[index];
      if (step.token === profile.roles.stop) continue; // a stop is always allowed and says nothing about being stuck
      if (step.gate && step.gate.allowed === false) refusalStreak += 1;
      else break;
    }

    let badStreak = 0;
    for (let index = steps.length - 1; index >= 0; index -= 1) {
      const score = steps[index].judge?.score;
      if (typeof score === "number" && score < 0) badStreak += 1;
      else break;
    }

    const recentTokens = steps.slice(-4).map(step => step.token).filter(Boolean);
    let oscillation = 0;
    for (let index = 1; index < recentTokens.length; index += 1) {
      if (isOpposite(recentTokens[index - 1], recentTokens[index], profile)) oscillation += 1;
    }

    // Stall: the last three executed forward pulses moved the centre depth by less than 5 cm in total.
    let stall = false;
    const forwardToken = profile.roles.forward ?? null;
    const forward = forwardToken === null ? [] : steps.slice(-3).filter(step => step.token === forwardToken && step.executed === true);
    if (forward.length === 3) {
      const first = forward[0].depth?.center_m;
      const lastDepth = forward[2].afterDepth?.center_m ?? forward[2].depth?.center_m;
      if (Number.isFinite(first) && Number.isFinite(lastDepth) && Math.abs(first - lastDepth) < 0.05) stall = true;
    }

    const latest = steps.at(-1) ?? null;
    return {
      steps: steps.length,
      refusalStreak,
      badStreak,
      oscillation,
      stall,
      lastToken: latest?.token ?? null,
      lastEngine: latest?.engine ?? null,
      lastSituation: latest?.situation ?? null,
      lastReview: latest?.review?.verdict ?? null,
      lastScore: latest?.judge?.score ?? null,
    };
  }

  /** One line per recent step, bounded, for the deliberate prompt and the reviewer. */
  function summary({ max = 5 } = {}) {
    return steps
      .slice(-max)
      .map(step => {
        const parts = [`#${step.seq ?? "?"} ${step.token ?? "?"}`];
        if (step.engine) parts.push(step.engine);
        if (step.gate && step.gate.allowed === false) parts.push(`vetoed(${(step.gate.reasons ?? []).join(",")})`);
        else if (step.executed) parts.push("executed");
        else parts.push("preview");
        if (step.review?.verdict) parts.push(`review:${step.review.verdict}${step.review.next_token ? `→${step.review.next_token}` : ""}`);
        if (typeof step.judge?.score === "number") parts.push(`judge:${step.judge.score > 0 ? "+" : ""}${step.judge.score.toFixed(1)}`);
        return parts.join(" ");
      })
      .join(" | ");
  }

  return {
    append,
    get,
    update,
    last,
    all: () => steps.slice(),
    size: () => steps.length,
    signals,
    summary,
    clear: () => steps.splice(0),
  };
}

// ---------------------------------------------------------------------------------------------------
// Experience memory
// ---------------------------------------------------------------------------------------------------

function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

function validLesson(lesson, profile = G1_PROFILE) {
  if (lesson === null || typeof lesson !== "object" || Array.isArray(lesson)) return "lesson must be an object";
  if (typeof lesson.lesson !== "string" || lesson.lesson.trim().length === 0 || lesson.lesson.length > 240) {
    return "lesson text must be 1..240 characters";
  }
  // Model-authored text becomes part of a later prompt: refuse anything that reads as an instruction.
  const textError = promptTextError(lesson.lesson);
  if (textError) return `lesson text rejected: ${textError}`;
  if (typeof lesson.score !== "number" || !Number.isFinite(lesson.score) || lesson.score < -1 || lesson.score > 1) {
    return "score must be a finite number in [-1, 1]";
  }
  if (lesson.token !== null && lesson.token !== undefined && !profile.tokenSet.has(lesson.token)) return "token is not in the token set";
  if (typeof lesson.outcome !== "string" || lesson.outcome.length === 0 || lesson.outcome.length > 40) return "outcome is required";
  return null;
}

/**
 * Cross-session lessons. `remember()` consolidates duplicates (same task key, situation, token and
 * outcome) by bumping `hits` and blending the score, so a lesson that keeps being re-learned gains
 * weight rather than crowding the store. `recall()` ranks by situation match, task match, weight and
 * recency; `render()` turns the top lessons into a size-capped prompt block.
 */
export function createExperienceMemory({
  path = null,
  maxLessons = 500,
  now = () => Date.now(),
  logger = null,
  idFactory = null,
  persistDelayMs = 250,
  profile = G1_PROFILE,
} = {}) {
  let lessons = [];
  let counter = 0;
  let persistQueue = Promise.resolve();
  let persistTimer = null;
  let dirty = false;
  let loaded = path === null;
  const rejected = {}; // reason -> count, so a judge that keeps writing instructions is visible
  const nextId = idFactory ?? (() => `l${now().toString(36)}${(++counter).toString(36)}`);

  async function load() {
    if (path === null) {
      loaded = true;
      return { loaded: 0 };
    }
    let text = "";
    try {
      text = await readFile(path, "utf8");
    } catch (error) {
      if (error.code !== "ENOENT") logger?.warn?.({ event: "experience_load_failed", path, error: error.message });
      loaded = true;
      return { loaded: 0 };
    }
    const parsed = [];
    let dropped = 0;
    for (const line of text.split(/\r?\n/u)) {
      if (!line.trim()) continue;
      try {
        const record = JSON.parse(line);
        const error = validLesson(record, profile);
        if (error === null) {
          // A hand-edited or pre-filter record may still carry a raw newline: one line, like remember().
          parsed.push({ ...record, lesson: flattenLine(record.lesson, 240) });
        } else {
          // Persisted before the filter existed, or edited by hand: dropped, and visible like a live rejection.
          dropped += 1;
          rejected[rejectionKey(error)] = (rejected[rejectionKey(error)] ?? 0) + 1;
          logger?.warn?.({ event: "experience_load_rejected", reason: error, id: record?.id ?? null, preview: typeof record?.lesson === "string" ? record.lesson.slice(0, 80) : null });
        }
      } catch {
        /* skip a corrupt line rather than refusing to start */
      }
    }
    lessons = parsed.slice(-maxLessons);
    loaded = true;
    if (dropped > 0) dirty = true; // rewrite the file without the dropped records on the next persist
    return { loaded: lessons.length, rejected: dropped };
  }

  function rejectionKey(error) {
    return error.replace(/^lesson text rejected: /u, "").replace(/[^a-z_]/gu, "_").slice(0, 40);
  }

  /** Rotate the file to `<path>.bak-<unix_ms>` (queued before any later persist). Returns the backup path or null. */
  function backup() {
    if (path === null) return null;
    const target = `${path}.bak-${Math.floor(now())}`;
    const run = async () => {
      try {
        await rename(path, target);
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
    };
    persistQueue = persistQueue.then(run, run).catch(error => logger?.warn?.({ event: "experience_backup_failed", path, error: error.message }));
    return target;
  }

  function writeNow() {
    if (path === null || !dirty) return persistQueue;
    dirty = false;
    const snapshot = lessons.map(lesson => JSON.stringify(lesson)).join("\n");
    const run = async () => {
      await mkdir(dirname(path), { recursive: true });
      const tmp = `${path}.tmp`;
      await writeFile(tmp, snapshot.length ? `${snapshot}\n` : "", "utf8");
      await rename(tmp, path);
    };
    persistQueue = persistQueue.then(run, run).catch(error => {
      logger?.warn?.({ event: "experience_persist_failed", path, error: error.message });
    });
    return persistQueue;
  }

  /**
   * The whole store is rewritten (tmp + rename), so writes are coalesced: a burst of lessons on the
   * reactive path costs one file write per `persistDelayMs`, not one per lesson. `flush()` writes now.
   */
  function persist() {
    if (path === null) return Promise.resolve();
    dirty = true;
    if (persistDelayMs <= 0) return writeNow();
    if (persistTimer === null) {
      persistTimer = setTimeout(() => {
        persistTimer = null;
        void writeNow();
      }, persistDelayMs);
      persistTimer.unref?.();
    }
    return persistQueue;
  }

  function flush() {
    if (persistTimer !== null) {
      clearTimeout(persistTimer);
      persistTimer = null;
    }
    return writeNow();
  }

  /**
   * Ranking weight. Corrections are the most valuable lessons: a reviewer or an external judge saw
   * the outcome and named what should have happened, whereas the heuristic judge only pattern-matched
   * the gate. So the source multiplies the weight, and a lesson that names an alternative token is
   * worth more than one that only says "that was bad".
   */
  function weight(lesson, at = now()) {
    const ageDays = Math.max(0, at - (lesson.lastAt ?? lesson.at ?? at)) / 86_400_000;
    const recency = 1 / (1 + ageDays / 14);
    const source = SOURCE_WEIGHT[lesson.source] ?? 1;
    const actionable = lesson.alternative ? ALTERNATIVE_WEIGHT : 1;
    return Math.abs(lesson.score) * (1 + Math.log(1 + (lesson.hits ?? 1))) * recency * source * actionable;
  }

  function evict() {
    if (lessons.length <= maxLessons) return;
    const at = now();
    lessons.sort((a, b) => weight(b, at) - weight(a, at));
    lessons = lessons.slice(0, maxLessons);
  }

  /** Store a lesson. Returns the stored (possibly consolidated) record, or `{ error }`. */
  function remember(input) {
    const error = validLesson(input, profile);
    if (error) {
      const reason = rejectionKey(error);
      rejected[reason] = (rejected[reason] ?? 0) + 1;
      logger?.warn?.({ event: "experience_lesson_rejected", reason: error, source: input?.source ?? null, preview: typeof input?.lesson === "string" ? input.lesson.slice(0, 80) : null });
      return { error };
    }
    const at = now();
    const record = {
      taskKey: typeof input.taskKey === "string" && input.taskKey ? input.taskKey : "generic",
      situation: typeof input.situation === "string" ? input.situation : situationKey(input.situation),
      token: input.token ?? null,
      outcome: input.outcome,
      score: clamp(input.score, -1, 1),
      // One line, printable, no brackets: model-authored text is rendered into a prompt, so a newline
      // or a bracket in it could forge a prompt line ("Advice: …", "[+] …") of its own.
      lesson: flattenLine(input.lesson, 240),
      source: typeof input.source === "string" && input.source ? input.source : "heuristic",
      alternative: profile.tokenSet.has(input.alternative) ? input.alternative : null,
      // The frame's perceptual hash, when the camera supplied one: what tells two doorways apart when
      // depth and lidar read the same.
      scene: typeof input.scene === "string" && /^[0-9a-f]{16}$/iu.test(input.scene) ? input.scene.toLowerCase() : null,
    };
    // Consolidation: same task, moment (a key without the asymmetry matches one with it), token and
    // outcome. With scenes on both sides they must be the same view. A record *without* a scene
    // (an external judge, a frame the worker could not hash) merges into a sceneless lesson first;
    // among scene-tagged lessons only when there is exactly one, and then it never overwrites that
    // lesson's text — a doorway-B judgement must not rewrite what was learned at doorway A.
    const sameKey = candidate =>
      candidate.taskKey === record.taskKey &&
      sameSituationKey(candidate.situation, record.situation) &&
      candidate.token === record.token &&
      candidate.outcome === record.outcome;
    const keyMatches = lessons.filter(sameKey);
    let existing = null;
    let overwriteText = true;
    if (record.scene) {
      existing =
        keyMatches.find(candidate => candidate.scene && hammingHex(candidate.scene, record.scene) <= SCENE_SAME_BITS) ??
        keyMatches.find(candidate => !candidate.scene) ??
        null;
    } else {
      existing = keyMatches.find(candidate => !candidate.scene) ?? null;
      if (!existing && keyMatches.length === 1) {
        existing = keyMatches[0];
        overwriteText = (SOURCE_WEIGHT[record.source] ?? 1) > (SOURCE_WEIGHT[existing.source] ?? 1);
      }
    }
    let stored;
    if (existing) {
      existing.hits = (existing.hits ?? 1) + 1;
      existing.score = clamp(0.7 * existing.score + 0.3 * record.score, -1, 1);
      if (overwriteText) {
        existing.lesson = record.lesson;
        existing.source = record.source;
      }
      existing.alternative = record.alternative ?? existing.alternative ?? null;
      existing.scene = existing.scene ?? record.scene;
      // Adopt the richer key when a pre-asymmetry lesson meets a moment that carries it.
      if (existing.situation !== record.situation && !parseSituationKey(existing.situation).asym && parseSituationKey(record.situation).asym) existing.situation = record.situation;
      existing.lastAt = at;
      stored = existing;
    } else {
      stored = { id: nextId(), at, lastAt: at, hits: 1, ...record };
      lessons.push(stored);
      evict();
    }
    void persist();
    return stored;
  }

  /**
   * Rank lessons for the moment at hand. `situation` may be a signature object or key; `taskKey` is
   * the current instruction's key; `token`, when given, prefers lessons about that token.
   */
  function recall({ taskKey: key = "generic", situation = null, token = null, scene = null, k = 3, minWeight = 0.05 } = {}) {
    const sig = typeof situation === "string" ? parseSituationKey(situation) : (situation ?? {});
    const keywords = new Set(String(key ?? "").split("-").filter(Boolean));
    const at = now();
    const scored = lessons
      .map(lesson => {
        const lessonSig = parseSituationKey(lesson.situation);
        let match = 0;
        if (sig.center && lessonSig.center === sig.center) match += 3;
        if (sig.lidar && lessonSig.lidar === sig.lidar) match += 2;
        if (sig.left && lessonSig.left === sig.left) match += 1;
        if (sig.right && lessonSig.right === sig.right) match += 1;
        if (sig.asym && sig.asym !== "unknown" && lessonSig.asym === sig.asym) match += 1;
        // Same view as the lesson was learned from: strong evidence; clearly a different place: weaker;
        // in between the hash says nothing.
        const bits = hammingHex(scene, lesson.scene);
        if (bits !== null) {
          if (bits <= SCENE_SAME_BITS) match += 3;
          else if (bits > SCENE_DIFFERENT_BITS) match -= 1;
        }
        if (lesson.taskKey === key) match += 2;
        else if (lesson.taskKey.split("-").some(word => keywords.has(word))) match += 1;
        if (token && lesson.token === token) match += 1;
        if (lesson.taskKey === "generic" && key !== "generic") match += 0.5;
        const w = weight(lesson, at) * (1 + match);
        return { lesson, match, weight: w };
      })
      .filter(entry => entry.match > 0 && entry.weight >= minWeight)
      .sort((a, b) => b.weight - a.weight);
    return scored.slice(0, k).map(entry => ({ ...entry.lesson, relevance: Number(entry.weight.toFixed(3)) }));
  }

  /** Prompt block. Capped so it can never dominate a one-token prompt. */
  function render(recalled, { maxChars = 240 } = {}) {
    if (!recalled || recalled.length === 0) return "";
    const lines = [];
    let used = 0;
    for (const lesson of recalled) {
      const sign = lesson.score < 0 ? "-" : "+";
      const line = `[${sign}] ${lesson.lesson}`;
      if (used + line.length + 1 > maxChars) break;
      lines.push(line);
      used += line.length + 1;
    }
    return lines.join("\n");
  }

  function forget({ taskKey: key = null, all = false } = {}) {
    dirty = true;
    const before = lessons.length;
    if (all) lessons = [];
    else if (key) lessons = lessons.filter(lesson => lesson.taskKey !== key);
    void persist();
    return { removed: before - lessons.length, size: lessons.length };
  }

  function stats() {
    const byOutcome = {};
    const bySource = {};
    for (const lesson of lessons) {
      byOutcome[lesson.outcome] = (byOutcome[lesson.outcome] ?? 0) + 1;
      bySource[lesson.source] = (bySource[lesson.source] ?? 0) + 1;
    }
    return { size: lessons.length, max: maxLessons, path, loaded, byOutcome, bySource, rejected: { ...rejected } };
  }

  return {
    load,
    remember,
    recall,
    render,
    forget,
    backup,
    stats,
    all: () => lessons.slice(),
    flush,
  };
}

export function parseSituationKey(key) {
  const out = {};
  for (const part of String(key ?? "").split("|")) {
    const [name, value] = part.split(":");
    if (!name || value === undefined) continue;
    if (name === "c") out.center = value;
    else if (name === "l") out.left = value;
    else if (name === "r") out.right = value;
    else if (name === "lidar") out.lidar = value;
    else if (name === "a") out.asym = value;
  }
  return out;
}

export { VELOCITY_TOKENS };
