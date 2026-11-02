// The brain: memory + judge + router composed around one control step.
//
//   plan()             before the step — read the episodic signals, route to an engine (ACRouter),
//                      recall lessons for this task and situation, build the deliberate context.
//   record()           after the step — judge the outcome, remember the lesson, update the router's
//                      per-dimension engine scores, append the episodic record. If a reflex token was
//                      vetoed and the deliberate engine answered inline, the refusal is recorded as a
//                      lesson against the reflex engine as well.
//   learnFromReview()  when the asynchronous reviewer's verdict arrives, stale or not — corrections
//                      are the most valuable lessons.
//   learnLesson()      an out-of-process judge posts a lesson (loopback endpoint).
//   learnFromJudgement() the reviewer's outcome assessment of a past step (the asynchronous judge).
//
// Two ways one step empowers the next:
//   directive          the deliberate engine may finish with a `next` line — what the following step
//                      should watch for or do. It is carried to exactly the next step's prompt as
//                      advice (never a command) and then dropped: a robot that has moved on has moved on.
//   reflection         when the episodic signals say the episode is going wrong (a veto, an oscillation,
//                      a stall, a refusal streak), a text-only model reflects over the last few steps and
//                      writes one lesson (Reflexion). Fire-and-forget, rate-limited, memory only.
//
// The brain changes what is asked and whom it is asked of. It never touches the actuator, and every
// token it influences still crosses the same fused gate.

import { performance } from "node:perf_hooks";

import { externalLessonError, judgeOutcome, judgeReview, judgeStep } from "./judge.mjs";
import { costUsd } from "./candidates.mjs";
import { createEpisodicMemory, dimensionOf, situationKey, situationOf, taskKey } from "./memory.mjs";
import { G1_PROFILE } from "./profiles/g1.mjs";
import { sanitisePromptText } from "./sanitise.mjs";
import { createSpatialMemory } from "./spatial.mjs";

/** The deliberate engine's `next` line is advice for one step only; longer than this and it is a plan, not a hint. */
export const DIRECTIVE_MAX_CHARS = 120;

function gateSummary(gate) {
  if (!gate) return null;
  return { allowed: gate.allowed === true, reasons: [...(gate.reasons ?? [])] };
}

function lidarSummary(lidar) {
  if (!lidar) return null;
  return {
    usable: lidar.usable ?? null,
    blocked: lidar.blocked ?? null,
    reason: lidar.reason ?? null,
    center: typeof lidar.center === "number" ? lidar.center : null,
    left: typeof lidar.left === "number" ? lidar.left : null,
    right: typeof lidar.right === "number" ? lidar.right : null,
  };
}

function stateSummary(state) {
  if (!state) return null;
  const rpy = Array.isArray(state.rpy) && state.rpy.length >= 3 ? state.rpy : null;
  return { ready: state.ready === true, reasons: [...(state.reasons ?? [])], yaw: rpy && Number.isFinite(rpy[2]) ? rpy[2] : null };
}

export function createBrain({
  experience,
  router,
  logger = null,
  now = () => performance.now(),
  episodicCapacity = 32,
  recallK = 3,
  promptMaxChars = 240,
  reflect = null,
  episodes = null,
  episodeRecallK = 2,
  candidates = null,
  prices = null,
  modelRouter = null,
  profile = G1_PROFILE,
}) {
  if (!experience || typeof experience.recall !== "function") throw new TypeError("brain needs an experience memory");
  if (!router || typeof router.route !== "function") throw new TypeError("brain needs a router");
  if (reflect !== null && (typeof reflect.infer !== "function" || typeof reflect.model !== "string" || !reflect.model)) {
    throw new TypeError("brain reflect needs { infer, model }");
  }
  const reflectCfg = reflect ? { deadlineMs: 6000, cooldown: 3, minScore: -0.4, ...reflect } : null;
  const nextRejected = {}; // reason -> count of `next` lines refused by the sanitiser
  const candidateById = candidates?.byId ?? new Map();
  const routerModelJobs = { started: 0, settled: 0, failed: 0 };
  const reflections = { inflight: null, started: 0, completed: 0, failed: 0, skipped: 0, lastLesson: null, lastError: null, lastTrigger: null };
  const motionTokens = new Set(profile.motion);

  function attach(session) {
    if (!session.episodic) session.episodic = createEpisodicMemory({ capacity: episodicCapacity, profile });
    // Where this session's robot has been, by dead reckoning from the pulses it dispatched.
    if (!session.spatial) session.spatial = createSpatialMemory();
    return session.episodic;
  }

  function recallFor({ instruction, situation, token = null, scene = null, k = recallK }) {
    if (k <= 0) return [];
    return experience.recall({ taskKey: taskKey(instruction), situation, token, scene, k });
  }

  function sceneOf(frame) {
    const hash = frame?.scene?.dhash;
    return typeof hash === "string" && /^[0-9a-f]{16}$/iu.test(hash) ? hash.toLowerCase() : null;
  }

  /**
   * The previous step's `next` line, consumed exactly once and only if it came from the step that
   * immediately precedes this one. Anything older describes a robot that has since moved.
   */
  function takeDirective(session) {
    const directive = session.directive ?? null;
    session.directive = null;
    if (!directive) return null;
    return directive.fromSeq === session.seq ? directive : null;
  }

  /**
   * Route and recall for the step about to run. `observe` is an optional read-only callback the
   * deliberate engine may call to look again; `explicitEngine` is a caller-forced engine or null.
   */
  function plan(session, { instruction, explicitEngine = null, observe = null, armHeld = null } = {}) {
    const episodic = attach(session);
    const spatial = session.spatial.signals();
    const signals = { ...episodic.signals(), revisits: spatial.revisits, recentRevisits: spatial.recentRevisits, refusedHere: spatial.refusedHere };
    const situation = signals.lastSituation;
    const lastScene = episodic.last(1)[0]?.scene ?? null;
    const route = router.route({ signals, situation, explicit: explicitEngine, seq: (session.seq ?? 0) + 1, sessionId: session.id });
    const key = taskKey(instruction);
    const recalled = recallFor({ instruction, situation, scene: lastScene });
    const lessonsText = experience.render(recalled, { maxChars: promptMaxChars });
    const recent = episodic.summary({ max: 5 });
    const where = episodic.size() > 0 ? session.spatial.summary() : null;
    // How the last sessions on this task went. Deliberate prompt and reviewer only: the reflex path
    // pays for every character, and a digest is context, not a lesson.
    const pastEpisodes = episodes ? episodes.recall({ taskKey: key, exclude: session.id, k: episodeRecallK }) : [];
    const history = episodes ? episodes.render(pastEpisodes, { maxChars: 400 }) || null : null;
    const directive = takeDirective(session);
    const context = {
      recent,
      where,
      history,
      lessons: lessonsText || null,
      armHeld,
      directive: directive?.text ?? null,
      evidence: signals.lastSituation ? { depth: episodic.last(1)[0]?.depth ?? null, lidar: episodic.last(1)[0]?.lidar ?? null, state: episodic.last(1)[0]?.state ?? null } : null,
      observe: typeof observe === "function" ? observe : null,
      recall: async query => {
        const lessons = recallFor({ instruction: query || instruction, situation, scene: lastScene, k: Math.max(recallK, 3) });
        return experience.render(lessons, { maxChars: Math.max(promptMaxChars, 480) });
      },
    };
    return { engine: route.engine, route, taskKey: key, signals, recalled, lessonsText, recent, where, history, pastEpisodes: pastEpisodes.length, directive, context };
  }

  /**
   * Reflexion: ask a text-only model for one lesson about the last few steps. Fire-and-forget (the
   * step never waits), one in flight at a time, at most one per `cooldown` steps per session. The
   * result only ever reaches memory.
   */
  function maybeReflect(session, { seq, trigger, situation, key, token, engine }) {
    if (!reflectCfg) return null;
    if (reflections.inflight) {
      reflections.skipped += 1;
      return { status: "busy", trigger };
    }
    if (Number.isFinite(session.lastReflectSeq) && seq - session.lastReflectSeq < reflectCfg.cooldown) {
      reflections.skipped += 1;
      return { status: "cooldown", trigger };
    }
    session.lastReflectSeq = seq;
    const episodic = attach(session);
    const known = experience.render(recallFor({ instruction: session.task, situation, k: Math.max(recallK, 3) }), { maxChars: 480 });
    const input = { instruction: session.task, situation, trigger, recent: episodic.summary({ max: 8 }), lessons: known || null, token };
    const controller = new AbortController();
    // The engine gets the deadline too, but the brain enforces it: a reflector that never settles
    // must not pin `busy` for the life of the process.
    const timer = setTimeout(() => controller.abort(new Error(`reflection deadline exceeded ${reflectCfg.deadlineMs} ms`)), reflectCfg.deadlineMs + 500);
    timer.unref?.();
    reflections.started += 1;
    reflections.lastTrigger = trigger;
    const run = (async () => {
      try {
        const out = await Promise.race([
          reflectCfg.infer(reflectCfg.model, input, { signal: controller.signal, deadlineMs: reflectCfg.deadlineMs }),
          new Promise((_, reject) => controller.signal.addEventListener("abort", () => reject(controller.signal.reason ?? new Error("reflection aborted")), { once: true })),
        ]);
        const confidence = Number.isFinite(out?.confidence) ? Math.min(1, Math.max(0, out.confidence)) : 0.5;
        const alternative = out?.alternative ? profile.parseToken(out.alternative) : null;
        const stored = experience.remember({
          taskKey: key,
          situation: situationKey(situation),
          scene: episodic.last(1)[0]?.scene ?? null,
          token,
          outcome: "reflection",
          // A reflection is only ever asked for after something went wrong, so it is a negative lesson
          // about the last token; confidence decides how negative.
          score: -(0.3 + 0.4 * confidence),
          lesson: String(out?.lesson ?? "").trim().slice(0, 240),
          alternative: alternative !== token ? alternative : null,
          source: "reflection",
        });
        if (!stored || stored.error) throw new Error(stored?.error ?? "reflection was not stored");
        reflections.completed += 1;
        reflections.lastLesson = stored.lesson;
        episodic.update(seq, { reflection: { lesson: stored.lesson, alternative: stored.alternative, confidence, trigger } });
        logger?.info?.({ event: "brain_reflection", session_id: session.id, seq, trigger, lesson: stored.lesson, alternative: stored.alternative, engine });
      } catch (error) {
        reflections.failed += 1;
        reflections.lastError = error?.message ?? String(error);
        logger?.warn?.({ event: "brain_reflection_failed", session_id: session.id, seq, trigger, error: reflections.lastError });
      } finally {
        clearTimeout(timer);
        if (reflections.inflight === run) reflections.inflight = null;
      }
    })();
    reflections.inflight = run;
    return { status: "started", trigger };
  }

  function learn(lesson, dimension, engine, score, { scene = null, latencyS = null, costUsd = null } = {}) {
    let stored = null;
    if (lesson) {
      stored = experience.remember(scene ? { ...lesson, scene } : lesson);
      if (stored?.error) {
        logger?.warn?.({ event: "brain_lesson_rejected", error: stored.error });
        stored = null;
      }
    }
    // Only real engines earn routing scores: a forced token or an unknown label says nothing about
    // which model to ask next time, and would pollute the per-dimension memory with phantom engines.
    const routable = typeof engine === "string" && engine && (typeof router.has !== "function" || router.has(engine));
    if (dimension && routable && Number.isFinite(score)) router.observe(dimension, engine, score, { latencyS, costUsd });
    return stored;
  }

  /** What one decision cost, from the provider's token counts and the candidate's (or model's) price. */
  function stepCost(result, candidateId) {
    if (Number.isFinite(result?.costUsd)) return result.costUsd;
    const model = result?.model ?? candidateById.get(candidateId)?.model ?? null;
    return prices && model ? costUsd(model, result?.usage ?? null, prices) : null;
  }

  /**
   * The rendered experience block for a decision about to be made outside the routed step path
   * (`/v1/act`, the pipeline). It uses the session's last known situation, so the reflex prompt on
   * the reactive path carries the same lessons the routed path would see.
   */
  function experienceFor(session, instruction) {
    const episodic = attach(session);
    const situation = episodic.signals().lastSituation;
    const recalled = recallFor({ instruction, situation, scene: episodic.last(1)[0]?.scene ?? null });
    return { recalled, text: experience.render(recalled, { maxChars: promptMaxChars }) || null };
  }

  /**
   * Judge and remember the step that just ran.
   *   result     the decision result (ok, token, engine, thought)
   *   executed   the executor's report (gate, executed, dispatch, arm_held, preview)
   *   frame      the frame the decision was made on
   *   reflex     when the deliberate engine answered inline: { token, engine, gate } of the vetoed reflex
   */
  function record(session, { seq, result, executed, frame, engine, candidate = null, route = null, inline = null, taskKey: key = "generic", reflex = null, routed = true, frameAt = null }) {
    // The router remembers candidates (engine@model), the episodic ring remembers the harness engine.
    const candidateId = candidate ?? engine;
    const stepCostUsd = stepCost(result, candidateId);
    const latencyS = Number.isFinite(result?.timing?.modelReadyS) ? result.timing.modelReadyS : null;
    const episodic = attach(session);
    const previous = episodic.last(1)[0] ?? null;
    const depth = frame?.depth ?? null;
    const wallAt = Date.now();
    const capturedAt = Number.isFinite(frameAt) ? frameAt : wallAt;
    const lidar = lidarSummary(executed?.gate?.lidar);
    const state = stateSummary(executed?.gate?.state);
    const gate = gateSummary(executed?.gate);
    const token = result.ok ? result.token : profile.roles.stop;
    const signals = episodic.signals();
    const situation = situationOf({ depth, lidar, state, lastToken: previous?.token ?? null, profile });
    const dimension = dimensionOf(situation);
    const scene = sceneOf(frame);

    // Retroactive progress judgement: this frame is the "after" view of the last forward pulse — but
    // only if it was captured after that pulse ended (a pipeline decision may sit on a frame from
    // *before* the pulse; judging that as "no progress" would teach a false stall), and only if no other
    // motion ran in between (a turn would confound the depth change). Stops and previews are skipped.
    let retro = null;
    let pulse = null;
    for (const entry of episodic.last(3).reverse()) {
      if (entry.executed && entry.token === profile.roles.forward && !entry.afterDepth) {
        pulse = entry;
        break;
      }
      if (entry.executed && motionTokens.has(entry.token)) break;
    }
    const afterPulse = pulse?.pulseEndAt == null || capturedAt >= pulse.pulseEndAt;
    if (pulse && depth && afterPulse) {
      pulse.afterDepth = depth;
      retro = judgeStep({
        token: pulse.token,
        engine: pulse.engine,
        ok: true,
        gate: pulse.gate,
        executed: true,
        dispatchOk: pulse.dispatchOk ?? null,
        depth: pulse.depth,
        afterDepth: depth,
        lidar: pulse.lidar,
        state: pulse.state,
        taskKey: pulse.taskKey ?? key,
        profile,
      });
      pulse.retro = { score: retro.score, outcome: retro.outcome, seq };
      if (retro.outcome !== "progress" || (pulse.judge?.score ?? 0) === 0) {
        learn(retro.lesson, dimensionOf(pulse.situation), pulse.engine, retro.score, { scene: pulse.scene ?? null });
        pulse.judge = { ...pulse.judge, score: retro.score, outcome: retro.outcome, retro: true };
      }
    }

    // The vetoed reflex attempt, when the deliberate engine took over inline (ACRouter: a cheap
    // attempt that applied but did not resolve is still an observation about the cheap model). When
    // the escalation failed the step's own judgement below already covers that same refused token.
    let reflexJudge = null;
    if (reflex && inline?.ok === true) {
      reflexJudge = judgeStep({
        token: reflex.token,
        engine: reflex.engine,
        ok: true,
        gate: gateSummary(reflex.gate),
        executed: false,
        depth: reflex.depth ?? depth,
        lidar: lidarSummary(reflex.gate?.lidar) ?? lidar,
        state: stateSummary(reflex.gate?.state) ?? state,
        signals,
        armHeld: executed?.arm_held ?? null,
        taskKey: key,
        profile,
      });
      learn(reflexJudge.lesson, dimension, reflex.engine, reflexJudge.score, { scene });
    }

    const judged = judgeStep({
      token,
      engine,
      ok: result.ok,
      errorType: result.errorType ?? null,
      gate,
      executed: executed?.executed === true,
      dispatchOk: executed?.dispatch ? executed.dispatch.ok === true : null,
      depth,
      lidar,
      state,
      signals,
      armHeld: executed?.arm_held ?? null,
      taskKey: key,
      profile,
    });
    learn(judged.lesson, dimension, judged.attributable === false ? null : candidateId, judged.score, { scene, latencyS, costUsd: stepCostUsd });
    // Only steps the router chose count against its budget and cooldown; a reactive act that never
    // consulted it still teaches the per-dimension scores above, but is not one of its decisions.
    if (routed) router.commit(candidateId, { inline: Boolean(inline) });

    // The deliberate engine's `next` line is model-authored text bound for the next prompt: sanitised
    // like a lesson, and a rejection is counted and logged rather than silently becoming "no plan".
    const nextLine = sanitisePromptText(result.next, DIRECTIVE_MAX_CHARS);
    const nextRejection = result.nextRejected ?? (nextLine.error && nextLine.error !== "not_text" && nextLine.error !== "empty" ? nextLine.error : null);
    if (nextRejection) {
      nextRejected[nextRejection] = (nextRejected[nextRejection] ?? 0) + 1;
      logger?.warn?.({ event: "brain_next_rejected", reason: nextRejection, engine, preview: typeof result.next === "string" ? result.next.slice(0, 80) : null });
    }
    const entry = episodic.append({
      seq,
      token,
      engine,
      candidate: candidateId,
      costUsd: stepCostUsd,
      usage: result?.usage ?? null,
      route: route?.reason ?? null,
      escalated: route?.escalated ?? false,
      inline: inline ? { from: inline.from, engine: inline.engine, ok: inline.ok } : null,
      thought: result.thought ?? null,
      gate,
      executed: executed?.executed === true,
      preview: executed?.preview ?? !executed?.executed,
      dispatchOk: executed?.dispatch ? executed.dispatch.ok === true : null,
      depth,
      lidar,
      state,
      situation,
      scene,
      taskKey: key,
      judge: { score: judged.score, outcome: judged.outcome, alternative: judged.alternative },
      next: nextLine.text,
      directiveUsed: result.directiveUsed ?? null,
      at: now(),
      wallAt,
      capturedAt,
      // When the pulse this step dispatched will have ended: the earliest a frame can judge it.
      pulseEndAt: executed?.executed === true && Number.isFinite(executed?.motion?.duration) ? wallAt + executed.motion.duration * 1000 : null,
    });

    // The deliberate engine's `next` line empowers exactly the following step of this session.
    if (entry.next) session.directive = { text: entry.next, fromSeq: seq, token, engine, at: entry.at };

    // Spatial memory: mark what happened at the pose the step was decided from, then move the pose by
    // the pulse that was actually dispatched (the actuator may have clamped it). The IMU yaw, when the
    // actuator reports one, snaps the heading so turns do not accumulate as pure dead reckoning.
    if (Number.isFinite(state?.yaw)) session.spatial.observeHeading(state.yaw);
    const marked = session.spatial.mark({
      seq,
      token,
      executed: entry.executed,
      refused: gate?.allowed === false,
      reasons: gate?.reasons ?? [],
      outcome: judged.outcome,
      centerM: depth?.center_m ?? null,
    });
    // A superseded cerebellum plan is not "executed" but its completed pulses did move the robot:
    // executed_motion is what actually ran, whatever the dispatch verdict.
    const motion = executed?.dispatch?.executed_motion ?? (entry.executed ? executed?.motion ?? null : null);
    if (motion) session.spatial.advance(motion);
    entry.pose = marked.visit;
    entry.revisits = marked.revisits;
    episodes?.record(session.id, {
      taskKey: key,
      task: session.task ?? null,
      seq,
      token,
      engine,
      route: route?.reason ?? null,
      escalated: route?.escalated === true,
      candidate: candidateId,
      cost: stepCostUsd,
      allowed: gate?.allowed ?? null,
      reasons: gate?.reasons ?? [],
      executed: entry.executed,
      outcome: judged.outcome,
      score: judged.score,
      situation: situationKey(situation),
      centerM: depth?.center_m ?? null,
      pose: { x: marked.visit.x, y: marked.visit.y, heading: marked.visit.heading },
      revisits: marked.revisits,
      at: wallAt,
    });

    // Reflect when the episode is going wrong, by the same signals the router escalates on.
    const after = episodic.signals();
    const triggers = [];
    if (reflectCfg && judged.score <= reflectCfg.minScore) triggers.push(`judge:${judged.outcome}`);
    if (after.oscillation >= 2) triggers.push("oscillation");
    if (after.stall) triggers.push("stall");
    if (after.refusalStreak >= 2) triggers.push("refusal_streak");
    if (marked.revisits >= 2 || (marked.revisits >= 1 && marked.refusedHere >= 1)) triggers.push("revisit");
    const reflection = triggers.length > 0 ? maybeReflect(session, { seq, trigger: triggers.join(","), situation, key, token, engine }) : null;

    // The model router thinks about the *next* step in the background; a late answer simply goes unused.
    if (modelRouter && routed) {
      routerModelJobs.started += 1;
      const lessons = recallFor({ instruction: session.task, situation, k: 3 }).map(l => l.lesson);
      modelRouter
        .suggestNext(session, {
          seq: seq + 1,
          dimension,
          situation,
          signals: after,
          lastStep: { candidate: candidateId, token, outcome: judged.outcome, score: judged.score, latencyS, refused: gate?.allowed === false },
          memoryLines: lessons,
        })
        .then(() => { routerModelJobs.settled += 1; }, error => { routerModelJobs.failed += 1; logger?.warn?.({ event: "router_model_job_failed", error: error?.message ?? String(error) }); });
    }

    return { judge: judged, reflexJudge, retro, record: entry, dimension, directive: entry.next, reflection, spatial: session.spatial.signals(), candidate: candidateId, cost_usd: stepCostUsd };
  }

  /** Everything the brain holds about one session: the episodic ring and the spatial estimate. */
  function sessionState(session) {
    if (!session) return null;
    const episodic = attach(session);
    return {
      session_id: session.id,
      task: session.task ?? null,
      seq: session.seq ?? null,
      steps: episodic.all().map(step => ({
        seq: step.seq,
        token: step.token,
        engine: step.engine,
        route: step.route,
        allowed: step.gate?.allowed ?? null,
        reasons: step.gate?.reasons ?? [],
        executed: step.executed,
        judge: step.judge ?? null,
        review: step.review ?? null,
        judgement: step.judgement ?? null,
        situation: step.situation ? situationKey(step.situation) : null,
        scene: step.scene ?? null,
        pose: step.pose ? { x: step.pose.x, y: step.pose.y, heading: step.pose.heading } : null,
        revisits: step.revisits ?? 0,
        next: step.next ?? null,
        at: step.wallAt ?? null,
      })),
      signals: episodic.signals(),
      spatial: session.spatial.snapshot(),
      trail: session.spatial.trail({ max: 32 }),
      directive: session.directive ?? null,
      digest: episodes ? episodes.digest(session.id) : null,
    };
  }

  /**
   * Forget what was learned, for an evaluation's cold arm. Experience and routing memory by default;
   * the episode store is kept unless asked (it is the evaluation's own evidence). Sessions in flight
   * keep their episodic rings.
   */
  function reset({ experience: clearExperience = true, router: clearRouter = true, episodes: clearEpisodes = false } = {}) {
    const before = { lessons: experience.stats().size, router_observations: router.state().observations ?? null, episodes: episodes ? episodes.stats().sessions : 0 };
    // Nothing persisted is truncated without a copy: each store first rotates its file to *.bak-<unix_ms>.
    const backups = {};
    if (clearExperience) {
      backups.experience = typeof experience.backup === "function" ? experience.backup() : null;
      experience.forget({ all: true });
    }
    if (clearRouter && typeof router.reset === "function") {
      backups.router = typeof router.backup === "function" ? router.backup() : null;
      router.reset();
    }
    let episodesCleared = false;
    if (clearEpisodes && episodes && typeof episodes.clear === "function") {
      backups.episodes = typeof episodes.backup === "function" ? episodes.backup() : null;
      episodes.clear();
      episodesCleared = true;
    }
    logger?.warn?.({ event: "brain_reset", experience: clearExperience, router: clearRouter, episodes: episodesCleared, before, backups });
    return { experience: clearExperience, router: clearRouter, episodes: episodesCleared, before, backups };
  }

  /** Digests of past sessions on a task, the way plan() would recall them. */
  function episodesFor({ instruction = "", taskKey: key = null, k = episodeRecallK } = {}) {
    if (!episodes) return { enabled: false, digests: [], rendered: "" };
    const digests = episodes.recall({ taskKey: key || taskKey(instruction), k: Math.max(1, Math.min(20, Number(k) || episodeRecallK)) });
    return { enabled: true, digests, rendered: episodes.render(digests, { maxChars: 2000 }) };
  }

  /** A reviewer verdict for a past step. Works for stale corrections too: a late lesson is still a lesson. */
  function learnFromReview(session, correction) {
    if (!session || !correction) return null;
    const episodic = attach(session);
    const entry = episodic.get(correction.basedOnSeq);
    if (!entry) return null;
    const judged = judgeReview({
      review: correction,
      token: entry.token,
      depth: entry.depth,
      lidar: entry.lidar,
      state: entry.state,
      taskKey: entry.taskKey ?? "generic",
      engine: entry.engine,
      profile,
    });
    if (!judged) return null;
    learn(judged.lesson, dimensionOf(entry.situation), entry.engine, judged.score, { scene: entry.scene ?? null });
    episodic.update(correction.basedOnSeq, {
      review: { verdict: correction.verdict, next_token: correction.next_token, brief: correction.brief, confidence: correction.confidence },
      reviewScore: judged.score,
    });
    episodes?.review(session.id, { seq: correction.basedOnSeq, verdict: correction.verdict, next: correction.next_token });
    logger?.info?.({ event: "brain_review_learned", session_id: session.id, seq: correction.basedOnSeq, verdict: correction.verdict, score: judged.score });
    return judged;
  }

  /**
   * The asynchronous judge's outcome assessment of a past step (the reviewer, having compared the
   * before and after frames). It arrives after the correction and is never on time to steer; it exists
   * to teach. A negative verdict also overrides a non-negative heuristic judgement of that step, so the
   * episodic signals the router escalates on reflect what actually happened, not only what the gate saw.
   *
   * The correction and the judgement come from the same reviewer call. The correction (learnFromReview)
   * already scored the engine for that step, so the judgement adds its lesson but not a second routing
   * observation — otherwise one reviewer opinion would count twice against the engine.
   */
  function learnFromJudgement(session, judgement) {
    if (!session || !judgement) return null;
    const episodic = attach(session);
    const entry = episodic.get(judgement.basedOnSeq);
    if (!entry) return null;
    const judged = judgeOutcome({
      judgement,
      token: entry.token,
      depth: entry.depth,
      lidar: entry.lidar,
      state: entry.state,
      taskKey: entry.taskKey ?? "generic",
      engine: entry.engine,
      profile,
    });
    // An unclear judgement with nothing to say was received but teaches nothing.
    if (!judged) return { score: 0, outcome: `review_${judgement.outcome}`, lesson: null, skipped: true };
    const alreadyScored = typeof entry.reviewScore === "number";
    learn(judged.lesson, dimensionOf(entry.situation), alreadyScored || judged.score === 0 ? null : entry.engine, judged.score, { scene: entry.scene ?? null });
    const patch = {
      judgement: { outcome: judgement.outcome, lesson: judgement.lesson ?? null, confidence: judgement.confidence ?? null, score: judged.score },
    };
    if (judged.score < 0 && (entry.judge?.score ?? 0) >= 0) patch.judge = { ...entry.judge, score: judged.score, outcome: judged.outcome, async: true };
    episodic.update(judgement.basedOnSeq, patch);
    episodes?.judgement(session.id, { seq: judgement.basedOnSeq, outcome: judgement.outcome });
    logger?.info?.({ event: "brain_judgement_learned", session_id: session.id, seq: judgement.basedOnSeq, outcome: judgement.outcome, score: judged.score, lesson: judged.lesson?.lesson ?? null });
    return judged;
  }

  /** An external judge's lesson (already shape-validated by the HTTP layer, validated again here). */
  function learnLesson(body, { source = "external" } = {}) {
    const error = externalLessonError(body, profile);
    if (error) return { accepted: false, reason: error };
    const situation = typeof body.situation === "string"
      ? body.situation
      : situationKey(body.situation ?? situationOf({ ...(body.evidence ?? {}), profile }));
    // The scene the judge looked at, when it says: a hash string, or the sense worker's scene object.
    const sceneHash = typeof body.scene === "string" ? body.scene : body.scene?.dhash ?? body.evidence?.scene?.dhash ?? null;
    const stored = experience.remember({
      taskKey: typeof body.taskKey === "string" && body.taskKey ? body.taskKey : taskKey(body.instruction),
      situation,
      token: body.token ? String(body.token).toUpperCase() : null,
      outcome: body.outcome,
      score: body.score,
      lesson: body.lesson,
      alternative: body.alternative ?? null,
      source: typeof body.source === "string" && body.source ? body.source : source,
      ...(typeof sceneHash === "string" ? { scene: sceneHash } : {}),
    });
    if (stored?.error) return { accepted: false, reason: stored.error };
    if (typeof body.engine === "string" && body.engine) {
      const dimension = typeof body.dimension === "string" && body.dimension ? body.dimension : dimensionOf(situationOf({ ...(body.evidence ?? {}), profile }));
      router.observe(dimension, body.engine, body.score);
    }
    return { accepted: true, id: stored.id, hits: stored.hits, size: experience.stats().size };
  }

  function recall({ instruction = "", situation = null, token = null, k = recallK } = {}) {
    const lessons = recallFor({ instruction, situation, token, k: Math.max(1, Math.min(20, Number(k) || recallK)) });
    return { lessons, rendered: experience.render(lessons, { maxChars: 2000 }) };
  }

  function reflectionStats() {
    return {
      enabled: reflectCfg !== null,
      model: reflectCfg?.model ?? null,
      cooldown: reflectCfg?.cooldown ?? null,
      busy: reflections.inflight !== null,
      started: reflections.started,
      completed: reflections.completed,
      failed: reflections.failed,
      skipped: reflections.skipped,
      last_trigger: reflections.lastTrigger,
      last_lesson: reflections.lastLesson,
      last_error: reflections.lastError,
    };
  }

  function state() {
    const memory = experience.stats();
    const routing = router.state();
    return {
      enabled: true,
      recall_k: recallK,
      prompt_max_chars: promptMaxChars,
      memory,
      router: routing,
      reflection: reflectionStats(),
      episodes: episodes ? episodes.stats() : { enabled: false },
      // `next` lines the deliberate engine wrote that read as instructions and were dropped, per reason.
      directives: { rejected: { ...nextRejected } },
      candidates: candidates ? candidates.list.map(c => ({ id: c.id, engine: c.engine, model: c.model, tier: c.tier, timeout_ms: c.timeoutMs, price_per_1m: c.price })) : null,
      model_router: modelRouter ? { ...modelRouter.stats(), jobs: { ...routerModelJobs } } : null,
      summary: {
        lessons: memory.size,
        episodes: episodes ? episodes.stats().sessions : 0,
        steps: routing.steps,
        deliberate_fraction: routing.deliberate_fraction,
        escalations: routing.escalations,
        inline_escalations: routing.inline_escalations,
        reflections: reflections.completed,
      },
    };
  }

  return {
    attach,
    plan,
    record,
    sessionState,
    takeDirective,
    learnFromReview,
    learnFromJudgement,
    learnLesson,
    recall,
    experienceFor,
    episodesFor,
    reset,
    state,
    router,
    experience,
    episodes,
    candidates,
    modelRouter,
    reflections: { idle: () => reflections.inflight ?? Promise.resolve(), stats: reflectionStats },
  };
}
