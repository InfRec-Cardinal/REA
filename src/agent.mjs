import { performance } from "node:perf_hooks";

import { defaultDecisionCfg } from "./config.mjs";
import { DEFAULT_INSTRUCTION, decide } from "./decide.mjs";
import { asEngineRegistry } from "./engines.mjs";
import { DEPTH_FLOOR_CENTER_M, DEPTH_FLOOR_SIDE_M, LIDAR_DEFAULTS, lidarGate } from "./gate.mjs";
import { externalJudgementError } from "./judge.mjs";
import { taskKey } from "./memory.mjs";
import { candidateCfg, costUsd, pricesFrom } from "./candidates.mjs";
import { DecisionPipeline } from "./pipeline.mjs";
import { G1_PROFILE } from "./profiles/g1.mjs";
import { SERVICE, VERSION } from "./version.mjs";

function seconds(ms) {
  return Number((ms / 1000).toFixed(4));
}

/**
 * The service object. Everything above it (HTTP, systemd) is transport; everything below it
 * (engines, gate, actuator) is mechanism. Three request shapes land here:
 *
 *   decide()  decision only, no actuator involved          (POST /v1/action)
 *   act()     decision -> fused gate -> motion, stateless  (POST /v1/act)
 *   step()    the same, but inside a session with the asynchronous thinker in the loop
 *             (POST /v1/control/step)
 */
export function createAgent({ camera, engines, executor, config, sessions, thinker = null, brain = null, profile = null, actionContract = null, logger = console, now = () => performance.now() }) {
  const contract = profile ?? actionContract ?? G1_PROFILE;
  // Engines arrive either as a registry or as a plain `{ name: infer }` map (tests, embedders);
  // wrap once so every lookup below goes through the same meta table.
  const registry = asEngineRegistry(engines);
  const { ACTION_TABLE, TOKENS, actionOrStop, legacyAction } = contract;
  const stop = contract.roles.stop;
  const pipeline = new DecisionPipeline({
    decide: (body) => runDecide(body ?? {}),
    now,
    logger,
  });
  const activeSessions = new Set();

  function mergeCfg(overrides = {}) {
    return { ...defaultDecisionCfg(config), ...overrides };
  }

  /**
   * Engine-specific decision knobs. The deliberate engine gets a normal token budget and its own
   * model and deadline; it never inherits the one-token primary's 6 s reflex timeout.
   */
  const prices = pricesFrom(config);

  function engineCfg(cfg, engine) {
    // A candidate id (engine@model, or a plain engine registered as a candidate) carries its own model and deadline.
    const candidate = brain?.candidates?.byId?.get(engine);
    if (candidate) return candidateCfg(cfg, candidate, config);
    if (!engine || engine === cfg.engine) return cfg;
    if (engine === (config.routerEscalateTo ?? "react") || engine === "react") {
      return {
        ...cfg,
        engine,
        primaryModel: config.reactModel ?? cfg.fallbackModel ?? cfg.primaryModel,
        fallbackModel: "",
        primaryTimeoutMs: config.reactTimeoutMs ?? 8000,
        maxOutputTokens: config.reactMaxOutputTokens ?? 160,
        reactMaxIterations: config.reactMaxIterations ?? 2,
        race: false,
      };
    }
    return { ...cfg, engine };
  }

  /**
   * The reactive path (`/v1/act`, the pipeline) has no HTTP session, but the robot still moves on it,
   * so it gets one process-wide episodic session of its own: every act is judged and remembered there,
   * and every fresh reactive decision recalls the lessons that session's last situation matches.
   */
  // One id per process: each run is its own episode in the store (an earlier run's steps are not this
  // run's duplicates, and old runs age out under G1_EPISODES_MAX_SESSIONS). "act" stays the alias.
  const actSession = { id: `act-${Date.now().toString(36)}`, task: DEFAULT_INSTRUCTION, seq: 0 };

  async function runDecide(body = {}) {
    const instruction = body.instruction ?? DEFAULT_INSTRUCTION;
    let experience = body.experience ?? null;
    let directive = body.directive ?? null;
    if (brain && body.experience === undefined) {
      // The reactive path: no routed plan, so recall and the previous act's `next` line come from the
      // process-wide act session instead.
      experience = brain.experienceFor(actSession, instruction).text;
      if (body.directive === undefined) directive = brain.takeDirective(actSession)?.text ?? null;
    }
    // cfg.engine may name a registered candidate id (engine@model): the candidate then supplies the
    // engine, model, deadline and provider, so stateless callers can address any arm the router knows.
    const requested = body.cfg?.engine;
    const candidate = typeof requested === "string" ? brain?.candidates?.byId?.get(requested) ?? null : null;
    const cfg = candidate ? candidateCfg(mergeCfg({ ...body.cfg, engine: candidate.engine }), candidate, config) : mergeCfg(body.cfg);
    const decided = await decide({
      camera,
      engines: registry,
      config,
      cfg,
      instruction,
      advice: body.advice ?? null,
      experience,
      context: body.context ?? null,
      directive,
      profile: contract,
      now,
    });
    if (candidate) decided.result.candidate = candidate.id;
    return decided;
  }

  /** Judge and remember one reactive act. Never on the actuator path: it runs after the executor answered. */
  function rememberAct({ instruction, result, executed, frame, frameAt = null }) {
    if (!brain) return null;
    actSession.task = instruction;
    actSession.seq += 1;
    try {
      const learned = brain.record(actSession, {
        seq: actSession.seq,
        result,
        executed,
        frame,
        frameAt,
        engine: result.engine ?? null,
        taskKey: taskKey(instruction),
        routed: false,
      });
      return learned?.judge ? { score: learned.judge.score, outcome: learned.judge.outcome, alternative: learned.judge.alternative } : null;
    } catch (error) {
      logger.warn?.({ event: "brain_record_failed", path: "act", error: error?.message ?? String(error) });
      return null;
    }
  }

  /**
   * Request-level knobs that may only tighten the service's safety settings, never loosen them:
   *   fresh_max_s   caps how old a pipeline decision may be; the config value is the ceiling
   *   use_lidar     may require the lidar veto, may not switch it off unless the deployment allows it
   *                 (contact work with the lidar producer stopped) — with lidar off, S/Q/E have no
   *                 obstacle sensing at all
   */
  const limits = {
    freshMaxS(body = {}) {
      const ceiling = Number(config.pipelineFreshMaxS) > 0 ? Number(config.pipelineFreshMaxS) : 1.0;
      const requested = Number(body.fresh_max_s);
      return Math.min(requested > 0 ? requested : ceiling, ceiling);
    },
    lidar(body = {}) {
      if (body.use_lidar === false && config.useLidar && !config.allowRequestLidarOff) return { useLidar: true, ignored: true };
      return { useLidar: body.use_lidar ?? config.useLidar, ignored: false };
    },
  };

  /**
   * The depth the gate should see when the decision was made on an older frame (a pipeline decision,
   * or the frame that preceded a deliberation). A depth-only capture costs the socket round-trip and
   * no JPEG encode; if it fails the decision frame is used and the response says so.
   */
  async function liveGateFrame(decisionFrame, cfg) {
    if (config.gateLiveDepth !== true) return { frame: decisionFrame, source: "decision" };
    const started = now();
    try {
      const fresh = await camera.capture({ ...cfg, needDepth: true, includeDepth: true, depthOnly: true });
      if (!fresh?.depth) return { frame: decisionFrame, source: "decision", elapsedS: seconds(now() - started) };
      return { frame: { ...(decisionFrame ?? {}), depth: fresh.depth, liveDepthAt: fresh.capturedAt ?? null }, source: "live", elapsedS: seconds(now() - started) };
    } catch (error) {
      logger.warn?.({ event: "gate_live_depth_failed", error: error?.message ?? String(error) });
      return { frame: decisionFrame, source: "decision", elapsedS: seconds(now() - started), error: error?.message ?? String(error) };
    }
  }

  /** Read-only look-again for the deliberate engine: a fresh frame plus the evidence the gate would use. */
  async function observeEvidence(cfg) {
    const frame = await camera.capture({ ...cfg, needDepth: true });
    const evidence = await executor.evidence({ frame, useLidar: config.useLidar });
    return { frame, depth: frame.depth ?? null, lidar: evidence.lidar, state: evidence.status };
  }

  /** Decision only. No gate, no actuator, no side effects. */
  async function decideOnly(body = {}) {
    const { result } = await runDecide(body);
    if (!Number.isFinite(result.costUsd)) result.costUsd = costUsd(result.model, result.usage ?? null, prices);
    return result;
  }

  /**
   * Stateless decide -> gate -> execute.
   * `pipeline: true` takes the freshest background decision instead of making a new one, which is
   * what turns a ~1.4 s request into a ~30 ms one. `force_token` skips the model entirely for
   * scripted approach phases; the gate still applies in full.
   */
  async function act(body = {}) {
    const started = now();
    const cfg = mergeCfg(body.cfg);
    let decided;
    let source = "fresh";

    if (body.force_token || body.force_action) {
      const token = actionOrStop(body.force_token ?? body.force_action).token;
      const frame = await camera.capture(cfg);
      source = "forced";
      decided = {
        result: {
          ok: true,
          token,
          record: ACTION_TABLE[token],
          action: legacyAction(token, body.duration_ms ?? null),
          durationMs: body.duration_ms ?? ACTION_TABLE[token].defaultMs,
          model: "forced",
          engine: "forced",
          fallbackUsed: false,
          attempts: [],
          timing: { captureS: null, modelReadyS: null, actionReadyS: null, totalS: null },
          errorType: null,
          error: null,
          tokenSource: "forced",
        },
        frame,
        cfg,
        at: now(),
      };
    } else if (body.pipeline) {
      if (!pipeline.running) pipeline.set({ on: true, concurrency: config.pipelineConcurrency, instruction: body.instruction, cfg: body.cfg });
      const fresh = pipeline.take(limits.freshMaxS(body));
      if (fresh) {
        decided = fresh;
        source = "pipeline";
      } else {
        decided = (await pipeline.inflight) ?? (await runDecide(body));
        decided.at ??= now();
        source = "pipeline_inflight";
      }
    } else {
      decided = await runDecide(body);
      decided.at = now();
    }

    const { result, frame } = decided;
    const decidedAt = now();
    const lidar = limits.lidar(body);
    const out = {
      ok: result.ok,
      harness: config.agentHarness ?? "prime-agent-rpc",
      harness_version: config.harnessVersion ?? config.primeAgentVersion ?? null,
      source,
      ...(lidar.ignored ? { lidar_override_ignored: true } : {}),
      token: result.token,
      action: result.action,
      record: result.record,
      model_used: result.model,
      usage: result.usage ?? null,
      cost_usd: Number.isFinite(result.costUsd) ? result.costUsd : costUsd(result.model, result.usage ?? null, prices),
      engine: result.engine,
      fallback: result.fallbackUsed,
      attempts: result.attempts,
      token_source: result.tokenSource ?? null,
      model_timings: result.modelTimings ?? null,
      output_text: result.outputText ?? null,
      cfg,
      image: frame ? { w: frame.width, h: frame.height, bytes: frame.jpegBytes, ...(frame.timing ?? {}) } : null,
      error_type: result.errorType,
      error: result.error,
      timing: {
        capture_s: result.timing?.captureS ?? null,
        model_s: result.timing?.modelReadyS ?? null,
        decided_s: seconds(decidedAt - started),
        ...(source.startsWith("pipeline")
          ? { decision_age_s: seconds(decidedAt - decided.at), pipeline_decide_s: decided.decideS ?? null }
          : {}),
      },
    };
    // A pipeline decision was made for the pipeline's instruction, whatever this request said.
    const instruction = (source.startsWith("pipeline") ? pipeline.body?.instruction : body.instruction) ?? pipeline.body?.instruction ?? DEFAULT_INSTRUCTION;
    const frameAt = decided.capturedAt ?? decided.wallAt ?? null;
    if (!result.ok) {
      // Fail closed: a decision we could not make becomes a stop, not a silent no-op.
      if (body.exec_mode === "real" || body.execute === true) {
        out.stopped = await executor.stop().catch((error) => ({ ok: false, error: error.message }));
      }
      out.brain = brain ? { judge: rememberAct({ instruction, result, executed: null, frame, frameAt }), memory_size: brain.experience.stats().size } : null;
      out.timing.total_s = seconds(now() - started);
      return out;
    }

    const mode = body.exec_mode ?? (body.execute === true ? "real" : "none");
    const token = mode === "wiggle" ? (contract.roles.turnLeft ?? stop) : result.token;
    // A pipeline decision is up to fresh_max_s old: the gate must see the depth as it is now.
    const gateFrame = source.startsWith("pipeline") ? await liveGateFrame(frame, cfg) : { frame, source: "decision" };
    const executed = await executor.execute({
      token,
      durationMs: mode === "wiggle" ? (body.wiggle_ms ?? 400) : result.durationMs,
      frame: gateFrame.frame,
      execute: mode === "real" || mode === "wiggle",
      useLidar: lidar.useLidar,
      nearCenterM: body.near_center_m ?? null,
      nearSideM: body.near_side_m ?? null,
    });

    Object.assign(out, {
      exec_mode: mode,
      // wiggle executes the turn-left token whatever the model said; say so instead of reporting the model's token
      token,
      gate_frame: gateFrame.source,
      ...(gateFrame.elapsedS !== undefined ? { live_depth_s: gateFrame.elapsedS } : {}),
      gate: executed.gate,
      motion: executed.motion ?? null,
      executed: executed.executed,
      preview: executed.preview,
      dispatch: executed.dispatch,
      arm_held: executed.arm_held,
    });
    out.error_type = out.error_type ?? executed.error_type ?? null;
    out.error = out.error ?? executed.error ?? null;
    out.timing.gate_s = executed.timing.gate_s;
    out.timing.exec_s = executed.timing.exec_s;
    // Learn from what just happened on the reactive path too: a veto, a rejected dispatch or a clean
    // pulse is evidence about this situation whichever endpoint asked for it.
    out.brain = brain
      ? {
          judge: rememberAct({ instruction, result: { ...result, token }, executed, frame, frameAt }),
          experience: result.experienceUsed ?? null,
          directive: result.directiveUsed ?? null,
          next: result.next ?? null,
          memory_size: brain.experience.stats().size,
        }
      : null;
    out.timing.total_s = seconds(now() - started);
    return out;
  }

  /**
   * Session step: the same pipeline as act(), plus sequence tracking, the thinker's advice on the
   * way in and a review job on the way out.
   */
  /**
   * With a brain attached the step becomes: route -> recall -> decide -> gate -> (verify-and-escalate
   * inline if the reflex token was vetoed and there is time) -> judge -> remember. Without one it is
   * the original decide -> gate path. Either way the actuator is only ever reached through
   * executor.execute(), with the same double opt-in.
   */
  async function step(body = {}) {
    const started = now();
    let session;
    if (body.session_id) {
      session = sessions.get(body.session_id);
      if (!session) {
        return { ok: false, session_id: body.session_id, error_type: "unknown_session", error: "unknown control session" };
      }
    } else {
      session = sessions.open(body.instruction || DEFAULT_INSTRUCTION, { cfg: body.cfg });
    }

    if (activeSessions.has(session.id)) {
      return { ok: false, session_id: session.id, error_type: "session_busy", error: "control session already has a step in progress" };
    }
    activeSessions.add(session.id);
    try {
      if (body.instruction) session.task = body.instruction;
      return await runSessionStep(body, session, started);
    } finally {
      activeSessions.delete(session.id);
    }
  }

  async function runSessionStep(body, session, started) {
    const advice = sessions.takeAdvice(session);
    const seq = session.seq + 1;
    const baseCfg = mergeCfg(body.cfg ?? session.cfg);
    const lidar = limits.lidar(body);
    const executeOptions = {
      execute: body.execute === true || body.exec_mode === "real",
      useLidar: lidar.useLidar,
      nearCenterM: body.near_center_m ?? null,
      nearSideM: body.near_side_m ?? null,
    };

    // ---- plan: route the step and recall what memory knows about this task and situation
    const armHeld = executor.armState?.().held ?? null;
    const plan = brain
      ? brain.plan(session, {
          instruction: session.task,
          explicitEngine: body.cfg?.engine ?? session.cfg?.engine ?? null,
          observe: () => observeEvidence(baseCfg),
          armHeld,
        })
      : null;
    const cfg = plan ? engineCfg(baseCfg, plan.engine) : baseCfg;

    let decided = await runDecide({
      instruction: session.task,
      cfg,
      advice,
      experience: plan?.lessonsText || null,
      context: plan?.context ?? null,
      directive: plan?.directive?.text ?? null,
    });
    let { result, frame } = decided;

    // Fail closed: any inference failure resolves to the profile's stop and is reported as a safe stop.
    let token = result.ok ? result.token : stop;
    let executed = await executor.execute({ token, durationMs: result.ok ? result.durationMs : 0, frame, ...executeOptions });

    // ---- verify-and-escalate: the reflex answer was vetoed; ask the deliberate engine once, re-gate
    let inline = null;
    let reflex = null;
    const deadlineLeftMs = (config.brainStepBudgetMs ?? 10000) - (now() - started);
    if (
      brain &&
      result.ok &&
      executed.gate?.allowed === false &&
      brain.router.shouldEscalateInline({
        engine: result.engine,
        token,
        gate: executed.gate,
        deadlineLeftMs,
      })
    ) {
      const escalateTo = brain.router.escalateTo;
      reflex = { token, engine: result.engine, gate: executed.gate, depth: frame?.depth ?? null };
      // The deliberate call is bounded by what is left of the step budget, not only by its own timeout.
      const deliberateCfg = engineCfg(baseCfg, escalateTo);
      deliberateCfg.primaryTimeoutMs = Math.max(1000, Math.min(deliberateCfg.primaryTimeoutMs, Math.floor(deadlineLeftMs - 300)));
      const second = await runDecide({
        instruction: session.task,
        cfg: deliberateCfg,
        advice,
        experience: plan.lessonsText || null,
        directive: plan.directive?.text ?? null,
        context: {
          ...plan.context,
          observation: `Safety gate vetoed ${token}: ${(executed.gate.reasons ?? []).join("; ")}`,
          evidence: { depth: executed.gate.depth ?? frame?.depth ?? null, lidar: executed.gate.lidar ?? null, state: executed.gate.state ?? null },
        },
      });
      inline = {
        from: token,
        engine: escalateTo,
        ok: second.result.ok,
        token: second.result.ok ? second.result.token : null,
        thought: second.result.thought ?? null,
        error_type: second.result.errorType ?? null,
      };
      if (second.result.ok) {
        decided = second;
        ({ result, frame } = second);
        token = result.token;
        // The deliberation took seconds; the gate sees the depth as it is now, not as the frame was.
        const gateFrame = await liveGateFrame(frame, baseCfg);
        inline.gate_frame = gateFrame.source;
        executed = await executor.execute({ token, durationMs: result.durationMs, frame: gateFrame.frame, ...executeOptions });
      }
      logger.info?.({ event: "brain_inline_escalation", session_id: session.id, seq, from: reflex.token, to: inline.token, ok: inline.ok });
    }

    sessions.commit(session, { seq, token });

    // ---- judge and remember
    const learned = brain
      ? brain.record(session, {
          seq,
          result,
          executed,
          frame,
          engine: result.engine ?? plan.engine,
          candidate: inline?.ok ? brain.router.escalateTo : plan.engine,
          route: plan.route,
          inline,
          taskKey: plan.taskKey,
          reflex,
          frameAt: decided.capturedAt ?? null,
        })
      : null;

    let thinkerQueued = false;
    if (thinker && frame) {
      thinkerQueued = thinker.submit({
        sessionId: session.id,
        seq,
        instruction: session.task,
        token,
        action: ACTION_TABLE[token],
        beforeFrame: frame,
        feedback: {
          executed: executed.executed,
          preview: executed.preview,
          gate: executed.gate?.reasons ?? [],
          dispatch_ok: executed.dispatch?.ok ?? null,
          error_type: executed.error_type ?? null,
        },
        ...(plan
          ? {
              context: {
                engine: result.engine ?? plan.engine,
                route: plan.route.reason,
                thought: result.thought ?? null,
                recent: plan.recent || null,
                where: plan.where || null,
                history: plan.history || null,
                lessons: plan.lessonsText || null,
                inline: inline ? { from: inline.from, to: inline.token } : null,
              },
            }
          : {}),
        captureAfter: executed.executed,
      });
    }

    const deliberate = Boolean(result.thought) || result.engine === (config.routerEscalateTo ?? "react");
    return {
      ok: result.ok && (executed.executed || executed.preview),
      harness: config.agentHarness ?? "prime-agent-rpc",
      harness_version: config.harnessVersion ?? config.primeAgentVersion ?? null,
      session_id: session.id,
      ...(lidar.ignored ? { lidar_override_ignored: true } : {}),
      seq,
      token,
      action: legacyAction(token, result.durationMs ?? null),
      record: ACTION_TABLE[token],
      safe_stop: !result.ok && token === stop,
      token_source: result.ok ? (result.tokenSource ?? "model") : "fail_closed",
      model_used: result.ok ? result.model : null,
      usage: result.usage ?? null,
      cost_usd: result.ok ? (Number.isFinite(result.costUsd) ? result.costUsd : costUsd(result.model, result.usage ?? null, prices)) : null,
      engine: result.engine,
      reasoning: deliberate,
      model_output_tokens: registry.meta(result.engine)?.outputTokens ?? null,
      advice_used: advice,
      thinker_queued: thinkerQueued,
      attempts: result.attempts,
      output_text: result.outputText ?? null,
      model_timings: result.modelTimings ?? null,
      thought: result.thought ?? null,
      gate: executed.gate,
      motion: executed.motion ?? null,
      executed: executed.executed,
      preview: executed.preview,
      dispatch: executed.dispatch,
      arm_held: executed.arm_held,
      brain: plan
        ? {
            engine: result.engine ?? plan.engine,
            candidate: learned?.candidate ?? plan.engine,
            route: plan.route.reason,
            router_mode: brain.router.mode ?? "rule",
            suggested: plan.route.suggested === true,
            escalated: plan.route.escalated || Boolean(inline?.ok),
            triggers: plan.route.triggers,
            dimension: plan.route.dimension,
            inline,
            thought: result.thought ?? null,
            trace: result.trace ?? null,
            lessons_used: plan.recalled.length,
            experience: plan.lessonsText || null,
            directive: plan.directive ? { text: plan.directive.text, from_seq: plan.directive.fromSeq, engine: plan.directive.engine } : null,
            next: result.next ?? null,
            reflection: learned?.reflection ?? null,
            judge: learned?.judge ? { score: learned.judge.score, outcome: learned.judge.outcome, alternative: learned.judge.alternative } : null,
            signals: plan.signals,
            where: plan.where,
            history: plan.history ?? null,
            episodes_used: plan.pastEpisodes ?? 0,
            spatial: learned?.spatial ?? null,
            memory_size: brain.experience.stats().size,
          }
        : null,
      timing: {
        capture_s: result.timing?.captureS ?? null,
        token_ready_s: result.timing?.modelReadyS ?? null,
        action_ready_s: result.timing?.actionReadyS ?? null,
        gate_s: executed.timing.gate_s,
        exec_s: executed.timing.exec_s,
        total_s: seconds(now() - started),
      },
      error_type: result.errorType ?? executed.error_type ?? null,
      error: result.error ?? executed.error ?? null,
    };
  }

  async function sense(body = {}) {
    const cfg = mergeCfg(body.cfg);
    const frame = await camera.capture({ ...cfg, needDepth: true });
    const [lidar, status] = await Promise.all([
      executor.obstacles ? executor.obstacles.read() : lidarGate({ snapshot: config.lidarSnapshot }),
      executor.status().catch((error) => ({ ok: false, ready: false, reasons: [error.message] })),
    ]);
    return {
      ok: true,
      depth: frame.depth,
      w: frame.width,
      h: frame.height,
      bytes: frame.jpegBytes,
      timing: frame.timing,
      image_b64: body.image ? frame.imageB64 : undefined,
      lidar,
      state: { ready: status?.ready ?? false, fsm_id: status?.fsm_id ?? null, soc: status?.soc ?? null, reasons: status?.reasons ?? [] },
    };
  }

  /**
   * Arm primitives. A U/P/R token goes through the normal executor (state gate + holding interlock);
   * any other firmware action id goes through the raw path, which applies the same two checks.
   *
   * The same double opt-in as every other motion applies: the service needs G1_EXECUTE_ENABLED and
   * the request needs `execute: true` (or `exec_mode: "real"`). Without the request flag the token
   * path returns a gated preview and the raw path is refused, so a stray POST cannot raise the arms.
   */
  async function arm(body = {}) {
    const shouldExecute = body.execute === true || body.exec_mode === "real";
    const record = body.token ? actionOrStop(body.token) : null;
    if (record && record.kind === "arm_primitive") {
      const out = await executor.execute({ token: record.token, frame: null, execute: shouldExecute, useLidar: false });
      // The executor reports outcomes, not success; a preview or an execution without an error is ok.
      return { ok: out.error_type == null, ...out };
    }
    if (!Number.isInteger(body.action_id)) {
      const armTokens = contract.arms ? ` or an arm token (${[...contract.arms.holding, contract.arms.release].join("/")})` : "";
      return { ok: false, error_type: "request_error", error: `action_id${armTokens} is required` };
    }
    if (!shouldExecute) {
      return {
        ok: false,
        error_type: "execute_required",
        error: "raw arm actions need execute:true (or exec_mode:\"real\") on the request",
        action_id: body.action_id,
        executed: false,
        preview: true,
      };
    }
    return executor.armRaw(body.action_id, { force: body.force === true, holding: body.holding ?? null });
  }

  function capabilities() {
    return {
      service: SERVICE,
      version: VERSION,
      harness: config.agentHarness ?? "prime-agent-rpc",
      harness_version: config.harnessVersion ?? config.primeAgentVersion ?? null,
      tokens: [...TOKENS],
      token_map: Object.fromEntries(
        Object.entries(ACTION_TABLE).map(([token, record]) => [
          token,
          { name: record.name, kind: record.kind, vx: record.vx ?? null, vy: record.vy ?? null, yaw: record.yaw ?? null, duration_s: record.durationS ?? null, action_id: record.actionId ?? null },
        ]),
      ),
      // One block per registered engine, each from its own registry meta: an engine that is not
      // registered (cli without G1_CLI_ENGINES) does not appear at all.
      engines: registry.describe(config),
      brain: brain
        ? {
            enabled: true,
            router: { cheap_chain: brain.router.cheapChain, escalate_to: brain.router.escalateTo },
            memory: { lessons: brain.experience.stats().size, path: brain.experience.stats().path },
            judge: ["heuristic", "reviewer", "external"],
            actuator_attached: false,
          }
        : { enabled: false },
      actuator: { kind: config.actuator, execute_enabled: config.executeEnabled, serialized: executor.actuatorInfo?.().serialized ?? true, latest_wins: executor.actuatorInfo?.().latestWins ?? false, lowcmd: false },
      gate: {
        use_lidar: config.useLidar,
        request_may_disable_lidar: Boolean(config.allowRequestLidarOff),
        live_depth_at_execution: config.gateLiveDepth === true,
        pipeline_fresh_max_s: config.pipelineFreshMaxS ?? null,
        depth_near_center_m: config.depthNearCenterM,
        depth_near_side_m: config.depthNearSideM,
        depth_floor_center_m: DEPTH_FLOOR_CENTER_M,
        depth_floor_side_m: DEPTH_FLOOR_SIDE_M,
        lidar: { snapshot: config.lidarSnapshot, max_age_s: LIDAR_DEFAULTS.maxAgeS, near_center_m: LIDAR_DEFAULTS.nearCenterM, near_side_m: LIDAR_DEFAULTS.nearSideM },
      },
      manipulation: contract.manipulation ?? {
        box_lift: contract.BOX_LIFT,
        available_arm_primitives: Object.values(ACTION_TABLE).filter(record => record.kind === "arm_primitive").map(record => record.primitive),
        gripper_available: false,
      },
      thinker: { asynchronous: thinker !== null, blocks_main: false, actuator_attached: false },
      pipeline: pipeline.status(),
    };
  }

  return {
    decideOnly,
    act,
    step,
    sense,
    arm,
    capabilities,
    limits,
    pipeline: {
      set: (body) => pipeline.set(body),
      status: () => pipeline.status(),
      stop: () => pipeline.stop(),
    },
    stop: async () => {
      pipeline.stop();
      return executor.stop();
    },
    status: () => executor.status(),
    /**
     * Reviewer corrections steer the next step only when they are fresh (the session protocol is
     * strict about that). Fresh or stale, a well-formed verdict is still a judgement of the step it
     * reviewed, so the brain learns from both.
     */
    acceptCorrection: (correction) => {
      const result = sessions.acceptCorrection(correction);
      if (brain && (result.accepted || result.reason === "stale_correction")) {
        const session = sessions.get(correction.sessionId);
        const learned = session ? brain.learnFromReview(session, correction) : null;
        return { ...result, learned: learned ? { score: learned.score, outcome: learned.outcome } : null };
      }
      return result;
    },
    /**
     * The asynchronous judge's outcome assessment of a past step. It is always late by construction
     * (it needs the after frame), so it never steers; it teaches memory and the router.
     */
    acceptJudgement: (body) => {
      if (!brain) return { accepted: false, reason: "brain_disabled" };
      const error = externalJudgementError(body);
      if (error) return { accepted: false, reason: "invalid_judgement", error };
      const session = sessions.get(body.sessionId);
      if (!session) return { accepted: false, reason: "unknown_session" };
      const learned = brain.learnFromJudgement(session, body);
      if (!learned) return { accepted: false, reason: "unknown_seq", sessionId: body.sessionId, basedOnSeq: body.basedOnSeq };
      return {
        accepted: true,
        sessionId: body.sessionId,
        basedOnSeq: body.basedOnSeq,
        learned: learned.skipped ? null : { score: learned.score, outcome: learned.outcome, lesson: learned.lesson?.lesson ?? null },
      };
    },
    brain: brain
      ? {
          state: () => brain.state(),
          recall: (query) => brain.recall(query),
          episodes: (query) => brain.episodesFor(query),
          reset: (options) => brain.reset(options),
          learnLesson: (body, options) => brain.learnLesson(body, options),
          session: (sessionId) => {
            if (sessionId === "act" || sessionId === actSession.id) return brain.sessionState(actSession);
            const session = sessions.get(sessionId);
            return session ? brain.sessionState(session) : null;
          },
        }
      : null,
  };
}
