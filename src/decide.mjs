import { performance } from "node:perf_hooks";

import { asEngineRegistry, UnknownEngineError } from "./engines.mjs";
import { G1_PROFILE } from "./profiles/g1.mjs";
import { sanitisePromptText } from "./sanitise.mjs";

export const DEFAULT_INSTRUCTION =
  "Move toward the nearest passable doorway or opening while avoiding collision. " +
  "From the current forward camera frame, choose exactly one immediate action. " +
  "Choose STOP when the safe path is unclear.";

function seconds(ms) {
  return Number((ms / 1000).toFixed(6));
}

function errorInfo(error, fallbackType = "inference_error") {
  return {
    errorType: error?.errorType ?? error?.kind ?? fallbackType,
    error: error instanceof Error ? error.message : String(error),
    status: Number.isInteger(error?.status) ? error.status : null,
    ...(error?.usage !== undefined ? { usage: error.usage } : {}),
    ...(error?.outputText !== undefined ? { outputText: error.outputText } : {}),
  };
}

function mayFallback(error) {
  if (error?.retryable === false) return false;
  if (error?.status === 401 || error?.status === 403) return false;
  return error?.errorType !== "config_error" && error?.kind !== "config_error";
}

async function withDeadline(run, timeoutMs, label) {
  const controller = new AbortController();
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      const error = new Error(`${label} deadline exceeded ${timeoutMs} ms`);
      error.errorType = "deadline";
      error.retryable = true;
      reject(error);
    }, timeoutMs);
  });
  try {
    return await Promise.race([run({ signal: controller.signal, deadlineMs: timeoutMs }), deadline]);
  } finally {
    clearTimeout(timer);
    controller.abort();
  }
}

/** Canonical decision shape every engine and every mode funnels into. */
function decision({ token, durationMs = null, model, engine, fallbackUsed, attempts, timing, extra = {}, actionContract }) {
  const record = actionContract.actionOrStop(token);
  return {
    ok: true,
    token: record.token,
    record,
    action: actionContract.legacyAction(record.token, durationMs),
    durationMs: Number.isInteger(durationMs) ? durationMs : record.defaultMs,
    model,
    engine,
    fallbackUsed,
    attempts,
    timing,
    errorType: null,
    error: null,
    ...extra,
  };
}

function failure({ attempts, timing, details, engine }) {
  return {
    ok: false,
    token: null,
    record: null,
    action: null,
    durationMs: null,
    model: null,
    engine,
    fallbackUsed: attempts.length > 1,
    attempts,
    timing,
    ...details,
  };
}

/**
 * Sequential primary -> fallback chain.
 * Cheaper in tokens than racing and it is what you want when the primary is reliable.
 */
async function chain({ engines, frame, cfg, instruction, advice, experience, context, directive, now }) {
  const started = now();
  const attempts = [];
  const models = [
    [cfg.primaryModel, cfg.primaryTimeoutMs],
    [cfg.fallbackModel, cfg.fallbackTimeoutMs],
  ].filter(([model]) => Boolean(model));
  let lastError = null;

  for (let index = 0; index < models.length; index += 1) {
    const [model, timeoutMs] = models[index];
    const attemptStarted = now();
    const engine = engines.get(cfg.engine);
    try {
      const outcome = await withDeadline(
        (options) => engine(model, frame, { ...options, instruction, advice, cfg, experience, context, directive }),
        timeoutMs,
        model,
      );
      const ended = now();
      const readyS = outcome.tokenReadyS ?? outcome.actionReadyS ?? seconds(ended - attemptStarted);
      attempts.push({
        model,
        engine: engine.engine ?? cfg.engine,
        ok: true,
        readyS,
        totalS: outcome.totalS ?? seconds(ended - attemptStarted),
        finishReason: outcome.finishReason ?? null,
        responseStatus: outcome.responseStatus ?? null,
        errorType: null,
        error: null,
        usage: outcome.usage ?? null,
      });
      return {
        outcome,
        model,
        engine: engine.engine ?? cfg.engine,
        fallbackUsed: index > 0,
        attempts,
        readyS,
        elapsedS: seconds(ended - started),
      };
    } catch (error) {
      const details = errorInfo(error);
      attempts.push({
        model,
        engine: engine.engine ?? cfg.engine,
        ok: false,
        readyS: null,
        totalS: seconds(now() - attemptStarted),
        finishReason: error?.finishReason ?? null,
        responseStatus: details.status,
        ...details,
      });
      lastError = error;
      if (index === 0 && mayFallback(error)) continue;
      break;
    }
  }
  return { outcome: null, attempts, error: lastError, elapsedS: seconds(now() - started) };
}

/**
 * Race mode: query primary and fallback concurrently and take
 * the first schema-valid answer. Costs one extra call per decision, wins when provider latency is
 * long-tailed rather than uniformly slow.
 */
async function race({ engines, frame, cfg, instruction, advice, experience, context, directive, now }) {
  const started = now();
  const models = [cfg.primaryModel, cfg.fallbackModel].filter(Boolean);
  const controllers = models.map(() => new AbortController());
  const attempts = [];
  const engine = engines.get(cfg.engine);

  const runs = models.map((model, index) =>
    engine(model, frame, {
      signal: controllers[index].signal,
      instruction,
      advice,
      deadlineMs: cfg.primaryTimeoutMs,
      cfg,
      experience,
      context,
      directive,
    }).then(
      (outcome) => {
        attempts.push({
          model,
          engine: engine.engine ?? cfg.engine,
          ok: true,
          readyS: outcome.tokenReadyS ?? outcome.actionReadyS ?? null,
          finishReason: outcome.finishReason ?? null,
          responseStatus: outcome.responseStatus ?? null,
          errorType: null,
          error: null,
        });
        return { model, outcome };
      },
      (error) => {
        const details = errorInfo(error);
        attempts.push({ model, engine: engine.engine ?? cfg.engine, ok: false, readyS: null, ...details });
        throw error;
      },
    ),
  );

  let winner = null;
  try {
    winner = await Promise.any(runs);
  } catch {
    winner = null;
  }
  for (const controller of controllers) {
    try {
      controller.abort();
    } catch {
      /* already aborted */
    }
  }
  const elapsedS = seconds(now() - started);
  if (!winner) {
    return { outcome: null, attempts, error: Object.assign(new Error("all models failed"), { errorType: "inference_error" }), elapsedS };
  }
  return {
    outcome: winner.outcome,
    model: winner.model,
    engine: engine.engine ?? cfg.engine,
    fallbackUsed: winner.model !== cfg.primaryModel,
    attempts,
    readyS: winner.outcome.tokenReadyS ?? winner.outcome.actionReadyS ?? null,
    elapsedS,
  };
}

/**
 * capture -> infer -> canonical decision.
 * Returns `{ result, frame, cfg }` so the executor can gate on the very frame the model saw.
 *
 * `experience` is the rendered lesson block from memory (reflex engines append it to the prompt,
 * size-capped); `context` is the deliberate engine's read-only context (recent steps, evidence,
 * observe/recall callbacks); `directive` is the previous step's `next` line (what to watch for now).
 * All are optional and ignored by engines that do not use them.
 */
export async function decide({
  camera,
  engines,
  config,
  cfg,
  instruction = DEFAULT_INSTRUCTION,
  advice = null,
  experience = null,
  context = null,
  directive = null,
  profile = null,
  actionContract = null,
  now = () => performance.now(),
}) {
  const contract = profile ?? actionContract ?? G1_PROFILE;
  const started = now();

  // An engine nobody registered is a bad request, answered before the camera is touched: silently
  // running a different engine than the caller asked for is worse than refusing.
  const registry = asEngineRegistry(engines);
  if (!registry.has(cfg.engine)) {
    return {
      result: failure({
        attempts: [],
        timing: { captureS: null, modelReadyS: null, actionReadyS: null, totalS: 0 },
        details: { errorType: "request_error", error: new UnknownEngineError(cfg.engine, registry.names()).message, status: null },
        engine: cfg.engine,
      }),
      frame: null,
      cfg,
    };
  }

  // CLI engines (claude -p / codex exec), the prime harness and the scripted engine bring their own
  // credentials or need none, so a missing gateway key only blocks the gateway engines.
  if (config.providerConfigured === false && registry.meta(cfg.engine).needsProvider && !cfg.apiKey) {
    return {
      result: failure({
        attempts: [],
        timing: { captureS: null, modelReadyS: null, actionReadyS: null, totalS: 0 },
        details: { errorType: "config_error", error: "REA_API_KEY is not set", status: null },
        engine: cfg.engine,
      }),
      frame: null,
      cfg,
    };
  }

  let frame;
  try {
    frame = await camera.capture(cfg);
  } catch (error) {
    const ended = now();
    return {
      result: failure({
        attempts: [],
        timing: { captureS: seconds(ended - started), modelReadyS: null, actionReadyS: null, totalS: seconds(ended - started) },
        details: errorInfo(error, "capture_error"),
        engine: cfg.engine,
      }),
      frame: null,
      cfg,
    };
  }

  const capturedAt = now();
  const capturedWallAt = Date.now();
  const run = cfg.race ? race : chain;
  const attempt = await run({ engines: registry, frame, cfg, instruction, advice, experience, context, directive, now });
  const ended = now();

  if (!attempt.outcome) {
    return {
      result: failure({
        attempts: attempt.attempts,
        timing: {
          captureS: seconds(capturedAt - started),
          modelReadyS: null,
          actionReadyS: null,
          totalS: seconds(ended - started),
        },
        details: errorInfo(attempt.error),
        engine: cfg.engine,
      }),
      frame,
      cfg,
      capturedAt: capturedWallAt,
    };
  }

  const readyS = attempt.readyS ?? seconds(ended - capturedAt);
  return {
    result: decision({
      actionContract: contract,
      token: attempt.outcome.token,
      durationMs: attempt.outcome.durationMs ?? null,
      model: attempt.model,
      engine: attempt.engine,
      fallbackUsed: attempt.fallbackUsed,
      attempts: attempt.attempts,
      timing: {
        captureS: seconds(capturedAt - started),
        modelReadyS: readyS,
        actionReadyS: seconds(capturedAt - started) + readyS,
        totalS: seconds(ended - started),
      },
      extra: {
        usage: attempt.outcome.usage ?? null,
        costUsd: Number.isFinite(attempt.outcome.costUsd) ? attempt.outcome.costUsd : null,
        modelTimings: attempt.outcome.timings ?? null,
        outputText: attempt.outcome.outputText ?? null,
        tokenSource: attempt.outcome.source ?? "model",
        adviceUsed: advice,
        experienceUsed: experience || null,
        thought: attempt.outcome.thought ?? null,
        trace: attempt.outcome.trace ?? null,
        next: sanitisePromptText(attempt.outcome.next, 120).text,
        nextRejected: attempt.outcome.nextRejected ?? (typeof attempt.outcome.next === "string" && attempt.outcome.next.trim() ? sanitisePromptText(attempt.outcome.next, 120).error : null),
        // The `min` prompt variant is the latency floor and carries no plan; only report what was rendered.
        directiveUsed: directive && cfg.promptVariant !== "min" ? directive : null,
      },
    }),
    frame,
    cfg,
    capturedAt: capturedWallAt,
  };
}
