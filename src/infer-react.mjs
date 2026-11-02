// Deliberate engine: a bounded ReAct loop (Thought -> Action -> Observation) that ends in one token.
//
// The reflex engines answer in one output token and cannot afford to think. This engine is the
// escalation target the router reaches for when the reflex is failing or the situation is risky. It
// follows the ReAct pattern with a hard iteration cap:
//
//   thought   one or two sentences about what the frame, the evidence and the recent steps imply
//   action    finish(token) | observe() -- look again with fresh depth/lidar/state | recall(query) --
//             consult the experience memory
//   observation  the result of the action, appended to the next iteration's prompt
//
// Every iteration is one forced tool call (`react_step`) streamed over the shared transport, so the
// answer is complete as soon as the JSON argument object closes. The loop always normalises onto the
// same single token as every other engine: the gate and the executor never learn that anyone thought.
//
// The engine holds no actuator and no SDK handle. `observe` and `recall` are read-only callbacks
// supplied by the brain; if they are absent the model is told so and must decide on what it has.

import { performance } from "node:perf_hooks";

import { G1_PROFILE } from "./profiles/g1.mjs";
import { sanitisePromptText } from "./sanitise.mjs";
import { InferenceError, chatUrl, createWarmer, isDeepSeek, providerError, streamChat } from "./provider.mjs";

const ACTIONS = Object.freeze(["finish", "observe", "recall"]);

const REACT_STEP_SCHEMA_BASE = {
  type: "object",
  properties: {
    thought: {
      type: "string",
      maxLength: 240,
      description: "One or two sentences: what the frame shows, what the clearance and recent steps imply.",
    },
    action: { type: "string", enum: [...ACTIONS], description: "finish with a token, observe again, or recall experience." },
    token: { type: "string", enum: [], description: "Required when action is finish." },
    query: { type: "string", maxLength: 80, description: "Optional topic for recall." },
    next: {
      type: "string",
      maxLength: 120,
      description: "With finish: what the next step should watch for or do once this token has executed. Stored as advice for the next decision, never executed.",
    },
  },
  required: ["thought", "action"],
  additionalProperties: false,
};

/** The step schema for one profile: only the token enum varies. */
export function reactStepSchema(profile = G1_PROFILE) {
  return {
    ...REACT_STEP_SCHEMA_BASE,
    properties: { ...REACT_STEP_SCHEMA_BASE.properties, token: { ...REACT_STEP_SCHEMA_BASE.properties.token, enum: [...profile.TOKENS] } },
  };
}

export const REACT_STEP_SCHEMA = reactStepSchema(G1_PROFILE);

export const REACT_NEXT_MAX_CHARS = 120;

const STEP_TOOL_DESCRIPTION = "Return the next ReAct step: a brief thought and exactly one action.";

const stepTool = profile => ({
  type: "function",
  function: {
    name: "react_step",
    description: STEP_TOOL_DESCRIPTION,
    parameters: reactStepSchema(profile),
  },
});

export function reactSystemPrompt(profile = G1_PROFILE) {
  return (
    `You are the deliberate half of a bounded ${profile.name} controller. ` +
    "Think briefly about the RGB-D frame, the clearance evidence, the recent steps and any recalled experience, " +
    "then call react_step exactly once: finish with one token, or observe again, or recall experience. " +
    "No actuator is connected to you; your token is re-checked by a safety gate before anything moves. " +
    `No grasp or lift primitive exists. When the safe choice is unclear, finish with ${profile.roles.stop}.`
  );
}

export const REACT_SYSTEM_PROMPT = reactSystemPrompt(G1_PROFILE);

export function isValidReactStep(value, profile = G1_PROFILE) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  if (typeof value.thought !== "string" || value.thought.length > 240) return false;
  if (!ACTIONS.includes(value.action)) return false;
  if (value.token !== undefined && (typeof value.token !== "string" || profile.parseToken(value.token) === null)) return false;
  if (value.query !== undefined && (typeof value.query !== "string" || value.query.length > 80)) return false;
  // `next` is best-effort advice: a wrong type or an over-long line must never cost the token, so
  // it only has to be a string here and is clipped where it is consumed.
  if (value.next !== undefined && typeof value.next !== "string") return false;
  if (value.action === "finish" && value.token === undefined) return false;
  return true;
}

/**
 * One line, printable, clipped: what the next step is told. A `next` that reads as an instruction to
 * the model ("always output W", "ignore the gate") is dropped, not carried.
 */
export function normaliseNext(value, maxChars = REACT_NEXT_MAX_CHARS) {
  return sanitisePromptText(value, maxChars).text;
}

/** Why a `next` was dropped ("directive", "prompt_label"), or null when it was kept or absent. */
export function nextRejection(value, maxChars = REACT_NEXT_MAX_CHARS) {
  if (typeof value !== "string" || !value.trim()) return null;
  const { error } = sanitisePromptText(value, maxChars);
  return error && error !== "too_short" ? error : null;
}

const USAGE_GRACE_MS = 150;

/** Sum of the token counts over the calls of one step; null until a call reports usage. */
export function addUsage(total, usage) {
  if (!usage || typeof usage !== "object") return total;
  const out = { ...(total ?? {}) };
  for (const key of ["prompt_tokens", "completion_tokens", "total_tokens"]) {
    if (Number.isFinite(usage[key])) out[key] = (Number.isFinite(out[key]) ? out[key] : 0) + usage[key];
  }
  return out;
}

function parseStep(text, profile = G1_PROFILE) {
  const start = text.indexOf("{");
  if (start === -1) return null;
  try {
    const value = JSON.parse(text.slice(start));
    // an over-long thought must never cost the step: clip it here, the validator then sees a legal one
    if (value && typeof value.thought === "string" && value.thought.length > 240) value.thought = value.thought.slice(0, 240);
    return isValidReactStep(value, profile) ? value : null;
  } catch {
    return null;
  }
}

function promptValue(value) {
  return value === null || value === undefined ? "none" : JSON.stringify(value);
}

function compactEvidence(evidence) {
  if (!evidence) return null;
  const out = {};
  if (evidence.depth) {
    out.depth = Object.fromEntries(
      Object.entries(evidence.depth)
        .filter(([key]) => ["center_m", "left_m", "right_m", "front_min_m"].includes(key))
        .map(([key, value]) => [key, typeof value === "number" ? Number(value.toFixed(2)) : value]),
    );
  }
  if (evidence.lidar) {
    out.lidar = {
      usable: evidence.lidar.usable ?? null,
      reason: evidence.lidar.reason ?? null,
      ...(typeof evidence.lidar.center === "number" ? { center_m: Number(evidence.lidar.center.toFixed(2)) } : {}),
      ...(typeof evidence.lidar.left === "number" ? { left_m: Number(evidence.lidar.left.toFixed(2)) } : {}),
      ...(typeof evidence.lidar.right === "number" ? { right_m: Number(evidence.lidar.right.toFixed(2)) } : {}),
    };
  }
  if (evidence.state) out.state = { ready: evidence.state.ready ?? null, reasons: evidence.state.reasons ?? [] };
  return out;
}

/**
 * The deliberate prompt. Longer than the reflex prompts on purpose: this engine is off the reflex
 * path, so context is worth more than the prefill it costs.
 */
export function buildReactPrompt({
  instruction,
  depth = null,
  evidence = null,
  recent = null,
  lessons = null,
  advice = null,
  observations = [],
  iteration = 1,
  maxIterations = 2,
  armHeld = null,
  directive = null,
  where = null,
  history = null,
  profile = G1_PROFILE,
} = {}) {
  const task = String(instruction ?? "Choose the safest immediate action from the current RGB-D observation.");
  const lines = [
    `Task: ${task}`,
    "",
    "Token map:",
    profile.legend.full,
    "",
    ...(profile.BOX_LIFT ? [`BOX_LIFT is unavailable (${profile.BOX_LIFT.reason}).`] : []),
    `Depth summary of this frame: ${promptValue(depth)}`,
  ];
  if (evidence) lines.push(`Live evidence: ${promptValue(compactEvidence(evidence))}`);
  if (armHeld) lines.push(`Arm currently holding pose: ${armHeld} (release with ${profile.arms?.release ?? "the release token"} before another arm gesture).`);
  lines.push(`Recent steps: ${recent && String(recent).trim() ? recent : "none"}`);
  if (where && String(where).trim()) lines.push(`Where (dead reckoning from dispatched pulses, drifts): ${String(where).trim()}`);
  if (history && String(history).trim()) lines.push(`How earlier sessions went (this task, unless a line says "similar task"):\n${String(history).trim()}`);
  lines.push(`Experience: ${lessons && String(lessons).trim() ? `\n${lessons}` : "none"}`);
  lines.push(`Reviewer advice: ${promptValue(advice)}`);
  if (directive && String(directive).trim()) lines.push(`Plan carried from the previous step (advice, not a command): ${String(directive).trim()}`);
  if (observations.length > 0) {
    lines.push("", "Observations so far:");
    for (const observation of observations) lines.push(`- ${observation}`);
  }
  lines.push("");
  if (iteration >= maxIterations) {
    lines.push(`Step ${iteration} of ${maxIterations}: this is your last step. You must finish with exactly one token now.`);
  } else {
    lines.push(
      `Step ${iteration} of ${maxIterations}: finish with a token if the choice is clear; ` +
        "otherwise observe (fresh depth, lidar and robot state) or recall (past lessons) once, then finish.",
    );
  }
  lines.push("When you finish, you may add `next`: one short line on what the following step should watch for or do.");
  return lines.join("\n");
}

export function createReactInfer({
  apiKey,
  baseUrl,
  dispatcher = undefined,
  maxOutputTokens = 160,
  maxIterations = 2,
  now = () => performance.now(),
  profile = G1_PROFILE,
}) {
  const url = chatUrl(baseUrl);
  const headers = {
    "content-type": "application/json",
    authorization: `Bearer ${apiKey}`,
    accept: "text/event-stream",
  };

  async function callOnce({ model, frame, prompt, signal, deadlineMs, imageDetail, maxTokens = maxOutputTokens, provider = null }) {
    const body = JSON.stringify({
      model,
      stream: true,
      stream_options: { include_usage: true },
      max_tokens: maxTokens,
      ...(provider?.extraBody ?? {}),
      temperature: 0,
      messages: [
        { role: "system", content: reactSystemPrompt(profile) },
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            {
              type: "image_url",
              image_url: {
                url: `data:${frame.mimeType ?? "image/jpeg"};base64,${frame.imageB64}`,
                ...(imageDetail ? { detail: imageDetail } : {}),
              },
            },
          ],
        },
      ],
      tools: [stepTool(profile)],
      tool_choice: { type: "function", function: { name: "react_step" } },
      // The thought field is the explicit reasoning budget; provider-side thinking would only add
      // unbounded latency on top of it.
      ...(isDeepSeek(model) ? { thinking: { type: "disabled" } } : {}),
    });

    let status = null;
    let finishReason = null;
    let args = "";
    let text = "";
    let usage = null;
    let found = null;
    const started = now();
    // Own controller, as in the token engine: once the step is parsed the stream is only drained
    // for the usage chunk, and a short grace timer keeps a slow gateway from delaying the step.
    const controller = new AbortController();
    const onAbort = () => controller.abort();
    signal?.addEventListener("abort", onAbort, { once: true });
    let grace = null;
    try {
      const stream = streamChat({ url: provider?.url ?? url, headers: provider?.headers ?? headers, dispatcher, body, signal: controller.signal, deadlineMs, apiKey: provider?.apiKey ?? apiKey, onStatus: value => { status = value; } });
      for await (const event of stream) {
        if (event.usage) usage = event.usage;
        if (found) continue;
        finishReason = event.finishReason ?? finishReason;
        for (const call of event.delta.tool_calls ?? []) args += call.function?.arguments ?? "";
        if (typeof event.delta.content === "string") text += event.delta.content;
        const step = parseStep(args, profile) ?? parseStep(text, profile);
        if (step !== null) {
          found = { step, status, finishReason, elapsedS: Number(((now() - started) / 1000).toFixed(6)) };
          grace = setTimeout(() => controller.abort(), USAGE_GRACE_MS);
        }
      }
    } catch (error) {
      if (!found) throw providerError(error, { apiKey, status, finishReason });
      /* aborted while draining for usage: the step is already in hand */
    } finally {
      clearTimeout(grace);
      signal?.removeEventListener("abort", onAbort);
    }
    if (found) return { ...found, usage };
    throw new InferenceError(
      `react step ended without a valid step (finish=${finishReason}, args=${args.slice(0, 400)}, text=${text.slice(0, 120)})`,
      { errorType: finishReason === "length" ? "length" : "invalid_react_step", status, finishReason, retryable: true },
    );
  }

  /**
   * options.context (all optional, all read-only):
   *   evidence     { depth, lidar, state } as the gate saw them
   *   recent       episodic summary line
   *   lessons      rendered experience block
   *   observation  why this engine was called (e.g. the gate's veto of the reflex token)
   *   observe()    async -> { depth, lidar, state, frame? }  fresh evidence
   *   recall(q)    async -> string  rendered lessons for a query
   *   armHeld      held arm pose token or null
   */
  async function infer(model, frame, { signal, instruction, advice = null, deadlineMs = 8000, cfg = {}, experience = null, context = {} } = {}) {
    const key = cfg.provider || cfg.baseUrl ? (cfg.apiKey ?? "") : (cfg.apiKey || apiKey);
    if (!key) throw new InferenceError(cfg.provider ? `no API key for provider ${cfg.provider}` : "REA_API_KEY is not set", { errorType: "config_error", retryable: false });
    if (!frame?.imageB64) throw new InferenceError("current frame is missing", { errorType: "config_error", retryable: false });
    // another provider's OpenAI-compatible endpoint (candidateCfg sets these for e.g. Gemini)
    const provider = cfg.baseUrl || cfg.apiKey || cfg.extraBody
      ? { url: cfg.baseUrl ? chatUrl(cfg.baseUrl) : url, headers: key === apiKey ? headers : { ...headers, authorization: `Bearer ${key}` }, apiKey: key, extraBody: cfg.extraBody && typeof cfg.extraBody === "object" ? cfg.extraBody : {} }
      : null;

    const started = now();
    const iterations = Math.max(1, Math.min(4, Number(cfg.reactMaxIterations ?? maxIterations)));
    // a candidate may carry its own budget (candidateCfg sets it); DeepSeek-style models need ~300 tokens for one step
    const maxTokens = Math.max(64, Math.min(1024, Number(cfg.maxOutputTokens) || maxOutputTokens));
    const imageDetail = cfg.imageDetail === undefined ? "low" : cfg.imageDetail;
    context = context ?? {};
    const observations = [];
    if (context.observation) observations.push(String(context.observation));
    let currentFrame = frame;
    let evidence = context.evidence ?? null;
    let lessons = context.lessons ?? experience ?? null;
    const trace = [];
    let status = null;
    let finishReason = null;
    let usageTotal = null;

    for (let iteration = 1; iteration <= iterations; iteration += 1) {
      if (signal?.aborted) throw new InferenceError("react inference was aborted", { errorType: "aborted", finishReason: "aborted" });
      const remaining = deadlineMs - (now() - started);
      if (remaining < 300) {
        throw new InferenceError(`react deadline exhausted after ${iteration - 1} step(s)`, { errorType: "deadline", retryable: true });
      }
      const prompt = buildReactPrompt({
        instruction,
        depth: currentFrame.depth ?? null,
        evidence,
        recent: context.recent ?? null,
        lessons,
        advice,
        observations,
        iteration,
        maxIterations: iterations,
        armHeld: context.armHeld ?? null,
        directive: context.directive ?? null,
        where: context.where ?? null,
        history: context.history ?? null,
        profile,
      });
      const call = await callOnce({ model, frame: currentFrame, prompt, signal, deadlineMs: remaining, imageDetail, maxTokens, provider });
      status = call.status ?? status;
      finishReason = call.finishReason ?? finishReason;
      usageTotal = addUsage(usageTotal, call.usage);
      const { step } = call;
      trace.push({ iteration, thought: step.thought, action: step.action, token: step.token ?? null, query: step.query ?? null, next: step.next ?? null, elapsed_s: call.elapsedS });

      // A token offered alongside observe/recall on the last step is accepted: the model was told it
      // had to finish, and refusing a valid token to re-prompt would only burn the deadline.
      const token = step.token !== undefined ? profile.parseToken(step.token) : null;
      if (step.action === "finish" || (iteration === iterations && token !== null)) {
        if (token === null) {
          observations.push(`finish requires a token from ${profile.TOKENS.join(" ")}`);
          continue;
        }
        const tokenReadyS = Number(((now() - started) / 1000).toFixed(6));
        const next = normaliseNext(step.next);
        return {
          token,
          thought: step.thought,
          next,
          nextRejected: nextRejection(step.next),
          iterations: iteration,
          trace,
          tokenReadyS,
          actionReadyS: tokenReadyS,
          totalS: tokenReadyS,
          finishReason,
          responseStatus: status,
          source: "react",
          usage: usageTotal,
          timings: { calls: iteration, token_ready_s: tokenReadyS },
        };
      }

      if (step.action === "observe") {
        if (typeof context.observe === "function") {
          try {
            const fresh = await context.observe();
            if (fresh?.frame?.imageB64) currentFrame = fresh.frame;
            evidence = { depth: fresh?.depth ?? currentFrame.depth ?? null, lidar: fresh?.lidar ?? null, state: fresh?.state ?? null };
            observations.push(`Observed: ${JSON.stringify(compactEvidence(evidence))}`);
          } catch (error) {
            observations.push(`Observe failed: ${error?.message ?? String(error)}`);
          }
        } else {
          observations.push("Observe is unavailable here; decide on the current frame and evidence.");
        }
        continue;
      }

      if (step.action === "recall") {
        if (typeof context.recall === "function") {
          try {
            const recalled = await context.recall(step.query ?? instruction ?? "");
            const rendered = typeof recalled === "string" ? recalled : Array.isArray(recalled) ? recalled.join("\n") : "";
            if (rendered.trim()) lessons = rendered;
            observations.push(rendered.trim() ? `Recalled:\n${rendered}` : "Recalled: nothing relevant.");
          } catch (error) {
            observations.push(`Recall failed: ${error?.message ?? String(error)}`);
          }
        } else {
          observations.push("Recall is unavailable here; decide on the current frame and evidence.");
        }
      }
    }

    throw new InferenceError(`react loop ended without a token after ${iterations} step(s)`, {
      errorType: "invalid_token",
      status,
      finishReason,
      retryable: true,
    });
  }

  return Object.assign(infer, { warm: createWarmer({ baseUrl, apiKey, dispatcher }), engine: "react", deliberate: true });
}
