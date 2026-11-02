// Asynchronous reviewer inference.
//
// The reviewer runs on the shared streaming transport, so it returns the moment its tool call is
// complete instead of draining the stream.
//
// The reviewer is advisory by construction: it holds no SDK handle, no actuator, and its output is a
// suggestion the main loop may apply to the *next* decision. It can never move the robot.

import { G1_PROFILE } from "./profiles/g1.mjs";
import { JUDGE_OUTCOMES } from "./judge.mjs";
import { InferenceError, chatUrl, isDeepSeek, providerError, streamChat } from "./provider.mjs";

export const THINKER_VERDICTS = Object.freeze(["KEEP", "CORRECT", "STOP"]);
export const THINKER_TOKENS = G1_PROFILE.TOKENS;

const VERDICT_SET = new Set(THINKER_VERDICTS);

/** The exact key set of a correction on the wire. The session protocol rejects anything else. */
export const THINKER_CORRECTION_KEYS = Object.freeze(["brief", "confidence", "next_token", "verdict"]);

/**
 * The reviewer answers two questions in one tool call. The steering half (verdict, next_token, brief,
 * confidence) is the strict correction protocol the main loop may apply to the next step. The judging
 * half (outcome, lesson) is what the reviewer concluded about the step it just watched; it is never
 * applied to a command, it goes to memory as a lesson. Both are optional on the model's side so a
 * provider that only returns the four correction keys keeps working.
 */
const THINKER_REVIEW_SCHEMA_BASE = {
  type: "object",
  properties: {
    verdict: { type: "string", enum: [...THINKER_VERDICTS] },
    next_token: { type: "string", enum: [] },
    brief: { type: "string", minLength: 1, maxLength: 200 },
    confidence: { type: "number", minimum: 0, maximum: 1 },
    outcome: {
      type: "string",
      enum: [...JUDGE_OUTCOMES],
      description: "What the reviewed step achieved: progress, no_progress, unsafe, or unclear.",
    },
    lesson: {
      type: "string",
      maxLength: 200,
      description: "One sentence a future controller should remember about this situation; omit if nothing.",
    },
  },
  required: ["verdict", "next_token", "brief", "confidence"],
  additionalProperties: false,
};

/** The review schema for one profile: only the next_token enum varies. */
export function reviewSchema(profile = G1_PROFILE) {
  return {
    ...THINKER_REVIEW_SCHEMA_BASE,
    properties: { ...THINKER_REVIEW_SCHEMA_BASE.properties, next_token: { type: "string", enum: [...profile.TOKENS] } },
  };
}

export const THINKER_REVIEW_SCHEMA = reviewSchema(G1_PROFILE);

const reviewTool = profile => ({
  type: "function",
  function: {
    name: "report_review",
    description: "Return the asynchronous action review without a reasoning trace.",
    parameters: reviewSchema(profile),
  },
});

export function thinkerSystemPrompt(profile = G1_PROFILE) {
  return (
    `You are an asynchronous visual reviewer for a ${profile.name} high-level action controller. ` +
    "Review the selected token against the before/after observations and execution feedback. " +
    "You are advisory only: no robot SDK or actuator is connected to you. " +
    (profile.BOX_LIFT ? "This robot has no verified gripper or box-lift primitive. Never reinterpret an arm gesture as grasping or lifting. " : "") +
    "Reason privately, never reveal chain-of-thought, and return only a short observable brief. " +
    "Call report_review exactly once."
  );
}

export const THINKER_SYSTEM_PROMPT = thinkerSystemPrompt(G1_PROFILE);

export function isValidThinkerReview(value, profile = G1_PROFILE) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  if (!VERDICT_SET.has(value.verdict) || !profile.tokenSet.has(value.next_token)) return false;
  if (typeof value.brief !== "string" || value.brief.length < 1 || value.brief.length > 200) return false;
  if (typeof value.confidence !== "number" || !Number.isFinite(value.confidence)) return false;
  if (value.confidence < 0 || value.confidence > 1) return false;
  // STOP must mean stop; a STOP verdict that recommends motion is a contradiction, not a review.
  if (value.verdict === "STOP" && value.next_token !== profile.roles.stop) return false;
  return Object.keys(value).sort().join("|") === THINKER_CORRECTION_KEYS.join("|");
}

/** The judging half of the reviewer's answer: a known outcome plus an optional short lesson. */
export function isValidThinkerJudgement(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  if (!JUDGE_OUTCOMES.includes(value.outcome)) return false;
  if (value.lesson !== undefined && value.lesson !== null && (typeof value.lesson !== "string" || value.lesson.length > 200)) return false;
  return true;
}

/**
 * Split the model's tool arguments into the strict correction and the optional judgement. The
 * correction keeps its exact key set so the session protocol downstream stays unchanged; any extra
 * key the model invents is dropped here rather than forwarded.
 */
export function splitThinkerOutput(value, profile = G1_PROFILE) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return null;
  const review = {};
  for (const key of THINKER_CORRECTION_KEYS) if (key in value) review[key] = value[key];
  if (!isValidThinkerReview(review, profile)) return null;
  const candidate = { outcome: value.outcome, ...(value.lesson !== undefined ? { lesson: value.lesson } : {}) };
  const judgement = isValidThinkerJudgement(candidate)
    ? { outcome: candidate.outcome, lesson: typeof candidate.lesson === "string" && candidate.lesson.trim() ? candidate.lesson.trim() : null }
    : null;
  return { review, judgement };
}

function parseReview(text, profile = G1_PROFILE) {
  const start = text.indexOf("{");
  if (start === -1) return null;
  try {
    return splitThinkerOutput(JSON.parse(text.slice(start)), profile);
  } catch {
    return null;
  }
}

function frameMetadata(frame) {
  if (!frame) return null;
  const { imageB64: _drop, ...metadata } = frame;
  return metadata;
}

/** Optional brain context: which engine decided, what it thought, what memory told it. */
function contextLines(context) {
  if (!context || typeof context !== "object") return [];
  const lines = [];
  if (context.engine) lines.push(`Deciding engine: ${context.engine}${context.route ? ` (route: ${context.route})` : ""}`);
  if (context.thought) lines.push(`Engine thought: ${String(context.thought).slice(0, 240)}`);
  if (context.inline?.from) lines.push(`The reflex token ${context.inline.from} was vetoed by the gate; the deliberate engine answered ${context.inline.to ?? "nothing"}.`);
  if (context.recent) lines.push(`Recent steps: ${String(context.recent).slice(0, 400)}`);
  if (context.where) lines.push(`Where the controller thinks it is (dead reckoning, drifts): ${String(context.where).slice(0, 200)}`);
  if (context.history) lines.push(`How earlier sessions went (this task, unless a line says "similar task"): ${String(context.history).slice(0, 400)}`);
  if (context.lessons) lines.push(`Lessons the controller was given: ${String(context.lessons).slice(0, 400)}`);
  return lines;
}

function promptFor(job) {
  return [
    `Review session ${job.sessionId}, sequence ${job.seq}.`,
    `Instruction: ${job.instruction}`,
    `Selected token: ${job.token}`,
    `Selected action: ${JSON.stringify(job.action)}`,
    ...contextLines(job.context),
    `Before-frame metadata: ${JSON.stringify(frameMetadata(job.beforeFrame))}`,
    `After-frame metadata: ${JSON.stringify(frameMetadata(job.afterFrame))}`,
    `Execution feedback: ${JSON.stringify(job.feedback ?? null)}`,
    job.afterFrame
      ? "Compare the supplied before and after frames."
      : "No after frame is available; make a conservative pre-execution review.",
    "KEEP: the selected token remains appropriate; repeat that token.",
    "CORRECT: choose a different token from the fixed token set.",
    "STOP: choose X when evidence is unsafe, contradictory, or too ambiguous.",
    "Also judge the step you reviewed. outcome: progress if it advanced the task safely; no_progress if nothing useful changed (deadband, blocked, oscillating); unsafe if it moved toward a hazard or should have been refused; unclear otherwise. If the step was not executed (preview), judge the decision, not the motion.",
    "lesson: one sentence a future controller should remember about this situation (what to do instead and why), or omit it. It is stored as memory, never executed.",
    "Token meanings: X stop, W forward, S backward, A strafe-left, D strafe-right, Q turn-left, E turn-right, U both-arms-up gesture, P forward-push gesture, R release/reset arms.",
    "BOX_LIFT is unavailable: U, P, R, and the firmware action named clamp do not grasp or lift a box.",
    "The brief must be one concise externally observable reason, never a reasoning trace.",
  ].join("\n");
}

function imagePart(frame, detail = "low") {
  return {
    type: "image_url",
    image_url: { url: `data:${frame.mimeType ?? "image/jpeg"};base64,${frame.imageB64}`, detail },
  };
}

function userContent(job) {
  const content = [{ type: "text", text: promptFor(job) }, { type: "text", text: "Before frame:" }, imagePart(job.beforeFrame)];
  if (job.afterFrame) content.push({ type: "text", text: "After frame:" }, imagePart(job.afterFrame));
  return content;
}

export function createThinkerInfer({ apiKey, baseUrl, maxOutputTokens = 320, dispatcher = undefined, now = () => performance.now(), profile = G1_PROFILE }) {
  const url = chatUrl(baseUrl);
  const headers = {
    "content-type": "application/json",
    authorization: `Bearer ${apiKey}`,
    accept: "text/event-stream",
  };

  return async function infer(model, job, { signal, deadlineMs = 12_000 } = {}) {
    if (!apiKey) {
      throw new InferenceError("REA_API_KEY is not set", { errorType: "config_error", retryable: false });
    }
    if (!job?.beforeFrame?.imageB64) {
      throw new InferenceError("thinker beforeFrame is missing", { errorType: "config_error", retryable: false });
    }
    if (job.afterFrame && !job.afterFrame.imageB64) {
      throw new InferenceError("thinker afterFrame is invalid", { errorType: "config_error", retryable: false });
    }
    if (!profile.tokenSet.has(job.token)) {
      throw new InferenceError("thinker token is invalid", { errorType: "config_error", retryable: false });
    }

    const started = now();
    const body = JSON.stringify({
      model,
      stream: true,
      max_tokens: maxOutputTokens,
      messages: [
        { role: "system", content: thinkerSystemPrompt(profile) },
        { role: "user", content: userContent(job) },
      ],
      tools: [reviewTool(profile)],
      tool_choice: { type: "function", function: { name: "report_review" } },
      // Unlike the main loop the reviewer is allowed to think: it is off the control path, so its
      // latency costs nothing and a considered second opinion is the entire point of running it.
      ...(isDeepSeek(model) ? { thinking: { type: "enabled" } } : {}),
    });

    let status = null;
    let finishReason = null;
    let args = "";
    let text = "";

    try {
      const stream = streamChat({
        url,
        headers,
        dispatcher,
        body,
        signal,
        deadlineMs,
        apiKey,
        onStatus: (value) => {
          status = value;
        },
      });

      for await (const event of stream) {
        const observedAt = now();
        finishReason = event.finishReason ?? finishReason;
        for (const call of event.delta.tool_calls ?? []) args += call.function?.arguments ?? "";
        if (typeof event.delta.content === "string") text += event.delta.content;
        const parsed = parseReview(args, profile) ?? parseReview(text, profile);
        if (parsed !== null) {
          const reviewReadyS = Number(((observedAt - started) / 1000).toFixed(6));
          return { review: parsed.review, judgement: parsed.judgement, reviewReadyS, totalS: reviewReadyS, finishReason, responseStatus: status };
        }
      }
    } catch (error) {
      throw providerError(error, { apiKey, status, finishReason });
    }

    throw new InferenceError("model ended without a valid thinker review", {
      errorType: "invalid_review",
      status,
      finishReason,
      retryable: true,
    });
  };
}
