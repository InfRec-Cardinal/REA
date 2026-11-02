// Reflection: the Reflexion pattern (verbal self-critique over a trajectory), bounded to one tool call.
//
// The heuristic judge sees one step; the reviewer sees one step with its after frame. Neither sees the
// shape of a failing episode — three vetoes in a row, a Q/E oscillation, a forward stall. When the
// brain's signals say the episode is going wrong, it hands the last few steps, the gate's reasons and
// the lessons it already knows to a text-only model and asks for ONE lesson: what to do instead in this
// situation and why. The answer is stored as a `reflection` lesson and recalled like any other.
//
// Off the control path by construction: it is fire-and-forget, rate-limited, text-only (no frame), and
// its only output is memory. Nothing here holds an actuator.

import { performance } from "node:perf_hooks";

import { G1_PROFILE } from "./profiles/g1.mjs";
import { situationPhrase } from "./memory.mjs";
import { InferenceError, chatUrl, isDeepSeek, providerError, streamChat } from "./provider.mjs";

export const REFLECTION_MAX_CHARS = 200;

const REFLECTION_SCHEMA_BASE = {
  type: "object",
  properties: {
    lesson: {
      type: "string",
      minLength: 1,
      maxLength: REFLECTION_MAX_CHARS,
      description: "One sentence: what to do instead in this situation and why.",
    },
    alternative: { type: "string", enum: [], description: "The token that would have been better, when one clearly is." },
    confidence: { type: "number", minimum: 0, maximum: 1 },
    applies_when: { type: "string", maxLength: 80, description: "The situation the lesson applies to, briefly." },
  },
  required: ["lesson", "confidence"],
  additionalProperties: false,
};

/** The reflection schema for one profile: only the alternative-token enum varies. */
export function reflectionSchema(profile = G1_PROFILE) {
  return {
    ...REFLECTION_SCHEMA_BASE,
    properties: { ...REFLECTION_SCHEMA_BASE.properties, alternative: { ...REFLECTION_SCHEMA_BASE.properties.alternative, enum: [...profile.TOKENS] } },
  };
}

export const REFLECTION_SCHEMA = reflectionSchema(G1_PROFILE);

const reflectTool = profile => ({
  type: "function",
  function: {
    name: "report_reflection",
    description: "Return one lesson learned from the recent steps.",
    parameters: reflectionSchema(profile),
  },
});

export function reflectSystemPrompt(profile = G1_PROFILE) {
  return (
    `You are the reflective half of a bounded ${profile.name} controller. ` +
    "You are shown the last few control steps of one session, what the safety gate and the judges said, and the lessons already known. " +
    "Write ONE lesson a future controller should apply in this situation: what to do instead and why, in one sentence. " +
    "Name an alternative token only when one is clearly better. " +
    "No actuator is connected to you; the lesson is stored as memory and re-checked by the gate whenever it is acted on. " +
    "Call report_reflection exactly once."
  );
}

export const REFLECT_SYSTEM_PROMPT = reflectSystemPrompt(G1_PROFILE);

export function isValidReflection(value, profile = G1_PROFILE) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  if (typeof value.lesson !== "string" || value.lesson.trim().length === 0 || value.lesson.length > REFLECTION_MAX_CHARS) return false;
  if (typeof value.confidence !== "number" || !Number.isFinite(value.confidence) || value.confidence < 0 || value.confidence > 1) return false;
  if (value.alternative !== undefined && profile.parseToken(value.alternative) === null) return false;
  if (value.applies_when !== undefined && (typeof value.applies_when !== "string" || value.applies_when.length > 80)) return false;
  return true;
}

function parseReflection(text, profile = G1_PROFILE) {
  const start = text.indexOf("{");
  if (start === -1) return null;
  try {
    const value = JSON.parse(text.slice(start));
    return isValidReflection(value, profile) ? value : null;
  } catch {
    return null;
  }
}

/** The reflection prompt: text only, bounded, and explicit that it is asking for memory, not a command. */
export function buildReflectPrompt({ instruction, situation = null, trigger = null, recent = null, lessons = null, token = null, profile = G1_PROFILE } = {}) {
  const lines = [
    `Task: ${String(instruction ?? "Choose the safest immediate action from the current RGB-D observation.")}`,
    `Token map: ${profile.legend.short}`,
    `Situation at the last step: ${situation ? situationPhrase(situation) : "unknown"}`,
    `Last token: ${token ?? "none"}`,
    `Why this reflection was triggered: ${trigger ?? "unspecified"}`,
    `Recent steps (newest last): ${recent && String(recent).trim() ? recent : "none"}`,
    `Lessons already known: ${lessons && String(lessons).trim() ? `\n${lessons}` : "none"}`,
    "",
    "Reflect on why these steps did not achieve the task, then report one lesson for this situation. Do not repeat a lesson already known; refine or contradict it if the evidence says so.",
  ];
  return lines.join("\n");
}

export function createReflectInfer({ apiKey, baseUrl, dispatcher = undefined, maxOutputTokens = 160, now = () => performance.now(), profile = G1_PROFILE }) {
  const url = chatUrl(baseUrl);
  const headers = {
    "content-type": "application/json",
    authorization: `Bearer ${apiKey}`,
    accept: "text/event-stream",
  };

  /** input: { instruction, situation, trigger, recent, lessons, token } — see buildReflectPrompt(). */
  async function infer(model, input, { signal, deadlineMs = 6000 } = {}) {
    if (!apiKey) throw new InferenceError("REA_API_KEY is not set", { errorType: "config_error", retryable: false });
    const started = now();
    const body = JSON.stringify({
      model,
      stream: true,
      max_tokens: maxOutputTokens,
      temperature: 0.2,
      messages: [
        { role: "system", content: reflectSystemPrompt(profile) },
        { role: "user", content: buildReflectPrompt({ ...(input ?? {}), profile }) },
      ],
      tools: [reflectTool(profile)],
      tool_choice: { type: "function", function: { name: "report_reflection" } },
      ...(isDeepSeek(model) ? { thinking: { type: "disabled" } } : {}),
    });

    let status = null;
    let finishReason = null;
    let args = "";
    let text = "";
    try {
      const stream = streamChat({ url, headers, dispatcher, body, signal, deadlineMs, apiKey, onStatus: value => { status = value; } });
      for await (const event of stream) {
        finishReason = event.finishReason ?? finishReason;
        for (const call of event.delta.tool_calls ?? []) args += call.function?.arguments ?? "";
        if (typeof event.delta.content === "string") text += event.delta.content;
        const reflection = parseReflection(args, profile) ?? parseReflection(text, profile);
        if (reflection !== null) {
          return {
            lesson: reflection.lesson.trim(),
            alternative: reflection.alternative !== undefined ? profile.parseToken(reflection.alternative) : null,
            confidence: reflection.confidence,
            appliesWhen: reflection.applies_when?.trim() || null,
            elapsedS: Number(((now() - started) / 1000).toFixed(6)),
            finishReason,
            responseStatus: status,
          };
        }
      }
    } catch (error) {
      throw providerError(error, { apiKey, status, finishReason });
    }
    throw new InferenceError(`model ended without a valid reflection (finish=${finishReason})`, {
      errorType: finishReason === "length" ? "length" : "invalid_reflection",
      status,
      finishReason,
      retryable: true,
    });
  }

  return Object.assign(infer, { engine: "reflect", deliberate: true });
}
