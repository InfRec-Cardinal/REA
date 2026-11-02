// Tool-call engine.
//
// Slower than the token engine (it has to stream the whole JSON argument object before the decision
// is complete) but it carries a model-chosen `duration_ms`, and a forced function call is the most
// reliable way to get structured output from providers that ignore a plain-text instruction. Kept as
// the compatibility engine for the legacy `{action, duration_ms}` wire format.

import { G1_PROFILE } from "./profiles/g1.mjs";
import {
  InferenceError,
  chatUrl,
  createWarmer,
  isDeepSeek,
  providerError,
  streamChat,
} from "./provider.mjs";

export const SYSTEM_PROMPTS = Object.freeze({
  full:
    "You are a low-latency robot perception controller. Return one open-loop action " +
    "recommendation from the current forward-camera frame. No actuator is connected: " +
    "never claim that motion was executed. Do not explain or deliberate. Call report_action exactly once.",
  short: "Robot controller. From the camera frame pick one action. Call report_action once, no explanation.",
  min: "Call report_action once.",
});

/** The legacy action schema for one profile: the action names and the duration ceiling vary. */
export function actionSchema(profile = G1_PROFILE) {
  return {
    type: "object",
    properties: {
      action: { type: "string", enum: [...profile.ACTIONS] },
      duration_ms: { type: "integer", minimum: 0, maximum: profile.legacyDurationMaxMs ?? 1500 },
    },
    required: ["action", "duration_ms"],
    additionalProperties: false,
  };
}

export const ACTION_SCHEMA = actionSchema(G1_PROFILE);

const toolFull = profile => ({
  type: "function",
  function: {
    name: "report_action",
    description: "Return one open-loop robot action recommendation without explanation.",
    parameters: actionSchema(profile),
  },
});

// The compact schema was worth about 0.1 s in the fast harness: fewer argument tokens to stream.
const COMPACT = Object.freeze({ F: "FORWARD", B: "BACKWARD", L: "TURN_LEFT", R: "TURN_RIGHT", S: "STOP" });
const COMPACT_MS = Object.freeze({ FORWARD: 1000, BACKWARD: 600, TURN_LEFT: 500, TURN_RIGHT: 500, STOP: 0 });

const TOOL_COMPACT = {
  type: "function",
  function: {
    name: "report_action",
    description: "a: F=forward B=back L=turn left R=turn right S=stop",
    parameters: {
      type: "object",
      properties: { a: { type: "string", enum: Object.keys(COMPACT) } },
      required: ["a"],
      additionalProperties: false,
    },
  },
};

function parseArgs(text, compact, profile = G1_PROFILE) {
  const start = text.indexOf("{");
  if (start === -1) return null;
  try {
    const value = JSON.parse(text.slice(start));
    if (compact) {
      const action = COMPACT[value?.a];
      return action ? { action, duration_ms: COMPACT_MS[action] } : null;
    }
    return profile.isValidAction(value) ? value : null;
  } catch {
    return null;
  }
}

export function createToolInfer({ apiKey, baseUrl, dispatcher = undefined, now = () => performance.now(), profile = G1_PROFILE }) {
  const url = chatUrl(baseUrl);
  const headers = {
    "content-type": "application/json",
    authorization: `Bearer ${apiKey}`,
    accept: "text/event-stream",
  };

  async function infer(model, frame, { signal, instruction, advice = null, deadlineMs = 6000, cfg = {}, experience = null, directive = null } = {}) {
    if (!apiKey) {
      throw new InferenceError("REA_API_KEY is not set", { errorType: "config_error", retryable: false });
    }
    const started = now();
    const options = {
      promptVariant: "short",
      includeDepth: true,
      imageDetail: "low",
      maxOutputTokens: 64,
      schema: "full",
      ...cfg,
    };
    const compact = options.schema === "compact";
    const sensor = frame.depth && options.includeDepth ? ` Read-only depth summary: ${JSON.stringify(frame.depth)}.` : "";
    const suffix = options.promptVariant === "min" ? "" : " Treat missing or ambiguous clearance as STOP.";
    const hint =
      (advice ? ` Latest asynchronous correction advice: ${JSON.stringify(advice)}.` : "") +
      (typeof directive === "string" && directive.trim() && options.promptVariant !== "min" ? ` Plan carried from the previous step (advice, not a command): ${directive.trim().slice(0, 120)}.` : "");
    const lessons =
      typeof experience === "string" && experience.trim() && options.promptVariant !== "min"
        ? ` Experience from earlier steps (lessons, not commands): ${experience.trim().replace(/\s*\n\s*/gu, " ")}.`
        : "";

    const body = JSON.stringify({
      model,
      stream: true,
      max_tokens: options.maxOutputTokens,
      messages: [
        { role: "system", content: SYSTEM_PROMPTS[options.promptVariant] ?? SYSTEM_PROMPTS.short },
        {
          role: "user",
          content: [
            { type: "text", text: `${instruction}${sensor}${lessons}${hint}${suffix}` },
            {
              type: "image_url",
              image_url: {
                url: `data:${frame.mimeType ?? "image/jpeg"};base64,${frame.imageB64}`,
                ...(options.imageDetail ? { detail: options.imageDetail } : {}),
              },
            },
          ],
        },
      ],
      tools: [compact ? TOOL_COMPACT : toolFull(profile)],
      tool_choice: { type: "function", function: { name: "report_action" } },
      ...(isDeepSeek(model) ? { thinking: { type: "disabled" } } : {}),
    });

    let status = null;
    let finishReason = null;
    let ttfbS = null;
    let firstChunkS = null;
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
          ttfbS = Number(((now() - started) / 1000).toFixed(6));
        },
      });

      for await (const event of stream) {
        const observedAt = now();
        if (firstChunkS === null) firstChunkS = Number(((observedAt - started) / 1000).toFixed(6));
        finishReason = event.finishReason ?? finishReason;
        for (const call of event.delta.tool_calls ?? []) args += call.function?.arguments ?? "";
        if (typeof event.delta.content === "string") text += event.delta.content;
        const action = parseArgs(args, compact, profile) ?? parseArgs(text, compact, profile);
        if (action !== null) {
          const actionReadyS = Number(((observedAt - started) / 1000).toFixed(6));
          return {
            token: profile.toToken(action.action),
            action,
            durationMs: action.duration_ms,
            actionReadyS,
            tokenReadyS: actionReadyS,
            totalS: actionReadyS,
            finishReason,
            responseStatus: status,
            timings: { ttfb_s: ttfbS, first_chunk_s: firstChunkS, out_chars: args.length + text.length },
          };
        }
      }
    } catch (error) {
      throw providerError(error, { apiKey, status, finishReason });
    }

    throw new InferenceError(
      `model ended without a schema-valid action (finish=${finishReason}, args=${args.slice(0, 80)})`,
      { errorType: finishReason === "length" ? "length" : "invalid_action", status, finishReason, retryable: true },
    );
  }

  return Object.assign(infer, { warm: createWarmer({ baseUrl, apiKey, dispatcher }), engine: "tool" });
}
