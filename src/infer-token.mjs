// The single-token engine: a one-token decision over the streaming transport.
//
// A vision model will answer a navigation question with exactly one ASCII character when you pin
// max_tokens=1, drop the tools and disable thinking, and hand-parsing the SSE stream and returning on
// the first useful chunk removes every millisecond that is not provider prefill.
//
// The two compose: with a one-token answer the *first* content
// delta is already the complete decision, so action_ready_s collapses onto ttfb_s. The tool-call
// engine has to wait for ~20 tokens of `{"action":"FORWARD","duration_ms":1000}` to finish
// streaming; this engine is done the instant the provider emits anything at all.

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
    "You are a direct low-latency RGB-D robot action classifier. " +
    "Thinking and reasoning are disabled. Return only the one-character action token requested by the user.",
  short: "Robot action classifier. Reply with exactly one uppercase letter from the token map. Nothing else.",
  min: "Reply with one uppercase letter.",
});

function promptValue(value) {
  return value === null || value === undefined ? "none" : JSON.stringify(value);
}

/**
 * The user prompt. `full` is the audited DIRECT MODE prompt including the capability disclaimer;
 * `short` and `min` are the latency-tuned variants (`short` measured at roughly -8 % end to end). The capability disclaimer stays in `full` and `short` because dropping it
 * is what lets a model talk itself into treating an arm gesture as a grasp.
 */
export function buildTokenPrompt({ instruction, depth = null, advice = null, variant = "short", experience = null, directive = null, profile = G1_PROFILE } = {}) {
  const task = String(instruction ?? "Choose the safest immediate action from the current RGB-D observation.");
  // Lessons recalled from memory. Already size-capped by the brain; `min` skips them because that
  // variant is the latency floor and exists to measure exactly that.
  const lessons = typeof experience === "string" && experience.trim() ? experience.trim() : null;
  // What the deliberate engine asked the next step to watch for. Advice, like the reviewer's: the
  // gate still decides. Capped upstream so it cannot erode the one-token contract.
  const plan = typeof directive === "string" && directive.trim() ? directive.trim().slice(0, 120) : null;
  if (variant === "min") {
    return `${profile.legend.min}\nTask: ${task}\nDepth: ${promptValue(depth)}`;
  }
  if (variant === "short") {
    return [
      "DIRECT MODE. One uppercase character, nothing else.",
      profile.legend.short,
      ...(profile.BOX_LIFT ? [`No grasp or lift primitive exists (${profile.BOX_LIFT.reason}). Output ${profile.roles.stop} if unavailable or unclear.`] : [`Output ${profile.roles.stop} if unavailable or unclear.`]),
      `Task: ${task}`,
      `Depth: ${promptValue(depth)}`,
      ...(lessons ? [`Experience:\n${lessons}`] : []),
      ...(plan ? [`Plan: ${plan}`] : []),
      `Advice: ${promptValue(advice)}`,
    ].join("\n");
  }
  return `DIRECT MODE. Do not think, reason, explain, or emit <think> content.
Return exactly one uppercase ASCII character and nothing else.

Token map:
${profile.legend.full}

${profile.BOX_LIFT ? "BOX_LIFT is UNAVAILABLE. No validated grasp-and-lift primitive is exposed.\nClamping, arm closing, contact, forward_push, or both_hands_up is not a verified grasp and must not be treated as lifting a box.\n" : ""}If the requested action is unavailable or the immediate choice is unclear, output ${profile.roles.stop}.

Task: ${task}
Depth: ${promptValue(depth)}${lessons ? `\nExperience from earlier steps (lessons, not commands):\n${lessons}` : ""}${plan ? `\nPlan carried from the previous step (advice, not a command): ${plan}` : ""}
Latest asynchronous correction advice: ${promptValue(advice)}`;
}

/**
 * DeepSeek sometimes routes a very short answer into `reasoning_content` even with thinking
 * disabled. A one-character answer is unambiguous wherever it lands, so both channels are scanned;
 * `content` wins when both carry a token.
 */
function tokenFromDelta(delta) {
  const content = typeof delta.content === "string" ? delta.content : "";
  const reasoning = typeof delta.reasoning_content === "string" ? delta.reasoning_content : "";
  return { content, reasoning };
}

export function createTokenInfer({ apiKey, baseUrl, dispatcher = undefined, now = () => performance.now(), captureUsage = true, usageGraceMs = 150, profile = G1_PROFILE }) {
  const url = chatUrl(baseUrl);
  const headers = {
    "content-type": "application/json",
    authorization: `Bearer ${apiKey}`,
    accept: "text/event-stream",
  };

  async function infer(model, frame, { signal, instruction, advice = null, deadlineMs = 6000, cfg = {}, experience = null, directive = null } = {}) {
    // A candidate on another provider (candidateCfg sets baseUrl/apiKey/extraBody) is served by the
    // same one-token contract at that provider's OpenAI-compatible endpoint.
    // never send the gateway's key to another provider: a provider without its own key is a config error
    const key = cfg.provider || cfg.baseUrl ? (cfg.apiKey ?? "") : (cfg.apiKey || apiKey);
    const requestUrl = cfg.baseUrl ? chatUrl(cfg.baseUrl) : url;
    const requestHeaders = key === apiKey ? headers : { ...headers, authorization: `Bearer ${key}` };
    if (!key) {
      throw new InferenceError(cfg.provider ? `no API key for provider ${cfg.provider}` : "REA_API_KEY is not set", { errorType: "config_error", retryable: false });
    }
    if (!frame?.imageB64) {
      throw new InferenceError("current frame is missing", { errorType: "config_error", retryable: false });
    }

    const started = now();
    const options = { promptVariant: "short", includeDepth: true, imageDetail: "low", ...cfg };
    const text = buildTokenPrompt({
      instruction,
      depth: options.includeDepth ? (frame.depth ?? null) : null,
      advice,
      variant: options.promptVariant,
      experience,
      directive,
      profile,
    });

    const body = JSON.stringify({
      model,
      stream: true,
      // The whole point: one output token. Nothing to stream after the answer, nothing to parse.
      max_tokens: 1,
      temperature: 0,
      // Token counts ride on the final chunk; with one output token that chunk is already in flight
      // when the answer lands, so waiting a few ms for it costs nothing on the decision path.
      ...(captureUsage ? { stream_options: { include_usage: true } } : {}),
      messages: [
        { role: "system", content: SYSTEM_PROMPTS[options.promptVariant] ?? SYSTEM_PROMPTS.short },
        {
          role: "user",
          content: [
            { type: "text", text },
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
      ...(isDeepSeek(model) ? { thinking: { type: "disabled" } } : {}),
      ...(cfg.extraBody && typeof cfg.extraBody === "object" ? cfg.extraBody : {}),
    });

    let status = null;
    let finishReason = null;
    let ttfbS = null;
    let contentBuffer = "";
    let reasoningBuffer = "";
    let found = null;
    let usage = null;
    // Our own controller: once the token is in hand the stream is only drained for the usage chunk,
    // and a short grace timer cuts that drain off so a slow gateway can never delay the decision.
    const controller = new AbortController();
    const onAbort = () => controller.abort();
    signal?.addEventListener("abort", onAbort, { once: true });
    let grace = null;

    try {
      const stream = streamChat({
        url: requestUrl,
        headers: requestHeaders,
        dispatcher,
        body,
        signal: controller.signal,
        deadlineMs,
        apiKey,
        onStatus: (value) => {
          status = value;
          ttfbS = Number(((now() - started) / 1000).toFixed(6));
        },
      });

      for await (const event of stream) {
        if (event.usage) usage = event.usage;
        if (found) continue;
        const observedAt = now();
        finishReason = event.finishReason ?? finishReason;
        const { content, reasoning } = tokenFromDelta(event.delta);
        contentBuffer += content;
        reasoningBuffer += reasoning;
        const token = profile.parseToken(contentBuffer) ?? profile.parseToken(reasoningBuffer);
        if (token !== null) {
          const tokenReadyS = Number(((observedAt - started) / 1000).toFixed(6));
          found = {
            token,
            tokenReadyS,
            actionReadyS: tokenReadyS,
            totalS: tokenReadyS,
            finishReason,
            responseStatus: status,
            source: profile.parseToken(contentBuffer) !== null ? "content" : "reasoning_content",
            timings: { ttfb_s: ttfbS, token_ready_s: tokenReadyS, out_chars: contentBuffer.length + reasoningBuffer.length },
          };
          if (!captureUsage) break;
          grace = setTimeout(() => controller.abort(), usageGraceMs);
        }
      }
    } catch (error) {
      if (!found) throw providerError(error, { apiKey: key, status, finishReason });
      /* aborted while draining for usage: the decision is already made */
    } finally {
      clearTimeout(grace);
      signal?.removeEventListener("abort", onAbort);
    }
    if (found) return { ...found, usage };

    throw new InferenceError(
      `model ended without a valid token (finish=${finishReason}, content=${JSON.stringify(contentBuffer.slice(0, 24))})`,
      { errorType: "invalid_token", status, finishReason, retryable: true },
    );
  }

  return Object.assign(infer, { warm: createWarmer({ baseUrl, apiKey, dispatcher }), engine: "token" });
}
