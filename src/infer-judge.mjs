// Visual completion judge: the model behind REA's judge gate (src/framework/judge-gate.mjs).
//
// One non-streaming chat completion per call: the judge has to return a whole JSON object, so
// first-token latency buys nothing and the plain response body is simpler to parse than SSE.
// The judge never sees simulator state: it answers from the rendered frame and the instruction.

import { performance } from "node:perf_hooks";

import { DEFAULT_PRICES, costUsd } from "./candidates.mjs";
import { InferenceError, chatUrl, classifyStatus, isDeepSeek, providerError, redact } from "./provider.mjs";

export const DEFAULT_JUDGE_THRESHOLD = 0.6;
export const DEFAULT_JUDGE_MAX_TOKENS = 64;
export const JUDGE_VIEWS = Object.freeze(["montage", "agentview", "front"]);
export const JUDGE_PROMPT_STYLES = Object.freeze(["default", "evidence", "pickup"]);
export const DEFAULT_JUDGE_PROMPT_STYLE = "default";
export const JUDGE_SYSTEM_PROMPT =
  "You are a strict visual inspector for a robot manipulation benchmark. Report only what is visible in the image. Answer with one JSON object and nothing else.";
// Evidence-first variant: the model names where the instruction's object and target are before it
// decides, and the JSON object comes last (parseJudgeOutput takes the last {...} in the text).
export const JUDGE_EVIDENCE_SYSTEM_PROMPT =
  "You are a strict visual inspector for a robot manipulation benchmark. Report only what is visible in the image. Give a brief visual description first, then finish with exactly one JSON object on the last line.";

// View notes describe the attached frames; the front view and the pickup style are for mobile pick-up tasks (BEHAVIOR R1Pro).
const MONTAGE_NOTE = "Left half: external front camera. Right half: wrist camera (its left/right is mirrored relative to the external view).";
const AGENTVIEW_NOTE = "External front camera.";
// BEHAVIOR/R1Pro "front" view (behavior_backend.py observe view=front): the task's external camera moved in front of the robot for the
// verdict, three views side by side. A held object usually appears high above the robot's head after ASPIRE's lift_arm.
const FRONT_NOTE = "The current image shows three views taken now by an external camera placed in front of the robot: left = from above head height looking at the head and any raised arm, middle = overview from the robot's front-left, right = overview from the robot's front-right. After a successful pick-up this robot raises the holding arm straight up, so a held object often appears high above the robot's head, near the top of the views, and may look stretched by the wide-angle lens.";
// Pick-up criterion: the default prompt can judge a pick-up against an imagined placement target and answer not_done after a
// successful grasp; this style states the end state a pick-up instruction asks for.
const PICKUP_CRITERION = "The instruction is complete when the object named in the instruction is held inside the closed fingers of one of the robot's grippers and lifted off the surface it was resting on. It does NOT have to be placed anywhere; holding it up counts as done. An object resting on the robot's wheeled base or platform, on the floor, on a table or on any furniture is NOT held, even if it touches the robot or a gripper.";

export function buildJudgePrompt({ instruction, view = "montage", style = DEFAULT_JUDGE_PROMPT_STYLE, withInitial = false } = {}) {
  const viewNote = view === "montage" ? MONTAGE_NOTE : view === "front" ? FRONT_NOTE : AGENTVIEW_NOTE;
  const header = [
    `Instruction given to the robot: "${instruction ?? ""}"`,
    withInitial
      ? (view === "front"
        ? `Two images are attached: the first is the robot's head camera at the start of the episode, before the robot moved (it shows where the object was). ${viewNote}`
        : `Two images are attached: the first shows the robot workspace at the start of the episode, before the robot moved; the second shows the workspace now. ${viewNote}`)
      : `The image shows the robot workspace now. ${viewNote}`,
  ];
  if (style === "pickup") {
    return [
      ...header,
      PICKUP_CRITERION,
      `Question: judging only from what is visible${withInitial ? " in the current image(s)" : ""}, has the instruction been completed as defined above? Answer done=true only if the target object is clearly inside the closed fingers of one gripper at the end of an arm, raised off the floor/table, in at least one view. If the object is on a surface, hidden, or you cannot tell, answer done=false or lower your confidence.`,
      'Answer exactly: {"done": true|false, "confidence": <number 0.0-1.0>, "why": "<at most 12 words>"}',
    ].join("\n");
  }
  if (style === "evidence") {
    return [
      ...header,
      withInitial
        ? "First, in one or two short sentences, compare the two images: which object moved between the start and now, where that object is now, and where the instruction's target location is (the object may be small, partly hidden by the robot, or resting inside/on top of the target). The object named in the instruction is the one the robot was asked to move; use the start image to identify it. Describe only what is visible; if the object or the target is hidden, say so."
        : "First, in one or two short sentences, state where the object named in the instruction is now and where the instruction's target location is in this image (the object may be small, partly hidden by the robot, or resting inside/on top of the target). Describe only what is visible; if the object or the target is hidden, say so.",
      `Then decide, judging only from what is visible${withInitial ? " in the second image" : ""}, whether the instruction has been completed: consider the end state the instruction asks for (object on/in the target, drawer open, switch on, ...), not what the robot seems to intend. If the target state is hidden or ambiguous, answer done=false or lower your confidence.`,
      'Finish with exactly one line: {"done": true|false, "confidence": <number 0.0-1.0>, "why": "<at most 12 words>"}',
    ].join("\n");
  }
  return [
    ...header,
    `Question: judging only from what is visible${withInitial ? " in the second image" : ""}, has the instruction been completed? Consider the end state the instruction asks for (object on/in the target, drawer open, switch on, ...), not what the robot seems to intend.`,
    "If the target state is hidden or ambiguous, answer done=false or lower your confidence.",
    'Answer exactly: {"done": true|false, "confidence": <number 0.0-1.0>, "why": "<at most 12 words>"}',
  ].join("\n");
}

export function judgeSystemPrompt(style = DEFAULT_JUDGE_PROMPT_STYLE) {
  return style === "evidence" ? JUDGE_EVIDENCE_SYSTEM_PROMPT : JUDGE_SYSTEM_PROMPT;  // pickup uses the strict inspector prompt
}

function coerceDone(value) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const word = value.trim().toLowerCase();
    if (word === "true" || word === "yes") return true;
    if (word === "false" || word === "no") return false;
  }
  return null;
}

function coerceConfidence(value) {
  if (value === undefined || value === null) return 0.5;
  const number = typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value) : NaN;
  if (!Number.isFinite(number)) return null;
  return Math.min(1, Math.max(0, number));
}

/** `{done, confidence, rationale}` from the model text, or null when it is not a usable answer. */
export function parseJudgeOutput(text) {
  if (typeof text !== "string") return null;
  const unfenced = text.replace(/```[A-Za-z]*\s*/gu, "");
  // First brace-delimited object that parses and carries `done` (prose or an evidence sentence may
  // precede it); otherwise the first object that parses at all, so a verdict without `done` is rejected.
  const matches = unfenced.match(/\{[^{}]*\}/gu);
  if (!matches) return null;
  let parsed = null;
  for (const candidate of matches) {
    let object;
    try {
      object = JSON.parse(candidate);
    } catch {
      continue;
    }
    if (parsed === null) parsed = object;
    if (object && typeof object === "object" && !Array.isArray(object) && "done" in object) {
      parsed = object;
      break;
    }
  }
  if (parsed === null) return null;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  const done = coerceDone(parsed.done);
  const confidence = coerceConfidence(parsed.confidence);
  if (done === null || confidence === null) return null;
  const rationale = typeof parsed.why === "string" ? parsed.why.slice(0, 200) : null;
  return { done, confidence, rationale };
}

/** "done" only when the model says done with confidence at or above the threshold; the runner recovers on "not_done". */
export function judgeVerdict({ done, confidence } = {}, { threshold = DEFAULT_JUDGE_THRESHOLD } = {}) {
  return done === true && Number(confidence) >= threshold ? "done" : "not_done";
}

function messageText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map(part => (typeof part === "string" ? part : part?.text ?? "")).join("");
  return "";
}

export function createJudgeInfer({
  apiKey, baseUrl, model, dispatcher = undefined, deadlineMs = 30000,
  maxTokens = DEFAULT_JUDGE_MAX_TOKENS, prices = DEFAULT_PRICES, fetchFn = globalThis.fetch, now = () => performance.now(),
  promptStyle = DEFAULT_JUDGE_PROMPT_STYLE,
} = {}) {
  if (!model) throw new TypeError("judge model is required");
  if (!JUDGE_PROMPT_STYLES.includes(promptStyle)) throw new Error(`judge promptStyle must be one of ${JUDGE_PROMPT_STYLES.join(", ")}`);
  const url = chatUrl(baseUrl ?? "");
  const headers = { "content-type": "application/json", authorization: `Bearer ${apiKey}` };

  // `maxTokens` per call lets the runner retry a `finish_reason: "length"` answer with a larger budget.
  return async function judge(frame, { instruction, view = "montage", signal, style = promptStyle, initialFrame = null, maxTokens: callMaxTokens = maxTokens } = {}) {
    if (!apiKey) throw new InferenceError("REA_API_KEY is not set", { errorType: "config_error", retryable: false });
    if (!frame?.imageB64) throw new InferenceError("judge frame is missing imageB64", { errorType: "config_error", retryable: false });
    if (initialFrame && !initialFrame.imageB64) throw new InferenceError("judge initialFrame is missing imageB64", { errorType: "config_error", retryable: false });
    const withInitial = Boolean(initialFrame);
    const imagePart = image => ({ type: "image_url", image_url: { url: `data:${image.mimeType ?? "image/jpeg"};base64,${image.imageB64}`, detail: "high" } });
    const started = now();
    const latencyS = () => Number(((now() - started) / 1000).toFixed(3));
    const fail = (error, extra = {}) => Object.assign(error, { usage: null, outputText: "", latencyS: latencyS(), maxTokens: callMaxTokens, ...extra });
    const body = JSON.stringify({
      model,
      stream: false,
      temperature: 0,
      max_tokens: callMaxTokens,
      messages: [
        { role: "system", content: judgeSystemPrompt(style) },
        {
          role: "user",
          content: [
            { type: "text", text: buildJudgePrompt({ instruction, view, style, withInitial }) },
            ...(withInitial ? [imagePart(initialFrame)] : []),
            imagePart(frame),
          ],
        },
      ],
      ...(isDeepSeek(model) ? { thinking: { type: "disabled" } } : {}),
    });

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), deadlineMs);
    const onAbort = () => controller.abort();
    signal?.addEventListener("abort", onAbort, { once: true });
    let status = null;
    let payload;
    try {
      let response;
      try {
        response = await fetchFn(url, { method: "POST", headers, body, dispatcher, signal: controller.signal });
      } catch (error) {
        if (controller.signal.aborted) {
          throw fail(new InferenceError(`judge deadline of ${deadlineMs} ms exceeded`, { errorType: "deadline", retryable: true }));
        }
        throw fail(providerError(error, { apiKey }));
      }
      status = response.status;
      if (!response.ok) {
        const detail = redact((await response.text()).slice(0, 300), apiKey);
        throw fail(new InferenceError(`upstream ${status}: ${detail}`, { status, ...classifyStatus(status) }));
      }
      try {
        payload = await response.json();
      } catch (error) {
        throw fail(new InferenceError(`judge response is not JSON: ${redact(error, apiKey)}`, { errorType: "invalid_response", status, retryable: true }));
      }
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    }

    const usage = payload?.usage ?? null;
    const choice = payload?.choices?.[0] ?? null;
    const finishReason = choice?.finish_reason ?? null;
    const outputText = messageText(choice?.message?.content);
    if (!outputText.trim()) {
      throw fail(new InferenceError(`judge returned no content (finish=${finishReason})`, { errorType: "empty_output", status, finishReason, retryable: true }), { usage, outputText });
    }
    const parsed = parseJudgeOutput(outputText);
    if (!parsed) {
      throw fail(new InferenceError(`judge output is not a {done, confidence} object (finish=${finishReason})`, { errorType: "invalid_judge_output", status, finishReason, retryable: true }), { usage, outputText });
    }
    return {
      ...parsed,
      usage,
      costUsd: costUsd(model, usage, prices),
      latencyS: latencyS(),
      model,
      promptStyle: style,
      withInitial,
      maxTokens: callMaxTokens,
      finishReason,
      responseStatus: status,
      outputText,
    };
  };
}
