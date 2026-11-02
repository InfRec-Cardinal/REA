import { InferenceError, chatUrl, isDeepSeek, providerError, streamChat } from "../provider.mjs";
import { createProgressPolicy } from "./progress.mjs";

const DEFAULT_SYSTEM_PROMPT = "You are an agent acting in an interactive environment. Choose one available action using the current observation and earlier outcomes. Infer rules from observed changes. Keep any explanation brief. Use notes to preserve useful facts, hypotheses, and the next objective across steps; return the complete replacement notes when updating them.";

function historyText(history, limit) {
  if (limit <= 0) return "[]";
  const entries = history.slice(-limit).map(entry => ({
    step: entry.step,
    action: entry.action,
    reward: entry.reward,
    terminated: entry.terminated,
    truncated: entry.truncated,
    success: entry.success,
    info: entry.info,
    thought: entry.thought ?? entry.result?.thought,
    observation: entry.observation,
    nextObservation: entry.nextObservation,
  }));
  return JSON.stringify(entries, (key, item) => key === "images" || key === "imageB64" ? undefined : item);
}

function decisionSchema(actions) {
  return {
    type: "object",
    properties: {
      name: { type: "string", enum: actions.map(action => action.name) },
      arguments: { type: "object", description: "Arguments matching the selected action's parameter schema in the prompt. Use {} for an action without parameters." },
      thought: { type: "string", description: "A brief explanation of the action." },
      notes: { type: "string", description: "Complete replacement episode notes, including useful rules, hypotheses, and next objective." },
      claim: {
        type: "object",
        description: "Whether you believe the task is already complete, with the observed evidence.",
        properties: {
          status: { type: "string", enum: ["working", "complete"] },
          evidence: { type: "string" },
        },
        required: ["status", "evidence"],
        additionalProperties: false,
      },
    },
    required: ["name", "arguments"],
    additionalProperties: false,
  };
}

function supportsToken(action) {
  const parameters = action.parameters ?? {};
  return /^[A-Za-z0-9]$/u.test(action.name)
    && Object.keys(parameters.properties ?? {}).length === 0
    && (parameters.required ?? []).length === 0
    && !parameters.additionalProperties;
}

export function parseDecision(text, actions) {
  let decision;
  try {
    decision = JSON.parse(text);
  } catch {
    throw new InferenceError("model returned invalid decision JSON", { errorType: "invalid_action" });
  }
  if (!decision || !actions.some(action => action.name === decision.name)
    || !decision.arguments || typeof decision.arguments !== "object" || Array.isArray(decision.arguments)
    || (decision.thought !== undefined && typeof decision.thought !== "string")
    || (decision.notes !== undefined && typeof decision.notes !== "string")
    || (decision.claim !== undefined && (!decision.claim || typeof decision.claim !== "object" || Array.isArray(decision.claim)
      || !["working", "complete"].includes(decision.claim.status) || typeof decision.claim.evidence !== "string"))) {
    throw new InferenceError("model returned an invalid action decision", { errorType: "invalid_action" });
  }
  return {
    action: { name: decision.name, arguments: decision.arguments },
    ...(decision.thought !== undefined ? { thought: decision.thought } : {}),
    ...(decision.notes !== undefined ? { notes: decision.notes } : {}),
    ...(decision.claim !== undefined ? { claim: { status: decision.claim.status, evidence: decision.claim.evidence } } : {}),
  };
}

export function buildDecisionPrompt({
  observation, actions = observation?.actions, history = [], historyForPrompt = history,
  instruction, budget = {}, notes = "", feedback = null, guidance = null,
  historyLimit = 8, deadlineMs = budget.decisionMs ?? 30000, format,
}) {
  return [
    `Task: ${instruction ?? observation?.instruction ?? "Make progress in the environment using its available actions."}`,
    `Remaining budget: ${JSON.stringify({ stepsRemaining: budget.stepsRemaining ?? null, remainingMs: budget.remainingMs ?? null, decisionMs: deadlineMs })}. Time values are milliseconds; null means the caller did not specify a limit.`,
    `Available actions and parameter schemas: ${JSON.stringify(actions)}`,
    `Current observation: ${JSON.stringify({ id: observation?.id, timestampMs: observation?.timestampMs, text: observation?.text, state: observation?.state, grid: observation?.grid })}`,
    `Episode notes: ${notes}`,
    `Recent transitions: ${historyText(historyForPrompt, historyLimit)}`,
    `Feedback: ${JSON.stringify(feedback)}`,
    ...(guidance ? [`Supervisor guidance: ${guidance}`] : []),
    format,
  ].join("\n\n");
}

export function createChatPolicy({
  apiKey = "",
  baseUrl,
  model,
  mode = "tool",
  maxOutputTokens = 1024,
  temperature = 0,
  thinking = isDeepSeek(model) ? { type: "disabled" } : undefined,
  dispatcher,
  systemPrompt = DEFAULT_SYSTEM_PROMPT,
  historyLimit = 8,
  progress = false,
} = {}) {
  if (!baseUrl || !model) throw new TypeError("chat policy requires baseUrl and model");
  if (!["tool", "json", "token"].includes(mode)) throw new TypeError(`unsupported chat policy mode: ${mode}`);
  const url = chatUrl(baseUrl);

  async function decide({ observation, actions = observation?.actions, history = [], historyForPrompt = history, instruction, budget = {}, notes = "", feedback = null, guidance = null, signal }) {
    signal?.throwIfAborted();
    if (!actions?.length) throw new TypeError("chat policy requires available action specifications");
    if (mode === "token" && !actions.every(supportsToken)) {
      throw new TypeError("token mode requires parameter-free single-character action names; use tool or json mode for parameterized actions");
    }
    const deadlineMs = budget.decisionMs ?? 30000;
    if (!(deadlineMs > 0)) throw Object.assign(new InferenceError("decision time budget exhausted", { errorType: "deadline", retryable: false }), { code: "deadline" });
    const format = mode === "tool"
      ? "Call take_action exactly once."
      : mode === "json"
        ? 'Return one JSON object: {"name":"available action name","arguments":{},"thought":"optional brief explanation","notes":"optional replacement notes","claim":{"status":"working or complete","evidence":"brief observed evidence"}}.'
        : "Return exactly one available action character. No explanation or notes.";
    const prompt = buildDecisionPrompt({ observation, actions, historyForPrompt, instruction, budget,
      notes, feedback, guidance, historyLimit, deadlineMs, format });
    const schema = decisionSchema(actions);
    const body = {
      model,
      stream: true,
      stream_options: { include_usage: true },
      max_tokens: mode === "token" ? 1 : maxOutputTokens,
      temperature,
      ...(thinking ? { thinking } : {}),
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: [
          { type: "text", text: prompt },
          ...(observation?.images ?? []).flatMap(image => [
            ...(image.name ? [{ type: "text", text: `Camera: ${image.name}` }] : []),
            { type: "image_url", image_url: { url: `data:${image.mimeType};base64,${image.data}` } },
          ]),
        ] },
      ],
      ...(mode === "tool" ? {
        tools: [{ type: "function", function: { name: "take_action", description: "Choose one available environment action and optionally update episode notes.", parameters: schema } }],
        tool_choice: { type: "function", function: { name: "take_action" } },
      } : {}),
      ...(mode === "json" ? { response_format: { type: "json_object" } } : {}),
    };
    const calls = new Map();
    let content = "";
    let usage = null;
    let status = null;
    let finishReason = null;
    try {
      for await (const event of streamChat({
        url,
        headers: { "content-type": "application/json", accept: "text/event-stream", ...(apiKey ? { authorization: `Bearer ${apiKey}` } : {}) },
        dispatcher,
        body: JSON.stringify(body),
        signal,
        deadlineMs,
        apiKey,
        onStatus: value => { status = value; },
        onUsage: value => { usage = value; },
      })) {
        finishReason = event.finishReason ?? finishReason;
        if (typeof event.delta.content === "string") content += event.delta.content;
        for (const delta of event.delta.tool_calls ?? []) {
          const index = delta.index ?? 0;
          const call = calls.get(index) ?? { name: "", arguments: "" };
          call.name += delta.function?.name ?? "";
          call.arguments += delta.function?.arguments ?? "";
          calls.set(index, call);
        }
      }
      if (finishReason === "length") {
        throw new InferenceError("model exhausted its output token budget before completing the action; increase maxOutputTokens or disable thinking", { errorType: "output_limit" });
      }
      let result;
      if (mode === "tool") {
        const call = [...calls.values()][0];
        if (calls.size !== 1 || call.name !== "take_action") {
          throw new InferenceError("model must call take_action exactly once", { errorType: "invalid_action" });
        }
        result = parseDecision(call.arguments, actions);
      } else if (mode === "json") {
        result = parseDecision(content, actions);
      } else {
        const name = content.trim();
        if (!actions.some(action => action.name === name)) {
          throw new InferenceError("model returned an invalid action token", { errorType: "invalid_action" });
        }
        result = { action: { name, arguments: {} } };
      }
      return { ...result, usage, model };
    } catch (error) {
      signal?.throwIfAborted();
      const classified = error?.name === "AbortError"
        ? new InferenceError(`decision exceeded ${deadlineMs} ms`, { errorType: "deadline", retryable: false })
        : providerError(error, { apiKey, status, finishReason });
      classified.code ??= classified.errorType;
      classified.usage = usage;
      throw classified;
    }
  }

  const rawPolicy = { id: `chat:${model}:${mode}`, decide };
  if (progress === false) return rawPolicy;
  const progressOptions = progress && typeof progress === "object" ? progress : {};
  return createProgressPolicy({ policy: rawPolicy, ...progressOptions });
}
