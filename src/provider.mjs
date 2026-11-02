// Shared OpenAI-compatible streaming transport for every inference engine.
//
// A hand-rolled fetch + SSE reader rather than an SDK: it can stop reading the moment the answer is
// complete instead of draining the stream.

export class InferenceError extends Error {
  constructor(message, { errorType = "inference_error", status = null, finishReason = null, retryable = true } = {}) {
    super(message);
    this.name = "InferenceError";
    this.errorType = errorType;
    this.status = status;
    this.upstreamStatus = status;
    this.finishReason = finishReason;
    this.retryable = retryable;
  }
}

export function apiRoot(baseUrl) {
  return String(baseUrl)
    .trim()
    .replace(/\/+$/u, "")
    .replace(/\/chat\/completions$/u, "");
}

export function isDeepSeek(model) {
  return /^deepseek-/iu.test(String(model));
}

export function redact(text, apiKey) {
  const raw = text instanceof Error ? text.message : String(text);
  return apiKey ? raw.split(apiKey).join("[redacted]") : raw;
}

export function classifyStatus(status) {
  if (status === 401 || status === 403) return { errorType: "auth_error", retryable: false };
  if (status !== null && status >= 400) {
    return { errorType: `upstream_${status}`, retryable: status === 429 || status >= 500 };
  }
  return { errorType: "stream_error", retryable: true };
}

/**
 * One shared keep-alive dispatcher. TLS handshakes and proxy CONNECT tunnels are not free on the
 * Jetson, and a reused socket removes them from the request path. Node's global fetch already pools
 * connections; when the standalone `undici` package is present we build a longer-lived pool instead.
 */
export async function createDispatcher(env = process.env) {
  try {
    const { ProxyAgent, Agent } = await import("undici");
    const proxy = env.HTTPS_PROXY || env.https_proxy;
    const options = { keepAliveTimeout: 60_000, keepAliveMaxTimeout: 300_000, connections: 4, pipelining: 1 };
    return proxy ? new ProxyAgent({ uri: proxy, ...options }) : new Agent(options);
  } catch {
    return undefined;
  }
}

export function chatUrl(baseUrl) {
  return `${apiRoot(baseUrl)}/chat/completions`;
}

/**
 * Cheap request that exercises the TLS session and proxy tunnel so the next real decision does not
 * pay for them. Returns the HTTP status, or the error message when the provider is unreachable.
 */
export function createWarmer({ baseUrl, apiKey, dispatcher }) {
  const url = `${apiRoot(baseUrl)}/models`;
  const headers = { authorization: `Bearer ${apiKey}` };
  return async function warm() {
    try {
      const response = await fetch(url, { dispatcher, headers, signal: AbortSignal.timeout(8000) });
      await response.arrayBuffer();
      return response.status;
    } catch (error) {
      return String(error?.message ?? error);
    }
  };
}

/**
 * Streams an OpenAI-compatible chat completion and yields each `choices[0]` delta as it arrives.
 *
 * The caller decides when it has enough and simply stops iterating; `abort()` on the returned
 * controller then drops the rest of the response. `onStatus` fires once with the HTTP status.
 */
export async function* streamChat({ url, headers, dispatcher, body, signal, deadlineMs = 6000, apiKey = "", onStatus, onUsage }) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), deadlineMs);
  signal?.addEventListener("abort", () => controller.abort(), { once: true });
  let status = null;
  try {
    const response = await fetch(url, {
      method: "POST",
      body,
      headers,
      dispatcher,
      signal: controller.signal,
    });
    status = response.status;
    onStatus?.(status);
    if (!response.ok) {
      const detail = redact((await response.text()).slice(0, 300), apiKey);
      throw new InferenceError(`upstream ${status}: ${detail}`, { status, ...classifyStatus(status) });
    }

    const decoder = new TextDecoder();
    let buffer = "";
    for await (const chunk of response.body) {
      buffer += decoder.decode(chunk, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        let event;
        try {
          event = JSON.parse(data);
        } catch {
          continue;
        }
        if (event.usage) onUsage?.(event.usage);
        const choice = event.choices?.[0];
        if (!choice) {
          // With stream_options.include_usage the last chunk carries only the token counts.
          if (event.usage) yield { delta: {}, finishReason: null, status, usage: event.usage };
          continue;
        }
        yield { delta: choice.delta ?? {}, finishReason: choice.finish_reason ?? null, status, usage: event.usage ?? null };
      }
    }
  } finally {
    clearTimeout(timer);
    // The consumer may have stopped early; drop the socket rather than draining the rest.
    try {
      controller.abort();
    } catch {
      /* already aborted */
    }
  }
}

/** Wraps any thrown value into an InferenceError with a provider-appropriate classification. */
export function providerError(error, { apiKey = "", status = null, finishReason = null } = {}) {
  if (error instanceof InferenceError) return error;
  const observed = Number.isInteger(error?.status) ? error.status : status;
  return new InferenceError(redact(error, apiKey), {
    status: observed,
    finishReason,
    ...classifyStatus(observed),
  });
}
