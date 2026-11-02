function endpoint(baseUrl, path) {
  return `${String(baseUrl).replace(/\/+$/u, "")}${path}`;
}

export function createThinkerClient({
  baseUrl = "http://127.0.0.1:8092",
  fetchFn = globalThis.fetch,
  submitTimeoutMs = 1500,
} = {}) {
  async function submit(job) {
    const response = await fetchFn(endpoint(baseUrl, "/v1/review"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(job),
      signal: AbortSignal.timeout(submitTimeoutMs),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok || payload?.ok === false) {
      throw new Error(payload?.error ?? `thinker returned HTTP ${response.status}`);
    }
    return payload;
  }

  return { submit, baseUrl: String(baseUrl).replace(/\/+$/u, "") };
}
