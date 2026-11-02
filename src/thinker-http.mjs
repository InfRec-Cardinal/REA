import { createServer } from "node:http";

import { ThinkerJobError } from "./thinker-runtime.mjs";

const DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024;

class RequestBodyError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.status = status;
  }
}

function sendJson(response, status, payload) {
  const data = Buffer.from(`${JSON.stringify(payload)}\n`);
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": data.length,
    "cache-control": "no-store",
  });
  response.end(data);
}

async function readJson(request, maxBodyBytes) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxBodyBytes) throw new RequestBodyError("request body is too large", 413);
    chunks.push(chunk);
  }
  if (size === 0) throw new RequestBodyError("JSON body is required");
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new RequestBodyError("malformed JSON");
  }
}

function log(logger, level, record) {
  const fn = logger?.[level];
  if (typeof fn === "function") fn.call(logger, record);
}

export function createThinkerHttpServer({
  runtime,
  logger = console,
  maxBodyBytes = DEFAULT_MAX_BODY_BYTES,
} = {}) {
  if (!runtime || typeof runtime.submit !== "function" || typeof runtime.health !== "function") {
    throw new TypeError("runtime with submit() and health() is required");
  }

  return createServer(async (request, response) => {
    const url = new URL(request.url ?? "/", "http://localhost");
    if (request.method === "GET" && url.pathname === "/health") {
      sendJson(response, 200, runtime.health());
      return;
    }

    if (!(request.method === "POST" && url.pathname === "/v1/review")) {
      sendJson(response, 404, { ok: false, error_type: "not_found", error: "not found" });
      return;
    }

    try {
      const body = await readJson(request, maxBodyBytes);
      if (body === null || typeof body !== "object" || Array.isArray(body)) {
        throw new RequestBodyError("JSON body must be an object");
      }
      const accepted = runtime.submit(body);
      log(logger, "info", {
        event: "thinker_job_accepted",
        session_id: accepted.sessionId,
        seq: accepted.seq,
        generation: accepted.generation,
        replaced: accepted.replaced,
      });
      sendJson(response, 202, {
        ok: true,
        accepted: true,
        asynchronous: true,
        sessionId: accepted.sessionId,
        seq: accepted.seq,
        generation: accepted.generation,
        replaced: accepted.replaced,
      });
    } catch (error) {
      const closed = error instanceof ThinkerJobError && /closed/u.test(error.message);
      const status = closed ? 503 : Number.isInteger(error?.status) ? error.status : 400;
      const errorType = closed
        ? "unavailable"
        : status === 413
          ? "request_too_large"
          : "request_error";
      sendJson(response, status, {
        ok: false,
        error_type: errorType,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
}
