import { createServer } from "node:http";
import { readFile, watchFile, unwatchFile } from "node:fs";
import { readFile as readText } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createAsyncSession } from "../src/framework/async-session.mjs";
import { createAsyncSystemDemo } from "../examples/async-system-demo.mjs";

const assets = new Map([
  ["/", ["index.html", "text/html"]],
  ["/app.js", ["app.js", "text/javascript"]],
  ["/style.css", ["style.css", "text/css"]],
]);
const publicDir = fileURLToPath(new URL("../public/async/", import.meta.url));

async function createReplaySession(path) {
  const id = "benchmark-replay";
  const listeners = new Set();
  let events = [];
  let summary = null;
  let latestTransition = null;
  let state = "replaying";
  let readError = null;
  async function refresh() {
    try {
      const content = await readText(path, "utf8");
      const lines = content.split("\n");
      // A writer may still be appending its final line.
      if (!content.endsWith("\n")) lines.pop();
      const records = lines.filter(Boolean).map(line => JSON.parse(line));
      for (const record of records.slice(events.length)) {
        const source = record.event ?? record;
        const event = { ...source, sessionId: id, sequence: events.length + 1 };
        events.push(event);
        if (event.transition) latestTransition = event.transition;
        if (event.summary) summary = event.summary;
        if (event.type === "episode" && ["completed", "cancelled", "failed"].includes(event.status)) state = event.status;
        for (const listener of [...listeners]) listener(event);
      }
      readError = null;
    } catch (error) { readError = { message: error.message }; }
  }
  await refresh();
  if (readError) throw new Error(readError.message);
  watchFile(path, { interval: 700, persistent: false }, refresh);
  return {
    id,
    status: () => ({ id, state, sequence: events.length, source: "benchmark", readOnly: true, mode: summary?.mode ?? events.find(event => event.mode)?.mode ?? "recorded", summary, latestTransition, error: readError }),
    history: ({ after = 0 } = {}) => events.filter(event => event.sequence > after),
    subscribe(listener, { after = 0, replay = true } = {}) {
      if (replay) for (const event of events) if (event.sequence > after) listener(event);
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    close() { unwatchFile(path, refresh); },
  };
}

/** Local zero-dependency SSE console. Replay only reads the supplied event file. */
export async function createAsyncConsole({ replayPath, demoOptions = {} } = {}) {
  const sessions = new Map();
  if (replayPath) {
    const replay = await createReplaySession(resolve(replayPath));
    sessions.set(replay.id, replay);
  }
  const server = createServer(async (request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    const json = (status, body) => {
      response.writeHead(status, { "content-type": "application/json", "cache-control": "no-store" });
      response.end(JSON.stringify(body));
    };
    try {
      if (request.method === "GET" && assets.has(url.pathname)) {
        const [name, type] = assets.get(url.pathname);
        readFile(resolve(publicDir, name), (error, content) => {
          if (error) return json(500, { error: error.message });
          response.writeHead(200, { "content-type": `${type}; charset=utf-8`, "cache-control": "no-cache" });
          response.end(content);
        });
        return;
      }
      if (request.method === "GET" && url.pathname === "/api/runs") {
        return json(200, { runs: [...sessions.values()].map(session => session.status()) });
      }
      if (request.method === "POST" && url.pathname === "/api/runs") {
        let body = "";
        for await (const chunk of request) body += chunk;
        const input = JSON.parse(body || "{}");
        if (!["async", "sync"].includes(input.mode ?? "async")) return json(400, { error: "mode must be async or sync" });
        const session = createAsyncSession(createAsyncSystemDemo({ ...demoOptions, mode: input.mode ?? "async" }));
        sessions.set(session.id, session);
        return json(202, { ...session.status(), source: "demonstration" });
      }
      const match = url.pathname.match(/^\/api\/runs\/([^/]+)(?:\/(events|cancel))?$/);
      if (match) {
        const session = sessions.get(match[1]);
        if (!session) return json(404, { error: "Unknown session" });
        if (match[2] === "cancel" && request.method === "POST") {
          if (!session.cancel) return json(409, { error: "A benchmark replay is read-only" });
          session.cancel();
          return json(202, session.status());
        }
        if (match[2] === "events" && request.method === "GET") {
          response.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", "connection": "keep-alive" });
          response.write(`event: snapshot\ndata: ${JSON.stringify(session.status())}\n\n`);
          const after = Number(request.headers["last-event-id"] ?? url.searchParams.get("after") ?? 0);
          const unsubscribe = session.subscribe(event => {
            response.write(`id: ${event.sequence}\nevent: runtime\ndata: ${JSON.stringify(event)}\n\n`);
          }, { after });
          const heartbeat = setInterval(() => response.write(": connected\n\n"), 15000);
          heartbeat.unref();
          response.on("close", () => { clearInterval(heartbeat); unsubscribe(); });
          return;
        }
        if (!match[2] && request.method === "GET") return json(200, { ...session.status(), events: session.history() });
      }
      json(404, { error: "Not found" });
    } catch (error) { json(400, { error: error.message }); }
  });
  server.on("close", () => {
    for (const session of sessions.values()) { session.cancel?.("console closed"); session.close?.(); }
  });
  return { server, sessions };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const option = name => args.includes(name) ? args[args.indexOf(name) + 1] : undefined;
  const { server } = await createAsyncConsole({ replayPath: option("--replay") });
  const port = Number(option("--port") ?? 4318);
  server.listen(port, "127.0.0.1", () => {
    process.stdout.write(`Async systems console: http://127.0.0.1:${server.address().port}\n`);
    if (option("--replay")) process.stdout.write(`Read-only benchmark replay: ${resolve(option("--replay"))}\n`);
  });
  const close = () => { server.close(); server.closeAllConnections(); };
  process.once("SIGINT", close);
  process.once("SIGTERM", close);
}
