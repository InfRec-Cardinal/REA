import { createServer } from "node:http";
import { pathToFileURL } from "node:url";

import { createAgent } from "../src/agent.mjs";
import { loadConfig } from "../src/config.mjs";
import { createExecutor, createNullActuator } from "../src/executor.mjs";
import { createHttpServer } from "../src/http.mjs";
import { createTokenInfer } from "../src/infer-token.mjs";
import { createSessionStore } from "../src/sessions.mjs";

const PIXEL = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a90sAAAAASUVORK5CYII=";
const SCENARIOS = [
  { name: "clear path", center: 3 },
  { name: "obstacle ahead", center: 0.3 },
];
const logger = { info() {}, warn() {}, error() {} };

async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  return `http://127.0.0.1:${server.address().port}`;
}

async function close(server) {
  await new Promise(resolve => {
    server.close(resolve);
    server.closeAllConnections();
  });
}

export async function runOfflineDemo() {
  const provider = createServer(async (request, response) => {
    for await (const chunk of request) void chunk;
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.end(`data: ${JSON.stringify({ choices: [{ delta: { content: "W" }, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`);
  });
  let server;
  try {
    const providerUrl = await listen(provider);
    const config = {
      ...loadConfig({
        REA_API_KEY: "offline-demo",
        REA_BASE_URL: `${providerUrl}/v1`,
        G1_ENGINE: "token",
        G1_PRIMARY_MODEL: "offline-fixed-forward",
        G1_AGENT_HOST: "127.0.0.1",
        G1_USE_LIDAR: "0",
        G1_EXECUTE_ENABLED: "0",
        G1_THINKER_ENABLED: "0",
        G1_BRAIN_ENABLED: "0",
      }),
      cameraSocket: null,
    };
    let scenario = SCENARIOS[0];
    const camera = {
      async capture() {
        return { imageB64: PIXEL, mimeType: "image/png", width: 1, height: 1, capturedAt: Date.now() / 1000, depth: { center_m: scenario.center, left_m: 3, right_m: 3 } };
      },
    };
    const actuator = {
      ...createNullActuator(),
      async status() { return { ok: true, ready: true, reasons: [], source: "offline-fixture" }; },
    };
    const agent = createAgent({
      camera,
      engines: { token: createTokenInfer({ apiKey: config.apiKey, baseUrl: config.baseUrl }) },
      executor: createExecutor({ actuator, config, logger }),
      config,
      sessions: createSessionStore(),
      logger,
    });
    server = createHttpServer({ agent, config, logger });
    const baseUrl = await listen(server);
    const results = [];
    let sessionId;
    for (scenario of SCENARIOS) {
      const response = await fetch(`${baseUrl}/v1/step`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, instruction: "Approach the doorway.", execute: false }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error ?? `offline step returned ${response.status}`);
      sessionId = result.session_id;
      results.push({ scenario: scenario.name, seq: result.seq, token: result.token, gate_allowed: result.gate.allowed, reasons: result.gate.reasons, preview: result.preview, executed: result.executed });
    }
    return results;
  } finally {
    if (server) await close(server);
    await close(provider);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  runOfflineDemo().then(results => {
    process.stdout.write(`${JSON.stringify({ mode: "offline", model: "fixed W response from a local fixture", results }, null, 2)}\n`);
  }).catch(error => {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  });
}
