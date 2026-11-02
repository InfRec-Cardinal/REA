import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { access, mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import test from "node:test";

import { loadConfig } from "../src/config.mjs";
import { createPrimeAgentInfer, withNodeOnPath } from "../src/infer-prime.mjs";

const execFileAsync = promisify(execFile);

// This is the only test that exercises the real Prime RPC binary, so it must not be silently
// skipped just because PRIME_AGENT_BIN is unset in the shell: the binary lives at a known
// repo-relative path and loadConfig already resolves it. The provider is a local mock, so the
// test needs no network and no API key -- only the binary has to be present.
const PRIME_BIN = process.env.PRIME_AGENT_BIN ?? loadConfig({}).primeAgentBin;
const PRIME_BIN_MISSING = !PRIME_BIN || !existsSync(PRIME_BIN);

async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  return server.address().port;
}

test("official Prime 0.8.0 RPC accepts a base64 image and returns one bounded token", {
  skip: PRIME_BIN_MISSING ? `prime-agent binary not installed at ${PRIME_BIN}` : false,
  timeout: 30_000,
}, async t => {
  let observedPayload = null;
  const mock = createServer(async (request, response) => {
    if (request.method === "POST" && request.url === "/v1/chat/completions") {
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      observedPayload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      response.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-store",
        connection: "keep-alive",
      });
      response.write(`data: ${JSON.stringify({
        id: "mock",
        object: "chat.completion.chunk",
        created: 1,
        model: "mock-vision",
        choices: [{ index: 0, delta: { role: "assistant", content: "X" }, finish_reason: null }],
      })}\n\n`);
      response.write(`data: ${JSON.stringify({
        id: "mock",
        object: "chat.completion.chunk",
        created: 1,
        model: "mock-vision",
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      })}\n\n`);
      response.end("data: [DONE]\n\n");
      return;
    }
    response.writeHead(404).end();
  });
  const port = await listen(mock);

  const configDir = await mkdtemp(join(tmpdir(), "fast-prime-smoke-"));
  await writeFile(join(configDir, "auth.json"), "{}\n");
  await writeFile(join(configDir, "settings.json"), `${JSON.stringify({
    compaction: { enabled: false },
    retry: {
      enabled: false,
      maxRetries: 0,
      provider: { timeoutMs: 10_000, maxRetries: 0 },
    },
  }, null, 2)}\n`);
  await writeFile(join(configDir, "models.json"), `${JSON.stringify({
    providers: {
      mock: {
        baseUrl: `http://127.0.0.1:${port}/v1`,
        api: "openai-completions",
        apiKey: "MOCK_API_KEY",
        compat: {
          supportsDeveloperRole: false,
          supportsReasoningEffort: false,
          supportsStore: false,
          supportsStrictMode: false,
          supportsUsageInStreaming: true,
          maxTokensField: "max_tokens",
        },
        models: [{
          id: "mock-vision",
          name: "Mock Vision",
          reasoning: false,
          input: ["text", "image"],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: 8192,
          maxTokens: 1,
        }],
      },
    },
  }, null, 2)}\n`);

  const infer = createPrimeAgentInfer({
    bin: PRIME_BIN,
    configDir,
    provider: "mock",
    model: "mock-vision",
    cwd: process.cwd(),
    env: { ...process.env, MOCK_API_KEY: "test-only" },
    startupTimeoutMs: 10_000,
  });
  t.after(async () => {
    await infer.close();
    await new Promise(resolve => mock.close(resolve));
    // Prime may finish an asynchronous daemon-worker bookkeeping write immediately after exit.
    await new Promise(resolve => setTimeout(resolve, 150));
    await rm(configDir, { recursive: true, force: true });
  });

  await infer.start();
  const result = await infer("mock-vision", {
    imageB64: Buffer.from("synthetic-jpeg").toString("base64"),
    mimeType: "image/jpeg",
    depth: { center_m: 1.5 },
  }, {
    instruction: "Return the safe stop token.",
    deadlineMs: 10_000,
  });

  assert.equal(result.token, "X");
  assert.equal(observedPayload.max_tokens, 1);
  assert.equal(observedPayload.tools, undefined);
  const imagePart = observedPayload.messages
    .flatMap(message => Array.isArray(message.content) ? message.content : [])
    .find(part => part.type === "image_url");
  assert.match(imagePart.image_url.url, /^data:image\/jpeg;base64,/u);

  const beforeClose = await execFileAsync(PRIME_BIN, ["status", "--json"], {
    cwd: process.cwd(),
    env: { ...withNodeOnPath(process.env), PRIME_AGENT_CODING_AGENT_DIR: configDir },
  });
  assert.ok(JSON.parse(beforeClose.stdout).some(record => record.socketPath === infer.daemonSocket));
  await infer.close();
  assert.equal(infer.state().running, false);
  await assert.rejects(access(infer.daemonSocket));
  const status = await execFileAsync(PRIME_BIN, ["status", "--json"], {
    cwd: process.cwd(),
    env: { ...withNodeOnPath(process.env), PRIME_AGENT_CODING_AGENT_DIR: configDir },
  });
  const daemons = JSON.parse(status.stdout);
  assert.equal(daemons.some(record => record.socketPath === infer.daemonSocket), false);
});
