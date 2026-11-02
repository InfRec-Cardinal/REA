import assert from "node:assert/strict";
import { execFile, spawn } from "node:child_process";
import { once } from "node:events";
import { cp, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import test from "node:test";

const execFileAsync = promisify(execFile);
const envModule = new URL("../src/env.mjs", import.meta.url).href;

test("dotenv loads file values and preserves the process environment", async context => {
  const directory = await mkdtemp(join(tmpdir(), "fast-env-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const envPath = join(directory, ".env");
  await writeFile(envPath, "REA_ENV_TEST_FROM_FILE=file-value\nREA_ENV_TEST_OVERRIDE=file-value\n");
  const result = await execFileAsync(process.execPath, ["--input-type=module", "-e", `
    import { loadDotEnv } from ${JSON.stringify(envModule)};
    const loaded = loadDotEnv();
    process.stdout.write(JSON.stringify({ loaded, file: process.env.REA_ENV_TEST_FROM_FILE, override: process.env.REA_ENV_TEST_OVERRIDE }));
  `], { env: { ...process.env, G1_ENV_FILE: envPath, REA_ENV_TEST_OVERRIDE: "process-value" } });
  assert.deepEqual(JSON.parse(result.stdout), { loaded: true, file: "file-value", override: "process-value" });
});

for (const entry of ["server.mjs", "thinker.mjs"]) {
  test(`${entry} loads its repository dotenv when started from another directory`, { timeout: 10000 }, async context => {
    const directory = await realpath(await mkdtemp(join(tmpdir(), "fast-startup-")));
    context.after(() => rm(directory, { recursive: true, force: true }));
    const checkout = join(directory, "checkout");
    await cp(new URL("../src", import.meta.url), join(checkout, "src"), { recursive: true });
    await cp(new URL("../package.json", import.meta.url), join(checkout, "package.json"));
    await cp(new URL(`../${entry}`, import.meta.url), join(checkout, entry));
    const reservation = createServer();
    await new Promise(resolve => reservation.listen(0, "127.0.0.1", resolve));
    const port = reservation.address().port;
    await new Promise(resolve => reservation.close(resolve));
    await writeFile(join(checkout, ".env"), [
      "REA_API_KEY=local-startup-fixture",
      "REA_BASE_URL=http://127.0.0.1:1/v1",
      "REA_MODEL=local-startup-model",
      "G1_AGENT_HOST=127.0.0.1",
      `G1_AGENT_PORT=${port}`,
      `G1_THINKER_PORT=${port}`,
      "G1_ENGINE=token",
      "G1_PERSISTENT_CAPTURE=0",
      "G1_PERSISTENT_CONTROL=0",
      "G1_EXECUTE_ENABLED=0",
      "G1_ACTUATOR=none",
      "G1_BRAIN_ENABLED=0",
      "G1_THINKER_ENABLED=0",
      "G1_WARM_INTERVAL_S=0",
    ].join("\n"));
    const child = spawn(process.execPath, [join(checkout, entry)], {
      cwd: directory,
      env: { PATH: process.env.PATH, HOME: directory },
      stdio: ["ignore", "pipe", "pipe"],
    });
    context.after(async () => {
      if (child.exitCode === null && child.signalCode === null) {
        const exited = once(child, "exit");
        child.kill("SIGTERM");
        await exited;
      }
    });
    const started = Promise.withResolvers();
    let output = "";
    child.stdout.on("data", chunk => {
      output += chunk;
      if (output.includes(entry === "server.mjs" ? '"event":"service_started"' : '"event":"thinker_started"')) started.resolve();
    });
    child.stderr.on("data", chunk => { output += chunk; });
    child.once("error", started.reject);
    child.once("exit", code => started.reject(new Error(`${entry} exited ${code}: ${output}`)));
    await started.promise;
    const health = await (await fetch(`http://127.0.0.1:${port}/health`)).json();
    assert.equal(health.service, entry === "server.mjs" ? "rea-agent" : "rea-thinker");
    if (entry === "server.mjs") {
      assert.equal(health.provider_configured, true);
      assert.equal(health.engine, "token");
    } else {
      assert.equal(health.ready, true);
    }
  });
}
