import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import test from "node:test";

const execFileAsync = promisify(execFile);

async function fixture(context) {
  const directory = await mkdtemp(join(tmpdir(), "fast-setup-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  await mkdir(join(directory, "scripts"));
  await mkdir(join(directory, "bin"));
  await copyFile(new URL("../scripts/setup-prime.sh", import.meta.url), join(directory, "scripts", "setup-prime.sh"));
  await copyFile(new URL("../scripts/setup.sh", import.meta.url), join(directory, "scripts", "setup.sh"));
  const node = join(directory, "bin", "node");
  const npm = join(directory, "bin", "npm");
  await writeFile(node, `#!/usr/bin/env bash
if [ "$1" = "-e" ]; then
  "$REA_TEST_REAL_NODE" -e 'Object.defineProperty(process.versions, "node", { value: process.env.REA_TEST_NODE_VERSION }); eval(process.argv[1])' "$2"
else
  printf 'v%s\\n' "$REA_TEST_NODE_VERSION"
fi
`, { mode: 0o755 });
  await writeFile(npm, `#!${process.execPath}
const fs = require("node:fs");
const path = require("node:path");
if (process.env.REA_TEST_NPM_FAIL) process.exit(7);
fs.appendFileSync(process.env.REA_TEST_INSTALL_LOG, JSON.stringify(process.argv.slice(2)) + "\\n");
const binary = path.join(process.argv[3], "node_modules", ".bin", "prime-agent");
fs.mkdirSync(path.dirname(binary), { recursive: true });
fs.writeFileSync(binary, "#!/usr/bin/env bash\\nprintf 'proxy warning\\n0.8.0\\\\n' >&2\\nif [ -n \\\"$REA_TEST_PRIME_FAIL\\\" ]; then exit 9; fi\\n", { mode: 0o755 });
`, { mode: 0o755 });
  const env = {
    ...process.env,
    REA_NODE: node,
    REA_NPM: npm,
    REA_TEST_REAL_NODE: process.execPath,
    REA_TEST_NODE_VERSION: "22.19.0",
    REA_TEST_INSTALL_LOG: join(directory, "install.jsonl"),
  };
  return {
    directory,
    env,
    run: (overrides = {}, script = "setup-prime.sh") => execFileAsync("bash", [join(directory, "scripts", script)], { env: { ...env, ...overrides } }),
  };
}

test("bootstrap installs the official release, accepts stderr version, and reuses the runtime", async context => {
  const setup = await fixture(context);
  const installed = await setup.run();
  assert.match(installed.stdout, /prime-agent: 0\.8\.0/u);
  const repeated = await setup.run();
  assert.match(repeated.stdout, /prime-agent: 0\.8\.0/u);
  const commands = (await readFile(setup.env.REA_TEST_INSTALL_LOG, "utf8")).trim().split("\n").map(JSON.parse);
  assert.deepEqual(commands, [["--prefix", join(setup.directory, "runtime", "prime-agent"), "install", "--no-save", "--package-lock=false", "--omit=dev", "https://github.com/PrimeIntellect-ai/prime-agent/releases/download/v0.8.0/prime-agent-0.8.0.tgz"]]);
});

test("setup rejects Node older than 22.19 before installing", async context => {
  const setup = await fixture(context);
  await assert.rejects(setup.run({ REA_TEST_NODE_VERSION: "22.18.0" }, "setup.sh"), error => {
    assert.match(error.stdout, /Node >= 22\.19\.0 is required; found v22\.18\.0/u);
    return true;
  });
  await assert.rejects(readFile(setup.env.REA_TEST_INSTALL_LOG), { code: "ENOENT" });
});

test("bootstrap accepts a local package and propagates npm installation failures", async context => {
  const setup = await fixture(context);
  const packagePath = join(setup.directory, "prime-agent-0.8.0.tgz");
  await assert.rejects(setup.run({ PRIME_AGENT_PACKAGE: packagePath, REA_TEST_NPM_FAIL: "1" }), { code: 7 });
  await setup.run({ PRIME_AGENT_PACKAGE: packagePath });
  const command = JSON.parse((await readFile(setup.env.REA_TEST_INSTALL_LOG, "utf8")).trim());
  assert.equal(command.at(-1), packagePath);
});

test("bootstrap preserves a nonzero Prime version status despite a matching version line", async context => {
  const setup = await fixture(context);
  await assert.rejects(setup.run({ REA_TEST_PRIME_FAIL: "1" }), error => {
    assert.equal(error.code, 1);
    assert.match(error.stdout, /Prime Agent 0\.8\.0 required; found 0\.8\.0/u);
    return true;
  });
});
