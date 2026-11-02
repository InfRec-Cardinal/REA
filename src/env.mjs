import { existsSync } from "node:fs";

export function loadDotEnv(path = process.env.REA_ENV_FILE ?? process.env.FAST_ENV_FILE ?? process.env.G1_ENV_FILE ?? new URL("../.env", import.meta.url)) {
  // An empty path means "no env file": the offline service demo pins its whole environment and passes "".
  if (!path || !existsSync(path)) return false;
  process.loadEnvFile(path);
  return true;
}

/** The public framework variables and the names they had in pre-release versions. */
export const ENV_ALIASES = Object.freeze({
  REA_MODEL: "FAST_MODEL",
  REA_BASE_URL: "FAST_BASE_URL",
  REA_API_KEY: "FAST_API_KEY",
  REA_ENV_FILE: "FAST_ENV_FILE",
  REA_POLICY_MODE: "FAST_POLICY_MODE",
});

/** The REA_* value when set and non-empty, else the deprecated FAST_* value, else undefined. Silent: the CLI prints the notices. */
export function readEnv(env, name) {
  if (env[name] !== undefined && env[name] !== "") return env[name];
  const legacy = ENV_ALIASES[name];
  if (legacy && env[legacy] !== undefined && env[legacy] !== "") return env[legacy];
  return undefined;
}

/** One line per deprecated name in use, for the CLI to print once at startup. */
export function deprecatedEnvNotices(env) {
  return Object.entries(ENV_ALIASES)
    .filter(([name, legacy]) => (env[name] === undefined || env[name] === "") && env[legacy] !== undefined && env[legacy] !== "")
    .map(([name, legacy]) => `${name} is not set; using the deprecated ${legacy} (rename it in .env)`);
}
