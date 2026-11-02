import { isAbsolute, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { defineProfile, isProfile, ProfileError } from "./define.mjs";
import { G1_PROFILE } from "./g1.mjs";
import { LIBERO_PROFILE } from "./libero.mjs";

export { defineProfile, isProfile, ProfileError, PROFILE, ROLE_NAMES } from "./define.mjs";
export { G1_PROFILE, LIBERO_PROFILE };

export const BUILTIN_PROFILES = Object.freeze({ g1: G1_PROFILE, libero: LIBERO_PROFILE });

/** A built-in profile by name; unset selects the G1 profile, an unknown name gives null. */
export function profileByName(name) {
  if (name === undefined || name === null || name === "") return G1_PROFILE;
  return BUILTIN_PROFILES[name] ?? null;
}

/**
 * `G1_ROBOT_PROFILE` resolution: a profile object passes through, `g1`/`libero` pick a built-in, any
 * other string is a module path (relative to `baseDir`) whose default export is a profile spec.
 */
export async function resolveProfile(value, { baseDir = process.cwd() } = {}) {
  if (isProfile(value)) return value;
  const builtin = profileByName(value);
  if (builtin) return builtin;
  if (typeof value !== "string") throw new ProfileError("G1_ROBOT_PROFILE", "must be g1, libero or a module path");
  const specifier = isAbsolute(value) || value.startsWith(".") ? pathToFileURL(resolve(baseDir, value)).href : value;
  const loaded = await import(specifier);
  const exported = loaded.default ?? loaded.profile;
  if (!exported) throw new ProfileError("G1_ROBOT_PROFILE", `${value} must export a profile as its default export`);
  return defineProfile(exported);
}
