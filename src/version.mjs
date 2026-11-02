// The one place the service version lives: package.json. Every health/capabilities/log field reads it
// from here, so a release bump is one edit.
import { readFileSync } from "node:fs";

const manifest = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));

export const SERVICE = "rea-agent";
export const VERSION = manifest.version;
