// A fixed token sequence as an engine: the offline service demo and CI run the real server, gate and
// executor without a model. Not a policy; it never looks at the frame beyond checking one exists.
import { performance } from "node:perf_hooks";

import { G1_PROFILE } from "./profiles/g1.mjs";
import { InferenceError } from "./provider.mjs";

/**
 * The default token script for a profile: its forward role, or its stop when it has no forward. A
 * literal "W" only exists on the G1, and createScriptedInfer() throws for a token outside the profile.
 */
export function defaultScriptedTokens(profile) {
  return [profile.roles.forward ?? profile.roles.stop];
}

export function createScriptedInfer({ tokens = ["W"], profile = G1_PROFILE, now = () => performance.now() } = {}) {
  const sequence = (Array.isArray(tokens) ? tokens : [tokens]).map(token => profile.parseToken(token));
  if (sequence.length === 0 || sequence.some(token => token === null)) {
    throw new TypeError(`scripted tokens must be from ${profile.TOKENS.join(" ")}; got ${JSON.stringify(tokens)}`);
  }
  let index = 0;
  async function infer(model, frame, { signal } = {}) {
    signal?.throwIfAborted?.();
    if (!frame?.imageB64) throw new InferenceError("current frame is missing", { errorType: "config_error", retryable: false });
    const started = now();
    const token = sequence[Math.min(index, sequence.length - 1)];
    index += 1;
    const elapsedS = Number(((now() - started) / 1000).toFixed(6));
    return { token, tokenReadyS: elapsedS, actionReadyS: elapsedS, totalS: elapsedS, finishReason: "stop", responseStatus: 200, source: "scripted", usage: null, timings: { ttfb_s: elapsedS, token_ready_s: elapsedS, out_chars: 1 } };
  }
  return Object.assign(infer, { engine: "scripted", reset: () => { index = 0; }, warm: async () => ({ ok: true, skipped: true }) });
}
