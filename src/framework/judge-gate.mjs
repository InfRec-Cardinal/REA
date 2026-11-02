import { readEnv } from "../env.mjs";
import { DEFAULT_JUDGE_THRESHOLD, createJudgeInfer, judgeVerdict } from "../infer-judge.mjs";

// System 2 at capability boundaries: a vision-language judge that answers whether the instruction is complete from the
// current image (and, optionally, the image at reset). It is called only between capability runs, never while a
// capability runs, and it never sees simulator success.

function frameOf(observation, name) {
  const images = observation?.images ?? [];
  const image = name ? images.find(item => item.name === name) : images[0];
  return image?.data ? { imageB64: image.data, mimeType: image.mimeType ?? "image/jpeg" } : null;
}

/**
 * createVlmJudge(options) -> { id, judge({ observation, initialObservation, instruction, signal }) }
 * judge() resolves to { verdict: "done" | "not_done" | "error", done, confidence, rationale, model, usage, latencyS };
 * "done" needs done=true with confidence >= threshold. Provider failures resolve to verdict "error" so System 1 decides.
 * Options: model (default REA_JUDGE_MODEL, then REA_MODEL), apiKey / baseUrl (default REA_API_KEY / REA_BASE_URL),
 * threshold, image (observation image name; default the first), view (agentview | montage | front), style
 * (default | evidence | pickup), withInitial (send the reset image too), maxTokens, deadlineMs, infer (inject a judge).
 */
export function createVlmJudge({
  model, apiKey, baseUrl, threshold = DEFAULT_JUDGE_THRESHOLD, image = null, view = "agentview", style = "default",
  withInitial = true, maxTokens, deadlineMs, env = process.env, infer = null,
} = {}) {
  const resolvedModel = model ?? env.REA_JUDGE_MODEL ?? readEnv(env, "REA_MODEL");
  const judge = infer ?? createJudgeInfer({
    model: resolvedModel,
    apiKey: apiKey ?? readEnv(env, "REA_API_KEY") ?? "",
    baseUrl: baseUrl ?? readEnv(env, "REA_BASE_URL"),
    ...(maxTokens === undefined ? {} : { maxTokens }),
    ...(deadlineMs === undefined ? {} : { deadlineMs }),
  });
  return {
    id: `judge:${resolvedModel ?? "custom"}`,
    async judge({ observation, initialObservation = null, instruction, signal } = {}) {
      const frame = frameOf(observation, image);
      if (!frame) return { verdict: "error", error: "the observation has no image for the judge" };
      try {
        const result = await judge(frame, {
          instruction, view, style, signal,
          initialFrame: withInitial ? frameOf(initialObservation, image) : null,
        });
        return {
          verdict: judgeVerdict(result, { threshold }), done: result.done, confidence: result.confidence,
          rationale: result.rationale ?? null, model: result.model ?? resolvedModel ?? null,
          usage: result.usage ?? null, latencyS: result.latencyS ?? null,
        };
      } catch (error) {
        if (signal?.aborted) throw error;
        return { verdict: "error", error: error?.message ?? String(error), usage: error?.usage ?? null };
      }
    },
  };
}
