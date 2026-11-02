import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { runEpisode } from "./runtime.mjs";
import { runAsyncEpisode } from "./async-runtime.mjs";
import { runRoutedEpisode } from "./routed-runtime.mjs";

async function finalizeEnvironment(environment) {
  const result = { officialScorecard: null };
  try {
    if (environment?.scorecard) result.officialScorecard = await environment.scorecard({ close: true });
  } catch (error) {
    result.scorecardError = error.message;
  }
  try {
    await environment?.close?.();
  } catch (error) {
    result.closeError = error.message;
  }
  return result;
}

export async function runEvaluation({ episodes, createEnvironment, createPolicy, createSystems, outputDir, budget = {}, metadata = {}, recordImages = false, reuseEnvironment = false, onProgress = () => {}, onEvent = () => {}, signal }) {
  await mkdir(outputDir, { recursive: true });
  // Per-episode policy or environment options can contain service credentials.
  const manifestReplacer = (key, value) => /^(apiKey|password|authorization|accessToken)$/iu.test(key) ? undefined : value;
  await writeFile(join(outputDir, "manifest.json"), `${JSON.stringify({ createdAt: new Date().toISOString(), episodes, budget, metadata, recordImages, reuseEnvironment }, manifestReplacer, 2)}\n`, { flag: "wx" });
  const results = [];
  let sharedEnvironment;
  let sharedFinalization = { officialScorecard: null };
  try {
    for (const [index, episode] of episodes.entries()) {
      if (signal?.aborted) break;
      const prefix = String(index + 1).padStart(4, "0");
      const tracePath = join(outputDir, `${prefix}.jsonl`);
      await writeFile(tracePath, "", { flag: "wx" });
      let environment;
      let policy;
      let systems;
      let eventWrites = Promise.resolve();
      let summary;
      let finalization;
      try {
        environment = sharedEnvironment ?? await createEnvironment(episode);
        if (reuseEnvironment) sharedEnvironment = environment;
        if (createSystems) systems = await createSystems(episode, environment);
        else policy = await createPolicy(episode, environment);
        const eventPath = join(outputDir, `${prefix}.events.jsonl`);
        if (systems) await writeFile(eventPath, "", { flag: "wx" });
        summary = await (!systems ? runEpisode : systems.mode === "routed" ? runRoutedEpisode : runAsyncEpisode)({
          ...systems,
          ...budget,
          ...episode.budget,
          environment,
          policy,
          task: episode.task,
          seed: episode.seed ?? 0,
          instruction: episode.instruction,
          signal,
          onEvent: event => {
            // Record asynchronously so disk I/O never delays System1 routing.
            const line = `${JSON.stringify(event)}\n`;
            eventWrites = eventWrites.then(() => appendFile(eventPath, line));
            eventWrites.catch(() => {});
            onEvent({ ...event, episodeId: episode.id ?? prefix });
          },
          onTransition: record => appendFile(tracePath, `${JSON.stringify(record, (key, value) => key === "images" && !recordImages ? undefined : value)}\n`),
        });
        await eventWrites;
      } catch (error) {
        summary = { task: episode.task, seed: episode.seed ?? 0, reason: signal?.aborted ? "cancelled" : "error", steps: 0, success: null, error: { code: error.code ?? "setup_error", message: error.message } };
      } finally {
        try {
          await (systems ?? policy)?.close?.();
        } catch (error) {
          summary.policyCloseError = error.message;
          summary.error ??= { code: "policy_close_error", message: error.message };
          summary.reason = "error";
          summary.success = null;
        }
        if (!reuseEnvironment) finalization = await finalizeEnvironment(environment);
      }
      if (finalization?.closeError) {
        summary.error ??= { code: "close_error", message: finalization.closeError };
        summary.reason = "error";
        summary.success = null;
      }
      const result = { id: episode.id ?? prefix, ...summary, ...finalization };
      await writeFile(join(outputDir, `${prefix}.json`), `${JSON.stringify(result, null, 2)}\n`, { flag: "wx" });
      results.push(result);
      await onProgress(result);
      if (signal?.aborted || result.reason === "cancelled" || (reuseEnvironment && result.error)) break;
    }
  } finally {
    if (reuseEnvironment) sharedFinalization = await finalizeEnvironment(sharedEnvironment);
  }
  const completed = results.filter(result => !result.error && result.reason !== "cancelled");
  const judged = completed.filter(result => typeof result.success === "boolean");
  const scorecardErrors = results.filter(result => result.scorecardError).length + Number(Boolean(sharedFinalization.scorecardError));
  const closeErrors = results.filter(result => result.closeError || result.policyCloseError).length + Number(Boolean(sharedFinalization.closeError));
  const summary = {
    episodesPlanned: episodes.length,
    episodesRun: results.length,
    completed: completed.length,
    successes: judged.filter(result => result.success).length,
    failures: judged.filter(result => !result.success).length,
    unscored: completed.length - judged.length,
    errors: results.filter(result => result.error).length,
    scorecardErrors,
    closeErrors,
    successRate: scorecardErrors === 0 && closeErrors === 0 && judged.length > 0 && judged.length === episodes.length ? judged.filter(result => result.success).length / judged.length : null,
    results: results.map((result, index) => ({ id: result.id, file: `${String(index + 1).padStart(4, "0")}.json`, reason: result.reason, success: result.success })),
    ...(reuseEnvironment ? sharedFinalization : {}),
  };
  await writeFile(join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, { flag: "wx" });
  return summary;
}
