/**
 * Frozen programs as resumable System 0 capabilities: each candidate runs for `chunkSteps` simulator control steps per
 * `continue_program`, and a different candidate continues from the current physical state. `client` speaks the program
 * backend protocol of scripts/async_libero_backend.py. Success is read only by score(), once, after the episode.
 * `phase` ('development' | 'heldout' | null) and `reset` ('aspire' | 'seeded_random') are passed to the backend, which
 * enforces the seed ranges of its protocol. Use reset 'seeded_random' for development seeds 51-100: under the 'aspire' rule
 * seed s replays official initial state (s - 1) mod 50, the state of an evaluation seed. Each transition reports info.capability (running | returned | failed); the
 * environment is resumable, so runRoutedEpisode can run another candidate after one returns.
 */
export function createLiberoProgramEnvironment({ client, candidateIds, chunkSteps = 100, horizon = 4000, maxDurationMs = 180000, phase = null, reset = 'aspire' }) {
  let resetMetadata;
  let lastRun = null;
  const actions = [{ name: 'continue_program', description: 'Continue this frozen controller for a bounded number of simulator control steps; a different controller starts from the current physical state.', parameters: {
    type: 'object', properties: { candidateId: { type: 'string', enum: candidateIds } }, required: ['candidateId'], additionalProperties: false,
  } }];
  function observation(reply) {
    return { id: reply.observationId, timestampMs: Date.parse(reply.frame.capturedAt), text: reply.task.language,
      state: { proprio: reply.proprio, controlSteps: reply.steps, stepsLeft: reply.stepsLeft,
        activeCandidateId: lastRun?.candidateId ?? null, lastRun },
      images: [{ mimeType: reply.frame.mimeType, data: reply.frame.imageB64 }], actions };
  }
  return {
    id: 'libero-frozen-programs', timing: 'turn-based', resumable: true,
    get resetMetadata() { return resetMetadata; },
    async reset({ task, seed, signal }) {
      signal?.throwIfAborted();
      lastRun = null;
      const reply = await client.request('reset', { suite: task.suite, task: task.task, seed,
        ...(phase ? { phase } : {}), reset, horizon, wall_budget_s: maxDurationMs / 1000 });
      resetMetadata = { instruction: reply.task.language, initialStateSource: reply.initialStateSource,
        initialStateIndex: reply.initialStateIndex, resetSemantics: reply.resetSemantics, gpuAssignment: reply.gpuAssignment };
      return { observation: observation(reply), terminated: false, truncated: false };
    },
    async step(action, { signal } = {}) {
      signal?.throwIfAborted();
      lastRun = await client.request('program_chunk', { candidateId: action.arguments.candidateId,
        controlSteps: chunkSteps, wall_timeout_s: 120 });
      if (lastRun.exception_class === 'infra') throw new Error(`LIBERO perception service failed: ${lastRun.exception}`);
      const reply = await client.request('observe', { judge_frame: false });
      const status = !lastRun.programDone ? 'running' : lastRun.exception ? 'failed' : 'returned';
      return { observation: observation(reply), terminated: lastRun.programDone && !lastRun.exception,
        truncated: lastRun.truncated,
        info: { ...lastRun, capability: { candidateId: lastRun.candidateId, status, ...(lastRun.exception ? { error: String(lastRun.exception) } : {}) } } };
    },
    async stop() { await client.request('stop_program'); },
    async score() {
      const reply = await client.request('score');
      return { ...reply, success: reply.task_completed === true || reply.task_completed === 1 };
    },
  };
}
