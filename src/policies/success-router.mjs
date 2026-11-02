export function createSuccessRouter({
  endpoint = "http://127.0.0.1:8810/predict",
  requestTimeoutMs = 30000,
  fetchImpl = fetch,
} = {}, context = {}) {
  let selection = null;

  async function route({ observation, instruction, candidates, episodeId = context.episode?.id, task, seed, step, signal }) {
    signal?.throwIfAborted();
    if (step !== 0) {
      if (!selection) throw new Error("Success router requires the initial observation at step 0");
      return { ...selection, meta: { ...selection.meta, reused: true } };
    }
    selection = null;
    const requestSignal = AbortSignal.timeout(requestTimeoutMs);
    const response = await fetchImpl(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal: signal ? AbortSignal.any([signal, requestSignal]) : requestSignal,
      body: JSON.stringify({
        observation, instruction, episodeId, task, seed, step,
        candidates: candidates.map(({ id, kind }) => ({ id, kind })),
      }),
    });
    if (!response.ok) throw new Error(`Success router prediction failed: HTTP ${response.status}`);
    const prediction = await response.json();
    signal?.throwIfAborted();
    const scores = {};
    for (const { id } of candidates) {
      const score = prediction.scores?.[id];
      if (!prediction.trained_candidates?.includes(id) || !Number.isFinite(score) || score < 0 || score > 1) {
        throw new Error(`Success router has no trained success probability for candidate ${id}`);
      }
      scores[id] = score;
    }
    const chosen = candidates.reduce((best, candidate) => scores[candidate.id] > scores[best.id] ? candidate : best);
    selection = {
      candidateId: chosen.id,
      reason: "Highest predicted success probability on the initial observation",
      model: prediction.model ?? null,
      meta: {
        scope: "episode",
        scores,
        trained_candidates: prediction.trained_candidates,
        initialObservationId: observation?.id ?? null,
        reused: false,
      },
    };
    return selection;
  }

  return { id: "trained-success-router", route };
}

export default createSuccessRouter;
