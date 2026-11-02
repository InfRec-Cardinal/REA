import { ACTION_TABLE } from "../libero-actions.mjs";

export function createLiberoEnvironment({ client }) {
  let current = null;
  let currentObservation = null;

  function observationFrom(response) {
    const { frame } = response;
    return {
      id: `libero:${frame.libero.observationId}`,
      timestampMs: Date.parse(frame.capturedAt),
      text: frame.libero.task.language,
      state: {
        task: frame.libero.task,
        proprio: frame.libero.proprio,
        controlSteps: response.steps,
        horizon: response.horizon,
      },
      images: [{ mimeType: frame.mimeType, data: frame.imageB64 }],
      actions: Object.values(ACTION_TABLE).map(record => ({
        name: record.token,
        description: record.description,
        parameters: {
          type: "object",
          properties: {
            repeatSteps: { type: "integer", minimum: 1, maximum: response.horizon, default: response.repeatSteps },
            amplitude: { type: "number", exclusiveMinimum: 0, maximum: 1, default: response.amplitude },
          },
          additionalProperties: false,
        },
      })),
    };
  }

  function transitionFrom(response, controlSteps) {
    current = response;
    currentObservation = observationFrom(response);
    const truncated = response.truncated ?? (response.terminated && !response.success && response.steps >= response.horizon);
    return {
      observation: currentObservation,
      terminated: Boolean(response.success || (response.terminated && !truncated)),
      truncated: Boolean(truncated),
      success: response.success,
      reward: response.reward ?? null,
      info: {
        controlSteps,
        totalControlSteps: response.steps,
        horizon: response.horizon,
        repeatSteps: response.repeatSteps,
        amplitude: response.amplitude,
        initialStateSource: response.initialStateSource,
        initialStateIndex: response.initialStateIndex,
        backendTerminated: response.terminated,
      },
    };
  }

  async function reset({ task, seed, signal }) {
    signal?.throwIfAborted();
    const response = await client.request("reset", {
      suite: task.suite,
      task: task.task,
      seed,
      ...(task.horizon === undefined ? {} : { horizon: task.horizon }),
      ...(task.repeatSteps === undefined ? {} : { repeatSteps: task.repeatSteps }),
      ...(task.amplitude === undefined ? {} : { amplitude: task.amplitude }),
    });
    const transition = transitionFrom(response, 0);
    signal?.throwIfAborted();
    return transition;
  }

  async function observe({ signal } = {}) {
    signal?.throwIfAborted();
    const response = await client.request("observe");
    const transition = transitionFrom(response, 0);
    signal?.throwIfAborted();
    return transition.observation;
  }

  async function step(action, { observation = currentObservation, signal } = {}) {
    signal?.throwIfAborted();
    if (!current || observation?.id !== currentObservation.id) throw new Error("LIBERO step requires the current observation");
    if (current.terminated) throw new Error("LIBERO episode has finished");
    if (!ACTION_TABLE[action.name]) throw new Error(`Unknown LIBERO action: ${action.name}`);
    const beforeSteps = current.steps;
    const { repeatSteps, amplitude } = action.arguments ?? {};
    const response = await client.request("step", {
      token: action.name,
      ...(repeatSteps === undefined ? {} : { repeatSteps }),
      ...(amplitude === undefined ? {} : { amplitude }),
    });
    const transition = transitionFrom(response, response.steps - beforeSteps);
    signal?.throwIfAborted();
    return transition;
  }

  async function close() {
    current = null;
    currentObservation = null;
    await client.close();
  }

  return { id: "libero", timing: "turn-based", reset, observe, step, close };
}
