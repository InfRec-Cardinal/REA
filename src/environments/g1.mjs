import { setTimeout as sleep } from "node:timers/promises";

import { G1_PROFILE } from "../profiles/g1.mjs";

export function createG1Environment({ camera, executor, config, execute = false, evaluate = null, waitForCompletion = null, profile = G1_PROFILE }) {
  const maximumDurationMs = Math.floor((config.maxPulseS ?? 2) * 1000);
  const actions = Object.values(profile.ACTION_TABLE).map(record => ({
    name: record.token,
    description: record.kind === "velocity"
      ? `${record.name}: vx=${record.vx} m/s, vy=${record.vy} m/s, yaw=${record.yaw} rad/s`
      : record.name,
    parameters: {
      type: "object",
      properties: record.kind === "velocity" ? {
        durationMs: { type: "integer", minimum: 400, maximum: maximumDurationMs, default: Math.min(record.durationS * 1000, maximumDurationMs) },
      } : {},
      additionalProperties: false,
    },
  }));
  let task = null;
  let episode = 0;
  let sequence = 0;
  let currentObservation = null;
  let currentFrame = null;
  let totalActionsAccepted = 0;

  async function observe({ signal } = {}) {
    signal?.throwIfAborted();
    const frame = await camera.capture({ ...config, needDepth: true }, { signal });
    const evidence = await executor.evidence({ frame, useLidar: config.useLidar });
    signal?.throwIfAborted();
    currentFrame = frame;
    currentObservation = {
      id: `g1:${episode}:${++sequence}`,
      timestampMs: frame.capturedAt == null ? Date.now() : frame.capturedAt * 1000,
      text: typeof task === "string" ? task : task?.instruction,
      state: { robot: evidence.status, depth: evidence.depth, lidar: evidence.lidar, armHeld: executor.armState?.().held ?? null },
      images: frame.imageB64 ? [{ mimeType: frame.mimeType ?? "image/jpeg", data: frame.imageB64 }] : [],
      actions,
    };
    return currentObservation;
  }

  async function reset({ task: nextTask, signal } = {}) {
    task = nextTask;
    episode += 1;
    sequence = 0;
    totalActionsAccepted = 0;
    currentObservation = null;
    currentFrame = null;
    return {
      observation: await observe({ signal }),
      terminated: false,
      truncated: false,
      success: null,
      reward: null,
      info: { physicalReset: false, actionsAccepted: 0, totalActionsAccepted },
    };
  }

  async function step(action, { observation = currentObservation, signal, reasoning } = {}) {
    signal?.throwIfAborted();
    if (!currentObservation || observation?.id !== currentObservation.id) throw new Error("G1 step requires the current observation");
    const record = profile.ACTION_TABLE[action.name];
    if (!record) throw new Error(`Unknown G1 action: ${action.name}`);
    const durationMs = action.arguments?.durationMs;
    if (durationMs !== undefined && (record.kind !== "velocity" || !Number.isInteger(durationMs) || durationMs < 400 || durationMs > maximumDurationMs)) {
      throw new Error(`G1 durationMs must be an integer between 400 and ${maximumDurationMs} for a velocity action`);
    }
    const before = observation;
    const liveDepth = await camera.capture({ ...config, needDepth: true, depthOnly: true }, { signal });
    signal?.throwIfAborted();
    const execution = await executor.execute({
      token: record.token,
      durationMs: durationMs ?? null,
      frame: { ...currentFrame, depth: liveDepth.depth, capturedAt: liveDepth.capturedAt },
      execute,
      useLidar: config.useLidar,
    });
    const actionsAccepted = execution.executed ? 1 : 0;
    totalActionsAccepted += actionsAccepted;
    let completion = execution.preview ? "preview" : "not_executed";
    try {
      signal?.throwIfAborted();
      if (execution.executed) {
        if (waitForCompletion) {
          await waitForCompletion({ action, execution, signal });
          completion = "completion_hook";
        } else if (execution.dispatch?.kind === "EXECUTION_COMPLETE") {
          completion = "execution_complete";
        } else if (record.kind === "velocity") {
          await sleep(execution.motion.duration * 1000 + 50, undefined, { signal });
          completion = "pulse_elapsed";
        } else {
          completion = "acknowledged";
        }
      }
      const after = await observe({ signal });
      const outcome = evaluate ? await evaluate({ task, action, before, observation: after, execution, reasoning, signal }) : null;
      return {
        observation: after,
        terminated: outcome?.terminated ?? false,
        truncated: outcome?.truncated ?? false,
        success: outcome?.success ?? null,
        reward: outcome?.reward ?? null,
        info: { ...outcome?.info, execution, completion, actionsAccepted, totalActionsAccepted },
      };
    } catch (error) {
      if (execution.executed && signal?.aborted) await stop();
      throw error;
    }
  }

  async function stop() {
    if (execute && config.executeEnabled) return executor.stop();
    return { executed: false, preview: true };
  }

  async function close() {
    currentObservation = null;
    currentFrame = null;
    await Promise.all([camera.close?.(), executor.close?.()]);
  }

  return { id: "g1", timing: "realtime", reset, observe, step, stop, close };
}
