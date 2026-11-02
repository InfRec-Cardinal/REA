import { validateAction } from "../framework/actions.mjs";

/** Adapt a robot driver without baking its joints or cameras into the runtime. */
export function createRobotEnvironment({ driver, id = "robot", timing = "realtime", execute = false, evaluate = null }) {
  let description;
  let current;
  let task;
  let episode = 0;
  let sequence = 0;
  let closed = false;

  function observation(snapshot) {
    current = {
      id: `${id}:${episode}:${++sequence}`,
      timestampMs: snapshot.timestampMs ?? Date.now(),
      text: snapshot.text ?? (typeof task === "string" ? task : task?.instruction),
      state: { ...snapshot.state, robot: snapshot.metadata ?? description.metadata ?? {} },
      images: snapshot.images ?? [],
      actions: snapshot.actions ?? description.actions,
    };
    return current;
  }

  function transition(snapshot) {
    return {
      observation: observation(snapshot),
      terminated: snapshot.terminated === true,
      truncated: snapshot.truncated === true,
      success: typeof snapshot.success === "boolean" ? snapshot.success : null,
      reward: Number.isFinite(snapshot.reward) ? snapshot.reward : null,
      info: snapshot.info ?? {},
    };
  }

  const environment = {
    id, timing,
    async reset({ task: nextTask = null, seed = 0, signal } = {}) {
      if (closed) throw new Error("Robot environment is closed");
      description ??= await driver.describe({ signal });
      environment.timing = description.timing ?? timing;
      task = nextTask;
      episode += 1;
      sequence = 0;
      return transition(await driver.reset({ task, seed, signal }));
    },
    async observe({ signal } = {}) {
      if (!description || closed) throw new Error("Reset the robot environment before observing");
      return observation(await driver.observe({ signal }));
    },
    async step(action, { observation: basedOn = current, signal, reasoning } = {}) {
      if (closed || !current) throw new Error("Reset the robot environment before stepping");
      if (basedOn?.id !== current.id) throw new Error("Robot action requires the current observation");
      const normalized = validateAction(action, current.actions);
      const before = current;
      const snapshot = await driver.step(normalized, { execute, signal });
      const next = transition(snapshot);
      if (evaluate) {
        const outcome = await evaluate({ task, action: normalized, before, observation: next.observation, execution: snapshot.info, reasoning, signal });
        if (outcome) {
          for (const key of ["terminated", "truncated", "success", "reward"]) if (outcome[key] !== undefined) next[key] = outcome[key];
          next.info = { ...next.info, ...outcome.info };
        }
      }
      return next;
    },
    async stop() {
      if (!description || closed || !execute) return { supported: false, preview: !execute };
      return driver.stop?.() ?? { supported: false };
    },
    async close() {
      if (closed) return;
      closed = true;
      current = null;
      await driver.close?.();
    },
  };
  return environment;
}
