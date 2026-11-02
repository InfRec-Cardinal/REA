import { isAbsolute, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { readFile } from "node:fs/promises";

import { readEnv } from "../env.mjs";
import { validateAction } from "./actions.mjs";
import { createVlmJudge } from "./judge-gate.mjs";
import { createChatPolicy } from "./policy.mjs";
import { createRouteMemoryRouter } from "./route-memory.mjs";

/** Adapt a chat agent to System2: its output advises the router at a later action boundary. */
export function createChatThinker(options = {}) {
  const policy = createChatPolicy({
    ...options,
    mode: options.mode ?? "tool",
    systemPrompt: options.systemPrompt ?? "You are System2, the deliberative brain of an embodied agent. System1 continues routing System0 controllers while you reason. Recommend one available controller and a concise directive using the observation and execution feedback. Your recommendation will be considered at a subsequent action boundary. Account for actions that may already be in progress. Do not assume a task is complete without evidence.",
  });
  return {
    id: `system2:${policy.id}`,
    async think(input) {
      const candidates = input.candidates.map(({ id, kind, description }) => ({ id, kind, description }));
      const action = {
        name: "recommend_controller",
        description: "Advise System1 which System0 controller to use next and why.",
        parameters: {
          type: "object",
          properties: {
            candidateId: { type: "string", enum: candidates.map(candidate => candidate.id) },
            directive: { type: "string" },
          },
          required: ["candidateId", "directive"],
          additionalProperties: false,
        },
      };
      const decision = await policy.decide({
        ...input,
        actions: [action],
        instruction: `${input.instruction ?? "Complete the task."}\nAvailable System0 controllers: ${JSON.stringify(candidates)}`,
      });
      try {
        validateAction(decision.action, [action]);
      } catch (error) {
        error.usage = decision.usage;
        throw error;
      }
      return { ...decision.action.arguments, reason: decision.thought ?? "System2 recommendation", usage: decision.usage, model: decision.model };
    },
  };
}

async function routeTable(table, baseDir) {
  if (typeof table !== "string") return table ?? null;
  return JSON.parse(await readFile(resolve(baseDir, table), "utf8"));
}

/** Built-in roles: route-memory (System 1), judge (System 2 in routed mode), chat (System 2 in async/sync mode); or a module. */
async function loadRole(spec, role, context) {
  if (typeof spec === "function" || typeof spec?.[role] === "function") return spec;
  if (spec.type === "route-memory" && role === "route") {
    const { table, ...options } = spec.options ?? {};
    return createRouteMemoryRouter({ ...options, table: await routeTable(table, context.baseDir) });
  }
  if (spec.type === "judge" && role === "judge") return createVlmJudge({ env: context.env ?? process.env, ...spec.options });
  if (spec.type === "chat" && role === "think") {
    const env = context.env ?? process.env;
    return createChatThinker({
      apiKey: readEnv(env, "REA_API_KEY") ?? "",
      baseUrl: readEnv(env, "REA_BASE_URL"),
      model: readEnv(env, "REA_MODEL"),
      ...spec.options,
    });
  }
  const builtIn = { route: "route-memory", think: "chat", judge: "judge" }[role];
  if (spec.type !== "module" || !spec.options?.module) throw new TypeError(`System role ${role} requires a module or ${builtIn} specification.`);
  const { module, exportName, ...options } = spec.options;
  const name = isAbsolute(module) || module.startsWith(".") ? pathToFileURL(resolve(context.baseDir, module)).href : module;
  const imported = await import(name);
  const factory = imported[exportName ?? "default"];
  if (typeof factory !== "function") throw new TypeError(`${module} must export a system role factory.`);
  return factory(options, context);
}

/** Build the three systems using the existing environment/policy registry. In routed mode System 2 judges at boundaries. */
export async function createConfiguredSystems(spec, context) {
  const { system0: specifications, system1, system2, ...runtime } = spec;
  const brainRole = runtime.mode === "routed" ? "judge" : "think";
  const resources = [];
  async function close() {
    const results = await Promise.allSettled(resources.map(resource => resource.close?.()));
    const failed = results.find(result => result.status === "rejected");
    if (failed) throw failed.reason;
  }
  try {
    const system0 = [];
    for (const candidate of specifications) {
      const policy = await context.registry.createPolicy(candidate.policy, context);
      resources.push(policy);
      system0.push({ ...candidate, policy });
    }
    const router = system1 ? await loadRole(system1, "route", context) : undefined;
    if (router) resources.push(router);
    const brain = system2 ? await loadRole(system2, brainRole, context) : undefined;
    if (brain) resources.push(brain);
    return {
      ...runtime, system0,
      system1: typeof router === "function" ? router : router?.route?.bind(router),
      system2: typeof brain === "function" ? { [brainRole]: brain } : brain,
      close,
    };
  } catch (error) {
    await close();
    throw error;
  }
}
