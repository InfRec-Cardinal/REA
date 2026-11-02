import { readEnv } from "../env.mjs";
import { validateAction } from "../framework/actions.mjs";
import { createChatPolicy } from "../framework/policy.mjs";

/** Select a controller from the current observation at each action boundary. */
export function createAlohaRouter({ generator = null, ...options } = {}, { env = process.env } = {}) {
  const policy = generator ?? createChatPolicy({
    model: env.REA_ROUTER_MODEL ?? readEnv(env, "REA_MODEL"),
    baseUrl: readEnv(env, "REA_BASE_URL"),
    apiKey: readEnv(env, "REA_API_KEY") ?? "",
    maxOutputTokens: 1024,
    systemPrompt: "You route controllers for a Classic ALOHA bimanual robot. Select exactly one supplied controller using the task, camera images, measured joints, controller descriptions and previous execution feedback. VLA controllers return learned absolute joint trajectories; the code controller generates Python to calculate explicit joint targets from observations. Use the configured capabilities to choose; do not invent model success probabilities. Continuous System0 execution takes priority: retain the current controller while it makes progress. Recommend a switch only for a concrete observed failure, lack of progress, or task phase better served by another controller; explain that evidence briefly. Your choice only selects a controller; it does not itself command the robot.",
    ...options,
  });
  return {
    id: "aloha-controller-router",
    async route(input) {
      const candidates = input.candidates.map(({ id, kind, description }) => ({ id, kind, description }));
      const action = {
        name: "select_controller",
        description: "Select the controller that should produce the next ALOHA action block.",
        parameters: {
          type: "object",
          properties: { candidateId: { type: "string", enum: candidates.map(candidate => candidate.id) } },
          required: ["candidateId"], additionalProperties: false,
        },
      };
      const decision = await policy.decide({
        ...input, actions: [action],
        instruction: `${input.instruction ?? input.observation?.text ?? "Complete the task."}\nAvailable controllers: ${JSON.stringify(candidates)}\nPrevious controller: ${input.previousCandidateId ?? "none"}.`,
      });
      validateAction(decision.action, [action]);
      return {
        candidateId: decision.action.arguments.candidateId,
        reason: decision.thought ?? "Selected from the current observation and execution feedback",
        model: decision.model, usage: decision.usage,
        meta: { scope: "action_boundary", observationId: input.observation?.id ?? null },
      };
    },
    async close() { await policy.close?.(); },
  };
}

/** Run the visual selector in the background; the runtime applies its choice at a later boundary. */
export function createAlohaRouterThinker(options = {}, context = {}) {
  const router = createAlohaRouter(options, context);
  return {
    id: "aloha-background-router",
    async think(input) {
      const selection = await router.route({
        ...input,
        instruction: `${input.instruction ?? input.observation?.text ?? "Complete the task."}\nThe current controller continues acting while you select. Your choice will be considered at a later action boundary, using a fresh observation to generate that controller's action. Account for progress since the supplied observation; select a controller, not joint targets.`,
      });
      return { ...selection, meta: { ...selection.meta, scope: "background", sourceStep: input.step } };
    },
    close: () => router.close(),
  };
}

export default createAlohaRouter;
