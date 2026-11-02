import { configVersions, loadLoopConfig } from "./loop/config.mjs";
import { armEvidence, executionOf } from "./loop/evidence.mjs";
import { screenAction } from "./loop/gate.mjs";
import { judgeCompletion } from "./loop/judge.mjs";
import { stalledSteps } from "./loop/progress.mjs";
import { planRecovery } from "./loop/recover.mjs";
import { createProgressPolicy } from "./progress.mjs";

/** Default stall test: the last `patience` transitions were refused, or unrewarded with exactly unchanged state. */
function exactlyStalled(history, patience) {
  const recent = history.slice(-patience);
  return recent.length >= patience && recent.every(record => (
    record.info?.gate?.allowed === false ||
    record.info?.execution?.gate?.allowed === false ||
    (record.success !== true && !(record.reward > 0) && JSON.stringify(record.observation?.state) === JSON.stringify(record.nextObservation?.state))
  ));
}

/**
 * Route between a fast reflex policy and a slower deliberate policy.
 * Without `loop`, a stall is `patience` refused or exactly unchanged transitions; `stallTolerance` (true or an override of the
 * loop `progress` node) counts transitions that moved no more than sensor jitter instead.
 * With `loop` set (true or a config override), each decision also passes an evidence loop:
 * a pre-execution gate, a rule judge for completion claims, and recovery macros.
 * `progress` wraps the router in the progress review layer; a required review asks for the deliberate policy.
 */
export function createRoutedPolicy({ reflex, deliberate, patience = 3, maxDeliberateFraction = 0.5, shouldDeliberate = null, loop = null, stallTolerance = null, progress = false, ...progressOptions }) {
  const nodes = loop ? loadLoopConfig(loop === true ? {} : loop) : loadLoopConfig({});
  const versions = configVersions(nodes);
  const deliberateCap = loop ? (loop.route?.maxDeliberateFraction ?? maxDeliberateFraction) : maxDeliberateFraction;
  const stallConfig = stallTolerance ? loadLoopConfig({ progress: stallTolerance === true ? {} : stallTolerance }).progress : null;
  let decisions = 0;
  let escalations = 0;
  let recoverySteps = 0;
  let queue = [];
  let attempts = {};
  let pendingGuidance = null;
  let deferredReview = false;
  let sinceDeliberate = Infinity;

  const canDeliberate = () => (escalations + 1) / (decisions + 1) <= deliberateCap;
  const evidenceOf = record => armEvidence(record.nextObservation?.state ?? record.observation?.state, executionOf(record), nodes.judge);

  async function call(policy, route, context, guidance) {
    const result = await policy.decide(guidance ? { ...context, guidance } : context);
    if (route === "deliberate") { escalations += 1; sinceDeliberate = 0; }
    return { ...result, model: result.model ?? policy.id, route };
  }

  async function decideLoop(context) {
    const { history, observation } = context;
    const state = observation?.state;
    const last = history.at(-1);
    const evidence = armEvidence(state, executionOf(last), nodes.judge);
    const trace = { versions };

    if (queue.length) {
      const failures = history.slice(-nodes.recover.abortOnFailures).filter(record => record.route === "recover" && nodes.route.failureStatuses.includes(executionOf(record)?.status)).length;
      if (failures >= nodes.recover.abortOnFailures) {
        trace.recover = { aborted: queue.length };
        queue = [];
        pendingGuidance = "The last recovery motions failed to plan or track. Choose a reachable approach from the images.";
      } else {
        // A progress review that falls on a macro step is kept for the first decision after the macro.
        if (context.progressReview?.required === true) deferredReview = true;
        recoverySteps += 1;
        const action = queue.shift();
        return { action, thought: `recovery macro step (${queue.length} left)`, route: "recover", model: "loop:recover", nodes: { ...trace, recover: { remaining: queue.length } } };
      }
    }

    const stalled = stalledSteps(history, nodes.progress);
    const repeats = history.slice(-nodes.route.repeatBlocks).filter(record => record.nodes?.gate?.verdict === "repeat").length;
    const failures = history.slice(-nodes.route.failureSteps);
    const failing = failures.length === nodes.route.failureSteps && failures.every(record => nodes.route.failureStatuses.includes(executionOf(record)?.status));
    const requested = deferredReview || context.progressReview?.required === true || (shouldDeliberate ? await shouldDeliberate(context) : false);
    deferredReview = false;
    const signals = { stalled, repeats, failing, requested, guidance: Boolean(pendingGuidance) };
    const wantDeliberate = requested || pendingGuidance || stalled >= nodes.route.stallSteps || repeats >= nodes.route.repeatBlocks || failing;
    let route = wantDeliberate && sinceDeliberate >= nodes.route.cooldown && canDeliberate() ? "deliberate" : "reflex";
    let guidance = route === "deliberate" ? [pendingGuidance, nodes.prompts.completion].filter(Boolean).join("\n") : null;
    pendingGuidance = null;
    sinceDeliberate += 1;
    let result = await call(route === "deliberate" ? deliberate : reflex, route, context, guidance);
    let calls = 1;
    let gate = screenAction({ decision: result, state, history, config: nodes.gate, progress: nodes.progress });

    while ((gate.verdict === "near_zero" || gate.verdict === "repeat") && calls < nodes.route.maxCallsPerStep && canDeliberate()) {
      guidance = `Your previous command for this step was rejected before execution: ${gate.reason}. ${nodes.prompts.completion}`;
      route = "deliberate";
      result = await call(deliberate, route, context, guidance);
      calls += 1;
      gate = screenAction({ decision: result, state, history, config: nodes.gate, progress: nodes.progress });
    }
    trace.route = { selected: route, calls, signals };
    trace.gate = { verdict: gate.verdict, reason: gate.reason ?? null, metrics: gate.metrics ?? null };

    // A blocked idle that survived re-deliberation is treated as an implicit completion claim.
    if (gate.verdict === "hold_claim" || gate.verdict === "near_zero" || gate.verdict === "repeat") {
      // The judge parses the task text itself: a progress wrapper appends review instructions to context.instruction.
      const verdict = judgeCompletion({ instruction: context.taskInstruction ?? context.instruction ?? observation?.text ?? "", evidence, history, evidenceOf, config: nodes.judge });
      trace.judge = verdict;
      if (verdict.verdict === "not_complete") {
        const attempt = attempts[verdict.reason] ?? 0;
        const budgetOk = recoverySteps < nodes.recover.maxFraction * Math.max(1, (context.budget?.stepsRemaining ?? 0) + decisions);
        const plan = attempt < nodes.recover.attemptsPerReason && budgetOk ? planRecovery({ verdict, evidence, history, attempt, config: nodes.recover }) : null;
        if (plan) attempts[verdict.reason] = attempt + 1;
        if (plan?.actions?.length) {
          queue = plan.actions.slice(1);
          recoverySteps += 1;
          trace.recover = { macro: plan.macro, arm: plan.arm, attempt, steps: plan.actions.length };
          return { action: plan.actions[0], thought: `judge: ${verdict.reason}; recovery ${plan.macro} on ${plan.arm} arm`, notes: result.notes, usage: result.usage, route: "recover", model: "loop:recover", nodes: trace };
        }
        pendingGuidance = plan?.directive ?? `Evidence says the task is not complete (${verdict.reason}). Recovery attempts for this reason are used up; re-plan from the images.`;
        trace.recover = { macro: plan?.macro ?? "exhausted" };
      } else if (verdict.verdict === "uncertain") {
        pendingGuidance = `A completion claim could not be verified from robot state (${verdict.reason}; evidence ${JSON.stringify(verdict.detail)}). Check the images and continue working unless the task is visibly complete.`;
      }
    }
    return { ...result, nodes: trace };
  }

  const routed = {
    id: `routed:${reflex.id ?? "reflex"}:${deliberate.id ?? "deliberate"}`,
    async reset(context) {
      decisions = 0;
      escalations = 0;
      recoverySteps = 0;
      queue = [];
      attempts = {};
      pendingGuidance = null;
      deferredReview = false;
      sinceDeliberate = Infinity;
      await reflex.reset?.(context);
      await deliberate.reset?.(context);
    },
    async decide(context) {
      if (loop) {
        const result = await decideLoop(context);
        decisions += 1;
        return result;
      }
      const stalled = stallConfig ? stalledSteps(context.history, stallConfig) >= patience : exactlyStalled(context.history, patience);
      const requested = context.progressReview?.required === true || (shouldDeliberate ? await shouldDeliberate(context) : stalled);
      const escalate = requested && canDeliberate();
      decisions += 1;
      if (escalate) escalations += 1;
      const selected = escalate ? deliberate : reflex;
      const result = await selected.decide(context);
      return { ...result, model: result.model ?? selected.id, route: escalate ? "deliberate" : "reflex" };
    },
    async close() {
      const results = await Promise.allSettled([reflex, deliberate].map(policy => Promise.resolve().then(() => policy.close?.())));
      const failed = results.find(result => result.status === "rejected");
      if (failed) throw failed.reason;
    },
  };
  if (progress === false) return routed;
  return createProgressPolicy({ policy: routed, ...(progress && typeof progress === "object" ? progress : progressOptions) });
}
