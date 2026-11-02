# System 1 routers and System 2 brains

System 1 decides which System 0 capability runs next. System 2 deliberates. Both have one small contract per runtime and
load from a configuration with `type: "module"` (a factory `(options, context) => role`), from a built-in type, or as
an object in a JavaScript configuration.

| Role | Routed mode (`runRoutedEpisode`) | Async / sync mode (`runAsyncEpisode`) |
|---|---|---|
| System 1 | `route(input)` before the first run and after every run; returns `{candidateId, reason}` or `{stop: true, reason}`. Built-in: `route-memory`. | `route(input)` at every action boundary; returns `{candidateId, reason, usedAdvice?}`. Built-in default: follow the latest advice, else keep the current capability. |
| System 2 | `judge(input)` when System 1 asks, between runs; returns `{verdict: "done" \| "not_done" \| "error", confidence?, rationale?, usage?}`. Built-in: `judge`. | `think(input)` in the background; returns `{candidateId?, directive?, reason?, usage?, model?}`. Built-in: `chat`. |

## Routed mode

### System 1 input

```js
{
  task, seed, instruction, candidates,      // candidates: [{ id, kind, description? }]
  observation, initialObservation,          // current and reset observations (no success field)
  runs,                                     // completed runs: [{ index, candidateId, status, steps, reason, error, verdict }]
  history, budget,                          // recent transitions; { stepsRemaining, remainingMs, runsRemaining }
  judge,                                    // async () => verdict, or null without System 2
  signal,
}
```

`status` is `returned`, `failed`, `budget` or `truncated` (see [capability boundaries](../routed-systems.md#capability-boundaries)).
Calling `judge()` is how System 1 gates System 2: the call is recorded on the last run, counts against the episode's
wall budget, and never sees success. A router that never calls it never pays for it.

A custom router, for example one that escalates to a VLA after any failed program and otherwise trusts the judge:

```js
export default function createRouter() {
  return {
    async route({ runs, candidates, judge }) {
      if (!runs.length) return { candidateId: "program", reason: "programs first" };
      const last = runs.at(-1);
      if (last.status === "failed") return { candidateId: "vla", reason: "program failed" };
      const verdict = await judge();
      if (verdict.verdict === "done") return { stop: true, reason: "judge_done" };
      return runs.length < 3 ? { candidateId: last.candidateId === "vla" ? "program" : "vla", reason: "not done" } : { stop: true, reason: "budget of runs" };
    },
  };
}
```

`createRouteMemoryRouter({ table, recoverOn, onJudgeError, maxRuns })` is the built-in; its `plan(task, candidateIds)`
returns the ladder a task will use.

### System 2 judge input

```js
{ observation, initialObservation, instruction, task, seed, runs, budget, signal }
```

`createVlmJudge(options)` (type `judge`) sends the current image, and by default the reset image, to a vision-language
model with a strict inspector prompt and returns `done` only when the model says done with confidence at or above
`threshold`. Prompt styles (`default`, `evidence`, `pickup`) and views are in `src/infer-judge.mjs`.
`tools/judge/qwen_judge_server.py` serves a local Qwen-VL judge over the same OpenAI-compatible API.

Calibrate a judge on development episodes before relying on it: a judge that says `not_done` after a success makes the
ladder run another capability on a solved scene, which can undo it.

## Async mode

System 1 input: `{observation, actions, history, notes, feedback, instruction, task, seed, step, budget, candidates,
advice, thinking, previousCandidateId, guidance, signal}`. System 2 input is the same snapshot; its answer becomes
`advice` for later routes, subject to `maxAdviceAgeSteps`. The [asynchronous systems guide](../async-systems.md) covers
scheduling (`thinkEverySteps`, `thinkingSchedule`), live events, the console and the evidence loop.

Two included System 1 examples beyond the default: the ALOHA hybrid configurations route among pi0.5, LingBot-VLA and
code controllers from current observations (`examples/aloha-pi05-lingbot-code.config.mjs`), and the RoboTwin runner's
success router (`src/policies/success-router.mjs`, served by `scripts/success_router_server.py`) scores every candidate
from the initial observation with a trained model and keeps the best for the episode.
