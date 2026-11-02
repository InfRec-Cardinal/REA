# Routed systems

`runRoutedEpisode` is REA's episode: route memory chooses the first capability, a judge gate runs only when a capability
returns, an escalation ladder decides what runs next, every run shares one budget, and success is read once at the end.

## Try it offline

```bash
npm run demo:routed
node examples/routed-system-demo.mjs --events
```

The demonstration is a drawer that two frozen "programs" push open. Route memory runs `quick` first (it needs fewer
steps per development success), the scripted judge reports `not_done` after `quick` and after `thorough`, the ladder
`[quick, thorough, quick]` runs to its end, the judge reports `done`, and the drawer's `score()` is read once.

## Configuration

```js
export default {
  environment: { type: "module", options: { module: "./my-environment.mjs" } },
  systems: {
    mode: "routed",
    system0: [
      { id: "fix", kind: "program", policy: { type: "module", options: { module: "./program-policy.mjs", program: "fix" } } },
      { id: "vla", kind: "vla", policy: { type: "module", options: { module: "./my-vla-policy.mjs", url: "ws://127.0.0.1:8000" } } },
    ],
    system1: { type: "route-memory", options: { table: "./route-table.json", recoverOn: "judge" } },
    system2: { type: "judge", options: { model: "your-vision-model", threshold: 0.6, image: "front" } },
    maxRuns: 3,              // hard cap on capability runs per episode (the ladder usually ends sooner)
    maxStepsPerRun: 50,      // ends a run of a capability that never returns on its own (VLAs, controllers)
  },
  episodes: [{ id: "drawer-1", task: { suite: "libero_goal_task", task: "open_the_top_drawer" }, seed: 1 }],
  budget: { maxSteps: 200, maxDurationMs: 900000 },
};
```

| Field | Meaning |
|---|---|
| `system1.type: "route-memory"` | Built-in System 1. `table` is a route table object or a path relative to the configuration. Without a table the capabilities run in configuration order. |
| `recoverOn` | What triggers the next rung: `judge` (ask System 2 after every run; escalate unless it says done), `exception` (escalate only after a failed or truncated run), `always` (blind retry), `never` (one run). |
| `onJudgeError` | Under `recoverOn: judge`, whether a judge error escalates (`escalate`, default) or stops (`stop`). |
| `system2.type: "judge"` | Built-in System 2: a vision-language judge over an OpenAI-compatible endpoint (`REA_BASE_URL`, `REA_API_KEY`, model from `REA_JUDGE_MODEL` or `REA_MODEL`). `image` selects an observation image by name; `withInitial` sends the reset image too. |
| `maxRuns`, `maxStepsPerRun`, `judgeTimeoutMs`, `scoreTimeoutMs` | Runtime limits. `budget.maxSteps` and `maxDurationMs` are shared by every run and every judge call. |

Both System 1 and System 2 also accept `type: "module"` (a factory returning `route` / `judge`) or an object from a
JavaScript configuration. See [Routers and brains](extending/systems.md).

## Route tables

A route table holds, per task, each capability's development evidence (`n`, `successes`, `steps`), the ranking and the
ladder. The rule (`ROUTE_RULE`):

1. higher development success rate;
2. then fewer development steps per success (steps of all scored episodes divided by successes);
3. then configuration order.

The ladder is `[first, the other capabilities by rank, first]`, cut to `maxRuns` (default 3): with two capabilities
`[first, other, first]`, with one `[first, first]`. Capabilities without development evidence for a task are not on its
ladder. The table records its own `sha256`; the router refuses a table whose content no longer matches.

Build a table from one development evaluation per capability:

```bash
npm run rea -- run configs/dev-fix.json --output outputs/dev-fix
npm run rea -- run configs/dev-evo.json --output outputs/dev-evo
node tools/route-table.mjs --run fix=outputs/dev-fix --run evo=outputs/dev-evo --output route-table.json
```

or in code with `buildRouteTable(results, {candidates})`. **Build tables from development episodes only** and evaluate
on different seeds or initial states: a table built on evaluation episodes selects on the test set. Episodes that
errored or have no boolean success carry no evidence.

## Capability boundaries

A capability run ends at the first of:

| Condition | Run status |
|---|---|
| a transition reports `info.capability.status === "returned"` | `returned` |
| a transition reports `info.capability.status === "failed"` (with `error`) | `failed` |
| a transition is `terminated` | `returned`; the episode also ends unless the environment declares `resumable: true` |
| a transition is `truncated` | `truncated`; the episode ends |
| the run used `maxStepsPerRun` steps, or the episode used `budget.maxSteps` | `budget` |

Environments that host resumable capabilities (such as frozen programs that can run one after another on the same scene)
set `resumable: true` and report `info.capability` on every step. `src/environments/libero-programs.mjs` is the reference.
For VLAs and other controllers that never return on their own, set `maxStepsPerRun`.

## The sealed score

The router, the judge and the capabilities never receive `success`. After the last run the runtime calls
`environment.score()` once and reads `score.success`. Environments without `score()` fall back to the `success` of the last
transition. Budget exhaustion (`max_steps`, `deadline`) is an outcome and is still scored; an error or a cancellation
leaves the episode unscored (`success: null`).

## Records

The episode summary adds `runs` (candidate, status, steps, the judge's verdict, duration), `stopReason`
(`judge_done`, `run_returned`, `ladder_exhausted …`), `score`, and `routedMetrics` (route and judge calls, judge time).
`onEvent` receives `routing.completed|stopped`, `run.started|completed`, `judge.started|completed|failed` and
`action.completed` events; evaluations write them to `NNNN.events.jsonl`.

## Programmatic use

```js
import { buildRouteTable, createRouteMemoryRouter, createVlmJudge, runRoutedEpisode } from "rea-agent";

const table = buildRouteTable(developmentResults, { candidates: ["fix", "evo"] });
const summary = await runRoutedEpisode({
  environment, task, seed,
  system0: [{ id: "fix", policy: fixPolicy }, { id: "evo", policy: evoPolicy }],
  system1: createRouteMemoryRouter({ table, recoverOn: "judge" }),
  system2: createVlmJudge({ model: "your-vision-model" }),
  maxSteps: 4000, maxDurationMs: 900000,
});
```

The caller owns `close()` of policies and environments; the runtime resets policies and stops the environment.
