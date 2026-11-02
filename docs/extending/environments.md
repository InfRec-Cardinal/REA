# Environments and benchmarks

An environment owns sensing, the available actions, units, control timing and the task outcome. Benchmarks,
simulators and real robots are all environments; policies and routers never import a simulator.

## Contract

```js
export function createMyEnvironment(options, context) {
  return {
    id: "my-benchmark",
    timing: "turn-based",                 // or "realtime"
    resumable: false,                     // routed mode: true if another capability may run after one terminates
    async reset({ task, seed, signal }) { return transition; },
    async step(action, { observation, signal, reasoning }) { return transition; },
    async observe({ signal }) {},         // optional: a fresh observation without acting
    async stop() {},                      // optional: halt motion at the end of an episode
    async score() { return { success }; },// optional: the sealed, once-per-episode success read (routed mode)
    async scorecard({ close }) {},        // optional: an official benchmark scorecard
    async close() {},                     // optional: release simulators, sockets, devices
  };
}
```

A transition is `{observation, terminated, truncated, success, reward, info}`; an observation is
`{id, timestampMs, text, state, images, actions}`. Return `success: null` unless the environment really evaluates the
task. Field-level details, the action-schema subset the validator supports, and timing rules are in the
[framework reference](../framework.md#environment-contract).

Load it from a configuration with `{"type": "module", "options": {"module": "./my-environment.mjs", ...}}`, or register
it: `registry.registerEnvironment("my-benchmark", createMyEnvironment, { description, timing, requires })`. `requires`
(`pythonEnv`, `pythonModules`, `sourceEnv`, `runtime`) lets `rea doctor` check the setup before a run.

## Python simulators: the JSON-lines backend pattern

Most simulators are Python. The included adapters keep Node in charge of the episode and run the simulator in a child
process that reads one JSON request per stdin line and writes one JSON reply per stdout line
(`{id, ok: true, ...}` or `{id, ok: false, error}`); all simulator logging goes to stderr. Use the same pattern:

- `src/workers.mjs` (`LineWorker`) and `src/libero-client.mjs` / `src/program-client.mjs` manage the process, request
  ids and per-command timeouts.
- `scripts/gymnasium_backend.py` is the smallest complete backend; `scripts/libero_backend.py` and
  `scripts/arc_agi3_backend.py` show image observations, action translation and official scoring.
- Start the simulator lazily on the first `reset`, so `rea doctor` and configuration loading never launch it.

## Hosting capabilities: frozen programs and resumable episodes

For routed episodes over capabilities that run inside the simulator, such as frozen programs, the environment:

1. declares `resumable: true` when a new capability can continue from the current physical state after one returns;
2. reports `info.capability = { candidateId, status: "running" | "returned" | "failed", error? }` on every step;
3. implements `score()` and reads the task predicate only there.

`src/environments/libero-programs.mjs` is the reference implementation over the [program backend protocol](program-backends.md).

## Included benchmark integrations

| Benchmark | Adapter | Guide |
|---|---|---|
| LIBERO / LIBERO-Pro | `libero` environment (token or parameterized control); frozen-program environment | [LIBERO](../benchmarks/libero.md) |
| BEHAVIOR-1K (R1Pro) | BEHAVIOR program backend; online agent runner; offline dataset check | [BEHAVIOR](../benchmarks/behavior.md) |
| RoboTwin 2.0 | VLA / code candidates with a trained success router; RPent services; OpenRAL scenes | [RoboTwin](../benchmarks/robotwin.md) |
| ARC-AGI-3 | official SDK, offline and online scorecards | [ARC-AGI-3](../benchmarks/arc-agi3.md) |
| Gymnasium | any registered environment, dynamic action spaces | [framework reference](../framework.md#gymnasium-and-additional-robots) |

Each guide states what the integration needs from upstream (simulators, assets, checkpoints) and what is not included.
