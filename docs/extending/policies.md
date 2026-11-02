# System 0 policies: agents, VLAs, code-as-policy and controllers

A System 0 policy turns the current observation into one action for the environment. Every kind of method, whether an
LLM agent, a VLA, generated or frozen code, or a motion controller, uses the same contract, so any of them can run alone,
as an async candidate, or as a rung of a routed ladder.

## Contract

```js
export function createMyPolicy(options, context) {
  return {
    id: "my-policy",
    async reset({ task, seed, environment, signal }) {},          // optional, once per episode
    async decide({ observation, actions, history, notes, feedback, instruction, budget, signal, ...rest }) {
      return {
        action: { name: "move_joints", arguments: { positions: [0.1, 0.2] } },  // must match one of `actions`
        thought: "brief rationale",                                  // optional, recorded
        notes: "replacement episode notes",                          // optional, passed back next step
        usage: { total_tokens: 123 },                                // optional token usage
        model: "model-or-policy-name",                               // optional, recorded
      };
    },
    async close() {},                                              // optional
  };
}
export default createMyPolicy;
```

- `observation` has `id`, `timestampMs`, `text`, `state`, `images` (`[{name, mimeType, data}]`) and `actions`.
- `actions` are JSON-Schema action specifications; the runtime validates the returned action against them.
- `feedback` is the previous step's `info` (execution feedback); `history` holds the recent transitions.
- `budget` has `stepsRemaining`, `remainingMs` and `decisionMs`. Honour `signal`: a decision past its deadline is
  discarded, and the runtime aborts the signal.
- Inside systems, `input.route` is System 1's selection and `input.directive` System 2's directive (async mode), or
  `input.run` the current capability run (routed mode).

Use it from a configuration without registering anything:

```json
{ "policy": { "type": "module", "options": { "module": "./my-policy.mjs", "anyOption": 1 } } }
```

or register it for reuse: `registry.registerPolicy("my-policy", createMyPolicy, { description: "…" })`, then
`createEmbodiedAgent(config, { registry })`.

## LLM and VLM agents

The built-in `chat` policy talks to any OpenAI-compatible Chat Completions endpoint (`REA_MODEL`, `REA_BASE_URL`,
`REA_API_KEY`, or `model` / `baseUrl` / `apiKey` options). `mode: "tool"` sends the environment's actions as tools,
`"json"` asks for a JSON decision, and `"token"` is for single-character action alphabets. `routed` combines a reflex and a
deliberate policy, and the `progress` layer reviews completion claims against execution evidence; see the
[framework reference](../framework.md#policies-memory-and-routing).

To wrap an existing agent (an SDK agent, a CLI agent, a planner), call it inside `decide` and translate its answer into
one action; keep long-running work behind the `signal`.

## VLA servers

| Policy type | Server | Notes |
|---|---|---|
| `lingbot-vla` | Any openpi-style WebSocket + MessagePack-NumPy server (LingBot-VLA 2.0 by default) | `imageMap`, `stateMap`, `actionKeys`, `actionName`, `actionArgument`, `chunkSteps` map any checkpoint's features to the environment's actions. [Guide](../policies/lingbot-vla.md) |
| `pi05` | openpi pi0.5 server for Classic ALOHA | absolute joint-14 targets, `chunkSteps` (default 8) |
| `src/policies/robotwin-vla.mjs` | RoboTwin-style VLA services | joint or end-effector chunks; used by the [RoboTwin runner](../../eval/robotwin/hybrid/README.md) |

Each decision runs fresh inference and executes the first `chunkSteps` actions of the predicted chunk; the unused tail is
discarded, so a router can switch away and back without replaying a stale prediction. The Python clients
(`scripts/lingbot_vla_client.py`, `scripts/pi05_client.py`, `scripts/robotwin_vla_client.py`) need only NumPy, Pillow,
MessagePack and websockets (`requirements-lingbot-vla.txt`); model weights and servers are not part of this repository.

A server that speaks another protocol needs a small module policy: send the mapped observation, receive a chunk, return
it as the environment's chunk action (for example `robot_trajectory` with `commands`).

## Code-as-policy

Two forms are supported.

**Generated each step.** `aloha-code` (Classic ALOHA) and `src/policies/robotwin-code.mjs` (RoboTwin) ask a model for one
`run_python` program per decision, run it in a persistent worker with NumPy and the observation, and return the one
action the program computes. Generated code runs in a separate process whose environment contains no credentials
(`codeWorkerEnv`), but **this is not a sandbox**: run code policies in a container or VM when the model is untrusted.
`src/policies/robotwin-codex.mjs` drives the same loop through the official Codex CLI in a read-only sandbox.

**Frozen programs.** A program written offline (by a person, by System 2, or by a program-search system) runs as a
System 0 capability inside a program backend that exposes the robot API to it. The LIBERO program environment
(`src/environments/libero-programs.mjs` with `scripts/async_libero_backend.py`) pauses programs at real control-step
boundaries, so a router can continue or switch them; `tools/build_program_bundle.py` freezes a program library into a
bundle. The protocol for adding a program backend for another simulator or robot is in
[program backends](program-backends.md).

## Motion and whole-body controllers

- **Scripted / analytical controllers**: a module policy whose `decide` computes the action (a planner, an IK solver, a
  servo loop step). The `scripted` policy type replays an explicit action list for tests and demos.
- **G1 cerebellum** (`agent-vla-async/`): a robot-side daemon that owns locomotion and arm execution, re-gates every
  pulse, and accepts bounded plans from the agent over a Unix socket. It hosts a GR00T policy client and a SONIC
  (whole-body) contract stub as controllers; the `g1` environment reaches it with `G1_ACTUATOR=cerebellum`. See
  [the cerebellum guide](../../agent-vla-async/README.md).
- **Your own whole-body or locomotion controller**: implement it behind the environment's action (for example a
  `velocity` or `whole_body_target` action whose arguments the controller tracks for a control interval), or as a
  System 0 policy that outputs those arguments. Keep safety checks in the environment or driver, where every policy's
  action passes through them.

## Recording and evaluation

Policies do not write files. The runtime records every decision, action, observation summary, usage and timing in
`NNNN.jsonl`, and the episode outcome in `NNNN.json`; see [Evaluation and outputs](../framework.md#evaluation-and-outputs).
