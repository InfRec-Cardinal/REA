# Framework guide

For REA's System 0/1/2 runtimes, use a `systems` configuration with `agent.run()`, `agent.evaluate()` or
`agent.start()`: the [routed systems guide](routed-systems.md) covers route memory, the boundary judge and the
escalation ladder, and the [asynchronous systems guide](async-systems.md) covers background thinking, live event
subscriptions, the console and the synchronous comparison. All paths share environment adapters, budgets and
evaluation reports; [the architecture overview](architecture.md) shows how they fit together.

Use `createEmbodiedAgent` to run a robot or benchmark with the same configuration,
episode lifecycle and result format. From a local/Git installation:

```js
import { createEmbodiedAgent } from "rea-agent";

const agent = await createEmbodiedAgent("demo");
try {
  const episode = await agent.run();
  console.log(episode.success, episode.steps);
  const evaluation = await agent.evaluate();
  console.log(evaluation.outputDir);
} finally {
  await agent.close();
}
```

Inside this repository, import from `./src/framework/index.mjs`. This example needs
only Node.js 22.19 or newer. `run()` runs the first configured episode;
`evaluate()` runs the whole plan and writes reports. The demo's grid and two-joint
fixtures use visible targets and a deterministic demo policy, so their success demonstrates
the interface rather than benchmark performance.

The environment owns sensing, available actions, units, control timing and task
outcomes. The policy chooses actions. The runtime manages recent transitions,
episode notes, budgets and recording. The asynchronous G1 service remains available
with its existing persistent pipeline and learning brain; its deployment is covered
in [the G1 guide](robots/unitree-g1.md).

## One configuration for CLI and API

List the included presets or run the offline demo:

```bash
npm run rea -- list
npm run demo
```

To configure ALOHA, initialize a separate file:

```bash
npm run rea -- init aloha --output aloha.config.json
```

The configuration has this shape:

```json
{
  "version": 1,
  "name": "aloha-transfer-cube",
  "environment": {
    "type": "aloha",
    "options": { "mode": "sim", "execute": true }
  },
  "policy": { "type": "chat", "options": {} },
  "episodes": [
    {
      "id": "transfer-cube:0",
      "task": { "taskName": "sim_transfer_cube_scripted" },
      "seed": 0,
      "instruction": "Transfer the cube between the two grippers."
    }
  ],
  "budget": {
    "maxSteps": 20,
    "maxDurationMs": 900000,
    "decisionTimeoutMs": 60000
  }
}
```

Install the selected backend using [the robots guide](robots/README.md#aloha-simulation), then
select its Python environment and SDK checkout through environment variables.
The CLI loads `.env` from the current directory, falling back to the repository's
file. `REA_ENV_FILE` (or the existing `G1_ENV_FILE`) selects another file;
exported shell variables take precedence:

```dotenv
ALOHA_PYTHON=/path/to/aloha-sim-venv/bin/python
ACT_ROOT=/path/to/act
REA_MODEL=your-model
REA_BASE_URL=http://localhost:8000/v1
REA_API_KEY=your-api-key
```

```bash
npm run rea -- doctor aloha.config.json
npm run rea -- run aloha.config.json --steps 5 --output outputs/aloha-first-run
```

`doctor` checks configuration and local dependencies. It does not connect to a
robot or model, initialize a simulator, or establish task success. `run` loads the
same file and runs the evaluation plan. The installed executable is
`rea`; inside the repository, use `npm run rea --` as shown above.

The API accepts a preset name, a configuration object or a `.json`/`.mjs` path:

```js
const agent = await createEmbodiedAgent("./aloha.config.json");
try {
  const episode = await agent.run({ seed: 7, budget: { maxSteps: 5 } });
  console.log(episode.reason);
} finally {
  await agent.close();
}
```

The API reads the supplied `env` object or `process.env`; `.env` loading is done by
the CLI entry point. Each agent permits one active run. It closes that run's
environment and policy after use, and `close()` cancels any active run.

## Configuration and paths

`environment` and `policy` select a registered `type` and its `options`. Presets
cover `demo`, `aloha`, `lerobot`, `openral`, `rpent`, `libero`, `arc-agi3`,
`gymnasium` and `g1`. `getPreset(name)` returns a fresh editable configuration.
The available policy types include `demo`, `scripted`, `chat`, `routed`, `progress`,
`lingbot-vla`, `pi05`, `aloha-code` and `module`. The
[ALOHA hardware entry point](robots/aloha-real.md) routes pi0.5, LingBot-VLA and
Python code controllers from current observations at each action boundary. The [VLA/code runner](../eval/robotwin/hybrid/README.md)
connects a trained success router that selects a candidate from the initial observation through the System1 interface.

Each episode can set `task`, `seed`, `instruction` and `budget`. `task` retains the
selected backend's native specification: an ACT task name, a LIBERO suite/task,
an ARC game ID, or a Gymnasium environment/reset configuration. `maxSteps` counts
policy decisions; adapters record executed control steps separately. Per-episode
`environment` and `policy` specifications allow heterogeneous plans; use
`reuseEnvironment: false` when changing environments between episodes.

Paths for Python interpreters, SDK/source directories, recordings and local
extension modules are resolved relative to the configuration file's directory.
For an inline configuration or preset they use the working directory, unless the
API receives `baseDir`. A bare Python executable such as `python3` is searched on
`PATH`. OpenRAL's `robotYaml` and `simEnvYaml` are relative to `openralRoot` when
provided, following the upstream catalogue layout.

For example, `configs/aloha.json` with `environment.options.actRoot` set to
`"../runtime/act"` selects `runtime/act` beside the `configs` directory. An absolute
`ACT_ROOT` or `ALOHA_PYTHON` is convenient when reusing the same SDK installation
across several configs. An explicit adapter option takes precedence over its
environment default. Other backends use `LEROBOT_PYTHON`, `OPENRAL_PYTHON` and
`OPENRAL_ROOT`, `RPENT_PYTHON` and `RPENT_ROOT`, `LIBERO_PYTHON`, `GYM_PYTHON`, or
`ARC_PYTHON` and `ARC_ENVIRONMENTS_DIR`; G1 uses its existing runtime configuration.
Use separate virtual environments for SDKs with different Python requirements.

The CLI resolves its explicit `--output` directory from the shell's working
directory. The API resolves `evaluate({outputDir})` from the configuration's base
directory. Both write the same report format described below.

Configuration fields can reference an environment variable using an object such
as `{ "env": "REA_MODEL" }`, with an optional `"default"`. This avoids embedding
machine-specific values in a shared recipe. Chat policy options also default to
`REA_MODEL`, `REA_BASE_URL` and `REA_API_KEY` without explicit references.

For a custom controller, register a factory on a `createRegistry()` instance and
pass it to `createEmbodiedAgent(config, {registry})`. A JSON config can load a local
factory with `type: "module"` and `options.module: "./my-policy.mjs"` (or an
environment module). Factories receive their options and context and return the
contracts below. SDK processes and device connections begin when an environment
is first used.

## Lower-level entry points

The underlying runner and individual adapters remain available:

```js
import { runEpisode, runEvaluation, createChatPolicy, createRoutedPolicy }
  from "rea-agent";
import { createArcAgi3Environment }
  from "rea-agent/environments/arc-agi3";
```

The package remains private to npm publishing. No Prime installation is required
for the generic runner. Model-backed examples use a streaming Chat Completions
endpoint. `tool` mode sends one structured action and arguments; `json` mode asks
for a JSON decision. `token` mode is available only for parameter-free,
single-character action names. Use `tool` or `json` for clicks and control vectors.

## Environment contract

An environment provides `id`, `timing` (`turn-based` or `realtime`),
`reset({task, seed, signal})`, `step(action, context)`, and optionally
`observe({signal})`, `stop()`, `close()` and `scorecard({close})`.

`reset` and `step` return:

```js
{
  observation: {
    id: "episode:step",         // identifies this observation
    timestampMs: Date.now(),    // acquisition time, Unix milliseconds
    text: "task or scene description",
    state: { /* JSON state, proprioception, grid, etc. */ },
    images: [{ mimeType: "image/png", data: "base64..." }],
    actions: [{
      name: "move_joints",
      description: "Target positions, radians, in shoulder/elbow order",
      parameters: {
        type: "object",
        properties: {
          positions: { type: "array", items: { type: "number", minimum: -1, maximum: 1 }, minItems: 2, maxItems: 2 }
        },
        required: ["positions"], additionalProperties: false
      }
    }]
  },
  terminated: false,            // environment terminal condition
  truncated: false,             // environment horizon/time limit
  success: null,                // boolean only when a task evaluator knows
  reward: null,                 // actual numeric environment reward, if any
  info: { /* metrics and execution feedback */ }
}
```

An action is `{name, arguments}`. The `step` context contains the exact observation
used by the policy, an abort signal and optional reasoning for recordings. Available
actions can change after every step. Units, coordinates, action duration and which
robot joints are controlled belong in the action description/schema.

The built-in parameter validator covers object properties/required fields,
`additionalProperties: false`, arrays with `items`/`prefixItems` and length bounds,
numbers/integers with inclusive or exclusive bounds, strings, booleans and enums.
It does not implement the entire JSON Schema standard or insert schema defaults;
the adapter applies omitted optional parameters. Invalid actions surface as errors.

An adapter should return after the chosen action's meaningful control interval and
then acquire the next observation. `timing` describes the environment; it does not
make `runEpisode` a realtime scheduler. A real robot adapter owns completion and
stop behavior. It also owns physical setup/reset, which differs from resetting a
simulator. Return `success: null` when there is no task evaluator.

## Policies, memory and routing

Implement `decide({observation, actions, history, instruction, budget, notes,
feedback, signal})`, returning:

```js
{
  action: { name: "move_joints", arguments: { positions: [0.2, -0.4] } },
  notes: "Complete replacement episode notes: tested actions, hypotheses, next goal",
  thought: "Brief action rationale",
  claim: { status: "working", evidence: "Observed execution feedback" },
  review: { required: false, reasons: [] },
  usage: { total_tokens: 123 },  // optional; unknown usage remains null
  model: "policy-or-model-name"
}
```

`reset({task, seed, environment})` is an optional policy hook. Every episode starts
with empty history/notes. A default maximum of 12 recent transitions and 4000 note
characters bounds memory. The chat policy renders at most 8 recent transitions by
default, configurable with `historyLimit`, excluding historical images. Retained
transition fields supplied to the policy are serialized without character
truncation. The former `historyMaxChars` option has been removed.
Large feedback can increase input tokens or exceed the provider's context window;
provider errors are surfaced without silently truncating history. Current images
and state remain available. Budget includes remaining steps, remaining wall time and the
current decision deadline. A scripted controller or local learned policy can use
the same interface without a model API.

DeepSeek chat policies default to `thinking: { type: "disabled" }`, as in the
existing LIBERO control path, so implicit reasoning cannot consume the entire
action-output budget. Set `policy.options.thinking` explicitly to enable it and
raise `maxOutputTokens` accordingly. Other providers receive no thinking option
by default. Output truncation is reported as `output_limit`; available API usage
is attached to inference errors as `error.usage`.

```js
const policy = createRoutedPolicy({
  reflex: createChatPolicy({ baseUrl, apiKey, model: smallModel }),
  deliberate: createChatPolicy({ baseUrl, apiKey, model: largeModel }),
  maxDeliberateFraction: 0.5,
  shouldDeliberate: ({ history }) =>
    history.slice(-3).length === 3 && history.slice(-3).every(t => t.reward === 0)
});
```

The default router considers repeated unchanged state (exact equality) or execution
refusals; `stallTolerance: true` (see the evidence loop below) ignores sensor jitter
instead. Supply `shouldDeliberate` for task-specific progress signals, especially
image-only environments. The fraction caps deliberate decisions, not monetary cost.
Routes and model usage are recorded; this router is not a trained policy and does not
itself establish an improvement over a single model.

### Progress review and evidence

The registry enables the `progress` layer for `chat` and `routed` policies by
default. Direct `createChatPolicy` and `createRoutedPolicy` calls keep the lower
level API compatible unless `progress: true` is supplied; set `progress: false`
explicitly in a registry policy when reproducing the old behavior. A standalone
wrapper is also available:

```js
const policy = createProgressPolicy({
  policy: createChatPolicy({ baseUrl, apiKey, model, progress: false }),
  reviewInterval: 6,
  repeatWindow: 5,
  patience: 3
});
```

Reviews happen inside the existing decision and compare expected effects with the
latest observation and execution feedback. They are requested periodically, after
approximately repeated actions (including holds), after repeated execution failures,
or after a policy returns a `claim.status: "complete"`. A claim is evidence to
check, not an environment success signal; only the environment's actual `success`
and termination fields decide an episode result. Reviews do not force release,
movement, recovery macros or an extra model call. The router receives the complete
history and can route a required review to its deliberate policy subject to its
existing fraction cap. The runtime records `claim` and `review` on each transition.

The wrapped chat policy receives `historyForPrompt`, which keeps recent actions and
execution feedback while dropping repeated old state snapshots and assistant notes.
Its full `history` remains available to routing. Current observation, images,
feedback and notes stay available. This avoids fixing an early “completed” statement
as a later premise while preserving the evidence needed to revise it.

### Evidence loop (experimental, opt-in)

`createRoutedPolicy({ ..., loop: true })`, or `loop` in a registry `routed`
policy's options, passes every decision through the nodes in `src/framework/loop/`:
tolerance-based motion and stall detection, a pre-execution gate that rejects
near-zero and repeated pose commands and reads idle completion claims, a rule
judge that checks completion claims against measured finger gap and lift, and
recovery macros (`regrasp`, `lift`, or a `directive` for the deliberate policy)
built from ordinary `move_gripper` pose commands. Pass an object instead of `true`
to override node parameters; every node is a versioned `id@version` configuration
and `INVARIANTS` lists what an override may not change. Without `loop` the router
behaves as described above.

Transitions record `route: "recover"` for macro steps and a `nodes` trace (node
versions, gate, judge, route and recovery). Each macro step consumes one decision
of the shared budget. Re-deliberation passes `guidance` to the deliberate policy;
the chat policy renders it as supervisor guidance. The judge only has rules for
pose-controlled bimanual state such as RoboTwin's; elsewhere it returns
`uncertain` and only forwards evidence to the deliberate policy. Under the
progress layer the judge reads the unmodified task text (`taskInstruction`), a
required review asks for the deliberate policy, and a review that falls on a macro
step is applied to the first decision after the macro. The asynchronous
System0/1/2 runtime accepts the same `loop` option at its action boundary (see
[the asynchronous systems guide](async-systems.md)).

`stallTolerance: true`, or an override of the loop `progress` node, changes the
default router's stall test to the loop's tolerance test: transitions that moved
the arms less than sensor jitter count as stalled, a transition without `state`
counts as motion, and only numeric state fields are compared (a change in a
string-only field such as ARC's `gameState` is not motion). The loop always uses
the tolerance test. The loop is not the default recovery strategy.

## Robot profiles

A profile is the vocabulary of one robot: its tokens, what each does, which tokens move it,
which undo each other, the arm interlock, the motion envelope and the gate thresholds. Two
are built in, `G1_PROFILE` and `LIBERO_PROFILE`; `G1_ROBOT_PROFILE` selects one for the G1
service and the `g1` environment, taking the names `g1` or `libero`, or a module path whose
default export is a profile spec. The path must start with `./` or be absolute: a bare name is
treated as a package specifier and fails to resolve.

```js
import { defineProfile } from "rea-agent";

export default defineProfile({
  id: "my-base",
  name: "my omnidirectional base",
  TOKENS: ["H", "F", "L", "R"],
  ACTION_TABLE: {
    H: { token: "H", name: "HALT", kind: "stop", action: "stop", vx: 0, vy: 0, yaw: 0, durationS: 0, defaultMs: 0 },
    F: { token: "F", name: "FORWARD", kind: "velocity", action: "move", vx: 0.2, vy: 0, yaw: 0, durationS: 1, defaultMs: 1000, veto: { depth: "center", lidar: null } },
    L: { token: "L", name: "LEFT", kind: "velocity", action: "move", vx: 0, vy: 0.2, yaw: 0, durationS: 1, defaultMs: 1000, veto: { depth: "left", lidar: null } },
    R: { token: "R", name: "RIGHT", kind: "velocity", action: "move", vx: 0, vy: -0.2, yaw: 0, durationS: 1, defaultMs: 1000, veto: { depth: "right", lidar: null } },
  },
  roles: { stop: "H", forward: "F", strafeLeft: "L", strafeRight: "R" },
  motion: ["F", "L", "R"],
  opposite: { L: "R", R: "L" },
  arms: null,
  envelope: { minLinear: 0.1, minYaw: 0.1, pulseRangeS: [0.4, 2.0] },
  sensing: { depth: { nearCenterM: 0.5, nearSideM: 0.3, floorCenterM: 0.25, floorSideM: 0.2 }, lidar: null },
  TOKEN_LEGEND: "H halt F forward L left R right",
});
```

`roles.stop` is required; every other role is optional and the judge, router and memory
simply have less to say without it (no turn role means a refused forward step recommends
stopping, not turning). A token's `veto` names the depth window and lidar sector that must be
clear before the gate lets it through; `usable` asks only for a usable scan. `sensing: null`
declares a robot the gate checks for readiness only, as with LIBERO. `defineProfile` validates
the spec and names the offending field in its error.

## Sensor sources

The camera contract is `capture(options, { signal }) -> frame`, where a frame carries
`imageB64`, `mimeType`, `width`, `height`, `capturedAt` (Unix seconds), `scene` and `depth`
(`center_m`, `left_m`, `right_m`, `front_min_m`, `center_valid_fraction`,
`front_valid_fraction`, `scale_m`). `options.depthOnly` asks for the depth summary alone; the
gate uses it to re-read depth at execution time. Three cameras exist: the persistent worker,
the per-capture subprocess and the in-process fixture (`createFixtureCamera`).

An obstacle source is `{ kind, read() -> Promise<summary> }`, the summary being `usable`,
`blocked`, `reason`, `age` and the sector distances `center`, `front`, `left`, `right`.
`createSnapshotObstacleSource` reads the Livox producer's file, `createFixtureObstacleSource`
serves fixed or computed distances, and `createExecutor({ obstacles })` consumes whichever it is
given. The cameras live in `src/camera.mjs` and the obstacle sources in
`src/sensors/obstacles.mjs`; the G1 service picks one of each from `G1_SENSE` and `G1_OBSTACLES`,
and neither is a package-root export.

## Engine registry

`createEngineRegistry()` holds the inference engines by name with what the service needs to
know about each: `outputTokens` (1 for the one-token engines), `deliberate`, `needsProvider`
and a `health(config)` function that renders its `/health` entry. `register(name, infer, meta)`
fills in the built-in metadata for the known names; `get(name)` throws for an unknown one, and
`decide()` turns that into a 400 with the registered names instead of quietly running another
engine. `asEngineRegistry(plainObject)` wraps the `{ name: infer }` objects the tests and the
older call sites pass.

## Adapters

| Adapter | Observations and actions | Outcome source |
|---|---|---|
| G1 | Camera, depth, lidar, robot state; bounded locomotion and arm primitives | Optional external `evaluate` callback |
| LIBERO-Pro/Long | External/wrist montage, proprioception; Cartesian/rotation/gripper primitives with amplitude and repeat count | Simulator task success and control horizon |
| ARC-AGI-3 | Every grid/frame, current available actions and coordinate clicks | Official game state and scorecard |
| Gymnasium | JSON observations and optional RGB rendering; Discrete, Box, MultiDiscrete, MultiBinary, Dict, Tuple actions | Gym reward, terminal/truncation flags and optional `info.is_success` |
| ALOHA | ACT simulation or classic ROS1 dual arms; 14 joint/gripper values, named cameras, control chunks | ACT task criterion in simulation; external task evaluator on hardware |
| LeRobot | Native named actions/cameras for SO followers, BiSO, Koch and Robot-compatible factories | External task evaluator |
| OpenRAL | Manifest-driven HALs for arms, dual arms and mobile/humanoid platforms; joint/control surfaces or native scene actions, named cameras | Native scene transitions; bare HAL `task_success` when available and task evaluator for episode termination |
| RPent | Official LIBERO, RoboCasa and RoboTwin HTTP/socket services; native controller vectors and chunks | Native environment outcomes and actual executed control steps |

See [ALOHA hardware](robots/aloha-real.md) and [the robots guide](robots/README.md) for ALOHA setup, LeRobot hardware configurations,
action chunks and adding another robot driver. [the OpenRAL and RPent guide](robots/openral-rpent.md)
lists OpenRAL/RPent configurations, setup, native action semantics and test scope.

### G1

`createG1Environment({camera, executor, config, execute = false, evaluate,
waitForCompletion, profile})` accepts the existing camera/executor instances and the
robot profile (default G1). It uses the existing gate and fresh depth check. Reset observes
the robot without repositioning it. Preview is the default; task success must come
from `evaluate`, not an accepted movement command. With `G1_SENSE=fixture`,
`G1_OBSTACLES=fixture` and `G1_ACTUATOR=mock` the environment runs offline;
`npm run demo:service` shows the same path through the HTTP service.
See [the G1 setup](robots/unitree-g1.md#prime-and-robot-setup).

For velocity commands, the adapter waits through the requested pulse plus 50 ms
before capturing the next image. A cerebellum `EXECUTION_COMPLETE` receipt already
represents completion and does not wait again. `waitForCompletion` can provide a
robot-specific completion check, including arm actions; otherwise arm receipts
remain acknowledgements. `info.completion` records which condition was observed.
None of these conditions substitutes for a task success check.

### LIBERO

Point `LIBERO_PYTHON` at a Python environment with the existing LIBERO-Pro/Long
backend and assets installed. The new example contains no host-specific path:

```bash
LIBERO_PYTHON=/path/to/libero/bin/python LIBERO_SUITE=libero_10_task \
LIBERO_TASKS=0,1 REA_SEEDS=51 REA_MODEL=your-model \
REA_BASE_URL=http://localhost:8000/v1 \
npm run rea -- run examples/libero.mjs
```

This adapter supports the eight LIBERO-Pro/Long suites of `scripts/libero_backend.py`, not every
upstream LIBERO distribution. Seeds 1–50 select official held-out initial states;
51–100 use development resets. `maxSteps` counts policy actions; simulator control
steps are recorded separately. Parameterized actions change the action contract
relative to the fixed-token controller. Setup is documented in [the LIBERO guide](benchmarks/libero.md).

### Gymnasium and additional robots

Install Gymnasium in a separate Python 3.10+ environment, then run:

```bash
python3 -m venv runtime/gym-venv
runtime/gym-venv/bin/python -m pip install -r requirements-gym.txt
GYM_PYTHON="$PWD/runtime/gym-venv/bin/python" \
GYM_ENVIRONMENTS=CartPole-v1,Pendulum-v1 REA_MODEL=your-model \
REA_BASE_URL=http://localhost:8000/v1 \
npm run rea -- run examples/gymnasium.mjs
```

Set `GYM_MAKE_OPTIONS='{"render_mode":"rgb_array"}'` to include rendered frames;
install the selected environment's rendering dependencies first. `GYM_MODULES`
imports comma-separated Python registration modules before `gym.make`.
Third-party Gym tasks need their own simulator/assets and registration. Robot
environments using this API can expose their native continuous action spaces
without changing the core. CartPole/Pendulum integration checks do not validate
MuJoCo, Isaac, ROS or a different physical robot.

For a non-Gym robot, implement the environment contract above, translate arguments
to its controller, and return its observations and task outcomes. Start from
[the small synthetic adapter](../examples/eval-fixtures.mjs). No edits to the episode
runner, chat policy or evaluator are needed.

## Evaluation and outputs

`npm run rea -- run path/to/config.json --output outputs/run-name` and
`agent.evaluate()` use the same evaluator and reports. JSON configurations select
registered environments and policies as shown above. Existing JavaScript
configurations can instead export `episodes`, `createEnvironment(episode)`,
`createPolicy(episode, env)`, optional `budget`, descriptive `metadata`, and
`reuseEnvironment`. Run them with `npm run rea -- run path/to/config.mjs`.

Every episode can override `task`, `seed`, `instruction` and `budget`. Default
budgets are 200 policy actions, 5 minutes per episode and 30 seconds per decision.
A timed-out policy cannot dispatch its late result. The runner calls `stop` at
episode end; evaluation owns final scorecard closure and `close`.

Output files are:

- `manifest.json`: episode plan, budgets, metadata and recording settings.
- `0001.jsonl`: reset, decisions/transitions and episode end; images omitted by
  default (`recordImages: true` retains them).
- `0001.json`: outcome, executed steps/action attempts, usage, timing, notes and
  errors. `modelCalls` counts policy decision attempts, including failed ones.
- `summary.json`: completed/failed/unscored counts and official scorecard where
  the adapter provides one.

Execution acknowledgements, a maximum-step exit and missing metrics do not become
wins. Unknown reward/usage/success stays `null`. Token usage is not a price quote.
Success rate is available only when all planned episodes have known boolean task
outcomes and evaluation finalization succeeded; it is not ARC's official score.

Most configurations create a fresh environment for each episode. The unified ARC
adapter defaults to sharing an environment and scorecard across its game plan.
With `reuseEnvironment: true`, every episode resets the same environment with a
fresh policy, and the whole run closes one scorecard. The lower-level
`runEvaluation` defaults to a fresh environment unless told to reuse it.
Shared-environment errors stop the batch. Existing output files are never
overwritten; this runner currently starts new runs rather than resuming
interrupted ones.

See [the ARC-AGI-3 guide](benchmarks/arc-agi3.md) for ARC setup, Competition mode and the remaining
work toward leaderboard submissions.
