# REA: Routed Embodied Agent

[![ci](https://github.com/InfRec-Cardinal/REA/actions/workflows/ci.yml/badge.svg)](https://github.com/InfRec-Cardinal/REA/actions/workflows/ci.yml)

REA is a framework for embodied agents that route among fast policies instead of putting a large model in the control
loop. It has three layers:

- **System 0**, capabilities that act at control rate: frozen code programs, VLA policies, LLM tool-calling
  controllers, motion and whole-body controllers.
- **System 1**, a cheap router: route memory picks the capability that did best on development episodes, a judge gate
  checks completion only when a capability returns, and an escalation ladder decides what runs next under one shared
  budget.
- **System 2**, the brain: it writes or selects capabilities offline, and online it only judges or advises at
  boundaries. No running capability waits for it.

The same interfaces connect benchmarks (LIBERO / LIBERO-Pro, BEHAVIOR-1K, RoboTwin, ARC-AGI-3, Gymnasium), real robots
(ALOHA, LeRobot SO-100/101 and Koch, OpenRAL and RPent robots, Unitree G1, your own driver) and methods (LLM agents,
VLAs, code-as-policy, controllers). Every episode records its decisions, actions, timing and usage in the same format.

## Start in one minute

Install [Node.js](https://nodejs.org/) 22.19 or newer:

```bash
git clone https://github.com/InfRec-Cardinal/REA.git
cd REA
npm run demo            # interface demo: a grid and a two-joint fixture, no model or hardware
npm run demo:routed     # a routed episode: route memory, boundary judge, escalation ladder, sealed score
npm run demo:async      # System 1 routing while System 2 thinks in the background
npm run rea -- list     # environments, policies and presets
```

These demos are deterministic and need no Python, credentials or hardware; they show the interfaces, not robot
performance.

## Configure once, choose a backend

A configuration names the environment, the policy (or the systems), the episodes and the budget:

```json
{
  "version": 1,
  "environment": { "type": "gymnasium", "options": { "environmentId": "CartPole-v1" } },
  "policy": { "type": "chat" },
  "episodes": [{ "id": "cartpole-0", "seed": 0, "instruction": "Balance the pole for as long as possible." }],
  "budget": { "maxSteps": 200, "maxDurationMs": 900000, "decisionTimeoutMs": 60000 }
}
```

Model-backed policies use any OpenAI-compatible Chat Completions endpoint. Put the provider in `.env`
(see [`env.example`](env.example)):

```dotenv
REA_MODEL=your-model
REA_BASE_URL=http://localhost:8000/v1
REA_API_KEY=your-api-key
```

```bash
npm run rea -- init gymnasium            # writes rea.config.json
python3 -m venv runtime/gym-venv && runtime/gym-venv/bin/python -m pip install -r requirements-gym.txt
export GYM_PYTHON="$PWD/runtime/gym-venv/bin/python"
npm run rea -- doctor rea.config.json    # checks configuration and local dependencies; connects to nothing
npm run rea -- run rea.config.json --steps 10 --output outputs/first-run
```

| Preset | What it connects | Guide |
|---|---|---|
| `demo` | Synthetic grid and joint fixtures | none needed |
| `gymnasium` | Registered Gymnasium tasks and their native action spaces | [framework reference](docs/framework.md#gymnasium-and-additional-robots) |
| `libero` | LIBERO / LIBERO-Pro suites | [LIBERO](docs/benchmarks/libero.md) |
| `arc-agi3` | Official ARC-AGI-3 SDK, games and scorecards | [ARC-AGI-3](docs/benchmarks/arc-agi3.md) |
| `aloha` | ACT dual-arm simulation; Classic ROS1 ALOHA hardware | [robots](docs/robots/README.md), [ALOHA hardware](docs/robots/aloha-real.md) |
| `lerobot` | SO-100/101, bimanual SO followers, Koch, Robot-compatible factories | [robots](docs/robots/README.md#lerobot-so-100so-101-biso-and-koch) |
| `openral` | Manifest-driven arms, dual arms, mobile platforms and humanoids | [OpenRAL and RPent](docs/robots/openral-rpent.md) |
| `rpent` | LIBERO, RoboCasa and RoboTwin environment services | [OpenRAL and RPent](docs/robots/openral-rpent.md) |
| `g1` | Unitree G1 camera, safety gate and executor | [Unitree G1](docs/robots/unitree-g1.md) |

BEHAVIOR-1K and RoboTwin have dedicated runners: [BEHAVIOR](docs/benchmarks/behavior.md),
[RoboTwin](docs/benchmarks/robotwin.md). Hardware presets start in preview (`execute: false`); task success on hardware
needs an evaluator, not a command acknowledgement.

## Routed and asynchronous systems

```js
systems: {
  mode: "routed",
  system0: [
    { id: "fix", kind: "program", policy: { type: "module", options: { module: "./fix-program.mjs" } } },
    { id: "vla", kind: "vla", policy: { type: "lingbot-vla", options: { url: "ws://127.0.0.1:8006" } } },
  ],
  system1: { type: "route-memory", options: { table: "./route-table.json", recoverOn: "judge" } },
  system2: { type: "judge", options: { model: "your-vision-model", threshold: 0.6 } },
  maxStepsPerRun: 50,
}
```

`tools/route-table.mjs` builds the route table from one development evaluation per capability. See
[routed systems](docs/routed-systems.md) and, for background thinking with per-action routing,
[asynchronous systems](docs/async-systems.md).

## Plug in your own

| To add | Implement | Guide |
|---|---|---|
| An LLM agent, VLA, code policy or controller | `decide(input) → {action}` | [System 0 policies](docs/extending/policies.md) |
| A router or a judge / brain | `route(input)`, `judge(input)` or `think(input)` | [Routers and brains](docs/extending/systems.md) |
| A benchmark or simulator | `reset`, `step` (and `score` for routed episodes) | [Environments](docs/extending/environments.md) |
| A program-executing simulator | the JSON-lines program protocol | [Program backends](docs/extending/program-backends.md) |
| A real robot | a Python driver class (`python-robot`) or a JavaScript driver | [Real robots](docs/extending/robots.md) |

Load any of them from a configuration with `type: "module"`, or register them on `createRegistry()`; nothing in `src/`
needs to change.

```js
import { createEmbodiedAgent } from "rea-agent";   // inside this repository: ./src/framework/index.mjs

const agent = await createEmbodiedAgent("./rea.config.json");
try {
  const episode = await agent.run({ seed: 3 });
  const evaluation = await agent.evaluate({ outputDir: "outputs/eval" });
} finally {
  await agent.close();
}
```

## Documentation

- [Architecture](docs/architecture.md): the three systems, the two runtimes, contracts and repository layout
- [Framework reference](docs/framework.md): configuration, CLI, environment and policy contracts, outputs
- [Routed systems](docs/routed-systems.md) · [Asynchronous systems](docs/async-systems.md)
- Extending: [policies](docs/extending/policies.md) · [routers and brains](docs/extending/systems.md) ·
  [environments](docs/extending/environments.md) · [program backends](docs/extending/program-backends.md) ·
  [real robots](docs/extending/robots.md)
- Benchmarks: [LIBERO](docs/benchmarks/libero.md) · [BEHAVIOR](docs/benchmarks/behavior.md) ·
  [RoboTwin](docs/benchmarks/robotwin.md) · [ARC-AGI-3](docs/benchmarks/arc-agi3.md)
- Robots: [ALOHA, LeRobot](docs/robots/README.md) · [ALOHA hardware](docs/robots/aloha-real.md) ·
  [OpenRAL and RPent](docs/robots/openral-rpent.md) · [Unitree G1](docs/robots/unitree-g1.md) ·
  [G1 cerebellum](agent-vla-async/README.md) · [LingBot-VLA](docs/policies/lingbot-vla.md)
- [Testing](docs/testing.md) · [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [Third-party notices](NOTICE.md)

## What is not included

Model weights, VLA servers, simulators and their assets (LIBERO, BEHAVIOR-1K / OmniGibson, RoboTwin), ASPIRE, datasets,
robot calibration and recorded runs are installed separately; each guide says what an integration expects. Benchmark
task prompts and API docstrings from third-party projects are not redistributed.

## License

MIT; see [LICENSE](LICENSE). Third-party components and attributions are listed in [NOTICE.md](NOTICE.md).
