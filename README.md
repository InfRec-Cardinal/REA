<h1 align="center">REA</h1>

<p align="center">
  <strong>A Routed Embodied Agent Runtime for Asynchronous Brain-Cerebellum Control</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2610.42053"><img src="https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square" alt="Paper PDF"></a>
  <a href="https://arxiv.org/abs/2610.42053"><img src="https://img.shields.io/badge/arXiv-2610.42053-B31B1B?style=flat-square&amp;logo=arxiv&amp;logoColor=white" alt="arXiv 2610.42053"></a>
  <a href="https://huggingface.co/papers/2610.42053"><img src="https://img.shields.io/badge/Hugging%20Face-Paper-FFD21E?style=flat-square&amp;logo=huggingface&amp;logoColor=white" alt="Hugging Face Paper"></a>
  <a href="https://github.com/InfRec-Cardinal/REA"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&amp;logo=github&amp;logoColor=white" alt="GitHub"></a>
  <a href="https://mikeeeeeeeeeeeeee.github.io/REA"><img src="https://img.shields.io/badge/Project%20Page-2088A2?style=flat-square&amp;logo=googlechrome&amp;logoColor=white" alt="Project Page"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-BibTeX-7042C1?style=flat-square&amp;logo=bookstack&amp;logoColor=white" alt="Citation BibTeX"></a>
  <a href="https://github.com/InfRec-Cardinal/REA/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/InfRec-Cardinal/REA/ci.yml?branch=main&amp;style=flat-square&amp;logo=githubactions&amp;logoColor=white&amp;label=CI" alt="CI"></a>
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

> 🚧 **Continuously updated.** REA is under active development. Support for more benchmark platforms (such as
> [RoboDojo](docs/benchmarks/robodojo.md)) and more real robots is added on an ongoing basis; see [Platforms](#platforms)
> and the [changelog](CHANGELOG.md). Contributions of new platforms and robots are welcome.

REA is a framework for embodied agents that route among fast policies instead of putting a large model in the control
loop. It has three layers:

- **System 0**, capabilities that act at control rate: frozen code programs, VLA policies, LLM tool-calling
  controllers, motion and whole-body controllers.
- **System 1**, a cheap router: route memory picks the capability that did best on development episodes, a judge gate
  checks completion only when a capability returns, and an escalation ladder decides what runs next under one shared
  budget.
- **System 2**, the brain: it writes or selects capabilities offline, and online it only judges or advises at
  boundaries. No running capability waits for it.

The same interfaces connect benchmark platforms (LIBERO / LIBERO-Pro, BEHAVIOR-1K, RoboTwin, RoboDojo, ARC-AGI-3,
Gymnasium), real robots (ALOHA, LeRobot SO-100/101 and Koch, OpenRAL and RPent robots, Unitree G1, RoboDojo-RealEval's
dual-arm robots, your own driver) and methods (LLM agents, VLAs, code-as-policy, controllers). REA can drive a platform's
environment, or be called as a policy by a platform that runs its own loop. Every episode records its decisions, actions,
timing and usage in the same format.

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

BEHAVIOR-1K, RoboTwin and RoboDojo have dedicated runners or adapters: [BEHAVIOR](docs/benchmarks/behavior.md),
[RoboTwin](docs/benchmarks/robotwin.md), [RoboDojo](docs/benchmarks/robodojo.md); every platform is listed under
[Platforms](#platforms). Hardware presets start in preview (`execute: false`); task success on hardware needs an
evaluator, not a command acknowledgement.

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

## Platforms

New platforms and robots are added continuously; the [changelog](CHANGELOG.md) lists each addition. **Supported**: the
adapter is included and tested against the platform's interface. **New**: the adapter is included and tested offline, and
validation on the platform is in progress.

| Benchmark platform | How REA connects | Status |
|---|---|---|
| [LIBERO / LIBERO-Pro](docs/benchmarks/libero.md) | `libero` environment; resumable frozen-program backend | Supported |
| [BEHAVIOR-1K](docs/benchmarks/behavior.md) (R1Pro) | program backend; online-agent episode runner | Supported |
| [RoboTwin 2.0](docs/benchmarks/robotwin.md) | VLA / code candidates with a success router; RPent and OpenRAL scenes | Supported |
| [ARC-AGI-3](docs/benchmarks/arc-agi3.md) | official SDK, games and scorecards | Supported |
| [Gymnasium](docs/framework.md#gymnasium-and-additional-robots) | any registered environment and its native action space | Supported |
| [RoboDojo](docs/benchmarks/robodojo.md) (simulation and RealEval) | REA as an XPolicyLab policy through `rea serve` | New |
| Other XPolicyLab platforms (RoboTwin, RMBench) | the same XPolicyLab adapter | New |
| Your platform | an [environment](docs/extending/environments.md) (REA runs the loop) or [`rea serve`](docs/extending/policy-server.md) (the platform runs it) | |

| Real robot | How REA connects | Status |
|---|---|---|
| ALOHA: ACT simulation, Classic ROS1 hardware | `aloha` | Supported |
| LeRobot: SO-100/101, bimanual SO, Koch, Robot-compatible factories | `lerobot` | Supported |
| OpenRAL robots: Franka, UR5e/UR10e, Sawyer, WidowX, OpenArm, humanoids and more | `openral` | Supported |
| Unitree G1 (service, safety gate, cerebellum) | `g1` | Supported |
| RoboDojo-RealEval: ARX X5, Piper, Piper X | the XPolicyLab adapter | New |
| Your robot | a [Python driver class](docs/extending/robots.md) (`python-robot`), a JavaScript driver, or `rea serve` from your robot stack | |

## Plug in your own

| To add | Implement | Guide |
|---|---|---|
| An LLM agent, VLA, code policy or controller | `decide(input) → {action}` | [System 0 policies](docs/extending/policies.md) |
| A router or a judge / brain | `route(input)`, `judge(input)` or `think(input)` | [Routers and brains](docs/extending/systems.md) |
| A benchmark or simulator | `reset`, `step` (and `score` for routed episodes) | [Environments](docs/extending/environments.md) |
| A program-executing simulator | the JSON-lines program protocol | [Program backends](docs/extending/program-backends.md) |
| A real robot | a Python driver class (`python-robot`) or a JavaScript driver | [Real robots](docs/extending/robots.md) |
| A platform that runs its own loop (XPolicyLab / RoboDojo, challenge evaluators, robot stacks) | `rea serve` plus a small adapter; a Python client is included | [Policy server](docs/extending/policy-server.md) |

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
  [real robots](docs/extending/robots.md) · [policy server](docs/extending/policy-server.md)
- Benchmarks: [LIBERO](docs/benchmarks/libero.md) · [BEHAVIOR](docs/benchmarks/behavior.md) ·
  [RoboTwin](docs/benchmarks/robotwin.md) · [RoboDojo](docs/benchmarks/robodojo.md) · [ARC-AGI-3](docs/benchmarks/arc-agi3.md)
- Robots: [ALOHA, LeRobot](docs/robots/README.md) · [ALOHA hardware](docs/robots/aloha-real.md) ·
  [OpenRAL and RPent](docs/robots/openral-rpent.md) · [Unitree G1](docs/robots/unitree-g1.md) ·
  [G1 cerebellum](agent-vla-async/README.md) · [LingBot-VLA](docs/policies/lingbot-vla.md)
- [Testing](docs/testing.md) · [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [Third-party notices](NOTICE.md)

## What is not included

Model weights, VLA servers, simulators and their assets (LIBERO, BEHAVIOR-1K / OmniGibson, RoboTwin), ASPIRE, datasets,
robot calibration and recorded runs are installed separately; each guide says what an integration expects. Benchmark
task prompts and API docstrings from third-party projects are not redistributed.

## Citation

```bibtex
@article{zhou2026rea,
  title   = {{REA}: A Routed Embodied Agent Runtime for Asynchronous Brain-Cerebellum Control},
  author  = {Zhou, Pengfei and Ma, Yixing and Zhang, Zhengfeiyang and Chen, Haoyuan and Wan, Zhenglin and Wang, Hexin and Zhao, Wangbo and You, Yang},
  journal = {arXiv preprint arXiv:2610.42053},
  year    = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE). Third-party components and attributions are listed in [NOTICE.md](NOTICE.md).
