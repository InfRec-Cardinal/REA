# ARC-AGI-3

The framework can run a structured-action policy against the official ARC-AGI-3 SDK through
the same episode/evaluation interface used for robots. The adapter supports
dynamic actions, ACTION6 coordinates, episode notes, action/time budgets,
recordings and official scorecards. This is a working integration and a baseline
policy interface; no leaderboard score is claimed.

## Install and verify locally

Use Node.js 22.19+ and Python 3.12+. ARC dependencies are separate from the G1
requirements. The integration was verified with `arc-agi 0.9.9` and
`arcengine 0.9.3`.

```bash
python3 -m venv runtime/arc-venv
runtime/arc-venv/bin/python -m pip install -r requirements-arc.txt
git clone --depth 1 https://github.com/arcprize/ARC-AGI.git runtime/arc-sdk
ARC_PYTHON="$PWD/runtime/arc-venv/bin/python" \
ARC_TEST_ENVIRONMENTS_DIR="$PWD/runtime/arc-sdk/test_environment_files" \
node --test test/arc-agi3.test.mjs
```

The opt-in SDK test initializes the official repository's `bt33` test environment,
reads all grid frames and PNG images, executes one coordinate action and closes
the official scorecard. It checks zero extra resets and correct action counting.
It uses no model or robot. These SDK test environments are integration fixtures,
not an evaluation on the full benchmark.

## Run a model policy

Use a vision-capable model endpoint, or a model able to consume the numeric grids.
The following runs the local `bt33` fixture. Replace the directory and game IDs
with your installed public environments for actual experiments:

```bash
ARC_PYTHON="$PWD/runtime/arc-venv/bin/python" \
ARC_ENVIRONMENTS_DIR="$PWD/runtime/arc-sdk/test_environment_files" \
ARC_RECORDINGS_DIR="$PWD/outputs/arc-recordings" \
ARC_GAMES=bt33 ARC_OPERATION_MODE=OFFLINE \
REA_MODEL=your-model REA_BASE_URL=http://localhost:8000/v1 \
REA_MAX_STEPS=200 \
npm run rea -- run examples/arc-agi3.mjs --output outputs/arc-baseline
```

Model credentials can be set with `REA_API_KEY` in `.env`; `REA_BASE_URL` and
`REA_API_KEY` are fallbacks. ARC service credentials use `ARC_API_KEY`, a separate
SDK setting. OFFLINE selects local environment execution; a remote model endpoint
still makes network requests and can incur its usual charges.

Useful settings: `ARC_GAMES` is comma-separated; `REA_SEEDS` defaults to `0`;
`REA_POLICY_MODE` is `tool` (default) or `json`; `REA_MAX_OUTPUT_TOKENS` defaults
to 2048. Episode and decision limits default to 900000 and 60000 milliseconds,
overridden by `REA_EPISODE_TIMEOUT_MS` and `REA_DECISION_TIMEOUT_MS`.

## SDK semantics

The adapter uses `Arcade.make`'s initial observation without an extra `reset`.
Each step refreshes available actions. ACTION6 accepts integer `x` (column) and
`y` (row) in 0–63; other action effects are left for the policy to discover. Every
frame is available as a numeric grid and a rendered PNG. The policy receives
public runtime observations, not game implementation files or baseline solutions.
These correspond to the official [EnvironmentWrapper interface](https://docs.arcprize.org/toolkit/environment_wrapper).

WIN ends the game. By default GAME_OVER leaves RESET available so the policy can
recover within its remaining budget; each reset is an explicit, counted policy
action. Set `endOnGameOver: true` on the environment factory for a single-attempt
episode. The policy cannot reset its own action budget. The SDK determines whether
a reset restarts a level or the game.

The example reuses one Arcade instance and scorecard across its game plan. It
closes the scorecard once at the end and writes the unchanged SDK response under
`summary.json` → `officialScorecard`. Intermediate game progress is not converted
into a synthetic reward or a guessed official score. Local seed values are passed
to the SDK; remote execution records the requested seed but does not claim the
remote API applied it.

## Competition and leaderboard scope

The adapter exposes `OFFLINE`, `ONLINE` and `COMPETITION`. OFFLINE is explicit by
default. A developer API leaderboard run can select `ARC_OPERATION_MODE=COMPETITION`
with the corresponding ARC API credentials. Use each game ID once and a single
seed; the example rejects repeated games or multi-seed Competition plans.

According to the [official Competition mode documentation](https://docs.arcprize.org/toolkit/competition_mode),
this mode uses API execution, permits one scorecard and one `make` per environment,
scores all available environments, and permits level resets. It does not expose
in-flight scorecards. The evaluator therefore requests only the final closed card.

The API leaderboard path and the Kaggle code competition require different
packaging. This repository does not yet generate a Kaggle submission notebook or
`submission.parquet`. The [official 2026 starter](https://docs.arcprize.org/arc-prize-2026)
documents the notebook/gateway submission flow and local model execution under
competition resource constraints. A remote-model development configuration alone
is not a validated Kaggle submission.

## Known limitations and next steps

- Run a fixed public game plan and held-out validation plan, saving official
  scores, action counts, failures, usage and runtime. SDK smoke tests do not measure
  problem-solving ability.
- Improve the policy's exploration and rule inference: track object changes,
  test hypotheses, retain learned rules in episode notes, and spend deliberate
  model calls where progress stalls. Compare against the same single-model/action
  budget before claiming the router helps.
- Implement and validate the chosen submission path, including its local model
  packaging and full game coverage. Publish the actual reproducible result and
  protocol, rather than treating adapter support as benchmark performance.

The same experiment loop can evaluate these policy changes across ARC, Gymnasium
and manipulation tasks. Whether a shared policy improves all of them remains an
empirical question.
