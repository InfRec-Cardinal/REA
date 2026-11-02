# LIBERO integration

REA connects its policies and runtimes to LIBERO / LIBERO-Pro through a Python backend.
The environment supplies task language, external and wrist RGB images, robot
proprioception and native task completion. Policies return Cartesian, rotation
and gripper actions; the backend applies the configured control-step repeat count.

## Setup

Install the LIBERO-Pro/Long distribution and its assets in a dedicated Python
environment. Set `LIBERO_PYTHON` to that environment's interpreter. The backend
is `scripts/libero_backend.py`; its JavaScript bridge is `src/libero-client.mjs`.
The adapter expects the Pro/Long suite names supported by that backend.

Configure a model through `REA_MODEL`, `REA_BASE_URL` and `REA_API_KEY`, then:

```bash
LIBERO_PYTHON=/path/to/libero-env/bin/python \
LIBERO_SUITE=libero_10_task LIBERO_TASKS=0,1 REA_SEEDS=51 \
REA_MODEL=your-model REA_BASE_URL=http://localhost:8000/v1 \
npm run rea -- run examples/libero.mjs --output outputs/libero
```

The example accepts comma-separated `LIBERO_TASKS` and `REA_SEEDS`.
`LIBERO_HORIZON` sets the native control-step budget; `REA_MAX_STEPS` limits
policy decisions. `REA_POLICY_MODE` selects tool-call or JSON policy output.
Use a rendering backend supported by the installed MuJoCo environment.

## Observation and action contract

The policy receives task text, camera images, proprioception and recent actions.
Object ground-truth state and simulator success predicates are not policy inputs.
The environment checks observation identity before dispatch and reports native
control steps separately from policy decisions.

Seeds 1–50 select the official initial-state array at index `seed - 1`; seeds
51–100 use seeded development resets. The task's success check and control horizon
supply termination. Configure action amplitude and repeat count in the episode's
`task` object; see [the framework reference](../framework.md#libero).

## Frozen programs

Code-as-policy programs run inside ASPIRE's LIBERO code-execution environment through a
[program backend](../extending/program-backends.md):

- `scripts/aspire_backend.py`: one program per `run_program` call, sealed `score`;
- `scripts/async_libero_backend.py` with `src/environments/libero-programs.mjs`: programs paused at real control-step
  boundaries (`requirements-async-libero.txt`), so a [routed](../routed-systems.md) or [asynchronous](../async-systems.md)
  episode can continue a program, switch to another one from the current physical state, and read success once at the
  end. `tools/build_program_bundle.py` freezes a program library (`<suite>/<task>/<candidate>.py`) into the bundle the
  backend reads from `REA_PROGRAM_BUNDLE`.

Programs are not included. Build route tables and calibrate judges on development seeds 51-100 with
`reset: "seeded_random"` (randomized initial states), and evaluate on the official initial states (seeds 1-50,
`reset: "aspire"`). Under ASPIRE's reset rule seed *s* uses official state (*s*-1) mod 50, so development seeds with
`reset: "aspire"` would replay the evaluation states; the backend's `phase: "development"` / `"heldout"` checks the seed
ranges, and `createLiberoProgramEnvironment({ phase, reset })` passes both through.

## Checks

```bash
node --test test/libero*.test.mjs test/program-client.test.mjs
python3 -m unittest discover -s test -p test_libero_backend.py
```

Simulator checks require the external runtime and assets. The shared runner writes
local manifests, transitions and summaries under the selected output directory;
these generated artifacts are not source files.
