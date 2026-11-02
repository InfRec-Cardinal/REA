# BEHAVIOR integration and testing

The framework's chat policy can consume BEHAVIOR R1Pro demonstration observations
and produce the default robot's 23-dimensional action vector. The offline runner
below compares those predictions with demonstration actions. It does not execute
actions or measure task success.

## Run an offline dataset check

Use a separate Python environment for the dataset reader:

```bash
python3 -m venv .venv-behavior-offline
.venv-behavior-offline/bin/python -m pip install -r requirements-behavior-offline.txt
.venv-behavior-offline/bin/python scripts/behavior_samples.py \
  --dataset /path/to/2026-challenge-demos \
  --output outputs/behavior-samples
```

The default selection is fixed before inference: task indices 0, 1 and 2, the
lowest episode index for each task, and frames at 0%, 25% and 50% of that episode.
`--tasks` and `--fractions` can specify another selection. Dataset files are read
only; use a new output directory for each export.

Set `REA_MODEL`, `REA_BASE_URL` and `REA_API_KEY` in your shell or `.env` using
the same configuration as the unified CLI. `REA_BASE_URL` and `REA_API_KEY`
are also accepted. Then run:

```bash
node scripts/behavior_offline.mjs \
  --samples outputs/behavior-samples/samples.jsonl \
  --output outputs/behavior-offline \
  --model your-vision-model
```

The runner uses the existing `createChatPolicy` through the policy registry, with
JSON actions, temperature 0, a 1024-token output budget, and a 60-second decision
timeout. `--mode tool` selects tool calls. DeepSeek defaults to non-thinking mode,
matching the existing LIBERO control path. Each frame is evaluated independently:
no previous predictions, future observations, reference actions, rewards or
termination flags are sent to the policy.

RGB images retain their native resolution: head 720×720, wrists 480×480. The
exporter maps each episode-local timestamp through its LeRobot v3 video-shard
offset and encodes the selected frame as JPEG. This check uses RGB and
proprioception; it does not use the provided depth videos.

Output files:

- `manifest.json`: selected samples, policy settings and measurement scope.
- `decisions.jsonl`: predictions, reference actions, comparison baseline, errors,
  brief model explanations, latency and available API usage.
- `summary.json`: action validity, grouped action MAE, latency and available usage.

MAE is calculated separately for base commands, torso radians, left/right arm
radians and left/right gripper commands. The baseline stops the base and retains
the observed joint positions. Agent and baseline use the same valid sample
subset; invalid predictions remain in the total sample/error counts. A lower
offline error does not necessarily imply greater task success, because more than
one action can make progress and a one-frame hold is a strong baseline at 30 Hz.

## Default R1Pro contract

`src/environments/behavior-actions.mjs` exports the action schema, state layout,
`describeR1ProState(raw61)` and `createR1ProHoldAction(raw61)`.

| Action indices | Meaning | Input units |
|---|---|---|
| 0–2 | Robot-local base vx, vy, yaw rate | `[-1,1]`, scaled by `[0.75,0.75,1]` to m/s, m/s, rad/s |
| 3–6 | Torso joint targets | Absolute radians |
| 7–13 | Left arm joint targets | Absolute radians |
| 14 | Left gripper | `-1` closed, `+1` open |
| 15–21 | Right arm joint targets | Absolute radians |
| 22 | Right gripper | `-1` closed, `+1` open |

These are the official default R1Pro controllers, not a contract for every custom
robot. The 61-value state contains measured joint positions/velocities and end
effector poses relative to the robot base. A zero vector is not a pose-holding
action: it would command torso and arm joints toward zero.

## Online evaluation prerequisites

For a closed-loop challenge score, use the official
[BEHAVIOR v3.9.2 evaluator](https://behavior.stanford.edu/challenge/evaluation.html).
Its score includes partial goal completion (`q_score`), simulated time and motion
efficiency. Reported evaluation uses public instance indices 0–9 and the official
default task horizons; video output is required. Public instances 10–19 can be
used for development.

Online evaluation requires an Isaac Sim-compatible GPU and an OmniGibson data
root containing simulation assets plus `2026-challenge-task-instances`, including
its metadata and scene instance files. A LeRobot demonstration directory is not
that data root. GPUs without RT cores are not supported by Isaac Sim (see NVIDIA's Isaac Sim requirements); on such
GPUs the BEHAVIOR backend can render perception frames through path tracing (`REA_BEHAVIOR_PT_OBS=1`).

The existing OpenRAL R1Pro scene adapter is an available simulation integration
path; see [the robot guide](../robots/openral-rpent.md). Official challenge evaluation
also supports a websocket policy server. The offline runner does not start either simulator integration; use the
[online adapter guide](../../eval/behavior/README.md) for the separate execution path.
