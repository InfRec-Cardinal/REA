# RoboTwin VLA and Code policies

This adapter connects native VLA services and a Python Code policy to REA's
System0 interface. A trained success router implements System1: it predicts an
independent success probability for each candidate from the initial observation,
selects the largest score, and retains that policy for the episode. The selected
policy continues to receive fresh observations after executing each action.

The VLA candidates are the keys of `vlaEndpoints` in the configuration: any service that speaks the protocol below can
be listed (the reference setup used `pi05`, `gigabrain`, `spatial_forcing`, `xvla`, `xwam`, `patchwam` and `medex`;
`xvla` and `xwam` are end-effector services, the others joint-space unless their metadata says otherwise). The router
needs a trained output for every candidate. The optional `code` candidate generates Python that computes one gripper
target per decision. Mid-episode policy switching is not implemented by this router.

## Requirements

- Node 22.19 or newer.
- A RoboTwin installation and a compatible numeric-control harness. Configure
  `harnessRoot` to a directory exposing `robotwin_subset.make_subset_env`,
  `robotwin_control` (`robot_state`, `camera_calibration`, `validate_action`,
  `execute`), and `observation_io.plain`. `subset` identifies the harness's scene
  bundle. These simulator and harness components are external dependencies.
- Separately running VLA WebSocket services. This adapter supplies clients, not
  model weights or model-server launchers. Each service must support metadata,
  reset, and inference using the protocol below.
- Install `requirements-lingbot-vla.txt` into `clientPython` for NumPy, Pillow,
  MessagePack and WebSocket support. Install `requirements-success-router.txt`
  into the Python environment used to serve or train the success router.
- A local Qwen3.5-4B model and a compatible trained success-router checkpoint.
  The checkpoint contains `router_config.json`, `training_plan.json`, `head.pt`,
  `adapter/`, and `processor/`. Model weights and collected episode data are not
  included in the repository.

The router's model and input encoder are included under `scripts/success_router/`.
The encoder combines the task text, three initial RGB images, robot end poses,
joint state and action-output contract. It uses the final non-padding hidden
state from Qwen with a LoRA adapter, followed by a linear head and independent
sigmoid outputs. An optional `--training-root` can load an external checkpoint's
matching `model.py` and `train.py` instead of the bundled implementation.

## Configuration and execution

Copy `config.example.json` and fill the interpreter paths, harness paths, GPU
assignment and service endpoints. Select simulator GPUs through `simulatorEnv`.
The example endpoints are placeholders and do not start any model services.

For the `chat` Code backend, provide `REA_BASE_URL` and `REA_API_KEY` through
the environment. For the official Codex CLI backend, add these configuration
fields, with a Responses-compatible endpoint and a local key file:

```json
{
  "codeBackend": "codex",
  "codeModel": "YOUR_MODEL_ID",
  "codeCodexPath": "/path/to/codex",
  "codeProvider": "configured_provider",
  "codeProviderBaseUrl": "https://api.example.com/v1",
  "codeApiKeyFile": "/path/to/provider.key",
  "codeWorkRoot": "/path/to/local-runs/code-cli"
}
```

The CLI receives four camera images and the observation/history prompt. Its
built-in tools are disabled; the Python worker executes the generated program.
Configure the same Code backend, model and action budget when collecting labels
and when using the fitted Code head.

Start the router, then run an instance manifest:

```bash
python scripts/success_router_server.py \
  --checkpoint /path/to/router-checkpoint \
  --model-path /path/to/Qwen3.5-4B --port 8810

node eval/robotwin/hybrid/run.mjs --config config.json \
  --instances instances.json --output runs/hybrid --mode hybrid
```

`instances.json` is a JSON array in the configured harness's scene format. Each
record must include `instance_id`, `task`, `setting`, `seed`, `instruction`, and
any scene-selection fields required by the harness, such as `setup_now_ep_num`.

Modes:

| Mode | Candidate pool |
| --- | --- |
| `hybrid` | Every VLA in `vlaEndpoints` plus Code; requires trained outputs for all of them |
| `vla` | The VLAs in `vlaEndpoints` only |
| `code` | Fixed Code policy, without a router call |
| A VLA identifier from `vlaEndpoints` | Fixed VLA policy, without a router call |

VLA clients preserve the service's native action chunks. Joint policies use
14-dimensional actions; end-effector policies use 16-dimensional actions.
Execution stops on the simulator's success condition or native action limit.
The Code policy receives four RGB views, robot state, calibration and execution
feedback. Its Python program uses NumPy and calls
`move_gripper(arm, position, quaternion, opening)` exactly once. The target uses
world-frame metres, a WXYZ quaternion, and a gripper opening from 0 to 1.
`codeDecisions` defaults to 40; a Python or target-validation error consumes one
decision and supplies feedback without moving the robot.

The `chat` backend uses 2,048 output tokens per call and temperature zero. The
`codex` backend uses CLI-default reasoning and temperature, with a 240-second
decision deadline and no configured output-token cap. These are distinct
execution protocols; keep the chosen protocol consistent when collecting labels.

Each output directory stores decisions, transitions and initial observations for
local inspection. Completed episodes are reused on restart. `--skip-errors`
preserves infrastructure errors as unknown outcomes and continues with untried
scenes; it does not turn them into unsuccessful training labels.

## VLA service protocol

The client uses the MessagePack/NumPy WebSocket transport implemented in
`scripts/lingbot_vla_client.py`. The service sends metadata on connection,
including `action_type` (`qpos` or `ee`) and `execute_horizon`, and accepts:

- Reset: `{"reset": true, "policy_seed": 42}`.
- Inference: `instruction`, `observation` with RGB arrays under `head_camera`,
  `left_camera`, and `right_camera`, plus `joint_action` and `endpose`.
- Reply: an `actions` array, optionally with `action_type`, `execute_horizon`
  and `inference_s`. An action horizon must come from metadata, the reply or the
  client's configured `chunkSteps`.

VLA sessions carry episode state. Give concurrent workers independent service
sessions. Use `parallel.py --configs worker-a.json worker-b.json ...` for static
shards, or `dynamic_parallel.py` for a persistent queue of Code episodes. Both
accept `--instances`, `--output`, `--mode` and `--skip-errors`. Each supplied
configuration represents one worker, so multiple workers can share a GPU when
resources permit.

## Adding a Code success head

Collect Code labels on separate training and development manifests. Run the
original VLA router on the development scenes to obtain paired outcomes:

```bash
node eval/robotwin/hybrid/run.mjs --config config.json \
  --instances calibration-train.json --output runs/code-train --mode code
node eval/robotwin/hybrid/run.mjs --config config.json \
  --instances calibration-dev.json --output runs/code-dev --mode code
node eval/robotwin/hybrid/run.mjs --config config.json \
  --instances calibration-dev.json --output runs/vla-dev --mode vla

python eval/robotwin/hybrid/export_labels.py \
  --run-root runs/code-train --output train.jsonl
python eval/robotwin/hybrid/export_labels.py \
  --run-root runs/code-dev --vla-run-dir runs/vla-dev --output dev.jsonl

python scripts/train_success_router_code_head.py \
  --checkpoint /path/to/vla-router --model-path /path/to/Qwen3.5-4B \
  --train-jsonl train.jsonl --dev-jsonl dev.jsonl \
  --test-instances test-instances.json --output checkpoints/vla-code
```

The trainer freezes Qwen, its LoRA adapter and all existing VLA outputs, caches
their shared features, and fits one new linear output with BCE. With paired
development outcomes, it selects the checkpoint by development routing success,
breaking ties by Code BCE. Without paired outcomes, it selects by Code BCE.
The supplied test manifest is used only to exclude its scenes from fitting.

The exporter accepts static or dynamic parallel output roots. `--instances`
selects a subset of a combined run; `--allow-partial` exports valid labels and
records omitted episodes. Restart the router with the new checkpoint to enable
`hybrid` mode. `capture_initials.py` and `score_initials.py` can also capture
initial inputs and score them independently of executing a policy.

## Tests

```bash
node --test test/success-router.test.mjs test/robotwin-vla.test.mjs \
  test/robotwin-code.test.mjs test/robotwin-codex.test.mjs \
  test/robotwin-hybrid-run.test.mjs
python -m unittest discover -s test -p 'test_success_router*.py'
python -m unittest discover -s test -p 'test_robotwin_hybrid*.py'
```
