# LingBot-VLA 2.0 RoboTwin policy

The built-in `lingbot-vla` policy connects to the official
[LingBot-VLA 2.0 WebSocket server](https://github.com/Robbyant/lingbot-vla-v2/blob/main/deploy/lingbot_vla_v2_policy.py).
It sends task language, named RGB cameras and robot state, then returns a native
environment trajectory from the predicted action chunk through the regular
framework `policy` contract.

The supplied [configuration](../../examples/lingbot-vla.config.mjs) connects a
RoboTwin checkpoint to the existing RPent RoboTwin adapter. It uses real service
protocols; both the model server and the simulator must be running separately.

## Start the services

### 1. Model server on the inference host

Install the inference environment using the
[official setup instructions](https://github.com/Robbyant/lingbot-vla-v2#installation).
Use a checkpoint trained for RoboTwin with its matching robot configuration and
normalization statistics. The general pretrained checkpoint alone does not establish
RoboTwin task performance.

Run from the LingBot-VLA 2.0 checkout:

```bash
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct \
python -m deploy.lingbot_vla_v2_policy \
  --model_path /path/to/experiment/checkpoints/global_step_50000/hf_ckpt \
  --port 8006 \
  --chunk_ret true \
  --use_length 50 \
  --use_bf16 false \
  --use_fp32 true \
  --use_compile false
```

`--model_path` points at the exported `hf_ckpt` directory. Keep the experiment's
`lingbotvla_cli.yaml` three parent directories above it, as required by the
[upstream loader](https://github.com/Robbyant/lingbot-vla-v2/blob/main/deploy/lingbot_vla_v2_policy.py).
Its `data.norm_stats_file` must resolve on the inference host. The server also
loads `configs/robot_configs/robotwin.yaml` relative to its working directory.
`QWEN3VL_PATH` selects the Qwen3-VL backbone/processor files.

The command uses the upstream FP32 evaluation setting and disables compilation
for a simpler first run. Use the upstream deployment guide for other precision
and compilation settings. Inference and training dependencies belong on this host;
the framework client does not need PyTorch or the model weights.

Keep **one server per active episode/client**. The official server owns a shared
policy whose reset, robot configuration and action history are mutable.

### 2. RPent RoboTwin service on the simulation host

Install RPent and its RoboTwin runtime/assets following the
[RPent setup](../robots/openral-rpent.md#rpent-coverage-and-setup). From the RPent checkout, run:

```bash
python robots/robotwin/env_server.py \
  --transport http --host 127.0.0.1 --port 8803 \
  --task-name adjust_bottle --task-config demo_clean --seed 0 \
  --assets-path /path/to/robotwin-assets
```

The service owns the task, task configuration and seed. Match those values in the
framework configuration; changing a framework episode does not reconfigure a
running RPent server. The service command and state contract are documented in
[RPent's server](https://github.com/RLinf/RPent/blob/main/robots/robotwin/env_server.py).

### 3. Framework client

From this repository, install the small policy-client dependencies in a Python
environment:

```bash
python3 -m venv .venv-lingbot-client
.venv-lingbot-client/bin/python -m pip install -r requirements-lingbot-vla.txt
```

Then run the supplied policy configuration. `RPENT_PYTHON` points at a Python
environment that can import RPent and NumPy; it may differ from the lightweight
policy-client environment.

```bash
LINGBOT_VLA_PYTHON="$PWD/.venv-lingbot-client/bin/python" \
LINGBOT_VLA_URL=ws://127.0.0.1:8006 \
RPENT_PYTHON=/path/to/rpent-env/bin/python \
RPENT_ROOT=/path/to/RPent \
RPENT_ENDPOINT=http://127.0.0.1:8803 \
RPENT_TASK=adjust_bottle RPENT_TASK_CONFIG=demo_clean REA_SEED=0 \
npm run rea -- run examples/lingbot-vla.config.mjs \
  --events --output outputs/lingbot-vla
```

Use service addresses reachable from the framework host. The configuration enables
execution in the turn-based RoboTwin simulator and keeps the external service alive
when the client closes.

`REA_INSTRUCTION` adds task guidance. The environment supplies the actual task
language. `REA_MAX_STEPS` defaults to 200 policy actions; each action can contain
several simulator control steps.

### Optional: run as a System0 candidate

The same policy can also be a System0 candidate in the experimental, opt-in
[System0/1/2 runtime](../async-systems.md). The
[async configuration](../../examples/lingbot-vla-async.config.mjs) uses the same
services and environment variables:

```bash
npm run rea -- run examples/lingbot-vla-async.config.mjs \
  --events --output outputs/lingbot-vla-async
```

To enable System2, set `REA_MODEL`, `REA_BASE_URL` and `REA_API_KEY` for an
OpenAI-compatible model before the same command. Without `REA_MODEL`, the async
example runs System0 through the default System1 router. With it, System2 reasons
in the background and can supply directives at later action boundaries; the policy
keeps such directives next to the environment's task text. The example has one
candidate; add other System0 policies to make controller switching meaningful.
In that runtime one trajectory is one System1 routing boundary: System2 can think
while RPent executes the chunk, and a new route takes effect after the chunk
returns.

## RoboTwin observation and action mapping

The client decodes the framework's named PNG images into RGB arrays for the
upstream NumPy/MessagePack protocol. The defaults match
[LingBot's RoboTwin evaluation client](https://github.com/Robbyant/lingbot-vla-v2/blob/main/experiment/robotwin/eval_policy_client_lingbotvla.py)
and [robot feature configuration](https://github.com/Robbyant/lingbot-vla-v2/blob/main/configs/robot_configs/robotwin.yaml):

| LingBot request field | Framework observation source |
| --- | --- |
| `observation.images.cam_high` | Image named `head` |
| `observation.images.cam_left_wrist` | Image named `left_wrist` |
| `observation.images.cam_right_wrist` | Image named `right_wrist` |
| `observation.state` | `state.robot_state.qpos_target14` |
| `task` | Task language and current instruction |

The state path is deliberate: RPent's
[`qpos_target14`](https://github.com/RLinf/RPent/blob/main/robots/robotwin/rlinf_env.py)
concatenates `get_left_arm_jointState()` and `get_right_arm_jointState()`.
RoboTwin builds the
[`joint_action.vector`](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/_base_task.py)
used by LingBot from those same getters. This follows the checkpoint's native
observation convention; it is not RPent's separate measured `arm_qpos_real12`.

The response's `action` feature is converted to:

```js
{
  name: "rpent_qpos_trajectory",
  arguments: { commands: actionRows }
}
```

Each row contains left six joint positions, left gripper, right six joint
positions, then right gripper. Joints use radians; grippers use `0` closed to `1`
open. These are absolute targets. The server converts its internal canonical
representation back to this native 14-value action; the framework does not send
the internal 55-dimensional vector to the simulator.

By default, `chunkSteps: 8` executes the first eight returned rows and discards
the remaining prediction. The next decision uses a new observation and a fresh
inference request. Use `LINGBOT_VLA_CHUNK_STEPS` to change this in the example.
Keep the server's `--chunk_ret true`: its single-action mode can reuse cached
predictions across requests, which is a different control schedule.

RPent executes its native chunk, can stop early on episode completion, and reports
actual `controlSteps` in the transition. Choosing eight rather than the upstream full 50-step chunk
changes the inference schedule, so this example is not a reproduction of the
published benchmark settings.

## Policy options

| Option | Default | Meaning |
| --- | --- | --- |
| `url` | `LINGBOT_VLA_URL` or `ws://127.0.0.1:8006` | Official model server |
| `python` | `LINGBOT_VLA_PYTHON` or `python3` | Python with `requirements-lingbot-vla.txt` installed |
| `robot` | `robotwin` | Server-side robot configuration selected on reset |
| `imageMap` | Three mappings above | Request field to framework image name |
| `stateMap` | `{"observation.state":"state.robot_state.qpos_target14"}` | Request field to a path within the observation |
| `actionKeys` | `["action"]` | Native response features concatenated in this order |
| `actionName` | `rpent_qpos_trajectory` | Environment action name |
| `actionArgument` | `commands` | Action argument receiving the mapped output |
| `actionFormat` | `chunk` | `chunk` for a list of commands; `single` for one command |
| `jointNames` | Omitted | Map each ordered output row to a named command object |
| `chunkSteps` | `8` | Maximum predicted rows executed per decision |
| `requestTimeoutMs` | `30000` | Policy request timeout |

The supplied example sets the timeout to 120 seconds for the first model request.
`LINGBOT_VLA_TIMEOUT_MS` changes both its policy timeout and decision budget.
`stateMap` accepts dot-separated paths or arrays of path segments when an
observation contains literal dots in its keys. For example,
`{"observation.state": ["state", "joint.position"]}` reads that exact field.

For [ALOHA hardware](../robots/aloha-real.md), a matching checkpoint can use `actionName: "joint_trajectory"` and
`actionArgument: "positions"`, with the appropriate state/camera mapping. For a
LeRobot driver, use `robot_trajectory` / `commands` and `jointNames` in the model's
output order, matching the robot's native action keys. Named mapping preserves
values; it does not convert radians to degrees or change gripper scales. Use the
checkpoint's matching embodiment, units and normalization. See the
[robot action contracts](../robots/README.md).

## Adapter tests

Run the adapter tests without a GPU or simulator:

```bash
node --test test/lingbot-vla.test.mjs
.venv-lingbot-client/bin/python -m unittest discover -s test -p test_lingbot_vla_client.py -v
```

Protocol tests can verify reset, image/state serialization, returned action
mapping and System0 execution in the async runtime without loading a model. They
do not demonstrate GPU inference quality, RoboTwin success rate or physical robot
operation. A live
check needs the matching checkpoint, inference host and RPent simulator above.
