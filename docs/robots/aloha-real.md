# REA on Classic ALOHA hardware

REA connects the same policy and routing interfaces to Classic ROS1 ALOHA through
the existing `tonyzhaozh/aloha` Python API. The adapter reads the two follower
arms and named cameras, and submits absolute bilateral joint targets. This guide
covers the two Interbotix ViperX 300 follower arms. ALOHA 2/ROS2, Mobile ALOHA and
AgileX/Piper hardware need their corresponding drivers; a shared ALOHA name does
not make those APIs interchangeable.

The framework entry points are:

- [Real-robot configuration](../../examples/aloha-real.config.mjs)
- [pi0.5 + Code configuration](../../examples/aloha-pi05-code.config.mjs)
- [pi0.5 + LingBot-VLA + Code configuration](../../examples/aloha-pi05-lingbot-code.config.mjs)
- [JavaScript environment](../../src/environments/aloha.mjs)
- [Python hardware driver](../../scripts/robot_drivers/aloha.py)
- [General robot interface](README.md)

## Robot host setup

Use the robot's installed ROS1, Interbotix and Classic ALOHA environment, with its
camera topics and arm configuration already running. Source that environment
before launching REA. `ALOHA_PYTHON` must select the interpreter that can import
`rospy`, NumPy and the robot SDK. `ALOHA_ROOT`
points to the Classic ALOHA checkout containing `aloha_scripts/real_env.py` and
`aloha_scripts/constants.py`.

From the REA repository:

```bash
npm install
export ALOHA_PYTHON=/path/to/robot-environment/bin/python
export ALOHA_ROOT=/path/to/aloha
export REA_MODEL=your-model
export REA_BASE_URL=https://your-model-endpoint/v1
# Set REA_API_KEY in the local environment when the endpoint requires it.
export REA_INSTRUCTION='Place the object in the tray.'

npm run rea -- run examples/aloha-real.config.mjs --events
```

The example defaults to preview: it acquires live observations and requests model
actions, but does not submit those actions to the arms. Set `ROBOT_EXECUTE=true`
for position-command execution with the same entry point.

Connecting calls `make_real_env(init_node=True, setup_robots=False)`; resetting
calls `reset(fake=True)`. Reset therefore reads the current robot instead of
homing it. Hardware setup and homing remain part of the robot's own workflow.

## Observation and action contract

Observations contain `state.qpos`, `state.qvel`, `state.effort`, camera images and
`state.robot` metadata describing units, joint order and the native control
interval. The default cameras are `cam_high`, `cam_low`, `cam_left_wrist` and
`cam_right_wrist`. Set `ALOHA_CAMERAS` to a comma-separated list for the cameras
present on the robot. `ALOHA_IMAGE_ENCODING=bgr` converts BGR source frames to RGB.

Both action types use absolute targets in this order:

```text
[left arm joints × 6, left gripper, right arm joints × 6, right gripper]
```

Arm joints are in radians; grippers use the SDK's normalized `[0, 1]` range.
`joint_position` accepts one 14-value `positions` vector. `joint_trajectory`
accepts 1–100 vectors and submits one per native control interval. Joint limits
come from the connected SDK. The driver reports submitted control steps; an SDK
submission does not establish that a target has been reached.

`REA_MAX_STEPS` bounds policy decisions, while `ALOHA_MAX_CONTROL_STEPS` bounds
submitted native targets. The example defaults to 20 decisions and 1,000 native
targets. A trajectory is one policy decision containing several native targets.

Task success is unknown unless the application supplies a task evaluator through
`createAlohaEnvironment({ evaluate })`. The adapter's `stop()` cancels remaining
targets in its queue. Classic ALOHA supplies no physical stop command through this
API, and the last submitted position target remains active; use the robot's
physical stop controls when a physical stop is needed.

## Route pi0.5 + Code

Use [aloha-pi05-code.config.mjs](../../examples/aloha-pi05-code.config.mjs) for a
candidate pool containing exactly `pi05` and `code`. Start the pi0.5 server with
a checkpoint and normalization statistics matching **Classic ALOHA**. This version
does not construct or connect a LingBot controller and requires no `LINGBOT_VLA_*`
settings or LingBot server.

```bash
# Source the robot's ROS1/ALOHA environment first.
export ALOHA_PYTHON=/path/to/robot-environment/bin/python
export ALOHA_ROOT=/path/to/aloha
export PI05_URL=ws://pi05-host:8000
export PI05_PYTHON=/path/to/pi05-client/bin/python
export REA_CODE_PYTHON=/path/to/code-environment/bin/python
export REA_CODE_MODEL=your-code-model
export REA_ROUTER_MODEL=your-routing-model
export REA_BASE_URL=https://your-model-endpoint/v1
# Set REA_API_KEY locally if required.
export REA_INSTRUCTION='Place the object in the tray.'

npm run rea -- run examples/aloha-pi05-code.config.mjs --events
```

Install [requirements-pi05.txt](../../requirements-pi05.txt) in `PI05_PYTHON` and
NumPy in `REA_CODE_PYTHON`. The pi0.5 client uses the shared WebSocket transport
dependencies; installing them does not start a LingBot service.

## Route pi0.5 + LingBot-VLA + Code

Use [aloha-pi05-lingbot-code.config.mjs](../../examples/aloha-pi05-lingbot-code.config.mjs)
for exactly `pi05`, `lingbot-vla` and `code`. Start both VLA servers with checkpoints,
robot configurations and normalization statistics matching **Classic ALOHA**:

```bash
# Source the robot's ROS1/ALOHA environment first.
export ALOHA_PYTHON=/path/to/robot-environment/bin/python
export ALOHA_ROOT=/path/to/aloha
export PI05_URL=ws://pi05-host:8000
export PI05_PYTHON=/path/to/pi05-client/bin/python
export LINGBOT_VLA_URL=ws://lingbot-host:8006
export LINGBOT_VLA_PYTHON=/path/to/lingbot-client/bin/python
export LINGBOT_VLA_ROBOT=your-aloha-robot-config
export REA_CODE_PYTHON=/path/to/code-environment/bin/python
export REA_CODE_MODEL=your-code-model
export REA_ROUTER_MODEL=your-routing-model
export REA_BASE_URL=https://your-model-endpoint/v1
# Set REA_API_KEY locally if required.
export REA_INSTRUCTION='Place the object in the tray.'

npm run rea -- run examples/aloha-pi05-lingbot-code.config.mjs --events
```

This version adds the LingBot endpoint, client interpreter and robot configuration.
Install [requirements-lingbot-vla.txt](../../requirements-lingbot-vla.txt) in
`LINGBOT_VLA_PYTHON`, in addition to the pi0.5 and Code dependencies above.

Both dedicated entry points select their candidate pools without an `ALOHA_POLICY`
setting. `REA_ROUTER_MODEL` and `REA_CODE_MODEL` each default to `REA_MODEL`, so
one shared model name can replace the two explicit names. The model services can
run on separate GPU hosts; ROS and the hardware driver stay on the robot host.
The client environments do not need model weights or ROS packages. Both versions
keep the preview default; `ROBOT_EXECUTE=true` enables command submission.

## Routing and controller contracts

| Candidate ID | Inference input | Native ALOHA output |
| --- | --- | --- |
| `pi05` | OpenPI `state`, three named `images`, `prompt` | `actions` mapped to `joint_trajectory.positions` |
| `lingbot-vla` | `observation.state`, three `observation.images.*` fields, `task` | `action` mapped to `joint_trajectory.positions` |
| `code` | Images, measured state, task and execution feedback | Generated Python computes `joint_position` or `joint_trajectory` |

Both entry points default to **asynchronous routing with System0 priority**.
System0 first generates and dispatches an action; only then can the visual selector
start a background System2 request. The selector reads camera observations,
measured joints and execution history. While it is thinking, the current controller
continues producing and executing action chunks; the first controller is `pi05`.
A completed selection never interrupts a chunk already in progress. System1 keeps
the current controller for at least four completed action blocks before accepting
a different candidate, giving System0 time to make progress.

Only the selected controller performs action inference, using the fresh measured
observation from the last completed chunk. Its inference and the resulting robot
action remain sequential. This overlaps **routing with controller inference and
motion**; it does not precompute the next joint trajectory during motion. The
robot executes one action block at a time, including when switching to Code.

```bash
# Optional: wait for the visual selector before each action for a serial comparison.
ALOHA_ROUTING_MODE=sync npm run rea -- run examples/aloha-pi05-code.config.mjs --events
# The same option works with aloha-pi05-lingbot-code.config.mjs.
```

In async mode, `ALOHA_ROUTE_EVERY_STEPS` defaults to `4`: System2 can start after
dispatching action blocks 1, 5, 9, and so on. One selection request runs at a time.
If it is still busy, that slot is skipped; completion does not immediately launch
another request. The next eligible dispatch supplies its current observation.
This keeps background work out of the System0 startup path and reduces its frequency.

`ALOHA_MIN_CONTROLLER_STEPS` defaults to `4` in async mode. It counts completed
action blocks, not routing calls or individual points within a VLA trajectory.
An early switch recommendation is deferred until that minimum is met and is
recorded as `route.deferredCandidateId`; newer advice can replace it in the meantime.
The selector is instructed to retain a controller making progress and to justify
switches with observed failure, lack of progress, or a change of task phase.
The sync comparison defaults both settings to `1` and waits for selection before
each action. Explicit environment settings override either mode's defaults.

`ALOHA_ROUTE_MAX_AGE_STEPS` defaults to `12`; older recommendations are discarded.
Without a usable new selection, including after a routing timeout, the runtime
continues the current controller. Routing uses `REA_DECISION_TIMEOUT_MS` as its
request timeout, separately from the selected controller's decision budget.
This priority controls client scheduling and switching; GPU scheduling on a shared
inference server remains the server's responsibility.

Events include `thinking.started/completed/applied`, the selected candidate and
reason. `asyncMetrics.overlapMs` measures overlap between background routing and
robot action intervals, and `routedWhileThinking` counts boundaries processed
while a routing request was pending. These are available through `--events` and
the existing [session API](../async-systems.md#start-subscribe-cancel).

`PI05_CHUNK_STEPS` and `LINGBOT_VLA_CHUNK_STEPS` default to 8. Set `PI05_DESCRIPTION` and
`LINGBOT_VLA_DESCRIPTION` to describe the task coverage of the deployed
checkpoints so the router can distinguish their capabilities.

Both routing versions use a vision-language controller selector. They do not load the
RoboTwin success router or assume that its candidate scores transfer to ALOHA.
No trained ALOHA model weights are included. Each configured VLA server must return
absolute joint14 targets with the joint/gripper convention above. The pi0.5
client uses the [OpenPI ALOHA interface](https://github.com/Physical-Intelligence/openpi/tree/main/examples/aloha_real);
checkpoint-specific resizing and normalization belong to the model server.

The [Code policy](../../src/policies/aloha-code.mjs) supplies Python with `np`,
`observation`, `instruction`, `joint_position(positions)` and
`joint_trajectory(positions)`. The program emits exactly one action, which returns
to the same ALOHA environment for validation and execution. It runs in the configured local
Python worker. Controller programs calculate targets through these helpers,
leaving ROS execution to the environment. This keeps both VLA and Code on the same hardware command path.

For single-controller operation, use the general
[aloha-real.config.mjs](../../examples/aloha-real.config.mjs) entry point with
`ALOHA_POLICY=pi05`, `lingbot-vla`, or `code`. `ALOHA_POLICY=chat` remains its
default, and `ALOHA_POLICY=hybrid` selects the three-controller pool.

The shared chat policy used by the chat, Code and router paths serializes supplied
transition history without a fixed character limit. `historyLimit` selects the
recent step window, and historical images are excluded. Provider context limits apply.

## Implementation paths

- [pi0.5 + Code configuration](../../examples/aloha-pi05-code.config.mjs)
- [pi0.5 + LingBot-VLA + Code configuration](../../examples/aloha-pi05-lingbot-code.config.mjs)
- [General real-robot configuration](../../examples/aloha-real.config.mjs)
- [Observation-based router](../../src/policies/aloha-router.mjs)
- [pi0.5 policy](../../src/policies/pi05.mjs) and [OpenPI client](../../scripts/pi05_client.py)
- [LingBot-VLA policy](../../src/policies/lingbot-vla.mjs) and [client](../../scripts/lingbot_vla_client.py)
- [Code policy](../../src/policies/aloha-code.mjs) and [Python worker](../../scripts/aloha_code_worker.py)

The `createAlohaRouterThinker` export adapts the visual selector to the existing
[System0/1/2 runtime](../async-systems.md). Custom System1 modules can replace the
fast selector and return `{ candidateId, reason }` for the same candidate pool.
The module's default `createAlohaRouter` export remains available for direct,
blocking System1 selection.

## Local interface checks

```bash
python3 -m pip install -r requirements-pi05.txt -r requirements-lingbot-vla.txt
npm run test:aloha
```

The suite exercises configuration, asynchronous two-way and three-way routing,
action-boundary switching, the synchronous comparison, Python controller execution,
VLA WebSocket protocols, and ALOHA SDK call contracts with local fixtures. These
are software integration checks; physical motion and trained-model task success
require the connected robot and matching checkpoints.
