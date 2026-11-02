# Robot adapters: ALOHA, LeRobot, OpenRAL and RPent

Robots share the framework's observation/action/episode interface. A reusable
`createRobotEnvironment` wraps sensing, dynamic action schemas, execution and an
optional task evaluator. ALOHA and LeRobot drivers run inside persistent Python
workers; their SDKs are optional dependencies installed in separate environments.
The existing G1 adapter continues to use its own executor.

[the OpenRAL and RPent guide](openral-rpent.md) covers OpenRAL's manifest-driven HALs
(Franka, UR, OpenArm, ALOHA, G1/H1 and other declared platforms), plus RPent's
LIBERO, RoboCasa and RoboTwin services, with a complete coverage table and
separate SDK, simulator and hardware validation scope.

## Scope

| Interface | Robot/configuration | Current integration |
|---|---|---|
| ALOHA simulation | Official ACT transfer-cube and insertion tasks | Bilateral joint positions, grippers, named cameras and action chunks |
| Classic ALOHA | ROS1, two Interbotix ViperX 300 arms | Classic `make_real_env` API; observation-only reset and position commands |
| LeRobot SO followers | SO-100 / SO-101 | Native named actions and camera features from the Robot API |
| LeRobot BiSO | Two SO follower arms | Native left/right action names through the same driver |
| LeRobot Koch | Koch follower | Same driver; separate robot configuration and units |
| Custom robot | JS driver or Python Robot-compatible factory | Robot-defined actions, units, cameras, completion and evaluation |

Hardware rows describe implemented interfaces, not new physical-robot validation
or learned-policy performance. Classic ALOHA's ROS1 API differs from the current
[Interbotix ROS2 ALOHA API](https://github.com/Interbotix/aloha/blob/2.0/aloha/real_env.py).
ALOHA 2, Mobile ALOHA and arbitrary mobile bases are not automatically covered by
the classic driver. The LeRobot bridge targets the stable **0.6.1 Robot API**.

## Dependency-free interface demo

```bash
npm run demo:robots
```

This drives synthetic single-arm and dual-arm fixtures using named joints and
two-command chunks. It needs only Node.js 22.19+. Output distinguishes policy
decisions from underlying control steps; task success remains unknown. This is an
interface demonstration, not a robot task benchmark.

## ALOHA simulation

Use Python **3.10 or 3.11** for the ACT simulation stack below, in a separate
environment from ARC or LeRobot:

```bash
git clone --depth 1 https://github.com/tonyzhaozh/act.git runtime/act
python3.11 -m venv runtime/aloha-sim-venv
runtime/aloha-sim-venv/bin/python -m pip install -r requirements-aloha-sim.txt
```

Run the simulator integration check without a model:

```bash
ALOHA_PYTHON="$PWD/runtime/aloha-sim-venv/bin/python" \
ALOHA_TEST_ACT_ROOT="$PWD/runtime/act" MPLBACKEND=Agg \
node --test test/robot-environments.test.mjs
```

This check runs both task families with a short hold-position trajectory and
verifies rendered frames and control-step accounting.

The adapter sets the initial object pose from the seed using the official task's
sampling ranges. Object ground-truth state is not placed in the policy observation.
Only robot `qpos`/`qvel` and selected camera images are exposed. The two task families
use the official ACT reward/success criterion and simulation horizon.

Run a model-backed development experiment:

```bash
ALOHA_PYTHON="$PWD/runtime/aloha-sim-venv/bin/python" \
ACT_ROOT="$PWD/runtime/act" ALOHA_MODE=sim \
ALOHA_TASKS=sim_transfer_cube_scripted,sim_insertion_scripted \
REA_MODEL=your-model REA_BASE_URL=http://localhost:8000/v1 \
REA_MAX_STEPS=20 \
npm run rea -- run examples/aloha.mjs
```

The simulation requires a working MuJoCo rendering backend on the host. The model
endpoint can be local or remote, configured as in [the framework reference](../framework.md).
For a state-only controller, construct the adapter with `cameraNames: []` to omit
rendering. Defaults are `mode: "sim"` and `execute: true`; setting `execute: false`
previews a target without advancing physics.

### ALOHA action contract

`joint_position` takes one `positions` vector of length 14. `joint_trajectory`
takes `positions` as a list of 1–100 such vectors, executed in order at the native
control interval. This is the classic [ACT action convention](https://github.com/tonyzhaozh/act/blob/main/sim_env.py):

```text
left arm:  waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
left gripper
right arm: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
right gripper
```

Arm positions are radians and each gripper is normalized to `[0, 1]`. Joint bounds
come from the installed simulator or robot SDK. The observation's `state.robot`
describes joint order, camera names and control frequency. Images retain camera
names; the chat policy labels them beside the corresponding images.

```js
import { createAlohaEnvironment } from "rea-agent/environments/aloha";

const environment = createAlohaEnvironment({
  mode: "sim",
  python: "/path/to/aloha-sim-venv/bin/python",
  actRoot: "/path/to/act",
  cameraNames: ["top", "angle"]
});
const initial = await environment.reset({ task: "sim_transfer_cube_scripted", seed: 0 });
const positions = initial.observation.state.qpos;
const next = await environment.step({
  name: "joint_trajectory", arguments: { positions: [positions, positions] }
});
await environment.close();
```

A learned ACT/VLA policy can return a trajectory action from `decide`; the Python
driver handles the control loop between model calls. This repository does not ship a
trained ALOHA policy checkpoint. One trajectory counts as one policy step, while
its actual control steps and native simulation limits are tracked separately.
`info.controlSteps` is the number executed in this call and `totalControlSteps`
is the episode total. A chunk's reward is the sum of its executed native step
rewards; `lastReward` and `maxObservedReward` retain the task's progress scale.

## Classic ALOHA hardware

Use the dedicated [real-robot guide](aloha-real.md) and
[configuration](../../examples/aloha-real.config.mjs) for a chat policy, a matching VLA
policy, or application-specific REA routing on Classic ROS1 ALOHA.

Use the robot's existing ROS1/Interbotix/ALOHA Python environment and source
checkout. `ALOHA_PYTHON` selects that interpreter and `alohaRoot`/`ALOHA_ROOT` selects
the classic [ALOHA source](https://github.com/tonyzhaozh/aloha).

```js
const environment = createAlohaEnvironment({
  mode: "real",
  python: "/path/to/robot/python",
  alohaRoot: "/path/to/aloha",
  execute: false
});
```

The worker calls `make_real_env(..., setup_robots=False)` and `reset(fake=True)`.
Reset reads the robot; it does not home the arms. Preview does not call `env.step`.
Execution is selected with `execute: true` or `ROBOT_EXECUTE=true` in the provided
example. The existing ROS system, calibrated arms and camera topics must already
be configured on the robot host. Connecting an SDK still initializes its clients.

Classic ALOHA's `step` submits joint targets and waits one control period. It does
not confirm that the arms have reached the targets. Its API has no general physical
stop primitive: this adapter can cancel pending targets, but the last position
target remains active. Completion and task success need robot/task-specific
feedback; `evaluate` can supply the latter. Camera transport color conventions can
be selected with `imageEncoding: "rgb"` or `"bgr"`.

## LeRobot: SO-100/SO-101, BiSO and Koch

Use Python **3.12+** with LeRobot 0.6.1 installed, normally in the robot's existing
environment. [requirements-lerobot.txt](../../requirements-lerobot.txt) declares the
version and hardware extras. It is independent of the core and ALOHA simulation
requirements. Setup/calibration follows the selected robot's upstream guide.

Starting configurations:

- [SO-101](../../examples/robots/so101.json): a single follower; change the type to
  `so100_follower` for SO-100.
- [Bimanual SO-101](../../examples/robots/bimanual-so101.json): separate left/right ports.
- [Koch](../../examples/robots/koch.json): Koch follower configuration.

Edit the ports, robot IDs, installed calibration and optional camera configuration
for your hardware. The example defaults to preview:

```bash
LEROBOT_PYTHON=/path/to/lerobot/bin/python \
LEROBOT_CONFIG=examples/robots/so101.json \
REA_MODEL=your-model REA_BASE_URL=http://localhost:8000/v1 \
REA_INSTRUCTION='Observe the work area and choose a next joint target.' \
npm run rea -- run examples/lerobot.mjs
```

The driver reads `action_features` and `observation_features` from the actual robot
object. Camera arrays become named PNG images; scalar proprioception remains in
state. `robot_command` accepts `{values: {"joint_name.pos": value, ...}}`;
`robot_trajectory` accepts `{commands: [values, ...]}`. Required joint names come
from the selected robot, including left/right prefixes for dual arms. This follows
the [LeRobot Robot contract](https://huggingface.co/docs/lerobot/main/api/robots).

`controlHz` controls command spacing, default 30 Hz. `units` documents the selected
hardware's native units; the robot SDK applies its own command limits. LeRobot's
feature types alone do not identify units: SO joints may be degrees or normalized
values depending on configuration, and its gripper scale differs from ALOHA.
The supplied configurations explicitly select degrees and declare gripper `[0,100]`.

Connection uses `calibrate=False`, so the runner does not launch interactive
calibration. The native connection routine can still configure motors. Reset only
observes. `send_action` may clip targets; `info.actualActions` stores the returned
commands, separately from `requestedActions`. A submitted command does not establish
arrival or task success. The base Robot API has no universal `stop`; an optional
robot `stop()` hook is used when present, otherwise support is reported as false.
Close uses the robot's own `disconnect()` behavior, including its torque policy.

### Bring another Python robot

A Robot-compatible factory avoids depending on a particular registry version:

```python
# my_robot.py, importable on PYTHONPATH
def make_robot(options):
    return MyRobot(options)  # supplies Robot-compatible features and I/O methods
```

```js
const environment = createLeRobotEnvironment({
  robotFactory: "my_robot:make_robot",
  factoryOptions: { device: "/dev/your-device" },
  units: { "joint_1.pos": "radians" },
  execute: false
});
```

The factory path does not import LeRobot itself. The returned object supplies
`action_features`, `observation_features`, `is_connected`, `connect(calibrate=False)`,
`get_observation()`, `send_action(dict)` and `disconnect()`. This lets a custom
robot use the bridge without adding a hard-coded robot model to the framework.

For a different SDK shape, implement a JS driver with `describe`, `reset`,
`observe`, `step`, `stop` and `close`, then pass it to
`createRobotEnvironment({driver, execute, evaluate})`. Driver snapshots contain
`state`, named `images`, timestamp, native outcome fields and `info`; the wrapper
adds observation identity and integrates with the existing episode/evaluation
runner. See [the offline driver example](../../examples/robots-offline.mjs).
