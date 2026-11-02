# OpenRAL and RPent robot integration

Two optional adapters connect the framework's existing policies and evaluation
runner to [OpenRAL](https://github.com/OpenRAL/openral) HALs and
[RPent](https://github.com/RLinf/RPent) environment services. A robot keeps its
native joint order, control modes, camera names and outcome semantics. Changing
the robot does not require changes to `runEpisode` or a separate agent loop.

The [machine-readable inventory](../../examples/robots/upstream-support.json) records
manifest and entrypoint mappings for the supported source versions, including
declarations that require an external implementation. Robot variants and simulated
embodiments are represented separately.

## OpenRAL coverage

Use a [robot configuration](../../examples/robots/openral) with `robotId`, or supply a
custom `robotYaml`. The driver calls the official resolver and HAL protocol.
New upstream manifests can use the same path without a framework robot registry edit.

| Manifest ID | Available integration | Example or requirement |
|---|---|---|
| `aloha_bimanual` | ALOHA dual-arm MuJoCo; real callback HAL | [configuration](../../examples/robots/openral/aloha_bimanual.json) |
| `aloha_agilex` | RoboTwin scene through `SimAttachedHAL` | [configuration](../../examples/robots/openral/aloha_agilex.json); RoboTwin runtime/assets |
| `anvil_openarm_v2` | Anvil OpenArm MuJoCo | [configuration](../../examples/robots/openral/anvil_openarm_v2.json) |
| `franka_panda` | Panda MuJoCo; real callback HAL | [configuration](../../examples/robots/openral/franka_panda.json) |
| `g1` | G1 MuJoCo | [configuration](../../examples/robots/openral/g1.json); the existing G1 adapter covers its separate sport-service path |
| `galaxea_a1` | Native real HAL through the vendor ROS1 sidecar | [configuration](../../examples/robots/openral/galaxea_a1.json); running A1 sidecar required |
| `google_robot` | Manifest only in the inspected release | Upstream MS3 Google tasks are not registered; no runnable example |
| `gr1` | GR1 RoboCasa community-fork scene | [configuration](../../examples/robots/openral/gr1.json); GR1 fork/assets |
| `h1` | H1 MuJoCo | [configuration](../../examples/robots/openral/h1.json) |
| `openarm` | OpenArm v2 MuJoCo; real callback HAL | [configuration](../../examples/robots/openral/openarm.json); manifest name is `openarm_v2` |
| `panda_mobile` | Native Python kinematic HAL; optional RoboCasa scene attachment | [configuration](../../examples/robots/openral/panda_mobile.json) |
| `panda_mobile_vslam` | Variant of the same kinematic HAL | [configuration](../../examples/robots/openral/panda_mobile_vslam.json) |
| `pusht_2d` | PushT scene, a 2D pusher rather than an arm | [configuration](../../examples/robots/openral/pusht_2d.json); gym-pusht |
| `r1pro` | BEHAVIOR / OmniGibson scene | [configuration](../../examples/robots/openral/r1pro.json); Isaac Sim runtime/assets |
| `rizon4` | Flexiv Rizon4 MuJoCo | [configuration](../../examples/robots/openral/rizon4.json) |
| `sawyer` | MetaWorld scene; real callback HAL | [configuration](../../examples/robots/openral/sawyer.json); MetaWorld |
| `so100_follower` | Generic MuJoCo arm from manifest; native LeRobot HAL | [configuration](../../examples/robots/openral/so100_follower.json) |
| `so101_follower` | Generic MuJoCo arm from manifest; native LeRobot HAL | [configuration](../../examples/robots/openral/so101_follower.json) |
| `ur5e` | UR5e MuJoCo; real callback HAL | [configuration](../../examples/robots/openral/ur5e.json) |
| `ur10e` | UR10e MuJoCo; real callback HAL | [configuration](../../examples/robots/openral/ur10e.json) |
| `widowx` | SimplerEnv / ManiSkill3 scene | [configuration](../../examples/robots/openral/widowx.json); SimplerEnv runtime/assets |

Google Robot's limitation is documented in the
[upstream SimplerEnv adapter](https://github.com/OpenRAL/openral/blob/master/python/sim/src/openral_sim/backends/simpler_env.py).
PandaMobile's default HAL is a kinematic state model, so a passing step does not
demonstrate mobile-manipulation physics or VSLAM performance. Attach an actual
RoboCasa scene when that is the intended experiment.

### Setup and use

OpenRAL 0.3.1 requires **Python 3.12**. Keep it in a separate environment from the
classic ACT ALOHA and RPent simulator environments. The source checkout supplies
robot manifests, scene files and source-relative HAL resources.

```bash
git clone https://github.com/OpenRAL/openral runtime/openral
python3.12 -m venv runtime/openral-venv
runtime/openral-venv/bin/python -m pip install -r requirements-openral.txt
```

Scene-based configurations additionally need their upstream simulator, assets and
controller dependencies; installing the generic HAL requirements does not install
every benchmark. Follow the relevant [OpenRAL scene setup](https://github.com/OpenRAL/openral/tree/master/scenes).
For the lightweight PushT and MetaWorld examples, install
`requirements-openral-scenes.txt`; it includes the Pymunk version needed by the
tested gym-pusht API.

```bash
OPENRAL_ROOT="$PWD/runtime/openral" \
OPENRAL_PYTHON="$PWD/runtime/openral-venv/bin/python" \
OPENRAL_CONFIG=examples/robots/openral/aloha_bimanual.json \
REA_MODEL=your-model REA_BASE_URL=http://localhost:8000/v1 \
REA_INSTRUCTION='Observe the robot and choose a joint target.' \
npm run rea -- run examples/openral.mjs
```

Simulation executes by default; real configurations default to preview. Set
`ROBOT_EXECUTE=true` to execute the selected real configuration. `cameraNames: []`
omits camera images; for bare HALs it also skips `read_images`. Native scenes may
still render internally during reset/step. Omit the option to include native HAL
cameras. `transport` passes constructor options such as an explicit `mjcf_path`
or a hardware port to the upstream resolver.

```js
import { createOpenRALEnvironment } from "rea-agent/environments/openral";

const env = createOpenRALEnvironment({
  openralRoot: "/path/to/openral",
  python: "/path/to/openral-venv/bin/python",
  robotId: "aloha_bimanual",
  cameraNames: [],
});
try {
  const initial = await env.reset();
  const hold = { controlMode: "joint_position", values: initial.observation.state.position };
  const next = await env.step({ name: "hal_sequence", arguments: { commands: [hold, hold] } });
  console.log(next.info.controlSteps); // 2 native submissions
} finally {
  await env.close();
}
```

`hal_action` takes `{controlMode, values, eeName?, frameId?, jointNames?}`.
`hal_sequence` takes `{commands: [...]}` and submits every command as an SDK
`Action(horizon=1)`, preserving waypoint order across HALs that interpret chunks
differently. `hal_group` combines the control surfaces for one atomic OpenArm tick
or a factory-supplied typed scene HAL. Available modes come from the instantiated HAL implementation;
`state.robot` describes the joint order, limits, units and camera names. Do not
reuse a controller vector across embodiments without translating those fields.
One sequence counts as one policy decision, and `info.controlSteps` records the
native steps actually submitted. Real sequences are paced by `controlHz`.

Configurations using `simEnvYaml` instead expose `scene_action` with
`{values: [...]}` and `scene_trajectory` with `{commands: [[...], ...]}`. They use
the native scene's action contract. PushT uses two pixel coordinates; MetaWorld
uses four controller values, and ALOHA scenes use their own 14D action contract.
No conversion from these vectors to robot joint angles is implied. Actions go
through the official `SimRollout.step`, preserving its observations, native reward
and termination flags. A sequence accumulates rewards and stops on a reported
terminal or truncated transition.
GR1 uses the upstream 29D order: waist (3), right arm (7), left arm (7), right hand
(6), left hand (6), including bounds in that order. The ALOHA scene adapter accepts
absolute arm radians and normalized grippers; it follows OpenRAL in ignoring
gym-aloha's nominal `[-1,1]` Box, which is not the actual joint command range.

`robotYaml` and `simEnvYaml` can be absolute or relative to `OPENRAL_ROOT`.
In `examples/openral.mjs`, a `simEnvYaml` starting with `.` is relative to the
configuration file. The supplied RoboTwin and Sawyer scene files adapt upstream
benchmark definitions to the HAL's `SimScene` format. Benchmark provenance and
multi-episode evaluation settings belong in the evaluation configuration.

For bare and real HALs, `reset` observes and records `seedApplied: false`.
`task_success()` is forwarded when present, while reward remains unknown and done
flags remain false. Use a task `evaluate` callback and a bounded episode for that
path. A successful command submission does not establish task completion.
`stop` invokes native `estop`; its expected `ROSEStopRequested` result is reported
explicitly. Bare and real HALs require a fresh driver after a stop.

Native scene configurations support `reset(seed)`: it starts a new scene episode,
forwards the seed, clears control-step accounting and uses the official
`reset_estop` lifecycle when recovering from a runner stop. `observe` does not
reset or advance physics. Task selection remains in `simEnvYaml`; task language
is included in the observation. Scene episode behavior follows the underlying
SDK; for example, the inspected LeRobot MetaWorld wrapper resets internally on a
terminal step. This path should not be treated as an official multi-episode
benchmark protocol merely because native outcomes are available.

### Real HAL transports

SO-100/SO-101 connect through the native LeRobot HAL and Galaxea A1 through its
native ROS1 sidecar. Franka, UR, Sawyer, ALOHA and OpenArm's real HAL constructors
need site-specific publish/state callbacks. Their default empty callbacks do not
control a robot, so the framework requires `halFactory: "my_robot:make_hal"` for these paths.
The factory receives `factoryOptions` and returns an unconnected official HAL
with those callbacks wired to the robot's working transport. It must expose
`description`, `connect`, `read_state`, `send_action`, `estop` and `disconnect`.

Preview skips action submission but still connects and reads the configured SDK.
The upstream HAL's connection, motor and disconnect semantics apply. None of the
software tests below constitutes a physical hardware validation.

## RPent coverage and setup

RPent currently publishes these `RobotSpec` packages:

| Backend | Embodiment | Native actions | Configuration |
|---|---|---|---|
| `libero` | Server-selected LIBERO robot, Panda by default; standard/pro/plus tasks | `rpent_action` / `rpent_trajectory`, 7 controller values | [LIBERO](../../examples/robots/rpent/libero.json) |
| `robocasa` | PandaOmron mobile manipulator | `rpent_action` / `rpent_trajectory`, 12 composite-controller values | [RoboCasa](../../examples/robots/rpent/robocasa.json) |
| `robotwin` | ALOHA-AgileX dual arms | `rpent_qpos` (14), `rpent_ee` (16), and corresponding `_trajectory` actions | [RoboTwin](../../examples/robots/rpent/robotwin.json) |

RPent's README roadmap also mentions Franka and SO-101, but their standalone RPent
implementations are not published in the inspected source. Use OpenRAL/LeRobot
for those robots; listing a roadmap row is not a working integration.

The framework worker only requires the official RPC client and NumPy:

```bash
git clone https://github.com/RLinf/RPent runtime/rpent
python3.11 -m venv runtime/rpent-client-venv
runtime/rpent-client-venv/bin/python -m pip install -r requirements-rpent.txt
```

Run the upstream environment server in its own fully provisioned simulator
environment, which may be on a different machine. For example, from the RPent
checkout after installing its LIBERO dependencies and assets:

```bash
python robots/libero/env_server.py --transport http --host 127.0.0.1 \
  --port 8801 --suite libero_spatial --task 0 --seed 0
```

RoboCasa uses `robots/robocasa/env_server.py --task-name ...`; RoboTwin uses
`robots/robotwin/env_server.py --task-name ... --task-config ... --assets-path ...`.
Follow each backend's [upstream usage guide](https://github.com/RLinf/RPent/tree/main/docs/source-en/rst_source/usage)
for installation and supported tasks. `http://host:port` and `socket://host:port`
endpoints both use RPent's official client and wire format.

```bash
RPENT_ROOT="$PWD/runtime/rpent" \
RPENT_PYTHON="$PWD/runtime/rpent-client-venv/bin/python" \
RPENT_CONFIG=examples/robots/rpent/libero.json \
REA_MODEL=your-model REA_BASE_URL=http://localhost:8000/v1 \
REA_SEED=0 REA_MAX_STEPS=20 \
npm run rea -- run examples/rpent.mjs
```

The server owns task and seed selection. `expectedMeta` can assert the expected
suite/task/seed from `get_env_meta`; `reset` rejects a different requested identity
instead of silently pretending to switch benchmarks. Configure a separate server
for another task. Repeated RoboCasa resets can advance the server's RNG unless
the upstream reset-seed mechanism is configured. `shutdownOnClose` defaults to
false, preserving an independently managed environment server.

Actions use `{values: [...]}` or `{commands: [[...], ...]}`. LIBERO and RoboCasa
vectors are native controller inputs, not absolute joint positions. RoboTwin
qpos contains left six joints plus gripper, then right six joints plus gripper;
EE actions contain left world xyz, quaternion **wxyz**, gripper, then the right
eight values. Joints are radians, positions metres and grippers normalized
closed `0` to open `1`. Named camera images and task language reach the policy.

The adapter preserves native reward/termination and reports actual control steps.
RoboTwin chunks can stop early and return `executed_actions`; RoboCasa sequences
loop over `step` and `check_success`. RPent's current LIBERO native chunk executes
the requested chunk and returns per-step signals, which the framework combines using any
termination/truncation flag. A chunk's reported reward sums its executed steps.
Observation state comes from the latest transition; RPC cameras are read
on observation. RPent has no common stop primitive; closing the client does not
claim to stop a physical robot.

## Verification

The dependency-free unit suite covers command order, preview, native outcomes,
camera encoding and lifecycle. Optional tests exercise the installed official
SDKs and services. The tests distinguish deterministic protocol fixtures from
physics; neither measures a trained policy's benchmark success rate.

```bash
python3 -m unittest discover -s test

OPENRAL_TEST_ROOT=/path/to/openral \
  /path/to/openral-venv/bin/python -m unittest discover -s test -p test_openral_driver.py

OPENRAL_SCENE_TEST_ROOT=/path/to/openral \
  /path/to/openral-venv/bin/python -m unittest discover -s test -p test_openral_scenes.py

RPENT_ROOT=/path/to/RPent RPENT_NODE=/path/to/node \
  /path/to/rpent-client-venv/bin/python test/test_rpent_driver.py

# Requires the full RPent LIBERO simulator runtime and assets:
RPENT_ROOT=/path/to/RPent RPENT_NODE=/path/to/node REA_RPENT_PHYSICS=libero \
  /path/to/rpent-simulator-venv/bin/python test/test_rpent_driver.py

# With both simulator stacks and the selected RoboCasa scene assets:
RPENT_ROOT=/path/to/RPent RPENT_NODE=/path/to/node \
REA_RPENT_PHYSICS=libero,robocasa REA_RPENT_ROBOCASA_TASK=NavigateKitchen \
  /path/to/rpent-simulator-venv/bin/python test/test_rpent_driver.py

OPENRAL_PYTHON=/path/to/openral-venv/bin/python \
OPENRAL_TEST_ROOT=/path/to/openral OPENRAL_TEST_PHYSICS=1 \
  node --test test/robot-environments.test.mjs

/path/to/openral-venv/bin/python scripts/inspect_robot_support.py \
  --openral-root /path/to/openral --rpent-root /path/to/RPent
```

The RPent opt-in test starts the official HTTP and socket servers for all three
backends, using deterministic inner simulator fixtures. It runs both the Python
driver and the complete Node → Python → official RPC chain, including camera
pixels, chunk truncation and executed-action accounting. OpenRAL opt-in tests use
all 21 actual manifest schemas and native `SimAttachedHAL`; transport tests use
native real HAL classes with simulated callbacks, digital twins or sidecars.

Install the SDK and simulator versions required by your selected backend. Fixture
tests exercise interface behavior; live operation requires the corresponding
robot or simulator runtime.
