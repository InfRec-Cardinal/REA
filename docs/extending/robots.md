# Real robots

A real robot is an environment like any other, wrapped around a **driver**. The driver talks to the robot SDK; the
wrapper (`createRobotEnvironment`) adds observation identity, action validation, preview mode and the episode lifecycle.
Every policy, router and runtime works unchanged on top.

## Included robots

| Robot | Environment type | Guide |
|---|---|---|
| ALOHA (ACT simulation, Classic ROS1 hardware) | `aloha` | [ALOHA](../robots/README.md#aloha-simulation), [ALOHA hardware](../robots/aloha-real.md) |
| LeRobot SO-100/SO-101, BiSO, Koch, any Robot-compatible factory | `lerobot` | [Robots](../robots/README.md#lerobot-so-100so-101-biso-and-koch) |
| OpenRAL manifests: Franka Panda, UR5e/UR10e, Sawyer, WidowX, Flexiv Rizon 4, SO-100/101, ALOHA, OpenArm, Galaxea A1 and R1Pro, Unitree G1/H1, Fourier GR1 and more | `openral` | [OpenRAL and RPent](../robots/openral-rpent.md) |
| RPent LIBERO, RoboCasa and RoboTwin services | `rpent` | [OpenRAL and RPent](../robots/openral-rpent.md) |
| Unitree G1 (service, safety gate, cerebellum) | `g1` | [Unitree G1](../robots/unitree-g1.md), [cerebellum](../../agent-vla-async/README.md) |
| Your Python driver class | `python-robot` | below |
| Your JavaScript driver | `robot` | below |

## Safety model

- **Preview by default.** `execute: false` validates actions and reports what would be sent; nothing moves. Python
  drivers receive `execute` on every `step` and the backend refuses execution unless it was configured with
  `executeEnabled`. Enable execution only after the robot, workspace and stop procedure are ready.
- **Stop.** `stop()` runs at the end of every episode and on cancellation; implement it as the SDK's safe halt and report
  `{supported: false}` honestly when the SDK has none.
- **Reset is physical.** `reset` on hardware should move to, or verify, a safe start pose. It is not a simulator reset;
  task success on hardware needs an external `evaluate` callback or a human-scored protocol.
- Keep velocity, workspace and force limits in the driver, where every policy's action passes through them.

## A Python driver in four steps

1. Copy [`examples/robots/template_driver.py`](../../examples/robots/template_driver.py) and call your SDK inside
   `reset`, `observe`, `step`, `stop` and `close`. Declare the actions with JSON Schema (units in the description).
2. Put its directory on `PYTHONPATH` for the robot's Python.
3. Configure the environment:

   ```json
   {
     "environment": { "type": "python-robot", "options": {
       "driver": "my_robot_driver:MyRobotDriver",
       "python": "/path/to/robot/venv/bin/python",
       "options": { "port": "/dev/ttyUSB0" },
       "execute": false
     } },
     "policy": { "type": "chat" }
   }
   ```

4. `npm run rea -- doctor my-robot.json`, then a preview run, then `execute: true`.

Driver snapshots are JSON: `state`, `images` (`[{name, mimeType, data}]`; `robot_drivers.common.image_observation`
encodes RGB arrays without extra dependencies), and optionally `text`, `terminated`, `truncated`, `success`, `reward`,
`info`. The driver object provides `id`, `timing`, `actions` and `metadata` attributes. `test/python-driver.test.mjs`
runs the template end to end.

## A JavaScript driver

Implement `describe()` → `{actions, timing, metadata}`, `reset({task, seed})`, `observe()`,
`step(action, {execute})`, `stop()` and `close()`, each returning the snapshot fields above, and pass it to
`createRobotEnvironment({driver, execute, evaluate})` or the `robot` environment type. See the offline example
[`examples/robots-offline.mjs`](../../examples/robots-offline.mjs).

## Action chunks and control rate

Chunked policies (VLAs) return several commands per decision; expose a trajectory action (`robot_trajectory` with
`commands`) and execute them at the robot's control rate inside `step`. `step` should return after the action's
meaningful control interval with a fresh observation, so routers see the effect of the chunk before the next decision.
