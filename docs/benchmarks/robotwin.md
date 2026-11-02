# RoboTwin 2.0

Three integrations reach RoboTwin 2.0 scenes:

| Integration | What it runs | Needs |
|---|---|---|
| [VLA / code runner](../../eval/robotwin/hybrid/README.md) (`eval/robotwin/hybrid/`) | VLA services and a Python code policy as System 0 candidates, chosen per episode by a trained success router (System 1) from the initial observation | RoboTwin plus a numeric-control harness exposing the interface below; VLA servers; a success-router checkpoint |
| RPent's RoboTwin service | any REA policy (chat, `lingbot-vla`, code) through the `rpent` environment; `examples/lingbot-vla.config.mjs` and `examples/lingbot-vla-async.config.mjs` run LingBot-VLA as a single policy and as an async System 0 candidate | a running RPent RoboTwin service ([OpenRAL and RPent](../robots/openral-rpent.md#rpent-coverage-and-setup)) |
| OpenRAL scene | `examples/robots/openral/scenes/robotwin_handover_block.yaml` through the `openral` environment | OpenRAL with its RoboTwin scene dependencies |

## Numeric-control harness interface

The VLA / code runner does not ship a simulator wrapper. It imports three modules from `harnessRoot`:

| Module | Functions it must provide |
|---|---|
| `robotwin_subset` | `make_subset_env(subset, instance, output_dir) -> (env, config)`: build the scene of one manifest record |
| `robotwin_control` | `robot_state(env, observation)`, `camera_calibration(observation)`, `validate_action(arguments)`, `execute(env, observation, arguments, ...) -> (execution, observation)` |
| `observation_io` | `plain(value)`: convert simulator values (arrays, tensors) into JSON |

`eval/robotwin/hybrid/environment.py` is the only file that calls them; read it for the exact arguments. Success is the
simulator's own success check, reported by the harness after execution.

## Tests

```bash
node --test test/robotwin-*.test.mjs test/success-router.test.mjs
python3 -m unittest discover -s test -p 'test_robotwin_*.py'
python3 -m unittest discover -s test -p 'test_success_router*.py'
```
