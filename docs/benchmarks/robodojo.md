# RoboDojo

[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) is a unified sim-and-real manipulation benchmark: 42
simulation tasks on Isaac Sim / Isaac Lab and 18 real-world tasks (RoboDojo-RealEval) on ARX X5, Piper and Piper X
dual-arm robots, scored on a shared leaderboard. Its evaluator owns the control loop and calls policies through
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), a policy-server standard (WebSocket + MessagePack) that also
serves RoboTwin and RMBench.

REA joins as an XPolicyLab policy. The package [`integrations/xpolicylab/REA`](../../integrations/xpolicylab/REA/README.md)
implements XPolicyLab's `Model` contract (`update_obs`, `get_action`, `reset`, batched variants) and forwards every call to
[`rea serve`](../extending/policy-server.md). Any REA configuration can be evaluated: an LLM agent, a VLA client, a
code policy, or a routed or async System 0/1/2 runtime.

**Status:** the adapter is new. It follows XPolicyLab's published contract and is tested offline against `rea serve`
(`test/test_xpolicylab_adapter.py`). It has not yet been run on RoboDojo itself, and REA reports no RoboDojo results yet.
This page will be updated as the integration is validated.

## Setup

```bash
git clone https://github.com/InfRec-Cardinal/REA.git /path/to/REA     # Node.js 22.19+ in the policy environment
cp -r /path/to/REA/integrations/xpolicylab/REA /path/to/RoboDojo/XPolicyLab/policy/REA
cd /path/to/RoboDojo/XPolicyLab/policy/REA
conda activate <policy env>
bash install.sh        # copies XPolicyLab's launch scripts from demo_policy, installs XPolicyLab, checks Node
```

Set `rea_root` and `rea_config` in `deploy.yml`, then evaluate with XPolicyLab's usual arguments:

```bash
# offline wiring check (no simulator)
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls rea arx_x5 joint 0 0 0 <policy env> <eval env>
# simulation
bash eval.sh RoboDojo stack_bowls rea arx_x5 joint 0 0 0 <policy env> <eval env>
```

## What REA receives and returns

| XPolicyLab | REA |
|---|---|
| `obs["instruction"]` | `observation.text` |
| `obs["vision"][<camera>]["color"]` | `observation.images` (JPEG per camera; `cameras` in deploy.yml selects them) |
| `obs["state"]` (`left_arm_joint_state`, `left_ee_joint_state`, ... or `*_ee_pose`) | `observation.state`, plus `state.packed` (XPolicyLab packing order) and `state.robot` (dimensions, packing) |
| an action chunk: a list of `{left_arm_joint_state, left_ee_joint_state, right_arm_joint_state, right_ee_joint_state}` (or poses) | `action_chunk` (named commands) or `packed_action_chunk` (one packed vector per command) |

`examples/xpolicylab/chat.config.json` serves a vision-language agent through the chat policy, and
`examples/xpolicylab/lingbot-vla.config.json` serves a LingBot-VLA server through `packed_action_chunk`. Rename the
camera names in `imageMap` to the ones your task reports.

## Notes

- **Episode outcome.** RoboDojo scores episodes itself. When the environment client reports a trial result through
  XPolicyLab's `on_trial_end` hook, REA records its success in the episode summary.
- **Hold, never zero.** When REA ends an episode before RoboDojo does, the adapter repeats the measured state (or REA's
  last command). It raises an error rather than sending a zero command.
- **Batched evaluation** (`eval_batch: true`) runs one `rea serve` process per environment index. Model-backed agents
  are usually better served unbatched.
- **RealEval.** The same package serves the real-robot benchmark. Follow RoboDojo-RealEval's deployment and safety
  procedures; REA adds no robot-side safety layer of its own on this path.
