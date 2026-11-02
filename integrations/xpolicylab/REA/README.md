# REA (XPolicyLab adapter)

**Paper:** [REA: A Routed Embodied Agent Runtime for Asynchronous Brain-Cerebellum Control](https://arxiv.org/abs/2610.42053) |
**Code:** [InfRec-Cardinal/REA](https://github.com/InfRec-Cardinal/REA)

This package runs a REA agent as an XPolicyLab policy, for RoboDojo simulation and RealEval, RoboTwin and other XPolicyLab
environments. `model.py` implements XPolicyLab's `Model` contract and forwards each call to a `rea serve` process, so any
REA configuration can be evaluated: an LLM agent, a VLA client, a code policy, or a routed or async System 0/1/2 runtime.

## Installation

Node.js 22.19 or newer must be on `PATH` (or in `REA_NODE`) in the policy environment.

```bash
git clone https://github.com/InfRec-Cardinal/REA.git /path/to/REA
cp -r /path/to/REA/integrations/xpolicylab/REA XPolicyLab/policy/REA
cd XPolicyLab/policy/REA
conda activate <policy env>
bash install.sh
```

`install.sh` copies XPolicyLab's policy-agnostic launchers (`eval.sh`, `setup_eval_policy_server.sh`,
`setup_eval_env_client.sh`, `deploy.py`) from `demo_policy`, installs XPolicyLab in editable mode, and checks for NumPy,
Pillow and Node.

## Configuration

Edit `deploy.yml`:

| Key | Meaning |
|---|---|
| `rea_root` | the REA checkout (default `$REA_ROOT`) |
| `rea_config` | the REA configuration to serve, relative to `rea_root` or absolute, e.g. `examples/xpolicylab/chat.config.json` |
| `rea_output` | optional directory for REA's per-episode traces |
| `cameras` | colour cameras to send (default: all) |
| `max_chunk` | the most commands REA may return per `get_action` (default 50) |
| `robot_action_dim_info` | `{arm_dim, ee_dim}`; by default read from `env_cfg_type` |
| `step_timeout_s` | seconds to wait for one REA decision |

Model credentials for REA's chat policies come from `REA_MODEL`, `REA_BASE_URL` and `REA_API_KEY` in the policy
environment.

## Evaluation

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls rea arx_x5 joint 0 0 0 <policy env> <eval env>   # wiring check
bash eval.sh RoboDojo stack_bowls rea arx_x5 joint 0 0 0 <policy env> <eval env>
```

The arguments are XPolicyLab's: bench, task, checkpoint name (unused by REA), environment configuration, action type,
seed, policy GPU, environment GPU, policy and environment conda environments.

## Behaviour

- `update_obs` stores the latest observation. `get_action` sends it to REA and returns REA's action as an XPolicyLab chunk
  (`action_chunk` keeps XPolicyLab's keys; `packed_action_chunk` vectors are split in packing order).
- `reset` ends the open REA episode. The next observation starts a new one with `task_name`, `seed` and the observation's
  instruction. `on_trial_end(result)` records the trial's success.
- When REA ends an episode before the benchmark does, the adapter holds the measured state (or REA's last command) and
  never sends zeros.

Status: new. It is tested offline against `rea serve` (`test/test_xpolicylab_adapter.py` in the REA repository) but not
yet on RoboDojo itself.
