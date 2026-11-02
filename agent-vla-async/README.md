# G1 cerebellum: asynchronous agent brain / VLA runtime

The REA Node service can use this runtime as its actuator with
`G1_ACTUATOR=cerebellum`; see [the G1 guide](../docs/robots/unitree-g1.md). An intent-level
`wait_for_policy` option controls whether a submitted plan waits for advisory VLA
output. `scripts/fake_cerebellum.py` supplies deterministic adapters for tests.

## Installation

The package's protocol, mailbox, RPC, and mock runtime use only the Python
standard library. From this `agent-vla-async` directory, create an environment
and install the core package:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Install `python -m pip install -e '.[policy]'` for the GR00T and SONIC clients.
This adds NumPy for state/action arrays plus `msgpack`, `msgpack-numpy`, and
`pyzmq` for the policy transport. The full unit suite needs the `test` extra:

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests
```

The parent repository's `requirements.txt` installs NumPy and headless OpenCV
for `sense_worker.py` and `g1_u2_sense.py`. From this directory, install it with
`python -m pip install -r ../requirements.txt` when using those sensing workers.
These declarations retain Python 3.8 support; pip selects package releases
compatible with the active interpreter. They cover runtime libraries, not the
GR00T model-serving environment or the separate LIBERO backend.

Physical G1 execution also requires the Unitree SDK (`unitree_sdk2py`), a navigation package that provides the
`g1nav` session interface (`NavSession`, `NavCfg`, `LidarCfg`; see `src/agent_vla_async/navigation.py`), a RealSense
client, and the robot's camera and Livox services. These are robot-side integrations and are not included.
`scripts/g1vla` uses `G1VLA_PYTHON` as the interpreter and adds `G1VLA_EXTRA_PYTHONPATH` to `PYTHONPATH` for those
packages; `UNITREE_SDK2_PYTHONPATH` adds the Unitree SDK's directories for the arm bridge (`src/agent_vla_async/arm.py`).
Configure the policy service with `--groot-host` and `--groot-port`, the SONIC service with `--sonic-policy-host` and
`--sonic-policy-port`, and the local RPC path with `--socket`.

The cerebellum joins a robot-side navigation stack with a GR00T policy server. The agent brain runs on a workstation
(or another orchestration host), while the long-running cerebellum runs on the G1's computer.

```text
Agent brain (workstation)
  submit / heartbeat / cancel / status
                 |
                 | SSH, JSON request
                 v
G1 cerebellum daemon (robot computer)
  perception loop ---- RGB/depth/Livox/state
  inference loop  ---- latest intent -> REAL_G1 over 127.0.0.1:5550
  SONIC probe     ---- status/open-loop validation over 127.0.0.1:5551
  one-slot mailbox ---- stale action chunks are overwritten/discarded
  plan loop       ---- explicit Agent plan -> SPORT / named arm action
  watchdog        ---- intent TTL and brain heartbeat
```

The Unix RPC socket is robot-local and mode `0600`; the workstation reaches it through
an SSH alias (`G1_SSH_HOST`, default `unitree@g1`). A new intent supersedes the old intent immediately.
Inference already in flight is allowed to return, but its generation is checked
and stale output is discarded before reaching the actuator.

## Defaults

- The daemon is dry-run unless it is started with `--execute`.
- The GR00T client expects a checkpoint with matching `REAL_G1` modalities.
- Its state adapter currently uses explicit neutral placeholders for the arm,
  hand, wrist EEF and waist fields. The status event exposes that fact.
- GR00T `REAL_G1` output is advisory only. Even if a diagnostic navigation
  calibration is supplied, its output is never converted into walking pulses.
- Physical execution requires both an execute-enabled daemon and an intent with
  `execute=true` plus an explicit structured plan. SPORT motion continues to
  pass through the navigation stack's depth/Livox/robot-state gating. Arm stages use
  `G1ArmActionClient` named high-level actions discovered with `GetActionList`.
  The Python 3.11 VLA daemon invokes the robot's Python 3.8 SDK through the
  one-shot `arm_bridge_worker.py`; the bridge exchanges one JSON request and
  response and never exposes low-level motor commands.

## Execution availability

Query the live capability handshake:

```bash
scripts/brainctl capabilities
```

`EXECUTION_AVAILABLE` means the daemon itself was started in execute mode and
the plan's required high-level capabilities are currently ready. A dry-run
daemon reports installed components separately while keeping execution
unavailable. The response always includes:

- `scope=HYBRID_HIGH_LEVEL`
- `direct_vla_actuation=false`
- current SPORT state/depth/lidar verdict
- dynamic availability of `arm_action:forward_push` and
  `arm_action:release_arm`

SONIC is reported separately because its latent whole-body contract is not a
SPORT capability. The main advisory policy remains on `127.0.0.1:5550`; the
SONIC client independently defaults to `127.0.0.1:5551` and can be changed with
`--sonic-policy-host` and `--sonic-policy-port`. Querying status sends only
`ping` and `get_modality_config`:

```bash
scripts/brainctl sonic-status
```

The reply is `SONIC_POLICY_AVAILABLE` only when all four modalities match the
official `UNITREE_G1_SONIC` contract exactly:

- video: `[ego_view]`, delta `[0]`
- state: `[left_leg, right_leg, waist, left_arm, right_arm, left_hand,
  right_hand, projected_gravity]`, delta `[0]`
- action: `motion_token[64] + left_hand_joints[7] +
  right_hand_joints[7]`, delta indices `0..39`
- language: `[annotation.human.task_description]`, delta `[0]`

Missing keys, extra keys, key-order changes, or different temporal indices are
reported as individual reasons. Status stays `scope=POLICY_OPEN_LOOP`,
`physical_execution=false`, and `controller_mode=not_connected`; it does not
alter the existing high-level execution availability.

Request one explicit open-loop action validation with the current RGB frame:

```bash
scripts/brainctl sonic-validate "touch the door"
```

This first repeats the exact contract probe, then sends one policy request with
the current RGB plus exact-shape state tensors marked
`state_source=synthetic_neutral_placeholder`. The returned 40-step action chunk
is checked for the 64+7+7 dimensions, finite values, and motion-token bound.
The runtime does not create the port-5556 action publisher and does not call a
controller; the result reports `action_publisher_started_by_runtime=false` and
`physical_commands_sent=0`.

## Controller modes

The control plane names controller ownership explicitly:

- `HYBRID_HIGH_LEVEL` is the existing SPORT plus named arm-action path.
- `SONIC_WBC` is reserved for an external, matching GEAR-SONIC deployment.

Inspect both modes, prepare one, or stop one over the existing persistent RPC
connection:

```bash
scripts/brainctl controller-status
scripts/brainctl controller-prepare HYBRID_HIGH_LEVEL
scripts/brainctl controller-prepare SONIC_WBC
scripts/brainctl controller-stop SONIC_WBC
```

The production default is an unconfigured SONIC supervisor. It reports the
task checkpoint, observation adapter, controller binary/transport, and action
bridge separately, and `SONIC_WBC` remains `EXECUTION_UNAVAILABLE` until those
components are supplied by a matching external deployment. `prepare` never
launches the official controller and this repository has no low-level command
publisher. This framework currently cannot activate physical SONIC; even a
successful open-loop validation does not change that boundary.

For RPC integration tests only, start the daemon with
`--sonic-supervisor simulation`. The in-memory supervisor implements
`prepare/status/stop` and can select `SONIC_WBC` for open-loop protocol testing,
but always reports `physical_execution=false`. A physical SONIC submission is
therefore still refused with zero robot commands. Return ownership with
`controller-stop SONIC_WBC` or `controller-prepare HYBRID_HIGH_LEVEL`.

Build a plan without deriving motion from prompt text:

```bash
scripts/make_hybrid_plan.py \
  --task "walk to the door and push it open" \
  --pulse 0.10,0.0,0.30 \
  --arm-action forward_push \
  --arm-action release_arm \
  --output /tmp/door-plan.json
```

The numeric pulse must come from the Agent's scene-specific plan. It is not
inferred from GR00T's base `navigate_command`. Preview the entire asynchronous
chain without motion:

```bash
scripts/brainctl run "walk to the door and push it open" \
  --plan-file /tmp/door-plan.json --seconds 30
```

For an execute-enabled deployment, install
`deploy/execute-mode.conf` as the user service drop-in and restart the daemon.
That only enables the backend; no motion happens until a separate request also
uses `--execute` with a valid plan:

```bash
scripts/brainctl run "walk to the door and push it open" \
  --plan-file /tmp/door-plan.json --execute --seconds 30
```

The terminal result is separate from availability. It reports
`EXECUTION_COMPLETE` or `EXECUTION_FAILED`, controller receipts, and actual
SPORT/arm command counts.

The fused navigation gate reads a lidar obstacle snapshot produced by a separate service on the robot (for a
MID-360, run that producer under user systemd so the gate survives daemon restarts). The executor can alternatively be
started with `--no-lidar`, in which case live depth remains mandatory and the capability response shows that lidar is
optional. `deploy/` contains user-unit templates for the cerebellum; adjust their paths to your checkout.

## Run on the robot

From the checkout on the robot:

```bash
G1VLA_PYTHON=/path/to/robot/venv/bin/python scripts/g1vla daemon --policy groot
```

Robot-local status:

```bash
printf '%s\n' '{"op":"status"}' | scripts/g1vla rpc
```

## Run the agent brain on a workstation

Set `--host` (or `G1_SSH_HOST`) to the SSH host or alias of the robot, and `--remote-cli` (or `G1VLA_REMOTE_CLI`) to the
`g1vla` path on it when it is not on the robot's `PATH`.

```bash
scripts/brainctl status
scripts/brainctl capabilities
scripts/brainctl run "look ahead and propose the next motion" --seconds 15
scripts/brainctl submit "find the doorway" --heartbeat-timeout 8
scripts/brainctl heartbeat INTENT_ID
scripts/brainctl cancel INTENT_ID
```

`brainctl` now opens one SSH process and keeps a JSON-lines RPC stream alive for
all submit/heartbeat/events/cancel calls in the command. `run` therefore does
not repeat the SSH handshake for each heartbeat. It cancels the intent when its
requested window ends.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The tests cover latest-wins delivery, superseded inference discard, heartbeat
expiry, dry-run non-actuation, explicit plan execution, high-level SPORT
receipts, dynamic arm action discovery, and the rule that raw `REAL_G1` output
cannot actuate the robot.

## Persistent physical-step missions

The authoritative implementation is
`src/agent_vla_async/physical_steps_mission.py`. A "step" is counted only after
the read-only low-state monitor detects a confirmed alternating left/right hip
phase. It is not inferred from elapsed seconds or command count.

The motion path refreshes a 0.45 s high-level sport lease on a fixed 0.12 s
deadline. No zero command is sent between gait phases. The finalizer sends one
stop after the requested confirmed phase count, cancellation, stall, excessive tilt,
or a hard close-range obstacle. Depth/Livox observation runs independently;
short depth errors receive a 0.6 s grace window, while the slower 1 Hz state
reader can reuse its latest valid report for 2.2 s. A real not-ready state,
excessive tilt, or measured obstacle still stops immediately. Normal perception
and policy updates do not interrupt the lease.

Preview without physical motion, on the robot:

```bash
scripts/g1vla walk --steps 20 --vx 0.15
```

Run one physical walk of 20 confirmed steps (1-40 are accepted; the completion timeout is not a commanded duration).
The runtime's own pulses are limited to 0.20 m/s, while this skill's SPORT backend accepts up to 0.40 m/s:

```bash
scripts/g1vla walk --steps 20 --execute --vx 0.15
```

The JSONL event stream reports every confirmed `PHYSICAL_STEP`, lease count,
maximum lease gap, maximum command latency, missed refresh deadlines, sensor
errors, the final stop result, and the terminal state. Success requires the exact
target count, an unbroken lease, `stop_rc=0`, and a ready postflight state. The
daemon and mission remain dry-run unless execution is explicitly enabled.
