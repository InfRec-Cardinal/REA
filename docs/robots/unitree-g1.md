# Unitree G1 service

The G1 service is rea-agent's robot control path: persistent sensing and control
workers, a one-token vision decision, an in-process fused safety gate, a decision pipeline,
an asynchronous advisory reviewer and a learning brain. Start with the
[framework quickstart](README.md) for the shared agent API and configuration CLI; this guide
covers the HTTP service, control path and deployment.

The token engine requests one action token. The tool engine accepts a structured
action with parameters. Pipeline mode maintains a recent decision and rechecks
the execution conditions before dispatch; responses report decision age and the
gate's observation source.

## How it works

```
                    ┌───────────── persistent, never restarted ─────────────┐
  RealSense ───────►│ sense_worker.py      (numpy/cv2 loaded, scale cached) │
                    │ control_daemon.py    (DDS + LocoClient held open)     │
                    └───────────────────────────┬───────────────────────────┘
                                                │
  frame + depth ──► Prime RPC pool ──► token ──► fused gate ──► SPORT / arm action
                     │                    ▲
                     │                    ├── robot state  (FSM, mode_machine, SOC, tilt)
                     │                    ├── depth veto   (RealSense, floor 0.25 m)
                     │                    └── lidar veto   (Livox MID-360, azimuth sectors)
                     │
                     └─ fresh session per frame; tools/thinking/retries disabled
                                                │
  asynchronous reviewer (separate process, loopback only, no actuator)
        ◄── before/after frames + execution feedback
        ──► correction applied to the NEXT decision, never this one
```

### The token space

Tokens are canonical because a token is one output token. The legacy `{action, duration_ms}` space
from both ancestors maps onto them, so old clients keep working.

| token | action | envelope |
|---|---|---|
| `X` | stop | zero velocity |
| `W` | forward | 0.30 m/s, 2.0 s, −0.075 rad/s yaw trim |
| `S` | backward | 0.30 m/s, 0.6 s |
| `A` / `D` | strafe left / right | 0.30 m/s, 0.6 s |
| `Q` / `E` | turn left / right | 0.35 rad/s, 0.6 s |
| `U` / `P` / `R` | arms up / forward push / release | predefined arm actions 18 / 36 / 99 |

Speeds sit deliberately above the walking-controller deadband: below ~0.15 m/s the G1 accepts the
RPC and never takes a step, so a smaller number is not a slower robot, it is a silently ignored
command. `motionFor()` enforces that floor — except on `W`'s yaw trim, which is a drift correction
and would become a curve if it were inflated.

### Engines

| engine | output | when |
|---|---|---|
| `prime` (default) | Prime Agent 0.8.0 persistent RPC pool, 1 token | production reactive path |
| `token` | raw streamed API, 1 token | latency baseline and explicit rollback |
| `tool` | forced `report_action` call, compact or full schema | when you want a model-chosen duration |
| `react` | bounded ReAct: thought → observe / recall / finish, ≤ `G1_REACT_MAX_ITERATIONS` tool calls | the router's escalation target when the reflex is failing or the situation is risky |
| `cli` (opt-in) | image and token prompt through the installed Claude or Codex CLI | deliberate candidates and asynchronous roles; enable `G1_CLI_ENGINES=1` |
| `scripted` | a fixed token sequence from `G1_SCRIPTED_TOKENS`, 1 token, no provider | the offline service demo and CI |

Engines are registered in `src/engines.mjs` with what the service needs to know about them (one
output token or not, deliberate tier, whether the gateway key is needed, what `/health` reports).
An unknown `cfg.engine` on a request is a 400 with the registered names; an unknown `G1_ENGINE`
fails at startup.

All engines normalise onto the same token, so the gate and executor never learn which one ran.
Prime uses two resident RPC processes by default, preserving the old two-way decision pipeline and
primary/fallback race without mixing event streams between concurrent prompts.
Startup warms every resident worker on the DeepSeek vision primary, so the first request does not
pay a model-switch penalty.

The `prime` and `token` engines default to no fallback. A fallback for the
single-token path must support the same output contract. `tool` allows
a larger response budget. Set `G1_FALLBACK_MODEL` explicitly to select a fallback
and configure its provider credentials on the service host.

CLI candidates require the selected official client to be installed and configured
on the service host. `G1_CLI_CLAUDE_BIN` and `G1_CLI_CODEX_BIN` select executables;
`G1_CLI_TIMEOUT_MS` (default 60000) and `G1_CLI_EFFORT` (default `low`) configure the
call. The CLI path parses the final answer and usage after the process completes,
so its reported token-ready time includes the whole client call. It does not have
the streamed one-token latency of the reflex path.

### Robot profile and sensors

The tokens, their motions, the arm interlock, the walking envelope and the gate thresholds are one
frozen object, the robot profile (`src/profiles/g1.mjs`). `G1_ROBOT_PROFILE` selects it: unset or
`g1` for this robot, `libero` for the manipulation vocabulary, or a module path whose default
export is a profile spec (see [the framework reference](../framework.md#robot-profiles)); the path must start
with `./` or be absolute, since a bare name is treated as a package specifier. Every module that
reasons about tokens takes the profile as a parameter, and the gate, executor, judge, memory and
brain never name `W` or `X` directly. The scripted engine's default token script is the profile's
forward role (its stop when the profile has none), so a profile without `W` still starts; an
explicit `G1_SCRIPTED_TOKENS` is always taken as written and must name tokens of the profile in
use. Residue: two router paths (`src/router.mjs`, `src/infer-router.mjs`) still compare the last
token against the G1 stop letter, which is harmless under another profile because a stop never has
a refused gate; `src/infer-tool.mjs`'s compact schema (`COMPACT`/`TOOL_COMPACT`, the default
`G1_SCHEMA=compact`) maps G1 action names only, so under another profile the name does not resolve
to a token and the step stops; and the `tool` engine sends an empty `enum` for a profile
with no `ACTIONS`, which providers reject. The last two are fail-closed: the step refuses rather
than guessing a motion.

Sensing is selected by `G1_SENSE`: `worker` (persistent `sense_worker.py`, the default when
`G1_PERSISTENT_CAPTURE=1`), `subprocess` (`g1_u2_sense.py` per capture) or `fixture` (in-process
synthetic frames; `G1_FIXTURE_DEPTH="c,l,r;c,l,r"` metres, cycled). Obstacles come from
`G1_OBSTACLES`: `snapshot` (the Livox producer's file at `G1_LIDAR_SNAPSHOT`) or `fixture`
(`G1_FIXTURE_LIDAR="center,left,right"` metres). Because both fixtures answer the gate with
constants whatever is in front of the robot, `loadConfig()` refuses to start when either is
combined with `G1_EXECUTE_ENABLED=1` and an actuator that can move the robot; use
`G1_ACTUATOR=mock` for fixture runs. The depth summary the gate consumes carries
`center_m`, `left_m`, `right_m`, `front_min_m`, `center_valid_fraction`, `front_valid_fraction`
and `scale_m`; lidar summaries carry `usable`, `blocked`, `reason`, `age` and the sector
distances `center`, `front`, `left`, `right`.

## The brain: memory, judge, router, deliberation

The reflex loop above answers in one token and cannot afford to think. The brain wraps `/v1/step`
with the two agent patterns that make a controller *learn from its own execution* without putting a
slow model on the reflex path:

- **Agent-as-a-Router** (ACRouter): every step is
  routed `cheap chain → verify → escalate`. The one-token reflex engine is the cheap chain, the fused
  gate plus the outcome judge are the verifier, and a bounded **ReAct** engine is the escalation
  target. Escalation happens *before* a step when the episodic signals say the reflex is failing
  (`k` refusals, `k` bad judgements, oscillation between opposite tokens, a forward stall, a reviewer
  STOP/CORRECT, or something near with a motion token pending) and *inside* a step when the reflex
  token is vetoed and there is deadline left: the deliberate engine is asked once with the veto as its
  first observation, and its answer is re-gated on a fresh frame. A budget caps the deliberate
  fraction; per-dimension engine scores (an EMA of judge scores, ACRouter's routing memory) let a
  clearly better engine win a situation class once it has enough samples.
- **ReAct** (Thought → Action → Observation): `src/infer-react.mjs` runs at most `G1_REACT_MAX_ITERATIONS`
  forced `react_step` tool calls. Each carries a brief thought and one action — `finish(token)`,
  `observe()` (fresh depth, lidar and robot state through a read-only callback) or `recall(query)`
  (the experience memory). It normalises onto the same single token as every other engine, so the
  gate and executor never learn that anyone thought.
- **Two-tier memory** (`src/memory.mjs`): an *episodic* ring per session (what was proposed, what the
  gate said, what happened) and a persisted *experience* store of lessons keyed by a coarse situation
  signature (`centre near / left far / lidar clear`) and a task key, consolidated by hits and
  bounded. Recalled lessons are rendered into a size-capped block (`G1_MEMORY_PROMPT_MAX_CHARS`,
  default 240) that the reflex prompts carry as `Experience:` — never on the `min` variant, which is
  the latency floor.
- **Judges** (`src/judge.mjs`): a heuristic judge scores every step from evidence already at hand
  (veto, actuator rejection, depth gained by a forward pulse, oscillation, fail-closed stops) and
  writes a lesson naming a geometric alternative; the asynchronous reviewer's KEEP / CORRECT / STOP
  is converted into a weighted lesson **whether or not it arrived in time to steer** — corrections
  are the most valuable lessons, and a stale one is still a judgement of the step it reviewed. An
  out-of-process judge (a stronger VLM, an offline replay, a human) can post lessons on
  `POST /internal/judge/lesson` (loopback only).
- **The reviewer is also the asynchronous judge.** Every review answers two questions in one tool
  call. The steering half (`verdict`, `next_token`, `brief`, `confidence`) is the strict correction
  protocol, unchanged. The judging half (`outcome` ∈ progress / no_progress / unsafe / unclear, plus a
  one-sentence `lesson`) is what the reviewer concluded about the step it watched, before and after
  frame in hand. The thinker posts it to `POST /internal/judge/review` (loopback only) *after* the
  correction, and still posts it when the correction was stale — a judgement is never on time to steer,
  it exists to teach. The brain stores it as a reviewer-sourced lesson (the model's own words, weighted
  by confidence), and a negative verdict overrides a non-negative heuristic score so the router's
  bad-step streak reflects what actually happened. The correction half already scored the engine for
  that step, so the judgement never scores it a second time.

- **One step empowers the next.** The deliberate engine may finish with a `next` line — what the
  following step should watch for or do ("expect the door frame at centre after the turn; W only if
  centre > 1.5 m"). The brain carries it to exactly the next step of that session as a `Plan:` line
  in the reflex prompt (advice, never a command — the gate still decides) and then drops it: a plan
  older than the immediately preceding step describes a robot that has since moved.
- **Reflection (Reflexion).** When a step is vetoed or judged bad, or the episode oscillates, stalls or
  keeps being refused — the same signals the router escalates on — a text-only model is handed the
  last few steps, the gate's reasons and the lessons already known, and asked for *one* lesson: what
  to do instead in this situation and why (`src/infer-reflect.mjs`). Fire-and-forget, one in flight,
  at most one per `G1_REFLECT_COOLDOWN` steps per session, and its only output is a `reflection`
  lesson in memory. The heuristic judge sees one step and the reviewer sees one step with its after
  frame; reflection is what sees the shape of a failing episode.
- **A finer situation signature.** Lessons include left/right clearance asymmetry
  and a scene descriptor from the sensing worker. Recall can prefer a similar
  viewpoint, while lessons from distinct scenes remain separate. Entries without
  scene metadata do not overwrite scene-tagged entries.
- **Model-authored text is data, never an instruction (`src/sanitise.mjs`).** Everything a model writes
  into memory — reviewer lessons, asynchronous judgements, reflections, external judges, the deliberate
  engine's `next` line — is flattened to one printable line without `[ ]` (so it cannot forge a
  `[+]`/`Advice:` line of its own) and refused when it reads as an instruction to the next model: a
  prompt label (`Advice:`, `Token map:`, or `Task:`/`Depth:`/`Answer:` at the start), "ignore the
  depth gate", "always output W", "Answer: W", "you must never output X", "regardless of obstacles",
  "BOX_LIFT is available", and the same in Chinese. Rejections — of lessons at write and at load, and
  of the deliberate engine's `next` lines — are counted per reason in `GET /v1/brain`
  (`memory.rejected`, `directives.rejected`) and logged with the source, so a judge that keeps writing
  instructions is visible. The filters are narrow on purpose so a judge's natural wording ("never pick
  W under 0.5 m", "you are 0.4 m from the wall", "centre depth: 0.42 m") passes, and a rejected text
  is dropped, never rewritten. It is a blocklist, not a proof: what bounds an instruction that slips
  through is that a lesson only biases the token asked for, and the gate decides on live evidence.
- **Episode store (`src/episodes.mjs`).** The third memory tier: every step of every session,
  appended to `G1_EPISODES_PATH` (JSONL, compacted to the newest `G1_EPISODES_MAX_SESSIONS` on load)
  and digested per session — steps, refusals and where, escalations, deliberate fraction, reviewer
  verdicts, asynchronous judgements, how it ended. A new session on the same task is told how the last
  `G1_EPISODES_RECALL_K` went ("Past episode: 12 steps; 3 refused (3x W at centre near); reviewer
  STOP x1; ended refused on W") in the deliberate prompt and the reviewer's context, never on the
  reflex path. The per-session episodic ring is `G1_EPISODIC_CAPACITY` steps (default 64). The same
  file is what an evaluation replays offline. `POST /v1/brain/episodes {instruction, k}` returns the
  digests.
- **Candidates and the model router (`src/candidates.mjs`, `src/infer-router.mjs`, `src/infer-cli.mjs`).**
  The router chooses between *candidates*, (engine harness, model, tier) triples declared in
  `G1_ROUTER_CANDIDATES`, so "route among agents" means route among models: a fast one-token
  reflex, a slower vision model, the ReAct engine on a bigger model, or a frontier model behind its
  official CLI (`cli@<model>`; `G1_CLI_ENGINES=1`) that only fits the
  deliberate tier and the asynchronous roles. Decision responses expose the provider's token counts
  and a USD cost (`usage`, `cost_usd`) when available, and the router keeps per-candidate latency and cost next to its
  per-situation quality scores. `G1_ROUTER_MODE=model|hybrid` adds Agent-as-a-Router's LLM router: after
  each step a lightweight model (`G1_ROUTER_MODEL`, default `REA_MODEL`) reads the candidates with
  their prices, measured latency and per-situation quality, the episodic signals and the last outcome,
  and suggests the candidate for the *next* step (with epsilon-greedy exploration). The suggestion is
  matched to that one step; late, absent or over-budget suggestions fall back to the rules, and in
  `hybrid` mode a safety trigger always outranks the model. [The evaluation guide](../../eval/README.md)
  covers the included checks.
- **Spatial memory (`src/spatial.mjs`).** The harness gets no odometry, so each session keeps a
  dead-reckoning pose integrated from the pulses that were actually dispatched (the cerebellum reports
  the clamped pulse it sent) and snapped to the IMU yaw when the actuator reports one. Every step is
  marked at the pose it was decided from — token, refusal, outcome, centre depth — and the trail
  answers "have I been here, facing this way, before?" Consecutive steps at one spot are one stay;
  coming *back* to a spot after moving is a revisit, and a revisit to a spot where a token was refused
  (or a third visit) is a router trigger (`revisit`) and a reflection trigger, the loop that the token
  history alone shows only later. The deliberate engine and the reviewer are told where the robot
  thinks it is, with the confidence (it drifts, and it is never safety evidence).
  `POST /v1/brain/session {session_id}` returns a session's steps, poses and trail.
- **The reactive path learns too.** `/v1/act` and the pipeline have no HTTP session, so they share one
  process-wide episodic session: every reactive act is judged and remembered exactly like a routed
  step (a veto, a rejected dispatch, a clean pulse, a fail-closed stop), and every fresh reactive
  decision recalls the lessons that session's last situation matches. The router is not consulted on
  this path, so those acts do not count against its deliberate budget, but they do feed its
  per-dimension engine scores.
- **Routing memory persists.** The router's per-dimension engine scores — ACRouter's routing memory,
  the part of the router that learns — are written to `G1_ROUTER_MEMORY_PATH` (default
  `runtime/memory/router.json`) and restored at start-up, next to the experience store.
- **Corrections rank first.** Recall weights a lesson by its source: a reviewer or external judge saw
  the outcome, so their lessons outrank a same-scored heuristic one, and a lesson that names an
  alternative token outranks one that only says "that was bad".

```
  /v1/step ─► route (signals, situation) ─► recall (task, situation) ─► reflex engine ─► gate
                 │                                                         │ vetoed & time left
                 │ k refusals / oscillation / stall / reviewer flag / risk   ▼
                 └──────────────────────────────────────────────► ReAct engine ─► gate (fresh frame)
                                                                              │
                                        judge ─► lesson ─► experience memory ◄┘  (+ router scores)
                                              ▲
             thinker verdict (fresh or stale) ─┘        external judge ─► /internal/judge/lesson
```

Every arrow above changes only what is asked of a model or which model is asked. The actuator is
reached exclusively through the executor, with the same double opt-in and the same gate.

| endpoint | purpose |
|---|---|
| `GET /v1/brain` | memory stats, router state (steps, deliberate fraction, per-dimension engine scores) |
| `POST /v1/brain/recall` | `{instruction, situation?, token?, k?}` → the lessons the next step would see |
| `POST /v1/brain/session` | `{session_id}` (or `"act"`) → that session's episodic steps, signals, dead-reckoning pose, trail and digest |
| `POST /v1/brain/episodes` | `{instruction \| task_key, k?}` → digests of the last sessions on that task, as the deliberate engine sees them |
| `POST /internal/judge/lesson` | loopback-only: `{instruction|taskKey, evidence|situation, token, outcome, score∈[-1,1], lesson, alternative?, engine?}` |
| `POST /internal/judge/review` | loopback-only, posted by the thinker: `{sessionId, basedOnSeq, outcome, lesson?, confidence?, verdict?, next_token?}` → a lesson about that step |
| `POST /internal/brain/reset` | loopback-only: `{confirm: true, experience?: true, router?: true, episodes?: false}` → forget what was learned (an evaluation's cold arm); each file is rotated to `*.bak-<unix_ms>` first; 400 without `confirm` |

`/v1/step` responses gain a `brain` block: `engine`, `route` (`default`, `refusal_streak`, `risk_near`,
`revisit`, `memory`, `explicit`, …), `escalated`, `inline` (`{from, engine, token, ok, thought}` when
verify-and-escalate ran), `thought`, `lessons_used`, `episodes_used`, `judge {score, outcome,
alternative}`, `directive` (the plan this step was given), `next` (the plan it left), `reflection
{status, trigger}`, `signals`, `where` (the dead-reckoning line the deliberate engine saw), `history`
(the past-episode lines), `spatial {revisits, recentRevisits, refusedHere, travelledM, turnedRad,
confidence}`, `experience`, `memory_size`. `cfg.engine` still forces an engine per request;
`G1_BRAIN_ENABLED=0` restores the plain decide → gate step.

Candidate specifications use `engine@model[:tier[:timeoutMs]]`, separated by
commas, or a JSON array in `G1_ROUTER_CANDIDATES`. Without an explicit list the
existing cheap chain and escalation engine remain the candidates. For example,
`token@your-fast-vision-model:reflex,cli@your-cli-model:deliberate:60000`
keeps a one-token reflex and makes a model behind its official CLI a deliberate option.
`G1_MODEL_PRICES` can override each model's input/output USD-per-million-token
weights. `cost_usd` uses a provider-reported cost when supplied, otherwise the
available usage and configured price table; unknown usage or prices remain
unknown. These routing estimates are not a provider invoice.

## Deployment layout

One checkout, one Node runtime. The service runs from the checkout directory (`@TARGET@` in the
unit templates) and expects a Node 22 runtime under `runtime/node` and the Prime Agent release
under `runtime/prime-agent`; `npm run setup` installs both. systemd user units are
`rea-agent.service` and `rea-thinker.service`, rendered from the
`scripts/g1/*.service.in` templates by `scripts/g1/deploy_to_g1.py`, which also stops and disables the units listed
in its `LEGACY_SERVICE_NAMES` (empty by default) before enabling the new ones.

## Safety

These are properties of the code, not conventions:

- **The default deployment is inference-only.** Its environment pins `G1_ACTUATOR=none`,
  `G1_PERSISTENT_CONTROL=0`, and `G1_EXECUTE_ENABLED=0`; an external controller (for example a console service
  on the robot) is the only component that converts a returned token into a bounded SPORT pulse.
- **Standalone execution remains an explicit mode.** It needs a real actuator, service execution
  enabled, and `execute`/`exec_mode` on the request. Missing any one of those leaves it preview-only.
- **The compatibility bridge never actuates.** `/v1/control/step` always forces `execute:false`; the external
  controller remains the only component allowed to turn a returned token into a bounded pulse.
- **A refused real execution actively stops.** A gate veto on an explicit execution request issues
  zero velocity rather than letting an earlier pulse run on. Preview traffic never touches the
  actuator, including when the gate refuses the proposed token.
- **Fail closed.** Any inference, capture, or parse failure resolves to `X`, reported as `safe_stop`.
  The exported `verdict()` fails closed too: a forward or strafe token without a gated depth reading
  is refused as `depth_missing`, whoever calls it.
- **Depth is re-read at execution time.** A pipeline decision is up to `fresh_max_s` old and a
  deliberated one follows seconds of thinking; before either moves the robot the gate takes one
  depth-only capture (`G1_GATE_LIVE_DEPTH=1`, default) and vetoes on *that* depth. The response
  reports `gate_frame: live | decision` (decision = the capture failed and the decision frame was used).
  Robot state and lidar were always read live.
- **Requests can only tighten safety.** `fresh_max_s` is capped by `G1_PIPELINE_FRESH_MAX_S`;
  `use_lidar:false` is ignored unless `G1_ALLOW_REQUEST_LIDAR_OFF=1` (the response then carries
  `lidar_override_ignored`); depth thresholds never go below the floors; `force` on `/v1/arm` overrides
  the arm-holding interlock only, never robot readiness.
- **A shared secret guards the network.** `G1_AGENT_TOKEN` makes every route except `GET /health`
  require `Authorization: Bearer <token>`; the service logs `unauthenticated_lan_bind` when it binds a
  non-loopback address without one. `G1_DIRECT_EXECUTE_ENABLED` (legacy alias) is parsed as a boolean,
  so `0`/`false`/`off` cannot enable execution.
- **A stop is never vetoed.** `X` executes regardless of gate state — refusing to stop is never safe.
- **Depth floors are hard.** Contact work can lower the veto by explicit opt-in, never below 0.25 m
  centre / 0.20 m side.
- **Lidar is read by azimuth, not radius.** The MID-360 on this robot is self-occluded on the right;
  a nearest-return radius check reports a permanent obstacle that is the robot's own shell.
- **Never `lowcmd`.** Only the sport-service `SetVelocity` and predefined arm-action RPCs. A lowcmd
  takeover means commanding all 17 joints correctly on the first packet or the robot falls.
- **Arm interlock.** A second holding pose without a release errors 7401 on the arm service, so it is
  refused locally instead of being sent.
- **No grasping.** `BOX_LIFT` is advertised as unavailable in every prompt variant and capability
  response. There is no validated grasp-and-lift primitive; clamping is not grasping, and the
  standing arm cannot reach below ~0.51 m regardless.
- **The reviewer cannot act.** It binds to loopback, holds no SDK handle, and the correction endpoint
  rejects non-loopback callers. Its advice applies to the *next* decision; the main loop never blocks
  on it.

## Run

### Offline quickstart

Install Node.js **22.19 or later**, then:

```bash
git clone https://github.com/InfRec-Cardinal/REA.git
cd REA
npm run demo:service
```

This starts the real service with a fixture camera, fixture lidar, the mock actuator and the
scripted engine, sends two preview requests and prints the gate allowing `W` on a 2.5 m frame
and refusing it on a 0.3 m frame. Nothing is installed, no key is needed and no robot is
observed. `npm run demo:offline` is the older, smaller variant: a local streaming provider
fixture behind the token engine. Both are runtime demonstrations, not robot results.

### Prime and robot setup

```bash
npm run setup
$EDITOR .env
```

Setup installs the official Prime Agent **0.8.0** release and its dependencies into
`runtime/prime-agent`, checks its version, creates `.env` if absent, and runs the
Node tests including the real Prime RPC smoke test against a local provider fixture.
The initial install requires network access and npm; the bridge tests also need
Python 3.8 or later. Existing `.env` values are
preserved. `PRIME_AGENT_PACKAGE=/path/to/prime-agent-0.8.0.tgz` selects a downloaded
release archive; npm still needs access to uncached dependencies. `REA_NODE` and
`REA_NPM` select alternate executables by absolute path.

Set `REA_API_KEY` in `.env` and configure the provider/model and robot services for
your host. Both entry points load the repository's `.env` automatically; exported
environment variables take precedence. `G1_ENV_FILE` selects a different file.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Set `G1_SENSE_PYTHON` in `.env` to the absolute path of `.venv/bin/python`.
Start the service with `npm start`. Run `npm run thinker` in a second terminal for
the optional asynchronous reviewer.
The sensing requirements install NumPy and headless OpenCV. For the Python
cerebellum's optional policy client, see
[agent-vla-async installation](../../agent-vla-async/README.md#installation).

The robot path additionally requires the RealSense socket producer configured by
`G1_REALSENSE_SOCKET`, the Livox snapshot producer configured by `G1_LIDAR_SNAPSHOT`,
and the selected actuator backend. These host services are not installed by setup.
The direct daemon uses Unitree SDK2 through `G1_CONTROL_SCRIPT`; the console backend
uses an existing console service on the robot at `G1_CONSOLE_BASE_URL`. Configure those existing
robot integrations before making live requests. The offline example is the
standalone path for a machine without them.

From a workstation, `python3 scripts/g1/deploy_to_g1.py deploy --host <robot-computer>` performs the same
installation over SSH (`--user` defaults to `unitree`, `--target` to `/home/<user>/rea-agent`;
`G1_DEPLOY_HOST` can stand in for `--host`). Units named in `LEGACY_SERVICE_NAMES` are stopped and disabled before
the new ones start. If startup or the bound-unit health check fails, it stops the replacement and restores the
previous service state.

The shipped environment is decision-only: it evaluates the fused gate and reports what it would
have sent, while `/v1/control/step` is additionally hard-pinned to `execute:false` in the HTTP
handler. Standalone lab execution can still be enabled deliberately by selecting `daemon` or
`console`, enabling persistent control as needed, and setting both execution opt-ins.

### Endpoints

| endpoint | purpose |
|---|---|
| `GET /health` | config, worker health, live robot readiness |
| `GET /v1/capabilities` | token map, engines, gate thresholds, manipulation limits |
| `GET /v1/control/capabilities` | website-compatible Prime harness capabilities |
| `POST /v1/action` | decision only — no gate, no actuator |
| `POST /v1/act` | decision → fused gate → motion; supports `pipeline`, `force_token` |
| `POST /v1/step` | the same inside a session, with the reviewer in the loop |
| `POST /v1/control/step` | website-compatible session step, forcibly preview-only |
| `POST /v1/pipeline` | start/stop the continuous decision loop |
| `POST /v1/sense` | one frame: depth summary, lidar sectors, robot state |
| `POST /v1/arm` | arm primitive by token (`U`/`P`/`R`) or raw firmware `action_id`; needs `execute:true` like every motion, otherwise a gated preview |
| `POST /v1/stop` | zero velocity |

`cfg` accepts `engine, promptVariant, schema, maxWidth, quality, filter, includeDepth, imageDetail,
maxOutputTokens, primaryModel, fallbackModel, race`.

Only one `step` may run in a session at a time. An overlapping request to
`/v1/step` or `/v1/control/step` returns HTTP 409 with `error_type: session_busy`;
send the next step after the active one completes. Different sessions can run
concurrently.

Reactive control:

```bash
curl -sX POST :8091/v1/pipeline -d '{"on":true,"concurrency":2,"cfg":{"engine":"prime"}}'
curl -sX POST :8091/v1/act -d '{"pipeline":true,"exec_mode":"real","fresh_max_s":1.0}'
```

## Tuning

`eval/prompt_latency.mjs` measures the decision path's latency for prompt variants against a provider
(`npm run eval:latency -- --frame frame.jpg`), with paired trials so a variant that is faster because it fails
cannot win.

## Tests

```bash
npm test
```

The suite covers the token/action mapping and deadband floors, gate vetoes per token kind, the
arm interlock, the double opt-in for execution, refusal-issues-a-stop, the fallback chain and race
cancellation, the single-token stream contract (including a token that lands in `reasoning_content`),
the correction protocol's staleness and schema rules, Prime LF-JSONL parsing, `agent_end` completion,
fresh sessions, model switching, abort/restart, two-process concurrency, and website compatibility.

The suite also covers the reviewer runtime and HTTP layer.

## Layout

```
server.mjs              service wiring
thinker.mjs             asynchronous reviewer (separate process)
scripts/g1/sense_worker.py      persistent RealSense capture
scripts/g1/control_daemon.py    persistent DDS + LocoClient
src/actions.mjs         unified token space and motion envelope
src/profiles/           robot profiles (G1, LIBERO)
src/provider.mjs        shared streaming transport, keep-alive dispatcher
src/infer-token.mjs     single-token engine
src/infer-prime.mjs     official Prime Agent 0.8.0 LF-JSONL RPC adapter
src/prime-pool.mjs      two-process persistent pool for pipeline/race concurrency
src/infer-tool.mjs      forced tool-call engine
src/infer-react.mjs     bounded ReAct deliberate engine (router escalation target)
src/infer-reflect.mjs   text-only reflection: one lesson over a failing episode (Reflexion)
src/decide.mjs          capture → infer → decision, fallback chain and race
src/brain.mjs           memory + judge + router composed around one step
src/memory.mjs          episodic ring per session, persisted experience lessons
src/spatial.mjs         dead-reckoning pose, visited places, revisit detection (memory, never safety evidence)
src/episodes.mjs        cross-session episode store: every step, per-session digests, JSONL
src/sanitise.mjs        model-authored text → one printable line, or refused as an instruction
src/judge.mjs           heuristic outcome judge, reviewer verdict → lesson
src/router.mjs          ACRouter-style cheap → verify → escalate engine routing
src/candidates.mjs      engine/model/tier candidates and configured price estimates
src/infer-router.mjs    asynchronous model suggestion for the next decision
src/infer-cli.mjs       opt-in Claude/Codex CLI inference adapter
src/infer-scripted.mjs  fixed-token engine for offline runs
src/engines.mjs         engine registry
src/gate.mjs            fused depth + lidar + state gate
src/sensors/            obstacle sources
src/executor.mjs        gate → actuator, arm interlock
src/cerebellum.mjs      agent-vla-async cerebellum as an actuator (Unix-socket RPC, bounded plans, heartbeat)
src/pipeline.mjs        continuous decision loop
src/sessions.mjs        session, sequence, correction protocol
src/agent.mjs           the service object
src/http.mjs            routing and response shaping
scripts/g1/              robot-side workers (sense_worker.py, g1_u2_sense.py, control_daemon.py), unit templates
scripts/g1/deploy_to_g1.py  bundle and deploy to the robot computer over SSH
prime-agent-config/     image-capable one-token model catalog and retry/compaction settings
agent-vla-async/        vendored asynchronous Agent-brain / VLA-cerebellum runtime (Python; see below)
scripts/setup.sh        local setup
```

## agent-vla-async (vendored)

`agent-vla-async/` contains the asynchronous **Agent brain / VLA cerebellum**
runtime in Python (`agent-vla-async/src/agent_vla_async`). Run its tests with `npm run test:py`.
It provides:

- a robot-side **cerebellum daemon** (`g1vla daemon`) that owns perception, a GR00T advisory policy
  loop, a one-slot latest-wins mailbox, generation-stamped stale-output discard, a heartbeat / TTL
  watchdog, and a structured **ExecutionPlan** executor (bounded SPORT pulses ≤ 0.20 m/s / 0.35 rad/s
  / 0.60 s plus named arm actions) behind a robot-local Unix-socket JSON RPC
  (`submit / heartbeat / cancel / status / events / capabilities`);
- a workstation-side **brain CLI** (`brainctl`, `g1-vla-brain`) that drives it over one SSH stream;
- a gait-verified bounded walk skill (`g1vla walk`) and example execution plans (`agent-vla-async/plans/`).


### The cerebellum as an actuator

`G1_ACTUATOR=cerebellum` makes the cerebellum this service's actuator (`src/cerebellum.mjs`), so
the one-token brain drives the robot *through* the asynchronous runtime instead of through
`scripts/g1/control_daemon.py`:

```
  token ─► fused gate ─► executor ─► cerebellum bridge ─► submit(plan, execute:true) ─► cerebellum
                                             │ heartbeat every 0.5 s                          │ re-gates each pulse
                                             ◄──────── events until a terminal kind ◄─────────┘ (depth, lidar, state)
```

- A decided pulse is re-shaped into the cerebellum's `NavigationPlan`: speed clamped into its envelope
  (|vx| ≤ 0.20 m/s, |yaw| ≤ 0.35 rad/s) and the duration split into `ceil(duration / 0.6)` equal
  sub-pulses. A decided `W` (0.30 m/s × 1.0 s default) becomes two 0.5 s pulses at 0.20 m/s; the
  2.0 s envelope cap becomes four — a bridged pulse covers *less* ground than a direct one, never more.
  Strafes (`A`/`D`) and `U` (the cerebellum advertises no `both_hands_up` action) have no cerebellum
  equivalent; the executor refuses them at the gate (`actuator:unsupported_token`) like any other veto,
  so the brain learns it.
- Latest wins, like the cerebellum itself: a second real pulse supersedes the one in flight (the
  cerebellum stops the first intent, then runs the second) and the first is reported as `superseded`
  with the pulses it did complete. Nothing is ever queued behind a running intent — a queued pulse
  would execute on stale gate evidence, and `stop()` must leave nothing waiting to move afterwards.
  If the transport fails while a plan runs, the bridge cancels the intent (the cerebellum's
  deactivation path issues the stop) and reports `TRANSPORT_ERROR`; if even that fails, the
  cerebellum's heartbeat watchdog expires the intent and stops the robot on its own.
- The intent is submitted with `execute:true` **and** `wait_for_policy:false` (a small addition to
  the vendored protocol): this service has already decided, so the bounded pulse is not held behind a GR00T
  round-trip. The cerebellum's own gate still runs before every pulse, and its heartbeat/TTL watchdog
  expires the intent — stopping the robot — if this process dies mid-pulse.
- `stop()` is `cancel` + `controller_stop`, the cerebellum's zero-velocity path. `status()` maps the
  capability handshake onto the gate's `{ready, reasons}`: `ready` is the robot's state (usable,
  standing, not in dry-run mode); the cerebellum's forward path-clear verdict is surfaced separately as
  `blocked_ahead`, because this service's gate already decides per token what a near obstacle means (an arm
  gesture in front of a door is fine, a forward pulse is not). Arm tokens map onto the cerebellum's
  named actions (`forward_push`, `release_arm`).
- The cerebellum socket is robot-local (`G1_CEREBELLUM_SOCKET`, mode 0600), so this only works with
  the service on the same host as `g1vla daemon --execute`. Both double opt-ins still apply: this service's
  (`G1_EXECUTE_ENABLED` + `execute:true`) and the cerebellum's (`--execute` + the intent's flag).

`test/cerebellum.test.mjs` drives the real cerebellum runtime (with its deterministic fake adapters,
`agent-vla-async/scripts/fake_cerebellum.py`) over a Unix socket and checks that a 2.0 s bridged `W`
arrives as four 0.2 m/s pulses and that `stop()` reaches `navigation.stop()`.

## Evaluation

`eval/` holds the scripts that check the brain's claims instead of asserting them: a replay sense
worker that runs the whole service on recorded frames without a robot, a paired latency test of the
Experience block, a cold-vs-hot memory comparison against a running service, and an offline analysis
of the episode store across sessions. [The evaluation guide](../../eval/README.md) says what each can and cannot show; none of them
ever actuates. `POST /internal/brain/reset` (loopback-only) is the cold arm's memory wipe.

## Caveats

- Pipeline mode steers on a decision up to `fresh_max_s` old. Safety does not depend on freshness —
  the gate re-reads robot state, lidar and (with `G1_GATE_LIVE_DEPTH=1`) depth at execution time — but
  steering quality does.
- Lidar and the depth camera both see through glass. Only the black frames of a glass door register.
- Some thinking models put short answers in `reasoning_content` and return an empty `content` unless
  `max_tokens` leaves room for the reasoning; give judge and reviewer calls at least a few hundred tokens.
- The reviewer is off the reactive path by design. Do not put it in a control loop.
