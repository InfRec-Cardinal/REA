# BEHAVIOR-1K integration

REA runs BEHAVIOR-1K's R1Pro navigate-and-pick-up tasks (`trash`, `radio`, `teddy`, `drill`, `camera`) on ASPIRE's
R1Pro code-execution environment over OmniGibson. Install ASPIRE with BEHAVIOR-1K support and OmniGibson separately; set
`ASPIRE_SIM_ROOT` (the runner) or `ASPIRE_ROOT` / `ASPIRE_PYTHON_ROOT` (the program backend) to the checkout. The
[BEHAVIOR guide](../../docs/benchmarks/behavior.md) covers the offline dataset check and the R1Pro action interface.

| Directory | Contents |
|---|---|
| `runner/` | `run_episode.py`: one episode inside ASPIRE's environment, with frozen programs (single or routed) or an online agent; `agents.py`: the online agents; `behavior_tasks.py`: the task table |
| `online/` | REA's online agent: the framework episode runtime (`runtime.mjs`), the per-decision model worker (`worker.py`), its prompt (`prompting.py`), routing rules (`behavior_routing.mjs`) and behavioural configuration (`config.example.json`) |
| `tasks/` | instance splits per task: development instances 26-35, held-out instances 1-25 (`*_heldout10.json`: 1-10), smoke instances |
| `prompts/` | task prompts, supplied by you (see its README) |

`scripts/behavior_backend.py` exposes the same environment as a [program backend](../../docs/extending/program-backends.md)
for other runners.

## Running an episode

Inside ASPIRE's Python environment:

```bash
# frozen programs: run the first; recover with the next ones while the visual judge says not done
python eval/behavior/runner/run_episode.py --task radio --seed 26 --arm routed-judge \
  --mode routed-judge --programs programs/radio/a.py,programs/radio/b.py --max-recoveries 2 \
  --judge-model your-vision-model --out outputs/behavior

# an online agent choosing one R1Pro skill per decision through REA's runtime
python eval/behavior/runner/run_episode.py --task radio --seed 26 --arm rea-online \
  --mode agents-sdk --agent-style rea-online --agent-model your-model --agent-api responses \
  --agent-python /path/to/agents-venv/bin/python --out outputs/behavior
```

Program modes: `single` (one program), `routed-exception` (the next program only after a program exception),
`routed-judge` (the next program while the visual judge says not done, up to `--max-recoveries`) and `routed-inhand`
(recover while neither gripper holds an object). All program runs of an episode share one budget (`--program-budget`,
default 1,500 s, measured from the first program statement); success is ASPIRE's `task_completed()`, read once after the
last run and never on the policy path. `--recovery-guard inhand` prepends a guard to recovery programs so they never undo a
grasp an earlier program achieved. Run `run_episode.py --help` for the judge views, prompts and video recording options.

Credentials: the online agents use `AGENT_API_KEY` / `AGENT_BASE_URL` when set, else `REA_API_KEY` / `REA_BASE_URL`; the
judge uses `REA_API_KEY` / `REA_BASE_URL`. Credentials are passed to the agent process through its environment only.

## The online agents

Both agent styles run through the same runner, reset, budget clock, `ToolServer` (which executes tools on the
budget-wrapped R1Pro API) and score:

| | `--agent-style sdk` | `--agent-style rea-online` |
|---|---|---|
| Loop | one OpenAI Agents SDK `Runner.run`, at most 60 model turns over all R1Pro tools | REA's episode runtime: at most 40 decisions, one skill per decision |
| Requests | one per turn | at most 4 model turns per decision (up to 3 read-only helper calls, then `take_action`) |
| Images | the head camera when the agent starts | the head camera at every decision |
| Clock | appended to every tool result (Responses API) | the decision index and remaining budget in every decision message |
| End | the agent stops calling tools, or the budget ends | `finish(arm)` accepted on an in-hand check, 40 decisions, or the budget ends |

`take_action(skill, arguments, notes, thought)` takes one of `find_object_base_rotate`, `find_object_torso_rotate`,
`navigate_to_pose`, `grasp_object`, `lift_arm`, `open_gripper`, `close_gripper`, `reset_torso` and `finish`; the helpers are
`get_object_pose`, `get_robot_position`, `compute_approach_pose`, `sample_grasp_pose`, `get_current_eef_pose`,
`get_sam3_mask`, `detect_objects` and `check_object_in_hand`. Both styles describe the API with the same `TOOL_SCHEMAS`
(`runner/agents.py`).

`finish(arm)` runs one `check_object_in_hand(arm)` on the named arm: true ends the episode, false refuses it and the
episode continues. The other arm is never probed, because `check_object_in_hand` switches the controller's active arm and
`task_completed()` reads the active arm.

How a `rea-online` decision runs: `agents.run_episode` starts `online/runtime.mjs` with Node and serves it over a JSON-lines
channel (`call` / `result` / `observe` / `done` / `log`). The runtime runs the framework's `runEpisode` with
`createRoutedPolicy`; the environment's `step` executes the chosen skill through the runner. Per decision it starts
`online/worker.py` with the agent interpreter, which builds the model input with `online/prompting.py` and makes one Agents
SDK run that stops as soon as `take_action` is accepted. Routing (`online/behavior_routing.mjs`) sends three failed
actions, a repeated action or a refused finish to the deliberate prompt with a reason.

`online/config.example.json` holds every behavioural key (decisions, history, notes, routing patience and deliberate
fraction, helper and turn limits, request timeout and retries, the reflex and deliberate instructions; `helpers: false`
removes the helper tools). The runner adds the model, effort, output tokens and decision limit from its flags and writes
the result to `<episode>/online/config.json`.

## Records

Per episode (`<out>/episodes/behavior/<task>/seed_<N>/`): `result.json` (outcome, runs, agent metadata and the sha256 of
the code that defines the agent), `runs/<i>.json`, initial and final head frames, and for `rea-online`
`online-events.jsonl`, `online-decisions.jsonl`, `online/decision-<k>/` (context, model input with the image as a path,
decision, record) and `online/observations/obs-<k>.jpg`. A rerun of an unscored episode keeps the previous attempt under
`online.prev-<k>/`.

## Provider failures

A provider that keeps failing after the client's retries (connection errors, timeouts including HTTP 408, rate limits,
authentication or permission errors, 5xx, an empty or non-JSON body, insufficient balance or quota) is infrastructure, not
a task outcome: the episode ends with `stop_reason: provider_error`, `success: null` and `unscored: true`, so it can be
rerun from scratch. A runner that hangs past the budget until the last-resort watchdog is infrastructure too.

## Tests

```bash
node --test test/behavior-routing.test.mjs
python3 -m unittest discover -s test -p 'test_behavior_*.py'
```

`test_behavior_online_runtime.py` runs the `rea-online` agent end to end against fake R1Pro functions, with
`test/fixtures/behavior_online_fake_worker.py` in place of the model. `test_behavior_agents_child.py` runs the real Agents
SDK child against a fake Responses relay; it needs an interpreter with `openai-agents` in `BEHAVIOR_SDK_PYTHON`.
