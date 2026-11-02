# Program backends

A program backend runs frozen robot programs (code-as-policy) inside a simulator or on a robot and answers over JSON
lines on stdin/stdout. `scripts/aspire_backend.py` (LIBERO / LIBERO-Pro through ASPIRE's code-execution environment),
`scripts/behavior_backend.py` (BEHAVIOR-1K R1Pro) and `scripts/async_libero_backend.py` (resumable LIBERO programs)
implement it; `src/program-client.mjs` is the Node client.

## Protocol

Every request is `{"id": <int>, "command": <name>, ...}` and every reply `{"id", "ok": true, ...}` or
`{"id", "ok": false, "error": "..."}`. Program output and simulator logs go to stderr, never to the protocol stream.

| Command | Request fields | Reply fields |
|---|---|---|
| `catalog` | | suites or tasks, seed ranges, backend settings |
| `reset` | task identity (for LIBERO `suite`, `task`), `seed`, `horizon`, `reset`, `phase`, `wall_budget_s` | the first `observe` reply plus initial-state provenance |
| `run_program` | `code` or `path`, `which` (a label), `allowed_sha256`, `max_steps`, `wall_timeout_s` | `exception`, `exception_class`, `truncated`, `truncated_reason`, `steps`, `wall_s`, `stdout_tail`, `stderr_tail` |
| `program_chunk` | `candidateId`, `controlSteps`, `wall_timeout_s` (resumable backends) | `programDone`, `exception`, `truncated`, `steps`, `boundary` |
| `stop_program` | (resumable backends) | `stopped` |
| `observe` | `view`, `judge_frame` | `frame` (`imageB64`, `mimeType`, `capturedAt`), `task.language`, `proprio`, `steps`, `stepsLeft` |
| `score` | | the task predicate (`task_completed`) and step count; **seals the episode** |
| `close` | | releases the simulator |

## Invariants

- Nothing on the program, judge or routing path reads the task predicate, reward or object poses. `score` is the only
  success read, it is called once per episode, and after it the episode accepts no more program commands.
- Programs execute through the code environment's own execution entry point, never through a step function that
  computes reward or success.
- One shared wall and step budget covers every program run of an episode: the caller passes the remaining budget as
  `wall_timeout_s` and the backend refuses to exceed the horizon.
- `allowed_sha256` pins the exact program bytes, so an evaluation can prove it ran the frozen programs it planned.
- Exceptions are classified (`program`, `timeout`, `infra`) so infrastructure failures can be retried without counting as
  task outcomes.

## Frozen bundles

`tools/build_program_bundle.py --library <dir> --output <bundle>` copies a library laid out as
`<suite>/<task>/<candidate>.py` into a bundle with `bundle.json` (`rea.program-bundle/v1`, per-candidate paths and
sha256). `scripts/libero_program_backend.py` and `scripts/async_libero_backend.py` read it from `REA_PROGRAM_BUNDLE` and
accept only its programs.

## Adding a backend for another simulator or robot

1. Expose the robot API your programs call (perception, motion primitives, grasping) as plain Python functions in the
   program's globals; keep simulator internals out of reach.
2. Implement `reset`, `run_program` (or `program_chunk` for resumable runs), `observe`, `score` and `close` with the
   invariants above; start from `scripts/aspire_backend.py`.
3. Wrap it as an environment: either one action `run_program` per step (a program per decision), or the resumable form of
   `src/environments/libero-programs.mjs` (`continue_program` with a candidate id, `info.capability`, `score()`).
4. Add tests with a fake environment injected through the backend's `env_factory`, as `test/test_aspire_backend.py`
   does, so the protocol is tested without the simulator.
