# Platforms that drive the policy: `rea serve`

REA connects to a benchmark or robot in one of two directions:

| The loop is owned by | Integrate as | Examples |
|---|---|---|
| REA | an [environment](environments.md) or a [robot driver](robots.md): REA calls `reset` / `step` | LIBERO, BEHAVIOR-1K, ARC-AGI-3, Gymnasium, ALOHA, LeRobot, OpenRAL, the G1 |
| the platform | a **policy server**: the platform sends observations, REA answers with actions | [RoboDojo](../benchmarks/robodojo.md) and other XPolicyLab platforms, challenge evaluators, robot deployment stacks |

`rea serve <config>` handles the second case. The configured policy, or the async or routed System 0/1/2 runtime, runs
unchanged on a bridge environment whose observations come from the platform. Every action the runtime dispatches goes
back to the platform.

```bash
npm run rea -- serve examples/xpolicylab/chat.config.json --output outputs/served
```

A served configuration names `"environment": { "type": "bridge" }`. JavaScript configurations with policy or systems
factories work too; their `createEnvironment` is replaced by the bridge. Unless the configuration sets a budget, serving
uses `maxSteps: 100000` and `maxDurationMs: 24 h`, because the platform decides when an episode ends.

## Protocol

JSON lines on stdin and stdout. Each request is `{"id": <int>, "op": <name>, ...}`; each reply is
`{"id", "ok": true, "result": ...}` or `{"id", "ok": false, "error": "..."}`. Requests are answered in order, and stdout
carries nothing but replies; REA's logs go to stderr.

| op | Request fields | Result |
|---|---|---|
| `configure` | `actions` (JSON-Schema action specifications), `timing`, `name`, `resumable` | the bridge settings and the served `mode` |
| `reset` | `task`, `seed`, `instruction` | `{episode}`; ends the previous episode first |
| `step` | `observation` (`{text, state, images: [{name, mimeType, data}]}`), optionally `terminated`, `truncated`, `success`, `reward`, `info` | `{action: {name, arguments}}`, or `{done: true, reason, summary}` once REA has ended the episode |
| `end` | `result`, e.g. `{"success": true}` | the episode summary; `result.success` becomes the episode's outcome (the routed runtime reads it as its sealed score) |
| `close` | | ends the episode, closes the agent and stops serving |

`step` delivers the observation that results from the previous action and waits for the next action, so a platform's
loop is: `reset`, then `step` with each new observation, then `end`. When REA ends an episode first (its own budget, a
routed System 1 stopping), every further `step` returns `done`; the caller decides what the robot does then. The
XPolicyLab adapter holds the measured state; never send a zero command to hardware.

With `--output DIR`, each episode writes `NNNN.jsonl` (transitions, images omitted) and `NNNN.json` (summary and the
platform's result).

## Python client

[`integrations/python/rea_client.py`](../../integrations/python/rea_client.py) needs only the standard library:

```python
import sys; sys.path.insert(0, "/path/to/REA/integrations/python")
from rea_client import ReaSession, encode_image

with ReaSession("examples/xpolicylab/chat.config.json", rea_root="/path/to/REA") as rea:
    rea.configure(actions=ACTION_SPECS)
    rea.reset(task="stack_bowls", seed=0, instruction=instruction)
    while not platform.episode_over():
        reply = rea.step({"text": instruction, "state": platform.state(),
                          "images": [encode_image(rgb, "head_camera")]})
        if reply.get("done"):
            platform.hold()
        else:
            platform.execute(reply["action"])
    rea.end({"success": platform.success()})
```

`encode_image` turns an RGB array into a JPEG with Pillow, or into a PNG without dependencies. Each `ReaSession` is one
`rea serve` process; batched evaluation uses one session per environment index.

## Writing an adapter for a new platform

1. Describe the platform's actions as JSON-Schema action specifications and send them with `configure`. Units, frames and
   joint order belong in the descriptions. Offer a flat-vector form as well if VLA clients will serve it (as the
   XPolicyLab adapter's `packed_action_chunk` does).
2. Convert each platform observation to `{text, state, images}` and send it with `step`; convert the returned action back.
3. Call `end` with the platform's success when it reports one, and `reset` for every new episode.
4. Test the adapter offline against `rea serve` with a scripted policy, as `test/test_xpolicylab_adapter.py` does.

The [XPolicyLab adapter](../../integrations/xpolicylab/REA/README.md) is the reference implementation.
