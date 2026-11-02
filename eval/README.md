# G1 service evaluation utilities

These utilities inspect service behavior through recorded frames, local runtime
events and saved episode logs. They do not include recorded experiment artifacts.
Replay and service requests use preview mode; they do not actuate a robot.

| Script | Purpose | Dependencies |
| --- | --- | --- |
| `replay_worker.py` | Serve recorded RGB frames through the sensing-worker protocol | Local image directory; Pillow for optional resizing |
| `prompt_latency.mjs` | Compare prompt variants and collect request timing | Configured model endpoint and a JPEG |
| `memory_ab.mjs` | Compare cold and accumulated memory in a local service | Running evaluation service |
| `analyse_episodes.mjs` | Summarize an existing episode store | Local episode JSONL |

## Recorded-frame service

Configure the model in `.env`, then start the service with a mock actuator and
frames you supply:

```bash
set -a; . ./.env; set +a
G1_ENGINE=token G1_PERSISTENT_CAPTURE=1 G1_ACTUATOR=mock \
G1_SENSE_WORKER=eval/replay_worker.py \
G1_REPLAY_FRAMES=/path/to/frames \
G1_REPLAY_DEPTH=0.5,2.2,0.6 \
G1_REPLAY_LIDAR_SNAPSHOT=/tmp/g1-replay/obstacles.json \
G1_LIDAR_SNAPSHOT=/tmp/g1-replay/obstacles.json \
G1_AGENT_HOST=127.0.0.1 node server.mjs
```

The worker serves frames in name order and loops. It supports `capture`,
`depth_only`, `no_depth` and `no_scene`. Depth can come from per-frame JSON
sidecars, a `--depth-json` map or a configured constant. A depth-only request
reuses the last served frame. The worker also maintains a simulated lidar snapshot.
Recorded frames do not respond to actions, so replay checks interfaces and
controller decisions rather than closed-loop task completion.

## Prompt timing

```bash
npm run eval:latency -- --frame /path/to/frame.jpg --trials 30
```

This interleaves prompt variants and records time to first byte and time to a
complete action. Keep model, frame and provider configuration fixed when comparing
variants.

## Memory comparison

```bash
npm run eval:memory -- --sessions 6 --steps 12 --instruction "reach the doorway" --yes
```

The script alternates cold and hot memory sessions using `execute:false`.
Cold sessions call the loopback-only `/internal/brain/reset` endpoint; `--keep-episodes`
retains the episode tier. Use a dedicated evaluation service with temporary
`G1_MEMORY_PATH`, `G1_ROUTER_MEMORY_PATH` and `G1_EPISODES_PATH`, because resetting
memory rotates and clears those stores. Run logs are written under `eval/out/`.

## Episode-store inspection

```bash
npm run eval:episodes -- runtime/memory/episodes.jsonl --task reach-doorway
```

The output groups recorded sessions by task and reports execution, routing,
review and timing fields. Task success requires an environment evaluator or
robot-side measurement; command acknowledgements alone do not provide it.

For simulator adapters, see [RoboTwin](../docs/benchmarks/robotwin.md),
[VLA/code routing](robotwin/hybrid/README.md) and [BEHAVIOR](behavior/README.md).
