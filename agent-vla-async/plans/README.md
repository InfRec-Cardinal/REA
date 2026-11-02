# Hybrid plan files

A plan is an explicit Agent-authored sequence. Prompt text is sent to GR00T
for advisory vision inference, but it is never converted into velocity or arm
commands.

```json
{
  "source": "agent_plan",
  "task": "approach an already identified door and push it",
  "navigation": {
    "target": "door",
    "pulses": [
      {"vx": 0.10, "yaw": 0.0, "duration_s": 0.30}
    ]
  },
  "arm_steps": [
    {"action": "forward_push"},
    {"action": "release_arm"}
  ]
}
```

Each SPORT pulse is limited to `|vx| <= 0.20 m/s`, `|yaw| <= 0.35 rad/s`,
`duration_s <= 0.60`, and implicit `vy = 0`. The robot-side navigation gate
checks state, depth and configured lidar immediately before every pulse.

`door-push-at-standoff.json` contains no locomotion. It is only the manipulation
half of the workflow, for use after an Agent has separately confirmed position
and alignment. Preview it before enabling the execute-mode service.
