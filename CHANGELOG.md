# Changelog

## 0.4.0: first public release

- **Routed systems.** `runRoutedEpisode` and `systems.mode: "routed"`: route memory (`route-memory`, `buildRouteTable`,
  `tools/route-table.mjs`) picks the first capability from development evidence, a boundary judge gate (`judge`,
  `createVlmJudge`) checks completion only when a capability returns, an escalation ladder decides what runs next under
  one shared budget, and success is read once from `environment.score()`. Offline demo: `npm run demo:routed`.
- **Asynchronous systems.** System 1 routing at every action boundary while System 2 thinks in the background; live
  events, sessions, a console and a sync comparison mode.
- **Extension points.** `python-robot` environments for your own Python driver class (template in
  `examples/robots/template_driver.py`); capability boundaries (`resumable`, `info.capability`) for environments that
  host frozen programs; `rea list` lists environments, policies and system roles.
- **Benchmarks and robots.** LIBERO / LIBERO-Pro (token control and resumable frozen programs), BEHAVIOR-1K program
  backend and online agent runner, RoboTwin VLA / code candidates with a success router, ARC-AGI-3, Gymnasium; ALOHA,
  LeRobot, OpenRAL, RPent and Unitree G1 (service, safety gate, cerebellum).
- **Safety.** Code-policy workers start without the parent's credentials; the G1 service binds to loopback by default.
- **Configuration.** Provider variables are `REA_MODEL`, `REA_BASE_URL`, `REA_API_KEY`, `REA_ENV_FILE` and
  `REA_POLICY_MODE`; the pre-release `FAST_*` names are read with a deprecation notice. The CLI is `rea`
  (`npm run rea --`).
