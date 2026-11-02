# Changelog

REA is updated continuously; each release lists the platforms, robots and interfaces it adds.

## 0.5.0

- **Platforms that run their own loop.** `rea serve <config>` runs any REA configuration (one policy, async or routed
  systems) as a policy for an external evaluator over JSON lines (`configure`, `reset`, `step`, `end`, `close`), on a
  bridge environment (`createBridge`, `servePolicy`). Standard-library Python client: `integrations/python/rea_client.py`.
- **RoboDojo.** `integrations/xpolicylab/REA`: REA as an XPolicyLab policy, for RoboDojo simulation and RealEval (ARX X5,
  Piper, Piper X) and the other XPolicyLab environments. New: tested offline; validation on RoboDojo is in progress.
- Example served configurations: `examples/xpolicylab/` (chat agent, LingBot-VLA client).
- README: project page, paper and Hugging Face links, a platform table with status, and a citation.

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
