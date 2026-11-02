# Contributing

Thanks for helping improve REA. Bug reports, new adapters (benchmarks, robots, VLA servers) and documentation fixes are
all welcome.

## Before you open a pull request

1. Run the checks that CI runs:

   ```bash
   npm test
   python3 -m unittest discover -s test
   (cd agent-vla-async && PYTHONPATH=src python3 -m unittest discover -s tests)
   npm run demo && npm run demo:routed && npm run demo:framework
   ```

2. Add tests that run offline. Simulators, robots and model APIs are replaced by fakes in this repository's tests
   (see `test/fixtures/` and the `env_factory` injection in the Python backends), so CI needs no GPU, hardware or keys.
3. Keep extensions out of the core: a new environment, policy, router or judge should be a module or a registry entry;
   change `src/framework/` only for behaviour every adapter needs.
4. Document the integration in `docs/` (setup, what it expects from upstream, what is not included) and link it from
   the README table.

## Conventions

- Node.js 22.19+ ES modules; Python 3.10+ for backends (robot-side workers keep Python 3.8 compatibility).
- No secrets, private hosts, personal paths or recorded run artifacts in the repository; use placeholders and
  environment variables (`REA_*`, `<ADAPTER>_PYTHON`, `<ADAPTER>_ROOT`). CI rejects private addresses and paths.
- Hardware adapters start in preview (`execute: false`) and implement `stop()`.
- Success is reported only by an environment's evaluator, never inferred from a command acknowledgement.
- Benchmark protocols: build route tables and tune judges on development episodes; evaluate on different episodes.

## Third-party code and assets

Do not add code, prompts, documentation or assets copied from other projects unless their license allows it; record the
source and license in [NOTICE.md](NOTICE.md).
