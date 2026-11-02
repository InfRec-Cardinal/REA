# Testing

Everything below runs offline: simulators, robots, VLA servers and model APIs are replaced by fakes.

```bash
npm test                                                   # Node tests (framework, adapters, G1 service)
python3 -m unittest discover -s test                       # Python backends, drivers, clients, runners
(cd agent-vla-async && PYTHONPATH=src python3 -m unittest discover -s tests)   # G1 cerebellum
npm run test:async                                         # async and routed runtimes only
npm run test:aloha                                         # ALOHA, pi0.5 and LingBot-VLA clients
```

Some Python tests skip themselves when optional packages are missing (NumPy, Pillow, MessagePack, websockets for the VLA
clients: `pip install -r requirements-pi05.txt -r requirements-lingbot-vla.txt`; `openai-agents` for the BEHAVIOR Agents
SDK child, selected with `BEHAVIOR_SDK_PYTHON`).

Demonstrations that exercise the full CLI path:

```bash
npm run demo                  # interface demo
npm run demo:framework        # evaluation fixtures through `rea run`
npm run demo:robots           # single-arm and bimanual action chunks through the robot contract
npm run demo:routed           # routed episode
npm run demo:async            # asynchronous systems
npm run demo:offline          # G1 agent with a fake provider
npm run demo:service          # the G1 HTTP service on fixtures
```

Against real backends, start with `npm run rea -- doctor <config>`: it checks configuration, interpreters, Python modules
and source directories without connecting to a robot, simulator or model. Then run one short episode with
`--steps 5 --output outputs/<name>` and read `NNNN.jsonl` before scaling up. Integration notes for each upstream system
are in its guide under [benchmarks](benchmarks/) and [robots](robots/).
