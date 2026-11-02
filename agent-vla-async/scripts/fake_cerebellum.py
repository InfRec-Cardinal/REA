#!/usr/bin/env python3
"""Run the cerebellum runtime on a Unix socket with the deterministic fake adapters.

For bridge tests only: FakeNavigationAdapter records pulses instead of moving a robot and MockPolicy
stands in for GR00T. Prints one JSON line when the socket is ready, serves until stdin closes, then
prints one JSON line with every pulse it was asked to run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import List, Optional

from agent_vla_async.navigation import FakeNavigationAdapter
from agent_vla_async.policy import MockPolicy
from agent_vla_async.runtime import CerebellumRuntime


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--policy-delay", type=float, default=0.02)
    args = parser.parse_args(argv)

    navigation = FakeNavigationAdapter()
    runtime = CerebellumRuntime(
        navigation=navigation,
        policy=MockPolicy(delay_s=args.policy_delay),
        socket_path=args.socket,
        execute_enabled=args.execute,
    )

    async def run() -> None:
        task = asyncio.create_task(runtime.run())
        while not any(event.kind == "READY" for event in runtime.events):
            await asyncio.sleep(0.01)
        print(json.dumps({"event": "ready", "socket": args.socket, "execute": args.execute}), flush=True)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, sys.stdin.read)
        await runtime.close()
        await task
        print(
            json.dumps(
                {
                    "event": "exit",
                    "steps": [pulse.to_dict() for pulse in navigation.steps],
                    "stops": navigation.stops,
                    "status": runtime.status(),
                }
            ),
            flush=True,
        )

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
