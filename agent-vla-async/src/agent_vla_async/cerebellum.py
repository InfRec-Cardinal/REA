"""Robot-side process entry point."""

from __future__ import annotations

import argparse
import asyncio
from typing import List, Optional

from .arm import HighLevelArmActionBackend
from .controller import SimulatedSonicSupervisor, UnavailableSonicSupervisor
from .navigation import G1NavigationAdapter
from .policy import GrootPolicy, MockPolicy, NavigationCalibration, SonicPolicyClient
from .runtime import CerebellumRuntime


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default="/run/user/1000/g1-vla-async/control.sock")
    parser.add_argument("--policy", choices=("groot", "mock"), default="groot")
    parser.add_argument("--groot-host", default="127.0.0.1")
    parser.add_argument("--groot-port", type=int, default=5550)
    parser.add_argument("--groot-timeout-ms", type=int, default=15000)
    parser.add_argument("--sonic-policy-host", default="127.0.0.1")
    parser.add_argument("--sonic-policy-port", type=int, default=5551)
    parser.add_argument(
        "--navigation-calibration",
        help="diagnostic scaling only; REAL_G1 output is never routed to SPORT",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--no-arm-actions", action="store_true")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument(
        "--sonic-supervisor",
        choices=("unavailable", "simulation"),
        default="unavailable",
        help="software control-plane supervisor; simulation never actuates the robot",
    )
    args = parser.parse_args(argv)

    calibration = (
        NavigationCalibration.load(args.navigation_calibration)
        if args.navigation_calibration
        else None
    )
    policy = (
        MockPolicy()
        if args.policy == "mock"
        else GrootPolicy(
            host=args.groot_host,
            port=args.groot_port,
            timeout_ms=args.groot_timeout_ms,
            calibration=calibration,
        )
    )
    navigation = G1NavigationAdapter(
        execute_enabled=args.execute,
        no_lidar=args.no_lidar,
    )
    arm = None if args.no_arm_actions else HighLevelArmActionBackend(
        execute_enabled=args.execute,
        network_interface=args.iface,
    )
    sonic_supervisor = (
        SimulatedSonicSupervisor()
        if args.sonic_supervisor == "simulation"
        else UnavailableSonicSupervisor()
    )
    sonic_policy = SonicPolicyClient(
        host=args.sonic_policy_host,
        port=args.sonic_policy_port,
        timeout_ms=args.groot_timeout_ms,
    )
    runtime = CerebellumRuntime(
        navigation=navigation,
        policy=policy,
        socket_path=args.socket,
        execute_enabled=args.execute,
        arm=arm,
        policy_wait_timeout_s=args.groot_timeout_ms / 1000.0,
        sonic_supervisor=sonic_supervisor,
        sonic_policy=sonic_policy,
    )
    try:
        asyncio.run(runtime.run())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
