#!/usr/bin/env python3
"""Build an explicit high-level G1 plan without deriving motion from prompt text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_vla_async.protocol import ExecutionPlan


def _pulse(value: str) -> Dict[str, float]:
    try:
        vx, yaw, duration = (float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pulse must be VX,YAW,DURATION_S") from exc
    return {"vx": vx, "yaw": yaw, "duration_s": duration}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--pulse",
        action="append",
        type=_pulse,
        default=[],
        help="repeatable bounded SPORT pulse: VX,YAW,DURATION_S",
    )
    parser.add_argument(
        "--arm-action",
        action="append",
        default=[],
        help="repeatable named high-level arm action",
    )
    parser.add_argument("--target", default="")
    parser.add_argument("--output")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    value: Dict[str, Any] = {
        "source": "agent_plan",
        "task": args.task,
        "navigation": (
            {"target": args.target, "pulses": args.pulse}
            if args.pulse
            else None
        ),
        "arm_steps": [{"action": name} for name in args.arm_action],
    }
    plan = ExecutionPlan.from_dict(value)
    rendered = json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
