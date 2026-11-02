"""Robot-local CLI for the cerebellum Unix socket."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

from .rpc import call


async def stream(socket_path: str) -> int:
    """Proxy newline-delimited requests over one long-lived stdin/stdout pipe."""
    loop = asyncio.get_running_loop()
    while True:
        raw = await loop.run_in_executor(None, sys.stdin.readline)
        if raw == "":
            return 0
        try:
            request: Dict[str, Any] = json.loads(raw)
            response = await call(socket_path, request)
        except Exception as exc:
            response = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        json.dump(response, sys.stdout, ensure_ascii=False, separators=(",", ":"))
        sys.stdout.write("\n")
        sys.stdout.flush()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default="/run/user/1000/g1-vla-async/control.sock")
    parser.add_argument("--request", help="JSON request; otherwise read JSON from stdin")
    parser.add_argument("--stream", action="store_true", help="serve JSONL requests until stdin closes")
    args = parser.parse_args(argv)
    if args.stream:
        return asyncio.run(stream(args.socket))
    raw = args.request if args.request is not None else sys.stdin.read()
    request: Dict[str, Any] = json.loads(raw)
    response = asyncio.run(call(args.socket, request))
    json.dump(response, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
