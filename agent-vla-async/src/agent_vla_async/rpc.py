"""JSON-lines RPC over a robot-local Unix socket."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict


Handler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


async def call(socket_path: str, request: Dict[str, Any]) -> Dict[str, Any]:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    try:
        writer.write(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")
        await writer.drain()
        line = await reader.readline()
        if not line:
            raise RuntimeError("cerebellum closed the RPC connection")
        return json.loads(line.decode("utf-8"))
    finally:
        writer.close()
        await writer.wait_closed()


async def serve(socket_path: str, handler: Handler) -> asyncio.AbstractServer:
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError("another cerebellum daemon owns %s" % socket_path) from exc
    if path.exists():
        path.unlink()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            request = json.loads(line.decode("utf-8"))
            response = await handler(request)
        except Exception as exc:
            response = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        writer.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(handle, path=socket_path)
    # Keep the advisory lock alive for exactly the server lifetime.
    setattr(server, "_g1_lock_file", lock_file)
    os.chmod(socket_path, 0o600)
    return server
