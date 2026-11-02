"""A one-slot async mailbox: producers overwrite stale work instead of queuing it."""

from __future__ import annotations

import asyncio
from typing import Generic, Optional, Tuple, TypeVar


T = TypeVar("T")


class LatestValue(Generic[T]):
    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._version = 0
        self._value: Optional[T] = None

    async def publish(self, value: T) -> int:
        async with self._condition:
            self._version += 1
            self._value = value
            self._condition.notify_all()
            return self._version

    async def clear(self) -> int:
        async with self._condition:
            self._version += 1
            self._value = None
            self._condition.notify_all()
            return self._version

    async def wait_after(self, version: int) -> Tuple[int, Optional[T]]:
        async with self._condition:
            await self._condition.wait_for(lambda: self._version > version)
            return self._version, self._value

    def snapshot(self) -> Tuple[int, Optional[T]]:
        return self._version, self._value

