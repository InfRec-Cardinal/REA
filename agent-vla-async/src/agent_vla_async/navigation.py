"""Adapter over the existing g1-nav-agent sensing and bounded sport control."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from .protocol import MotionPulse


class G1NavigationAdapter:
    def __init__(self, execute_enabled: bool = False, no_lidar: bool = False):
        self.execute_enabled = bool(execute_enabled)
        self.no_lidar = bool(no_lidar)
        self._nav: Any = None
        self._camera: Any = None

    async def __aenter__(self) -> "G1NavigationAdapter":
        from g1nav.config import LidarCfg, NavCfg
        from g1nav.session import NavSession
        from g1_smolvla.realsense_client import UnifiedRealSenseClient

        base = NavCfg()
        cfg = NavCfg(
            lidar=LidarCfg(required=not self.no_lidar),
            depth=base.depth,
            motion=base.motion,
        )
        self._nav = NavSession(cfg, execute=self.execute_enabled)
        await self._nav.__aenter__()
        self._camera = UnifiedRealSenseClient(
            socket_path=cfg.depth.socket,
            timeout_s=cfg.depth.timeout_s,
        )
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._camera is not None:
            self._camera.close()
        if self._nav is not None:
            await self._nav.__aexit__(exc_type, exc, traceback)

    def observe(self) -> Dict[str, Any]:
        return self._nav.observe().to_dict()

    async def capture_rgb(self) -> Any:
        loop = asyncio.get_running_loop()
        color, _ = await loop.run_in_executor(None, self._camera.read)
        return color

    async def step(self, pulse: MotionPulse) -> Dict[str, Any]:
        return await self._nav.step(pulse.vx, pulse.yaw, pulse.duration_s)

    async def stop(self) -> Dict[str, Any]:
        return await self._nav.stop()


class FakeNavigationAdapter:
    """Deterministic adapter used by the integration tests."""

    def __init__(self) -> None:
        self.steps = []
        self.stops = 0

    async def __aenter__(self) -> "FakeNavigationAdapter":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def observe(self) -> Dict[str, Any]:
        return {
            "state": {"usable": True, "ready": True, "reason": "fake_ready"},
            "depth": {"usable": True, "blocked": False, "reason": "fake_clear"},
            "lidar": {"usable": True, "blocked": False, "reason": "fake_clear"},
            "verdict": {"allowed": True, "reasons": []},
            "source": "fake",
        }

    async def capture_rgb(self) -> Any:
        return None

    async def step(self, pulse: MotionPulse) -> Dict[str, Any]:
        self.steps.append(pulse)
        return {
            "ok": True,
            "moved": True,
            "refused": False,
            "controller": {
                "ok": True,
                "move_rc": 0,
                "stop_rc": 0,
                "postflight": {"ready": True},
            },
        }

    async def stop(self) -> Dict[str, Any]:
        self.stops += 1
        return {"ok": True}
