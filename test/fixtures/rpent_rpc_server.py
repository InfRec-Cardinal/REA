"""Deterministic simulated responses served by RPent's real RPC transports."""

import argparse
import json
import os
from pathlib import Path
import sys


class RPentFixture:
    def __init__(self, robot="libero", seed=0, finish_after=3, arrays=False):
        self.robot = robot
        self.seed = seed
        self.finish_after = finish_after
        self.arrays = arrays
        self.steps = 0
        self.reset_count = 0
        self.calls = []
        self.closed = False
        self.meta = {"seed": seed}
        if robot == "libero":
            self.meta.update({"suite": "libero_spatial", "task": 0, "max_episode_steps": 20})
        elif robot == "robocasa":
            self.meta.update({"task_name": "OpenDrawer", "split": "target", "camera_h": 2, "camera_w": 1})
        else:
            self.meta.update({"runtime": "rlinf_robotwin_env", "task_name": "adjust_bottle", "task_config": "demo_clean", "action_layouts": ["qpos14", "eef16"]})

    def array(self, value):
        if self.arrays:
            import numpy as np
            return np.asarray(value)
        return value

    def image(self):
        values = [[[255, 0, 0]], [[0, 0, 255]]]
        if self.arrays:
            import numpy as np
            return np.asarray(values, dtype=np.uint8)
        return values

    def observation(self):
        return {"states": self.array([float(self.steps), 0, 0]), "main_images": self.image(), "agentview_depth": self.array([[1.0], [2.0]])}

    def info(self, executed=1):
        if self.robot == "robotwin":
            return {
                "executed_actions": executed,
                "episode_status": {"eval_success": self.steps >= self.finish_after, "take_action_cnt": self.steps, "step_lim": 20, "actual_seed": self.seed},
                "robot_state": {"qpos_target14": self.array([0.0] * 14), "arm_qpos_real12": self.array([0.0] * 12)},
            }
        return {"native_steps": self.steps}

    def call(self, method, args=(), kwargs=None, timeout_s=None):
        kwargs = kwargs or {}
        self.calls.append((method, args, kwargs))
        if method == "env.get_env_meta":
            return dict(self.meta)
        if method == "env.get_task_language":
            return f"Complete the {self.robot} fixture task"
        if method == "env.reset":
            if args or kwargs:
                raise ValueError("Official reset takes no task or seed arguments")
            self.reset_count += 1
            self.steps = 0
            obs = self.observation()
            return obs if self.robot == "robocasa" else (obs, self.info(0))
        if method == "env.check_success":
            return self.steps >= self.finish_after
        if method == "env.render_camera":
            if self.robot == "robotwin" and ("height" in kwargs or "width" in kwargs):
                raise ValueError("RoboTwin render_camera does not accept image dimensions")
            return self.image()
        if method in ("env.step", "env.chunk_step"):
            chunk = method == "env.chunk_step"
            if chunk and self.robot == "robocasa":
                raise NotImplementedError("RoboCasa does not implement chunk_step")
            commands = args[0] if chunk else [args[0]]
            dimension = 7 if self.robot == "libero" else 12 if self.robot == "robocasa" else 16 if kwargs.get("action_type") == "ee" else 14
            for command in commands:
                if len(command) != dimension:
                    raise ValueError(f"Expected {dimension} values")
            rewards, terminated, infos = [], [], []
            for command in commands:
                self.steps += 1
                done = self.steps >= self.finish_after
                rewards.append(float(done))
                terminated.append(done)
                infos.append(self.info())
                if done and self.robot != "libero":
                    break
            info = self.info(len(rewards))
            if self.robot == "robocasa":
                return self.observation(), rewards[-1], False, info
            if chunk:
                return self.observation(), self.array(rewards), self.array(terminated), self.array([False] * len(rewards)), infos if self.robot == "libero" else info
            return self.observation(), rewards[-1], terminated[-1], False, info
        if method == "shutdown":
            self.closed = True
            return {"ok": True}
        if method == "fixture.inspect":
            return {"reset_count": self.reset_count, "steps": self.steps, "closed": self.closed}
        raise ValueError(f"Unknown fixture RPC: {method}")

    def close(self):
        self.closed = True


def make_client(options):
    return RPentFixture(**options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["libero", "robocasa", "robotwin"], required=True)
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--rpent-root", default=os.environ.get("RPENT_ROOT"))
    args = parser.parse_args()
    if args.rpent_root:
        sys.path.insert(0, str(Path(args.rpent_root).resolve()))
    from rpent.utils.rpc.http_rpc import HttpRpcServer
    from rpent.utils.rpc.socket_rpc import SocketRpcServer
    server_type = HttpRpcServer if args.transport == "http" else SocketRpcServer
    fixture = RPentFixture(args.robot, arrays=True)
    server = server_type(("127.0.0.1", 0), lambda method, call_args, kwargs: fixture.call(method, call_args, kwargs))
    print(json.dumps({"endpoint": f"{args.transport}://127.0.0.1:{server.server_address[1]}", "robot": args.robot}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
