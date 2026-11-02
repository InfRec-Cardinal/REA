"""REA as an XPolicyLab policy: RoboDojo simulation and RealEval, RoboTwin and the other XPolicyLab environments.

Copy this directory to XPolicyLab/policy/REA and run install.sh (see README.md). XPolicyLab's policy server builds
Model(deploy_cfg) and calls update_obs / get_action / reset on it. Every call is forwarded to a `rea serve` process that
runs the REA configuration named by `rea_config` in deploy.yml, so any REA policy (an LLM agent, a VLA client, a code
policy) or System 0/1/2 runtime runs on the benchmark without changes.

Observations reach REA as {text: instruction, state: {<XPolicyLab state keys>, packed, robot}, images: [one per colour
camera]}. REA answers with one of two actions: `action_chunk` (commands keyed like XPolicyLab actions) or
`packed_action_chunk` (each command one vector in XPolicyLab's packing order). When REA ends an episode before the
benchmark does, the adapter holds the measured state; it never sends a zero command.
"""

import os
import sys
from pathlib import Path

import numpy as np

try:
    from XPolicyLab.model_template import ModelTemplate
except ImportError:  # outside an XPolicyLab checkout (tests)
    class ModelTemplate:
        def __init__(self):
            self.model = None


def robot_layout(action_type, dim_info):
    """XPolicyLab's keys for one robot, as XPolicyLab.utils.process_data and the demo adapter define them.

    Returns [(arm action key, arm state key, arm dim, gripper key, gripper dim), ...] in packing order.
    """
    if action_type not in ("joint", "ee"):
        raise ValueError(f"action_type must be joint or ee, got {action_type!r}")
    arms, grippers = list(dim_info["arm_dim"]), list(dim_info["ee_dim"])
    if len(arms) != len(grippers):
        raise ValueError("arm_dim and ee_dim must list the same number of arms")
    if len(arms) == 1:
        arm_actions = ["arm_joint_state"] if action_type == "joint" else ["ee_pose"]
        arm_states = ["joint_state"] if action_type == "joint" else ["ee_pose"]
        gripper_keys = ["ee_joint_state"]
    elif len(arms) == 2:
        arm_actions = ["left_arm_joint_state", "right_arm_joint_state"] if action_type == "joint" else ["left_ee_pose", "right_ee_pose"]
        arm_states = arm_actions
        gripper_keys = ["left_ee_joint_state", "right_ee_joint_state"]
    else:
        raise NotImplementedError(f"unsupported number of arms: {len(arms)}")
    pose = action_type == "ee"
    return [(arm_actions[i], arm_states[i], 7 if pose else int(arms[i]), gripper_keys[i], int(grippers[i])) for i in range(len(arms))]


def action_specs(action_type, layout, max_chunk):
    """The two actions REA may return, as JSON-Schema action specifications."""
    vector = lambda size: {"type": "array", "items": {"type": "number"}, "minItems": size, "maxItems": size}  # noqa: E731
    fields = [(arm_key, arm_dim) for arm_key, _, arm_dim, _, _ in layout] + [(key, dim) for _, _, _, key, dim in layout]
    ordered = []
    for arm_key, _, arm_dim, gripper_key, gripper_dim in layout:
        ordered += [(arm_key, arm_dim), (gripper_key, gripper_dim)]
    packed = sum(size for _, size in ordered)
    arm_units = "absolute joint positions in radians" if action_type == "joint" else "end-effector poses [x, y, z, qw, qx, qy, qz]"
    detail = (f"Arm fields are {arm_units}; *ee_joint_state fields are gripper commands. Units and frames follow the benchmark's "
              "robot configuration (the same values as the matching observation state keys).")
    command = {"type": "object", "properties": {key: vector(size) for key, size in fields},
               "required": [key for key, _ in fields], "additionalProperties": False}
    return [
        {"name": "action_chunk", "description": f"Execute 1-{max_chunk} commands in order, one per control step. {detail}",
         "parameters": {"type": "object", "properties": {"commands": {"type": "array", "items": command, "minItems": 1, "maxItems": max_chunk}},
                        "required": ["commands"], "additionalProperties": False}},
        {"name": "packed_action_chunk",
         "description": f"The same as action_chunk, each command one {packed}-value vector in the order {', '.join(key for key, _ in ordered)}.",
         "parameters": {"type": "object", "properties": {"commands": {"type": "array", "items": vector(packed), "minItems": 1, "maxItems": max_chunk}},
                        "required": ["commands"], "additionalProperties": False}},
    ]


def _plain(value):
    return value.tolist() if hasattr(value, "tolist") else value


def packed_state(state, layout):
    """The measured state in XPolicyLab packing order [arm_0, gripper_0, arm_1, gripper_1], or None when a key is missing."""
    parts = []
    for _, arm_state, _, gripper_key, _ in layout:
        if arm_state not in state or gripper_key not in state:
            return None
        parts += [np.asarray(state[arm_state], dtype=np.float32).reshape(-1), np.asarray(state[gripper_key], dtype=np.float32).reshape(-1)]
    return np.concatenate(parts).tolist()


def to_rea_observation(obs, layout, action_type, encode_image, quality=90, cameras=None):
    images = []
    for name, camera in (obs.get("vision") or {}).items():
        if cameras and name not in cameras:
            continue
        color = (camera.get("color") if camera.get("color") is not None else camera.get("rgb")) if isinstance(camera, dict) else camera
        if color is None:
            continue
        array = np.asarray(color)
        if array.ndim != 3:
            continue
        if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if np.issubdtype(array.dtype, np.floating):
            array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        images.append(encode_image(array.astype(np.uint8), name, quality))
    measured = obs.get("state") or {}
    state = {key: _plain(value) for key, value in measured.items()}
    state["packed"] = packed_state(measured, layout)
    state["robot"] = {"action_type": action_type, "packing": [key for item in layout for key in (item[1], item[3])],
                      "arm_dim": [item[2] for item in layout], "ee_dim": [item[4] for item in layout]}
    return {"text": obs.get("instruction") or "", "state": state, "images": images}


def to_xpolicylab_actions(action, layout):
    """REA's action -> XPolicyLab's action chunk (a list of {key: float32 array})."""
    name, commands = action.get("name"), (action.get("arguments") or {}).get("commands")
    if not commands:
        raise ValueError(f"REA returned {name!r} without commands")
    chunk = []
    for command in commands:
        out = {}
        if name == "action_chunk":
            for arm_key, _, arm_dim, gripper_key, gripper_dim in layout:
                out[arm_key] = np.asarray(command[arm_key], dtype=np.float32).reshape(arm_dim)
                out[gripper_key] = np.asarray(command[gripper_key], dtype=np.float32).reshape(gripper_dim)
        elif name == "packed_action_chunk":
            vector, offset = np.asarray(command, dtype=np.float32).reshape(-1), 0
            for arm_key, _, arm_dim, gripper_key, gripper_dim in layout:
                out[arm_key] = vector[offset:offset + arm_dim]
                offset += arm_dim
                out[gripper_key] = vector[offset:offset + gripper_dim]
                offset += gripper_dim
            if offset != vector.size:
                raise ValueError(f"packed command has {vector.size} values, expected {offset}")
        else:
            raise ValueError(f"REA returned an unsupported action {name!r}")
        chunk.append(out)
    return chunk


def hold_action(obs, layout):
    """One command that keeps the measured state, or None when the observation does not report it."""
    state = obs.get("state") or {}
    command = {}
    for arm_key, arm_state, arm_dim, gripper_key, gripper_dim in layout:
        if arm_state not in state or gripper_key not in state:
            return None
        command[arm_key] = np.asarray(state[arm_state], dtype=np.float32).reshape(arm_dim)
        command[gripper_key] = np.asarray(state[gripper_key], dtype=np.float32).reshape(gripper_dim)
    return [command]


def success_of(result):
    if isinstance(result, bool):
        return result
    if isinstance(result, dict):
        for key in ("success", "is_success", "succ", "succeeded"):
            if isinstance(result.get(key), (bool, int)):
                return bool(result[key])
    return None


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = dict(model_cfg)
        self.action_type = self.cfg.get("action_type") or "joint"
        dim_info = self.cfg.get("robot_action_dim_info")
        if dim_info is None:
            from XPolicyLab.utils.process_data import get_robot_action_dim_info
            dim_info = get_robot_action_dim_info(self.cfg["env_cfg_type"])
        self.layout = robot_layout(self.action_type, dim_info)
        self.max_chunk = int(self.cfg.get("max_chunk") or 50)
        self.quality = int(self.cfg.get("jpeg_quality") or 90)
        self.cameras = self.cfg.get("cameras") or None
        root = self.cfg.get("rea_root") or os.environ.get("REA_ROOT")
        if not root:
            raise RuntimeError("set rea_root in deploy.yml (or REA_ROOT) to a REA checkout")
        self.rea_root = Path(root).expanduser().resolve()
        client_dir = str(self.rea_root / "integrations" / "python")
        if client_dir not in sys.path:
            sys.path.insert(0, client_dir)
        import rea_client
        self.client = rea_client
        config = self.cfg.get("rea_config")
        if not config:
            raise RuntimeError("set rea_config in deploy.yml to the REA configuration to serve")
        config = Path(config).expanduser()
        self.rea_config = config if config.is_absolute() else self.rea_root / config
        self.sessions, self.latest, self.open, self.finished, self.last = {}, {}, set(), {}, {}
        print(f"[REA] serving {self.rea_config} for {self.cfg.get('bench_name')}/{self.cfg.get('task_name')} "
              f"({self.action_type}, {len(self.layout)} arm(s))", file=sys.stderr)

    def _session(self, index):
        if index not in self.sessions:
            output = self.cfg.get("rea_output")
            session = self.client.ReaSession(self.rea_config, rea_root=self.rea_root, node=self.cfg.get("node"),
                                             output=(Path(output) / f"env-{index}") if output else None,
                                             timeout_s=float(self.cfg.get("step_timeout_s") or 600))
            session.configure(actions=action_specs(self.action_type, self.layout, self.max_chunk),
                              name=f"xpolicylab:{self.cfg.get('bench_name') or 'bench'}:{self.cfg.get('task_name') or 'task'}")
            self.sessions[index] = session
        return self.sessions[index]

    def _act(self, index):
        obs = self.latest.get(index)
        if obs is None:
            raise RuntimeError(f"get_action before update_obs for environment {index}")
        session = self._session(index)
        if index not in self.open:
            seed = self.cfg.get("seed")
            session.reset(task=self.cfg.get("task_name"), seed=int(seed) if seed not in (None, "", "null") else 0,
                          instruction=obs.get("instruction"))
            self.open.add(index)
            self.finished.pop(index, None)
            self.last.pop(index, None)
        if index not in self.finished:
            reply = session.step(to_rea_observation(obs, self.layout, self.action_type, self.client.encode_image, self.quality, self.cameras))
            if not reply.get("done"):
                chunk = to_xpolicylab_actions(reply["action"], self.layout)
                self.last[index] = chunk
                return chunk
            if reply.get("reason") == "error":
                raise RuntimeError(f"REA stopped with an error: {reply.get('error')}")
            self.finished[index] = reply
        hold = hold_action(obs, self.layout) or (self.last.get(index) or [None])[-1:]
        if not hold or hold[0] is None:
            raise RuntimeError("REA ended the episode and the observation reports no state to hold")
        return hold

    def update_obs(self, obs):
        self.latest[0] = obs

    def update_obs_batch(self, obs_list):
        for position, obs in enumerate(obs_list):
            self.latest[obs.get("env_idx", position)] = obs

    def get_action(self):
        return self._act(0)

    def get_action_batch(self, env_idx_list=None):
        indices = list(env_idx_list) if env_idx_list is not None else sorted(self.latest)
        return [self._act(index) for index in indices]

    def _end(self, result=None):
        for index in sorted(self.open):
            try:
                self.sessions[index].end(result)
            except self.client.ReaError as error:
                print(f"[REA] ending environment {index}: {error}", file=sys.stderr)
        self.open.clear()

    def on_trial_end(self, result=None):
        self._end({"success": success_of(result)} if success_of(result) is not None else None)

    def reset(self):
        self._end(None)
        self.latest.clear()
        self.finished.clear()
        self.last.clear()

    def close(self):
        self._end(None)
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass
