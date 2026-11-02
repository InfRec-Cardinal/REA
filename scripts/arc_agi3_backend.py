import base64
from enum import Enum
from importlib.metadata import version
import io
import json
import logging
import os
import sys
import time


def json_value(value):
    if isinstance(value, Enum):
        return value.name
    if hasattr(value, "model_dump"):
        return json_value(value.model_dump())
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def frame_images(frames):
    import numpy as np
    from PIL import Image
    from arc_agi.rendering import frame_to_rgb_array

    images = []
    for frame in frames:
        rgb = frame_to_rgb_array(0, np.asarray(frame), scale=4)
        encoded = io.BytesIO()
        Image.fromarray(rgb).save(encoded, format="PNG")
        images.append({"mimeType": "image/png", "data": base64.b64encode(encoded.getvalue()).decode("ascii")})
    return images


def load_sdk(options):
    mode = options.get("operationMode", "OFFLINE").upper()
    if mode not in ("OFFLINE", "ONLINE", "COMPETITION"):
        raise ValueError("operationMode must be OFFLINE, ONLINE, or COMPETITION")
    os.environ["OPERATION_MODE"] = mode
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState

    logger = logging.getLogger("fast.arc-agi-3")
    logger.handlers = [logging.StreamHandler(sys.stderr)]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    kwargs = {"operation_mode": OperationMode[mode], "logger": logger}
    for key, sdk_key in (
        ("environmentsDir", "environments_dir"),
        ("recordingsDir", "recordings_dir"),
        ("baseUrl", "arc_base_url"),
        ("apiKey", "arc_api_key"),
    ):
        if options.get(key) is not None:
            kwargs[sdk_key] = options[key]
    return Arcade(**kwargs), GameAction, GameState, version("arc-agi")


def action_spec(action):
    if action.name == "ACTION6":
        parameters = {
            "type": "object",
            "properties": {axis: {"type": "integer", "minimum": 0, "maximum": 63} for axis in ("x", "y")},
            "required": ["x", "y"],
            "additionalProperties": False,
        }
        description = "Interact at grid coordinates x (column), y (row), each from 0 to 63."
    else:
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}
        description = "Reset using the environment's reset semantics." if action.name == "RESET" else "Perform this available action; infer its effect from observations."
    return {"name": action.name, "description": description, "parameters": parameters}


class ArcAgi3Backend:
    def __init__(self, sdk_factory=load_sdk, image_encoder=frame_images):
        self.sdk_factory = sdk_factory
        self.image_encoder = image_encoder
        self.arcade = None
        self.environment = None
        self.observation = None
        self.episode = 0
        self.sequence = 0
        self.options = {}
        self.closed_scorecard = None
        self.made_game_ids = set()

    def configure(self, options):
        if self.arcade is not None:
            raise ValueError("ARC backend is already configured")
        self.options = {"operationMode": "OFFLINE", "endOnGameOver": False, **options}
        self.options["operationMode"] = self.options["operationMode"].upper()
        self.arcade, self.actions, self.states, self.sdk_version = self.sdk_factory(self.options)
        return {"sdkVersion": self.sdk_version, "operationMode": self.options["operationMode"]}

    def reset(self, request):
        if self.arcade is None:
            raise ValueError("Configure the ARC backend before reset")
        task = request["task"]
        game_id = task.get("gameId") if isinstance(task, dict) else task
        seed = task.get("seed", request.get("seed", 0)) if isinstance(task, dict) else request.get("seed", 0)
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("task must be a gameId string or an object with gameId")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        game_key = game_id.split("-", 1)[0]
        if self.options["operationMode"] == "COMPETITION" and game_key in self.made_game_ids:
            raise ValueError(f"COMPETITION permits one initialization per game: {game_key}. Use the RESET action to retry the current level.")
        self.environment = self.arcade.make(
            game_id, seed=seed, scorecard_id=self.options.get("scorecardId"),
            save_recording=self.options.get("saveRecording", False), include_frame_data=True,
        )
        if self.environment is None or self.environment.observation_space is None:
            raise RuntimeError(f"ARC could not initialize {game_id}")
        self.made_game_ids.add(game_key)
        self.game_id = game_id
        self.seed = seed
        self.episode += 1
        self.sequence = 0
        return self.transition(self.environment.observation_space)

    def available_actions(self, frame=None):
        frame = frame if frame is not None else self.environment.observation_space
        if frame.state == self.states.GAME_OVER:
            return [self.actions.RESET]
        actions = list(self.environment.action_space)
        if self.actions.RESET not in actions:
            actions.append(self.actions.RESET)
        return actions

    def transition(self, frame):
        if frame is None:
            raise RuntimeError("ARC returned no observation")
        state = json_value(frame)
        state["frame"] = json_value(frame.frame)
        if "levels_completed" not in state and "levels_complete" in state:
            state["levels_completed"] = state["levels_complete"]
        self.sequence += 1
        self.observation = {
            "id": f"arc:{self.episode}:{self.sequence}",
            "timestampMs": time.time_ns() // 1_000_000,
            "text": f"ARC-AGI-3 {self.game_id}. Discover the rules from the frames and available actions.",
            "state": state,
            "images": self.image_encoder(state.get("frame", [])),
            "actions": [action_spec(action) for action in self.available_actions(frame)],
        }
        won = frame.state == self.states.WIN
        lost = frame.state == self.states.GAME_OVER
        terminal_loss = lost and self.options["endOnGameOver"]
        return {
            "observation": self.observation,
            "terminated": won or terminal_loss,
            "truncated": False,
            "success": True if won else False if terminal_loss else None,
            "reward": None,
            "info": {
                "gameId": state.get("game_id", self.game_id),
                "seed": self.seed if self.options["operationMode"] == "OFFLINE" else None,
                "requestedSeed": self.seed,
                "gameState": state["state"], "levelsCompleted": state.get("levels_completed"),
                "gameOver": lost,
                "operationMode": self.options["operationMode"], "sdkVersion": self.sdk_version,
                "scorecardId": self.environment.scorecard_id,
            },
        }

    def step(self, request):
        if self.observation is None:
            raise ValueError("Reset an episode before stepping")
        if request.get("observationId") != self.observation["id"]:
            raise ValueError("Action observation is stale")
        action = request["action"]
        if not isinstance(action, dict) or set(action) - {"name", "arguments"}:
            raise ValueError("action must contain name and optional arguments")
        available = {item.name: item for item in self.available_actions()}
        name = action.get("name")
        if name not in available:
            raise ValueError(f"Action {name} is not available")
        arguments = action.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("Action arguments must be an object")
        if name == "ACTION6":
            if set(arguments) != {"x", "y"} or any(type(value) is not int or not 0 <= value <= 63 for value in arguments.values()):
                raise ValueError("ACTION6 requires integer x and y coordinates in 0..63")
        elif arguments:
            raise ValueError(f"{name} does not accept arguments")
        frame = self.environment.step(available[name], data=arguments, reasoning=request.get("reasoning"))
        return self.transition(frame)

    def scorecard(self, close=False):
        if self.arcade is None:
            return self.closed_scorecard
        scorecard_id = self.environment.scorecard_id if self.environment is not None else self.options.get("scorecardId")
        if close:
            self.closed_scorecard = json_value(self.arcade.close_scorecard(scorecard_id))
            return self.closed_scorecard
        if self.closed_scorecard is not None:
            return self.closed_scorecard
        return json_value(self.arcade.get_scorecard(scorecard_id))

    def close(self):
        self.environment = None
        self.arcade = None
        return {"closed": True}

    def dispatch(self, request):
        operation = request["op"]
        if operation == "configure":
            return self.configure(request.get("options", {}))
        if operation == "reset":
            return self.reset(request)
        if operation == "observe":
            if self.observation is None:
                raise ValueError("Reset an episode before observing")
            return self.observation
        if operation == "step":
            return self.step(request)
        if operation == "scorecard":
            return self.scorecard(close=request.get("close", False))
        if operation == "close":
            return self.close()
        raise ValueError(f"Unknown operation: {operation}")


def main():
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    backend = ArcAgi3Backend()
    protocol_output.write(json.dumps({"event": "ready"}) + "\n")
    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                response = {"id": request.get("id"), "ok": True, "result": backend.dispatch(request)}
            except Exception as error:
                print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
                response = {"id": request.get("id"), "ok": False, "error": str(error)}
            protocol_output.write(json.dumps(response, separators=(",", ":")) + "\n")
            if request.get("op") == "close":
                break
    finally:
        backend.close()


if __name__ == "__main__":
    main()
