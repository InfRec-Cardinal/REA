"""Compute an ALOHA action from generated Python without connecting to a robot."""

import contextlib
import io
import json
import sys


def execute_program(code, observation, instruction):
    import numpy as np

    commands = []

    def positions_array(positions, trajectory):
        values = np.asarray(positions, dtype=float)
        if trajectory:
            if values.ndim != 2 or values.shape[1] != 14 or not 1 <= values.shape[0] <= 100:
                raise ValueError("joint_trajectory requires 1 to 100 vectors of 14 absolute positions")
        elif values.shape != (14,):
            raise ValueError("joint_position requires 14 absolute positions")
        if not np.isfinite(values).all():
            raise ValueError("Joint positions must be finite")
        return values.tolist()

    def joint_position(positions):
        commands.append({"name": "joint_position", "arguments": {
            "positions": positions_array(positions, trajectory=False),
        }})

    def joint_trajectory(positions):
        commands.append({"name": "joint_trajectory", "arguments": {
            "positions": positions_array(positions, trajectory=True),
        }})

    scope = {
        "np": np, "observation": observation, "instruction": instruction,
        "joint_position": joint_position, "joint_trajectory": joint_trajectory,
    }
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "<aloha-policy>", "exec"), scope)
    if len(commands) != 1:
        raise ValueError("A decision must call joint_position or joint_trajectory exactly once")
    return {"action": commands[0], "stdout": output.getvalue()}


def main():
    print(json.dumps({"event": "ready"}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            result = execute_program(request["code"], request["observation"], request["instruction"])
            reply = {"id": request["id"], "ok": True, "result": result}
        except Exception as error:
            reply = {"id": request["id"], "ok": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
