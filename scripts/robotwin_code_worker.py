"""Evaluate one observation-grounded Python controller decision."""
import contextlib
import io
import json
import sys


def execute_program(code, observation, instruction):
    import numpy as np

    commands = []

    def move_gripper(arm, position, quaternion, opening):
        commands.append({
            "arm": arm, "position": np.asarray(position, dtype=float).tolist(),
            "quaternion": np.asarray(quaternion, dtype=float).tolist(), "opening": float(opening),
        })

    scope = {"np": np, "observation": observation, "instruction": instruction, "move_gripper": move_gripper}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "<robotwin-policy>", "exec"), scope)
    if len(commands) != 1:
        raise ValueError("A decision must call move_gripper exactly once")
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
