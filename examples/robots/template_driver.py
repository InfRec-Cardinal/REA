"""A minimal REA robot driver. Copy this file, keep the attribute and method names, and call your robot SDK inside them.

Use it from a configuration, with this file's directory on PYTHONPATH:

    "environment": {"type": "python-robot", "options": {
        "driver": "template_driver:TemplateDriver", "python": "/path/to/robot/venv/bin/python",
        "options": {"joints": 2}, "execute": false}}

REA calls the methods below one at a time from scripts/robot_backend.py. Snapshots are JSON: "state" (any JSON),
"images" ([{"name", "mimeType", "data": base64}], see robot_drivers.common.image_observation), and optionally "text",
"terminated", "truncated", "success" (only when the driver can evaluate the task), "reward" and "info".
With execute=False the driver must not move the robot: step() only validates and reports what it would do.
"""


class TemplateDriver:
    def __init__(self, options):
        self.joints = int(options.get("joints", 2))
        self.id = "template-robot"
        self.timing = "turn-based"
        self.actions = [{
            "name": "move_joints",
            "description": "Absolute joint targets in radians, in joint order",
            "parameters": {
                "type": "object",
                "properties": {"positions": {"type": "array", "items": {"type": "number", "minimum": -3.14, "maximum": 3.14},
                                             "minItems": self.joints, "maxItems": self.joints}},
                "required": ["positions"], "additionalProperties": False,
            },
        }]
        self.metadata = {"jointNames": [f"joint_{index}" for index in range(self.joints)], "units": "radians"}
        self.positions = [0.0] * self.joints

    def reset(self, task=None, seed=0):
        # On hardware: move to, or verify, a safe start pose. A physical reset is not a simulator reset.
        self.positions = [0.0] * self.joints
        return self.observe()

    def observe(self):
        return {"state": {"positions": list(self.positions)}, "images": [], "success": None}

    def step(self, action, execute=False):
        targets = [float(value) for value in action["arguments"]["positions"]]
        if execute:
            self.positions = targets  # send the targets to the robot SDK here and wait for the control interval
        return {**self.observe(), "info": {"executed": execute, "requested": targets}}

    def stop(self):
        # Bring the robot to a safe halt; report whether the SDK supports it.
        return {"supported": True}

    def close(self):
        pass
