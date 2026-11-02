"""An in-memory Robot-compatible fixture; never opens a hardware connection."""


class NamedRobot:
    name = "named-fixture"

    def __init__(self, options):
        arms = options.get("arms", ["left", "right"])
        self.positions = {f"{arm}.joint.pos": 0.0 for arm in arms}
        self.action_features = {key: float for key in self.positions}
        self.observation_features = {**self.action_features, "top": (1, 1, 3)}
        self.is_connected = False

    def connect(self, calibrate=True):
        if calibrate:
            raise AssertionError("fixture must not be calibrated by the driver")
        self.is_connected = True

    def get_observation(self):
        return {**self.positions, "top": [[[255, 0, 0]]]}

    def send_action(self, action):
        self.positions = {key: min(0.5, value) for key, value in action.items()}
        return dict(self.positions)

    def disconnect(self):
        self.is_connected = False


def make_robot(options):
    return NamedRobot(options)
