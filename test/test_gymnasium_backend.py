import base64
import importlib.util
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
import zlib


source = Path(__file__).resolve().parents[1] / "scripts" / "gymnasium_backend.py"
spec = importlib.util.spec_from_file_location("gymnasium_backend", source)
backend_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_module)


class Array:
    def __init__(self, values, dtype=None):
        self.values = values
        self.dtype = dtype

    def tolist(self):
        return self.values


class Discrete:
    def __init__(self, n, start=0):
        self.n = n
        self.start = start

    def contains(self, value):
        return type(value) is int and self.start <= value < self.start + self.n


class Box:
    def __init__(self, low, high, shape, kind="f"):
        self.low = Array(low)
        self.high = Array(high)
        self.shape = shape
        self.dtype = SimpleNamespace(kind=kind)

    def contains(self, value):
        return isinstance(value, Array) and value.dtype is self.dtype


class MultiDiscrete:
    def __init__(self, nvec, start, shape):
        self.nvec = Array(nvec)
        self.start = Array(start)
        self.shape = shape
        self.dtype = SimpleNamespace(kind="i")


class MultiBinary:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = SimpleNamespace(kind="i")


class Dict:
    def __init__(self, spaces):
        self.spaces = spaces


class Tuple:
    def __init__(self, spaces):
        self.spaces = spaces


SPACES = SimpleNamespace(Discrete=Discrete, Box=Box, MultiDiscrete=MultiDiscrete, MultiBinary=MultiBinary, Dict=Dict, Tuple=Tuple)
NUMPY = SimpleNamespace(asarray=lambda value, dtype=None: Array(value, dtype))


class Environment:
    def __init__(self, action_space):
        self.action_space = action_space
        self.render_mode = None
        self.render_calls = 0
        self.closed = False
        self.steps = []
        self.resets = []
        self.next_result = (Array([0.2, 0.3]), 1.5, False, False, {})

    def reset(self, **kwargs):
        self.resets.append(kwargs)
        return Array([0.1, 0.2]), {"mask": Array([1, 1])}

    def step(self, action):
        self.steps.append(action)
        return self.next_result

    def render(self):
        self.render_calls += 1
        return Array([[[255, 0, 0], [0, 255, 0]]])

    def close(self):
        self.closed = True


class GymnasiumBackendTests(unittest.TestCase):
    def setUp(self):
        self.environments = []
        self.make_calls = []
        self.selected_space = Discrete(2)
        def make(environment_id, **kwargs):
            self.make_calls.append((environment_id, kwargs))
            env = Environment(self.selected_space)
            env.render_mode = kwargs.get("render_mode")
            self.environments.append(env)
            return env
        gym = SimpleNamespace(make=make, spaces=SPACES)
        self.backend = backend_module.GymnasiumBackend(sdk_factory=lambda options: (gym, NUMPY, "1.3.0"))
        self.backend.configure({"environmentId": "Fixture-v1", "makeOptions": {"limit": 3}})

    def step(self, value, **kwargs):
        return self.backend.step({
            "action": {"name": "step", "arguments": {"action": value}},
            "observationId": self.backend.observation["id"], **kwargs,
        })

    def test_reset_options_and_observation_are_preserved_and_old_env_closes(self):
        initial = self.backend.reset({"seed": 7})
        self.assertEqual(self.make_calls, [("Fixture-v1", {"limit": 3})])
        self.assertEqual(self.environments[0].resets, [{"seed": 7, "options": None}])
        self.assertEqual(initial["observation"]["state"], {"observation": [0.1, 0.2], "info": {"mask": [1, 1]}})
        self.assertIsNone(initial["reward"])
        self.assertIsNone(initial["success"])
        self.assertEqual(self.backend.dispatch({"op": "observe"})["id"], initial["observation"]["id"])
        self.backend.reset({"task": {"environmentId": "Other-v2", "makeOptions": {"limit": 5}, "resetOptions": {"start": 1}}, "seed": 9})
        self.assertTrue(self.environments[0].closed)
        self.assertEqual(self.make_calls[-1], ("Other-v2", {"limit": 5}))
        self.assertEqual(self.environments[1].resets, [{"seed": 9, "options": {"start": 1}}])
        self.backend.close()
        self.assertTrue(self.environments[1].closed)
        self.backend.close()

    def test_discrete_start_bounds_are_checked_before_execution(self):
        self.selected_space = Discrete(3, start=2)
        first = self.backend.reset({"task": "Offset-v1"})
        schema = first["observation"]["actions"][0]["parameters"]["properties"]["action"]
        self.assertEqual(schema, {"type": "integer", "minimum": 2, "maximum": 4})
        for value in (1, 5, True, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.step(value)
        self.assertEqual(self.environments[0].steps, [])
        self.step(3)
        self.assertEqual(self.environments[0].steps, [3])
        with self.assertRaisesRegex(ValueError, "stale"):
            self.step(3, observationId=first["observation"]["id"])

    def test_box_per_coordinate_bounds_and_dtype_survive_transport(self):
        self.selected_space = Box([-2.0, 0.0], [2.0, 5.0], (2,))
        initial = self.backend.reset({})
        schema = initial["observation"]["actions"][0]["parameters"]["properties"]["action"]
        self.assertEqual([item["minimum"] for item in schema["prefixItems"]], [-2.0, 0.0])
        with self.assertRaisesRegex(ValueError, "bounds"):
            self.step([0.0, -1.0])
        with self.assertRaisesRegex(ValueError, "items"):
            self.step([0.0])
        self.step([1.5, 2.0])
        action = self.environments[0].steps[0]
        self.assertEqual(action.tolist(), [1.5, 2.0])
        self.assertIs(action.dtype, self.selected_space.dtype)

    def test_nested_multidiscrete_dict_tuple_and_multibinary(self):
        space = Dict({
            "selection": MultiDiscrete([[2, 3]], [[1, -1]], (1, 2)),
            "other": Tuple((Discrete(2), Box([-float("inf")], [float("inf")], (1,)), MultiBinary((2,)))),
        })
        schema = backend_module.space_schema(space, SPACES)
        value = {"selection": [[2, 1]], "other": [1, [0.5], [0, 1]]}
        backend_module.validate_value(value, schema)
        action = backend_module.decode_action(value, space, SPACES, NUMPY)
        self.assertEqual(action["selection"].tolist(), [[2, 1]])
        self.assertIsInstance(action["other"], tuple)
        self.assertEqual(action["other"][1].tolist(), [0.5])
        self.assertEqual(action["other"][2].tolist(), [0, 1])
        unbounded = schema["properties"]["other"]["prefixItems"][1]["prefixItems"][0]
        self.assertEqual(unbounded, {"type": "number"})
        value["selection"][0][1] = 2
        with self.assertRaisesRegex(ValueError, "bounds"):
            backend_module.validate_value(value, schema)

    def test_terminal_flags_do_not_invent_success(self):
        for info, success, terminated, truncated in (({}, None, True, False), ({"is_success": False}, False, False, True), ({"is_success": Array(True)}, True, True, False)):
            with self.subTest(info=info):
                self.backend.reset({})
                self.environments[-1].next_result = (Array([1.0]), -4.5, terminated, truncated, info)
                transition = self.step(0)
                self.assertEqual(transition["reward"], -4.5)
                self.assertEqual(transition["terminated"], terminated)
                self.assertEqual(transition["truncated"], truncated)
                self.assertIs(transition["success"], success)
                with self.assertRaisesRegex(ValueError, "Episode finished"):
                    self.step(0)

    def test_rgb_render_uses_standard_png_encoding_and_is_opt_in(self):
        initial = self.backend.reset({})
        self.assertEqual(initial["observation"]["images"], [])
        self.assertEqual(self.environments[0].render_calls, 0)
        rendered = self.backend.reset({"task": {"makeOptions": {"render_mode": "rgb_array"}}})
        image = rendered["observation"]["images"][0]
        data = base64.b64decode(image["data"])
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack("!II", data[16:24]), (2, 1))
        offset = data.index(b"IDAT")
        length = struct.unpack("!I", data[offset - 4:offset])[0]
        self.assertEqual(zlib.decompress(data[offset + 4:offset + 4 + length]), b"\x00\xff\x00\x00\x00\xff\x00")

    def test_unsupported_action_space_reports_its_type(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Gymnasium action space"):
            backend_module.space_schema(object(), SPACES)


if __name__ == "__main__":
    unittest.main()
