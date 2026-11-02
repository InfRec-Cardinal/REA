from enum import Enum
import importlib.util
from pathlib import Path
import unittest


source = Path(__file__).resolve().parents[1] / "scripts" / "arc_agi3_backend.py"
spec = importlib.util.spec_from_file_location("arc_agi3_backend", source)
backend_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_module)


class Action(Enum):
    RESET = 0
    ACTION1 = 1
    ACTION6 = 6


class State(Enum):
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


class Frame:
    def __init__(self, state=State.NOT_FINISHED, levels=0, legacy=False):
        self.state = state
        self.levels = levels
        self.legacy = legacy
        self.frame = [[[1, 2], [3, 4]], [[4, 3], [2, 1]]]

    def model_dump(self):
        return {
            "state": self.state,
            "game_id": "fixture-v1",
            "levels_complete" if self.legacy else "levels_completed": self.levels,
            "available_actions": [1],
        }


class Environment:
    def __init__(self):
        self.scorecard_id = "official-card"
        self.observation_space = Frame()
        self.action_space = [Action.ACTION1]
        self.calls = []
        self.reset_calls = 0
        self.next_frame = Frame()

    def reset(self):
        self.reset_calls += 1
        return self.observation_space

    def step(self, action, data=None, reasoning=None):
        self.calls.append((action, data, reasoning))
        self.observation_space = self.next_frame
        self.action_space = [Action.ACTION6]
        return self.observation_space


class Arcade:
    def __init__(self):
        self.environment = Environment()
        self.calls = []
        self.closed = []
        self.card = {"score": 17.25, "environments": [{"score": 19.0}], "official_extra": True}

    def make(self, game_id, **kwargs):
        self.calls.append((game_id, kwargs))
        return self.environment

    def get_scorecard(self, scorecard_id):
        return self.card

    def close_scorecard(self, scorecard_id):
        self.closed.append(scorecard_id)
        return {**self.card, "closed": True}


class ArcBackendTests(unittest.TestCase):
    def setUp(self):
        self.arcade = Arcade()
        self.options = []
        def sdk(options):
            self.options.append(options)
            return self.arcade, Action, State, "0.9.9"
        self.backend = backend_module.ArcAgi3Backend(
            sdk_factory=sdk,
            image_encoder=lambda frames: [{"mimeType": "image/png", "data": "fixture"} for _ in frames],
        )
        self.backend.configure({})

    def reset(self):
        return self.backend.reset({"task": "fixture", "seed": 2})

    def step(self, action, **options):
        return self.backend.step({"action": action, "observationId": self.backend.observation["id"], **options})

    def test_make_uses_its_initial_observation_without_an_extra_reset(self):
        transition = self.backend.reset({"task": {"gameId": "fixture", "seed": 7}, "seed": 2})
        self.assertEqual(self.arcade.calls[0][1]["seed"], 7)
        self.assertEqual(self.arcade.environment.reset_calls, 0)
        self.assertEqual(self.options[0]["operationMode"], "OFFLINE")
        self.assertEqual(transition["observation"]["state"]["frame"], self.arcade.environment.observation_space.frame)
        self.assertEqual(len(transition["observation"]["images"]), 2)
        self.assertIsNone(transition["reward"])
        self.assertIsNone(transition["success"])
        self.assertFalse(transition["terminated"])
        self.assertEqual(transition["observation"]["state"]["state"], "NOT_FINISHED")

    def test_actions_refresh_and_coordinates_reasoning_reach_the_sdk(self):
        first = self.reset()["observation"]
        self.assertEqual([item["name"] for item in first["actions"]], ["ACTION1", "RESET"])
        second = self.step({"name": "ACTION1"})["observation"]
        self.assertEqual([item["name"] for item in second["actions"]], ["ACTION6", "RESET"])
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(second["actions"][0]["parameters"]["required"], ["x", "y"])
        self.step({"name": "ACTION6", "arguments": {"x": 63, "y": 0}}, reasoning={"thought": "test click"})
        self.assertEqual(self.arcade.environment.calls[-1], (Action.ACTION6, {"x": 63, "y": 0}, {"thought": "test click"}))
        self.assertEqual(self.backend.dispatch({"op": "observe"}), self.backend.observation)

    def test_invalid_or_unavailable_actions_do_not_step(self):
        self.reset()
        with self.assertRaisesRegex(ValueError, "not available"):
            self.step({"name": "ACTION6", "arguments": {"x": 1, "y": 1}})
        with self.assertRaisesRegex(ValueError, "does not accept"):
            self.step({"name": "ACTION1", "arguments": {"x": 1}})
        self.assertEqual(self.arcade.environment.calls, [])
        self.step({"name": "ACTION1"})
        for arguments in ({"x": 64, "y": 0}, {"x": 0}, {"x": True, "y": 1}):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "coordinates"):
                self.step({"name": "ACTION6", "arguments": arguments})
        self.assertEqual(len(self.arcade.environment.calls), 1)

    def test_old_observation_is_rejected_without_execution(self):
        first = self.reset()["observation"]
        self.step({"name": "ACTION1"})
        with self.assertRaisesRegex(ValueError, "stale"):
            self.backend.step({"observationId": first["id"], "action": {"name": "RESET"}})
        self.assertEqual(len(self.arcade.environment.calls), 1)

    def test_win_and_legacy_levels_are_reported_without_synthetic_rewards(self):
        self.reset()
        self.arcade.environment.next_frame = Frame(State.WIN, levels=3, legacy=True)
        transition = self.step({"name": "RESET"})
        self.assertTrue(transition["terminated"])
        self.assertTrue(transition["success"])
        self.assertIsNone(transition["reward"])
        self.assertEqual(transition["observation"]["state"]["levels_completed"], 3)
        self.assertEqual(transition["info"]["levelsCompleted"], 3)
        self.assertFalse(transition["info"]["gameOver"])

    def test_game_over_requires_an_explicit_reset_and_preserves_the_game_budget(self):
        self.reset()
        self.arcade.environment.next_frame = Frame(State.GAME_OVER, levels=1)
        lost = self.step({"name": "ACTION1"})
        self.assertFalse(lost["terminated"])
        self.assertIsNone(lost["success"])
        self.assertTrue(lost["info"]["gameOver"])
        self.assertEqual([item["name"] for item in lost["observation"]["actions"]], ["RESET"])
        self.assertEqual(len(self.arcade.environment.calls), 1)
        self.assertEqual(self.arcade.environment.reset_calls, 0)
        with self.assertRaisesRegex(ValueError, "not available"):
            self.step({"name": "ACTION6", "arguments": {"x": 1, "y": 1}})
        self.arcade.environment.next_frame = Frame(State.NOT_FINISHED, levels=1)
        recovered = self.step({"name": "RESET"})
        self.assertFalse(recovered["terminated"])
        self.assertFalse(recovered["info"]["gameOver"])
        self.assertIsNone(recovered["success"])
        self.assertEqual([item["name"] for item in recovered["observation"]["actions"]], ["ACTION6", "RESET"])
        self.assertEqual(self.arcade.environment.calls[-1], (Action.RESET, {}, None))
        self.arcade.environment.next_frame = Frame(State.WIN, levels=2)
        won = self.step({"name": "ACTION6", "arguments": {"x": 1, "y": 1}})
        self.assertTrue(won["terminated"])
        self.assertTrue(won["success"])
        self.assertEqual(len(self.arcade.environment.calls), 3)
        self.assertEqual(len(self.arcade.calls), 1)

    def test_end_on_game_over_retains_single_attempt_terminal_failure(self):
        self.backend = backend_module.ArcAgi3Backend(self.backend.sdk_factory, self.backend.image_encoder)
        self.backend.configure({"endOnGameOver": True})
        self.reset()
        self.arcade.environment.next_frame = Frame(State.GAME_OVER)
        lost = self.step({"name": "ACTION1"})
        self.assertTrue(lost["terminated"])
        self.assertFalse(lost["success"])
        self.assertTrue(lost["info"]["gameOver"])
        self.assertEqual([item["name"] for item in lost["observation"]["actions"]], ["RESET"])

    def test_competition_rejects_reinitializing_the_same_game_before_calling_sdk(self):
        self.backend = backend_module.ArcAgi3Backend(self.backend.sdk_factory, self.backend.image_encoder)
        self.backend.configure({"operationMode": "COMPETITION"})
        self.backend.reset({"task": "fixture-v1", "seed": 2})
        with self.assertRaisesRegex(ValueError, "one initialization per game"):
            self.backend.reset({"task": "fixture", "seed": 3})
        self.assertEqual(len(self.arcade.calls), 1)
        self.step({"name": "RESET"})
        self.assertEqual(self.arcade.environment.calls[-1][0], Action.RESET)
        self.backend.reset({"task": "other", "seed": 2})
        self.assertEqual(len(self.arcade.calls), 2)

    def test_offline_allows_multiple_seeds_for_one_game(self):
        self.backend.reset({"task": "fixture", "seed": 2})
        self.backend.reset({"task": "fixture", "seed": 3})
        self.assertEqual([options["seed"] for _, options in self.arcade.calls], [2, 3])

    def test_scorecard_is_official_data_and_closing_is_explicit(self):
        self.reset()
        self.assertEqual(self.backend.scorecard(), self.arcade.card)
        self.assertEqual(self.arcade.closed, [])
        scorecard = self.backend.dispatch({"op": "scorecard", "close": True})
        self.assertEqual(scorecard, {**self.arcade.card, "closed": True})
        self.assertEqual(self.arcade.closed, ["official-card"])
        self.backend.close()
        self.assertEqual(self.arcade.closed, ["official-card"])


if __name__ == "__main__":
    unittest.main()
