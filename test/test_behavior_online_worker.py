"""online BEHAVIOR worker: the take_action schema and argument checks, the tools the model receives and the failure labels
(no model calls; openai-agents is replaced by stubs)."""
import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arm = load("behavior_agents_sdk_arm_worker", ROOT / "eval/behavior/runner/agents.py")
worker = load("behavior_online_worker_unit", ROOT / "eval/behavior/online/worker.py")


class WorkerActionTest(unittest.TestCase):
    SKILLS, HELPERS = arm.online_tool_specs()

    def test_take_action_schema_is_strict_with_nullable_arguments(self):
        schema = worker.take_action_schema(self.SKILLS)
        self.assertEqual(schema["required"], ["skill", "arguments", "notes", "thought"])
        arguments = schema["properties"]["arguments"]
        self.assertFalse(arguments["additionalProperties"])
        self.assertEqual(sorted(arguments["required"]), sorted(arguments["properties"]))
        self.assertEqual(arguments["properties"]["arm"], {"type": ["integer", "null"]})
        self.assertIn("finish", schema["properties"]["skill"]["enum"])
        self.assertNotIn("get_object_pose", schema["properties"]["skill"]["enum"])

    def test_normalize_action(self):
        self.assertEqual(worker.normalize_action(self.SKILLS, "grasp_object", {"object_name": "drill", "candidate_id": 2.0, "arm": 1, "x": None}),
                         ({"skill": "grasp_object", "arguments": {"object_name": "drill", "candidate_id": 2, "arm": 1}}, None))
        self.assertEqual(worker.normalize_action(self.SKILLS, "reset_torso", {"arm": None}), ({"skill": "reset_torso", "arguments": {}}, None))
        self.assertRegex(worker.normalize_action(self.SKILLS, "navigate_to_pose", {"x": 1, "y": 2})[1], "needs yaw")
        self.assertRegex(worker.normalize_action(self.SKILLS, "finish", {"arm": 2})[1], "arm must be 0")
        self.assertRegex(worker.normalize_action(self.SKILLS, "grasp_object", {"object_name": "d", "candidate_id": 1.5, "arm": 0})[1], "must be of type integer")
        self.assertRegex(worker.normalize_action(self.SKILLS, "get_object_pose", {"object_name": "d"})[1], "unknown skill")


    def test_model_input_record_keeps_the_text_and_drops_the_image_bytes(self):
        request = {"route": "reflex", "instructions": "i", "inputs": [{"role": "user", "content": [
            {"type": "input_text", "text": "t"}, {"type": "input_image", "detail": "high", "image_url": "data:image/jpeg;base64,AAAA"}]}]}
        record = worker.model_input_record(request, "/ep/online/observations/obs-001.jpg")
        self.assertEqual(record["inputs"][0]["content"][1], {"type": "input_image", "image_path": "/ep/online/observations/obs-001.jpg", "detail": "high"})
        self.assertNotIn("AAAA", str(record))


class ModelBehaviorError(Exception):
    pass


def stub_sdk(built, error, runs):
    """Stand-ins for agents, agents.agent and openai: FunctionTool records name -> description, Runner.run raises `error`."""
    agents, agents_agent, openai = types.ModuleType("agents"), types.ModuleType("agents.agent"), types.ModuleType("openai")

    class FunctionTool:
        def __init__(self, name, description, params_json_schema, on_invoke_tool, strict_json_schema):
            built[name] = description

    class Plain:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class Runner:
        @staticmethod
        async def run(agent, inputs, max_turns, hooks=None, run_config=None):
            runs.append(max_turns)
            raise error

    class AsyncOpenAI(Plain):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False
    agents.FunctionTool, agents.Runner = FunctionTool, Runner
    agents.Agent = agents.ModelSettings = agents.OpenAIResponsesModel = agents.RunConfig = Plain
    agents.RunHooks = object
    agents.set_tracing_disabled = lambda flag: None
    agents_agent.ToolsToFinalOutputResult = lambda **kwargs: kwargs
    openai.AsyncOpenAI = AsyncOpenAI
    return {"agents": agents, "agents.agent": agents_agent, "openai": openai}


def decide(error, config=None):
    built, runs = {}, []
    config = {"model": "m", "reasoning_effort": "medium", "base_url": "http://x.invalid", "helpers": True, **(config or {})}
    with patch.dict(sys.modules, stub_sdk(built, error, runs)):
        record = asyncio.run(worker.decide(None, config, {"notes": ""}, {"route": "reflex", "instructions": "i", "inputs": []}, arm, None))
    return record, built, runs


class WorkerToolsTest(unittest.TestCase):
    def test_helper_descriptions_are_the_sdk_arm_words_exactly(self):
        _, built, _ = decide(RuntimeError("stop"))
        described = {name: desc for name, desc, _ in arm.TOOL_SCHEMAS}
        self.assertEqual(set(built) - {"take_action"}, set(arm.REA_HELPERS))
        for name in arm.REA_HELPERS:
            with self.subTest(name=name):
                self.assertEqual(built[name], described[name])

    def test_helpers_false_offers_take_action_only(self):
        _, built, _ = decide(RuntimeError("stop"), {"helpers": False})
        self.assertEqual(set(built), {"take_action"})

    def test_failure_labels_and_no_decision_retry_by_default(self):
        failed = ModelBehaviorError("Responses stream ended with terminal event `response.failed`. status=failed; "
                                    "error=ResponseError(code='server_error', message='m', misalignment=None).")
        record, _, runs = decide(failed)
        self.assertEqual((record["status"], record["provider_error"], len(runs)), ("provider_error", "response_failed", 1))
        record, _, runs = decide(failed, {"decision_retries": 1})
        self.assertEqual(len(runs), 2)
        image = ModelBehaviorError("Responses stream ended with terminal event `response.failed`. status=failed; "
                                   "error=ResponseError(code='invalid_image', message='m', misalignment=None).")
        record, _, runs = decide(image)
        self.assertEqual((record["status"], len(runs)), ("model_invalid_action", 1))
        record, _, _ = decide(ModelBehaviorError("... `response.incomplete`. incomplete_details=IncompleteDetails(reason='max_output_tokens')"))
        self.assertEqual(record["status"], "model_invalid_action")


if __name__ == "__main__":
    unittest.main()
