"""Prompt construction of the online BEHAVIOR adapter (rules 'v7'): a pure function of the task context, online's episode
context and the head-camera image."""
import base64
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / 'eval' / 'behavior' / 'online'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# by path, under its own name: the RoboTwin adapter's prompting module is imported as `prompting` by other tests
prompting = load('online_behavior_prompting_test', ADAPTER / 'prompting.py')
arm = load('behavior_agents_sdk_arm_prompting', ROOT / 'eval/behavior/runner/agents.py')
CONFIG = json.loads((ADAPTER / 'config.example.json').read_text())
TASK = arm.online_task_context((ROOT / 'test/fixtures/behavior_task_prompt.txt').read_text(), 1500)
IMAGE = b'\xff\xd8head-camera\xff\xd9'


def context(**over):
    value = {'decision': 3, 'max_decisions': 40, 'remaining_s': 1123.4, 'budget_s': 1500.0, 'guidance': None, 'notes': 'radio on the table ahead',
             'instruction': 'Pick up the red radio from the table and hold it in the gripper.',
             'history': [{'step': 1, 'skill': 'find_object_base_rotate', 'arguments': {'object_name': 'red radio'}, 'result': 'true', 'seconds': 41.2, 'thought': 'look for it'},
                         {'step': 2, 'skill': 'sample_grasp_pose', 'arguments': {'object_name': 'red radio'}, 'result': 'x' * 1000, 'seconds': 12.0, 'thought': None}]}
    value.update(over)
    return value


def texts(request):
    return [item['text'] for item in request['inputs'][0]['content'] if item['type'] == 'input_text']


class BehaviorPromptTest(unittest.TestCase):
    def test_user_message_layout(self):
        request = prompting.build(TASK, context(), 'reflex', CONFIG, IMAGE)
        first, label = texts(request)
        self.assertTrue(first.startswith(TASK.strip()))
        lines = first[len(TASK.strip()):].splitlines()[1:]
        self.assertEqual(lines[0], 'Task: Pick up the red radio from the table and hold it in the gripper.')
        self.assertEqual(lines[1], 'Decision 3 of at most 40; about 1123 s of the 1500 s budget remain.')
        self.assertEqual(lines[2], 'Your episode notes: radio on the table ahead')
        self.assertEqual(lines[3], 'Your recent reasoning, oldest first: [{"step": 1, "thought": "look for it"}]')
        actions = json.loads(lines[4].split(': ', 1)[1])
        self.assertEqual(actions[0], {'step': 1, 'skill': 'find_object_base_rotate', 'arguments': {'object_name': 'red radio'}, 'result': 'true', 'seconds': 41})
        self.assertEqual(len(actions[1]['result']), 300)
        self.assertTrue(actions[1]['result'].endswith('...'))
        self.assertEqual(lines[5], 'Choose the next action now.')
        self.assertEqual(label, 'Image: head camera now')
        image = request['inputs'][0]['content'][-1]
        self.assertEqual(image['type'], 'input_image')
        self.assertEqual(image['detail'], 'high')
        self.assertEqual(base64.b64decode(image['image_url'].split(',', 1)[1]), IMAGE)
        self.assertNotIn('Supervisor guidance', first)

    def test_deliberate_route_carries_guidance_and_instructions(self):
        guidance = 'finish was refused: check_object_in_hand(0) was False, so arm 0 does not hold the object.'
        request = prompting.build(TASK, context(guidance=guidance), 'deliberate', CONFIG, IMAGE)
        self.assertIn('\nSupervisor guidance: ' + guidance + '\nYour episode notes:', texts(request)[0])
        self.assertTrue(request['instructions'].endswith(CONFIG['deliberate_instructions'] + ' Call take_action exactly once.'))
        self.assertTrue(request['instructions'].startswith(prompting.SYSTEM))
        self.assertEqual(request['route'], 'deliberate')

    def test_system_text_has_the_v7_behavior_rules_and_helpers(self):
        instructions = prompting.build(TASK, context(), 'reflex', CONFIG, IMAGE)['instructions']
        self.assertIn('The episode is scored once, when you call finish or when the budget ends: the target object must be held in a gripper and lifted.', instructions)
        self.assertIn('If you are asked for another decision, the task is not finished: something is still missing.', instructions)
        self.assertIn('Do not repeat a skill or call finish again without a change; find what is missing and act.', instructions)
        self.assertIn('earlier notes can be wrong', instructions)
        self.assertIn('every decision must make progress toward it', instructions)
        self.assertIn('yaw in radians about +z', instructions)
        self.assertIn('Colour, size and part words in the instruction describe the target object', instructions)
        self.assertIn('False, "none", "not found" or ERROR means it did not do what you wanted', instructions)
        self.assertIn('up to 3 helper calls', instructions)
        self.assertNotIn('active arm', instructions)  # scorer-internal: neither arm is told (same words in both arms)
        self.assertTrue(instructions.endswith(CONFIG['reflex_instructions'] + ' Call take_action exactly once.'))

    def test_first_decision_and_missing_image(self):
        request = prompting.build(TASK, context(decision=1, history=[], notes=''), 'reflex', CONFIG, None)
        first, label = texts(request)
        self.assertIn('Your episode notes: (none yet)', first)
        self.assertIn('Your recent actions, oldest first: (none yet)', first)
        self.assertNotIn('recent reasoning', first)
        self.assertIn('unavailable', label)
        self.assertFalse(any(item['type'] == 'input_image' for item in request['inputs'][0]['content']))

    def test_only_the_last_six_rationales_and_twelve_actions(self):
        history = [{'step': i, 'skill': 'lift_arm', 'arguments': {'arm': 0}, 'result': 'true', 'seconds': 1, 'thought': f't{i}'} for i in range(1, 15)]
        first = texts(prompting.build(TASK, context(history=history), 'reflex', CONFIG, IMAGE))[0]
        reasoning = json.loads(first.split('Your recent reasoning, oldest first: ', 1)[1].split('\n', 1)[0])
        self.assertEqual([item['thought'] for item in reasoning], [f't{i}' for i in range(9, 15)])
        actions = json.loads(first.split('Your recent actions, oldest first: ', 1)[1].split('\n', 1)[0])
        self.assertEqual([item['step'] for item in actions], list(range(3, 15)))

    def test_pure(self):
        a = prompting.build(TASK, context(), 'reflex', CONFIG, IMAGE)
        b = prompting.build(TASK, context(), 'reflex', CONFIG, IMAGE)
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
