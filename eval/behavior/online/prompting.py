"""Model input for the online agent's BEHAVIOR decisions (rules 'v7'), as a pure function of what the runner gives every arm.

build() turns the task context (the same ASPIRE task prompt and typical-loop / budget paragraph the Agents SDK arm gets as
its instructions, with the SDK's stop sentence replaced by the finish wording), the online agent's episode context (decision index,
remaining budget, supervisor guidance, notes, recent rationales and actions) and the current head-camera JPEG into the system
instructions and the user message of one decision. It reads no simulator state.
"""
import base64
import json

SYSTEM = """You control a simulated R1Pro mobile manipulator (BEHAVIOR-1K) one skill per decision. Work like a careful engineer: decide from evidence, act, then verify.
- Each decision ends with exactly one take_action call: one skill with its arguments. The skill runs after the decision; its result appears in your action history at the next decision, together with a new head-camera image.
- Navigation and grasping take minutes each and the budget includes your own thinking time, so choose each skill deliberately and avoid needless calls.
- Re-check the current image against the instruction at every decision; earlier notes can be wrong. While the task is not finished, every decision must make progress toward it.
- Keep the episode notes short and factual: what you observed, the current sub-goal, and what you expect the next action to change. Replace them completely each time."""

V7_RULES = """- The episode is scored once, when you call finish or when the budget ends: the target object must be held in a gripper and lifted. finish is accepted only when check_object_in_hand is True for the arm you name; otherwise it is refused and the episode continues. If you are asked for another decision, the task is not finished: something is still missing. Do not repeat a skill or call finish again without a change; find what is missing and act.
- World frame: positions are in world metres, and the base pose is (x, y, yaw) with yaw in radians about +z. get_robot_position gives the current base pose; compute_approach_pose gives the base pose that faces an object from a chosen standoff.
- Colour, size and part words in the instruction describe the target object: use them in object_name for the perception tools and pick the matching object.
- Check each skill's result before the next: False, "none", "not found" or ERROR means it did not do what you wanted - change something instead of repeating it."""

HELPERS = """Helpers in this decision: get_object_pose, get_robot_position, compute_approach_pose, sample_grasp_pose, get_current_eef_pose, get_sam3_mask, detect_objects and check_object_in_hand run now and return their result to you before you choose the skill. They move neither the base nor the arms, but they take time. grasp_object uses the candidates of the latest sample_grasp_pose of that object. You may make up to 3 helper calls, then you must call take_action; skip them when the next skill is obvious."""

RULES = {'v7': (V7_RULES,)}


def _truncate(text, limit):
    text = text if isinstance(text, str) else json.dumps(text)
    return text if len(text) <= limit else text[:max(0, limit - 3)] + '...'


def history_view(history, limit=300):
    """Compact action history: skill, arguments, result truncated to `limit` characters, seconds."""
    return [{'step': item.get('step'), 'skill': item.get('skill'), 'arguments': item.get('arguments') or {},
             'result': _truncate(item.get('result'), limit),
             'seconds': round(item['seconds']) if isinstance(item.get('seconds'), (int, float)) else None}
            for item in history]


def build(task_context, context, route, config, image_bytes):
    """System instructions and user message for one decision.

    context: decision, max_decisions, remaining_s, budget_s, instruction, guidance, notes and history (oldest first; items with
    step, skill, arguments, result, seconds, thought). image_bytes: the current head-camera JPEG, or None when it could not be
    rendered.
    """
    text = task_context.strip()
    if context.get('instruction'):
        text += '\nTask: ' + context['instruction']
    text += (f"\nDecision {context['decision']} of at most {context['max_decisions']}; "
             f"about {max(0, round(context['remaining_s']))} s of the {round(context['budget_s'])} s budget remain.")
    if context.get('guidance'):
        text += '\nSupervisor guidance: ' + context['guidance']
    text += '\nYour episode notes: ' + (context.get('notes') or '(none yet)')
    history = context.get('history') or []
    thoughts = [(item.get('step'), item.get('thought')) for item in history if item.get('thought')][-config.get('thought_history', 6):]
    if thoughts:
        text += '\nYour recent reasoning, oldest first: ' + json.dumps([{'step': s, 'thought': t} for s, t in thoughts])
    actions = history_view(history[-config.get('history_limit', 12):], config.get('result_chars', 300))
    text += '\nYour recent actions, oldest first: ' + (json.dumps(actions) if actions else '(none yet)')
    text += '\nChoose the next action now.'
    content = [{'type': 'input_text', 'text': text}]
    if image_bytes:
        content += [{'type': 'input_text', 'text': 'Image: head camera now'},
                    {'type': 'input_image', 'detail': config.get('image_detail', 'high'),
                     'image_url': 'data:image/jpeg;base64,' + base64.b64encode(image_bytes).decode()}]
    else:
        content.append({'type': 'input_text', 'text': 'Image: head camera now (unavailable: the render failed)'})
    system = SYSTEM + ''.join('\n' + block for block in RULES.get(config.get('prompt_rules'), ()))
    if config.get('helpers'):
        system += '\n' + HELPERS
    return {'route': route, 'instructions': system + '\n' + config[route + '_instructions'] + ' Call take_action exactly once.',
            'inputs': [{'role': 'user', 'content': content}]}
