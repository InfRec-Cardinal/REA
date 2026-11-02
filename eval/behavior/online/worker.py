"""One rea-online BEHAVIOR decision through the OpenAI Agents SDK (a model served over the Responses API).

runtime.mjs runs `<agent python> worker.py <decision dir>` once per decision. The worker reads <dir>/context.json and the run's
config, builds the model input with prompting.build, and runs one Runner.run (max_turns 4; decision_retries more after a
provider failure, 0 by default) whose tools are take_action and
the read-only helpers. Helper calls are executed live by the runner: the worker writes {"type":"call","id","tool","args"} on
stdout, the runtime relays it, and the result comes back as {"type":"result","id","value"} on stdin. At most 3 helper calls per
decision; the run stops as soon as take_action is accepted. The outcome goes to <dir>/decision.json.

The tool descriptions come from TOOL_SCHEMAS in eval/behavior/runner/agents.py (imported, not copied, nothing
appended), so both BEHAVIOR arms describe the R1Pro API with the same words. Nothing here reads simulator state.
"""
import sys

if sys.platform.startswith('linux'):  # PR_SET_PDEATHSIG: die with the runtime even if it is SIGKILLed (before the heavy imports)
    import ctypes
    import signal
    try:
        ctypes.CDLL('libc.so.6').prctl(1, signal.SIGKILL)
    except (OSError, AttributeError):
        pass

import asyncio
import importlib.util
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Loaded by path under its own name: eval/robotwin/online has a prompting.py too.
prompting = _load('online_behavior_prompting', HERE / 'prompting.py')

HELPER_BUDGET_TEXT = 'Helper budget used up; call take_action now.'
ACCEPTED_TEXT = 'ACTION_ACCEPTED. The skill runs once after this decision; its result is shown at the next decision.'


def load_arm_module(config):
    return _load('behavior_agents_sdk_arm', Path(config.get('agents_sdk_arm') or HERE.parent / 'runner' / 'agents.py'))


def schema_type(kind):
    return {'string': 'string', 'integer': 'integer', 'number': 'number'}[kind]


def take_action_schema(skills):
    """take_action(skill, arguments, notes, thought): arguments has every skill parameter, nullable (strict schemas list every
    property); the skill's own parameters must be given, the others are null."""
    params = {}
    for skill in skills:
        for name, kind in skill['parameters'].items():
            params.setdefault(name, kind)
    arguments = {'type': 'object', 'properties': {name: {'type': [schema_type(kind), 'null']} for name, kind in params.items()},
                 'required': list(params), 'additionalProperties': False}
    return {'type': 'object', 'properties': {
        'skill': {'type': 'string', 'enum': [skill['name'] for skill in skills]},
        'arguments': arguments,
        'notes': {'type': 'string'},
        'thought': {'type': 'string'}},
        'required': ['skill', 'arguments', 'notes', 'thought'], 'additionalProperties': False}


def take_action_description(skills):
    lines = [f"- {s['name']}({', '.join(s['parameters'])}): {s['description']}" for s in skills]
    return ('Run one skill as this decision\'s action. Skills:\n' + '\n'.join(lines) +
            '\nPut the skill\'s arguments in `arguments` and set the ones it does not take to null. arm is 0 (left) or 1 (right). '
            'Also give complete replacement episode notes and a short rationale (thought).')


def _coerce(value, kind):
    if isinstance(value, bool):
        raise ValueError
    if kind == 'integer':
        number = float(value)
        if not number.is_integer():
            raise ValueError
        return int(number)
    if kind == 'number':
        return float(value)
    if not isinstance(value, str):
        raise ValueError
    return value


def normalize_action(skills, skill, arguments):
    """(action, None) for a valid take_action, or (None, error text for the model)."""
    spec = {s['name']: s['parameters'] for s in skills}
    if skill not in spec:
        return None, f"ERROR: unknown skill {skill!r}; choose one of {', '.join(spec)}."
    given = arguments or {}
    out, missing = {}, []
    for name, kind in spec[skill].items():
        if given.get(name) is None:
            missing.append(name)
            continue
        try:
            out[name] = _coerce(given[name], kind)
        except (TypeError, ValueError):
            return None, f'ERROR: {skill} argument {name} must be of type {kind}.'
    if missing:
        return None, f"ERROR: {skill} needs {', '.join(missing)}."
    if 'arm' in out and out['arm'] not in (0, 1):
        return None, 'ERROR: arm must be 0 (left) or 1 (right).'
    return {'skill': skill, 'arguments': out}, None


class Relay:
    """Helper calls to the runner through the runtime (JSON lines on stdout / stdin)."""

    def __init__(self, out=None, inp=None):
        self.out, self.inp, self.n = out or sys.stdout, inp or sys.stdin, 0

    def call(self, tool, args):
        self.n += 1
        self.out.write(json.dumps({'type': 'call', 'id': self.n, 'tool': tool, 'args': args}) + '\n')
        self.out.flush()
        while True:
            line = self.inp.readline()
            if not line:
                raise RuntimeError('runtime closed the helper channel')
            msg = json.loads(line)
            if msg.get('type') == 'result' and msg.get('id') == self.n:
                return str(msg.get('value'))


async def decide(directory, config, context, request, sdk_arm, relay):
    from agents import Agent, FunctionTool, ModelSettings, OpenAIResponsesModel, RunConfig, Runner, RunHooks, set_tracing_disabled
    from agents.agent import ToolsToFinalOutputResult
    from openai import AsyncOpenAI
    set_tracing_disabled(True)
    key = os.environ.get('AGENT_API_KEY') or ''
    redact = (lambda text: text.replace(key, '[REDACTED]')) if key else (lambda text: text)
    skills, helpers = sdk_arm.online_tool_specs()  # TOOL_SCHEMAS descriptions, imported from the SDK arm
    accepted, helper_calls, attempts = {}, [], []
    record = {'status': 'error', 'action': None, 'notes': context.get('notes') or '', 'thought': None, 'route': request['route'],
              'helper_calls': helper_calls, 'attempts': attempts, 'usage': None, 'model': config['model']}

    async def take_action(ctx, args_json):
        if accepted:
            return 'Already accepted; end this turn.'
        try:
            args = json.loads(args_json or '{}')
        except ValueError as exc:
            return f'ERROR: bad arguments ({exc})'
        action, error = normalize_action(skills, args.get('skill'), args.get('arguments'))
        if error:
            return error
        accepted.update(action=action, notes=str(args.get('notes') or '')[:config.get('max_notes_chars', 4000)], thought=str(args.get('thought') or ''))
        return ACCEPTED_TEXT

    def helper_tool(spec):
        properties = {name: {'type': schema_type(kind)} for name, kind in spec['parameters'].items()}
        schema = {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}
        description = spec['description']

        async def invoke(ctx, args_json, _name=spec['name']):
            if len(helper_calls) >= config.get('max_helper_calls', 3):
                return HELPER_BUDGET_TEXT
            try:
                args = json.loads(args_json or '{}')
            except ValueError as exc:
                return f'ERROR: bad arguments ({exc})'
            started = time.monotonic()
            value = relay.call(_name, args)
            helper_calls.append({'tool': _name, 'args': args, 'result': value, 's': round(time.monotonic() - started, 3)})
            return value
        return FunctionTool(name=spec['name'], description=description, params_json_schema=schema, on_invoke_tool=invoke, strict_json_schema=True)

    tools = [FunctionTool(name='take_action', description=take_action_description(skills), params_json_schema=take_action_schema(skills),
                          on_invoke_tool=take_action, strict_json_schema=True)]
    if config.get('helpers', True):
        tools += [helper_tool(spec) for spec in helpers]

    def stop_when_accepted(ctx, results):
        return ToolsToFinalOutputResult(is_final_output=bool(accepted), final_output='Action queued.')

    settings = ModelSettings(reasoning={'effort': config['reasoning_effort']}, max_tokens=config.get('max_output_tokens', 2048),
                             store=False, tool_choice='required', parallel_tool_calls=False)
    started = time.monotonic()
    # The retry window (sets the per-request timeout, at most request_timeout_s); the runtime's real decision deadline caps it.
    budget_s = config.get('decision_timeout_ms', 600000) / 1000.0 - 15.0
    if context.get('decision_deadline_epoch'):  # it also covers the observe render and this process's start
        budget_s = min(budget_s, context['decision_deadline_epoch'] - time.time() - 15.0)
    usage = {'input_tokens': 0, 'output_tokens': 0, 'requests': 0}
    responses = []

    class Count(RunHooks):
        """Every model response of the decision, also those of an attempt that fails later (usage, raw output), and since when
        a model request is in flight (a relay that hangs at the decision deadline is a provider timeout, not a slow model)."""
        since = None

        async def on_llm_start(self, context, agent, system_prompt, input_items):
            self.since = time.monotonic()

        async def on_llm_end(self, context, agent, response):
            self.since = None
            for name in ('input_tokens', 'output_tokens'):
                usage[name] += int(getattr(response.usage, name, 0) or 0)
            usage['requests'] += 1
            responses.append({'id': response.response_id, 'output': [x.model_dump(mode='json') for x in response.output]})
    hooks = Count()
    for attempt in range(1 + int(config.get('decision_retries', 0))):
        remaining = budget_s - (time.monotonic() - started)
        timeout = min(float(config.get('request_timeout_s', 180)), 240.0, max(30.0, remaining / (int(config.get('max_retries', 2)) + 1)))
        entry = {'attempt': attempt, 'timeout_s': round(timeout, 1)}
        attempts.append(entry)
        try:
            async with AsyncOpenAI(api_key=key, base_url=config['base_url'], timeout=timeout, max_retries=int(config.get('max_retries', 2))) as client:
                agent = Agent(name='rea-r1pro-controller', instructions=request['instructions'], tools=tools,
                              tool_use_behavior=stop_when_accepted, reset_tool_choice=False,
                              model=OpenAIResponsesModel(config['model'], client), model_settings=settings)
                run = asyncio.ensure_future(Runner.run(agent, request['inputs'], max_turns=int(config.get('max_turns', 4)), hooks=hooks,
                                                       run_config=RunConfig(tracing_disabled=True)))
                hung = False
                while not run.done() and not hung:
                    await asyncio.wait({run}, timeout=max(0.5, budget_s - (time.monotonic() - started)))
                    # at the deadline, only a request in flight for a full client timeout (an HTTP try already timed out) is
                    # the provider's; a slow but answering model keeps running into the runtime's own deadline
                    hung = (not run.done() and time.monotonic() - started >= budget_s and hooks.since is not None
                            and time.monotonic() - hooks.since >= timeout)
                if hung:
                    run.cancel()
                    await asyncio.gather(run, return_exceptions=True)
                    entry.update(status='provider_error', provider_error='APITimeoutError', error_type='APITimeoutError',
                                 error=f'model request in flight {time.monotonic() - hooks.since:.0f} s (client timeout {timeout:.0f} s, '
                                       f'{int(config.get("max_retries", 2))} retries) at the decision deadline')
                    break
                run.result()
            entry['status'] = 'ok' if accepted else 'model_invalid_action'
            break
        except Exception as exc:  # noqa: BLE001 - every failure is recorded and classified
            label = sdk_arm.provider_failure(exc)
            entry.update(status='provider_error' if label else 'error', error_type=type(exc).__name__, error=redact(str(exc))[:1000])
            if label:
                entry['provider_error'] = label
            if accepted:
                break
            # a provider-labelled failure (also a ModelBehaviorError for response.failed server_error) stays provider_error
            invalid = not label and (type(exc).__name__ in ('MaxTurnsExceeded', 'ModelBehaviorError') or 'max_output_tokens' in str(exc))
            if invalid:
                entry['status'] = 'model_invalid_action'
                break
            if not label or budget_s - (time.monotonic() - started) < 60:
                break
    elapsed = time.monotonic() - started
    helper_s = sum(call['s'] for call in helper_calls)
    last = attempts[-1]
    if accepted:
        record.update(status='ok', action=accepted['action'], notes=accepted['notes'], thought=accepted['thought'])
    else:
        record['status'] = last.get('status', 'error')
        if last.get('provider_error'):
            record['provider_error'] = last['provider_error']
        if last.get('error'):
            record['error'] = f"{last.get('error_type')}: {last['error']}"
    record.update(usage=usage if usage['requests'] else None, responses=responses, elapsed_s=round(elapsed, 3),
                  helper_s=round(helper_s, 3), model_seconds=round(max(0.0, elapsed - helper_s), 3))
    return record


def model_input_record(request, image_path):
    """The model input with the image replaced by its file path (the JPEG itself is in the observations directory)."""
    content = [item if item['type'] != 'input_image' else {'type': 'input_image', 'image_path': image_path, 'detail': item.get('detail')}
               for item in request['inputs'][0]['content']]
    return {'route': request['route'], 'instructions': request['instructions'], 'inputs': [{'role': 'user', 'content': content}]}


def main(directory):
    directory = Path(directory)
    context = json.loads((directory / 'context.json').read_text())
    config = json.loads(Path(context['config_path']).read_text())
    image = context.get('image')
    image_bytes = Path(image).read_bytes() if image and Path(image).exists() else None
    request = prompting.build(config['task_context'], context, context['route'], config, image_bytes)
    (directory / 'model-input.json').write_text(json.dumps(model_input_record(request, image if image_bytes else None), indent=1))
    try:
        record = asyncio.run(decide(directory, config, context, request, load_arm_module(config), Relay()))
    except Exception as exc:  # noqa: BLE001
        key = os.environ.get('AGENT_API_KEY') or ''
        text = f'{type(exc).__name__}: {exc}'
        record = {'status': 'error', 'action': None, 'notes': context.get('notes') or '', 'route': context['route'],
                  'error': text.replace(key, '[REDACTED]') if key else text}
    tmp = directory / 'decision.json.tmp'
    tmp.write_text(json.dumps(record, indent=1, default=str))
    tmp.replace(directory / 'decision.json')


if __name__ == '__main__':
    main(sys.argv[1])
