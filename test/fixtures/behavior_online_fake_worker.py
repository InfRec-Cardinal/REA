"""Scripted stand-in for eval/behavior/online/worker.py (no model): one decision per call, read from the JSON script named by
REA_FAKE_SCRIPT ({"<decision>": {"helpers": [[tool, args], ...], "action": {...} | null, "status": "ok", "think_s": 0, ...}}).

It speaks the worker's helper protocol on stdout / stdin, builds the real prompt with prompting.build (so the test sees the
model input the online agent would send), and writes decision.json like the real worker. The prompt text goes to seen.json."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location('online_behavior_prompting', Path(__file__).resolve().parents[2] / 'eval' / 'behavior' / 'online' / 'prompting.py')
prompting = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prompting)

directory = Path(sys.argv[1])
context = json.loads((directory / 'context.json').read_text())
config = json.loads(Path(context['config_path']).read_text())
step = json.loads(Path(os.environ['REA_FAKE_SCRIPT']).read_text()).get(str(context['decision']), {'action': None, 'status': 'model_invalid_action'})
(directory / 'pid').write_text(str(os.getpid()))
image = context.get('image')
request = prompting.build(config['task_context'], context, context['route'], config, Path(image).read_bytes() if image else None)
helper_calls = []
for n, (tool, args) in enumerate(step.get('helpers', []), start=1):
    sys.stdout.write(json.dumps({'type': 'call', 'id': n, 'tool': tool, 'args': args}) + '\n')
    sys.stdout.flush()
    while True:
        message = json.loads(sys.stdin.readline())
        if message.get('type') == 'result' and message.get('id') == n:
            helper_calls.append({'tool': tool, 'args': args, 'result': message['value']})
            break
(directory / 'seen.json').write_text(json.dumps({'instructions': request['instructions'],
                                                 'texts': [item['text'] for item in request['inputs'][0]['content'] if item['type'] == 'input_text'],
                                                 'images': sum(item['type'] == 'input_image' for item in request['inputs'][0]['content']),
                                                 'helper_results': helper_calls, 'key_seen': bool(os.environ.get('AGENT_API_KEY'))}))
time.sleep(float(step.get('think_s', 0)))  # model time
decision = {'status': step.get('status', 'ok'), 'action': step.get('action'), 'notes': f"notes after decision {context['decision']}",
            'thought': f"thought {context['decision']}", 'helper_calls': helper_calls, 'route': context['route'],
            'usage': {'input_tokens': 100, 'output_tokens': 10, 'requests': 1 + len(helper_calls)}, 'model_seconds': 0.01}
if step.get('provider_error'):
    decision.update(provider_error=step['provider_error'], error='APIConnectionError: Connection error.')
(directory / 'decision.json').write_text(json.dumps(decision))
