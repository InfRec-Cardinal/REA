import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[2]


def load_server(platform):
    source = Path(os.environ["RPENT_ROOT"]) / "robots" / platform / "env_server.py"
    spec = importlib.util.spec_from_file_location("fast_rpent_physics_server", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_smoke(platform):
    sys.path.insert(0, os.environ["RPENT_ROOT"])
    module = load_server(platform)
    if platform == "libero":
        native_env = module.make_env(task_id=0, seed=0, suite_name="libero_spatial", max_episode_steps=20)
        facade = module.LiberoEnvFacade(native_env, meta={"suite": "libero_spatial", "task": 0, "seed": 0, "max_episode_steps": 20})
    else:
        facade = module.RoboCasaEnvFacade(task_name=os.environ.get("REA_RPENT_ROBOCASA_TASK", "OpenDrawer"), seed=0, camera_h=64, camera_w=64)
    options = {
        "robot": platform,
        "python": sys.executable,
        "rpentRoot": os.environ["RPENT_ROOT"],
        "imageHeight": 64,
        "imageWidth": 64,
        "shutdownOnClose": True,
    }
    ready = threading.Event()
    bind = facade._bind_and_announce

    def bind_and_record(*args, **kwargs):
        server = bind(*args, **kwargs)
        options["endpoint"] = f"http://127.0.0.1:{server.server_address[1]}"
        ready.set()
        return server

    facade._bind_and_announce = bind_and_record
    script = """
import assert from 'node:assert/strict';
import {createRPentEnvironment} from './src/environments/rpent.mjs';
const options = JSON.parse(process.env.REA_RPENT_TEST_OPTIONS);
const environment = createRPentEnvironment({...options, execute: true});
try {
  const initial = await environment.reset({seed: 0});
  assert(initial.observation.images.length > 0);
  assert.equal(initial.terminated, false);
  const size = options.robot === 'libero' ? 7 : 12;
  const result = await environment.step({name: 'rpent_action', arguments: {values: Array(size).fill(0)}});
  assert.equal(result.info.controlSteps, 1);
  assert.equal(result.info.preview, false);
  assert(result.observation.images.every(image => image.name && image.mimeType === 'image/png' && image.data.length > 100));
  assert(Object.keys(result.observation.state).length > 0);
  const trajectory = await environment.step({name: 'rpent_trajectory', arguments: {commands: [Array(size).fill(0), Array(size).fill(0)]}});
  assert.equal(trajectory.info.controlSteps, 2);
  assert.equal(trajectory.info.episodeControlSteps, 3);
  process.stdout.write(JSON.stringify({platform: options.robot, reset: true, step: true, transport: 'http', bridge: 'node-python', images: result.observation.images.map(image => image.name), controlSteps: result.info.controlSteps, trajectoryControlSteps: trajectory.info.controlSteps, reward: trajectory.reward, terminated: trajectory.terminated, truncated: trajectory.truncated, success: trajectory.success}) + '\\n');
} finally {
  await environment.close();
}
"""
    results = []

    def run_node():
        ready.wait()
        try:
            results.append(subprocess.run(
                [os.environ.get("RPENT_NODE", "node"), "--input-type=module", "-e", script],
                cwd=ROOT, env={**os.environ, "REA_RPENT_TEST_OPTIONS": json.dumps(options)},
                capture_output=True, text=True, timeout=180,
            ))
        except Exception as error:
            results.append(error)
        finally:
            facade._shutdown_event.set()

    thread = threading.Thread(target=run_node, daemon=True)
    thread.start()
    facade.serve(transport="http", host="127.0.0.1", port=0)
    thread.join(timeout=5)
    result = results[0]
    if isinstance(result, Exception):
        raise result
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("platform", choices=["libero", "robocasa"])
    arguments = parser.parse_args()
    print(json.dumps(run_smoke(arguments.platform)), flush=True)
