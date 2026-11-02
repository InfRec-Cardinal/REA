#!/usr/bin/env python3
"""Serve a trained RoboTwin success router through the REA policy interface."""

import argparse
import base64
import importlib.util
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import time
import traceback
from types import SimpleNamespace


CAMERAS = {
    "head_camera": ("head_camera", "head"),
    "left_camera": ("left_camera", "left_wrist"),
    "right_camera": ("right_camera", "right_wrist"),
}
OUTPUT_CONTRACT = {
    "modality": "robot_action_sequence",
    "embodiment": "aloha-agilex",
    "accepted_spaces": ["joint14", "ee16"],
    "frame": "RoboTwin world",
    "joint_units": "radian; normalized gripper",
    "ee_units": "meter; quaternion wxyz; normalized gripper",
    "stop": "official success or official task step limit",
    "modalities": ["robot_action_sequence"],
}


def robot_state_text(observation):
    state = observation.get("state", {})
    robot_state = state.get("vla", state.get("robot_state", state))
    for source in (robot_state, state, observation):
        if isinstance(source.get("state_text"), str):
            return source["state_text"]
    endpose = robot_state.get("endpose")
    joint = robot_state.get("joint")
    if joint is None:
        joint = robot_state.get("joint_action", {}).get("vector")
    if joint is None:
        joint = robot_state.get("qpos_target14")
    if endpose is None or joint is None:
        raise ValueError("RoboTwin router requires endpose and joint14 in observation.state.")
    ordered_endpose = {
        key: endpose[key]
        for key in ("left_endpose", "left_gripper", "right_endpose", "right_gripper")
    }
    return json.dumps({"endpose": ordered_endpose, "joint": joint}, ensure_ascii=False)


def input_record(payload):
    observation = payload["observation"]
    task_text = observation.get("text") or payload.get("instruction")
    if not isinstance(task_text, str) or not task_text.strip():
        raise ValueError("RoboTwin router requires the task instruction.")
    by_name = {image["name"]: image for image in observation.get("images", [])}
    images = []
    for camera, aliases in CAMERAS.items():
        image = next((by_name[name] for name in aliases if name in by_name), None)
        if image is None:
            raise ValueError(f"RoboTwin router requires camera {camera}.")
        images.append(io.BytesIO(base64.b64decode(image["data"])))
    return {
        "id": observation.get("id", str(payload.get("episodeId", "online"))),
        "task_text": task_text,
        "images": images,
        "state_text": robot_state_text(observation),
        "output_contract": dict(OUTPUT_CONTRACT),
    }


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_paths(training_root=None):
    """Use bundled encoding by default, or the sources matching an external checkpoint."""
    if training_root is None:
        code = Path(__file__).resolve().parent / "success_router"
        return code / "model.py", code / "encoding.py"
    root = Path(training_root).resolve()
    code = root / "code" if (root / "code/model.py").exists() else root
    return code / "model.py", code / "train.py"


class CheckpointPredictor:
    def __init__(self, training_root, checkpoint, model_path=None, device="cuda"):
        import torch

        checkpoint = Path(checkpoint).resolve()
        model_source_path, encoding_source_path = source_paths(training_root)
        config = json.loads((checkpoint / "router_config.json").read_text())
        plan = json.loads((checkpoint / "training_plan.json").read_text())
        arguments = plan["arguments"]
        self.model_ids = config["model_ids"]
        self.device = torch.device(device)
        self.torch = torch
        self.training = load_source("rea_success_training", encoding_source_path)
        model_source = load_source("rea_success_model", model_source_path)
        self.args = SimpleNamespace(
            model_path=model_path or config["metadata"]["model_path"],
            processor_path=str(checkpoint / "processor"),
            chat_template_path=None,
            seq_length=arguments["seq_length"],
            max_image_tokens=arguments["max_image_tokens"],
            image_token_budget=arguments.get("image_token_budget", 2048),
        )
        self.processor = self.training.load_processor(self.args)
        self.model = model_source.load_model(
            self.args.model_path, self.model_ids, self.device, arguments["lora_r"],
        )
        self.model.restore_weights(checkpoint)
        self.model.eval()
        self.model.encoder.gradient_checkpointing_disable()
        self.metadata = {
            "schema_version": config["schema_version"],
            "checkpoint": str(checkpoint),
            "model_path": self.args.model_path,
            "training_root": str(model_source_path.parent),
            "training": config["metadata"],
            "training_input_stage": "initial_observation",
            "input_fields": ["task_text", "images", "state_text", "output_contract"],
            "camera_order": list(CAMERAS),
            "output_contract": OUTPUT_CONTRACT,
            "score_type": "independent_sigmoid_success_probability",
        }

    def __call__(self, row):
        with self.torch.inference_mode():
            logits, encoding = self.training.forward_row(
                self.model, row, self.processor, self.args, self.device,
            )
            probabilities = logits.float().sigmoid()[0].cpu().tolist()
        return dict(zip(self.model_ids, probabilities)), encoding


class RouterService:
    def __init__(self, predictor, model_ids, metadata):
        self.predictor = predictor
        self.model_ids = list(model_ids)
        self.metadata = metadata

    def health(self):
        return {"status": "ready", "trained_candidates": self.model_ids, "metadata": self.metadata}

    def predict(self, payload):
        started = time.monotonic()
        candidates = payload.get("candidates")
        if candidates is None:
            candidates = [{"id": model_id, "kind": "vla"} for model_id in self.model_ids]
        trained = [candidate["id"] for candidate in candidates if candidate["id"] in self.model_ids]
        untrained = [
            {"id": candidate["id"], "kind": candidate.get("kind"), "reason": "no_trained_head"}
            for candidate in candidates if candidate["id"] not in self.model_ids
        ]
        scores, encoding = {}, None
        if trained:
            probabilities, encoding = self.predictor(input_record(payload))
            scores = {model_id: probabilities[model_id] for model_id in trained}
        return {
            "scores": scores,
            "trained_candidates": self.model_ids,
            "untrained_candidates": untrained,
            "selected_candidate": max(scores, key=scores.get) if scores else None,
            "episodeId": payload.get("episodeId"),
            "step": payload.get("step"),
            "encoding": encoding,
            "inference_seconds": time.monotonic() - started,
            "metadata": self.metadata,
        }


def make_server(service, host, port):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, service.health())
            else:
                self.respond(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if self.path != "/predict":
                self.respond(404, {"error": "Unknown endpoint"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                result = service.predict(payload)
            except (KeyError, TypeError, ValueError) as error:
                self.respond(400, {"error": str(error)})
            except Exception as error:
                traceback.print_exc()
                self.respond(500, {"error": str(error)})
            else:
                self.respond(200, result)

    return HTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path,
                        help="Optional external model.py/train.py directory; defaults to bundled router sources")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8810)
    args = parser.parse_args()
    predictor = CheckpointPredictor(args.training_root, args.checkpoint, args.model_path, args.device)
    service = RouterService(predictor, predictor.model_ids, predictor.metadata)
    with make_server(service, args.host, args.port) as server:
        print(json.dumps({"event": "ready", "host": args.host, "port": server.server_port,
                          **service.health()}, ensure_ascii=False), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
