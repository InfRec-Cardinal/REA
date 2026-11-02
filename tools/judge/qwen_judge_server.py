#!/usr/bin/env python3
"""Local OpenAI-compatible chat endpoint serving Qwen3-VL as the visual judge (no gateway dependency).

Implements POST /v1/chat/completions (non-streaming) for the subset src/infer-judge.mjs sends: a system
message, one user message with text parts and image_url data URLs, `max_tokens`, `temperature` (greedy
decoding is always used). Returns choices[0].message.content plus a usage block so the runner's cost
accounting stays defined. One model instance; requests are serialised on the GPU.

  python qwen_judge_server.py --model /path/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port 8130 [--device cuda:0]

Health: GET /healthz -> {"ok": true, "model": ...}. Stdlib only besides torch/transformers/Pillow.
"""

import argparse
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image


def load_model(path, device, dtype):
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as Cls
    except ImportError:  # older transformers
        from transformers import AutoModelForImageTextToText as Cls
    # no device_map: it needs `accelerate`, which the shared simulator venv does not ship; load on CPU then move
    model = Cls.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa").to(device)
    model.eval()
    return model, AutoProcessor.from_pretrained(path)


def decode_image(url):
    if not url.startswith("data:") or ";base64," not in url:
        raise ValueError("only data:image/...;base64 image URLs are supported")
    return Image.open(io.BytesIO(base64.b64decode(url.split(";base64,", 1)[1]))).convert("RGB")


def to_chat(messages):
    """OpenAI messages -> (qwen chat messages, PIL images in order)."""
    chat, images = [], []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, str):
            chat.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        parts = []
        for item in content or []:
            if item.get("type") == "text":
                parts.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                images.append(decode_image(item.get("image_url", {}).get("url", "")))
                parts.append({"type": "image"})
        chat.append({"role": role, "content": parts})
    return chat, images


class Judge:
    def __init__(self, path, device, dtype):
        self.path = path
        self.device = device
        self.model, self.processor = load_model(path, device, dtype)
        self.lock = threading.Lock()
        self.calls = 0

    def complete(self, messages, max_tokens):
        chat, images = to_chat(messages)
        text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to(self.device)
        with self.lock, torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        completion = out[:, prompt_tokens:]
        content = self.processor.batch_decode(completion, skip_special_tokens=True)[0]
        self.calls += 1
        return content, prompt_tokens, int(completion.shape[1])


def make_handler(judge, model_name):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status, body):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") in ("/healthz", "/v1/models"):
                self._send(200, {"ok": True, "model": model_name, "calls": judge.calls, "object": "list", "data": [{"id": model_name, "object": "model"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length))
                if payload.get("stream"):
                    raise ValueError("streaming is not supported by this judge server")
                max_tokens = int(payload.get("max_tokens") or 64)
                started = time.time()
                content, prompt_tokens, completion_tokens = judge.complete(payload.get("messages", []), max_tokens)
                self._send(200, {
                    "id": f"chatcmpl-local-{judge.calls}", "object": "chat.completion", "created": int(started), "model": model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": "length" if completion_tokens >= max_tokens else "stop"}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
                    "latency_s": round(time.time() - started, 3),
                })
            except Exception as error:  # noqa: BLE001
                self._send(400, {"error": {"message": f"{type(error).__name__}: {error}", "type": "judge_server_error"}})

        def log_message(self, format, *args):  # noqa: A002
            print(f"[judge] {format % args}", flush=True)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", default=None, help="name reported to clients (default: basename of --model)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()
    name = args.model_name or os.path.basename(args.model.rstrip("/"))
    t0 = time.time()
    judge = Judge(args.model, args.device, getattr(torch, args.dtype))
    print(f"[judge] loaded {name} on {args.device} in {time.time() - t0:.1f}s; serving http://{args.host}:{args.port}/v1/chat/completions", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(judge, name)).serve_forever()


if __name__ == "__main__":
    main()
