"""Local stand-in for the Responses API relay (no model) for the BEHAVIOR SDK-arm tests. Prints "ready <port>" once listening.

MODE (env) is a comma list, one entry per request in order (the last entry repeats):
  tool      a get_robot_position function call
  final     an assistant message (the agent stops)
  cut       status incomplete, incomplete_details.reason max_output_tokens
  slow      sleeps 5 s, then a get_robot_position function call
  http400   HTTP 400 {"error": {"message": "content refused"}}
  http408   HTTP 408 {"error": {"message": "upstream request timeout"}}
  null      HTTP 200 with the body `null`
Every request body is appended to BODIES (JSON lines)."""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODES = os.environ.get("MODE", "final").split(",")
BODIES = os.environ.get("BODIES")
lock = threading.Lock()
count = {"n": 0}


def response(n, output, status="completed", incomplete=None):
    return {"id": f"resp_{n}", "object": "response", "created_at": int(time.time()), "status": status, "model": "gpt-6-astra",
            "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "error": None,
            "incomplete_details": incomplete, "instructions": None, "metadata": {}, "temperature": 1.0, "top_p": 1.0,
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 10}}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, data):
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        with lock:
            count["n"] += 1
            n = count["n"]
            if BODIES:
                with open(BODIES, "a") as f:
                    f.write(json.dumps({"n": n, "body": body}) + "\n")
        mode = MODES[min(n, len(MODES)) - 1]
        call = {"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n}", "name": "get_robot_position", "arguments": "{}",
                "status": "completed"}
        if mode == "slow":
            time.sleep(5)
        if mode in ("tool", "slow"):
            return self.reply(200, json.dumps(response(n, [call])).encode())
        if mode == "cut":
            reasoning = {"type": "reasoning", "id": f"rs_{n}", "summary": []}
            return self.reply(200, json.dumps(response(n, [reasoning], "incomplete", {"reason": "max_output_tokens"})).encode())
        if mode == "http400":
            return self.reply(400, json.dumps({"error": {"message": "content refused", "type": "invalid_request_error"}}).encode())
        if mode == "http408":
            return self.reply(408, json.dumps({"error": {"message": "upstream request timeout", "type": "timeout"}}).encode())
        if mode == "null":
            return self.reply(200, b"null")
        message = {"type": "message", "id": f"msg_{n}", "role": "assistant", "status": "completed",
                   "content": [{"type": "output_text", "text": "done", "annotations": []}]}
        return self.reply(200, json.dumps(response(n, [message])).encode())


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
server.daemon_threads = True
print("ready", server.server_address[1], flush=True)
server.serve_forever()
