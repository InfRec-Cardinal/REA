#!/usr/bin/env python3
"""Run independent RoboTwin shards, one supplied configuration per GPU worker."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


VLA_IDS = ("pi05", "gigabrain", "spatial_forcing", "xvla", "xwam", "patchwam", "medex")
TERMINAL_REASONS = {"success", "max_steps", "environment_limit", "terminated", "policy_output_limit", "policy_refusal"}
RUNNER = Path(__file__).with_name("run.mjs")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def check_vla_endpoints(configs, mode):
    """VLA sessions carry episode state, so different workers need different services."""
    owners = {}
    for shard, config in enumerate(configs):
        vla_ids = tuple(config.get("vlaEndpoints", {}))
        candidates = vla_ids if mode in ("hybrid", "vla") else (() if mode == "code" else (mode,))
        endpoints = {config["vlaEndpoints"][candidate].rstrip("/") for candidate in candidates}
        for endpoint in endpoints:
            if endpoint in owners:
                raise ValueError(f"Shards {owners[endpoint]} and {shard} share VLA service {endpoint}")
            owners[endpoint] = shard


def prepare(config_paths, instances_path, output, mode, node="node", skip_errors=False):
    configs = [json.loads(Path(path).read_text()) for path in config_paths]
    instances = json.loads(Path(instances_path).read_text())
    check_vla_endpoints(configs, mode)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "instances.json", instances)
    shards = []
    for index, config in enumerate(configs):
        directory = output / f"shard-{index:02d}"
        directory.mkdir(exist_ok=True)
        selected = instances[index::len(configs)]
        write_json(directory / "config.json", config)
        write_json(directory / "instances.json", selected)
        command = [node, str(RUNNER), "--config", str(directory / "config.json"),
                   "--instances", str(directory / "instances.json"), "--output", str(directory), "--mode", mode]
        if skip_errors:
            command.append("--skip-errors")
        shards.append({"index": index, "output": str(directory), "requested": len(selected),
                       "instanceIds": [row["instance_id"] for row in selected], "command": command})
    plan = {"mode": mode, "requested": len(instances), "workers": len(shards),
            "output": str(output), "shards": shards}
    write_json(output / "plan.json", plan)
    return plan, instances


def summarize(plan, instances, exit_codes=None):
    """Read each original instance's assigned result; pending rows stay in denominators."""
    records = []
    assignments = {instance_id: shard for shard in plan["shards"] for instance_id in shard.get("instanceIds", [])}
    for index, instance in enumerate(instances):
        assigned = assignments.get(instance["instance_id"])
        if assigned is None and plan.get("scheduling") != "dynamic":
            assigned = plan["shards"][index % plan["workers"]]
        shard = assigned["index"] if assigned is not None else None
        result_path = Path(assigned["output"]) / instance["instance_id"] / "result.json" if assigned else None
        try:
            result = json.loads(result_path.read_text()) if result_path else None
        except (FileNotFoundError, json.JSONDecodeError):
            result = None
        status = "pending" if result is None else ("completed" if result.get("reason") in TERMINAL_REASONS else "error")
        records.append({"instance": instance, "shard": shard, "status": status, "result": result})
    workers_finished = exit_codes is not None and len(exit_codes) == plan["workers"] and all(code == 0 for code in exit_codes)
    groups = []
    for setting in ("clean", "randomized", "all"):
        selected = [row for row in records if setting == "all" or row["instance"].get("setting") == setting]
        completed = sum(row["status"] == "completed" for row in selected)
        errors = sum(row["status"] == "error" for row in selected)
        pending = len(selected) - completed - errors
        successes = sum(row["status"] == "completed" and row["result"].get("success") is True for row in selected)
        groups.append({"setting": setting, "total": len(selected), "completed": completed,
                       "errors": errors, "pending": pending, "successes": successes,
                       "successRate": successes / completed if completed else None,
                       "fullPanelSuccessRate": successes / len(selected)
                       if workers_finished and completed == len(selected) and selected else None})
    return {"mode": plan["mode"], "requested": len(instances),
            "processed": sum(row["status"] != "pending" for row in records),
            "workers": plan["workers"], "exitCodes": exit_codes,
            "groups": groups, "results": records}


def run(plan, instances, skip_errors=False):
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    processes, logs = [], []
    exit_codes = [None] * plan["workers"]

    def save_summary():
        summary = summarize(plan, instances, exit_codes)
        summary.update(startedAt=started_at, elapsedSeconds=round(time.monotonic() - started, 3))
        summary["queueExhausted"] = all(code == 0 for code in exit_codes) and not summary["groups"][-1]["pending"]
        summary["skippedErrors"] = [{"instance_id": row["instance"]["instance_id"], "shard": row["shard"],
                                     "reason": row["result"].get("reason")}
                                    for row in summary["results"] if skip_errors and row["status"] == "error"]
        summary["status"] = ("completed_with_errors" if summary["skippedErrors"] else "completed") if summary["queueExhausted"] else "running"
        write_json(Path(plan["output"]) / "combined-summary.json", summary)

    save_summary()
    try:
        for shard in plan["shards"]:
            log = open(Path(shard["output"]) / "run.log", "a", buffering=1)
            logs.append(log)
            command = shard["command"]
            if skip_errors and "--skip-errors" not in command:
                command = [*command, "--skip-errors"]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        while any(code is None for code in exit_codes):
            exit_codes = [process.poll() for process in processes]
            save_summary()
            if any(code is None for code in exit_codes):
                time.sleep(5)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for index, process in enumerate(processes):
            exit_codes[index] = process.wait()
        for log in logs:
            log.close()
        save_summary()
    return 0 if all(code == 0 for code in exit_codes) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True, help="One config JSON per worker")
    parser.add_argument("--instances", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="hybrid", help="hybrid, vla, code, or one VLA id from config.vlaEndpoints")
    parser.add_argument("--node", default="node")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    args = parser.parse_args()
    plan, instances = prepare(args.configs, args.instances, args.output, args.mode, args.node, args.skip_errors)
    print(json.dumps({"plan": str(Path(plan["output"]) / "plan.json"), "workers": plan["workers"],
                      "requested": plan["requested"]}), flush=True)
    if args.prepare_only:
        write_json(Path(plan["output"]) / "combined-summary.json", summarize(plan, instances))
        return 0
    return run(plan, instances, skip_errors=args.skip_errors)


if __name__ == "__main__":
    raise SystemExit(main())
