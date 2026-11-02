#!/usr/bin/env python3
"""Dispatch code episodes from a shared scene queue with expandable worker slots."""

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from parallel import RUNNER, TERMINAL_REASONS, summarize


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def configure_worker(shard, config_path, mode, node, runner):
    directory = Path(shard["output"])
    config = json.loads(Path(config_path).read_text())
    write_json(directory / "config.json", config)
    shard["command"] = [str(node), str(runner), "--config", str(directory / "config.json"),
                        "--instances", str(directory / "current-instance.json"),
                        "--output", str(directory), "--mode", mode]


def extend_workers(plan, config_paths, node, runner):
    if len(config_paths) < plan["workers"]:
        raise ValueError("An existing dynamic queue cannot shrink its worker count")
    added = []
    for index in range(plan["workers"], len(config_paths)):
        directory = Path(plan["output"]) / f"shard-{index:02d}"
        directory.mkdir(exist_ok=True)
        shard = {"index": index, "output": str(directory), "requested": 0, "instanceIds": []}
        configure_worker(shard, config_paths[index], plan["mode"], node, runner)
        plan["shards"].append(shard)
        added.append(shard)
    plan["workers"] = len(config_paths)
    return added


def prepare(config_paths, instances_path, output, mode="code", node="node", runner=RUNNER):
    if mode not in ("code", "hybrid"):
        raise ValueError("Dynamic queue supports code and the hybrid router's code group")
    instances = json.loads(Path(instances_path).read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan.get("scheduling") != "dynamic" or plan["mode"] != mode:
            raise ValueError("Resume requires a dynamic code plan in this output directory")
        if json.loads((output / "instances.json").read_text()) != instances:
            raise ValueError("Resume must retain the same scene manifest")
    else:
        plan = {"mode": mode, "requested": len(instances), "workers": 0,
                "output": str(output), "scheduling": "dynamic", "shards": []}
    extend_workers(plan, config_paths, node, runner)
    for shard, path in zip(plan["shards"], config_paths):
        configure_worker(shard, path, mode, node, runner)
    write_json(output / "instances.json", instances)
    write_json(plan_path, plan)
    return plan, instances


def completed(shard, instance, mode="code"):
    path = Path(shard["output"]) / instance["instance_id"] / "result.json"
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    if result.get("instance") != instance or result.get("mode") != mode:
        raise ValueError(f"Existing result has a different scene or mode: {instance['instance_id']}")
    return result.get("reason") in TERMINAL_REASONS and type(result.get("success")) is bool


def run(plan, instances, poll_seconds=1, lock_fd=None, configs_file=None, skip_errors=False):
    root = Path(plan["output"])
    by_id = {row["instance_id"]: row for row in instances}
    assignments = {instance_id: shard for shard in plan["shards"] for instance_id in shard["instanceIds"]}
    pending = deque(row for row in instances if row["instance_id"] not in assignments)
    skipped = {}

    def skip(shard, instance):
        instance_id = instance["instance_id"]
        path = Path(shard["output"]) / instance_id / "result.json"
        reason = json.loads(path.read_text()).get("reason", "infrastructure_error") if path.exists() else "missing_result"
        skipped[instance_id] = {"instance_id": instance_id, "shard": shard["index"], "reason": reason}

    # The explicit skip mode preserves every previous attempt, including interrupted
    # assignments without a result. Only scenes never assigned enter the shared queue.
    retries = {shard["index"]: deque() for shard in plan["shards"]}
    for shard in plan["shards"]:
        for instance_id in shard["instanceIds"]:
            instance = by_id[instance_id]
            if not completed(shard, instance, plan["mode"]):
                if skip_errors:
                    skip(shard, instance)
                else:
                    retries[shard["index"]].append(instance)
    states = [{"index": shard["index"], "pid": None, "instanceId": None, "launches": 0,
               "completedThisRun": 0, "exitCode": None} for shard in plan["shards"]]
    processes, logs = {}, {}
    previous = json.loads((root / "combined-summary.json").read_text()) if (root / "combined-summary.json").exists() else {}
    started_at = previous.get("startedAt", datetime.now(timezone.utc).isoformat())
    stopped = False
    configs_file = Path(configs_file) if configs_file else None
    configs_mtime = None

    def save():
        summary = summarize(plan, instances, [state["exitCode"] for state in states])
        for record in summary["results"]:
            instance_id = record["instance"]["instance_id"]
            if instance_id in skipped:
                record.update(status="error", skipReason=skipped[instance_id]["reason"])
            if record["status"] == "error" and record["result"]:
                # Full provider diagnostics remain in the original result, not in summaries.
                record["result"] = {key: record["result"].get(key)
                                    for key in ("instance", "mode", "reason", "success")}
        for group in summary["groups"]:
            records = [record for record in summary["results"]
                       if group["setting"] == "all" or record["instance"].get("setting") == group["setting"]]
            group["errors"] = sum(record["status"] == "error" for record in records)
            group["pending"] = sum(record["status"] == "pending" for record in records)
            if group["errors"]:
                group["fullPanelSuccessRate"] = None
        summary["processed"] = sum(record["status"] != "pending" for record in summary["results"])
        remaining = len(pending) + sum(len(queue) for queue in retries.values())
        exhausted = remaining == 0 and not processes
        summary.update(startedAt=started_at, supervisorPid=os.getpid(), scheduling="dynamic", workerStates=states,
                       queueRemaining=remaining, queueExhausted=exhausted, skipErrors=skip_errors,
                       skippedErrors=list(skipped.values()), skippedErrorIds=list(skipped),
                       status=("paused_error" if stopped else "completed_with_errors" if exhausted and skipped
                               else "completed" if exhausted else "running"),
                       elapsedSeconds=round((datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds(), 3))
        write_json(root / "combined-summary.json", summary)

    def launch(shard, state, instance):
        index, instance_id = shard["index"], instance["instance_id"]
        if instance_id not in assignments:
            assignments[instance_id] = shard
            shard["instanceIds"].append(instance_id)
            shard["requested"] = len(shard["instanceIds"])
        write_json(root / "plan.json", plan)
        write_json(Path(shard["output"]) / "current-instance.json", [instance])
        log = open(Path(shard["output"]) / "run.log", "a", buffering=1)
        process = subprocess.Popen(shard["command"], stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=() if lock_fd is None else (lock_fd,))
        processes[index], logs[index] = process, log
        state.update(pid=process.pid, instanceId=instance_id, launches=state["launches"] + 1, exitCode=None)

    save()
    try:
        while True:
            if configs_file and not stopped:
                current_mtime = configs_file.stat().st_mtime_ns
                if current_mtime != configs_mtime:
                    config_paths = json.loads(configs_file.read_text())
                    node, runner = plan["shards"][0]["command"][:2]
                    for shard in extend_workers(plan, config_paths, node, runner):
                        retries[shard["index"]] = deque()
                        states.append({"index": shard["index"], "pid": None, "instanceId": None,
                                       "launches": 0, "completedThisRun": 0, "exitCode": None})
                    write_json(root / "plan.json", plan)
                    configs_mtime = current_mtime
            for shard, state in zip(plan["shards"], states):
                index = shard["index"]
                process = processes.get(index)
                if process is None or process.poll() is None:
                    continue
                code = process.wait()
                del processes[index]
                logs.pop(index).close()
                instance = by_id[state["instanceId"]]
                state["pid"] = None
                if code == 0 and completed(shard, instance, plan["mode"]):
                    state.update(completedThisRun=state["completedThisRun"] + 1, instanceId=None)
                elif skip_errors:
                    skip(shard, instance)
                    state.update(instanceId=None, exitCode=None, lastSkippedInstance=instance["instance_id"],
                                 lastWorkerExitCode=code or 1)
                else:
                    stopped = True
                    state.update(exitCode=code or 1, error="Episode failed outside valid policy outcomes; queue paused without retry")
                write_json(Path(shard["output"]) / "instances.json", [by_id[key] for key in shard["instanceIds"]])
            if not stopped:
                for shard, state in zip(plan["shards"], states):
                    index = shard["index"]
                    if index in processes:
                        continue
                    instance = retries[index].popleft() if retries[index] else pending.popleft() if pending else None
                    if instance is not None:
                        launch(shard, state, instance)
                    else:
                        state["exitCode"] = 0
            save()
            if not processes:
                break
            time.sleep(poll_seconds)
    finally:
        for index, process in processes.items():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            logs[index].close()
            states[index].update(pid=None, exitCode=process.returncode or 1)
        save()
    return 1 if stopped or any(state["exitCode"] not in (0,) for state in states) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    configs = parser.add_mutually_exclusive_group(required=True)
    configs.add_argument("--configs", nargs="+")
    configs.add_argument("--configs-file", type=Path,
                         help="JSON list of config paths; append entries to add workers while running")
    parser.add_argument("--instances", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("code", "hybrid"), default="code")
    parser.add_argument("--node", default="node")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-errors", action="store_true",
                        help="Preserve and skip failed/interrupted scenes, continuing only unattempted scenes")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This output already has an active queue or episode worker") from None
        config_paths = json.loads(args.configs_file.read_text()) if args.configs_file else args.configs
        plan, instances = prepare(config_paths, args.instances, root, args.mode, args.node)
        print(json.dumps({"plan": str(root / "plan.json"), "workers": plan["workers"],
                          "requested": plan["requested"], "scheduling": "dynamic"}), flush=True)
        if args.prepare_only:
            write_json(root / "combined-summary.json", summarize(plan, instances))
            return 0
        return run(plan, instances, lock_fd=lock.fileno(), configs_file=args.configs_file, skip_errors=args.skip_errors)


if __name__ == "__main__":
    raise SystemExit(main())
