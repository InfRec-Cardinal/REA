#!/usr/bin/env python3
"""Export a complete code-policy run as observed success labels for router training."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from success_router_server import OUTPUT_CONTRACT


COMPLETED_REASONS = {"success", "max_steps", "environment_limit", "terminated", "policy_output_limit", "policy_refusal"}


def episode_directories(run_root, instances):
    """Resolve parallel.py assignments, independently of shard or filesystem order."""
    plan_path = run_root / "plan.json"
    if not plan_path.exists():
        return {row["instance_id"]: run_root / row["instance_id"] for row in instances}
    plan = json.loads(plan_path.read_text())
    directories = {}
    for shard in plan["shards"]:
        directory = run_root / f"shard-{shard['index']:02d}"
        for instance_id in shard["instanceIds"]:
            if instance_id in directories:
                raise ValueError(f"Episode assigned to multiple shards: {instance_id}")
            directories[instance_id] = directory / instance_id
    return directories


def export_labels(run_root, output, vla_run_dir=None, instances_path=None, allow_partial=False):
    run_root, output = Path(run_root).resolve(), Path(output)
    vla_run_dir = Path(vla_run_dir).resolve() if vla_run_dir else None
    instances = json.loads((run_root / "instances.json").read_text())
    if instances_path:
        original = {row["instance_id"]: row for row in instances}
        instances = json.loads(Path(instances_path).read_text())
        if len({row["instance_id"] for row in instances}) != len(instances):
            raise ValueError("Selected instances contain duplicate episode IDs")
        for instance in instances:
            if original.get(instance["instance_id"]) != instance:
                raise ValueError(f"Selected instance does not match the run manifest: {instance['instance_id']}")
    code_directories = episode_directories(run_root, instances)
    vla_directories = episode_directories(vla_run_dir, instances) if vla_run_dir else None
    labels, missing = [], []
    for instance in instances:
        directory = code_directories.get(instance["instance_id"])
        if directory is None:
            missing.append({"instance_id": instance["instance_id"], "reason": "missing_code_shard_assignment"})
            continue
        result_path = directory / "result.json"
        input_path = directory / "initial-router-input.json"
        absent = [path.name for path in (result_path, input_path) if not path.exists()]
        if absent:
            missing.append({"instance_id": instance["instance_id"], "reason": "missing_files", "files": absent})
            continue
        result = json.loads(result_path.read_text())
        reason = result.get("reason")
        if result.get("mode") != "code" or reason not in COMPLETED_REASONS or type(result.get("success")) is not bool:
            missing.append({
                "instance_id": instance["instance_id"], "reason": "no_completed_code_outcome",
                "mode": result.get("mode"), "termination": reason, "success": result.get("success"),
            })
            continue
        if any(result["instance"][key] != instance[key] for key in ("task", "setting", "seed")):
            raise ValueError(f"Episode result does not match the requested scene: {instance['instance_id']}")
        original = json.loads(input_path.read_text())
        images = [str((directory / path).resolve()) for path in original["images"]]
        absent_images = [path for path in images if not Path(path).is_file()]
        if absent_images:
            missing.append({"instance_id": instance["instance_id"], "reason": "missing_images", "files": absent_images})
            continue
        label = {
            "instance": instance,
            "input": {
                "task_text": original["task_text"],
                "images": images,
                "state_text": original["state_text"],
                "output_contract": original.get("output_contract", OUTPUT_CONTRACT),
            },
            "code_success": result["success"],
            "policy": {
                "model": result["codeModel"], "protocol": result["protocol"],
                **({"provider": result["codeProvider"]} if "codeProvider" in result else {}),
                **({"code_protocol": result["codeProtocol"]} if "codeProtocol" in result else {}),
            },
        }
        if vla_run_dir:
            vla_directory = vla_directories.get(instance["instance_id"])
            if vla_directory is None:
                missing.append({"instance_id": instance["instance_id"], "reason": "missing_vla_shard_assignment"})
                continue
            vla_result_path = vla_directory / "result.json"
            decisions_path = vla_directory / "decisions.jsonl"
            absent = [path.name for path in (vla_result_path, decisions_path) if not path.exists()]
            if absent:
                missing.append({"instance_id": instance["instance_id"], "reason": "missing_vla_files", "files": absent})
                continue
            vla_result = json.loads(vla_result_path.read_text())
            decisions = [json.loads(line) for line in decisions_path.read_text().splitlines() if line.strip()]
            if (vla_result.get("mode") != "vla" or vla_result.get("reason") not in COMPLETED_REASONS
                    or type(vla_result.get("success")) is not bool or not decisions):
                missing.append({"instance_id": instance["instance_id"], "reason": "no_completed_vla_outcome",
                                "termination": vla_result.get("reason")})
                continue
            if any(vla_result["instance"][key] != instance[key] for key in ("task", "setting", "seed")):
                raise ValueError(f"VLA result does not match the code scene: {instance['instance_id']}")
            label.update(vla_selected_model=decisions[0]["candidate"], vla_success=vla_result["success"])
        labels.append(label)
    report = {
        "status": "completed_with_errors" if missing and allow_partial and labels else "incomplete" if missing or not labels else "completed",
        "run_root": str(run_root), "expected": len(instances),
        "completed_labels": len(labels), "missing": missing,
        "output": None, "vla_run_dir": str(vla_run_dir) if vla_run_dir else None, "allow_partial": allow_partial,
    }
    if not labels or (missing and not allow_partial):
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(json.dumps(label, ensure_ascii=False) + "\n" for label in labels)
    if output.exists():
        if output.read_text() != serialized:
            raise ValueError(f"Existing labels differ from the current valid outcomes: {output}")
        report["reused"] = True
    else:
        output.write_text(serialized)
        report["reused"] = False
    report["output"] = str(output.resolve())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vla-run-dir", type=Path)
    parser.add_argument("--instances", type=Path,
                        help="Export only this ordered subset of the run's instances (for train/dev splits)")
    parser.add_argument("--allow-partial", action="store_true", help="Export valid outcomes and report every excluded episode")
    args = parser.parse_args()
    report = export_labels(args.run_root, args.output, args.vla_run_dir, args.instances, args.allow_partial)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in ("completed", "completed_with_errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
