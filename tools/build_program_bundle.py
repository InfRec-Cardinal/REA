#!/usr/bin/env python3
"""Freeze a library of program candidates into a bundle for the LIBERO program backends.

A library is a directory of frozen programs, one file per candidate:

    <library>/<suite>/<task>/<candidate>.py

    python3 tools/build_program_bundle.py --library programs/ --output outputs/bundle [--tasks tasks.json]

The bundle copies every selected program to <output>/programs/<suite>/<task>/<candidate>.py and writes
<output>/bundle.json:

    {"schema": "rea.program-bundle/v1", "createdAt": ..., "tasks": [
        {"suite": ..., "task": ..., "candidates": {"<candidate>": {"path": ..., "sha256": ...}}}]}

scripts/libero_program_backend.py and scripts/async_libero_backend.py read it from REA_PROGRAM_BUNDLE.
--tasks restricts the bundle to a JSON list of {"suite", "task"} entries, in that order. Program bytes
are copied unchanged; a candidate whose bytes repeat another candidate of the same task is dropped.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SCHEMA = "rea.program-bundle/v1"


def library_tasks(library):
    """(suite, task) pairs that have at least one candidate, in sorted order."""
    pairs = []
    for suite_dir in sorted(path for path in library.iterdir() if path.is_dir()):
        for task_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
            if any(task_dir.glob("*.py")):
                pairs.append((suite_dir.name, task_dir.name))
    return pairs


def prepare_bundle(library, output, tasks_file=None, created_at=None):
    library, output = Path(library), Path(output)
    if not library.is_dir():
        raise ValueError(f"Library directory not found: {library}")
    if (output / "bundle.json").exists():
        raise FileExistsError(f"Bundle already exists: {output / 'bundle.json'}")
    if tasks_file is None:
        pairs = library_tasks(library)
    else:
        entries = json.loads(Path(tasks_file).read_text(encoding="utf-8"))
        pairs = [(entry["suite"], entry["task"]) for entry in entries]
    if not pairs:
        raise ValueError("The bundle must contain at least one task")
    tasks, files = [], []
    for suite, task in pairs:
        sources = sorted((library / suite / task).glob("*.py"))
        if not sources:
            raise ValueError(f"No program candidates for {suite}/{task}")
        candidates, seen = {}, set()
        for source in sources:
            code = source.read_bytes()
            digest = hashlib.sha256(code).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            relative = Path("programs") / suite / task / source.name
            candidates[source.stem] = {"path": relative.as_posix(), "sha256": digest}
            files.append((relative, code))
        tasks.append({"suite": suite, "task": task, "candidates": candidates})
    bundle = {
        "schema": SCHEMA,
        "createdAt": created_at or datetime.now(timezone.utc).isoformat(),
        "tasks": tasks,
    }
    for relative, code in files:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(code)
    (output / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", required=True, help="directory of <suite>/<task>/<candidate>.py programs")
    parser.add_argument("--output", required=True, help="new directory for bundle.json and the copied programs")
    parser.add_argument("--tasks", help="JSON list of {suite, task} entries to include (default: every task in the library)")
    args = parser.parse_args()
    bundle = prepare_bundle(args.library, args.output, args.tasks)
    count = sum(len(task["candidates"]) for task in bundle["tasks"])
    print(f"wrote {Path(args.output) / 'bundle.json'}: {len(bundle['tasks'])} tasks, {count} candidates")


if __name__ == "__main__":
    main()
