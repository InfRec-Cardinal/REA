"""Inspect upstream robot manifests and packages without importing their SDKs."""

import argparse
import ast
import json
from pathlib import Path
import sys


def asset_information(reference, manifest_directory, repository):
    if isinstance(reference, dict):
        reference = reference.get("ref")
    if reference is None:
        return None
    if reference.startswith("file:"):
        relative = Path(reference.removeprefix("file:")).expanduser()
        candidates = [manifest_directory / relative, repository / relative]
        selected = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        return {"ref": reference, "path": str(selected.resolve()), "exists": selected.is_file(), "resolution": "local"}
    return {"ref": reference, "path": None, "exists": None, "resolution": "external_not_checked"}


def inspect_openral(repository):
    import yaml

    repository = Path(repository).expanduser().resolve()
    directory = repository / "robots"
    if not directory.is_dir():
        raise ValueError(f"OpenRAL robots directory does not exist: {directory}")
    robots = []
    for path in sorted(directory.glob("*/robot.yaml")):
        manifest = yaml.safe_load(path.read_text())
        hal = manifest.get("hal") or {}
        simulation_configured = manifest.get("sim") is not None
        assets = {
            name: asset_information(reference, path.parent, repository)
            for name, reference in (manifest.get("assets") or {}).items()
        }
        robots.append({
            "id": path.parent.name,
            "name": manifest["name"],
            "manifest": path.relative_to(repository).as_posix(),
            "path": str(path),
            "embodimentKind": manifest.get("embodiment_kind"),
            "hal": {"sim": hal.get("sim"), "real": hal.get("real")},
            "sim": {"configured": simulation_configured},
            "assets": assets,
            "sceneRequired": not hal.get("sim") and not simulation_configured,
        })
    return {
        "root": str(repository),
        "count": len(robots),
        "sceneRequiredMeaning": "No bare simulation entry is declared; simulation requires a separate scene adapter. This does not confirm that such a scene is available.",
        "robots": robots,
    }


def robot_spec_name(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for function in tree.body:
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and function.name == "get_robot_spec":
            for call in ast.walk(function):
                if not isinstance(call, ast.Call):
                    continue
                constructor = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
                if constructor == "RobotSpec":
                    for keyword in call.keywords:
                        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                            return keyword.value.value
    return None


def inspect_rpent(repository):
    repository = Path(repository).expanduser().resolve()
    directory = repository / "robots"
    if not directory.is_dir():
        raise ValueError(f"RPent robots directory does not exist: {directory}")
    robots = []
    for path in sorted(directory.glob("*/robot_spec.py")):
        package = path.parent
        if package.name.startswith("_") or not (package / "__init__.py").is_file():
            continue
        robots.append({
            "id": package.name,
            "name": robot_spec_name(path),
            "path": str(package),
            "spec": path.relative_to(repository).as_posix(),
            "envServer": (package / "env_server.py").is_file(),
            "envClient": (package / "env_client.py").is_file(),
            "vlaServer": (package / "vla_server.py").is_file(),
            "vlaClient": (package / "vla_client.py").is_file(),
        })
    return {"root": str(repository), "count": len(robots), "robots": robots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openral-root", type=Path)
    parser.add_argument("--rpent-root", type=Path)
    arguments = parser.parse_args()
    if arguments.openral_root is None and arguments.rpent_root is None:
        parser.error("provide --openral-root, --rpent-root, or both")
    inventory = {}
    if arguments.openral_root is not None:
        inventory["openral"] = inspect_openral(arguments.openral_root)
    if arguments.rpent_root is not None:
        inventory["rpent"] = inspect_rpent(arguments.rpent_root)
    json.dump(inventory, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
