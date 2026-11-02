#!/usr/bin/env python3
"""Bundle and deploy the REA G1 service to the robot's computer over SSH."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_NODE_VERSION = "22.23.2"
DEFAULT_PRIME_AGENT_VERSION = "0.8.0"
MAIN_SERVICE_NAME = "rea-agent.service"
THINKER_SERVICE_NAME = "rea-thinker.service"
SERVICE_NAMES = (THINKER_SERVICE_NAME, MAIN_SERVICE_NAME)
# Older units that serve the same ports and must stop before the new units start (and come back if the deployment
# fails). Oldest first; stop_legacy_services stops in reverse (request-serving units before their advisers). Add the
# names of any earlier deployment here.
LEGACY_SERVICE_NAMES = ()
HEALTH_URLS = {
    MAIN_SERVICE_NAME: "http://127.0.0.1:8091/health",
    THINKER_SERVICE_NAME: "http://127.0.0.1:8092/health",
}
# Everything the service needs at runtime plus the operator tooling. The robot-side Python workers live in
# scripts/g1 and ride along because the whole scripts directory is bundled.
OPTIONAL_BUNDLE_FILES = ("node_modules", "src", "test", "scripts", "README.md")


class DeployError(RuntimeError):
    pass


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("deploy", "check"), nargs="?", default="deploy")
    host = os.environ.get("G1_DEPLOY_HOST")
    result.add_argument("--host", default=host, required=host is None, help="the robot computer's SSH host, or set G1_DEPLOY_HOST")
    result.add_argument("--user", default=os.environ.get("G1_DEPLOY_USER", "unitree"), help="SSH user; the stock Unitree account is unitree")
    result.add_argument("--port", type=int, default=22)
    result.add_argument("--identity-file")
    result.add_argument("--target", default=os.environ.get("G1_DEPLOY_TARGET"), help="remote checkout directory; default /home/<user>/rea-agent")
    result.add_argument("--node-version", default=DEFAULT_NODE_VERSION)
    result.add_argument("--node-archive", type=Path)
    result.add_argument("--prime-agent-version", default=DEFAULT_PRIME_AGENT_VERSION)
    result.add_argument("--prime-agent-package", type=Path)
    result.add_argument("--npm", default="npm", help="workstation npm executable")
    result.add_argument("--primary-model", default=os.environ.get("G1_PRIMARY_MODEL"), required=not os.environ.get("G1_PRIMARY_MODEL"),
                        help="vision model for the one-token engines, written to the remote .env and checked after start (default G1_PRIMARY_MODEL)")
    result.add_argument("--fallback-model", default=os.environ.get("G1_FALLBACK_MODEL", ""),
                        help="optional fallback model checked the same way (default G1_FALLBACK_MODEL; empty for none)")
    result.add_argument(
        "--keep-key",
        action="store_true",
        help="preserve REA_API_KEY already present in the remote .env",
    )
    result.add_argument("--connect-timeout", type=int, default=8)
    result.add_argument("--health-timeout", type=float, default=30.0)
    return result


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.target:
        args.target = f"/home/{args.user}/rea-agent"
    return args


def connection_target(args: argparse.Namespace) -> str:
    return f"{args.user}@{args.host}"


def ssh_prefix(args: argparse.Namespace) -> list[str]:
    command = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
        "-p",
        str(args.port),
    ]
    if args.identity_file:
        command.extend(["-i", args.identity_file])
    command.append(connection_target(args))
    return command


def scp_prefix(args: argparse.Namespace) -> list[str]:
    command = [
        "scp",
        "-q",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
        "-P",
        str(args.port),
    ]
    if args.identity_file:
        command.extend(["-i", args.identity_file])
    return command


def run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    child_env = os.environ.copy()
    child_env.pop("REA_API_KEY", None)
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=child_env,
            input=input_text,
            text=True,
            capture_output=capture,
            check=False,
        )
    except FileNotFoundError as exc:
        raise DeployError(f"executable not found: {command[0]}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise DeployError(f"command failed ({completed.returncode}): {detail[-1000:]}")
    return completed


def node_archive(args: argparse.Namespace) -> Path:
    if args.node_archive:
        archive = args.node_archive.expanduser().resolve()
        if not archive.is_file():
            raise DeployError(f"Node archive not found: {archive}")
        return archive

    parts = args.node_version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise DeployError("node-version must look like 22.23.2")
    filename = f"node-v{args.node_version}-linux-arm64.tar.gz"
    cache_root = Path.home() / "Library" / "Caches" / "rea-agent"
    cache_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / filename
    if archive.is_file() and archive.stat().st_size > 0:
        return archive

    url = f"https://nodejs.org/dist/v{args.node_version}/{filename}"
    partial = cache_root / f"{filename}.part"
    print(f"[deploy] downloading {url}", file=sys.stderr)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
    except (OSError, urllib.error.URLError) as exc:
        raise DeployError(f"cannot download Node ARM64 runtime: {exc}") from exc
    os.replace(partial, archive)
    return archive


def prime_agent_package(args: argparse.Namespace) -> Path:
    if args.prime_agent_package:
        package = args.prime_agent_package.expanduser().resolve()
        if not package.is_file():
            raise DeployError(f"Prime Agent package not found: {package}")
        return package

    version = args.prime_agent_version.strip()
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise DeployError("prime-agent-version must look like 0.8.0")
    filename = f"prime-agent-{version}.tgz"
    cache_root = Path.home() / "Library" / "Caches" / "rea-agent"
    cache_root.mkdir(parents=True, exist_ok=True)
    package = cache_root / filename
    if package.is_file() and package.stat().st_size > 0:
        return package

    url = (
        "https://github.com/PrimeIntellect-ai/prime-agent/releases/download/"
        f"v{version}/{filename}"
    )
    partial = cache_root / f"{filename}.part"
    print(f"[deploy] downloading {url}", file=sys.stderr)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
    except (OSError, urllib.error.URLError) as exc:
        raise DeployError(f"cannot download Prime Agent release: {exc}") from exc
    os.replace(partial, package)
    return package


def prepare_app(args: argparse.Namespace, work: Path) -> Path:
    required = [
        ROOT / "package.json",
        ROOT / "env.example",
        HERE / "g1_u2_sense.py",
        HERE / "sense_worker.py",
        HERE / "control_daemon.py",
        ROOT / "prime-agent-config" / "models.json",
        ROOT / "prime-agent-config" / "settings.json",
        ROOT / "prime-agent-config" / "auth.example.json",
        HERE / "rea-agent.service.in",
        HERE / "rea-thinker.service.in",
        ROOT / "server.mjs",
        ROOT / "thinker.mjs",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise DeployError("missing deployment source: " + ", ".join(missing))

    stage = work / "app"
    stage.mkdir(parents=True)
    shutil.copy2(ROOT / "package.json", stage / "package.json")
    shutil.copy2(ROOT / "env.example", stage / "env.example")
    shutil.copy2(ROOT / "server.mjs", stage / "server.mjs")
    shutil.copy2(ROOT / "thinker.mjs", stage / "thinker.mjs")
    shutil.copytree(ROOT / "prime-agent-config", stage / "prime-agent-config")
    if (ROOT / "README.md").is_file():
        shutil.copy2(ROOT / "README.md", stage / "README.md")
    # scripts/ carries the Python workers (scripts/g1): they are not optional, the agent spawns them at boot.
    for directory in ("src", "test", "scripts"):
        if (ROOT / directory).is_dir():
            shutil.copytree(ROOT / directory, stage / directory, ignore=shutil.ignore_patterns("__pycache__"))

    service_templates = {
        MAIN_SERVICE_NAME: HERE / "rea-agent.service.in",
        THINKER_SERVICE_NAME: HERE / "rea-thinker.service.in",
    }
    for service_name, template_path in service_templates.items():
        unit = template_path.read_text(encoding="utf-8")
        (stage / service_name).write_text(
            unit.replace("@TARGET@", args.target),
            encoding="utf-8",
        )

    print("[deploy] installing Linux ARM64 optional dependencies", file=sys.stderr)
    npm_command = [
        args.npm,
        "install",
        "--omit=dev",
        "--ignore-scripts",
        "--package-lock=false",
        "--platform=linux",
        "--arch=arm64",
    ]
    try:
        run(npm_command, cwd=stage)
    except Exception as error:  # noqa: BLE001 - dependencies are optional, keep deploying
        print(f"[deploy] optional dependency install failed ({error}); continuing", file=sys.stderr)
    (stage / "node_modules" / ".package-lock.json").unlink(missing_ok=True)
    # undici only tunes the streaming transport. Prime Agent is installed from its official release below.

    archive_path = work / "rea-agent-app.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        names = [
            "package.json",
            "env.example",
            MAIN_SERVICE_NAME,
            THINKER_SERVICE_NAME,
            "server.mjs",
            "thinker.mjs",
            "prime-agent-config",
        ]
        for optional in OPTIONAL_BUNDLE_FILES:
            if (stage / optional).exists():
                names.append(optional)
        for name in names:
            archive.add(stage / name, arcname=name)
    return archive_path


REMOTE_INSTALL = r"""set -eu
target=$1
app_archive=$2
node_archive=$3
prime_agent_archive=$4
main_service_name=$5
thinker_service_name=$6

cleanup() {
  rm -f "$app_archive" "$node_archive" "$prime_agent_archive"
}
trap cleanup EXIT

mkdir -p "$target" "$target/runtime/node" "$target/runtime/prime-agent" "$HOME/.config/systemd/user"
tar --warning=no-timestamp -xzf "$node_archive" -C "$target/runtime/node" --strip-components=1
tar --warning=no-timestamp -xzf "$app_archive" -C "$target"
test -x "$target/runtime/node/bin/node"
test -f "$target/server.mjs"
test -f "$target/thinker.mjs"
test -f "$target/prime-agent-config/models.json"

if [ -f "$target/.env" ]; then
  set -a
  . "$target/.env"
  set +a
fi
PATH="$target/runtime/node/bin:$PATH" \
  "$target/runtime/node/bin/npm" install \
  --prefix "$target/runtime/prime-agent" \
  --omit=dev \
  "$prime_agent_archive"
test -x "$target/runtime/prime-agent/node_modules/.bin/prime-agent"
PATH="$target/runtime/node/bin:$PATH" \
  "$target/runtime/prime-agent/node_modules/.bin/prime-agent" --version

if [ ! -f "$target/.env" ]; then
  umask 077
  cp "$target/env.example" "$target/.env"
fi
chmod 600 "$target/.env"
install -m 0644 "$target/$main_service_name" "$HOME/.config/systemd/user/$main_service_name"
install -m 0644 "$target/$thinker_service_name" "$HOME/.config/systemd/user/$thinker_service_name"

uid=$(id -u)
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$uid}
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}
systemctl --user daemon-reload
"$target/runtime/node/bin/node" --version
"""


REMOTE_SET_KEY = r"""import os
import sys

path = sys.argv[1]
key = sys.stdin.readline().rstrip("\r\n")
if not key:
    raise SystemExit("REA_API_KEY cannot be empty")

with open(path, "r", encoding="utf-8") as source:
    lines = source.read().splitlines()

updated = []
found = False
for line in lines:
    if line.startswith("REA_API_KEY="):
        updated.append("REA_API_KEY=" + key)
        found = True
    else:
        updated.append(line)
if not found:
    updated.append("REA_API_KEY=" + key)

temporary = path + ".deploy-" + str(os.getpid())
with open(temporary, "w", encoding="utf-8") as output:
    output.write("\n".join(updated) + "\n")
os.chmod(temporary, 0o600)
os.replace(temporary, path)
"""


REMOTE_MIGRATE_PRIME = r"""import os
import sys

path, target, version, primary_model, fallback_model = sys.argv[1:6]
updates = {
    "G1_ENGINE": "prime",
    "G1_PRIMARY_MODEL": primary_model,
    "G1_FALLBACK_MODEL": fallback_model,
    "G1_EXECUTE_ENABLED": "0",
    "G1_ACTUATOR": "none",
    "G1_PERSISTENT_CONTROL": "0",
    "G1_PRIME_AGENT_VERSION": version,
    "G1_PRIME_AGENT_PROVIDER": "openai-compatible",
    "G1_PRIME_AGENT_POOL_SIZE": "2",
    "G1_PRIME_WARMUP": "1",
    "PRIME_AGENT_BIN": target + "/runtime/prime-agent/node_modules/.bin/prime-agent",
    "PRIME_AGENT_CODING_AGENT_DIR": target + "/prime-agent-config",
    "G1_PRIME_AGENT_CWD": target,
}

with open(path, "r", encoding="utf-8") as source:
    lines = source.read().splitlines()

kept = []
for line in lines:
    name = line.split("=", 1)[0] if "=" in line else None
    if name not in updates:
        kept.append(line)
kept.extend(name + "=" + value for name, value in updates.items())

temporary = path + ".prime-" + str(os.getpid())
with open(temporary, "w", encoding="utf-8") as output:
    output.write("\n".join(kept) + "\n")
os.chmod(temporary, 0o600)
os.replace(temporary, path)
"""


def systemctl_command(arguments: str) -> str:
    return (
        "uid=$(id -u); "
        "export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$uid}; "
        "export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}; "
        f"systemctl --user {arguments}"
    )


SYSTEMD_RUNTIME_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "MainPID",
    "ControlGroup",
    "WorkingDirectory",
)


def service_properties(args: argparse.Namespace, service_name: str) -> dict[str, str]:
    properties = " ".join(
        f"--property={shlex.quote(name)}" for name in SYSTEMD_RUNTIME_PROPERTIES
    )
    command = systemctl_command(
        f"show --no-pager {properties} {shlex.quote(service_name)}"
    )
    output = run([*ssh_prefix(args), command], capture=True).stdout
    return {
        name: value
        for line in output.splitlines()
        if "=" in line
        for name, value in (line.split("=", 1),)
    }


def service_snapshot(
    args: argparse.Namespace, service_names: Sequence[str]
) -> dict[str, dict[str, str]]:
    return {
        service_name: service_properties(args, service_name)
        for service_name in service_names
    }


def systemctl_each(
    args: argparse.Namespace, action: str, service_names: Sequence[str]
) -> None:
    for service_name in service_names:
        run(
            [
                *ssh_prefix(args),
                systemctl_command(f"{action} {shlex.quote(service_name)}"),
            ]
        )


def stop_legacy_services(
    args: argparse.Namespace, snapshot: dict[str, dict[str, str]]
) -> None:
    loaded = [
        service_name
        for service_name in LEGACY_SERVICE_NAMES
        if snapshot.get(service_name, {}).get("LoadState") == "loaded"
    ]
    # Stop the request-serving service before its advisory dependency.
    systemctl_each(args, "stop", tuple(reversed(loaded)))
    systemctl_each(args, "disable", tuple(reversed(loaded)))


def restore_legacy_services(
    args: argparse.Namespace, snapshot: dict[str, dict[str, str]]
) -> list[str]:
    errors: list[str] = []

    def attempt(action: str, service_name: str) -> None:
        try:
            systemctl_each(args, action, (service_name,))
        except DeployError as exc:
            errors.append(f"{action} {service_name}: {exc}")

    # A failed replacement must stop retrying on the occupied website ports before legacy returns.
    for service_name in reversed(SERVICE_NAMES):
        attempt("stop", service_name)
        attempt("disable", service_name)

    for service_name in LEGACY_SERVICE_NAMES:
        previous = snapshot.get(service_name, {})
        if previous.get("LoadState") != "loaded":
            continue
        if previous.get("UnitFileState") == "enabled":
            attempt("enable", service_name)
        if previous.get("ActiveState") == "active":
            attempt("start", service_name)
    return errors


REMOTE_PROCESS_RUNTIME = r"""import json
import os
import sys

pid = int(sys.argv[1])
target = os.path.realpath(sys.argv[2])
with open(f"/proc/{pid}/cgroup", "r", encoding="utf-8") as source:
    cgroups = [line.rstrip("\n").split(":", 2)[-1] for line in source]
print(json.dumps({
    "cwd": os.path.realpath(f"/proc/{pid}/cwd"),
    "target": target,
    "cgroups": cgroups,
}))
"""


def process_runtime(
    args: argparse.Namespace, pid: int
) -> dict[str, Any]:
    command = "python3 -c " + shlex.quote(REMOTE_PROCESS_RUNTIME) + " " + " ".join(
        (shlex.quote(str(pid)), shlex.quote(args.target))
    )
    output = run([*ssh_prefix(args), command], capture=True).stdout.strip()
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise DeployError(f"cannot inspect MainPID {pid}") from exc


def validate_service_runtime(
    args: argparse.Namespace,
    service_name: str,
    properties: dict[str, str],
) -> dict[str, Any]:
    if properties.get("LoadState") != "loaded":
        raise DeployError(f"{service_name} is not loaded")
    if properties.get("ActiveState") != "active" or properties.get("SubState") != "running":
        raise DeployError(
            f"{service_name} service state is "
            f"{properties.get('ActiveState')!r}/{properties.get('SubState')!r}"
        )
    try:
        main_pid = int(properties.get("MainPID", "0"))
    except ValueError as exc:
        raise DeployError(f"{service_name} reports an invalid MainPID") from exc
    if main_pid <= 0:
        raise DeployError(f"{service_name} has no running MainPID")

    expected_working_directory = os.path.normpath(args.target)
    working_directory = os.path.normpath(properties.get("WorkingDirectory") or "/")
    if working_directory != expected_working_directory:
        raise DeployError(
            f"{service_name} WorkingDirectory is {working_directory!r}, "
            f"expected {expected_working_directory!r}"
        )

    control_group = properties.get("ControlGroup") or ""
    if not control_group.endswith(f"/{service_name}"):
        raise DeployError(
            f"{service_name} reports an unexpected ControlGroup {control_group!r}"
        )

    process = process_runtime(args, main_pid)
    if process.get("cwd") != process.get("target"):
        raise DeployError(
            f"{service_name} MainPID {main_pid} is not running from the deployment target"
        )
    if control_group not in process.get("cgroups", []):
        raise DeployError(
            f"{service_name} MainPID {main_pid} is outside its reported ControlGroup"
        )
    return {
        "main_pid": main_pid,
        "control_group": control_group,
        "working_directory": working_directory,
    }


def read_deployment_key(args: argparse.Namespace) -> str | None:
    if args.keep_key:
        return None
    key = os.environ.get("REA_API_KEY")
    if not key:
        try:
            key = getpass.getpass("REA_API_KEY: ")
        except (EOFError, KeyboardInterrupt) as exc:
            raise DeployError("REA_API_KEY input was cancelled") from exc
    if not key:
        raise DeployError("REA_API_KEY is required unless --keep-key is used")
    return key


def set_remote_key(args: argparse.Namespace, key: str) -> None:
    command = "python3 -c " + shlex.quote(REMOTE_SET_KEY) + " " + shlex.quote(
        f"{args.target}/.env"
    )
    run([*ssh_prefix(args), command], input_text=key + "\n")


def migrate_remote_prime_config(args: argparse.Namespace) -> None:
    command = "python3 -c " + shlex.quote(REMOTE_MIGRATE_PRIME) + " " + " ".join(
        shlex.quote(value)
        for value in (f"{args.target}/.env", args.target, args.prime_agent_version, args.primary_model, args.fallback_model)
    )
    run([*ssh_prefix(args), command])


def deploy(args: argparse.Namespace) -> dict[str, Any]:
    api_key = read_deployment_key(args)
    runtime = node_archive(args)
    prime_agent = prime_agent_package(args)
    token = str(time.time_ns())
    remote_app = f"/tmp/rea-agent-app-{token}.tar.gz"
    remote_node = f"/tmp/rea-agent-node-{token}.tar.gz"
    remote_prime_agent = f"/tmp/prime-agent-{token}.tgz"

    with tempfile.TemporaryDirectory(prefix="rea-agent-") as temp:
        app_archive = prepare_app(args, Path(temp))
        print(f"[deploy] copying application to {connection_target(args)}", file=sys.stderr)
        run([*scp_prefix(args), str(app_archive), f"{connection_target(args)}:{remote_app}"])
        print("[deploy] copying Node Linux ARM64 runtime", file=sys.stderr)
        run([*scp_prefix(args), str(runtime), f"{connection_target(args)}:{remote_node}"])
        print("[deploy] copying Prime Agent release", file=sys.stderr)
        run([*scp_prefix(args), str(prime_agent), f"{connection_target(args)}:{remote_prime_agent}"])
        remote_command = "bash -s -- " + " ".join(
            shlex.quote(value)
            for value in (
                args.target,
                remote_app,
                remote_node,
                remote_prime_agent,
                MAIN_SERVICE_NAME,
                THINKER_SERVICE_NAME,
            )
        )
        run([*ssh_prefix(args), remote_command], input_text=REMOTE_INSTALL)

    try:
        if api_key is not None:
            print("[deploy] updating remote REA_API_KEY from SSH stdin", file=sys.stderr)
            set_remote_key(args, api_key)
        migrate_remote_prime_config(args)
    finally:
        api_key = None

    legacy = service_snapshot(args, LEGACY_SERVICE_NAMES)
    try:
        stop_legacy_services(args, legacy)
        systemctl_each(args, "enable", SERVICE_NAMES)
        systemctl_each(args, "restart", SERVICE_NAMES)
        return check(args, wait=True)
    except (Exception, KeyboardInterrupt) as exc:
        rollback_errors = restore_legacy_services(args, legacy)
        recovery = "restored legacy services"
        if rollback_errors:
            recovery = "legacy recovery was incomplete: " + "; ".join(rollback_errors)
        raise DeployError(
            f"replacement deployment failed; {recovery}: {exc}"
        ) from exc


def check(args: argparse.Namespace, wait: bool = False) -> dict[str, Any]:
    deadline = time.monotonic() + (args.health_timeout if wait else 0)
    last_error: Exception | None = None
    while True:
        try:
            service_details = {
                service_name: service_properties(args, service_name)
                for service_name in SERVICE_NAMES
            }
            service_runtime = {
                service_name: validate_service_runtime(
                    args, service_name, service_details[service_name]
                )
                for service_name in SERVICE_NAMES
            }
            mode = run(
                [*ssh_prefix(args), "stat -c %a " + shlex.quote(f"{args.target}/.env")],
                capture=True,
            ).stdout.strip()
            health: dict[str, Any] = {}
            for service_name, health_url in HEALTH_URLS.items():
                health_code = (
                    "import urllib.request;"
                    f"r=urllib.request.urlopen({health_url!r},timeout=3);"
                    "print(r.read().decode('utf-8'))"
                )
                health_text = run(
                    [
                        *ssh_prefix(args),
                        "python3 -c " + shlex.quote(health_code),
                    ],
                    capture=True,
                ).stdout.strip()
                health[service_name] = json.loads(health_text)

            if mode != "600":
                raise DeployError(f"environment file mode is {mode!r}, expected 600")
            for service_name, service_health in health.items():
                if service_health.get("ok") is not True:
                    raise DeployError(
                        f"{service_name} health check failed: {service_health}"
                    )
                if service_health.get("ready") is not True:
                    raise DeployError(
                        f"{service_name} health is up but not ready"
                    )
            main_health = health[MAIN_SERVICE_NAME]
            if main_health.get("harness") != "prime-agent-rpc":
                raise DeployError("main service is not using the Prime Agent RPC harness")
            if main_health.get("engine") != "prime":
                raise DeployError("main service engine is not Prime Agent")
            if main_health.get("harness_version") != args.prime_agent_version:
                raise DeployError("main service reports the wrong Prime Agent version")
            if main_health.get("primary_model") != args.primary_model:
                raise DeployError(f"main service is not using the configured primary model {args.primary_model}")
            if (main_health.get("fallback_model") or "") != args.fallback_model:
                raise DeployError(f"main service is not using the configured fallback model {args.fallback_model or '(none)'}")
            if main_health.get("actuator_kind") != "none":
                raise DeployError("main service must leave robot actuation to the external controller")
            if main_health.get("execute_enabled") is not False:
                raise DeployError("main service execution must remain disabled")
            if main_health.get("persistent_control") is not False:
                raise DeployError("main service must not start a second persistent control worker")
            if (main_health.get("harness_state") or {}).get("ready") is not True:
                raise DeployError("Prime Agent RPC pool is not ready")
            if (main_health.get("harness_state") or {}).get("modelReady") is not True:
                raise DeployError("Prime Agent model warm-up did not pass")
            model_checks = (main_health.get("harness_state") or {}).get("models") or {}
            expected_models = {"primary": args.primary_model, **({"fallback": args.fallback_model} if args.fallback_model else {})}
            for role, model in expected_models.items():
                if (model_checks.get(role) or {}).get("ready") is not True:
                    raise DeployError(f"Prime Agent {role} model warm-up did not pass")
                if (model_checks.get(role) or {}).get("model") != model:
                    raise DeployError(f"Prime Agent {role} model is not {model}")
            return {
                "ok": True,
                "host": args.host,
                "target": args.target,
                "service": MAIN_SERVICE_NAME,
                "service_state": service_details[MAIN_SERVICE_NAME]["ActiveState"],
                "service_runtime": service_runtime[MAIN_SERVICE_NAME],
                "thinker_service": THINKER_SERVICE_NAME,
                "thinker_service_state": service_details[THINKER_SERVICE_NAME]["ActiveState"],
                "thinker_service_runtime": service_runtime[THINKER_SERVICE_NAME],
                "environment_mode": mode,
                "health": health[MAIN_SERVICE_NAME],
                "thinker_health": health[THINKER_SERVICE_NAME],
                "services": {
                    service_name: {
                        "state": service_details[service_name]["ActiveState"],
                        "runtime": service_runtime[service_name],
                        "health": health[service_name],
                    }
                    for service_name in SERVICE_NAMES
                },
            }
        except (DeployError, json.JSONDecodeError) as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                raise DeployError(f"health check did not pass: {last_error}") from exc
            time.sleep(1)


def main(argv: list[str] | None = None) -> int:
    args = apply_defaults(parser().parse_args(argv))
    if args.port <= 0 or args.connect_timeout <= 0 or args.health_timeout <= 0:
        print("error: port and timeouts must be positive", file=sys.stderr)
        return 2
    try:
        result = deploy(args) if args.action == "deploy" else check(args)
    except DeployError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
