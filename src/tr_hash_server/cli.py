from __future__ import annotations

import argparse
import grp
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SERVICE = "tr-hash-i64.service"
TIMER = "tr-hash-healthcheck.timer"
DEFAULT_ENV = Path("/etc/tr-hash-server/server.env")
DEFAULT_STATE = Path("/run/tr-hash-server/readiness-failures")


def _value(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def build_server_command() -> list[str]:
    executable = _value("TR_HASH_EXECUTABLE", "/home/boris/pytorch/bin/tr-hash-i64")
    command = [
        executable,
        "serve",
        _value("TR_HASH_MODEL", "tr-hash-moe-200m"),
        "--host",
        _value("TR_HASH_HOST", "127.0.0.1"),
        "--port",
        _value("TR_HASH_PORT", "7860"),
        "--dtype",
        _value("TR_HASH_DTYPE", "float16"),
        "--quantization",
        _value("TR_HASH_QUANTIZATION", "none"),
        "--max-batch-size",
        _value("TR_HASH_MAX_BATCH_SIZE", "8"),
        "--chunk-size",
        _value("TR_HASH_CHUNK_SIZE", "256"),
        "--max-kv-blocks",
        _value("TR_HASH_MAX_KV_BLOCKS", "256"),
        "--max-pending",
        _value("TR_HASH_MAX_PENDING", "32"),
    ]
    checkpoint = _value("TR_HASH_CHECKPOINT", "")
    if checkpoint:
        command.extend(("--checkpoint", checkpoint))
    api_key_file = _value("TR_HASH_API_KEY_FILE", "")
    if api_key_file:
        command.extend(("--api-key-file", api_key_file))
    if _value("TR_HASH_CUDA_GRAPHS", "1") == "0":
        command.append("--no-cuda-graphs")
    return command


def build_launch_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = _value(
        "TR_HASH_CUDA_DEVICE_ORDER", "PCI_BUS_ID"
    )
    environment["CUDA_VISIBLE_DEVICES"] = _value("TR_HASH_DEVICES", "1")
    return environment


def cmd_launch(_: argparse.Namespace) -> int:
    command = build_server_command()
    executable = Path(command[0])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise SystemExit(f"TR-Hash-i64 executable is not usable: {executable}")
    working_directory = Path(_value("TR_HASH_WORKING_DIRECTORY", "/home/boris"))
    if not working_directory.is_dir():
        raise SystemExit(f"Working directory does not exist: {working_directory}")
    environment = build_launch_environment()
    os.chdir(working_directory)
    os.execve(str(executable), command, environment)
    return 0


def _ready(url: str, timeout: float) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if 200 <= response.status < 300:
                return True, f"HTTP {response.status}"
            return False, f"HTTP {response.status}"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return False, str(error)


def cmd_healthcheck(args: argparse.Namespace) -> int:
    url = _value("TR_HASH_READY_URL", "http://127.0.0.1:7860/ready")
    threshold = int(_value("TR_HASH_HEALTH_FAILURES", "3"))
    state = Path(args.state_file)
    healthy, detail = _ready(url, args.timeout)
    state.parent.mkdir(parents=True, exist_ok=True)
    if healthy:
        state.write_text("0\n", encoding="utf-8")
        print(f"READY {url} ({detail})")
        return 0

    try:
        failures = int(state.read_text(encoding="utf-8").strip()) + 1
    except (FileNotFoundError, ValueError):
        failures = 1
    state.write_text(f"{failures}\n", encoding="utf-8")
    print(f"NOT READY {url} ({detail}); failure {failures}/{threshold}")
    if failures >= threshold and not args.no_restart:
        subprocess.run(["systemctl", "restart", SERVICE], check=True)
        state.write_text("0\n", encoding="utf-8")
        print(f"Restarted {SERVICE}")
    return 1


def _gpu_rows() -> list[str]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def cmd_doctor(args: argparse.Namespace) -> int:
    failures = 0

    def report(ok: bool, label: str, detail: str) -> None:
        nonlocal failures
        symbol = "OK" if ok else "FAIL"
        print(f"{symbol:<4} {label:<18} {detail}")
        failures += int(not ok)

    executable = Path(_value("TR_HASH_EXECUTABLE", "/home/boris/pytorch/bin/tr-hash-i64"))
    report(executable.is_file() and os.access(executable, os.X_OK), "executable", str(executable))
    workdir = Path(_value("TR_HASH_WORKING_DIRECTORY", "/home/boris"))
    report(workdir.is_dir(), "working directory", str(workdir))
    api_key = _value("TR_HASH_API_KEY_FILE", "")
    if api_key:
        path = Path(api_key)
        mode = path.stat().st_mode & 0o777 if path.exists() else 0
        report(path.is_file() and mode & 0o007 == 0, "API key file", f"{path} mode={mode:04o}")
    rows = _gpu_rows()
    report(bool(rows), "NVIDIA driver", "; ".join(rows) if rows else "nvidia-smi failed")
    devices = _value("TR_HASH_DEVICES", "1")
    indexes = {row.split(",", 1)[0].strip() for row in rows}
    requested = {part.strip() for part in devices.split(",") if part.strip()}
    report(bool(requested) and requested <= indexes, "CUDA devices", devices)
    healthy, detail = _ready(_value("TR_HASH_READY_URL", "http://127.0.0.1:7860/ready"), args.timeout)
    report(healthy, "readiness", detail)
    return 1 if failures else 0


def _run(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def cmd_status(_: argparse.Namespace) -> int:
    return _run(["systemctl", "--no-pager", "--full", "status", SERVICE])


def cmd_restart(_: argparse.Namespace) -> int:
    return _run(["systemctl", "restart", SERVICE])


def cmd_logs(args: argparse.Namespace) -> int:
    command = ["journalctl", "-u", SERVICE, "-n", str(args.lines)]
    if args.follow:
        command.append("-f")
    return _run(command)


def cmd_install(args: argparse.Namespace) -> int:
    if os.geteuid() != 0:
        raise SystemExit("install must be run with sudo")
    project = Path(args.project).resolve()
    units = project / "systemd"
    example = project / "config" / "server.env.example"
    source_package = project / "src" / "tr_hash_server"
    for source in (
        units / SERVICE,
        units / "tr-hash-healthcheck.service",
        units / TIMER,
        example,
        source_package / "cli.py",
    ):
        if not source.is_file():
            raise SystemExit(f"Missing installation file: {source}")
    service_group = grp.getgrnam(args.group).gr_gid
    config_directory = Path("/etc/tr-hash-server")
    config_directory.mkdir(mode=0o750, parents=True, exist_ok=True)
    os.chown(config_directory, 0, service_group)
    os.chmod(config_directory, 0o750)
    Path("/var/lib/tr-hash-server").mkdir(mode=0o755, parents=True, exist_ok=True)
    library = Path("/usr/local/lib/tr-hash-server")
    destination_package = library / "tr_hash_server"
    library.mkdir(mode=0o755, parents=True, exist_ok=True)
    shutil.copytree(
        source_package,
        destination_package,
        dirs_exist_ok=True,
        copy_function=shutil.copyfile,
    )
    launcher = Path("/usr/local/bin/tr-hash-server")
    launcher.write_text(
        "#!/usr/bin/python3\n"
        "import sys\n"
        "sys.path.insert(0, '/usr/local/lib/tr-hash-server')\n"
        "from tr_hash_server.cli import main\n"
        "raise SystemExit(main())\n",
        encoding="utf-8",
    )
    os.chmod(launcher, 0o755)
    for unit in (SERVICE, "tr-hash-healthcheck.service", TIMER):
        destination = Path("/etc/systemd/system") / unit
        shutil.copyfile(units / unit, destination)
        os.chmod(destination, 0o644)
    if not DEFAULT_ENV.exists():
        shutil.copyfile(example, DEFAULT_ENV)
    os.chown(DEFAULT_ENV, 0, service_group)
    os.chmod(DEFAULT_ENV, 0o640)
    restorecon = shutil.which("restorecon")
    if restorecon:
        subprocess.run(
            [
                restorecon,
                "-RF",
                "/usr/local/bin/tr-hash-server",
                "/usr/local/lib/tr-hash-server",
                "/etc/tr-hash-server",
                f"/etc/systemd/system/{SERVICE}",
                "/etc/systemd/system/tr-hash-healthcheck.service",
                f"/etc/systemd/system/{TIMER}",
            ],
            check=True,
        )
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    print(f"Installed units and {DEFAULT_ENV}")
    print("Edit the configuration, run doctor, then enable the service and timer.")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="tr-hash-server")
    sub = root.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("launch", help="Launch TR-Hash-i64 from the host environment")
    launch.set_defaults(func=cmd_launch)
    health = sub.add_parser("healthcheck", help="Check /ready and restart after repeated failures")
    health.add_argument("--state-file", default=str(DEFAULT_STATE))
    health.add_argument("--timeout", type=float, default=5.0)
    health.add_argument("--no-restart", action="store_true")
    health.set_defaults(func=cmd_healthcheck)
    doctor = sub.add_parser("doctor", help="Validate the host configuration")
    doctor.add_argument("--timeout", type=float, default=5.0)
    doctor.set_defaults(func=cmd_doctor)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("restart").set_defaults(func=cmd_restart)
    logs = sub.add_parser("logs")
    logs.add_argument("-f", "--follow", action="store_true")
    logs.add_argument("-n", "--lines", type=int, default=100)
    logs.set_defaults(func=cmd_logs)
    install = sub.add_parser("install", help="Install systemd units on Fedora")
    install.add_argument("--project", default=".")
    install.add_argument("--group", default="boris", help="Group allowed to read host configuration")
    install.set_defaults(func=cmd_install)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
