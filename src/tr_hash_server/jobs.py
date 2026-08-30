"""Generic, restart-safe jobs for the TR-HASH eGPU host.

The manager deliberately knows nothing about model sizes, datasets, trainers,
or token budgets.  A job is an argv vector plus an optional checkpoint resume
contract.  The wrapper owns only process supervision and eGPU safety.
"""

from __future__ import annotations

import json
import math
import os
import pwd
import grp
import re
import shlex
import signal
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any


DEFAULT_JOBS_ROOT = Path("/var/lib/tr-hash-server/jobs")
DEFAULT_TENSORBOARD_ROOT = Path("/var/lib/tr-hash-server/tensorboard")
SYSTEMD_ROOT = Path("/etc/systemd/system")
JOBS_ENV = Path("/etc/tr-hash-server/jobs.env")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def unit_name(name: str) -> str:
    _validate_name(name)
    return f"tr-hash-job-{name}.service"


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError("job name must match [a-z0-9][a-z0-9_-]{0,62}")


def load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("rb") as stream:
        if suffix == ".toml":
            data = tomllib.load(stream)
        elif suffix == ".json":
            data = json.load(stream)
        else:
            raise ValueError("job config must be TOML or JSON")
    if not isinstance(data, dict):
        raise ValueError("job config must contain an object/table")
    return validate_config(data)


def validate_config(data: dict[str, Any]) -> dict[str, Any]:
    name = data.get("name")
    _validate_name(name)

    command = data.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise ValueError("command must be a non-empty array of strings")

    working_directory = data.get("working_directory")
    if (
        not isinstance(working_directory, str)
        or not Path(working_directory).is_absolute()
    ):
        raise ValueError("working_directory must be an absolute path")

    devices = data.get("gpu_devices", ["1"])
    if not isinstance(devices, list):
        raise ValueError("gpu_devices must be an array")
    data["gpu_devices"] = [str(device) for device in devices]
    if not all(device.isdigit() for device in data["gpu_devices"]):
        raise ValueError(
            "gpu_devices currently accepts numeric nvidia-smi indexes only"
        )

    environment = data.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("environment must be a table/object")
    for key, value in environment.items():
        if not isinstance(key, str) or not _ENV_NAME.fullmatch(key):
            raise ValueError(f"invalid environment variable name: {key!r}")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"environment value for {key} must be scalar")

    checkpoint = data.get("checkpoint", {})
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a table/object")
    if checkpoint:
        directory = checkpoint.get("directory")
        if not isinstance(directory, str) or not Path(directory).is_absolute():
            raise ValueError("checkpoint.directory must be an absolute path")
        pattern = checkpoint.get("pattern", "*")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("checkpoint.pattern must be a non-empty string")
        arguments = checkpoint.get("resume_arguments", [])
        if not isinstance(arguments, list) or not all(
            isinstance(part, str) for part in arguments
        ):
            raise ValueError("checkpoint.resume_arguments must be an array of strings")

    tensorboard_logdir = data.get("tensorboard_logdir")
    if tensorboard_logdir is not None and (
        not isinstance(tensorboard_logdir, str)
        or not Path(tensorboard_logdir).is_absolute()
    ):
        raise ValueError("tensorboard_logdir must be an absolute path")

    egpu = data.get("egpu", {})
    if not isinstance(egpu, dict):
        raise ValueError("egpu must be a table/object")
    for key, default in (("stable_seconds", 30), ("poll_seconds", 5)):
        value = egpu.get(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"egpu.{key} must be a non-negative number")
    power_limit_w = egpu.get("power_limit_w")
    if power_limit_w is not None and (
        not isinstance(power_limit_w, (int, float))
        or isinstance(power_limit_w, bool)
        or not math.isfinite(power_limit_w)
        or power_limit_w <= 0
    ):
        raise ValueError("egpu.power_limit_w must be a positive number")

    return data


def _job_root(name: str, jobs_root: Path = DEFAULT_JOBS_ROOT) -> Path:
    _validate_name(name)
    return jobs_root / name


def _config_path(name: str, jobs_root: Path = DEFAULT_JOBS_ROOT) -> Path:
    return _job_root(name, jobs_root) / "job.json"


def _state_path(name: str, jobs_root: Path = DEFAULT_JOBS_ROOT) -> Path:
    return _job_root(name, jobs_root) / "state.json"


def register_tensorboard_run(
    config: dict[str, Any],
    tensorboard_root: Path = DEFAULT_TENSORBOARD_ROOT,
) -> Path | None:
    """Expose a job's event directory below TensorBoard's stable root."""
    logdir = config.get("tensorboard_logdir")
    if logdir is None:
        return None
    tensorboard_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    os.chmod(tensorboard_root, 0o755)
    link = tensorboard_root / config["name"]
    if link.exists() and not link.is_symlink():
        raise RuntimeError(f"TensorBoard registry entry is not a symlink: {link}")
    link.unlink(missing_ok=True)
    link.symlink_to(Path(logdir), target_is_directory=True)
    return link


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {} if default is None else default.copy()


def latest_checkpoint(config: dict[str, Any]) -> Path | None:
    checkpoint = config.get("checkpoint", {})
    if not checkpoint:
        return None
    directory = Path(checkpoint["directory"])
    candidates = [
        candidate
        for candidate in directory.glob(checkpoint.get("pattern", "*"))
        if candidate.is_dir()
        and (
            (candidate / "checkpoint.pt").is_file()
            or (candidate / ".metadata").is_file()
        )
    ]
    if not candidates:
        return None
    return max(
        candidates, key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name)
    )


def build_command(config: dict[str, Any]) -> tuple[list[str], Path | None]:
    command = list(config["command"])
    checkpoint = latest_checkpoint(config)
    if checkpoint is not None:
        for argument in config.get("checkpoint", {}).get("resume_arguments", []):
            command.append(argument.replace("{checkpoint}", str(checkpoint)))
    return command, checkpoint


def _probe_devices(devices: list[str]) -> tuple[bool, str]:
    details: list[str] = []
    for device in devices:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--id",
                    str(device),
                    "--query-gpu=name,pci.bus_id,uuid",
                    "--format=csv,noheader",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            return False, "nvidia-smi not found"
        detail = (result.stdout or result.stderr).strip()
        if result.returncode != 0 or not detail:
            return False, f"GPU {device}: {detail or 'unavailable'}"
        details.append(detail)
    return True, "; ".join(details)


def render_unit(config: dict[str, Any], user: str, group: str) -> str:
    name = config["name"]
    stable_seconds = float(config.get("egpu", {}).get("stable_seconds", 30))
    devices = ",".join(config["gpu_devices"])
    power_limit = config.get("egpu", {}).get("power_limit_w")
    power_limit_lines = ""
    if devices and power_limit is not None:
        rendered_limit = f"{float(power_limit):g}"
        power_limit_lines = "".join(
            f"ExecStartPre=+/usr/bin/nvidia-smi --id {device} "
            f"--power-limit {rendered_limit}\n"
            for device in config["gpu_devices"]
        )
    gpu_unit = ""
    gpu_environment = "Environment=CUDA_VISIBLE_DEVICES="
    if devices:
        gpu_unit = """Conflicts=tr-hash-i64.service
Before=tr-hash-i64.service
"""
        gpu_environment = f"""Environment=TR_HASH_DEVICES={devices}
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=CUDA_VISIBLE_DEVICES={devices}
ExecStartPre=/usr/local/bin/tr-hash-server wait-gpu --timeout 300 --stable-for {stable_seconds:g}
{power_limit_lines.rstrip()}""".rstrip()
    return f"""[Unit]
Description=TR-Hash generic eGPU job: {name}
After=network-online.target nvidia-persistenced.service
Wants=network-online.target nvidia-persistenced.service
{gpu_unit.rstrip()}
StartLimitIntervalSec=5min
StartLimitBurst=1

[Service]
Type=simple
User={user}
Group={group}
EnvironmentFile=-{JOBS_ENV}
{gpu_environment}
ExecStart=/usr/local/bin/tr-hash-server run-job {name}
Restart=no
TimeoutStartSec=15min
TimeoutStopSec=5min
KillMode=control-group
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
UMask=0077

[Install]
WantedBy=multi-user.target
"""


def submit_job(
    config_path: Path,
    *,
    user: str,
    group: str,
    start: bool,
    enable_on_boot: bool,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    systemd_root: Path = SYSTEMD_ROOT,
    tensorboard_root: Path = DEFAULT_TENSORBOARD_ROOT,
) -> int:
    if os.geteuid() != 0:
        raise SystemExit("job submit must be run with sudo")
    config = load_config(config_path)
    name = config["name"]
    account = pwd.getpwnam(user)
    group_entry = grp.getgrnam(group)
    root = _job_root(name, jobs_root)
    root.mkdir(parents=True, exist_ok=True)
    os.chown(root, account.pw_uid, group_entry.gr_gid)
    os.chmod(root, 0o750)
    stored_config = _config_path(name, jobs_root)
    _write_json(stored_config, config)
    os.chown(stored_config, account.pw_uid, group_entry.gr_gid)
    _write_json(
        _state_path(name, jobs_root), {"status": "submitted", "updated_at": time.time()}
    )
    os.chown(_state_path(name, jobs_root), account.pw_uid, group_entry.gr_gid)
    register_tensorboard_run(config, tensorboard_root)

    unit = systemd_root / unit_name(name)
    unit.write_text(render_unit(config, user, group), encoding="utf-8")
    os.chmod(unit, 0o644)
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    if enable_on_boot:
        command = ["systemctl", "enable"]
        if start:
            command.append("--now")
        command.append(unit.name)
        subprocess.run(command, check=True)
    elif start:
        subprocess.run(["systemctl", "start", unit.name], check=True)
    print(f"Submitted {name}: {shlex.join(config['command'])}")
    return 0


def _terminate_process(process: subprocess.Popen[Any], timeout: float = 30.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_job(name: str, jobs_root: Path = DEFAULT_JOBS_ROOT) -> int:
    config = validate_config(_read_json(_config_path(name, jobs_root)))
    state_path = _state_path(name, jobs_root)
    previous = _read_json(state_path)
    if previous.get("status") == "completed":
        print(f"Job {name} already completed; use `job resume {name}` to run it again")
        return 0

    command, checkpoint = build_command(config)
    environment = os.environ.copy()
    environment.update(
        {key: str(value) for key, value in config.get("environment", {}).items()}
    )
    devices = config["gpu_devices"]
    if devices:
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    working_directory = Path(config["working_directory"])
    if not working_directory.is_dir():
        raise SystemExit(f"working directory does not exist: {working_directory}")

    try:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            start_new_session=True,
        )
    except OSError as error:
        _write_json(
            state_path,
            {
                "status": "failed",
                "reason": str(error),
                "updated_at": time.time(),
            },
        )
        print(f"Could not start job {name}: {error}")
        return 127
    _write_json(
        state_path,
        {
            "status": "running",
            "pid": process.pid,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "command": command,
            "started_at": time.time(),
            "updated_at": time.time(),
        },
    )

    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        _terminate_process(process)

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    failures = 0
    poll_seconds = float(config.get("egpu", {}).get("poll_seconds", 5))
    try:
        while process.poll() is None:
            if not devices:
                time.sleep(max(0.1, poll_seconds))
                continue
            healthy, detail = _probe_devices(devices)
            failures = 0 if healthy else failures + 1
            if failures >= 2:
                _terminate_process(process)
                _write_json(
                    state_path,
                    {
                        "status": "recovery_required",
                        "reason": detail,
                        "checkpoint": str(latest_checkpoint(config) or ""),
                        "updated_at": time.time(),
                    },
                )
                print(
                    f"eGPU unavailable: {detail}; reboot required, restart suppressed"
                )
                return 79
            time.sleep(max(0.1, poll_seconds))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    return_code = int(process.returncode or 0)
    if return_code != 0:
        healthy, detail = _probe_devices(devices)
        if not healthy:
            _write_json(
                state_path,
                {
                    "status": "recovery_required",
                    "reason": detail,
                    "exit_code": return_code,
                    "checkpoint": str(latest_checkpoint(config) or ""),
                    "updated_at": time.time(),
                },
            )
            print(f"Job exited while eGPU was unavailable: {detail}; reboot required")
            return 79
    status = "stopped" if stopping else ("completed" if return_code == 0 else "failed")
    _write_json(
        state_path,
        {
            "status": status,
            "exit_code": return_code,
            "checkpoint": str(latest_checkpoint(config) or ""),
            "updated_at": time.time(),
        },
    )
    return return_code


def state_for(name: str, jobs_root: Path = DEFAULT_JOBS_ROOT) -> dict[str, Any]:
    return _read_json(_state_path(name, jobs_root), {"status": "unknown"})


def mark_queued(name: str, jobs_root: Path = DEFAULT_JOBS_ROOT) -> None:
    state = state_for(name, jobs_root)
    state.update({"status": "queued", "updated_at": time.time()})
    _write_json(_state_path(name, jobs_root), state)
