from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tr_hash_server.jobs import (
    build_command,
    latest_checkpoint,
    load_config,
    render_unit,
    run_job,
    submit_job,
    unit_name,
    validate_config,
)


def _config(tmp_path: Path) -> dict:
    return {
        "name": "any-training",
        "command": ["/usr/bin/python3", "train.py", "--tokens", "arbitrary"],
        "working_directory": str(tmp_path),
        "gpu_devices": ["1"],
        "environment": {"PYTHONUNBUFFERED": "1"},
        "checkpoint": {
            "directory": str(tmp_path / "checkpoints"),
            "pattern": "step_*",
            "resume_arguments": ["--resume", "{checkpoint}"],
        },
        "egpu": {"stable_seconds": 30, "poll_seconds": 5},
    }


def test_config_is_workload_size_agnostic(tmp_path):
    config = validate_config(_config(tmp_path))
    assert config["command"][-1] == "arbitrary"
    assert "token_budget" not in config
    assert unit_name(config["name"]) == "tr-hash-job-any-training.service"


@pytest.mark.parametrize("name", ["../escape", "UPPER", "bad name", ""])
def test_job_name_rejects_unit_and_path_injection(tmp_path, name):
    config = _config(tmp_path)
    config["name"] = name
    with pytest.raises(ValueError):
        validate_config(config)


def test_gpu_device_rejects_systemd_environment_injection(tmp_path):
    config = _config(tmp_path)
    config["gpu_devices"] = ["1\nEnvironment=BAD=1"]
    with pytest.raises(ValueError):
        validate_config(config)


def test_latest_checkpoint_expands_resume_arguments(tmp_path):
    config = _config(tmp_path)
    checkpoint_dir = Path(config["checkpoint"]["directory"])
    checkpoint_dir.mkdir()
    old = checkpoint_dir / "step_000100"
    new = checkpoint_dir / "step_000200"
    old.mkdir()
    new.mkdir()
    os.utime(old, ns=(1, 1))
    os.utime(new, ns=(2, 2))

    assert latest_checkpoint(config) == new
    command, checkpoint = build_command(config)
    assert checkpoint == new
    assert command[-2:] == ["--resume", str(new)]


def test_rendered_unit_has_egpu_safety_and_no_restart(tmp_path):
    unit = render_unit(_config(tmp_path), "boris", "boris")
    assert "wait-gpu --timeout 300 --stable-for 30" in unit
    assert "Restart=no" in unit
    assert "CUDA_VISIBLE_DEVICES=1" in unit
    assert "run-job any-training" in unit
    assert "Conflicts=tr-hash-i64.service" in unit
    assert "systemd-inhibit --what=sleep:idle" in unit


def test_load_json_config(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")
    assert load_config(path)["name"] == "any-training"


def test_submit_persists_config_unit_and_state(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source = tmp_path / "source.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    jobs_root = tmp_path / "jobs"
    systemd_root = tmp_path / "systemd"
    systemd_root.mkdir()
    commands = []

    monkeypatch.setattr("tr_hash_server.jobs.os.geteuid", lambda: 0)
    monkeypatch.setattr("tr_hash_server.jobs.os.chown", lambda *_args: None)
    monkeypatch.setattr(
        "tr_hash_server.jobs.pwd.getpwnam", lambda _name: SimpleNamespace(pw_uid=1000)
    )
    monkeypatch.setattr(
        "tr_hash_server.jobs.grp.getgrnam", lambda _name: SimpleNamespace(gr_gid=1000)
    )
    monkeypatch.setattr(
        "tr_hash_server.jobs.subprocess.run",
        lambda command, **_kwargs: (
            commands.append(command) or SimpleNamespace(returncode=0)
        ),
    )

    result = submit_job(
        source,
        user="boris",
        group="boris",
        start=True,
        enable_on_boot=False,
        jobs_root=jobs_root,
        systemd_root=systemd_root,
    )

    assert result == 0
    assert (jobs_root / "any-training" / "job.json").is_file()
    assert (
        json.loads((jobs_root / "any-training" / "state.json").read_text())["status"]
        == "submitted"
    )
    assert (systemd_root / "tr-hash-job-any-training.service").is_file()
    assert ["systemctl", "daemon-reload"] in commands
    assert ["systemctl", "start", "tr-hash-job-any-training.service"] in commands


def test_failed_child_with_lost_egpu_requires_recovery(tmp_path, monkeypatch):
    config = _config(tmp_path)
    jobs_root = tmp_path / "jobs"
    job_root = jobs_root / config["name"]
    job_root.mkdir(parents=True)
    (job_root / "job.json").write_text(json.dumps(config), encoding="utf-8")
    (job_root / "state.json").write_text(
        json.dumps({"status": "queued"}), encoding="utf-8"
    )

    class FailedProcess:
        pid = 123
        returncode = 17

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        "tr_hash_server.jobs.subprocess.Popen",
        lambda *_args, **_kwargs: FailedProcess(),
    )
    monkeypatch.setattr(
        "tr_hash_server.jobs._probe_devices",
        lambda _devices: (False, "GPU 1: Unknown Error"),
    )

    result = run_job(config["name"], jobs_root=jobs_root)
    state = json.loads((job_root / "state.json").read_text())

    assert result == 79
    assert state["status"] == "recovery_required"
    assert state["exit_code"] == 17
