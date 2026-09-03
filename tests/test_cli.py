from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tr_hash_server.cli import (
    _load_environment_file,
    _migrate_server_env,
    _probe_nvidia_devices,
    build_launch_environment,
    build_server_command,
    build_tensorboard_command,
    cmd_doctor,
    cmd_healthcheck,
    cmd_prepare_gpu,
    cmd_wait_gpu,
)


def test_jobs_env_example_does_not_export_an_empty_hf_token():
    example = (Path(__file__).parents[1] / "config" / "jobs.env.example").read_text(
        encoding="utf-8"
    )
    assert "\nHF_TOKEN=\n" not in example


def test_i64_unit_prepares_egpu_with_sandboxed_root_privileges():
    unit = (Path(__file__).parents[1] / "systemd" / "tr-hash-i64.service").read_text(
        encoding="utf-8"
    )
    assert (
        "ExecStartPre=!/usr/local/bin/tr-hash-server prepare-gpu --timeout 180"
        in unit
    )
    assert "ExecStartPre=+/usr/local/bin/tr-hash-server prepare-gpu" not in unit
    assert "ExecStartPre=/usr/local/bin/tr-hash-server wait-gpu" not in unit


def test_server_env_caps_egpu_to_150_watts():
    example = (Path(__file__).parents[1] / "config" / "server.env.example").read_text(
        encoding="utf-8"
    )
    assert "TR_HASH_EGPU_POWER_LIMIT_W=150" in example


def test_server_env_selects_egpu_by_stable_uuid():
    example = (Path(__file__).parents[1] / "config" / "server.env.example").read_text(
        encoding="utf-8"
    )
    assert (
        "TR_HASH_DEVICES=GPU-757192a7-6497-c48a-1543-12b92e74ee41"
        in example
    )


def test_load_environment_file_supplies_manual_cli_configuration(
    tmp_path,
    monkeypatch,
):
    config = tmp_path / "server.env"
    config.write_text(
        "# host settings\n"
        "TR_HASH_DEVICES=GPU-stable\n"
        "TR_HASH_EGPU_POWER_LIMIT_W=150\n"
        "TR_HASH_WORKING_DIRECTORY=/srv/TR Hash\n"
        "TR_HASH_READY_URL=http://127.0.0.1/ready#probe\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TR_HASH_DEVICES", raising=False)
    monkeypatch.setenv("TR_HASH_EGPU_POWER_LIMIT_W", "140")

    _load_environment_file(config)

    assert os.environ["TR_HASH_DEVICES"] == "GPU-stable"
    assert os.environ["TR_HASH_EGPU_POWER_LIMIT_W"] == "140"
    assert os.environ["TR_HASH_WORKING_DIRECTORY"] == "/srv/TR Hash"
    assert os.environ["TR_HASH_READY_URL"] == "http://127.0.0.1/ready#probe"


def test_install_migrates_numeric_device_to_stable_uuid_and_power_cap(tmp_path):
    installed = tmp_path / "server.env"
    installed.write_text(
        "TR_HASH_DEVICES=1\nTR_HASH_CUDA_GRAPHS=1\nTR_HASH_DEVICES=GPU-stable\n",
        encoding="utf-8",
    )
    example = tmp_path / "server.env.example"
    example.write_text(
        "TR_HASH_DEVICES=GPU-stable\n"
        "TR_HASH_EGPU_POWER_LIMIT_W=150\n"
        "TR_HASH_EGPU_STABLE_SECONDS=15\n",
        encoding="utf-8",
    )

    _migrate_server_env(installed, example)
    first = installed.read_text(encoding="utf-8")
    _migrate_server_env(installed, example)

    assert "TR_HASH_DEVICES=GPU-stable" in first
    assert "TR_HASH_EGPU_POWER_LIMIT_W=150" in first
    assert "TR_HASH_EGPU_STABLE_SECONDS=15" in first
    assert "TR_HASH_CUDA_GRAPHS=1" in first
    assert first.count("TR_HASH_DEVICES=") == 1
    assert installed.read_text(encoding="utf-8") == first


def test_home_server_defaults(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("TR_HASH_"):
            monkeypatch.delenv(name, raising=False)
    command = build_server_command()
    assert command[:3] == [
        "/home/boris/pytorch/bin/tr-hash-i64",
        "serve",
        "tr-hash-moe-200m",
    ]
    assert command[command.index("--quantization") + 1] == "none"
    assert command[command.index("--device") + 1] == "cuda"
    assert "--api-key-file" not in command


def test_explicit_device_is_forwarded(monkeypatch):
    monkeypatch.setenv("TR_HASH_DEVICE", "cpu")
    command = build_server_command()
    assert command[command.index("--device") + 1] == "cpu"


def test_optional_checkpoint_and_key(monkeypatch):
    monkeypatch.setenv("TR_HASH_CHECKPOINT", "/models/v1")
    monkeypatch.setenv("TR_HASH_API_KEY_FILE", "/run/secrets/api.key")
    command = build_server_command()
    assert command[-4:] == [
        "--checkpoint",
        "/models/v1",
        "--api-key-file",
        "/run/secrets/api.key",
    ]


def test_cuda_graphs_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TR_HASH_CUDA_GRAPHS", "0")
    assert build_server_command()[-1] == "--no-cuda-graphs"


def test_gpu_order_matches_nvidia_smi(monkeypatch):
    monkeypatch.delenv("TR_HASH_CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setenv("TR_HASH_DEVICES", "1")
    environment = build_launch_environment()
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"


def test_tensorboard_defaults_are_private(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("TR_HASH_TENSORBOARD_"):
            monkeypatch.delenv(name, raising=False)

    command = build_tensorboard_command()

    assert command[:3] == [
        "/home/boris/pytorch/bin/python",
        "-m",
        "tensorboard.main",
    ]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "6006"
    assert command[command.index("--logdir") + 1] == (
        "/var/lib/tr-hash-server/tensorboard"
    )


def test_gpu_probe_uses_nvidia_smi_without_pytorch(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="NVIDIA GeForce RTX 5060 Ti, 00000000:07:00.0, GPU-test\n",
            stderr="",
        )

    monkeypatch.setenv("TR_HASH_DEVICES", "1")
    monkeypatch.setattr("tr_hash_server.cli.subprocess.run", fake_run)
    healthy, detail = _probe_nvidia_devices()

    assert healthy
    assert "RTX 5060 Ti" in detail
    assert calls == [
        [
            "nvidia-smi",
            "--id",
            "1",
            "--query-gpu=name,pci.bus_id,uuid",
            "--format=csv,noheader",
        ]
    ]
    assert all("python" not in part.lower() for part in calls[0])


def test_doctor_accepts_gpu_uuid(monkeypatch, tmp_path, capsys):
    executable = tmp_path / "tr-hash-i64"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    gpu_uuid = "GPU-757192a7-6497-c48a-1543-12b92e74ee41"
    monkeypatch.setenv("TR_HASH_EXECUTABLE", str(executable))
    monkeypatch.setenv("TR_HASH_WORKING_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("TR_HASH_API_KEY_FILE", "")
    monkeypatch.setenv("TR_HASH_DEVICES", gpu_uuid)
    monkeypatch.setattr(
        "tr_hash_server.cli._gpu_rows",
        lambda: ["0, NVIDIA GeForce RTX 5060 Ti, 16384, 610.57.04"],
    )
    monkeypatch.setattr(
        "tr_hash_server.cli._probe_nvidia_devices",
        lambda: (True, f"GPU {gpu_uuid}: NVIDIA GeForce RTX 5060 Ti"),
    )
    monkeypatch.setattr(
        "tr_hash_server.cli._ready", lambda _url, _timeout: (True, "HTTP 200")
    )

    result = cmd_doctor(Namespace(timeout=0.1))

    assert result == 0
    assert "OK   CUDA devices" in capsys.readouterr().out


def test_wait_gpu_requires_stable_visibility(monkeypatch, capsys):
    probes = iter(
        [
            (False, "GPU 1 unavailable"),
            (True, "GPU 1 ready"),
            (True, "GPU 1 ready"),
        ]
    )
    clock = iter([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    monkeypatch.setattr(
        "tr_hash_server.cli._probe_nvidia_devices", lambda: next(probes)
    )
    monkeypatch.setattr("tr_hash_server.cli.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("tr_hash_server.cli.time.sleep", lambda _seconds: None)

    result = cmd_wait_gpu(Namespace(timeout=30.0, interval=0.0, stable_for=1.0))

    assert result == 0
    assert "GPU READY" in capsys.readouterr().out


def test_prepare_gpu_rescans_and_applies_default_power_limit_before_stability(
    monkeypatch, tmp_path
):
    events = []
    rescan = tmp_path / "rescan"
    rescan.write_text("", encoding="utf-8")
    gpu_uuid = "GPU-757192a7-6497-c48a-1543-12b92e74ee41"
    monkeypatch.setenv("TR_HASH_DEVICES", gpu_uuid)
    monkeypatch.delenv("TR_HASH_EGPU_POWER_LIMIT_W", raising=False)
    monkeypatch.setattr(
        "tr_hash_server.cli.cmd_wait_gpu",
        lambda args: events.append(("wait", args.stable_for)) or 0,
    )
    monkeypatch.setattr(
        "tr_hash_server.cli.subprocess.run",
        lambda command, **_kwargs: events.append(command) or SimpleNamespace(returncode=0),
    )

    result = cmd_prepare_gpu(
        Namespace(
            timeout=30.0,
            interval=0.0,
            stable_for=1.0,
            rescan_path=str(rescan),
        )
    )

    assert result == 0
    assert rescan.read_text(encoding="utf-8") == "1\n"
    assert events == [
        ("wait", 0.0),
        ["nvidia-smi", "--id", gpu_uuid, "--power-limit", "150"],
        ("wait", 1.0),
    ]


def test_prepare_gpu_rejects_mutable_numeric_device_selection(monkeypatch, tmp_path):
    rescan = tmp_path / "rescan"
    rescan.write_text("", encoding="utf-8")
    monkeypatch.setenv("TR_HASH_DEVICES", "1")

    with pytest.raises(SystemExit, match="stable GPU UUID"):
        cmd_prepare_gpu(
            Namespace(
                timeout=30.0,
                interval=0.0,
                stable_for=1.0,
                rescan_path=str(rescan),
            )
        )

    assert rescan.read_text(encoding="utf-8") == ""


def test_healthcheck_refuses_restart_when_egpu_is_lost(monkeypatch, tmp_path, capsys):
    restarts = []
    monkeypatch.setenv("TR_HASH_HEALTH_FAILURES", "1")
    monkeypatch.setattr(
        "tr_hash_server.cli._ready",
        lambda _url, _timeout: (False, "connection refused"),
    )
    monkeypatch.setattr(
        "tr_hash_server.cli._probe_nvidia_devices",
        lambda: (False, "GPU 1: Unknown Error"),
    )
    monkeypatch.setattr(
        "tr_hash_server.cli.subprocess.run",
        lambda command, **_kwargs: restarts.append(command),
    )

    result = cmd_healthcheck(
        Namespace(state_file=str(tmp_path / "failures"), timeout=0.1, no_restart=False)
    )

    assert result == 2
    assert restarts == []
    assert "refusing a restart loop" in capsys.readouterr().out
