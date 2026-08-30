from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from tr_hash_server.cli import (
    _probe_nvidia_devices,
    build_launch_environment,
    build_server_command,
    build_tensorboard_command,
    cmd_healthcheck,
    cmd_wait_gpu,
)


def test_jobs_env_example_does_not_export_an_empty_hf_token():
    example = (Path(__file__).parents[1] / "config" / "jobs.env.example").read_text(
        encoding="utf-8"
    )
    assert "\nHF_TOKEN=\n" not in example


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
