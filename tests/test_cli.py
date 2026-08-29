from __future__ import annotations

import os

from tr_hash_server.cli import build_server_command


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
    assert "--api-key-file" not in command


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

