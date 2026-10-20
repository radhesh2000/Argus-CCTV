"""Supervising the vLLM process — the parts that can be checked without a GPU.

The command line and the environment are worth their own tests because both
have already been wrong once on this hardware, and both fail *slowly*: a missing
PATH entry does not surface until the model has loaded, allocated 16 GB of KV
cache and captured its CUDA graphs, roughly three minutes in.
"""

from __future__ import annotations

import os

import pytest

from argus.config import ArgusConfig
from argus.vlm import server as vlm_server


@pytest.fixture
def served(config: ArgusConfig) -> ArgusConfig:
    config.vlm.model = "Qwen/Qwen3-VL-8B-Instruct"
    config.vlm.server.port = 18000
    config.vlm.server.venv = config.data_dir / "venv"
    config.vlm.server.models_dir = config.data_dir / "models"
    return config


def test_command_serves_the_local_path_under_the_configured_name(served):
    command = vlm_server.build_command(served)

    assert command[1] == "serve"
    # A path, not a HuggingFace id: a running appliance must never reach for
    # the network to find its own weights.
    assert command[2] == str(served.vlm_model_path)
    assert not command[2].startswith("Qwen/")
    # ...but clients still ask for it by id.
    assert command[command.index("--served-model-name") + 1] == "Qwen/Qwen3-VL-8B-Instruct"
    assert command[command.index("--port") + 1] == "18000"
    assert command[command.index("--max-num-seqs") + 1] == "4"


def test_command_uses_the_serving_venvs_own_vllm(served):
    """Not whatever `vllm` happens to be on PATH — the argus env has none."""
    assert vlm_server.build_command(served)[0] == str(served.vlm_venv / "bin" / "vllm")


def test_extra_args_are_appended_shell_style(served):
    served.vlm.server.extra_args = "--enforce-eager --seed 7"
    assert vlm_server.build_command(served)[-3:] == ["--enforce-eager", "--seed", "7"]


def test_env_puts_the_serving_venv_first_on_path(served):
    """vLLM's warm-up shells out to `ninja`, which lives in the serving venv.
    Launching its `vllm` by absolute path does not put that bin on PATH, and the
    resulting failure arrives minutes into startup."""
    environment = vlm_server.server_env(served)
    assert environment["PATH"].split(os.pathsep)[0] == str(served.vlm_venv / "bin")
    assert environment["VIRTUAL_ENV"] == str(served.vlm_venv)


def test_env_forbids_downloading_at_runtime(served):
    assert vlm_server.server_env(served)["HF_HUB_OFFLINE"] == "1"


def test_status_reports_missing_prerequisites_rather_than_raising(served):
    state = vlm_server.status(served)
    assert state["venv_ready"] is False
    assert state["model_ready"] is False
    assert state["available"] is False
    assert state["reason"]
    assert state["endpoint"] == "http://127.0.0.1:18000/v1"


def test_start_refuses_without_a_venv(served):
    with pytest.raises(vlm_server.VlmServerError, match="setup_vlm_server"):
        vlm_server.start(served)


def test_start_refuses_without_weights(served):
    (served.vlm_venv / "bin").mkdir(parents=True)
    (served.vlm_venv / "bin" / "vllm").write_text("#!/bin/sh\n")
    with pytest.raises(vlm_server.VlmServerError, match="fetch_vlm"):
        vlm_server.start(served)


def test_model_ready_needs_both_config_and_weights(served):
    paths = vlm_server.paths_for(served)
    paths.model.mkdir(parents=True)
    assert paths.model_ready is False
    (paths.model / "config.json").write_text("{}")
    assert paths.model_ready is False          # config alone is not a model
    (paths.model / "model-00001-of-00004.safetensors").write_bytes(b"\x00")
    assert paths.model_ready is True


def test_stop_is_a_no_op_when_nothing_is_recorded(served):
    result = vlm_server.stop(served)
    assert result["stopped"] is False
    assert "no running server" in result["reason"]


def test_wait_ready_gives_up_when_the_process_dies(served):
    result = vlm_server.wait_ready(served, timeout=5.0, poll=0.1, is_alive=lambda: False)
    assert result["available"] is False
    assert "exited during startup" in result["reason"]
