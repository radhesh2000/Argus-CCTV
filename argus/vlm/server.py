"""Supervising the local inference server — what ``argus vlm`` drives.

The server is a **separate process in a separate virtualenv**, and that is the
design, not an accident of packaging:

* **Licensing.** vLLM's tree is large and pulls its own torch build. The argus
  package ships permissive dependencies only, and the cleanest way to keep that
  true is for the inference server never to enter it. Argus speaks HTTP to an
  OpenAI-compatible endpoint; what is behind that endpoint is deployment, not
  code.
* **Lifecycle.** Model load is ~2 min and CUDA-graph capture longer. The engine
  must be restartable in seconds, so it cannot own the model's process.
* **Blast radius.** A CUDA fault in the server kills the server. The cameras
  keep decoding.

Nothing here reaches the network at runtime: the model is passed to vLLM as a
**local path** and ``HF_HUB_OFFLINE=1`` is set, so a missing file fails loudly at
start rather than silently downloading 17 GB on a plant LAN.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.config import ArgusConfig


class VlmServerError(RuntimeError):
    """The server could not be started, or its prerequisites are missing."""


@dataclass(slots=True)
class ServerPaths:
    """Where the pieces live, and whether they are actually there."""

    venv: Path
    executable: Path
    model: Path
    log: Path
    pidfile: Path

    @property
    def venv_ready(self) -> bool:
        return self.executable.is_file()

    @property
    def model_ready(self) -> bool:
        return (self.model / "config.json").is_file() and any(self.model.glob("*.safetensors"))


def paths_for(config: ArgusConfig) -> ServerPaths:
    venv = config.vlm_venv
    return ServerPaths(
        venv=venv,
        executable=venv / "bin" / "vllm",
        model=config.vlm_model_path,
        log=config.logs_dir / "vlm-server.log",
        pidfile=config.data_dir / "vlm-server.pid",
    )


def build_command(config: ArgusConfig) -> list[str]:
    """The exact ``vllm serve`` invocation, so it can be inspected and tested."""
    paths = paths_for(config)
    server = config.vlm.server
    command = [
        str(paths.executable), "serve", str(paths.model),
        # Serve under the configured id even though we load from a path, so
        # clients ask for "Qwen/Qwen3-VL-8B-Instruct" rather than a filesystem
        # path that differs between the appliance and a developer's checkout.
        "--served-model-name", config.vlm.model,
        "--host", server.host,
        "--port", str(server.port),
        "--gpu-memory-utilization", str(server.gpu_memory_utilization),
        "--max-model-len", str(server.max_model_len),
        "--max-num-seqs", str(server.max_num_seqs),
        # One image per prompt: every policy asks about a single frame, and
        # letting the scheduler reserve multi-image slots wastes KV cache.
        "--limit-mm-per-prompt.image", "1",
    ]
    if server.extra_args:
        command += shlex.split(server.extra_args)
    return command


def server_env(config: ArgusConfig) -> dict[str, str]:
    """Environment for the server process.

    The PATH entry is not cosmetic. vLLM's warm-up compiles custom ops through
    torch's C++ extension JIT, which shells out to ``ninja`` — and ninja lives in
    the serving venv's ``bin``, not the system's. Launching the venv's ``vllm``
    by absolute path does *not* put that directory on PATH (only activation
    would), so without this the model loads, allocates its KV cache, captures
    CUDA graphs, and only then dies with ``FileNotFoundError: 'ninja'`` about
    three minutes in. Prepending the venv's bin is what activation would have
    done, and it costs nothing.
    """
    paths = paths_for(config)
    environment = dict(os.environ)
    bin_dir = str(paths.venv / "bin")
    environment["PATH"] = bin_dir + os.pathsep + environment.get("PATH", "")
    environment["VIRTUAL_ENV"] = str(paths.venv)
    # Fully local at runtime. If a file is missing, say so — do not fetch it.
    environment["HF_HUB_OFFLINE"] = "1"
    environment.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    return environment


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------


def probe(config: ArgusConfig, timeout: float = 3.0) -> dict[str, Any]:
    """Ask the endpoint what it is serving. Never raises."""
    url = f"{config.vlm.endpoint}/models"
    started = time.monotonic()
    try:
        request = urllib.request.Request(  # noqa: S310 - loopback http from config
            url, headers={"Authorization": f"Bearer {config.vlm.api_key}"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        # Name the URL. This string is what a degraded health panel and the
        # engine log both print, and "Connection refused" without saying to
        # what sends an operator looking at the wrong box.
        return {"available": False, "endpoint": config.vlm.endpoint,
                "reason": f"{url} unreachable: {type(exc).__name__}: {exc}"[:240]}
    models = [str(entry.get("id")) for entry in payload.get("data", []) if entry.get("id")]
    return {
        "available": True,
        "endpoint": config.vlm.endpoint,
        "model": models[0] if models else None,
        "models": models,
        "probe_sec": round(time.monotonic() - started, 3),
    }


def wait_ready(config: ArgusConfig, timeout: float, *, poll: float = 3.0,
               is_alive=None) -> dict[str, Any]:
    """Poll until the endpoint answers, the process dies, or time runs out."""
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {"available": False, "reason": "not probed"}
    while time.monotonic() < deadline:
        last = probe(config, timeout=min(poll, 5.0))
        if last.get("available"):
            return last
        if is_alive is not None and not is_alive():
            last["reason"] = "server process exited during startup"
            return last
        time.sleep(poll)
    last["reason"] = f"not ready within {timeout:.0f}s ({last.get('reason')})"
    return last


# --------------------------------------------------------------------------
# process control
# --------------------------------------------------------------------------


def _read_pid(pidfile: Path) -> int | None:
    try:
        pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def start(config: ArgusConfig, *, wait: bool = True) -> dict[str, Any]:
    """Launch the server detached, and (by default) wait for it to answer.

    Detached rather than a child of this CLI invocation: an operator running
    ``argus vlm start`` over ssh expects the model to survive the session, the
    same way ``argus run`` is expected to be supervised by systemd rather than
    by a terminal.
    """
    paths = paths_for(config)
    running = probe(config, timeout=2.0)
    if running.get("available"):
        return {"started": False, "already_running": True, **running}

    if not paths.venv_ready:
        raise VlmServerError(
            f"no vLLM environment at {paths.venv}. Run scripts/setup_vlm_server.sh "
            "(it builds the venv this command drives)."
        )
    if not paths.model_ready:
        raise VlmServerError(
            f"no model weights at {paths.model}. Run scripts/fetch_vlm.sh "
            f"(or set vlm.model / vlm.server.models_dir to where they live)."
        )

    paths.log.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(config)
    with paths.log.open("ab") as log:
        log.write(f"\n=== argus vlm start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())
        log.flush()
        process = subprocess.Popen(  # noqa: S603 - argv built from typed config
            command, stdout=log, stderr=subprocess.STDOUT, env=server_env(config),
            start_new_session=True,
        )
    paths.pidfile.write_text(str(process.pid))

    result: dict[str, Any] = {
        "started": True, "pid": process.pid, "log": str(paths.log),
        "command": " ".join(shlex.quote(part) for part in command),
    }
    if not wait:
        return result
    ready = wait_ready(
        config, config.vlm.server.startup_timeout_sec,
        is_alive=lambda: process.poll() is None,
    )
    return {**result, **ready}


def stop(config: ArgusConfig, timeout: float = 30.0) -> dict[str, Any]:
    """Terminate the server. SIGTERM to the process group, then SIGKILL."""
    paths = paths_for(config)
    pid = _read_pid(paths.pidfile)
    if pid is None:
        paths.pidfile.unlink(missing_ok=True)
        return {"stopped": False, "reason": "no running server recorded"}

    # The group, not the process: vLLM forks an EngineCore worker, and killing
    # only the API server would leave the model resident and the GPU claimed.
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError as exc:
        paths.pidfile.unlink(missing_ok=True)
        return {"stopped": False, "reason": f"{type(exc).__name__}: {exc}"}

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _read_pid(paths.pidfile) is None:
            paths.pidfile.unlink(missing_ok=True)
            return {"stopped": True, "pid": pid}
        time.sleep(0.5)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except OSError:
        pass
    paths.pidfile.unlink(missing_ok=True)
    return {"stopped": True, "pid": pid, "forced": True}


def installed(config: ArgusConfig) -> dict[str, Any]:
    """What is on disk for the local server — **without touching the network**.

    Split out of :func:`status` because ``/api/health`` is polled every few
    seconds by an open console, and the useful half of this answer ("are the
    weights even here?") is three stat calls while the other half is an HTTP
    round trip. It is also the half that explains a degraded panel: "no vLLM
    environment installed" and "installed but not started" are different
    problems with different fixes, and neither is visible from a refused
    connection alone.
    """
    paths = paths_for(config)
    return {
        "venv": str(paths.venv),
        "venv_ready": paths.venv_ready,
        "model_path": str(paths.model),
        "model_ready": paths.model_ready,
        "pid": _read_pid(paths.pidfile),
        "log": str(paths.log),
    }


def status(config: ArgusConfig) -> dict[str, Any]:
    """Everything ``argus vlm health`` prints, as data."""
    return {
        "endpoint": config.vlm.endpoint,
        "configured_model": config.vlm.model,
        **installed(config),
        **probe(config),
    }


__all__ = [
    "ServerPaths",
    "VlmServerError",
    "build_command",
    "installed",
    "paths_for",
    "probe",
    "server_env",
    "start",
    "status",
    "stop",
    "wait_ready",
]
