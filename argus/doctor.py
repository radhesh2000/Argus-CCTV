"""``argus doctor`` — one command that says whether this box is an appliance.

This is the support tool. When a site calls, the first request is "run
``argus doctor`` and paste the output", so every check here is written to be
readable by somebody who has never seen the codebase, and every failure carries
the command that fixes it. It exits non-zero when something is broken, so it is
also usable as the last step of an install script and as a monitoring probe.

Three rules it follows, all of them learned from what makes a support tool
useless:

* **A check reports what it measured, not what it assumed.** "gateway binary
  missing" and "gateway binary present but not executable" are different
  problems with different fixes, so they are different messages.
* **``warn`` is not ``fail``.** A missing VLM is a warning: the appliance is
  specified to run without one. A missing detector checkpoint is a failure:
  nothing works. Exit status follows that distinction rather than counting
  yellow text.
* **Nothing here touches the network.** Ports are probed on loopback and on
  this box's own address; no check downloads, resolves or phones anywhere. A
  doctor that hangs for 30 s on a plant with no uplink is a doctor nobody runs.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from argus.config import ROOT, ArgusConfig
from argus.db import SCHEMA_VERSION, Database

OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

_ORDER = {FAIL: 0, WARN: 1, SKIP: 2, OK: 3}

#: Connect timeout for a port probe. Loopback and the LAN address of this very
#: box both answer in microseconds or not at all; anything longer is a hang.
_PROBE_TIMEOUT = 0.35


@dataclass(slots=True)
class Check:
    """One line of the report."""

    group: str
    name: str
    status: str
    detail: str
    fix: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = {"group": self.group, "name": self.name, "status": self.status,
               "detail": self.detail}
        if self.fix:
            out["fix"] = self.fix
        if self.data:
            out["data"] = self.data
        return out


def _run(argv: list[str], timeout: float = 10.0) -> "subprocess.CompletedProcess[str] | None":
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------
# host
# --------------------------------------------------------------------------


def check_host(config: ArgusConfig) -> list[Check]:
    machine = platform.machine()
    python = ".".join(str(p) for p in sys.version_info[:3])
    status = OK if machine in ("aarch64", "arm64") else WARN
    detail = f"{platform.system()} {platform.release()} on {machine}, python {python}"
    return [
        Check("host", "platform", status, detail,
              fix=None if status == OK else
              "Argus targets linux/arm64 (DGX Spark). Other architectures are "
              "development only — the pinned go2rtc and ffmpeg builds are arm64.",
              data={"machine": machine, "python": python}),
        Check("host", "config", OK if config.source_path else WARN,
              f"{config.source_path or 'built-in defaults (no argus.yaml loaded)'}",
              fix=None if config.source_path else
              "Pass -c/--config, or set ARGUS_CONFIG, so services and the CLI agree.",
              data={"data_dir": str(config.data_dir)}),
    ]


def check_gpu() -> list[Check]:
    if shutil.which("nvidia-smi") is None:
        return [Check("gpu", "driver", FAIL, "nvidia-smi not found",
                      fix="Install the NVIDIA driver. Without it every frame is "
                          "decoded and detected on the Grace cores.")]
    result = _run(["nvidia-smi",
                   "--query-gpu=name,driver_version,utilization.gpu,temperature.gpu",
                   "--format=csv,noheader,nounits"])
    if result is None or result.returncode != 0:
        return [Check("gpu", "driver", FAIL, "nvidia-smi failed to answer",
                      fix="Check `nvidia-smi` by hand; the driver may be mid-upgrade.")]
    line = (result.stdout or "").strip().splitlines()
    if not line:
        return [Check("gpu", "driver", FAIL, "nvidia-smi reported no GPU")]
    name, driver, util, temp = (part.strip() for part in line[0].split(",")[:4])
    checks = [Check("gpu", "driver", OK, f"{name}, driver {driver}, {util}% busy, {temp} C",
                    data={"name": name, "driver": driver})]

    # Torch is what actually has to see the GPU; nvidia-smi answering proves
    # only that the driver is loaded. This import is the slowest thing in the
    # doctor (~2 s) and is worth it: "nvidia-smi works but torch sees no CUDA"
    # is a real and confusing failure after a driver upgrade.
    try:
        import torch

        if torch.cuda.is_available():
            capability = ".".join(str(v) for v in torch.cuda.get_device_capability(0))
            checks.append(Check("gpu", "torch", OK,
                                f"torch {torch.__version__} sees {torch.cuda.get_device_name(0)} "
                                f"(sm_{capability.replace('.', '')})"))
        else:
            checks.append(Check("gpu", "torch", FAIL,
                                f"torch {torch.__version__} reports no CUDA device",
                                fix="`uv sync` reinstalls the cu130 build; a driver "
                                    "upgrade without a reboot also does this."))
    except Exception as exc:  # noqa: BLE001 - a broken torch is a finding, not a crash
        checks.append(Check("gpu", "torch", FAIL, f"torch unusable: {type(exc).__name__}: {exc}",
                            fix="uv sync"))
    return checks


# --------------------------------------------------------------------------
# memory, and who else is using it
# --------------------------------------------------------------------------
#
# The GB10's memory is *unified*: the detector, the VLM, the decoders and every
# other process on the box draw on one 121 GiB pool, and vLLM refuses to start
# unless `gpu_memory_utilization` x that pool is free at the moment it looks.
# Measured on this box on 2026-09-19: a collaborator's own llama.cpp server
# (63 GiB of CUDA allocations plus 32 GiB anonymous RSS — 95 GiB of the 121)
# had left 13.3 GiB, the VLM wanted 36.5, and `argus-vlm` had been dead since
# 2026-08-30 — twenty days in which five prompt policies judged nothing. This
# doctor reported `[ok] vlm` throughout, because what it checked was that the
# weights were on disk. A tenant check is the difference between "the VLM is
# broken" and "the box is full, and here is by whom".

_VLM_MEMORY_ERROR = "Free memory on device"


def gpu_tenants() -> list[dict[str, Any]]:
    """Compute processes holding GPU memory: pid, MiB, owner, command.

    `nvidia-smi` names the pid and the size; `/proc` names the owner, which is
    the half that matters — this appliance's own engine is a tenant too, and
    only someone else's is news.
    """
    result = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                   "--format=csv,noheader,nounits"])
    if result is None or result.returncode != 0:
        return []
    import pwd

    tenants = []
    for line in (result.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        try:
            mib = float(parts[1])
        except ValueError:
            mib = 0.0       # "[N/A]" on some drivers: still a tenant, size unknown
        try:
            uid = os.stat(f"/proc/{pid}").st_uid
            try:
                owner = pwd.getpwuid(uid).pw_name
            except KeyError:
                owner = str(uid)
            command = Path(f"/proc/{pid}/comm").read_text().strip()
        except OSError:
            continue        # exited between the two reads
        tenants.append({"pid": pid, "mib": mib, "owner": owner, "command": command,
                        "foreign": uid != os.getuid()})
    return tenants


def memory_gib() -> tuple[float, float] | None:
    """(available, total) GiB from /proc/meminfo.

    Deliberately not `torch.cuda.mem_get_info()`: that needs a CUDA context,
    a context costs a few hundred MiB, and the moment this check matters is
    exactly the moment the box cannot spare them.
    """
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        kib = lambda key: float(fields[key].split()[0])  # noqa: E731
        return kib("MemAvailable") / 1024**2, kib("MemTotal") / 1024**2
    except (OSError, KeyError, ValueError, IndexError):
        return None


def vlm_start_failure(config: ArgusConfig) -> str | None:
    """vLLM's own sentence for why it would not start, if memory was why."""
    log = config.logs_dir / "vlm-server.log"
    try:
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 200_000))
            tail = handle.read().decode("utf-8", "replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        if _VLM_MEMORY_ERROR in line:
            return line[line.index(_VLM_MEMORY_ERROR):].strip()
    return None


def check_memory(
    config: ArgusConfig,
    *,
    tenants: list[dict[str, Any]] | None = None,
    memory: tuple[float, float] | None = None,
    vlm_listening: bool | None = None,
) -> list[Check]:
    """Is there room for what this box is configured to run, and if not, who has it.

    The keyword arguments are the system reads, injectable so the arithmetic
    can be tested without a GPU or a second user.
    """
    memory = memory_gib() if memory is None else memory
    if memory is None:
        return [Check("gpu", "memory", SKIP, "/proc/meminfo unreadable")]
    available, total = memory
    tenants = gpu_tenants() if tenants is None else tenants
    foreign = sorted((t for t in tenants if t["foreign"]), key=lambda t: -t["mib"])
    foreign_gib = sum(t["mib"] for t in foreign) / 1024
    who = "; ".join(f"{t['command']} (pid {t['pid']}, user {t['owner']}) {t['mib'] / 1024:.1f} GiB"
                    for t in foreign[:3])
    data = {"available_gib": round(available, 1), "total_gib": round(total, 1),
            "foreign_gib": round(foreign_gib, 1), "foreign": foreign}

    if vlm_listening is None:
        vlm_listening = config.vlm.enabled and _listening(config.vlm.server.host,
                                                          config.vlm.server.port)
    if config.vlm.enabled and not vlm_listening:
        needed = config.vlm.server.gpu_memory_utilization * total
        if available < needed:
            detail = (f"the VLM cannot start: it needs {needed:.1f} GiB free "
                      f"(gpu_memory_utilization {config.vlm.server.gpu_memory_utilization:g} "
                      f"x {total:.1f} GiB) and {available:.1f} GiB is available")
            if foreign:
                detail += f" — held by another user: {who}"
            fix = ("Free the memory, then `systemctl --user reset-failed argus-vlm.service && "
                   "systemctl --user start argus-vlm.service`.")
            if foreign:
                fix = (f"Ask {foreign[0]['owner']} to stop {foreign[0]['command']} (this account "
                       f"cannot, and should not, stop another user's process). " + fix)
            return [Check("gpu", "memory", FAIL, detail, fix=fix, data=data)]

    # A tenth of the pool is the line: below it is somebody's notebook, above
    # it is a second product sharing the appliance.
    if foreign_gib > 0.10 * total:
        return [Check("gpu", "memory", WARN,
                      f"{available:.1f} of {total:.1f} GiB available; another user's processes "
                      f"hold {foreign_gib:.1f} GiB of GPU memory — {who}",
                      fix="This box is sized for Argus alone; agree a budget with whoever owns "
                          "those processes before the next VLM restart needs the room.",
                      data=data)]
    return [Check("gpu", "memory", OK,
                  f"{available:.1f} of {total:.1f} GiB available, no other user's GPU processes",
                  data=data)]


def check_vlm_policies(
    config: ArgusConfig, database: Database | None = None, *, vlm_listening: bool | None = None,
) -> list[Check]:
    """What a dead VLM is costing, in policies — the line a manager reads.

    `[WARN] argus-vlm.service failed` is a sentence about systemd. "Five of
    this site's policies have judged nothing for twenty days" is the same fact
    as a sentence about the factory, and it is the one that gets acted on.
    """
    from argus import processing

    if not config.vlm.enabled:
        return []
    if not processing.is_enabled(config):
        return []           # paused on purpose; check_services already says so
    if vlm_listening is None:
        vlm_listening = _listening(config.vlm.server.host, config.vlm.server.port)
    if vlm_listening:
        return []
    try:
        db = database or Database(config.db_path)
        with db.conn() as conn:
            dark = [row["id"] for row in conn.execute(
                "SELECT id FROM policies WHERE kind = 'vlm' AND enabled = 1 ORDER BY id")]
            last = conn.execute(
                "SELECT MAX(e.ts) AS ts FROM events e JOIN policies p ON p.id = e.policy_id "
                "WHERE p.kind = 'vlm'").fetchone()["ts"]
    except sqlite3.Error as exc:
        return [Check("engine", "prompt policies", SKIP, f"database unreadable: {exc}")]
    if not dark:
        return []
    detail = (f"{len(dark)} enabled prompt polic{'y is' if len(dark) == 1 else 'ies are'} not being "
              f"evaluated — the VLM server is not running ({', '.join(dark[:4])}"
              f"{', …' if len(dark) > 4 else ''})")
    if last:
        detail += f"; their last event was {(time.time() - float(last)) / 86400:.1f} days ago"
    reason = vlm_start_failure(config)
    if reason:
        detail += f". vLLM's last word: {reason}"
    return [Check("engine", "prompt policies", FAIL, detail,
                  fix="See `gpu / memory` above, then `systemctl --user reset-failed "
                      "argus-vlm.service && systemctl --user start argus-vlm.service`.",
                  data={"policies": dark, "last_event_ts": last})]


# --------------------------------------------------------------------------
# binaries
# --------------------------------------------------------------------------


def check_ffmpeg() -> list[Check]:
    """The vendored ffmpeg, which is the only one the engine will use."""
    from argus.ingest.decoder import FFMPEG

    binary = Path(FFMPEG)
    if not binary.is_file():
        resolved = shutil.which(FFMPEG)
        if resolved is None:
            return [Check("binaries", "ffmpeg", FAIL, f"not found at {FFMPEG}",
                          fix="./scripts/build_ffmpeg.sh")]
        binary = Path(resolved)

    version = _run([str(binary), "-version"], timeout=15.0)
    if version is None or version.returncode != 0:
        return [Check("binaries", "ffmpeg", FAIL, f"{binary} will not run",
                      fix="./scripts/build_ffmpeg.sh")]
    banner = (version.stdout or "").splitlines()[0]
    gpl = "--enable-gpl" in (version.stdout or "")

    decoders = _run([str(binary), "-hide_banner", "-decoders"], timeout=15.0)
    nvdec = bool(decoders and "h264_cuvid" in (decoders.stdout or ""))
    hwaccels = _run([str(binary), "-hide_banner", "-hwaccels"], timeout=15.0)
    cuda = bool(hwaccels and "cuda" in (hwaccels.stdout or ""))

    checks = [Check("binaries", "ffmpeg", OK, f"{banner} at {binary}",
                    data={"path": str(binary), "gpl": gpl})]
    if nvdec and cuda:
        checks.append(Check("binaries", "ffmpeg nvdec", OK,
                            "h264_cuvid and cuda hwaccel present — hardware decode available"))
    else:
        checks.append(Check(
            "binaries", "ffmpeg nvdec", WARN,
            f"no hardware decode (cuvid={nvdec}, cuda hwaccel={cuda}) — every camera "
            "will be decoded on the CPU",
            fix="./scripts/build_ffmpeg.sh (the ffmpeg on PATH is usually built without CUDA)"))
    if gpl:
        checks.append(Check(
            "binaries", "ffmpeg licence", FAIL,
            "this ffmpeg is built --enable-gpl and cannot ship inside the appliance",
            fix="./scripts/build_ffmpeg.sh builds an LGPL one into bin/"))
    return checks


def check_gateway(config: ArgusConfig) -> list[Check]:
    binary = config.gateway_binary
    if not binary.is_file():
        return [Check("binaries", "go2rtc", FAIL, f"not found at {binary}",
                      fix="./scripts/fetch_gateway.sh")]
    if not os.access(binary, os.X_OK):
        return [Check("binaries", "go2rtc", FAIL, f"{binary} is not executable",
                      fix=f"chmod +x {binary}")]
    result = _run([str(binary), "--version"])
    version = ""
    if result is not None:
        text = (result.stdout or result.stderr or "").strip()
        version = text.splitlines()[0] if text else ""
    return [Check("binaries", "go2rtc", OK, version or str(binary),
                  data={"path": str(binary)})]


# --------------------------------------------------------------------------
# model weights
# --------------------------------------------------------------------------


def check_models(config: ArgusConfig) -> list[Check]:
    checks: list[Check] = []

    variant = config.detection.model.variant
    matches = sorted(config.models_dir.glob(f"rf-detr-{variant}*.pth"))
    if matches:
        size = round(matches[0].stat().st_size / 1e6, 1)
        checks.append(Check("models", "detector", OK,
                            f"rf-detr-{variant}: {matches[0].name} ({size} MB)",
                            data={"path": str(matches[0])}))
    else:
        # This is a failure rather than a warning because the units set
        # HF_HUB_OFFLINE=1 and RF-DETR fetches from storage.googleapis.com: a
        # missing checkpoint on a site with no uplink is an engine that never
        # produces a frame, discovered at 3am.
        checks.append(Check("models", "detector", FAIL,
                            f"no rf-detr-{variant} checkpoint under {config.models_dir}",
                            fix="./scripts/fetch_detector.sh"))

    # A site-specific fine-tune, if this deployment adopted one. FAIL rather
    # than WARN for the same reason as above and one more: the engine refuses to
    # start on an unusable checkpoint, so a box in this state is not degraded,
    # it is down — and `argus doctor` is what somebody runs to find out why
    # before restarting it a second time.
    finetuned = config.detector_checkpoint_path
    if finetuned is not None:
        from argus.vision.checkpoint import CheckpointError, load_provenance

        try:
            provenance = load_provenance(finetuned)
        except CheckpointError as exc:
            checks.append(Check("models", "detector fine-tune", FAIL, str(exc),
                                fix="Fix or clear detection.model.checkpoint in "
                                    "config/argus.yaml — empty means stock COCO weights",
                                data={"path": str(finetuned)}))
        else:
            checks.append(Check("models", "detector fine-tune", OK, provenance.summary(),
                                data=provenance.as_dict()))

    index_path = config.index_model_path
    if config.index.model == "fake":
        checks.append(Check("models", "index", WARN,
                            "index.model is 'fake' — the deterministic stub, not SigLIP",
                            fix="Set index.model back to a checkpoint id and "
                                "./scripts/fetch_siglip.sh"))
    elif (index_path / "config.json").is_file() and any(index_path.glob("*.safetensors")):
        checks.append(Check("models", "index", OK, f"{config.index.model} at {index_path}"))
    elif config.index.enabled:
        checks.append(Check("models", "index", FAIL,
                            f"{config.index.model} missing from {index_path} "
                            "(index.enabled is true)",
                            fix="./scripts/fetch_siglip.sh"))
    else:
        checks.append(Check("models", "index", WARN,
                            f"{config.index.model} missing and indexing is disabled",
                            fix="./scripts/fetch_siglip.sh"))

    from argus.vlm import server as vlm_server

    installed = vlm_server.installed(config)
    if not config.vlm.enabled:
        checks.append(Check("models", "vlm", SKIP, "vlm.enabled is false"))
    elif installed["venv_ready"] and installed["model_ready"]:
        checks.append(Check("models", "vlm", OK,
                            f"{config.vlm.model} and its serving venv are present"))
    else:
        missing = []
        if not installed["venv_ready"]:
            missing.append(f"venv {installed['venv']}")
        if not installed["model_ready"]:
            missing.append(f"weights {installed['model_path']}")
        # Warning, not failure: the appliance is specified to run without a VLM
        # and the console reports it as down. Only prompt policies stop working.
        checks.append(Check("models", "vlm", WARN,
                            "optional VLM not installed — missing " + ", ".join(missing),
                            fix="./scripts/setup_vlm_server.sh && ./scripts/fetch_vlm.sh"))
    return checks


def check_console(config: ArgusConfig) -> list[Check]:
    dist = config.ui_dist_dir
    if (dist / "index.html").is_file():
        assets = len(list((dist / "assets").glob("*"))) if (dist / "assets").is_dir() else 0
        return [Check("console", "ui", OK, f"built console at {dist} ({assets} assets)")]
    return [Check("console", "ui", WARN,
                  f"no built console at {dist} — `argus api` will serve the JSON API alone",
                  fix="(cd ui && npm ci && npm run build)")]


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------


def check_database(config: ArgusConfig, database: Database | None = None) -> list[Check]:
    db = database or Database(config.db_path)
    if not config.db_path.is_file():
        return [Check("database", "schema", FAIL, f"no database at {config.db_path}",
                      fix="argus seed")]
    checks: list[Check] = []
    try:
        version = db.schema_version()
    except sqlite3.Error as exc:
        return [Check("database", "schema", FAIL, f"unreadable: {exc}",
                      fix="argus backup list, then argus backup restore <ts>")]

    if version == SCHEMA_VERSION:
        checks.append(Check("database", "schema", OK, f"v{version} (current)"))
    elif version < SCHEMA_VERSION:
        checks.append(Check("database", "schema", WARN,
                            f"v{version}, code expects v{SCHEMA_VERSION}",
                            fix="Any argus command applies pending migrations; "
                                "`argus status` is the cheapest."))
    else:
        checks.append(Check("database", "schema", FAIL,
                            f"v{version} is newer than this build (v{SCHEMA_VERSION})",
                            fix="This database was written by a newer Argus. Upgrade "
                                "the software rather than downgrading the schema."))

    try:
        with db.conn() as conn:
            integrity = conn.execute("PRAGMA quick_check").fetchone()[0]
    except sqlite3.Error as exc:
        integrity = f"unreadable: {exc}"
    checks.append(Check("database", "integrity",
                        OK if integrity == "ok" else FAIL,
                        f"quick_check: {integrity}",
                        fix=None if integrity == "ok" else "argus backup restore latest"))

    try:
        counts = db.counts()
        size = round(config.db_path.stat().st_size / 1e6, 1)
        checks.append(Check("database", "contents", OK,
                            f"{size} MB — " + ", ".join(f"{k}={v}" for k, v in counts.items()),
                            data=counts))
    except sqlite3.Error:
        pass

    if config.index.enabled or config.index_db_path.is_file():
        if config.index_db_path.is_file():
            size = round(config.index_db_path.stat().st_size / 1e6, 1)
            checks.append(Check("database", "index store", OK,
                                f"{config.index_db_path.name}: {size} MB"))
        else:
            checks.append(Check("database", "index store", WARN,
                                "no vector store yet — nothing has been indexed",
                                fix="argus index backfill"))
    return checks


def check_accounts(config: ArgusConfig, database: Database | None = None) -> list[Check]:
    from argus import accounts

    db = database or Database(config.db_path)
    try:
        rows = db.list_users()
    except sqlite3.Error as exc:
        # Almost always "no such table: users" on a box that has never been
        # seeded, which is a fixable state rather than a mystery.
        return [Check("accounts", "roster", FAIL, f"unreadable: {exc}",
                      fix="argus seed, then argus user add ops --role ADMIN")]
    hint = accounts.bootstrap_hint(db)
    if hint:
        return [Check("accounts", "roster", FAIL, hint,
                      fix="argus user add ops --role ADMIN")]
    active = [r for r in rows if not r["disabled"]]
    admins = [r for r in active if r["role"] == "ADMIN"]
    return [Check("accounts", "roster", OK,
                  f"{len(active)} active account(s), {len(admins)} admin(s)",
                  data={"total": len(rows), "active": len(active)})]


def check_disk(config: ArgusConfig) -> list[Check]:
    from argus.health import disk_info

    info = disk_info(config)
    limit = config.retention.max_disk_percent
    percent = info["percent"]
    if percent > limit:
        status, detail = FAIL, (f"{percent}% full, over the {limit}% guard — retention "
                                "is reclaiming oldest-first")
    elif percent > limit - 10:
        status, detail = WARN, f"{percent}% full, approaching the {limit}% guard"
    else:
        status, detail = OK, f"{percent}% full, {info['free_gb']} GB free"
    breakdown = ", ".join(f"{name} {mb:.0f} MB"
                          for name, mb in sorted(info["by_directory_mb"].items())
                          if mb >= 1)
    checks = [
        Check("storage", "filesystem", status, detail,
              fix=None if status != FAIL else "argus retention",
              data={"percent": percent, "free_gb": info["free_gb"]}),
        Check("storage", "argus", OK, f"{info['argus_mb']:.0f} MB total — {breakdown or 'empty'}",
              data=info["by_directory_mb"]),
    ]

    # The evidence budget, separately from the filesystem — because the two
    # answer different questions and this box proved it. `data/clips` reached
    # 578 GB on a 3.7 TB disk that was still only 36% full, so every check that
    # watched the *filesystem* said OK while the appliance was 2.5 days from
    # needing one. Over budget is a WARN and not a FAIL: the next sweep evicts
    # it automatically, and an install script should not exit non-zero because
    # housekeeping is due.
    evidence = info["evidence"]
    over = [name for name, kind in evidence["kinds"].items() if kind["over_budget"]]
    unbounded = [name for name, kind in evidence["kinds"].items()
                 if kind["budget_gb"] is None]
    def gb(value: float) -> str:
        # A support tool prints what it measured. `0.1 GB/0 GB` on a small
        # appliance is a rounding artefact that reads as a division by zero.
        return f"{value:.0f}" if value >= 10 or value == int(value) else f"{value:.2f}"

    usage = ", ".join(
        f"{name} {gb(kind['used_gb'])} GB"
        + (f"/{gb(kind['budget_gb'])} GB ({kind['percent']:.0f}%)"
           if kind["budget_gb"] else " (no budget)")
        for name, kind in sorted(evidence["kinds"].items())
    )
    if over:
        checks.append(Check(
            "storage", "evidence budget", WARN,
            f"over budget and not reclaimed: {usage} — the sweep has not kept up, "
            "or min_keep_hours is protecting evidence too young to evict",
            fix="argus retention", data=evidence["kinds"]))
    elif unbounded:
        checks.append(Check(
            "storage", "evidence budget", WARN,
            f"{usage} — no hard ceiling on " + " and ".join(unbounded)
            + "; ages alone do not bound a site that raises more events than expected",
            fix="Set retention.max_clips_gb (and max_snapshots_gb) in config/argus.yaml",
            data=evidence["kinds"]))
    else:
        tiering = ("per-stage windows in force" if evidence["tiered"]
                   else f"flat windows (clips {config.retention.clips_days} d)")
        # Sitting exactly at the ceiling is the budget doing its job, not a
        # fault — the sweep reclaims to the budget and the cameras keep
        # recording. Worth saying out loud, because it tells an operator that
        # this site's evidence is bounded by size rather than by age.
        held = " — held at the ceiling by the sweep" if evidence["at_budget"] else ""
        checks.append(Check("storage", "evidence budget", OK,
                            f"{usage} — {tiering}{held}", data=evidence["kinds"]))
    return checks


def check_backups(config: ArgusConfig) -> list[Check]:
    from argus import backup

    if not config.backup.enabled:
        return [Check("backup", "schedule", WARN, "backup.enabled is false — no snapshots",
                      fix="Set backup.enabled: true in config/argus.yaml")]

    entries = backup.list_backups(config)
    complete = [e for e in entries if e.complete]
    if not complete:
        return [Check("backup", "latest", FAIL,
                      f"no complete backup in {backup.backup_dir(config)}",
                      fix="argus backup now")]
    latest = complete[0]
    age_hours = (latest.age_sec() or 0.0) / 3600.0
    overdue = age_hours > config.backup.interval_hours * 2
    checks = [Check("backup", "latest", WARN if overdue else OK,
                    f"{latest.stamp} — {age_hours:.1f} h old, {latest.mb} MB, "
                    f"schema v{latest.schema_version}",
                    fix="argus backup now" if overdue else None,
                    data={"stamp": latest.stamp, "mb": latest.mb})]
    checks.append(Check("backup", "generations", OK,
                        f"{len(complete)} kept of {config.backup.keep} in "
                        f"{backup.backup_dir(config)}"))

    # A copy on the disk it is protecting is a copy of a mistake, not of a
    # failed disk. Said out loud once, here, rather than implied nowhere.
    try:
        same = (os.stat(backup.backup_dir(config)).st_dev == os.stat(config.data_dir).st_dev)
    except OSError:
        same = True
    if same:
        checks.append(Check("backup", "destination", WARN,
                            "backups are on the same filesystem as the data they protect",
                            fix="Set backup.directory to a mounted share to survive "
                                "this disk failing."))
    incomplete = [e for e in entries if not e.complete]
    if incomplete:
        checks.append(Check("backup", "incomplete", WARN,
                            f"{len(incomplete)} interrupted backup(s): "
                            + ", ".join(e.stamp for e in incomplete[:3]),
                            fix="argus backup now (prunes them)"))
    return checks


def check_logs(config: ArgusConfig) -> list[Check]:
    directory = config.logs_dir
    if not directory.is_dir():
        return [Check("logs", "rotation", OK, "no log directory yet")]
    files = sorted(directory.glob("*.log"))
    if not files:
        return [Check("logs", "rotation", OK, "no supervised-child logs yet")]
    limit = config.logs.max_mb * 1e6
    oversized = [f for f in files if f.stat().st_size > limit]
    summary = ", ".join(f"{f.name} {f.stat().st_size / 1e6:.1f} MB" for f in files)
    if not config.logs.rotate:
        return [Check("logs", "rotation", WARN, f"rotation disabled — {summary}",
                      fix="Set logs.rotate: true in config/argus.yaml")]
    if oversized:
        return [Check("logs", "rotation", WARN,
                      f"over the {config.logs.max_mb} MB threshold and not yet rotated: "
                      + ", ".join(f.name for f in oversized),
                      fix="argus retention (housekeeping rotates logs), or wait for "
                          "the next sweep")]
    return [Check("logs", "rotation", OK, summary)]


# --------------------------------------------------------------------------
# services and ports
# --------------------------------------------------------------------------


def check_services(config: ArgusConfig) -> list[Check]:
    from argus import processing, services

    directory = services.unit_dir()
    installed = [name for name in services.UNIT_NAMES if (directory / name).is_file()]
    if not installed:
        return [Check("services", "units", WARN,
                      f"no systemd units in {directory} — Argus is being run by hand",
                      fix="argus install-services")]

    # When processing is paused the engine and VLM are *meant* to be inactive
    # (their ExecCondition skips the start), so their being down is not a fault.
    paused = not processing.is_enabled(config)

    checks: list[Check] = []
    state = services.states(installed)
    for name in installed:
        info = state.get(name, {})
        active = info.get("active_state", "unknown")
        enabled = info.get("unit_file_state", "")
        restarts = info.get("restarts", "0")
        outcome = info.get("result", "")
        detail = f"{active}/{info.get('sub_state', '')} ({enabled or 'not enabled'})"
        if restarts and restarts not in ("0", ""):
            detail += f", {restarts} restart(s)"
        if outcome and outcome != "success":
            detail += f", last result: {outcome}"
        # `Result` rather than a restart count: a unit restarted once an hour
        # ago and then stopped on purpose is stopped, not broken.
        never_started = active == "inactive" and outcome in ("success", "")
        if (paused and name in (services.ENGINE_UNIT, services.VLM_UNIT)
                and active != "active"):
            # Paused on purpose: the ExecCondition kept it down. Any not-running
            # state is expected here — including `failed`, which is how a vLLM
            # that had to be SIGKILLed on stop (it was mid-load) reports, and
            # which self-heals on the next resume.
            status = OK
            detail += " — processing paused"
        elif name == services.VLM_UNIT and active in ("inactive", "unknown"):
            status = SKIP  # optional equipment, off by default
        elif name == services.VLM_UNIT and active == "failed" and config.vlm.enabled:
            # `vlm.enabled: true` is this site saying it depends on the VLM, and
            # the unit file being `disabled` (it is started by argus.target, not
            # on its own) used to route a crashed server into the WARN below —
            # one quiet line among thirty, for twenty days, on 2026-08-30.
            status = FAIL
        elif active == "active":
            status = OK
        elif name == services.TARGET_UNIT:
            status = OK if active in ("active", "inactive") else WARN
        elif enabled != "enabled":
            status = WARN
        elif never_started:
            # "Enabled and not running, and the last time it ran it exited
            # cleanly" is "you have not turned it on yet" — a different phone
            # call from "it died", which is what a non-success Result means.
            status = WARN
            detail += " — enabled but not running"
        else:
            status = FAIL
        fix = None
        if status == WARN and never_started and enabled == "enabled":
            fix = f"systemctl --user start {services.TARGET_UNIT}"
        elif status in (WARN, FAIL):
            fix = f"systemctl --user status {name}"
        checks.append(Check("services", name, status, detail, fix=fix, data=info))

    linger = services.linger_enabled()
    if linger is True:
        checks.append(Check("services", "linger", OK,
                            "enabled — units survive logout and start at boot"))
    elif linger is False:
        checks.append(Check("services", "linger", FAIL,
                            "disabled — every Argus service stops when this user logs out",
                            fix=f"sudo loginctl enable-linger {os.environ.get('USER', '$USER')}"))
    else:
        checks.append(Check("services", "linger", SKIP, "loginctl unavailable"))
    return checks


def local_address() -> str | None:
    """This box's primary address, found without sending a packet or resolving
    a name: connecting a UDP socket only picks a route."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1, never routed
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


def _listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(_PROBE_TIMEOUT)
        return sock.connect_ex((host, port)) == 0


def check_ports(config: ArgusConfig) -> list[Check]:
    """Which doors are open, and to whom.

    This is the check that earns the doctor its place in a security review: it
    answers "is the gateway's unauthenticated API reachable from the plant LAN"
    with a connection attempt rather than with a config value, and the two have
    disagreed before.
    """
    lan = local_address()
    checks: list[Check] = []

    def probe(name: str, port: int, *, lan_expected: bool, note: str) -> None:
        loopback = _listening("127.0.0.1", port)
        exposed = _listening(lan, port) if (lan and loopback) else False
        if not loopback:
            checks.append(Check("ports", name, SKIP, f"nothing listening on :{port}"))
            return
        where = f"127.0.0.1:{port}" + (f" and {lan}:{port}" if exposed else " only")
        if exposed and not lan_expected:
            checks.append(Check("ports", name, FAIL,
                                f"reachable from the network on {lan}:{port} — {note}",
                                fix="Set gateway.api_host: 127.0.0.1 and restart the engine.",
                                data={"port": port, "exposed": True}))
        else:
            checks.append(Check("ports", name, OK, f"{where} — {note}",
                                data={"port": port, "exposed": exposed}))

    probe("api", config.api.port, lan_expected=True,
          note="the console and JSON API, session-authenticated")
    probe("gateway api", config.gateway.api_port, lan_expected=False,
          note="go2rtc's own unauthenticated API and admin UI: loopback only by design")
    probe("gateway rtsp", config.gateway.rtsp_port, lan_expected=True,
          note="internal RTSP; open to the LAN so a site's VMS can re-use the streams")
    probe("gateway webrtc", config.gateway.webrtc_port, lan_expected=True,
          note="WebRTC media; open by necessity — ICE connects the browser directly "
               "(docs/SECURITY.md, 'Residual')")
    if config.vlm.enabled:
        probe("vlm", config.vlm.server.port, lan_expected=False,
              note="the local inference server: loopback only")

    if lan is None:
        checks.append(Check("ports", "address", WARN,
                            "could not determine this box's own address; only loopback "
                            "was probed"))
    return checks


def check_engine(config: ArgusConfig, database: Database | None = None) -> list[Check]:
    from argus import processing
    from argus.health import read_engine_stats

    stats = read_engine_stats(config)
    if not stats.get("running"):
        # A paused appliance's engine is *meant* to be down — that is the whole
        # point of the switch, and reporting it as a warning would train an
        # operator to ignore the one check that also catches a real crash.
        if not processing.is_enabled(config):
            return [Check("engine", "running", OK,
                          "processing is paused — engine intentionally stopped, GPU idle",
                          fix="Resume from the console, or `argus processing resume`.")]
        return [Check("engine", "running", WARN,
                      f"not running — {stats.get('reason') or 'no statistics published'}",
                      fix="systemctl --user start argus-engine  (or `argus run`)")]

    cameras = stats.get("cameras") or []
    live = sum(1 for c in cameras if c.get("status") == "live")
    uptime = (stats.get("uptime_sec") or 0) / 3600.0
    checks = [Check("engine", "running", OK,
                    f"up {uptime:.1f} h, {live}/{len(cameras)} cameras live",
                    data={"cameras": len(cameras), "live": live})]
    checks.extend(_check_enrolled_but_not_decoding(config, database, stats))
    return checks


def _check_enrolled_but_not_decoding(
    config: ArgusConfig, database: Database | None, stats: dict[str, Any]
) -> list[Check]:
    """Cameras the operator enrolled that this engine has never picked up.

    The engine reads the camera table **once**, at startup: ``Engine._refresh``
    re-reads zones and policies every 15 s so an edit lands without a restart,
    but a camera means a decoder process and a regenerated gateway config, and
    neither is hot-swapped. That is a defensible design; being quiet about it
    is not.

    Before this check, enrolling a camera from the console produced three
    reassuring surfaces and no working camera: the console listed it ``offline``
    (which is what a *broken* camera also says), the engine's own line read
    "2/2 cameras live" because two is all it had ever been told about, and
    ``argus doctor`` — the one command support asks a site to run — printed that
    same 2/2 in green. The camera had probed ``OPENED`` a minute earlier.
    """
    db = database or Database(config.db_path)
    try:
        enrolled = {str(row["id"]): row for row in db.list_cameras(enabled_only=True)}
    except sqlite3.Error as exc:
        return [Check("engine", "fleet", WARN,
                      f"could not read the camera table: {exc}",
                      fix="See the `database` checks above — this one cannot answer "
                          "until the database can be read.")]

    running = {str(camera.get("camera_id")) for camera in (stats.get("cameras") or [])}
    missing = [row for camera_id, row in enrolled.items() if camera_id not in running]
    if not missing:
        return []

    names = ", ".join(sorted(str(row["id"]) for row in missing))
    return [Check(
        "engine", "fleet", WARN,
        f"{len(missing)} enabled camera(s) are enrolled but not being decoded: {names}",
        fix="The engine reads the camera table at startup only. Restart it: "
            "systemctl --user restart argus-engine  (or stop and re-run `argus run`).",
        data={"enrolled": len(enrolled), "running": len(running),
              "missing": sorted(str(row["id"]) for row in missing)},
    )]


# --------------------------------------------------------------------------
# the whole examination
# --------------------------------------------------------------------------


def run_checks(config: ArgusConfig, database: Database | None = None) -> list[Check]:
    checks: list[Check] = []
    for producer in (
        lambda: check_host(config),
        check_gpu,
        lambda: check_memory(config),
        check_ffmpeg,
        lambda: check_gateway(config),
        lambda: check_models(config),
        lambda: check_console(config),
        lambda: check_database(config, database),
        lambda: check_accounts(config, database),
        lambda: check_disk(config),
        lambda: check_backups(config),
        lambda: check_logs(config),
        lambda: check_services(config),
        lambda: check_ports(config),
        lambda: check_engine(config, database),
        lambda: check_vlm_policies(config, database),
    ):
        try:
            checks.extend(producer())
        except Exception as exc:  # noqa: BLE001 - a check that crashes is a finding
            checks.append(Check("doctor", getattr(producer, "__name__", "check"), FAIL,
                                f"check crashed: {type(exc).__name__}: {exc}"))
    return checks


def summarise(checks: list[Check]) -> dict[str, int]:
    counts = {OK: 0, WARN: 0, FAIL: 0, SKIP: 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return counts


def exit_code(checks: list[Check], *, strict: bool = False) -> int:
    counts = summarise(checks)
    if counts.get(FAIL):
        return 1
    if strict and counts.get(WARN):
        return 1
    return 0


_GLYPH = {OK: "ok  ", WARN: "WARN", FAIL: "FAIL", SKIP: "--  "}


def format_report(checks: list[Check], config: ArgusConfig, *, elapsed: float = 0.0) -> str:
    lines: list[str] = [f"argus doctor — {ROOT}", ""]
    group = None
    for check in checks:
        if check.group != group:
            group = check.group
            lines.append(f"{group}:")
        lines.append(f"  [{_GLYPH[check.status]}] {check.name:<20} {check.detail}")
        if check.fix and check.status in (WARN, FAIL):
            lines.append(f"{'':<10}-> {check.fix}")
    counts = summarise(checks)
    lines.append("")
    lines.append(
        f"{counts[OK]} ok, {counts[WARN]} warning(s), {counts[FAIL]} failure(s), "
        f"{counts[SKIP]} skipped in {elapsed:.1f}s"
    )
    if counts[FAIL]:
        lines.append("FAILING — the arrows above are the commands that fix it.")
    return "\n".join(lines)


def report(config: ArgusConfig, database: Database | None = None) -> tuple[list[Check], float]:
    started = time.monotonic()
    checks = run_checks(config, database)
    # Stable order: groups in the order they were produced, failures first
    # inside each group, so the thing to act on is never buried.
    order = {name: index for index, name in
             enumerate(dict.fromkeys(check.group for check in checks))}
    checks.sort(key=lambda c: (order[c.group], _ORDER[c.status]))
    return checks, time.monotonic() - started


__all__ = [
    "FAIL",
    "OK",
    "SKIP",
    "WARN",
    "Check",
    "exit_code",
    "format_report",
    "local_address",
    "report",
    "run_checks",
    "summarise",
]
