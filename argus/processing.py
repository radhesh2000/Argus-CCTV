"""The appliance's pause switch: is the GPU plane allowed to run?

The product's control plane — the API and the console it serves — is *always*
up, because taking a plant's monitoring offline is a thing an operator does on
purpose and then needs a way to undo. The **GPU workloads** are the toggled
thing: the vision engine (`argus run`: decode, detect, track, index, evidence,
housekeeping) and the VLM inference server. This module owns the one durable bit
that decides whether they run, and the one mechanism that actuates it.

Where the bit lives, and why a file
-----------------------------------
A single JSON document under ``data_dir`` — ``data/processing.json`` — beside
the engine's other cross-process state (``engine_stats.json``, the vLLM pidfile).
It is durable, it is trivially inspectable with ``cat``, and — the property that
matters most — it is read **at unit-start time**, so a rebooted appliance that
was paused comes up paused. A row in the database would have been the other
option and was rejected: the gate that reads this (see :func:`is_enabled` and
``argus processing gate``, wired as the units' ``ExecCondition``) must answer
before the engine has opened anything, and must not depend on a schema
migration having run first.

Why not ``systemctl disable``
-----------------------------
The obvious "don't start it" lever is to disable the unit. It is the wrong one:
``systemctl --user disable argus-engine`` fights the operator who then runs
``systemctl --user restart argus.target``, and it conflates "the site paused
processing" with "this box was never provisioned to process". The flag keeps the
units *enabled* — what changes is that each one asks this file, at every start
attempt, whether it is allowed to proceed. A manual ``restart argus.target`` on
a paused box therefore brings the API up and leaves the GPU idle, which is the
whole point.

The tradeoff, stated plainly
----------------------------
Retention and the nightly backup run on a thread *inside* ``argus run`` (see
``argus/engine.py``), so while processing is paused they do not run. This is
benign and mostly self-correcting: a paused appliance writes no new clips,
snapshots or events, so there is nothing for the age sweep or the budget to
reclaim; and the backup is *age*-based (``backup.interval_hours``), so the first
housekeeping tick after a resume takes the snapshot that came due while paused.
A site that pauses for weeks and wants a backup in the meantime runs
``argus backup now`` or ``argus retention`` by hand — both work with the engine
down, by construction.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from argus.config import ArgusConfig
from argus.services import (
    ENGINE_UNIT,
    VLM_UNIT,
    ServiceError,
    control,
    states,
    systemctl_available,
)

#: The file under ``data_dir`` that holds the desired state. Named for what it
#: answers, not for the module.
STATE_FILENAME = "processing.json"

#: The one key that decides everything. Spelled out (not ``enabled``) so a human
#: reading ``cat data/processing.json`` knows what it governs without context.
FLAG_KEY = "processing_enabled"


def state_path(config: ArgusConfig) -> Path:
    """Where the flag lives — beside the engine's other cross-process state."""
    return config.data_dir / STATE_FILENAME


# --------------------------------------------------------------------------
# the desired state, read and written
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProcessingState:
    """The persisted desired-state, plus the small audit trail of who set it."""

    enabled: bool
    updated_at: str | None = None
    updated_by: str | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            FLAG_KEY: self.enabled,
            "updated_at": self.updated_at,
            "updated_by": self.updated_by,
            "note": self.note,
        }


def read_state(config: ArgusConfig) -> ProcessingState:
    """Read the flag. **A missing or unreadable file means enabled.**

    The bias is deliberate and it is toward *watching the plant*: an appliance
    whose whole job is safety monitoring must not silently stop because a file
    is absent (a fresh, never-configured box) or has a stray byte in it. Pausing
    is an explicit act that leaves a valid document behind (see
    :func:`write_state`, which writes atomically so a half-write cannot happen);
    anything else resolves to "run".
    """
    path = state_path(config)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return ProcessingState(enabled=True)
    except (OSError, json.JSONDecodeError):
        # Present but unreadable — a bug, not a pause. Fail toward running and
        # let the engine's own startup surface anything genuinely wrong.
        return ProcessingState(enabled=True, note="processing.json unreadable; assuming enabled")
    if not isinstance(payload, dict) or FLAG_KEY not in payload:
        return ProcessingState(enabled=True, note="processing.json malformed; assuming enabled")
    return ProcessingState(
        enabled=bool(payload.get(FLAG_KEY)),
        updated_at=payload.get("updated_at"),
        updated_by=payload.get("updated_by"),
        note=payload.get("note"),
    )


def is_enabled(config: ArgusConfig) -> bool:
    """Whether the GPU plane is allowed to run. Never raises — the gate that
    depends on it must answer even for a broken config."""
    try:
        return read_state(config).enabled
    except Exception:  # noqa: BLE001 - a gate that raises is a gate that hangs boot
        return True


def write_state(config: ArgusConfig, enabled: bool, *, actor: str | None = None,
                note: str | None = None) -> ProcessingState:
    """Persist the desired state atomically (tmp + ``os.replace``).

    Atomic because this file is read by a boot-time gate in another process: a
    reader must never see a half-written document, so it is never written in
    place. Returns the state as stored, timestamp and all.
    """
    state = ProcessingState(
        enabled=bool(enabled),
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        updated_by=actor,
        note=note,
    )
    path = state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state.as_dict(), indent=2) + "\n")
    os.replace(tmp, path)
    return state


# --------------------------------------------------------------------------
# what the flag governs
# --------------------------------------------------------------------------


def vlm_should_run(config: ArgusConfig) -> bool:
    """Whether *this* box has a VLM worth starting on resume.

    The engine always runs when processing is enabled; the VLM is optional
    equipment. It is only started — and only ever expected to answer — when it
    is enabled in config and actually installed (venv + weights on disk). A box
    with the model disabled or never fetched must not be told it is "starting" a
    server that will never come, and resume must not try to start a unit that
    would only fail.
    """
    if not config.vlm.enabled:
        return False
    from argus.vlm.server import paths_for

    paths = paths_for(config)
    return paths.venv_ready and paths.model_ready


def warmup_grace_sec(config: ArgusConfig) -> float:
    """How long after a resume a not-yet-answering VLM still reads as *warming
    up* rather than *down*. The model's own cold-start budget — generous by
    construction (docs/PHASE2_NOTES.md: a cold 8B is ~2 min, a 27B FP8 longer),
    and the number past which a silent VLM is a health problem, not a resume in
    progress."""
    return float(config.vlm.server.startup_timeout_sec)


def _target_units(config: ArgusConfig, *, enabled: bool) -> list[str]:
    """Which units an actuation touches. Resume starts only what should run;
    pause stops both (stopping an already-stopped unit is a no-op, and a VLM
    that was left running by hand must still come down)."""
    if enabled:
        units = [ENGINE_UNIT]
        if vlm_should_run(config):
            units.append(VLM_UNIT)
        return units
    return [ENGINE_UNIT, VLM_UNIT]


# --------------------------------------------------------------------------
# the derived, human-facing status
# --------------------------------------------------------------------------


def processing_view(
    config: ArgusConfig,
    *,
    engine_running: bool,
    engine_uptime_sec: float | None,
    vlm_available: bool,
    state: ProcessingState | None = None,
) -> dict[str, Any]:
    """The light, subprocess-free status block for ``/api/live/state`` and
    ``/api/health`` — the one the console polls to drive the banner.

    ``status`` is the headline, keyed on the engine and bounded by the VLM's
    cold-start grace:

    * ``paused``   — the flag is off. The GPU plane is intentionally down.
    * ``starting`` — the flag is on but the box has not settled: the engine is
      not yet publishing stats, or it is and the VLM is still within its
      cold-start grace. This is the "models warming up" state.
    * ``running``  — the flag is on, the engine is up, and the VLM is either
      answering, not required, or has been given longer than its grace to answer
      (past which a silent model is a *health* problem shown on Model Health,
      not a resume still in progress — so the banner stops crying wolf).

    ``vlm`` is the model's own sub-status, for the panels that show it in full.
    """
    state = state or read_state(config)
    required = vlm_should_run(config)
    uptime = float(engine_uptime_sec or 0.0)
    within_grace = uptime < warmup_grace_sec(config)

    if not state.enabled:
        status, vlm_status = "paused", "paused"
    elif not engine_running:
        status = "starting"
        vlm_status = "starting" if required else "disabled"
    elif not required:
        status, vlm_status = "running", "disabled"
    elif vlm_available:
        status, vlm_status = "running", "ready"
    elif within_grace:
        status, vlm_status = "starting", "starting"
    else:
        # Engine healthy, model silent well past its cold-start budget: degraded,
        # not warming. Model Health owns the alarm; the banner clears.
        status, vlm_status = "running", "down"

    return {
        FLAG_KEY: state.enabled,
        "status": status,
        "vlm": vlm_status,
        "vlm_required": required,
        "updated_at": state.updated_at,
        "updated_by": state.updated_by,
    }


def _service_state(unit_info: dict[str, str], *, enabled: bool,
                   required: bool = True) -> str:
    """Map one unit's systemd ``active_state`` — and the desired flag — onto a
    word the console renders. A unit skipped by its ``ExecCondition`` looks
    exactly like a clean stop (``inactive``/``Result=success``), so the flag,
    not systemd, decides whether that reads as ``paused``."""
    active = unit_info.get("active_state", "unknown")
    if not required:
        return "disabled"
    if not enabled:
        return "paused"
    if active == "active":
        return "running"
    if active in ("activating", "reloading", "deactivating"):
        return "starting"
    if active == "failed":
        return "failed"
    # Enabled but not yet active: either the start job has not run yet, or a
    # crash the flag says should not have happened. "starting" is the honest read
    # right after a resume; a genuine failure shows as ``failed`` above.
    return "starting"


def processing_report(config: ArgusConfig, *, runner=None,
                      engine_running: bool = False,
                      engine_uptime_sec: float | None = None,
                      vlm_available: bool = False) -> dict[str, Any]:
    """The full control-plane view for ``GET /api/system/processing`` and
    ``argus processing`` — the light view plus each unit's actual systemd state.

    This one shells out (``systemctl show`` per unit via
    :func:`argus.services.states`), so it is the on-demand detail endpoint, not
    something polled every two seconds.
    """
    state = read_state(config)
    view = processing_view(
        config,
        engine_running=engine_running,
        engine_uptime_sec=engine_uptime_sec,
        vlm_available=vlm_available,
        state=state,
    )
    required = view["vlm_required"]
    unit_states = states([ENGINE_UNIT, VLM_UNIT], runner=runner)
    engine_info = unit_states.get(ENGINE_UNIT, {})
    vlm_info = unit_states.get(VLM_UNIT, {})
    return {
        **view,
        "systemctl": systemctl_available(),
        "services": {
            "engine": {
                "unit": ENGINE_UNIT,
                "active_state": engine_info.get("active_state"),
                "sub_state": engine_info.get("sub_state"),
                "state": _service_state(engine_info, enabled=state.enabled),
            },
            "vlm": {
                "unit": VLM_UNIT,
                "active_state": vlm_info.get("active_state"),
                "sub_state": vlm_info.get("sub_state"),
                "state": _service_state(vlm_info, enabled=state.enabled, required=required),
                "available": bool(vlm_available),
            },
        },
    }


# --------------------------------------------------------------------------
# actuation
# --------------------------------------------------------------------------


def set_processing(config: ArgusConfig, enabled: bool, *, actor: str | None = None,
                   note: str | None = None, runner=None) -> dict[str, Any]:
    """Flip the durable flag and actuate it.

    Order matters: the flag is written **first**, so that the units'
    ``ExecCondition`` gate agrees with the intent before anything is asked to
    start, and so a paused box stays paused even against a race. Then the two
    units are started or stopped.

    Non-blocking on purpose. ``argus-vlm`` is ``Type=forking`` and its
    ``ExecStart`` waits for the model to serve — minutes on a cold cache — so a
    blocking ``systemctl start`` would hang this call (and the HTTP request
    behind it) for the whole load. ``--no-block`` hands the job to systemd and
    returns; the caller learns readiness by polling health, where the engine
    shows ``running`` quickly and the VLM shows ``starting`` until it answers.

    Where there is no user systemd (a dev box, CI), the flag is still persisted
    and ``actuated`` comes back false — the desired state survives even when
    nothing here can enforce it.
    """
    state = write_state(config, enabled, actor=actor, note=note)
    units = _target_units(config, enabled=enabled)
    action = "start" if enabled else "stop"

    result: dict[str, Any] = {
        "state": state.as_dict(),
        FLAG_KEY: state.enabled,
        "action": action,
        "units": units,
        "actuated": False,
        "command": None,
        "detail": None,
    }

    if not systemctl_available():
        result["detail"] = (
            "flag persisted, but systemctl --user is unavailable on this box — "
            "nothing was started or stopped"
        )
        return result

    try:
        completed = control(action, units, no_block=True, runner=runner)
    except ServiceError as exc:
        result["detail"] = str(exc)
        return result

    result["command"] = f"systemctl --user {action} --no-block {' '.join(units)}"
    if completed.returncode != 0:
        result["detail"] = (
            (completed.stderr or completed.stdout or "").strip()
            or f"systemctl {action} exited {completed.returncode}"
        )
        return result
    result["actuated"] = True
    return result


__all__ = [
    "FLAG_KEY",
    "STATE_FILENAME",
    "ProcessingState",
    "is_enabled",
    "processing_report",
    "processing_view",
    "read_state",
    "set_processing",
    "state_path",
    "vlm_should_run",
    "warmup_grace_sec",
    "write_state",
]
