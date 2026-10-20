"""Import ``config/seed.yaml`` into the appliance database.

The v1 demo cameras, zones and policies existed only as rows in a SQLite file
that was never checked in — reinstalling meant recreating them by hand. This
makes the fleet declarative: the YAML is the source of truth, `argus seed` is
idempotent, and running it twice changes nothing.

Rows are upserted, never deleted: a camera removed from the YAML stays in the
database (it may own events and evidence). Deletion is an explicit operator
action, not a side effect of editing a file.

A camera may be marked ``optional: true``, which means "import this only if its
footage is on this box". That exists because ``config/seed.anmol.yaml`` enrols a
customer's video files that are gitignored: on the machine holding them it is a
six-camera plant, and on a fresh clone it has to be a clear skip rather than a
stack trace. The skip cascades — a zone belongs to a camera, and a policy scoped
only to skipped cameras is skipped too. That last rule is the one worth stating
out loud: an emptied scope means "every camera" everywhere else in this file, so
letting a scope *become* empty would silently point a plant's policy at the demo
fleet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from argus.config import ROOT
from argus.db import Camera, Database, Policy, Zone

DEFAULT_SEED_PATH = ROOT / "config" / "seed.yaml"

_CAMERA_KEYS = {"id", "name", "source", "enabled", "detect_fps", "kind", "optional"}
_ZONE_KEYS = {"id", "camera_id", "name", "polygon", "sanctioned_limit", "color"}
_POLICY_KEYS = {
    "id", "kind", "detector_key", "label", "enabled", "stage", "lifecycle", "severity",
    "routed_to", "prompt", "interval_sec", "cooldown_sec", "threshold", "pilot_duration_days",
    "cameras",
}


class SeedError(ValueError):
    """Raised when the seed file is malformed."""


@dataclass(slots=True)
class SeedResult:
    cameras: int = 0
    zones: int = 0
    policies: int = 0
    skipped_cameras: int = 0
    skipped_zones: int = 0
    skipped_policies: int = 0
    # One line per skip, in the order they happened, so `argus seed` can say
    # *which* file it could not find rather than a count. A silent skip and a
    # successful import look identical in a database, and the difference is a
    # plant that is not being watched.
    notes: list[str] = field(default_factory=list)

    @property
    def skipped(self) -> int:
        return self.skipped_cameras + self.skipped_zones + self.skipped_policies

    def __str__(self) -> str:
        text = f"{self.cameras} camera(s), {self.zones} zone(s), {self.policies} policy(ies)"
        if self.skipped:
            text += (
                f" (skipped {self.skipped_cameras} camera(s), {self.skipped_zones} zone(s), "
                f"{self.skipped_policies} policy(ies) — footage not on this box)"
            )
        return text


def _require(entry: Any, keys: set[str], kind: str, index: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise SeedError(f"{kind}[{index}]: expected a mapping, got {type(entry).__name__}")
    unknown = set(entry) - keys
    if unknown:
        raise SeedError(f"{kind}[{index}]: unknown key(s): {', '.join(sorted(unknown))}")
    return entry


def _resolve_source(source: str) -> str:
    """File sources are stored as absolute paths so the gateway and decoder do
    not depend on anyone's working directory. URLs pass through untouched."""
    if source.lower().startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://")):
        return source
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return str(path)


def _scope(cameras: Any, index: int) -> list[Any]:
    """Read a policy's ``cameras:`` list. Absent or empty means every camera.

    Two spellings, because a plant scopes by camera nine times out of ten and
    the tenth wants a zone::

        cameras:
          - packing_line_demo
          - {camera: packing_line_west_demo, zone: packing_line_west_demo_conveyor_side}

    The ids are not checked here — ``set_policy_scope`` does it against the
    tables, which is the only place that knows whether the zone belongs to the
    camera beside it.
    """
    if cameras is None:
        return []
    if not isinstance(cameras, list):
        raise SeedError(f"policies[{index}]: 'cameras' must be a list")
    scope: list[Any] = []
    for position, entry in enumerate(cameras):
        if isinstance(entry, str):
            scope.append(entry)
        elif isinstance(entry, dict):
            unknown = set(entry) - {"camera", "zone", "camera_id", "zone_id"}
            if unknown:
                raise SeedError(
                    f"policies[{index}].cameras[{position}]: unknown key(s): "
                    f"{', '.join(sorted(unknown))}"
                )
            if not (entry.get("camera") or entry.get("camera_id")):
                raise SeedError(f"policies[{index}].cameras[{position}]: missing 'camera'")
            scope.append(entry)
        else:
            raise SeedError(
                f"policies[{index}].cameras[{position}]: expected a camera id or a "
                f"{{camera, zone}} mapping, got {type(entry).__name__}"
            )
    return scope


def _scope_camera(entry: Any) -> str:
    """The camera id inside one scope entry, whichever of the two spellings."""
    if isinstance(entry, dict):
        return str(entry.get("camera") or entry.get("camera_id") or "")
    return str(entry)


def load_seed(path: str | Path | None = None) -> dict[str, Any]:
    seed_path = Path(path) if path is not None else DEFAULT_SEED_PATH
    if not seed_path.exists():
        raise SeedError(f"seed file not found: {seed_path}")
    data = yaml.safe_load(seed_path.read_text()) or {}
    if not isinstance(data, dict):
        raise SeedError(f"{seed_path}: top level must be a mapping")
    unknown = set(data) - {"cameras", "zones", "policies"}
    if unknown:
        raise SeedError(f"{seed_path}: unknown top-level key(s): {', '.join(sorted(unknown))}")
    return data


def _missing_optional_source(entry: dict[str, Any], source: str) -> bool:
    """Is this an ``optional:`` camera whose footage is not on this box?

    Only *file* sources can be answered here. An optional camera on an RTSP URL
    is always imported: proving a camera absent means opening a socket to it,
    which is a thing `argus doctor` does deliberately and an import must not do
    at all — a seed that takes 30 s per unreachable camera is a seed nobody runs.
    """
    if not entry.get("optional"):
        return False
    if source.lower().startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://")):
        return False
    return not Path(source).exists()


def apply_seed(database: Database, data: dict[str, Any]) -> SeedResult:
    result = SeedResult()
    skipped_cameras: set[str] = set()

    for index, entry in enumerate(data.get("cameras") or []):
        entry = _require(entry, _CAMERA_KEYS, "cameras", index)
        for required in ("id", "name", "source"):
            if not entry.get(required):
                raise SeedError(f"cameras[{index}]: missing '{required}'")
        source = _resolve_source(str(entry["source"]))
        if _missing_optional_source(entry, source):
            camera_id = str(entry["id"])
            skipped_cameras.add(camera_id)
            result.skipped_cameras += 1
            result.notes.append(f"camera {camera_id!r}: no footage at {source}")
            continue
        database.upsert_camera(
            Camera(
                id=str(entry["id"]),
                name=str(entry["name"]),
                source=source,
                enabled=bool(entry.get("enabled", True)),
                detect_fps=(float(entry["detect_fps"]) if entry.get("detect_fps") is not None else None),
                kind=entry.get("kind"),
            )
        )
        result.cameras += 1

    known_cameras = {camera["id"] for camera in database.list_cameras()}
    for index, entry in enumerate(data.get("zones") or []):
        entry = _require(entry, _ZONE_KEYS, "zones", index)
        for required in ("id", "camera_id", "name", "polygon"):
            if not entry.get(required):
                raise SeedError(f"zones[{index}]: missing '{required}'")
        if entry["camera_id"] in skipped_cameras:
            result.skipped_zones += 1
            continue
        if entry["camera_id"] not in known_cameras:
            raise SeedError(f"zones[{index}]: unknown camera_id {entry['camera_id']!r}")
        polygon = entry["polygon"]
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise SeedError(f"zones[{index}]: polygon needs at least 3 points")
        database.upsert_zone(
            Zone(
                id=str(entry["id"]),
                camera_id=str(entry["camera_id"]),
                name=str(entry["name"]),
                polygon=polygon,
                sanctioned_limit=(
                    int(entry["sanctioned_limit"]) if entry.get("sanctioned_limit") is not None else None
                ),
                color=str(entry.get("color", "#22d3ee")),
            )
        )
        result.zones += 1

    for index, entry in enumerate(data.get("policies") or []):
        entry = _require(entry, _POLICY_KEYS, "policies", index)
        for required in ("id", "kind", "detector_key", "label"):
            if not entry.get(required):
                raise SeedError(f"policies[{index}]: missing '{required}'")
        scope = _scope(entry.get("cameras"), index)
        if scope and skipped_cameras:
            kept = [item for item in scope if _scope_camera(item) not in skipped_cameras]
            if not kept:
                # Every camera this policy named was skipped. It must NOT be
                # imported with an empty scope: empty means "every camera" in
                # this file, so a plant policy would land on the demo fleet and
                # start alerting about a recording of somebody else's factory.
                policy_id = str(entry["id"])
                result.skipped_policies += 1
                result.notes.append(
                    f"policy {policy_id!r}: every camera it is scoped to was skipped"
                )
                continue
            scope = kept
        policy = Policy(
            id=str(entry["id"]),
            kind=str(entry["kind"]),
            detector_key=str(entry["detector_key"]),
            label=str(entry["label"]),
            enabled=bool(entry.get("enabled", True)),
            stage=str(entry.get("stage", "shadow")),
            lifecycle=str(entry.get("lifecycle", "draft")),
            severity=str(entry.get("severity", "medium")),
            routed_to=entry.get("routed_to"),
            prompt=(entry["prompt"].strip() if entry.get("prompt") else None),
            interval_sec=(float(entry["interval_sec"]) if entry.get("interval_sec") is not None else None),
            cooldown_sec=float(entry.get("cooldown_sec", 30.0)),
            threshold=(float(entry["threshold"]) if entry.get("threshold") is not None else None),
            pilot_duration_days=float(entry.get("pilot_duration_days", 2.0)),
            # Always passed, never None: this file is declarative, so a policy
            # with no `cameras:` key means "every camera" and must clear a scope
            # somebody left behind — the same rule that makes editing `prompt:`
            # here override an edit made in the console.
            scope=scope,
        )
        try:
            database.upsert_policy(policy)
        except ValueError as exc:
            # `set_policy_scope` refuses an id that names nothing. Re-raised
            # with the position in the file, because "no such camera: line_3"
            # without it means reading the whole YAML to find which policy.
            raise SeedError(f"policies[{index}]: {exc}") from exc
        result.policies += 1

    return result


def seed(database: Database, path: str | Path | None = None) -> SeedResult:
    return apply_seed(database, load_seed(path))
