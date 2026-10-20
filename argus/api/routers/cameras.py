"""Cameras and their zones.

Zones hang off cameras (``/api/cameras/{id}/zones`` to list and create) but are
addressed directly once they exist (``/api/zones/{id}``), because a zone id is
unique on its own and the UI edits one without knowing which camera it came
from.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from argus.api.auth import NeedsManage, NeedsView
from argus.api.deps import ConfigDep, DbDep, changed, found
from argus.api.schemas import CameraCreate, CameraProbe, CameraUpdate, ZoneCreate, ZoneUpdate
from argus.db import Camera, Zone, slugify, source_kind
from argus.ingest.gateway import probe_camera
from argus.redact import PasswordNotKept, is_masked, redact_camera, restore_password

# The fleet is readable by anyone who may see the plant at all, and editable
# only by RESPOND and ADMIN: adding a camera or moving a zone polygon changes
# what the appliance watches and what it counts against a sanctioned limit,
# which is a supervisor's decision rather than a reviewer's.
#
# Readable, except for one thing: a camera's password. Every row leaves through
# `redact_camera`, for every role — an edit round-trips the masked source and
# `_unmasked` puts the stored password back (argus/redact.py).
router = APIRouter(tags=["fleet"], dependencies=[NeedsView])


def _unmasked(source: str, stored: dict[str, Any] | None) -> str:
    """``source`` with a ``***`` password filled in from an enrolled camera."""
    try:
        return restore_password(source, stored["source"] if stored else None)
    except PasswordNotKept as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# --------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------


@router.get("/cameras")
def list_cameras(db: DbDep, enabled_only: bool = False, with_zones: bool = True) -> dict[str, Any]:
    cameras = [redact_camera(camera) for camera in db.list_cameras(enabled_only=enabled_only)]
    if with_zones:
        # One query for every zone beats one per camera: a 40-camera fleet is
        # 41 round trips otherwise, on every poll of the live wall.
        zones: dict[str, list[dict[str, Any]]] = {}
        for zone in db.list_zones():
            zones.setdefault(zone["camera_id"], []).append(zone)
        for camera in cameras:
            camera["zones"] = zones.get(camera["id"], [])
    return {"cameras": cameras}


@router.post("/cameras", status_code=201, dependencies=[NeedsManage])
def create_camera(body: CameraCreate, db: DbDep) -> dict[str, Any]:
    if body.id and db.get_camera(body.id):
        raise HTTPException(status_code=409, detail=f"camera already exists: {body.id}")
    camera_id = body.id or db.unique_id("cameras", slugify(body.name))
    db.upsert_camera(Camera(
        id=camera_id, name=body.name, source=body.source,
        enabled=body.enabled, detect_fps=body.detect_fps,
    ))
    return {"camera": redact_camera(db.get_camera(camera_id))}


@router.post("/cameras/probe", dependencies=[NeedsManage])
def probe(body: CameraProbe, config: ConfigDep, db: DbDep) -> dict[str, Any]:
    """Does this source actually answer?

    For a camera that is already enrolled, the answer comes from **go2rtc**:
    it is the component holding the connection, and "the gateway is serving this
    stream over rtsp+tcp with an H264 track on it" is a stronger statement than
    anything a second connection could produce. For a URL an operator has just
    typed there is no such stream yet, so the appliance opens it with its own
    ffprobe — the same binary the gateway is pointed at. The response says which
    route answered; see ``argus.ingest.gateway.probe_camera``.

    Never 502s for a camera that does not answer: "unreachable" is the result of
    a connection test, not an error in it.
    """
    if (body.source is None) == (body.camera_id is None):
        raise HTTPException(
            status_code=422, detail="probe exactly one of `source` or `camera_id`"
        )
    if body.camera_id is not None:
        camera = found(db.get_camera(body.camera_id), "camera", body.camera_id)
        return probe_camera(config, camera, prefer_gateway=True)
    # A throwaway row, never stored. `source` has already been through
    # `assert_safe_source` at the schema, which is what makes it safe to hand to
    # something whose job is to run ffmpeg.
    source = str(body.source)
    if body.password_from is not None:
        source = _unmasked(source, found(db.get_camera(body.password_from),
                                         "camera", body.password_from))
    elif is_masked(source):
        source = _unmasked(source, None)
    camera = {"id": "probe", "source": source, "kind": source_kind(source)}
    return probe_camera(config, camera, prefer_gateway=False)


@router.get("/cameras/{camera_id}")
def get_camera(camera_id: str, db: DbDep) -> dict[str, Any]:
    camera = redact_camera(found(db.get_camera(camera_id), "camera", camera_id))
    return {"camera": {**camera, "zones": db.list_zones(camera_id)}}


@router.patch("/cameras/{camera_id}", dependencies=[NeedsManage])
def update_camera(camera_id: str, body: CameraUpdate, db: DbDep) -> dict[str, Any]:
    stored = found(db.get_camera(camera_id), "camera", camera_id)
    fields = body.model_dump(exclude_unset=True)
    if "source" in fields:
        # The form sends back the masked source it was shown. An edit that does
        # not retype the password keeps the one already stored.
        fields["source"] = _unmasked(fields["source"], stored)
        # kind is derived, never supplied: a source edited from a file path to
        # an rtsp:// URL must not keep claiming to be a file.
        fields["kind"] = source_kind(fields["source"])
    if fields:
        db.update_camera(camera_id, **fields)
    return {"camera": redact_camera(db.get_camera(camera_id))}


@router.delete("/cameras/{camera_id}", dependencies=[NeedsManage])
def delete_camera(camera_id: str, db: DbDep) -> dict[str, Any]:
    """Deletes the camera and, by cascade, its zones, events and occupancy.

    Evidence files on disk are left for retention to reap: unlinking clips
    inside a request would make a mistaken click unrecoverable and slow.
    """
    changed(db.delete_camera(camera_id), "camera", camera_id)
    return {"deleted": camera_id}


# --------------------------------------------------------------------------
# zones
# --------------------------------------------------------------------------


@router.get("/cameras/{camera_id}/zones")
def list_camera_zones(camera_id: str, db: DbDep) -> dict[str, Any]:
    found(db.get_camera(camera_id), "camera", camera_id)
    return {"zones": db.list_zones(camera_id)}


@router.post("/cameras/{camera_id}/zones", status_code=201, dependencies=[NeedsManage])
def create_zone(camera_id: str, body: ZoneCreate, db: DbDep) -> dict[str, Any]:
    found(db.get_camera(camera_id), "camera", camera_id)
    if body.id and db.get_zone(body.id):
        raise HTTPException(status_code=409, detail=f"zone already exists: {body.id}")
    zone_id = body.id or db.unique_id("zones", slugify(f"{camera_id}_{body.name}"))
    db.upsert_zone(Zone(
        id=zone_id, camera_id=camera_id, name=body.name, polygon=body.polygon,
        sanctioned_limit=body.sanctioned_limit, color=body.color,
    ))
    return {"zone": db.get_zone(zone_id)}


@router.get("/zones")
def list_zones(db: DbDep, camera_id: str | None = None) -> dict[str, Any]:
    return {"zones": db.list_zones(camera_id)}


@router.get("/zones/{zone_id}")
def get_zone(zone_id: str, db: DbDep) -> dict[str, Any]:
    return {"zone": found(db.get_zone(zone_id), "zone", zone_id)}


@router.patch("/zones/{zone_id}", dependencies=[NeedsManage])
def update_zone(zone_id: str, body: ZoneUpdate, db: DbDep) -> dict[str, Any]:
    found(db.get_zone(zone_id), "zone", zone_id)
    fields = body.model_dump(exclude_unset=True)
    if fields:
        db.update_zone(zone_id, **fields)
    return {"zone": db.get_zone(zone_id)}


@router.delete("/zones/{zone_id}", dependencies=[NeedsManage])
def delete_zone(zone_id: str, db: DbDep) -> dict[str, Any]:
    changed(db.delete_zone(zone_id), "zone", zone_id)
    return {"deleted": zone_id}
