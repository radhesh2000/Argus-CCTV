"""What is happening right now, per camera.

The live wall needs three things the rest of the API does not provide together:
the zone polygons to draw, the number of people currently inside each of them,
and the boxes to put around those people. Zones come from the database, the
counts and boxes come from the engine's stats file (see
``Engine._record_live_state``), and this route is the join.

Two deliberate choices:

* **No fabricated numbers.** When the engine is not running, or has not yet
  analysed a frame for a camera, ``count`` and ``tracks`` are ``null`` rather
  than zero. Zero people and no information are different states and the wall
  must be able to say which one it is showing.
* **Zones are always listed**, even for a camera the engine never picked up, so
  a disabled or crashed camera still draws its geometry instead of vanishing.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter

from argus import processing
from argus.api.auth import NeedsView
from argus.api.deps import ConfigDep, DbDep
from argus.config import ArgusConfig
from argus.health import read_engine_stats
from argus.ingest.gateway import stream_name
from argus.redact import redact_text

router = APIRouter(tags=["live"], dependencies=[NeedsView])


def current_shift(config: ArgusConfig, when: float | None = None) -> dict[str, Any]:
    """The shift the appliance's local clock is in right now."""
    moment = datetime.fromtimestamp(when if when is not None else time.time())
    return {**config.site.shift_for_hour(moment.hour), "local_time": moment.isoformat(timespec="seconds")}


@router.get("/live/state")
def live_state(config: ConfigDep, db: DbDep) -> dict[str, Any]:
    engine = read_engine_stats(config)
    running = bool(engine.get("running"))
    measured = {entry.get("camera_id"): entry for entry in engine.get("cameras", [])}

    zones_by_camera: dict[str, list[dict[str, Any]]] = {}
    for zone in db.list_zones():
        zones_by_camera.setdefault(zone["camera_id"], []).append(zone)

    cameras: list[dict[str, Any]] = []
    for camera in db.list_cameras():
        stats = measured.get(camera["id"], {}) if running else {}
        counts = {
            entry["zone_id"]: entry
            for entry in (stats.get("occupancy") or [])
            if entry.get("zone_id")
        }
        zones = []
        for zone in zones_by_camera.get(camera["id"], []):
            live = counts.get(zone["id"])
            count = live.get("count") if live else None
            limit = zone["sanctioned_limit"]
            # The engine's own answer when it has one, because since
            # `detection.continuity` that is the *smoothed* judgement and
            # recomputing it here from the instantaneous count would throw the
            # smoothing away at the last step — putting the amber border back on
            # a strobe while the alert it is supposed to foreshadow never comes.
            # The fallback is the old arithmetic, for an engine from before this
            # change or a zone no policy is watching.
            over = live.get("over_limit") if live and "over_limit" in live else None
            zones.append({
                "id": zone["id"],
                "name": zone["name"],
                "polygon": zone["polygon"],
                "color": zone["color"],
                "sanctioned_limit": limit,
                "count": count,
                "over_limit": bool(over) if over is not None
                else bool(count is not None and limit is not None and count > limit),
            })
        cameras.append({
            "id": camera["id"],
            "name": camera["name"],
            "enabled": bool(camera["enabled"]),
            "status": camera["status"] if running or camera["status"] != "live" else "offline",
            # The decoder's last error, which is ffmpeg quoting the URL it could
            # not open — password and all, on a camera decoded directly.
            "status_detail": redact_text(camera["status_detail"], camera["source"]),
            "last_seen_at": camera["last_seen_at"],
            # The go2rtc stream to point a WebRTC player at. Equal to the camera
            # id today, but the UI should not have to know that.
            "stream": stream_name(camera["id"]),
            "target_fps": camera["detect_fps"] or config.detection.fps,
            "fps": stats.get("fps"),
            "lag_ms": stats.get("lag_ms"),
            "decode_mode": stats.get("mode"),
            "zones": zones,
            "tracks": stats.get("tracks") if running else None,
            "live_ts": stats.get("live_ts") if running else None,
            "people": len(stats.get("tracks") or []) if running and stats.get("tracks") is not None else None,
        })

    vlm = engine.get("vlm") or {}
    return {
        "time": time.time(),
        "engine": {
            "running": running,
            "reason": engine.get("reason"),
            "age_sec": engine.get("age_sec"),
        },
        # The pause switch, on the two-second poll the console already runs — so
        # the paused banner and the masthead react without a second endpoint.
        # No subprocess here: it is the flag file plus figures already in hand.
        "processing": processing.processing_view(
            config,
            engine_running=running,
            engine_uptime_sec=engine.get("uptime_sec"),
            vlm_available=bool(vlm.get("available")) if running else False,
        ),
        "gateway": engine.get("gateway"),
        "shift": current_shift(config),
        "cameras": cameras,
    }
