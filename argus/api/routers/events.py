"""Events, verdicts, and the live stream.

Route order matters here: ``/events/stream`` is declared before
``/events/{event_id}``, or Starlette matches "stream" as an id and the SSE feed
answers 422 forever. v1 hit exactly that and left a comment about it; the
ordering is load-bearing, not stylistic.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from argus.api.auth import NeedsView, TriageDep
from argus.api.deps import ConfigDep, DbDep, found
from argus.api.schemas import BoxCorrectionCreate, ReviewCreate
from argus.db import Database, decoded_meta
from argus.redact import redact_camera

# VIEW for everything here, including the SSE stream: an alert carries a
# description of who was where, and the stream is the same rows arriving
# sooner. Recording a verdict needs TRIAGE, which is the whole point of the
# REVIEW role existing.
router = APIRouter(tags=["events"], dependencies=[NeedsView])

# Sent when a poll finds nothing, so proxies and load balancers do not decide
# the connection is idle and close it.
_KEEPALIVE = ": keepalive\n\n"

# How long to keep asking whether a streamed event's clip has landed. The cut is
# scheduled for `evidence.clip_delay_sec` (~ the pad, 8 s) after the event and
# takes a second or two, so this is generous by an order of magnitude — and it
# is a ceiling on work, not a promise: an event whose clip never appears simply
# stops being watched.
_CLIP_GRACE_SEC = 120.0


def _decode(event: dict[str, Any]) -> dict[str, Any]:
    """Hand ``meta`` to the client as an object, not as a string of JSON."""
    event["meta"] = decoded_meta(event)
    return event


def _sse(name: str, payload: Any, *, event_id: int | None = None) -> str:
    """One SSE frame. The ``id:`` field is what lets a browser resume with
    Last-Event-ID after a dropped connection instead of missing events."""
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}event: {name}\ndata: {json.dumps(payload, default=str)}\n\n"


# --------------------------------------------------------------------------
# list & filters
# --------------------------------------------------------------------------


@router.get("/events")
def list_events(
    db: DbDep,
    camera_id: str | None = None,
    zone_id: str | None = None,
    policy_id: str | None = None,
    detector: str | None = Query(default=None, description="detector_name, e.g. a policy id"),
    detector_type: Literal["detector", "vlm"] | None = None,
    stage: Literal["shadow", "reviewed", "auto_routed"] | None = None,
    severity: Literal["low", "medium", "high"] | None = None,
    verdict: Literal["confirmed", "false_positive", "unreviewed"] | None = None,
    since_ts: float | None = Query(default=None, description="unix epoch, inclusive"),
    until_ts: float | None = Query(default=None, description="unix epoch, inclusive"),
    since_id: int | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    filters = {
        "camera_id": camera_id, "zone_id": zone_id, "policy_id": policy_id,
        "detector": detector, "detector_type": detector_type, "stage": stage,
        "severity": severity, "verdict": verdict, "since_ts": since_ts,
        "until_ts": until_ts, "since_id": since_id,
    }
    events = [_decode(event) for event in db.recent_events(limit=limit, offset=offset, **filters)]
    return {
        "events": events,
        # The unpaginated total, so a UI can say "showing 100 of 4,312" rather
        # than discovering the end by walking off it.
        "total": db.count_events(**filters),
        "limit": limit,
        "offset": offset,
        "detectors": db.distinct_detector_names(),
    }


@router.get("/triage")
def triage_queue(
    db: DbDep,
    limit: int = Query(default=200, ge=1, le=1000),
    order: str = Query(default="oldest", pattern="^(oldest|newest)$"),
) -> dict[str, Any]:
    """Everything past shadow that nobody has ruled on yet.

    ``total`` is the whole backlog, not this page. The console's nav badge used
    to report ``events.length``, which on this appliance meant it said **200**
    while 7 682 alerts were outstanding — and downloaded a quarter of a megabyte
    every 15 seconds to say it.

    ``order=newest`` is the other half of the same finding. A queue drained from
    the front is right, but a front that is a fortnight old puts *today's* alert
    beyond reach of the only screen that can rule on it. Oldest stays the
    default: the toggle is for a reviewer who wants the shift they just worked.
    """
    events = db.unreviewed_events(limit=limit, newest_first=order == "newest")
    return {
        "events": [_decode(event) for event in events],
        "total": db.unreviewed_count(),
        "order": order,
    }


# --------------------------------------------------------------------------
# live stream (declared before /events/{event_id} — see module docstring)
# --------------------------------------------------------------------------


async def _event_source(
    request: Request, db: Database, poll_sec: float, last_id: int
):
    yield _sse("hello", {"last_id": last_id, "poll_sec": poll_sec})
    # Events sent before their clip existed, and when we sent them. See
    # _CLIP_GRACE_SEC.
    awaiting_clip: dict[int, float] = {}
    # The second cursor. `last_id` only moves forward, so it can only ever
    # announce rows that did not exist before — and an ongoing condition is an
    # old row that changed (one more observation, a newer sentence, a fresher
    # snapshot, or its close). Without this the console would hold a card
    # reading "1 observation" while the appliance was on its fortieth.
    condition_marker = db.condition_marker()
    while True:
        if await request.is_disconnected():
            return
        # Cheap enough to run on the event loop: an indexed SELECT on a WAL
        # database, once every couple of seconds per connected client.
        fresh = db.events_since(last_id)
        just_sent: set[int] = set()
        if fresh:
            for event in fresh:
                last_id = int(event["id"])
                just_sent.add(last_id)
                if not event.get("clip_path"):
                    awaiting_clip[last_id] = time.monotonic()
                yield _sse("event", _decode(event), event_id=last_id)
        else:
            yield _KEEPALIVE
        for event in _completed_clips(db, awaiting_clip):
            yield _sse("update", _decode(event))
        for event in db.conditions_updated_since(condition_marker):
            condition_marker = max(condition_marker,
                                   float(event.get("condition_updated_at") or 0.0))
            event_id = int(event["id"])
            if event_id in just_sent or event_id > last_id:
                # A condition that has only just opened. It went out as an
                # `event` frame a moment ago (or is about to on the next pass),
                # and every opening also stamps `condition_updated_at` — so
                # without this every new condition would arrive twice, the
                # second time as an update carrying nothing the client did not
                # already have. The marker still advances: the row is accounted
                # for, it simply has no news yet.
                continue
            # Same `update` frame the clip completion uses, and deliberately
            # so: it carries no `id:` field, so it cannot drag a reconnecting
            # browser's Last-Event-ID cursor backwards, and every client already
            # knows to merge it over the row it holds rather than prepend it.
            # That is what keeps the feed from reordering under the reader.
            yield _sse("update", _decode(event))
        await asyncio.sleep(poll_sec)


def _completed_clips(db: Database, awaiting: dict[int, float]) -> list[dict[str, Any]]:
    """Events whose clip has appeared since we streamed them, clip-less.

    An event row is written the instant a violation is decided; its ±8 s clip
    cannot be, because the seconds *after* the event have not been recorded yet
    (`Engine._schedule_clip`). So the row this stream sends is true and
    incomplete, and every live alert would carry a snapshot and no video — not
    because the clip failed, but because a client that only reads the stream
    never hears that it landed. Holding the alert back until the clip is cut is
    the wrong trade: it delays the alarm by the length of the evidence.

    So the completion is itself an event. Ids are tracked per connection, only
    while their clip is outstanding, so this costs one keyed SELECT for the
    handful of alerts raised in the last half-minute and nothing at all the rest
    of the time.
    """
    if not awaiting:
        return []
    now = time.monotonic()
    done: list[dict[str, Any]] = []
    for event_id, sent_at in list(awaiting.items()):
        if now - sent_at > _CLIP_GRACE_SEC:
            # The cut failed, there was no ring footage, or evidence is off.
            # Stop asking; the REST row stays authoritative either way.
            del awaiting[event_id]
            continue
        row = db.get_event(event_id)
        if row is None:  # deleted by a retention sweep mid-stream
            del awaiting[event_id]
            continue
        if row.get("clip_path"):
            del awaiting[event_id]
            done.append(row)
    return done


@router.get("/events/stream")
async def stream_events(
    request: Request,
    db: DbDep,
    config: ConfigDep,
    since_id: int | None = None,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """Server-sent events for new alerts.

    The engine and the API are separate processes, so the SQLite file is the
    bus: poll for rows past the last id seen. At a handful of events a minute
    this is the right amount of machinery — a real broker would be a second
    thing to run, supervise and explain, for no gain a dashboard can perceive.

    Two frame types:

    * ``event`` — a new alert, carrying an ``id:`` field. Its ``clip_path`` is
      usually ``null``, because the ±8 s of video around an alert includes
      seconds that have not happened yet.
    * ``update`` — the same event again, with something about it changed. Two
      things send one: its clip landing on disk, and an **ongoing condition**
      being observed again or closed (schema migration 7). No ``id:`` field on
      either: it must not move a reconnecting client's ``Last-Event-ID`` cursor
      backwards. Clients should merge it over the row they already hold — *in
      place*, never re-sorting the list, because a feed that reorders itself
      while a reviewer is reading it is a feed that gets misclicked, and the
      misclick is a safety verdict.

    A reconnecting client resumes from ``Last-Event-ID`` (sent automatically by
    every browser's EventSource) or an explicit ``?since_id=``; without either
    it starts from now and misses nothing it did not already have. Clip updates
    are tracked per connection, so a reconnect re-reads the completed rows from
    ``GET /api/events`` rather than being replayed them.
    """
    if since_id is None and last_event_id is not None:
        try:
            since_id = int(last_event_id)
        except ValueError:
            since_id = None
    start = since_id if since_id is not None else db.max_event_id()
    return StreamingResponse(
        _event_source(request, db, max(0.25, config.api.sse_poll_sec), start),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx buffers proxied responses by default, which turns a live
            # feed into a feed that arrives in bursts of 4 KB.
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------
# one event
# --------------------------------------------------------------------------


@router.get("/events/{event_id}")
def get_event(event_id: int, db: DbDep) -> dict[str, Any]:
    event = _decode(found(db.get_event(event_id), "event", event_id))
    return {
        "event": event,
        "camera": redact_camera(db.get_camera(str(event["camera_id"]))),
        "zone": db.get_zone(str(event["zone_id"])) if event.get("zone_id") else None,
        "review": db.get_review(event_id),
        "notifications": db.notifications_for_event(event_id),
        "box_corrections": db.list_box_corrections(event_id),
    }


@router.post("/events/{event_id}/review")
def review_event(event_id: int, body: ReviewCreate, reviewer: TriageDep,
                 db: DbDep) -> dict[str, Any]:
    """Record a human verdict.

    This is the single most valuable write in the system: it is what turns
    alerts into a precision number, gates promotion to auto_routed, and marks
    the false positives that feed retraining. The refreshed accuracy row for
    this detector comes back with the response so a triage UI can update its
    running precision without a second request.

    **The reviewer is the session, not the request body.** ``body.reviewer`` is
    still accepted so an integration written against the pre-auth API does not
    break, and is ignored: a verdict is an act by a person, and a name a client
    can choose is not a person. This is the column that answers "who cleared
    that alert" a year later, and it is now answerable.
    """
    event = found(db.get_event(event_id), "event", event_id)
    if event["stage"] == "shadow":
        # Shadow events are the detector talking to itself; letting them into
        # the precision statistics would mix unvalidated output with the
        # reviewed record that promotion decisions rest on.
        raise HTTPException(status_code=409, detail="shadow events are not reviewable")
    db.record_review(event_id, body.verdict, reviewer=reviewer.username, note=body.note)
    accuracy = next(iter(db.detector_accuracy(event["detector_name"])), None)
    return {
        "event": _decode(db.get_event(event_id)),
        "review": db.get_review(event_id),
        "accuracy": accuracy,
    }


@router.post("/events/{event_id}/boxes/{track_id}/correction")
def correct_box(event_id: int, track_id: int, body: BoxCorrectionCreate, reviewer: TriageDep,
                db: DbDep) -> dict[str, Any]:
    """Record a human verdict on one detection within this event, not the
    whole alert. See `BoxCorrectionCreate` for how this differs from
    `/review`. The path's `track_id` and the body's must agree — the id is in
    the URL so a correction is a resource the console can link to and
    re-submit idempotently, the way `/review` is; it is also in the body so a
    client that already built the payload before knowing the URL shape (or a
    future bulk endpoint) is not forced to duplicate it awkwardly.
    """
    if track_id != body.track_id:
        raise HTTPException(
            status_code=422,
            detail=f"track_id in the URL ({track_id}) does not match the body ({body.track_id})",
        )
    event = found(db.get_event(event_id), "event", event_id)
    meta = decoded_meta(event) or {}
    known_tracks = {
        int(box[4]) for box in (meta.get("boxes") or []) if len(box) > 4 and box[4] is not None
    }
    if track_id not in known_tracks:
        raise HTTPException(
            status_code=422,
            detail=f"track {track_id} is not one of the detections recorded on event {event_id}",
        )
    db.record_box_correction(event_id, track_id, body.verdict,
                             reviewer=reviewer.username, note=body.note)
    return {"box_corrections": db.list_box_corrections(event_id)}


# --------------------------------------------------------------------------
# derived views
# --------------------------------------------------------------------------


@router.get("/accuracy")
def accuracy(db: DbDep, days: int = Query(default=14, ge=1, le=365)) -> dict[str, Any]:
    detectors = db.detector_accuracy()
    return {
        "detectors": detectors,
        "trend": db.review_trend(days),
        "total_reviewed": sum(row["reviewed"] for row in detectors),
        "total_pending": sum(row["pending"] for row in detectors),
    }


@router.get("/occupancy")
def occupancy(
    db: DbDep,
    camera_id: str | None = None,
    zone_id: str | None = None,
    since_ts: float | None = None,
    limit: int = Query(default=2000, ge=1, le=20000),
) -> dict[str, Any]:
    return {
        "series": db.occupancy_series(
            camera_id=camera_id, zone_id=zone_id, since_ts=since_ts, limit=limit
        )
    }


@router.get("/notifications")
def notifications(db: DbDep, limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    """Who was actually told, and whether it worked."""
    return {"notifications": db.recent_notifications(limit=limit)}
