"""Read-only aggregates for the compliance analytics screen.

One request answers the whole screen. The alternative — five endpoints, or the
UI paging through raw events and counting them in JavaScript — either multiplies
round trips on a page that is a single glance, or ships a month of events to a
browser to compute four numbers.

What the server does *not* do is presentation: no colours, no bar heights, no
percentages of a maximum. It returns counts, a dense day series and the shift
rota it used, and the UI decides how tall a bar is. That split keeps this
testable as arithmetic and keeps the design's styling in the design's language.

Everything is bucketed on the appliance's local clock, and the shift boundaries
come from ``site.shifts`` in argus.yaml.

**Two counts, everywhere, since schema migration 7.** ``raw`` is *incidents* —
one row per thing that happened, however long it went on. ``observations`` is
the sampled verdicts behind them: a VLM condition that was true all night is one
incident and forty observations. Both are returned and neither is derived from
the other, because each alone lies in a different direction — counting
observations turns one mess into forty violations and makes the trend a picture
of the sampling interval, while counting only incidents hides that the mess sat
there for nine hours. The screen leads with incidents and says the second number
out loud.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from statistics import median
from typing import Any

from fastapi import APIRouter, Query

from argus.api.auth import NeedsView
from argus.api.deps import ConfigDep, DbDep
from argus.config import ArgusConfig
from argus.db import Database

router = APIRouter(prefix="/analytics", tags=["analytics"], dependencies=[NeedsView])

# Anything past this is not a dashboard, it is an export.
_MAX_DAYS = 365


def _day_span(days: int, until: datetime | None = None) -> tuple[datetime, datetime]:
    """The window as local datetimes: midnight ``days-1`` ago -> now.

    Inclusive of today, so "30 days" is 30 bars ending with the one still being
    written, which is what a shift supervisor means by the last 30 days.
    """
    end = until or datetime.now()
    start = (end - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return start, end


def _dense_days(rows: list[dict[str, Any]], start: date, end: date) -> list[dict[str, Any]]:
    """Zero-fill the day series.

    A day with no violations is a fact worth plotting; a chart that silently
    omits it compresses the x axis and makes a quiet week look like a busy one.
    """
    found = {row["day"]: row for row in rows}
    series, cursor = [], start
    while cursor <= end:
        key = cursor.isoformat()
        row = found.get(key)
        series.append({
            "day": key,
            "raw": int(row["raw"]) if row else 0,
            # Sampled verdicts behind those incidents. See the note above
            # `events_by_day` in argus/db.py: `raw` is what happened,
            # `observations` is how many times it was seen happening, and a
            # chart of the second is describing the sampling interval.
            "observations": int(row["observations"] or 0) if row else 0,
            "confirmed": int(row["confirmed"] or 0) if row else 0,
            "false_positive": int(row["false_positive"] or 0) if row else 0,
        })
        cursor += timedelta(days=1)
    return series


def _median_response_sec(db: Database, since_ts: float, until_ts: float) -> float | None:
    """Median seconds between a violation and a human ruling on it."""
    deltas: list[float] = []
    for row in db.review_latencies(since_ts=since_ts, until_ts=until_ts):
        reviewed_at = row.get("reviewed_at")
        if not reviewed_at:
            continue
        try:
            recorded = datetime.fromisoformat(str(reviewed_at)).timestamp()
        except ValueError:
            continue
        delta = recorded - float(row["event_ts"])
        # A verdict recorded before its event means clock skew or an imported
        # row, not a negative response time.
        if delta >= 0:
            deltas.append(delta)
    return round(median(deltas), 1) if deltas else None


def _zone_shift_table(db: Database, config: ArgusConfig, since_ts: float,
                      until_ts: float) -> list[dict[str, Any]]:
    """Violations per zone per shift — the design's heatmap, as counts."""
    shift_ids = [shift["id"] for shift in config.site.shifts]
    table: dict[str, dict[str, Any]] = {}
    for row in db.events_by_zone_hour(since_ts=since_ts, until_ts=until_ts):
        zone_id = row["zone_id"] or "__unzoned__"
        entry = table.setdefault(zone_id, {
            "zone_id": row["zone_id"],
            "zone_name": row["zone_name"] or "Unzoned",
            "camera_id": row["camera_id"],
            "raw": {key: 0 for key in shift_ids},
            "observations": {key: 0 for key in shift_ids},
            "confirmed": {key: 0 for key in shift_ids},
            "total": 0,
        })
        shift = config.site.shift_for_hour(int(row["hour"]))["id"]
        entry["raw"][shift] += int(row["raw"])
        entry["observations"][shift] += int(row["observations"] or 0)
        entry["confirmed"][shift] += int(row["confirmed"] or 0)
        entry["total"] += int(row["raw"])
    return sorted(table.values(), key=lambda item: (-item["total"], item["zone_name"]))


def _type_severity_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """``{detector_name: {severity: count}}`` — real severity, so a policy
    report card's grade split is the same field the triage queue sorts by,
    not an invented minor/major scale."""
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        out.setdefault(row["detector_name"], {})[row["severity"]] = int(row["n"])
    return out


def _type_shift_map(
    rows: list[dict[str, Any]], config: ArgusConfig,
) -> dict[str, dict[str, dict[str, int]]]:
    """``{detector_name: {shift_id: {raw, observations, confirmed}}}`` — the
    per-policy analogue of ``_zone_shift_table``, folded the same way: the
    database hands back policy x hour, this folds hours into shifts because
    the rota is configuration, not a database concern."""
    shift_ids = [shift["id"] for shift in config.site.shifts]
    out: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        entry = out.setdefault(row["detector_name"], {key: {"raw": 0, "observations": 0, "confirmed": 0} for key in shift_ids})
        shift = config.site.shift_for_hour(int(row["hour"]))["id"]
        entry[shift]["raw"] += int(row["raw"])
        entry[shift]["observations"] += int(row["observations"] or 0)
        entry[shift]["confirmed"] += int(row["confirmed"] or 0)
    return out


def _type_day_map(
    rows: list[dict[str, Any]], start: date, end: date,
) -> dict[str, list[dict[str, Any]]]:
    """``{detector_name: [{day, raw, confirmed}, ...]}``, zero-filled across
    the window — the per-card trend line, one query for every policy instead
    of one round trip each."""
    by_policy: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_policy.setdefault(row["detector_name"], {})[row["day"]] = row
    out: dict[str, list[dict[str, Any]]] = {}
    for policy, found in by_policy.items():
        series, cursor = [], start
        while cursor <= end:
            key = cursor.isoformat()
            row = found.get(key)
            series.append({
                "day": key,
                "raw": int(row["raw"]) if row else 0,
                "confirmed": int(row["confirmed"] or 0) if row else 0,
            })
            cursor += timedelta(days=1)
        out[policy] = series
    return out


@router.get("/summary")
def summary(config: ConfigDep, db: DbDep,
            days: int = Query(default=30, ge=1, le=_MAX_DAYS)) -> dict[str, Any]:
    start, end = _day_span(days)
    since_ts, until_ts = start.timestamp(), end.timestamp()
    # The equivalent window immediately before this one, so every scorecard can
    # carry a delta that means something rather than an arrow pointing nowhere.
    previous_start = start - timedelta(days=days)

    totals = db.event_totals(since_ts=since_ts, until_ts=until_ts)
    totals["median_response_sec"] = _median_response_sec(db, since_ts, until_ts)
    previous = db.event_totals(since_ts=previous_start.timestamp(), until_ts=since_ts)
    previous["median_response_sec"] = _median_response_sec(
        db, previous_start.timestamp(), since_ts
    )

    # Severity, shift and day breakdowns per policy — folded onto `by_type`
    # here rather than fetched per card, so the whole report-cards section is
    # still the one request this router's own docstring promises.
    severity_by_type = _type_severity_map(
        db.events_by_type_severity(since_ts=since_ts, until_ts=until_ts)
    )
    shift_by_type = _type_shift_map(
        db.events_by_type_hour(since_ts=since_ts, until_ts=until_ts), config
    )
    day_by_type = _type_day_map(
        db.events_by_type_day(since_ts=since_ts, until_ts=until_ts), start.date(), end.date()
    )
    empty_shifts = {shift["id"]: {"raw": 0, "observations": 0, "confirmed": 0} for shift in config.site.shifts}
    by_type = [
        {
            "key": row["detector_name"],
            "label": row["label"],
            "detector_type": row["detector_type"],
            "raw": int(row["raw"]),
            "observations": int(row["observations"] or 0),
            "confirmed": int(row["confirmed"] or 0),
            "false_positive": int(row["false_positive"] or 0),
            "by_severity": severity_by_type.get(row["detector_name"], {}),
            "by_shift": shift_by_type.get(row["detector_name"], empty_shifts),
            "by_day": day_by_type.get(row["detector_name"], []),
        }
        for row in db.events_by_type(since_ts=since_ts, until_ts=until_ts)
    ]

    return {
        "generated_at": time.time(),
        "window": {
            "days": days,
            "since_ts": since_ts,
            "until_ts": until_ts,
            "since_day": start.date().isoformat(),
            "until_day": end.date().isoformat(),
        },
        "shifts": config.site.shift_windows,
        "totals": totals,
        "previous": previous,
        "by_day": _dense_days(
            db.events_by_day(since_ts=since_ts, until_ts=until_ts), start.date(), end.date()
        ),
        "by_type": by_type,
        "by_zone_shift": _zone_shift_table(db, config, since_ts, until_ts),
    }
