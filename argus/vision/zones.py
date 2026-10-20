"""Zone geometry and occupancy.

Polygons are stored normalized (0-1) so a zone drawn once survives any change of
resolution on the same camera view — the detector runs at 960x544, the UI draws
on a 1920x1080 WebRTC frame, and both agree about where the zone is.

Ported from v1 (``pipeline/geometry.py`` on the ``v1-archive`` branch). Two things are kept exactly as they
were, because they are product behaviour rather than implementation detail:

* **The anchor is the bottom-centre of the box** (a person's feet), not the box
  centre. With a centre anchor, someone standing just *behind* a zone boundary
  counts as inside it, which is how v1's first occupancy numbers were wrong.
* **Even-odd ray casting** with the same strict/non-strict comparisons, so a
  point on an edge falls the same side of the test it did in v1.

The vectorized implementation here is exercised against the scalar one in
``tests/test_zones.py`` on random points, so "faster" cannot quietly become
"different".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

Point = tuple[float, float]


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def parse_polygon(polygon: str | Sequence[Sequence[float]]) -> list[Point]:
    """Accept either the JSON text stored in SQLite or an already-decoded list."""
    points = json.loads(polygon) if isinstance(polygon, str) else polygon
    parsed = [(float(x), float(y)) for x, y in points]
    if len(parsed) < 3:
        raise ValueError(f"a polygon needs at least 3 points, got {len(parsed)}")
    return parsed


def point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    """Ray-casting test in the polygon's own (normalized) coordinate space."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def points_in_polygon(points: np.ndarray, polygon: Sequence[Point]) -> np.ndarray:
    """Vectorized :func:`point_in_polygon` for an ``(N, 2)`` array of points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    poly = np.asarray(polygon, dtype=np.float64)
    xs, ys = points[:, 0][:, None], points[:, 1][:, None]      # (N, 1)
    xi, yi = poly[:, 0][None, :], poly[:, 1][None, :]           # (1, E) edge starts
    xj, yj = np.roll(poly[:, 0], 1)[None, :], np.roll(poly[:, 1], 1)[None, :]

    straddles = (yi > ys) != (yj > ys)
    # Horizontal edges never straddle, so their zero denominator is masked out
    # before it is used; substituting 1.0 keeps numpy from warning about it.
    denominator = np.where(straddles, yj - yi, 1.0)
    crossing_x = (xj - xi) * (ys - yi) / denominator + xi
    return (straddles & (xs < crossing_x)).sum(axis=1) % 2 == 1


def polygon_bounds(
    polygon: Sequence[Point], *, pad: float = 0.04, pad_top: float = 0.12,
) -> tuple[float, float, float, float]:
    """Normalised axis-aligned bounds of a polygon, padded and clamped to 0-1.

    The sub-picture a claim about this zone is really about — used by the VLM
    verification pass, which is asked *"are there 5 people in the Packing
    floor?"* and was until now shown the whole camera view to answer it.

    **The top pad is larger than the others, and that is the anchor's fault.**
    Zone membership is decided on the bottom-centre of a person's box — their
    feet — so a polygon is a region of *floor*, and somebody standing on its top
    edge has their entire body above it. Cropping tight to the polygon would cut
    the heads off the very people the question is about, which is a strange way
    to help something count them. The sides and bottom need much less: a body is
    narrow, and feet at the bottom edge means the body is already inside.
    """
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return (0.0, 0.0, 1.0, 1.0)
    pad = max(0.0, float(pad))
    x1 = float(points[:, 0].min()) - pad
    x2 = float(points[:, 0].max()) + pad
    y1 = float(points[:, 1].min()) - max(pad, float(pad_top))
    y2 = float(points[:, 1].max()) + pad
    clamp = lambda value: min(1.0, max(0.0, value))  # noqa: E731
    return (clamp(x1), clamp(y1), clamp(x2), clamp(y2))


def anchors(xyxy: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
    """Where people *are* for zone membership: feet, normalized to 0-1."""
    boxes = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2.0 / frame_width, boxes[:, 3] / frame_height],
        axis=1,
    )


def person_anchor(xyxy: Sequence[float], frame_width: int, frame_height: int) -> Point:
    """Scalar anchor for one box (kept for parity with v1 and for tests)."""
    x1, _y1, x2, y2 = xyxy
    return ((x1 + x2) / 2 / frame_width, y2 / frame_height)


# --------------------------------------------------------------------------
# zones
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Zone:
    """A camera zone, as configured — polygon plus the limit it is judged against."""

    id: str
    camera_id: str
    name: str
    polygon: tuple[Point, ...]
    sanctioned_limit: int | None = None
    color: str = "#22d3ee"

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Zone":
        limit = row.get("sanctioned_limit")
        return cls(
            id=str(row["id"]),
            camera_id=str(row["camera_id"]),
            name=str(row["name"]),
            polygon=tuple(parse_polygon(row["polygon"])),
            sanctioned_limit=(int(limit) if limit is not None else None),
            color=str(row.get("color") or "#22d3ee"),
        )

    def contains(self, x: float, y: float) -> bool:
        return point_in_polygon(x, y, self.polygon)

    def members(self, xyxy: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
        """Boolean mask over boxes: is this person standing in this zone?"""
        return points_in_polygon(anchors(xyxy, frame_width, frame_height), self.polygon)


@dataclass(frozen=True, slots=True)
class Occupancy:
    """One zone's answer for one frame."""

    zone: Zone
    count: int
    inside: np.ndarray          # bool mask, aligned with the detections passed in

    @property
    def limit(self) -> int | None:
        return self.zone.sanctioned_limit

    @property
    def over_limit(self) -> bool:
        """A zone with no ``sanctioned_limit`` is counted but never alerts."""
        return self.limit is not None and self.count > self.limit

    @property
    def excess(self) -> int:
        return 0 if self.limit is None else max(0, self.count - self.limit)


class ZoneSet:
    """The zones of one camera, evaluated together against one frame."""

    __slots__ = ("zones",)

    def __init__(self, zones: Iterable[Zone]):
        self.zones = list(zones)

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> "ZoneSet":
        return cls(Zone.from_row(row) for row in rows)

    def __len__(self) -> int:
        return len(self.zones)

    def __iter__(self):
        return iter(self.zones)

    def occupancy(self, xyxy: np.ndarray, frame_width: int, frame_height: int) -> list[Occupancy]:
        """Per-zone counts for one frame's person boxes.

        The anchors are computed once for the whole frame and reused for every
        zone: the per-zone work is then one masked comparison, so overlapping
        zones cost almost nothing.
        """
        points = anchors(xyxy, frame_width, frame_height)
        result = []
        for zone in self.zones:
            inside = points_in_polygon(points, zone.polygon)
            result.append(Occupancy(zone=zone, count=int(inside.sum()), inside=inside))
        return result


__all__ = [
    "Occupancy",
    "Point",
    "Zone",
    "ZoneSet",
    "anchors",
    "parse_polygon",
    "person_anchor",
    "point_in_polygon",
    "points_in_polygon",
    "polygon_bounds",
]
