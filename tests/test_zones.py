"""Zone geometry: the vectorized path must agree with the ported scalar one."""

from __future__ import annotations

import json

import numpy as np
import pytest

from argus.vision.zones import (
    Occupancy,
    Zone,
    ZoneSet,
    anchors,
    parse_polygon,
    person_anchor,
    point_in_polygon,
    points_in_polygon,
    polygon_bounds,
)

SQUARE = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
# A concave polygon: the notch is where a convex-hull shortcut would go wrong.
L_SHAPE = [(0.0, 0.0), (0.6, 0.0), (0.6, 0.4), (1.0, 0.4), (1.0, 1.0), (0.0, 1.0)]


def test_parse_polygon_accepts_json_and_lists():
    assert parse_polygon(json.dumps(SQUARE)) == SQUARE
    assert parse_polygon(SQUARE) == SQUARE


def test_parse_polygon_rejects_degenerate():
    with pytest.raises(ValueError):
        parse_polygon([[0.0, 0.0], [1.0, 1.0]])


@pytest.mark.parametrize(
    "point, expected",
    [((0.5, 0.5), True), ((0.1, 0.5), False), ((0.5, 0.1), False), ((0.9, 0.9), False)],
)
def test_point_in_polygon_square(point, expected):
    assert point_in_polygon(*point, SQUARE) is expected


def test_point_in_polygon_concave_notch():
    # Inside the bounding box of the L, outside the L itself.
    assert point_in_polygon(0.8, 0.2, L_SHAPE) is False
    assert point_in_polygon(0.3, 0.2, L_SHAPE) is True
    assert point_in_polygon(0.8, 0.7, L_SHAPE) is True


@pytest.mark.parametrize("polygon", [SQUARE, L_SHAPE])
def test_vectorized_matches_scalar_on_random_points(polygon):
    rng = np.random.default_rng(20260809)
    points = rng.random((2000, 2))
    vectorized = points_in_polygon(points, polygon)
    scalar = np.array([point_in_polygon(x, y, polygon) for x, y in points])
    assert np.array_equal(vectorized, scalar)


def test_points_in_polygon_handles_empty():
    assert points_in_polygon(np.zeros((0, 2)), SQUARE).shape == (0,)


def test_anchor_is_feet_not_centre():
    # A person whose box straddles the bottom zone edge is *outside* it: the
    # anchor is where they stand, not where their chest is.
    box = [400.0, 100.0, 500.0, 500.0]
    assert person_anchor(box, 1000, 1000) == (0.45, 0.5)
    assert np.allclose(anchors(np.array([box]), 1000, 1000), [[0.45, 0.5]])


def test_anchor_vectorized_matches_scalar():
    rng = np.random.default_rng(7)
    boxes = rng.random((50, 4)) * [960, 544, 960, 544]
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    expected = np.array([person_anchor(b, 960, 544) for b in boxes])
    assert np.allclose(anchors(boxes, 960, 544), expected)


def test_polygon_bounds_pads_the_top_hardest():
    """The anchor is the feet, so a zone is a patch of floor and the people
    standing on its far edge are drawn *above* it. A tight crop decapitates the
    people the verifier is being asked to count."""
    bounds = polygon_bounds(SQUARE, pad=0.04, pad_top=0.12)
    assert bounds == pytest.approx((0.16, 0.08, 0.84, 0.84))
    top_pad, side_pad = 0.2 - bounds[1], 0.2 - bounds[0]
    assert top_pad > side_pad


def test_polygon_bounds_never_leaves_the_frame():
    edge = [(0.0, 0.9), (0.1, 0.9), (0.1, 1.0), (0.0, 1.0)]
    assert polygon_bounds(edge) == pytest.approx((0.0, 0.78, 0.14, 1.0))


def test_polygon_bounds_of_nothing_is_the_whole_frame():
    """An empty polygon cannot say which part of the picture matters, and
    guessing a corner would be worse than declining to crop."""
    assert polygon_bounds([]) == (0.0, 0.0, 1.0, 1.0)


def _zone(limit: int | None = 2, polygon=SQUARE) -> Zone:
    return Zone(id="z1", camera_id="cam", name="Packing floor",
                polygon=tuple(polygon), sanctioned_limit=limit)


def test_zone_from_row_parses_json_polygon():
    zone = Zone.from_row(
        {"id": "z", "camera_id": "c", "name": "n", "polygon": json.dumps(SQUARE), "sanctioned_limit": 3}
    )
    assert zone.polygon == tuple(SQUARE)
    assert zone.sanctioned_limit == 3


def test_zoneset_occupancy_counts_feet_inside():
    zone = _zone(limit=2)
    boxes = np.array([
        [400, 200, 500, 400],    # feet at (0.45, 0.4)  -> inside
        [400, 200, 500, 900],    # feet at (0.45, 0.9)  -> below the zone
        [500, 100, 700, 600],    # feet at (0.6,  0.6)  -> inside
    ], dtype=float)
    occupancy = ZoneSet([zone]).occupancy(boxes, 1000, 1000)
    assert len(occupancy) == 1
    result: Occupancy = occupancy[0]
    assert result.count == 2
    assert list(result.inside) == [True, False, True]
    assert result.over_limit is False        # 2 people, limit 2 -> at the limit
    assert result.excess == 0


def test_over_limit_is_strictly_greater():
    zone = _zone(limit=1)
    boxes = np.array([[400, 200, 500, 400], [500, 100, 700, 600]], dtype=float)
    result = ZoneSet([zone]).occupancy(boxes, 1000, 1000)[0]
    assert result.count == 2
    assert result.over_limit is True
    assert result.excess == 1


def test_zone_without_limit_counts_but_never_alerts():
    zone = _zone(limit=None)
    boxes = np.array([[400, 200, 500, 400]] * 9, dtype=float)
    result = ZoneSet([zone]).occupancy(boxes, 1000, 1000)[0]
    assert result.count == 9
    assert result.over_limit is False
    assert result.excess == 0


def test_occupancy_with_no_detections():
    result = ZoneSet([_zone()]).occupancy(np.zeros((0, 4)), 960, 544)[0]
    assert result.count == 0
    assert result.inside.shape == (0,)


def test_zoneset_is_built_straight_from_database_rows(database):
    from argus.db import Camera, Zone as ZoneRow

    database.upsert_camera(Camera(id="cam", name="Cam", source="/tmp/x.mp4"))
    database.upsert_zone(ZoneRow(id="z1", camera_id="cam", name="Floor", polygon=SQUARE,
                                 sanctioned_limit=1))
    zone_set = ZoneSet.from_rows(database.list_zones("cam"))
    occupancy = zone_set.occupancy(np.array([[400, 200, 500, 400]], dtype=float), 1000, 1000)
    assert {o.zone.id: o.count for o in occupancy} == {"z1": 1}


def test_overlapping_zones_both_count_the_same_person():
    left = Zone(id="left", camera_id="c", name="Left", polygon=tuple(SQUARE))
    wide = Zone(id="wide", camera_id="c", name="Wide",
                polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    occupancy = ZoneSet([left, wide]).occupancy(np.array([[400, 200, 500, 400]], dtype=float), 1000, 1000)
    assert [o.count for o in occupancy] == [1, 1]
