"""The leak fix has to remove a neighbourhood, not a match.

`scripts/leakage_check.py` found that four of the sixteen val frames are inside
the student's own training set (docs/STUDENT_DETECTOR.md §6). Excluding only
the frames it *flagged* would fix nothing: the flag is an RMSE comparison
against a threshold, and a harvested frame 1.4 s from a val frame can sit just
the wrong side of that threshold while being, to a detector, the same moment
with one worker's arm moved. So the exclusion is temporal — every harvested
second within a window of the val frame's anchor — and these pin that.

The anchor itself is a guess: the hand-labelled corpus never recorded when its
frames were cut, so nearest-by-RMSE is the only estimate there is. Excluding
around a wrong guess costs training frames and cannot create a leak, which is
why `exclusions_for` does it for every checked frame rather than only the
flagged ones. That asymmetry is the whole design and it is tested below.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.leakage_check import classify, exclusions_for, without


def times(scene: str, seconds: range) -> dict[str, list[tuple[float, str]]]:
    return {scene: [(float(t), f"{scene}/t{t:03d}.jpg") for t in seconds]}


def row(frame: str, scene: str, t: float | None, verdict: str = "near-duplicate") -> dict:
    return {"frame": frame, "scene": scene, "nearest_t_sec": t, "rmse": 0.02, "verdict": verdict}


# --- the window ------------------------------------------------------------


def test_the_window_is_inclusive_on_both_sides():
    excluded, anchors = exclusions_for(
        [row("v", "S", 18.0)], times("S", range(40)), 2.0
    )

    assert excluded == [f"S/t{t:03d}.jpg" for t in (16, 17, 18, 19, 20)]
    assert anchors[0]["anchor_t_sec"] == 18.0
    assert anchors[0]["excludes"] == excluded


def test_a_window_at_the_end_of_the_clip_is_truncated_not_shifted():
    """A val frame anchored at the last harvested second costs three frames,
    not five. Sliding the window inwards to keep its width would exclude
    seconds that are further from the val frame than ones it left in."""
    excluded, _ = exclusions_for([row("v", "S", 39.0)], times("S", range(40)), 2.0)

    assert excluded == ["S/t037.jpg", "S/t038.jpg", "S/t039.jpg"]


def test_overlapping_windows_are_a_union_not_a_sum():
    excluded, anchors = exclusions_for(
        [row("a", "S", 5.0), row("b", "S", 6.0)], times("S", range(40)), 2.0
    )

    assert excluded == [f"S/t{t:03d}.jpg" for t in (3, 4, 5, 6, 7, 8)]
    assert len(anchors) == 2
    assert sum(len(a["excludes"]) for a in anchors) == 10   # 5 + 5, deduplicated to 6


def test_a_distinct_frame_still_costs_its_neighbourhood():
    """The flag is not the criterion. `distinct` means the RMSE bar was not
    met, which is not the same as "no harvested frame is near this moment"."""
    excluded, _ = exclusions_for(
        [row("v", "S", 12.0, verdict="distinct")], times("S", range(40)), 2.0
    )

    assert len(excluded) == 5


def test_a_scene_that_was_never_harvested_excludes_nothing():
    excluded, anchors = exclusions_for(
        [row("v", "Premixing", None, verdict="no-harvest")], times("S", range(40)), 2.0
    )

    assert excluded == []
    assert anchors == []


# --- re-checking the training set rather than the harvest ------------------


def signatures(scene: str, seconds: range) -> dict[str, list[tuple[float, np.ndarray, str]]]:
    return {scene: [(float(t), np.full((4, 4), t / 100, np.float32), f"{scene}/t{t:03d}.jpg")
                    for t in seconds]}


def test_the_reference_set_drops_what_training_dropped():
    kept = without(signatures("S", range(5)), frozenset({"S/t001.jpg", "S/t003.jpg"}))

    assert [file for _, _, file in kept["S"]] == ["S/t000.jpg", "S/t002.jpg", "S/t004.jpg"]


def test_excluding_nothing_returns_the_same_object():
    """Not merely equal: the no-exclusions path must not pay to rebuild the
    dictionary, and a copy here would silently double peak memory on a corpus
    that is already 200 thumbnails per scene."""
    original = signatures("S", range(5))

    assert without(original, frozenset()) is original


# --- the bar itself --------------------------------------------------------


@pytest.mark.parametrize("distance, expected", [
    (0.010, "duplicate"),        # closer than the closest consecutive pair
    (0.017, "duplicate"),        # exactly the closest pair — still a duplicate
    (0.020, "near-duplicate"),
    (0.027, "near-duplicate"),   # exactly the median — still flagged
    (0.030, "distinct"),
])
def test_the_verdict_boundaries_are_inclusive(distance, expected):
    assert classify(distance, 0.017, 0.027) == expected
