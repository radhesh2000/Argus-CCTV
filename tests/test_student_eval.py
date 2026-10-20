"""The arithmetic in the comparison harness, without a GPU.

`scripts/student_eval.py` decides whether a checkpoint ships. Everything in it
that runs a model needs a GPU and a customer's footage; everything in it that
turns detections into a table does not, and that second half is where a wrong
answer would be quiet. A foot-anchor error with the sign flipped, a leak-free
subset that silently kept every frame, a latency ratio computed against the
wrong row — none of those would crash, and all of them would change a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from scripts.student_eval import (
    BANDS,
    foot_anchor,
    head_matrix,
    label_reference,
    markdown,
    parse_students,
    tally_frames,
)


@dataclass
class FakeFrame:
    stem: str
    boxes: np.ndarray


@dataclass
class FakeSubject:
    label: str


def frame(stem: str, *boxes: tuple[float, float, float, float]) -> FakeFrame:
    return FakeFrame(stem, np.asarray(boxes, dtype=np.float32).reshape(-1, 4))


def detections(*boxes: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return array, np.full(len(array), 0.9, dtype=np.float32)


# --- naming the subjects ---------------------------------------------------


def test_a_student_can_be_restricted_to_the_paths_it_could_ship_on():
    parsed = parse_students(["engineonly=models/b@engine"], ["engine", "two-pass"])

    assert parsed[0][2] == ["engine"]


def test_a_student_with_no_at_sign_gets_the_default_paths():
    parsed = parse_students(["v1=models/a"], ["engine", "two-pass"])

    assert parsed[0][2] == ["engine", "two-pass"]


@pytest.mark.parametrize("entry", ["no-equals-sign", "a=x@sliced", "a=x@engine,tiles"])
def test_a_malformed_or_unknown_spec_is_refused(entry):
    with pytest.raises(SystemExit):
        parse_students([entry], ["engine"])


def test_two_students_cannot_share_a_name():
    """Labels are `name/path` and they key every table. A collision would
    silently drop one checkpoint's rows and report the other's twice."""
    with pytest.raises(SystemExit):
        parse_students(["a=models/one", "a=models/two"], ["engine"])


# --- the leak-free subset --------------------------------------------------


def test_the_clean_subset_scores_fewer_frames_than_the_whole_split():
    frames = [frame("clean", (0, 0, 10, 100)), frame("flagged", (0, 0, 10, 100))]
    subjects = [FakeSubject("s/engine")]
    cache = {"s/engine": [detections((0, 0, 10, 100)), detections()]}

    everything = tally_frames(subjects, frames, cache, thresholds=(0.5,))
    clean = tally_frames(subjects, frames, cache, thresholds=(0.5,), keep={"clean"})

    assert everything["s/engine"]["0.50"]["iou"]["truth"] == 2
    assert everything["s/engine"]["0.50"]["iou"]["recall"] == pytest.approx(0.5)
    # Dropping the frame this subject missed is exactly the frame it was
    # flagged on, and the recall must move because of it.
    assert clean["s/engine"]["0.50"]["iou"]["truth"] == 1
    assert clean["s/engine"]["0.50"]["iou"]["recall"] == pytest.approx(1.0)


def test_keeping_nothing_is_not_the_same_as_keeping_everything():
    frames = [frame("a", (0, 0, 10, 100))]
    cache = {"s/engine": [detections((0, 0, 10, 100))]}

    empty = tally_frames([FakeSubject("s/engine")], frames, cache, thresholds=(0.5,), keep=set())

    assert empty["s/engine"]["0.50"]["iou"]["truth"] == 0
    assert empty["s/engine"]["0.50"]["iou"]["recall"] is None


# --- the feet --------------------------------------------------------------


def test_a_box_that_stops_at_the_shoulders_reads_negative():
    """The sign carries the finding. A head-only box ends far above the
    person's feet, and `argus/vision/zones.py:anchors` anchors zone membership on
    that bottom edge — so a negative median is a worker placed up-frame of
    where they are standing, not a cosmetic extent error."""
    truth = frame("f", (100, 100, 140, 200))               # 100 px tall person
    cache = {"s/engine": [detections((100, 100, 140, 150))]}   # box ends 50 px high

    result = foot_anchor([FakeSubject("s/engine")], [truth], cache, threshold=0.5)
    band = result["s/engine"]["mid 60-150"]

    assert band["n"] == 1
    assert band["median_px"] == pytest.approx(-50.0)
    assert band["median_frac"] == pytest.approx(-0.5)
    assert band["median_abs_px"] == pytest.approx(50.0)


def test_a_box_that_overshoots_the_feet_reads_positive():
    truth = frame("f", (100, 100, 140, 200))
    cache = {"s/engine": [detections((100, 100, 140, 220))]}

    band = foot_anchor([FakeSubject("s/engine")], [truth], cache, threshold=0.5)["s/engine"]
    assert band["mid 60-150"]["median_px"] == pytest.approx(20.0)


def test_an_unmatched_box_contributes_nothing():
    """A box on empty floor has no truth height to be wrong about and must not
    be averaged in as a zero."""
    truth = frame("f", (100, 100, 140, 200))
    cache = {"s/engine": [detections((100, 100, 140, 200), (900, 900, 940, 1000))]}

    result = foot_anchor([FakeSubject("s/engine")], [truth], cache, threshold=0.5)

    assert sum(result["s/engine"][name]["n"] for name, _, _ in BANDS) == 1


def test_errors_are_bucketed_by_the_truth_height_not_the_prediction():
    """A detector that emits a 40 px box on a 200 px person must appear in the
    near band, saying it was wrong there — not in the far band, where its own
    wrong box would suggest it was measuring a distant worker correctly."""
    truth = frame("f", (100, 100, 140, 300))              # 200 px: near band
    cache = {"s/engine": [detections((100, 100, 140, 140))]}

    result = foot_anchor([FakeSubject("s/engine")], [truth], cache, threshold=0.5)

    assert result["s/engine"]["near >150"]["n"] == 1
    assert result["s/engine"]["far <60"]["n"] == 0


def test_the_median_is_not_moved_by_one_absurd_box():
    truth = FakeFrame("f", np.asarray([[0, 0, 40, 100]] * 5, dtype=np.float32))
    # Four boxes 10 px short, one 800 px long on the same people.
    boxes = [[0, 0, 40, 90]] * 4 + [[0, 0, 40, 900]]
    cache = {"s/engine": [(np.asarray(boxes, dtype=np.float32), np.full(5, 0.9, np.float32))]}

    band = foot_anchor([FakeSubject("s/engine")], [truth], cache, threshold=0.5)
    assert band["s/engine"]["mid 60-150"]["median_px"] == pytest.approx(-10.0)


# --- the derived reference for the third clip ------------------------------


def test_the_label_reference_is_the_median_of_the_scenes_labelled_frames(tmp_path):
    for split, counts in (("train", [18, 19, 24]), ("val", [22, 20])):
        directory = tmp_path / "labels" / split
        directory.mkdir(parents=True)
        for index, count in enumerate(counts):
            body = "".join("0 0.5 0.5 0.1 0.1\n" for _ in range(count))
            (directory / f"Scene_X_frame_{split}{index}.txt").write_text(body)
        (directory / "Other_Scene_frame_99.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    reference = label_reference(tmp_path, "Scene_X")

    assert reference["labelled_frames"] == 5          # the other scene is not counted
    assert sorted(reference["counts"]) == [18, 19, 20, 22, 24]
    assert reference["median_low"] == 20
    assert reference["mean"] == pytest.approx(20.6)


def test_a_scene_with_no_labels_is_an_error_not_a_zero(tmp_path):
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)

    with pytest.raises(SystemExit):
        label_reference(tmp_path, "Scene_X")


# --- the tables ------------------------------------------------------------


def test_latency_is_reported_relative_to_production_on_the_engine_path():
    """The absolutes were measured beside a live appliance and are inflated by
    whatever else the box was doing; the ratio is the part that transfers."""
    text = markdown({
        "operating_point": 0.5,
        "latency": {
            "production/engine": {"views_per_frame": 1, "ms_per_frame_batch1": 6.0,
                                  "ms_per_frame_batch8": 4.0},
            "parity/two-pass": {"views_per_frame": 7, "ms_per_frame_batch1": 60.0,
                                "ms_per_frame_batch8": 32.0},
        },
    })

    assert "| production/engine | 1 | 6.00 ms | 1.00x | 4.00 ms | 1.00x |" in text
    assert "| parity/two-pass | 7 | 60.00 ms | 10.00x | 32.00 ms | 8.00x |" in text


def test_a_derived_headcount_reference_says_so_in_the_table():
    text = markdown({
        "operating_point": 0.5,
        "headcount": {"packing5": {
            "clip": "c.MP4", "hand_count": 19, "frames": 300, "fps": 5.0,
            "reference_source": "label-derived reference",
            "reference_detail": {"labelled_frames": 10, "median_low": 19, "mean": 20.4,
                                 "min": 18, "max": 24},
            "by_subject": {"production/engine": {
                "mean_count": 20.1, "mean_smoothed": 20.0, "within_1": 0.5,
                "within_2": 0.8, "identities": 70}},
        }},
    })

    assert "label-derived reference ~19" in text
    assert "10 harmonised" in text
    assert "A weak reference" in text


def test_both_val_scopes_are_labelled_so_they_cannot_be_confused():
    block = {"s/engine": {f"{t:.2f}": {c: {
        "precision": 0.9, "recall": 0.8, "f1": 0.85, "duplicates": 0, "predictions": 10,
        "truth": 12, "tp": 9,
        "bands": {name: {"found": 1, "total": 2} for name, _, _ in BANDS},
    } for c in ("iou", "centre")} for t in (0.30, 0.40, 0.50, 0.60)}}

    text = markdown({"operating_point": 0.5, "val": block, "val_clean": block,
                     "val_frames": {"val": 16, "val_clean": 12}})

    assert "every val frame (16 frames" in text
    assert "leak-free val frames only (12 frames" in text


# --- the matcher the foot diagnostic needs ---------------------------------


def test_a_head_only_box_matches_the_whole_person_it_is_drawn_on():
    """The point of `head_matrix`. Under centre-in-box this pair does not
    match — the truth's mid-torso centre is below the head box — so the
    production Nano's defining defect would never reach the table."""
    from argus.eval.detector import centre_matrix

    truth = np.asarray([[100, 100, 140, 300]], dtype=np.float32)      # standing worker
    head = np.asarray([[100, 100, 140, 145]], dtype=np.float32)       # head-only box

    assert centre_matrix(head, truth)[0, 0] == 0.0
    assert head_matrix(head, truth)[0, 0] > 0.0


def test_a_box_on_the_person_behind_does_not_match_at_the_head():
    """Eligibility is horizontal centre plus top edge. A box starting at a
    different height is a different person, however much it overlaps."""
    truth = np.asarray([[100, 100, 140, 300]], dtype=np.float32)
    lower = np.asarray([[100, 200, 140, 300]], dtype=np.float32)      # top edge 100 px down

    assert head_matrix(lower, truth)[0, 0] == 0.0


def test_a_box_beside_the_person_does_not_match():
    truth = np.asarray([[100, 100, 140, 300]], dtype=np.float32)
    beside = np.asarray([[200, 100, 260, 300]], dtype=np.float32)

    assert head_matrix(beside, truth)[0, 0] == 0.0


def test_the_top_edge_tolerance_scales_with_the_person():
    """30% of a 200 px worker is 60 px; 30% of a 40 px one is 12. A fixed
    pixel tolerance would be most of a far-field person and none of a near
    one, which is the band the diagnostic is reported by."""
    tall = np.asarray([[100, 100, 140, 300]], dtype=np.float32)
    short = np.asarray([[100, 100, 140, 140]], dtype=np.float32)
    offset = np.asarray([[100, 130, 140, 300]], dtype=np.float32)     # top 30 px down

    assert head_matrix(offset, tall)[0, 0] > 0.0                      # 30 <= 0.3 * 200
    assert head_matrix(offset, short)[0, 0] == 0.0                    # 30 >  0.3 * 40
