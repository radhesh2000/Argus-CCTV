"""One teacher box may lend its extent to one head, and never to two.

`scripts/harmonize_labels.py` rewrites the only ground truth this project has,
from synthetic geometry rather than from a model, so every rule in it is
testable without a GPU, a checkpoint or a frame of the customer's footage —
which is exactly what these do.

The case worth the most is `test_one_teacher_box_cannot_absorb_two_humans`. In
the packing-line clusters the teacher stacks overlapping boxes on partly
occluded workers; a rule that let one of those boxes claim two hand labels would
merge two workers into one and silently delete a person from the corpus, and no
count printed by the script would say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.harmonize_labels import (
    AGREE_IOU, TEACHER_ONLY_THRESHOLD, assign_one_to_one, harmonise_frame, plausible_pairs,
)


def boxes(*values: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 4)


def scores(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def origins(result) -> list[str]:
    return [box.origin for box in result.boxes]


# --- the repair -----------------------------------------------------------


def test_head_label_takes_the_teacher_body():
    """A 40x40 square on a head inside a 40x200 body: the body wins the extent."""
    head = boxes((100, 100, 140, 140))
    body = boxes((98, 96, 142, 296))
    result = harmonise_frame("f", "s", "train", 2560, 1440, head, body, scores(0.9))

    assert origins(result) == ["extent_from_teacher"]
    assert result.boxes[0].xyxy == pytest.approx((98, 96, 142, 296))
    assert result.boxes[0].original_xyxy == pytest.approx((100, 100, 140, 140))


def test_agreeing_boxes_keep_the_human_extent():
    """Where the two conventions already coincide, the human is the localiser.

    Adopting the teacher here would import its error for no gain: the whole
    argument for taking its extent is that the hand label is a head, and this
    hand label is not.
    """
    human = boxes((100, 100, 160, 300))
    teacher = boxes((104, 104, 158, 296))
    result = harmonise_frame("f", "s", "train", 2560, 1440, human, teacher, scores(0.9))

    assert origins(result) == ["human_kept_agreed"]
    assert result.boxes[0].xyxy == pytest.approx((100, 100, 160, 300))


def test_a_squarish_label_the_teacher_agrees_with_is_not_a_head():
    """Squarish is evidence, not proof.

    docs/TEACHER_LABELS.md §3.2 uses aspect alone as the proxy for "this is a
    head", which is right as a population statistic and wrong per box: a worker
    bent over a bench is genuinely square. The height ratio is what separates
    them, and without it this box would be stretched onto a body that is not
    there.
    """
    human = boxes((100, 100, 200, 200))
    teacher = boxes((98, 98, 202, 205))
    result = harmonise_frame("f", "s", "train", 2560, 1440, human, teacher, scores(0.9))

    assert origins(result) == ["human_kept_agreed"]


def test_unmatched_human_is_kept_and_flagged():
    human = boxes((100, 100, 140, 140))
    teacher = boxes((900, 900, 950, 1100))
    result = harmonise_frame("f", "s", "train", 2560, 1440, human, teacher, scores(0.9))

    assert origins(result) == ["extent_unverified", "teacher_only"]
    assert result.boxes[0].xyxy == pytest.approx((100, 100, 140, 140))


# --- many-to-one ----------------------------------------------------------


def test_one_teacher_box_cannot_absorb_two_humans():
    """Two heads, one teacher body. One is repaired; the other keeps its own box.

    The alternative — both heads taking the same body — writes the identical box
    twice and turns two workers into one duplicated person. Losing the contest
    is the correct outcome for the second head: it is still a person (a human
    drew it), its extent is simply unconfirmed.
    """
    heads = boxes((100, 100, 140, 140), (150, 110, 190, 150))
    body = boxes((96, 96, 200, 300))
    result = harmonise_frame("f", "s", "train", 2560, 1440, heads, body, scores(0.9))

    assert sorted(origins(result)) == ["extent_from_teacher", "extent_unverified"]
    assert result.conflicts == 1
    kept = [b for b in result.boxes if b.origin == "extent_unverified"][0]
    assert kept.xyxy in (pytest.approx((100, 100, 140, 140)), pytest.approx((150, 110, 190, 150)))
    # No two output boxes are the same box.
    assert len({b.xyxy for b in result.boxes}) == len(result.boxes)


def test_agreement_outranks_inference_for_a_contested_box():
    """An IoU 0.5 pair is an identification and may not be outbid by a head.

    Ranking purely on containment would hand the body to the fully-contained
    head and leave the person whose box *is* that body unverified — the exact
    inversion this two-phase assignment exists to prevent.
    """
    human = boxes((100, 100, 140, 140), (98, 96, 142, 296))
    teacher = boxes((96, 96, 144, 300))
    result = harmonise_frame("f", "s", "train", 2560, 1440, human, teacher, scores(0.9))

    assert origins(result) == ["extent_unverified", "human_kept_agreed"]
    assert result.conflicts == 1


# --- teacher-only ---------------------------------------------------------


def test_teacher_only_needs_the_higher_threshold():
    teacher = boxes((900, 900, 950, 1100), (500, 500, 550, 700))
    result = harmonise_frame(
        "f", "s", "train", 2560, 1440, boxes(), teacher,
        scores(TEACHER_ONLY_THRESHOLD + 0.05, TEACHER_ONLY_THRESHOLD - 0.05),
    )

    assert origins(result) == ["teacher_only"]
    assert result.boxes[0].xyxy == pytest.approx((900, 900, 950, 1100))


def test_teacher_only_is_dropped_when_it_lands_on_somebody_already_here():
    """A stacked box on a worker the set already has is a duplicate, not a find.

    The teacher's own NMS runs at 0.55, so two of its boxes on one occluded
    worker can both survive at IoU 0.4 — under the merge's bar and over the bar
    for "this is the same person".
    """
    human = boxes((100, 100, 160, 300))
    teacher = boxes((104, 104, 158, 296), (108, 150, 165, 330))
    result = harmonise_frame("f", "s", "train", 2560, 1440, human, teacher, scores(0.9, 0.9))

    assert origins(result) == ["human_kept_agreed"]
    assert result.teacher_only_suppressed == 1


# --- plausibility ---------------------------------------------------------


def test_centre_outside_the_teacher_box_is_never_a_pair():
    """The necessary condition, on its own.

    Two workers side by side: the left one's head overlaps the right one's body
    box, but its centre is outside it, so no extent crosses between them.
    """
    human = boxes((100, 100, 140, 140))
    teacher = boxes((139, 96, 240, 300))
    assert plausible_pairs(human, teacher) == []


def test_a_box_low_in_the_teacher_box_needs_real_overlap():
    """Position or overlap — a small box on somebody's shins is neither.

    Its centre is inside the teacher box and it is nowhere near the upper part,
    so it must clear the containment bar on its own to pair; a box mostly
    outside the teacher's does not.
    """
    shins = boxes((100, 250, 190, 290))
    body = boxes((96, 96, 144, 300))
    assert plausible_pairs(shins, body) == []


def test_assignment_is_stable_and_one_to_one():
    human = boxes((100, 100, 140, 140), (300, 100, 340, 140), (500, 100, 540, 140))
    teacher = boxes((96, 96, 144, 300), (296, 96, 344, 300), (496, 96, 544, 300))
    pairs = plausible_pairs(human, teacher)
    accepted, displaced = assign_one_to_one(pairs)

    assert len(accepted) == 3
    assert displaced == []
    assert len({pair.teacher for pair in accepted.values()}) == 3


def test_agree_iou_matches_the_projects_matching_threshold():
    """The harmoniser and every table in docs/ must mean the same thing by "agree"."""
    from argus.eval.detector import MATCH_IOU

    assert AGREE_IOU == MATCH_IOU
