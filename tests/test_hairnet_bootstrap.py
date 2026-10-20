"""The hairnet label bootstrap — the decisions, without the model.

`scripts/bootstrap_hairnet.py` proposes head crops and a first guess at hair
covering, for a human to correct. Nothing here loads OWLv2 or touches a GPU:
what is worth testing is the part between the model's scores and the directory
a crop lands in, because that is where a bootstrap quietly becomes a training
set full of coin tosses.

The rule under test, stated once: a box whose winning phrase is the neutral
anchor goes to `unsure/`, and so does a box whose two best phrases vote for
*different* classes and are within `--margin` of each other. A 0.31-vs-0.29
split is a coin toss wearing a decimal point, and the whole value of the pass is
that the coin tosses are separated out rather than mixed in.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from argus.config import ROOT

SCRIPT = ROOT / "scripts" / "bootstrap_hairnet.py"
GUIDE = ROOT / "docs" / "HAIRNET_BOOTSTRAP.md"


@pytest.fixture(scope="module")
def bootstrap():
    """The script as a module. Loaded by path, like the trainer: it is a script
    and every heavy import (torch, transformers, PIL) is inside a function."""
    spec = importlib.util.spec_from_file_location("argus_bootstrap_hairnet", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# the phrase list
# --------------------------------------------------------------------------


def test_every_phrase_votes_for_a_real_bucket(bootstrap):
    votes = {votes for _, votes in bootstrap.PHRASES}
    assert votes <= set(bootstrap.BUCKETS)
    assert {"hairnet", "no_hairnet", "unsure"} <= votes, (
        "the neutral anchor is what lets the model say 'a head, and no more'"
    )


def test_every_phrase_is_about_a_head_not_about_a_net(bootstrap):
    """OWLv2 scores a phrase, so 'a hairnet' boxes the fabric rather than the
    head and the crops stop being comparable to each other."""
    for text, _ in bootstrap.PHRASES:
        assert "head" in text, text


# --------------------------------------------------------------------------
# which bucket a box lands in
# --------------------------------------------------------------------------


def _index_of(bootstrap, votes: str) -> int:
    return next(i for i, (_, v) in enumerate(bootstrap.PHRASES) if v == votes)


def test_a_confident_winner_takes_its_own_class(bootstrap):
    covered = _index_of(bootstrap, "hairnet")
    assert bootstrap.bucket_for(covered, 0.40, 0.10, margin=0.03) == "hairnet"
    bare = _index_of(bootstrap, "no_hairnet")
    assert bootstrap.bucket_for(bare, 0.40, 0.10, margin=0.03) == "no_hairnet"


def test_the_neutral_anchor_always_abstains(bootstrap):
    anchor = _index_of(bootstrap, "unsure")
    assert bootstrap.bucket_for(anchor, 0.90, 0.0, margin=0.03) == "unsure"


def test_a_narrow_split_between_the_classes_abstains(bootstrap):
    covered = _index_of(bootstrap, "hairnet")
    assert bootstrap.bucket_for(covered, 0.31, 0.29, margin=0.03) == "unsure"
    assert bootstrap.bucket_for(covered, 0.31, 0.20, margin=0.03) == "hairnet"


def test_no_runner_up_is_not_a_narrow_split(bootstrap):
    """A box only one phrase ever scored is not ambiguous, it is unopposed."""
    covered = _index_of(bootstrap, "hairnet")
    assert bootstrap.bucket_for(covered, 0.13, 0.0, margin=0.03) == "hairnet"


# --------------------------------------------------------------------------
# merging one head's several descriptions into one box
# --------------------------------------------------------------------------


def test_nms_is_class_agnostic_so_one_head_is_one_crop(bootstrap):
    """Seven phrases are seven ways of describing one head. Per-class NMS would
    emit the same head three times under three labels."""
    covered = _index_of(bootstrap, "hairnet")
    bare = _index_of(bootstrap, "no_hairnet")
    kept = bootstrap.merge_boxes([
        (0.40, [10, 10, 50, 50], covered),
        (0.28, [11, 11, 51, 51], bare),
    ], iou=0.5)
    assert len(kept) == 1
    score, _, index, runner, runner_score = kept[0]
    assert (score, index) == (0.40, covered)
    assert runner_score == 0.28 and runner == bootstrap.PHRASES[bare][0]


def test_a_suppressed_box_from_the_same_class_is_not_a_runner_up(bootstrap):
    """Two hairnet phrases agreeing says nothing about the abstention band, so
    recording one as the runner-up would make an unopposed box look contested."""
    hairnets = [i for i, (_, v) in enumerate(bootstrap.PHRASES) if v == "hairnet"]
    kept = bootstrap.merge_boxes([
        (0.40, [10, 10, 50, 50], hairnets[0]),
        (0.38, [11, 11, 51, 51], hairnets[1]),
    ], iou=0.5)
    assert len(kept) == 1
    assert kept[0][4] == 0.0
    assert bootstrap.bucket_for(kept[0][2], kept[0][0], kept[0][4], margin=0.03) == "hairnet"


def test_two_separate_heads_stay_two_boxes(bootstrap):
    covered = _index_of(bootstrap, "hairnet")
    kept = bootstrap.merge_boxes([
        (0.40, [10, 10, 50, 50], covered),
        (0.35, [200, 200, 240, 240], covered),
    ], iou=0.5)
    assert len(kept) == 2


# --------------------------------------------------------------------------
# the crop a human is asked to judge
# --------------------------------------------------------------------------


def test_a_head_box_is_padded_into_something_judgeable(bootstrap):
    """A tight head box at this camera distance is 20-40 px on a side, and
    nobody says 'hairnet or not' to a 24x24 thumbnail."""
    box = bootstrap.expand([100, 100, 130, 130], 2560, 1440, pad=1.4, min_px=96)
    left, top, right, bottom = box
    assert right - left >= 96 and bottom - top >= 96
    assert (left + right) / 2 == pytest.approx(115, abs=1)


def test_a_crop_never_leaves_the_frame(bootstrap):
    left, top, right, bottom = bootstrap.expand([0, 0, 20, 20], 640, 480, pad=1.4, min_px=96)
    assert (left, top) == (0, 0)
    assert right <= 640 and bottom <= 480
    assert right > left and bottom > top


def test_a_box_at_the_far_corner_is_still_a_crop(bootstrap):
    left, top, right, bottom = bootstrap.expand(
        [630, 470, 640, 480], 640, 480, pad=1.4, min_px=96,
    )
    assert right > left and bottom > top


# --------------------------------------------------------------------------
# provenance and the warning that has to travel with it
# --------------------------------------------------------------------------


def test_a_scene_name_survives_every_frame_id_spelling(bootstrap):
    """Three spellings reach this function: the labelled dataset's stills, the
    stills this tool cuts from the videos, and the hyphenated form. Missing one
    is not cosmetic — SCENE_REGIONS is keyed by the answer, so an unstripped
    suffix silently gives Broken_Biscuits its participant tiles back."""
    assert bootstrap.scene_of(Path("Loading_Point_frame_00.jpg")) == "Loading_Point"
    assert bootstrap.scene_of(Path("Loading_Point_t000300.jpg")) == "Loading_Point"
    assert bootstrap.scene_of(Path("Broken-Biscuits-frame-07.png")) == "Broken-Biscuits"
    assert bootstrap.scene_of(Path("Broken_Biscuits_t005700.jpg")) == "Broken_Biscuits"
    assert bootstrap.scene_of(Path("Broken_Biscuits_t005700.jpg")) in bootstrap.SCENE_REGIONS


def test_the_guide_exists_and_refuses_to_claim_a_detector():
    """The one thing this whole area must not do is get described as a hairnet
    detector before it is one."""
    text = GUIDE.read_text()
    assert "There is no hairnet detector in Argus." in text
    assert "NOT YET CONFIRMED" in text, (
        "the site's own rule about nets vs caps vs scarves is not a model's to guess"
    )


def test_the_bootstrap_output_is_gitignored():
    """Somebody's people's heads, and derived data a re-run reproduces."""
    assert "datasets/" in (ROOT / ".gitignore").read_text().splitlines()


# --------------------------------------------------------------------------
# the part of the frame that is a camera
# --------------------------------------------------------------------------


def test_the_screen_recording_scene_is_restricted_to_the_shared_picture(bootstrap):
    """`Broken_Biscuits.mov` is a screen recording of a video call, with
    participant tiles across the top. On the first run of this tool six of
    thirteen candidates were the participant's face — filed as `hairnet`,
    `no_hairnet` and `unsure` across three crops of the same head, with the
    largest head boxes in the set. A named person on a video call, harvested
    from a customer's file into a training set, is the one thing this must not
    do quietly."""
    region = bootstrap.SCENE_REGIONS["Broken_Biscuits"]
    x1, y1, x2, y2 = region
    assert 0.0 < x1 < x2 < 1.0 and 0.0 < y1 < y2 <= 1.0
    # The tiles are the top ~15% of the file; the region has to start below them.
    assert y1 >= 0.15


def test_a_region_is_a_constant_not_a_flag(bootstrap):
    """Forgetting the flag is the failure mode, so there is no flag. Every scene
    that needs one is named in the source and reported in provenance.json."""
    names = {action.dest for action in bootstrap.build_parser()._actions}
    assert "scene_regions" not in names
    assert "region" not in names


def test_a_scene_without_a_region_is_scanned_whole(bootstrap):
    assert bootstrap.SCENE_REGIONS.get("Loading_Point") is None
    assert bootstrap.SCENE_REGIONS.get("Line_No.3_Packing3_Bas") is None
