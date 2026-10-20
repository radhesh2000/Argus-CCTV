"""A sliver of shoulder at a tile edge is not a person.

`scripts/build_student_dataset.py` cuts every frame into the same 3x2 native
tiles the teacher and the two-pass inference path use, and the only judgement
call in it is what to do with a box the tile cuts through. Teach the model that
a 10% sliver is a person and it will put boxes on shoulders; drop everything
that touches an edge and the tiles teach nothing about the people standing near
one. These pin where the line is drawn and that it is drawn on *area kept*,
which is the thing a reader of the corpus would want to know.

The other half is the split rule, which is not a judgement call at all: the
guard band exists to be unused and the heldout frames are the only measurement
this student has. That refusal is tested because a typo in a shell command is
exactly how a test set gets trained on, and nothing downstream would ever say so.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from scripts.build_student_dataset import (
    ENGINE_HEIGHT,
    ENGINE_WIDTH,
    MIN_KEPT_AREA,
    clip_to_tile,
    engine_view,
    read_harvest,
    tiles_kept,
)
from scripts.teacher_label import Tile, plan_tiles


def tile(x0: int, y0: int, x1: int, y1: int) -> Tile:
    return Tile(row=0, col=0, x0=x0, y0=y0, x1=x1, y1=y1,
                interior_left=True, interior_top=True, interior_right=True, interior_bottom=True)


def boxes(*values: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 4)


def test_a_box_wholly_inside_is_moved_not_changed():
    kept, dropped = clip_to_tile(boxes((120, 140, 180, 340)), tile(100, 100, 1124, 923))

    assert dropped == 0
    assert kept[0] == pytest.approx((20, 40, 80, 240))


def test_a_mostly_present_box_is_clipped_and_kept():
    """75% of the person is in the tile: taught, clipped to what the tile shows.

    This is not a compromise. The tile is the image, and in that image the
    person genuinely ends at its edge — which is also exactly what the model is
    asked to emit there at inference, before the seam rule discards it.
    """
    kept, dropped = clip_to_tile(boxes((0, 100, 400, 300)), tile(100, 0, 1124, 823))

    assert dropped == 0
    assert kept[0] == pytest.approx((0, 100, 300, 300))


def test_a_sliver_is_dropped():
    """10% of a person left. A box on that is a box on a shoulder."""
    kept, dropped = clip_to_tile(boxes((0, 100, 400, 300)), tile(360, 0, 1384, 823))

    assert len(kept) == 0
    assert dropped == 1


def test_the_boundary_is_the_area_kept():
    height, width = 200.0, 400.0
    # Cut so that exactly MIN_KEPT_AREA of the width survives.
    cut = int(width * (1 - MIN_KEPT_AREA))
    kept, _ = clip_to_tile(boxes((0, 0, width, height)), tile(cut, 0, cut + 1024, 823))
    assert len(kept) == 1

    kept, dropped = clip_to_tile(boxes((0, 0, width, height)), tile(cut + 8, 0, cut + 1032, 823))
    assert len(kept) == 0 and dropped == 1


def test_a_box_elsewhere_is_not_counted_as_a_dropped_sliver():
    """"Dropped" must mean truncated, not absent.

    Counting every box outside the tile would make the drop rate a statement
    about tile geometry — six tiles, so five-sixths of every frame's boxes
    "dropped" — and the number would be useless for spotting a real problem.
    """
    kept, dropped = clip_to_tile(boxes((2000, 1000, 2100, 1300)), tile(0, 0, 1024, 823))

    assert len(kept) == 0
    assert dropped == 0


def test_a_flat_box_is_dropped_whatever_its_area_ratio():
    """A 2 px strip keeps most of a 2 px-tall box's area and is still not a person."""
    kept, dropped = clip_to_tile(boxes((100, 100, 300, 102)), tile(0, 0, 1024, 823))

    assert len(kept) == 0
    assert dropped == 1


# --- the split rule --------------------------------------------------------


@pytest.mark.parametrize("split", ["heldout", "guard"])
def test_the_harvest_measurement_splits_cannot_be_trained_on(split, tmp_path):
    with pytest.raises(SystemExit) as raised:
        read_harvest(tmp_path, tmp_path, [split], 0.5)

    assert split in str(raised.value)


# --- the tiles are the teacher's tiles -------------------------------------


def test_tiles_here_are_the_tiles_at_inference():
    """One geometry, imported, never re-derived.

    If this corpus were cut on a grid that differed from the one
    `scripts/teacher_label.py` runs at inference by even a row of pixels, the
    student would be trained on crops it never sees and the whole argument for
    tiling in training would quietly not apply.
    """
    tiles = plan_tiles(2560, 1440)

    assert len(tiles) == 6
    assert {(t.x1 - t.x0, t.y1 - t.y0) for t in tiles} == {(1024, 823)}
    assert tiles[0].x0 == 0 and tiles[-1].x1 == 2560
    assert tiles[0].y0 == 0 and tiles[-1].y1 == 1440


# --- the parity knobs ------------------------------------------------------
#
# docs/STUDENT_DETECTOR.md §8: v1's corpus was 14% full frames and the model
# optimised for the other 86%. These pin the two mechanisms that change the
# mix, because both of them are the kind of thing that can be subtly wrong and
# still produce a corpus that trains.


def test_dropping_tiles_rotates_so_no_region_is_lost_for_good():
    """Keeping the *first* K tiles of every frame would delete a strip of the
    image from the corpus — the exact failure `plan_tiles` exists to avoid."""
    planned = plan_tiles(2560, 1440)

    seen = Counter()
    for index in range(30):
        kept = tiles_kept(planned, 4, index)
        assert len(kept) == 4
        assert len({(t.row, t.col) for t in kept}) == 4      # no tile twice in a frame
        seen.update((t.row, t.col) for t in kept)

    assert len(seen) == 6                                     # every position appears
    assert set(seen.values()) == {20}                         # and equally often: 4/6 of 30


def test_keeping_every_tile_is_the_identity():
    planned = plan_tiles(2560, 1440)

    assert tiles_kept(planned, 6, 7) == planned
    assert tiles_kept(planned, 99, 3) == planned


def test_the_engine_view_is_the_size_the_decoder_hands_over(tmp_path):
    """960x544 with the aspect ratio ignored, because the appliance ignores it.

    A view that preserved 16:9 would be a picture the engine never produces,
    and training on it would leave the student in exactly v1's position: fluent
    in a view nobody runs.
    """
    from PIL import Image

    source = tmp_path / "frame.jpg"
    Image.new("RGB", (2560, 1440), (30, 40, 50)).save(source)

    out = engine_view(source, tmp_path / "cache" / "frame.jpg", 92)

    with Image.open(out) as handle:
        assert handle.size == (ENGINE_WIDTH, ENGINE_HEIGHT)
    assert (ENGINE_WIDTH / ENGINE_HEIGHT) != pytest.approx(2560 / 1440)


def test_the_engine_view_matches_the_resize_the_eval_harness_uses(tmp_path):
    """One resize, two call sites. `scripts/student_eval.py` builds its engine
    view with `argus.eval.detector._resize`; if this drifted from that, the
    student would be trained on one filter and scored on another."""
    import numpy as np
    from PIL import Image

    from argus.eval.detector import _load_bgr, _resize

    rng = np.random.default_rng(0)
    source = tmp_path / "frame.jpg"
    Image.fromarray(rng.integers(0, 255, (288, 512, 3), dtype=np.uint8)).save(source, quality=100)

    mine = _load_bgr(engine_view(source, tmp_path / "engine.jpg", 100))
    theirs = _resize(_load_bgr(source), ENGINE_WIDTH, ENGINE_HEIGHT)

    # JPEG round-trip at q100 is not bit-exact; the resample must still be the
    # same one, which a mean absolute difference of a couple of levels shows
    # and a different filter (nearest, bicubic) would not.
    assert np.abs(mine.astype(float) - theirs.astype(float)).mean() < 3.0
