"""Evidence drawing: the snapshot has to show why the alert fired."""

from __future__ import annotations

import numpy as np
from PIL import Image

from argus.media.annotate import (
    SEVERITY_COLORS,
    _font,
    annotate_zone_event,
    drawable,
    draw_banner,
    draw_box,
    draw_zone_overlay,
    hex_to_rgb,
    save_snapshot,
    to_image,
    track_label,
)

WIDTH, HEIGHT = 320, 180
SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def _frame(value: int = 30) -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)


def test_bgr_frames_are_converted_to_rgb():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, :, 2] = 255                      # red in BGR
    assert to_image(frame).getpixel((0, 0)) == (255, 0, 0)


def test_hex_to_rgb_parses_zone_colours_and_survives_junk():
    assert hex_to_rgb("#22d3ee") == (34, 211, 238)
    assert hex_to_rgb("22d3ee") == (34, 211, 238)
    assert hex_to_rgb("not-a-colour") == (34, 211, 238)


def test_banner_covers_the_top_and_leaves_the_rest_alone():
    image = to_image(_frame())
    draw_banner(image, "Packing floor: 4 people - limit 2")
    pixels = np.asarray(image)
    assert pixels[:5].mean() != 30            # strip drawn
    assert pixels[-5:].mean() == 30           # bottom untouched


def test_zone_overlay_tints_inside_and_outlines_the_edge():
    image = to_image(_frame())
    draw_zone_overlay(image, SQUARE, (34, 211, 238))
    pixels = np.asarray(image)
    assert pixels[HEIGHT // 2, WIDTH // 2].tolist() != [30, 30, 30]     # inside tinted
    assert pixels[2, 2].tolist() == [30, 30, 30]                        # outside clean


def test_box_labels_stay_inside_the_frame():
    image = to_image(_frame())
    draw_box(image, [4, 1, 60, 40], "Worker 3", SEVERITY_COLORS["high"])
    assert np.asarray(image)[0:3, 4:40].mean() != 30    # chip pushed inside, not clipped away


def test_annotated_event_draws_zone_and_banner_but_no_per_person_boxes():
    """Per-person boxes used to be burned into these pixels — permanently, on
    top of each other, with no way to tell one crowded detection from another
    once saved. They are carried separately in the event's `meta.boxes` and
    drawn as an interactive overlay by the console instead, so the saved
    snapshot only shows the zone shading and the headline: nothing at the
    frame's corner (outside both the banner and the zone) should be touched,
    which a leftover per-person box drawn there would break."""
    frame = _frame()
    image = annotate_zone_event(
        frame,
        polygon_norm=SQUARE,
        headline="Packing floor: 2 people - limit 1",
        color=SEVERITY_COLORS["high"],
    )
    assert image.size == (WIDTH, HEIGHT)
    pixels = np.asarray(image)
    assert pixels[:5].mean() != 30                       # banner strip drawn
    assert pixels[HEIGHT // 2, WIDTH // 2].tolist() != [30, 30, 30]  # zone tinted
    assert pixels[HEIGHT - 1, 0].tolist() == [30, 30, 30]  # bottom-left corner untouched
    # The source frame is never modified in place — it is still live video.
    assert (frame == 30).all()


def test_annotation_works_with_no_polygon_area():
    image = annotate_zone_event(
        _frame(), polygon_norm=SQUARE, headline="empty", color=SEVERITY_COLORS["medium"],
    )
    assert image.size == (WIDTH, HEIGHT)


def test_snapshot_is_written_atomically(tmp_path):
    destination = tmp_path / "cam" / "zone_1700000000.jpg"
    written = save_snapshot(to_image(_frame()), destination, quality=80)
    assert written == destination and destination.exists()
    assert not list(tmp_path.glob("**/*.tmp"))
    with Image.open(destination) as image:
        assert image.size == (WIDTH, HEIGHT)
        assert image.format == "JPEG"


def test_track_labels_read_like_the_product():
    assert track_label(7) == "Worker 7"
    assert track_label(7, 0.884) == "Worker 7 · 88%"
    assert track_label(None) == "Worker"


def test_the_font_cannot_draw_an_em_dash_and_we_no_longer_ask_it_to():
    """Pillow's bundled Aileron has no em dash, arrow, ellipsis or geometric
    shape, and draws `.notdef` — a tofu rectangle — for each. Found burned into
    a real snapshot on 2026-08-25: the headline `Idle or conversing manpower —
    Gangway by the sorting bench` had a box where the dash should be, in both
    the banner and the box chip."""
    font = _font(22)
    assert drawable("a — b", font) == "a - b"
    assert drawable("5 × 3 ≤ 4 … → done", font) == "5 x 3 <= 4 ... -> done"
    # The characters Aileron *does* have are left exactly as written: the
    # product's own typography (`Worker 7 · 88%`, `24 °C`) must not be mangled.
    assert drawable("Worker 7 · 88% ± 2 °C", font) == "Worker 7 · 88% ± 2 °C"
    assert drawable("plain ascii", font) == "plain ascii"


def test_a_character_nothing_can_draw_becomes_a_question_mark():
    """Honest placeholder over a rectangle a reviewer reads as a corrupt file.
    This matters because half the headline is the model's own sentence."""
    assert drawable("cyrillic \u0416 here", _font(22)) == "cyrillic ? here"


def test_the_banner_is_sized_around_what_it_will_actually_draw():
    """`…` becomes three characters, so measuring before substituting would
    size the strip for the wrong string."""
    wide = Image.new("RGB", (WIDTH, HEIGHT), (10, 10, 10))
    draw_banner(wide, "count: 3 … of 4")
    assert wide.getpixel((WIDTH // 2, 2)) != (10, 10, 10)
