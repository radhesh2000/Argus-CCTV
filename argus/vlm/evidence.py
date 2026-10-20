"""The pictures the VLM path composes: what a reviewer is shown, and what the
model is shown.

Same rule as ``argus.media.annotate`` for the first: **every snapshot shows its
own violation.** A detector event proves itself with boxes and a zone; a VLM
event has only the model's sentence and, when it grounded its answer, one box.
So the box is drawn if there is one, and the reason is burned into the banner
either way — a reviewer must never have to open another screen to learn why a
frame is in front of them.

The second is :func:`crop_to_zone`, which frames the *question* rather than the
answer: a claim about a zone is asked over that zone, not over the whole camera
view. It is the only function here that takes pixels away instead of adding
them.

This lives in ``argus.vlm`` rather than in ``argus.media.annotate`` on purpose:
it composes that module's primitives and adds nothing to them, so the Phase-1
drawing code stays exactly as it was.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PIL import Image

from argus.media.annotate import SEVERITY_COLORS, RGB, draw_banner, draw_box, to_image
from argus.vision.zones import polygon_bounds
from argus.vlm.schema import PolicyVerdict

# Below this, a crop is a postage stamp: the sides are grown around the zone's
# centre until it is at least this many pixels each way. A vision encoder sees a
# fixed grid of patches, so a 140x90 cut-out of a 960x544 frame is not a closer
# look at the zone, it is fewer pixels of it than the wide shot already had.
MIN_CROP_PX = 320

# Above this share of the frame, cropping is not worth doing: the zone *is* the
# view. The demo box has one of each — 'Packing floor' spans 0.05-0.95 x
# 0.20-0.95 and is left alone, 'Conveyor side' spans 0.35-0.95 x 0.30-0.90 and
# is cropped. Returning the frame untouched says "there was no sub-picture to
# find here", which is true, and is better than a 3% crop that pretends there
# was.
MAX_CROP_FRACTION = 0.8


def annotate_policy_event(
    frame: np.ndarray | Image.Image,
    verdict: PolicyVerdict,
    *,
    label: str,
    severity: str = "high",
    box_color: RGB | None = None,
) -> Image.Image:
    """Compose the snapshot for a fired VLM policy. Returns a new image.

    ``verdict.box_2d`` is a 0-1 fraction of the frame (the client normalises it
    on the way out, because the model saw a downscaled copy). Scaling it here
    means the box lands correctly whatever resolution this frame happens to be —
    the live 960x544 decoder output, or a full-resolution still from the API.
    """
    image = to_image(frame)
    color = box_color or SEVERITY_COLORS.get(severity, SEVERITY_COLORS["high"])

    if verdict.box_2d is not None:
        width, height = image.size
        x1, y1, x2, y2 = verdict.box_2d
        draw_box(image, (x1 * width, y1 * height, x2 * width, y2 * height), label, color)

    headline = f"{label}: {verdict.reason}" if verdict.reason else label
    draw_banner(image, headline[:120], color)
    return image


def annotate_veto(
    frame: np.ndarray | Image.Image, *, headline: str, reason: str
) -> Image.Image:
    """Re-banner an already-annotated snapshot as suppressed.

    The frame keeps its detector overlay — boxes, zone, counts — and gains a
    second strip saying the verifier disagreed and why. A reviewer opening a
    suppressed event needs to see *both* claims at once to judge which was right,
    which is the entire purpose of keeping suppressed events visible.
    """
    image = to_image(frame)
    # A middle dot rather than an em dash — the bundled font has no em dash and
    # draws a tofu box for it (argus/media/annotate.py, `drawable`).
    draw_banner(image, f"SUPPRESSED · {reason}"[:120], SEVERITY_COLORS["low"])
    return image


def box_meta(verdict: PolicyVerdict) -> dict[str, Any] | None:
    """The grounding box as it is stored in ``events.meta``.

    Normalised 0-1, same convention as the detector's ``meta['boxes']``, so the
    UI redraws either kind of event with one code path.
    """
    if verdict.box_2d is None:
        return None
    x1, y1, x2, y2 = verdict.box_2d
    return {"box_2d": [round(float(v), 4) for v in (x1, y1, x2, y2)], "coords": "normalized"}


def clamp_polygon(points: Sequence[Sequence[float]]) -> list[list[float]]:
    """Keep a normalized polygon inside the frame (defensive, for API callers)."""
    return [[min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))] for x, y in points]


def zone_crop(
    frame: np.ndarray | Image.Image,
    polygon: Sequence[Sequence[float]],
    *,
    min_px: int = MIN_CROP_PX,
    max_fraction: float = MAX_CROP_FRACTION,
) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    """:func:`crop_to_zone`, plus the rectangle it was cut from.

    The rectangle is normalized 0-1 against the *source* frame, and it is what
    lets a caller undo the crop: a model shown a zone answers with ``box_2d`` in
    the crop's coordinates, and an evidence snapshot that drew those on the wide
    shot would put the box in the wrong place — near the top-left corner, which
    reads as a detection rather than as a bug. Scheduled zone policies map the
    box back through this and annotate the full frame, so the reviewer keeps the
    context and the box still lands on the person.

    The verification pass does not need it (it asks for ``box_2d`` null), which
    is why :func:`crop_to_zone` still exists as the one-value form.
    """
    try:
        points = [(float(x), float(y)) for x, y in polygon]
    except (TypeError, ValueError):
        return None
    if len(points) < 3:
        return None

    image = to_image(frame)
    width, height = image.size
    if width < 2 or height < 2:
        return None

    x1, y1, x2, y2 = polygon_bounds(points)
    left, top = round(x1 * width), round(y1 * height)
    right, bottom = round(x2 * width), round(y2 * height)
    left, top, right, bottom = _grow_to(
        (left, top, right, bottom), width=width, height=height, min_px=int(min_px)
    )
    if right - left < 2 or bottom - top < 2:
        return None
    if (right - left) * (bottom - top) >= max_fraction * width * height:
        return None
    return image.crop((left, top, right, bottom)), (
        left / width, top / height, right / width, bottom / height,
    )


def box_in_frame(
    box: tuple[float, float, float, float], crop: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """A 0-1 box inside a crop -> the same box 0-1 inside the whole frame."""
    left, top, right, bottom = crop
    span_x, span_y = right - left, bottom - top
    x1, y1, x2, y2 = box
    return (left + x1 * span_x, top + y1 * span_y, left + x2 * span_x, top + y2 * span_y)


def crop_to_zone(
    frame: np.ndarray | Image.Image,
    polygon: Sequence[Sequence[float]],
    *,
    min_px: int = MIN_CROP_PX,
    max_fraction: float = MAX_CROP_FRACTION,
) -> Image.Image | None:
    """The zone as its own picture, or None when there is no point cropping.

    A headcount claim in this product is always about a zone — *"Zone 'Packing
    floor' over sanctioned limit: 5 people (limit 3)"* — and the verification
    pass was asking that question over the entire camera view. That is the
    difference between a question a model can answer by counting and a question
    it can only answer by impression, and the eval harness measured exactly how
    much it costs: on the same model, the same prompt and the same footage,
    over-claims at twice the plausible headcount were vetoed **12/12 on zone
    crops and 3/12 on the wide shots they were cut from** (docs/VLM_EVAL.md).
    Every failure the production wording had on that variant was a wide shot.

    None means "send what you already had". Two cases produce it, and neither is
    an error: a zone whose bounds cover most of the view (nothing to cut), and a
    polygon that is unusable (fewer than three points, or degenerate). The caller
    must treat None as "no crop", never as "do not verify" — a verification that
    silently stopped happening because a zone was drawn badly is the failure this
    subsystem exists to avoid.
    """
    found = zone_crop(frame, polygon, min_px=min_px, max_fraction=max_fraction)
    return None if found is None else found[0]


def _grow_to(
    box: tuple[int, int, int, int], *, width: int, height: int, min_px: int
) -> tuple[int, int, int, int]:
    """Widen a pixel box to ``min_px`` a side, around its own centre, inside the frame.

    Grown symmetrically and then slid back inside the frame rather than clamped
    edge by edge, so a zone against the left wall keeps its full width instead of
    losing half of it to a clamp at x=0.
    """
    left, top, right, bottom = box
    for axis in (0, 1):
        low, high = (left, right) if axis == 0 else (top, bottom)
        limit = width if axis == 0 else height
        want = min(min_px, limit)
        if high - low < want:
            centre = (low + high) / 2.0
            low, high = round(centre - want / 2), round(centre - want / 2) + want
            if low < 0:
                low, high = 0, want
            elif high > limit:
                low, high = limit - want, limit
        if axis == 0:
            left, right = low, high
        else:
            top, bottom = low, high
    return (max(0, left), max(0, top), min(width, right), min(height, bottom))


__all__ = [
    "MAX_CROP_FRACTION",
    "MIN_CROP_PX",
    "annotate_policy_event",
    "annotate_veto",
    "box_in_frame",
    "box_meta",
    "clamp_polygon",
    "crop_to_zone",
    "zone_crop",
]
