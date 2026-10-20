"""Drawing the evidence snapshot.

Ported from v1 (``pipeline/frames.py`` on ``v1-archive``), keeping the rule that made v1's evidence
readable: **every snapshot shows its own violation.** A reviewer sees the zone
shaded, the people *inside* it boxed in the severity colour, everyone else
dimmed to grey, and a headline strip stating the count against the limit. No
cross-referencing another screen to work out what tripped.

Rewritten on Pillow instead of OpenCV. v2 has no other use for OpenCV — the
decoder hands us raw BGR arrays and the detector does its own resizing on the
GPU — and Pillow is already in the dependency tree, ships a scalable font
(``ImageFont.load_default(size=...)``, so no font file to install and nothing
fetched at runtime), and draws translucent fills natively.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RGB = tuple[int, int, int]

# The product's severity palette, RGB (v1 stored the same colours BGR for cv2).
SEVERITY_COLORS: dict[str, RGB] = {
    "high": (208, 59, 59),      # critical red   #d03b3b
    "medium": (250, 178, 25),   # warning amber  #fab219
    "low": (12, 163, 12),       # good green     #0ca30c
}
MUTED: RGB = (140, 140, 140)
BANNER_BACKGROUND: RGB = (8, 10, 12)
TEXT: RGB = (255, 255, 255)


def _font(size: int) -> ImageFont.ImageFont:
    """Pillow's bundled scalable default font — local, no file to ship.

    Deliberately **not** cached. A FreeType face is mutable state inside
    Pillow's C extension and snapshots are drawn from more than one thread
    (the vision loop for zone events, a worker thread for policy events);
    handing them a shared face to save a few microseconds is the wrong trade in
    the one code path whose output is the product's evidence. :func:`_renders`
    memoises the *answer* instead, so the extra faces are built once per
    (size, character) and never on the drawing path.
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow < 10.1
        return ImageFont.load_default()


# --------------------------------------------------------------------------
# text the font can actually draw
# --------------------------------------------------------------------------

#: Characters Pillow's bundled Aileron has no glyph for, and what to draw in
#: their place. Anything not listed and not drawable becomes '?'.
#:
#: **This is not a style preference, it is a rendering bug fix.** Aileron
#: covers Latin-1 and little else: it has `·`, `°`, `±` and curly quotes, and it
#: has **no em dash, en dash, arrow, multiplication sign, ellipsis or geometric
#: shape**. Every one of those came out as a `.notdef` box burned into the
#: evidence. Found on 2026-08-25 in a real snapshot, where the headline
#: `Idle or conversing manpower — Gangway by the sorting bench` rendered the
#: dash as tofu in both the banner and the box chip.
#:
#: Fixing our own strings would not have been enough. The headline's second half
#: is **the model's own sentence**, and a language model writes em dashes and
#: ellipses by habit — so the substitution has to happen at the point of
#: drawing, over text nobody in this repository wrote.
_TRANSLITERATIONS: dict[str, str] = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",                       # dashes and minus
    "…": "...",                                    # ellipsis
    "→": "->", "←": "<-", "⇒": "=>",     # arrows
    "×": "x", "÷": "/",                       # times, divide
    "≤": "<=", "≥": ">=", "≠": "!=",     # comparisons
    "•": "·", "●": "·", "○": "o",   # bullets and circles
    "▲": "^", "▼": "v", "■": "#", "□": "#",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    " ": " ", " ": " ", " ": " ",        # non-breaking spaces
    "′": "'", "″": '"',
}

#: A permanently unassigned code point: whatever a font draws for this is its
#: `.notdef` glyph, which is what every missing character draws too.
_NOTDEF_PROBE = "￾"


@lru_cache(maxsize=4096)
def _renders(size: int, char: str) -> bool:
    """Can the font at this size draw this character as anything but tofu?"""
    font = _font(size)
    try:
        return bytes(font.getmask(char)) != bytes(font.getmask(_NOTDEF_PROBE))
    except Exception:  # noqa: BLE001 - an unaskable font is not a reason to draw nothing
        return True


def drawable(text: str, font: Any) -> str:
    """``text`` with every character this font cannot draw replaced.

    Known typography is transliterated (see :data:`_TRANSLITERATIONS`); anything
    else the font lacks becomes ``?``, which is at least honestly a placeholder
    rather than a rectangle a reviewer reads as a corrupt file.
    """
    size = int(getattr(font, "size", 0) or 0)
    out: list[str] = []
    for char in text:
        replacement = _TRANSLITERATIONS.get(char, char)
        if char.isascii() or _renders(size, replacement[0]):
            out.append(replacement)
        else:
            out.append("?")
    return "".join(out)


def to_image(frame: np.ndarray | Image.Image) -> Image.Image:
    """Decoder frames are BGR uint8; Pillow wants RGB."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR frame, got shape {array.shape}")
    return Image.fromarray(array[:, :, ::-1].copy(), mode="RGB")


def hex_to_rgb(value: str, default: RGB = (34, 211, 238)) -> RGB:
    text = (value or "").lstrip("#")
    if len(text) != 6:
        return default
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:
        return default


def _blend(image: Image.Image, layer: Image.Image, at: tuple[int, int] = (0, 0)) -> None:
    """Alpha-composite an RGBA layer onto an RGB image, in place.

    Pillow cannot paste RGBA onto RGB with per-pixel alpha, so both translucent
    effects here (the zone fill and the banner strip) go through this.
    """
    left, top = at
    region = (left, top, left + layer.size[0], top + layer.size[1])
    merged = Image.alpha_composite(image.crop(region).convert("RGBA"), layer)
    image.paste(merged.convert("RGB"), at)


def draw_zone_overlay(
    image: Image.Image, polygon_norm: Sequence[Sequence[float]], color: RGB, fill_alpha: float = 0.18
) -> None:
    """Shade a normalized (0-1) polygon onto the snapshot, in place."""
    width, height = image.size
    points = [(x * width, y * height) for x, y in polygon_norm]
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).polygon(points, fill=(*color, int(255 * fill_alpha)))
    _blend(image, overlay)
    ImageDraw.Draw(image).line([*points, points[0]], fill=color, width=2)


def draw_box(image: Image.Image, xyxy: Sequence[float], label: str, color: RGB, width: int = 2) -> None:
    """A labelled box with the label chip pinned inside the frame."""
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if not label:
        return
    font = _font(max(12, image.size[0] // 60))
    label = drawable(label, font)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = right - left, bottom - top
    pad = 3
    chip_y = y1 - text_h - 2 * pad
    if chip_y < 0:                       # box at the top edge: put the chip inside
        chip_y = min(y1, image.size[1] - text_h - 2 * pad)
    draw.rectangle([x1, chip_y, x1 + text_w + 2 * pad, chip_y + text_h + 2 * pad], fill=color)
    draw.text((x1 + pad - left, chip_y + pad - top), label, fill=TEXT, font=font)


# A string spanning the font's full ink extent — capital, ascender, descender.
# `banner_height` measures this rather than the real headline so that its answer
# is an upper bound for *any* headline, which is what a caller cropping the
# strip back off needs.
_BANNER_EXTENT = "AXyjgq|"


def _banner_metrics(width: int) -> tuple[Any, int, int]:
    """Font, padding and strip height for a banner on an image this wide.

    One function so that :func:`draw_banner` and :func:`banner_height` cannot
    drift apart — the second exists to undo the first (see
    ``argus/index/backfill.py``) and a formula copied into two places would
    silently start cropping the wrong number of rows.
    """
    font = _font(max(14, width // 44))
    pad = max(8, width // 100)
    _left, top, _right, bottom = ImageDraw.Draw(
        Image.new("RGB", (1, 1))
    ).textbbox((0, 0), _BANNER_EXTENT, font=font)
    return font, pad, (bottom - top) + 2 * pad


def banner_height(width: int) -> int:
    """Rows :func:`draw_banner` occupies at the top of a ``width``-wide image.

    An upper bound over headlines, plus the accent rule and a pixel of slack.
    """
    return _banner_metrics(width)[2] + 5


def draw_banner(image: Image.Image, text: str, color: RGB = SEVERITY_COLORS["high"]) -> None:
    """Burn a headline strip across the top: the one line a reviewer reads first."""
    width, _height = image.size
    font, pad, _bound = _banner_metrics(width)
    # Sanitised before it is measured, so the strip is sized around what will
    # actually be drawn. See `drawable`.
    text = drawable(text, font)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    strip_height = (bottom - top) + 2 * pad

    _blend(image, Image.new("RGBA", (width, strip_height), (*BANNER_BACKGROUND, 184)))
    draw.rectangle([0, strip_height - 3, width, strip_height], fill=color)
    draw.text((pad - left, pad - top), text, fill=TEXT, font=font)


def annotate_zone_event(
    frame: np.ndarray | Image.Image,
    *,
    polygon_norm: Sequence[Sequence[float]],
    headline: str,
    color: RGB = SEVERITY_COLORS["high"],
    zone_color: RGB | None = None,
) -> Image.Image:
    """Compose the over-limit snapshot. Returns a new image; ``frame`` is untouched.

    Per-person boxes are deliberately **not** drawn into these pixels. They used
    to be — burned in permanently, on top of each other, with no way to tell one
    crowded detection from another once saved. The per-track boxes (with their
    ids and confidences) are already carried separately in the event's
    ``meta.boxes``, and the console now draws them as an interactive overlay on
    top of this same, unmodified frame: hover, select, toggle visibility, flag
    one as wrong without needing a second image. Baking them into the JPEG would
    make all of that impossible after the fact. The zone shading and the
    headline stay, because they are the one fact every reviewer needs before
    looking at anything else: which area, and by how much.
    """
    image = to_image(frame)
    draw_zone_overlay(image, polygon_norm, zone_color or color)
    draw_banner(image, headline, color)
    return image


def save_snapshot(image: Image.Image | np.ndarray, path: str | Path, quality: int = 85) -> Path:
    """Write a snapshot atomically, so a reader never sees a half-written JPEG."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    picture = to_image(image) if not isinstance(image, Image.Image) else image
    temporary = destination.with_name(destination.name + ".tmp")
    picture.save(temporary, format="JPEG", quality=int(quality), optimize=True)
    temporary.replace(destination)
    return destination


def track_label(tracker_id: Any, confidence: float | None = None) -> str:
    """'Worker 7' / 'Worker 7 · 88%' — v1's wording, kept for continuity."""
    name = f"Worker {int(tracker_id)}" if tracker_id is not None else "Worker"
    if confidence is None:
        return name
    return f"{name} · {round(float(confidence) * 100)}%"


__all__ = [
    "MUTED",
    "SEVERITY_COLORS",
    "annotate_zone_event",
    "banner_height",
    "draw_banner",
    "draw_box",
    "draw_zone_overlay",
    "hex_to_rgb",
    "save_snapshot",
    "to_image",
    "track_label",
]
