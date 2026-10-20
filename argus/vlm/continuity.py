"""Ongoing conditions: deciding when two triggered verdicts are one problem.

A prompt policy asks about a **state**, and a state persists. Before this
module, the only thing standing between "there is loose laminate on the wrapper
apron" and a row in the events table was ``cooldown_sec``, so a condition that
was true all night wrote one row per cooldown: 140 rows about one apron on this
appliance, 40 of them between dusk and dawn. The cooldown was doing the only job
it can do — rate-limiting — when the question being asked was *identity*.

This is the identity test, kept in its own module and written as pure functions
over plain values so it can be argued with in a test rather than in a live run.
Nothing here touches the database, the clock or the model.

**Detector-origin events are not in scope, but they do have the same problem.**
This file used to say ``argus/vision/debounce.py`` already held a count in a
polygon steady across frames. It did not — measured 2026-09-19, one headcount
policy had written 112 151 rows the same way one apron wrote 140 — and the
answer for counts is a different one, because a count has a *magnitude* and a
model's grounding box does not: see ``argus/vision/zone_state.py``, which
smooths and applies hysteresis where this module compares boxes. A detector
event still never reaches this code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Box = Sequence[float]

#: Why a triggered verdict did not continue the condition on file. These are the
#: strings the scheduler logs, so they are written to be read at 3 a.m.
NO_OPEN_CONDITION = "no open condition"
STALE = "the open condition went unobserved for too long"
MOVED = "the grounding box disagrees with the open condition"


def iou(a: Box | None, b: Box | None) -> float | None:
    """Intersection over union of two normalized ``[x1, y1, x2, y2]`` boxes.

    ``None`` — not 0.0 — when either box is missing, because "the model did not
    locate its answer" and "the model located it somewhere else" are opposite
    facts and a caller that cannot tell them apart will treat the first as
    disagreement. Every VLM verdict may legally carry ``box_2d: null``.
    """
    if not a or not b or len(a) < 4 or len(b) < 4:
        return None
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Match:
    """The answer, and the sentence explaining it."""

    ok: bool
    reason: str
    #: Set when both observations located themselves and the boxes were compared.
    iou: float | None = None

    def __bool__(self) -> bool:  # so `if match:` reads the way it looks
        return self.ok


def box_of(event: Mapping[str, Any]) -> list[float] | None:
    """The grounding box recorded on an event row, if it has one.

    ``meta`` arrives either decoded (the API) or as a JSON string (a raw row),
    and this is called from both, so it accepts a mapping and nothing else —
    callers hand it ``decoded_meta(event)``.
    """
    meta = event.get("meta")
    if not isinstance(meta, Mapping):
        return None
    box = meta.get("box_2d")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        return [float(v) for v in box[:4]]
    except (TypeError, ValueError):
        return None


def continues(
    *,
    last_seen_ts: float,
    ts: float,
    stale_after_sec: float,
    open_box: Box | None = None,
    new_box: Box | None = None,
    zone_scoped: bool = False,
    min_iou: float = 0.1,
) -> Match:
    """Does a triggered verdict at ``ts`` continue the condition on file?

    The caller has already established the cheap half of identity — same policy,
    same camera, same zone, and the verdict is triggered — by looking the
    condition up under its own key. What is left is the two questions a key
    cannot answer.

    **Is it still the same episode?** A condition nobody has observed since
    ``stale_after_sec`` ago is over, whatever the next verdict says. Without this
    a camera that comes back after a week would extend a condition from last
    Tuesday and file a seven-day span that never happened.

    **Is it in the same place?** Only for a policy with no zone. A zone-scoped
    policy is already anchored: the model is shown the zone crop and the question
    is about the zone, so laminate that moves along the apron is the same
    problem seen again. Measured before it was assumed — over this appliance's
    own 139 consecutive laminate observations the median IoU between one
    sighting and the next is **0.000**, with box centres a median 0.11 of the
    frame apart against a box 0.077 wide, so comparing boxes inside a zone would
    have split one ongoing mess into roughly a hundred. An unscoped policy has no
    polygon to stand in, so its box has to, and ``min_iou`` is the bar.

    A missing box on either side is not disagreement (see :func:`iou`): it is
    the model declining to point, and it leaves the zone-or-key identity to
    carry the match on its own.
    """
    if ts - last_seen_ts > stale_after_sec:
        return Match(False, STALE)
    if zone_scoped:
        return Match(True, "same zone, still fresh")
    overlap = iou(open_box, new_box)
    if overlap is None:
        return Match(True, "no box to disagree with")
    if overlap < min_iou:
        return Match(False, MOVED, iou=overlap)
    return Match(True, "the boxes agree", iou=overlap)


__all__ = ["Box", "Match", "MOVED", "NO_OPEN_CONDITION", "STALE", "box_of", "continues", "iou"]
