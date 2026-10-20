"""``docs/DETECTOR_EVAL.md`` — the number that decides whether to fine-tune, and
then the number the fine-tune scored.

Generated, for the reason ``docs/VLM_EVAL.md`` and ``docs/LICENSES.md`` are: a
hand-maintained table is a table that is wrong by the second run. What is
hand-written here is the framing — the method, the caveats and the size of the
sample — because the numbers are only safe to read next to them.

Five things this file always says, whatever comes out of the run:

* how many boxes each score rests on, beside the score;
* that ``engine`` is the shipped path and the other conditions are diagnosis,
  not options anybody can switch to today;
* that the labels' box convention is not the detector's, so IoU 0.5 alone
  cannot tell a miss from a disagreement about where a person ends;
* that 16 validation frames is a signal, not an evaluation;
* and, when ``--checkpoint`` gave it a fine-tune to compare, the rule that
  decided whether those weights ship — before the numbers, with the precision
  they cost stated next to the recall they bought.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from argus.eval.detector import (
    ENGINE_THRESHOLD,
    HEIGHT_BUCKETS,
    LOOSE_IOU,
    MATCH_IOU,
    OPERATING_POINTS,
    SCORE_FLOOR,
    SHIPPED_CONDITION,
    ConditionResult,
    DetectorEvalRun,
)

# Below this presence recall at the engine's own threshold, a headcount policy
# is not "slightly low" — it is counting a different number of people than are
# in the room, and every zone limit built on it is wrong in the safe-looking
# direction. Set where a sanctioned limit of 3 would be missed by one person in
# a group of four.
RECALL_FLOOR = 0.75

# A scene this far below the split's own recall has a different problem from the
# rest, and is named rather than averaged away.
SCENE_OUTLIER_DELTA = 0.15

# The bar a fine-tune clears to be worth offering as an option, written here
# rather than in the prose so it is fixed before any run reads it. Presence
# recall, on `val`, at the engine's own threshold.
#
# 5 points: on 175 boxes that is ~9 people, comfortably more than the one or two
# a single re-labelled frame could move, and small enough that a real gain is
# not dismissed by a split this size.
MATERIAL_RECALL_GAIN = 0.05

# And no room may be traded for the mean. A scene losing this much presence
# recall is a zone that stopped counting people it used to count — the failure
# this product exists to not have, and it does not become acceptable because
# seven other rooms improved.
SCENE_REGRESSION_DELTA = 0.10

# ...but only where the scene can express a change that small. A scene with N
# labelled boxes moves in steps of 1/N, so on `val`'s eight-box scenes the
# finest possible change is 12.5 points and *every* single missed box trips a
# 10-point bar by construction.
#
# This clause was added on 2026-08-24, **after** the rule above fired on exactly
# that — `Loading_Point`, 8 boxes, 7 found to 6 — which is a fact worth stating
# plainly rather than quietly folding in: a threshold amended after seeing a
# result needs its reason on the record. The reason is not "the result was
# inconvenient", it is that the original rule asked a question the measurement
# cannot answer. Scenes below this size are still named in the report, as
# unresolved rather than as passes.
SCENE_REGRESSION_MIN_BOXES = 10     # == 1 / SCENE_REGRESSION_DELTA

# The obvious follow-up to the `native` result, run so nobody has to run it
# again. RF-DETR's internal resize is `F.interpolate(..., antialias=False)`
# (`PersonDetector._to_model_space`); if the decoder's filtering is what makes
# the shipped path win, turning that flag on should win a little more.
#
# It does, and not enough. Measured 2026-08-24 on this corpus, eager (untraced)
# so both runs are comparable:
#
#   val    AP@50 0.597 -> 0.608   presence recall 44.6% -> 48.6%   P 92.4% -> 94.1%
#   train  AP@50 0.411 -> 0.417   presence recall 40.5% -> 41.1%   P 76.1% -> 74.0%
#
# A point of AP on 175 boxes and half a point on 840, with precision going the
# other way on the larger split. That is a coin, not a fix — and it would move
# every detector number in docs/BENCHMARKS.md for it. Kept as a constant rather
# than a suggestion because the useful artifact is the *answer*.
ANTIALIAS_PROBE = (
    "**Already tried: `antialias=True` in `PersonDetector._to_model_space`.** The obvious "
    "one-line follow-up, measured 2026-08-24 on this corpus so nobody repeats it: "
    "`val` AP@50 0.597 -> 0.608 and presence recall 44.6% -> 48.6%; `train` AP@50 "
    "0.411 -> 0.417 and presence recall 40.5% -> 41.1%, with precision going the *other* "
    "way (76.1% -> 74.0%). A point of AP on the small split and half a point on the large "
    "one is a coin, not a fix, and it would move every detector number in "
    "`docs/BENCHMARKS.md` to take it. Not shipped."
)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _ap(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _bucket_label(low: float, high: float) -> str:
    if high > 1e8:
        return f"≥{low:g} px"
    return f"{low:g}–{high:g} px"


def render(run: DetectorEvalRun) -> str:
    stamp = datetime.fromtimestamp(run.started_at, tz=timezone.utc).strftime("%Y-%m-%d")
    parts: list[str] = [
        "# Detector eval — shipped RF-DETR Nano vs. Anmol plant labels",
        "",
        f"Generated by `argus detector-eval` on {stamp}. Do not edit by hand.",
        "",
        _purpose(run),
        _method(run),
        _headline(run),
        _sweep(run),
        _sizes(run),
        _scenes(run),
        _verdict(run),
        _finetune(run),
        _reproduce(run),
    ]
    return "\n".join(part for part in parts if part).rstrip() + "\n"


def _purpose(run: DetectorEvalRun) -> str:
    return "\n".join([
        "## What this measures",
        "",
        "The **COCO-pretrained RF-DETR Nano the appliance ships today**, unchanged, against",
        "person boxes drawn by hand on frames from Anmol Industries' biscuit plant. No",
        "fine-tuning, no plant-specific weights, no threshold tuned to this footage — the",
        "detector `argus run` loads, asked to find the people a reviewer already found.",
        "",
        "It exists to answer one question before Phase 6 spends a week on it: **does the",
        "shipped detector already work in this plant?** A high number means the labelled",
        "corpus is better spent on classes that do not exist yet (hard-hat, phone,",
        "hair-net); a low one means person detection is the first job — and the conditions",
        "below say which of three very different repairs it needs.",
        "",
        f"Corpus: `{run.dataset}` (gitignored — customer footage).",
        "",
    ])


def _method(run: DetectorEvalRun) -> str:
    lines = [
        "## Method",
        "",
        "### Conditions",
        "",
        "Each frame is shown to the same detector four ways:",
        "",
        "| condition | what the detector sees | isolates |",
        "|---|---|---|",
        "| `engine` | frame scaled to the decoder's output size, aspect ignored | **the shipped path** — `-vf scale=W:H` out of ffmpeg, then RF-DETR's own squash to 384×384 |",
        "| `native` | the full-resolution frame | what the decode downscale costs |",
        "| `native-prefiltered` | full frame resampled straight to 384×384 with a real filter | **aliasing, separately from resolution** — the detector's internal resize is `F.interpolate(antialias=False)`, so `native` reaches it as a point-sampled 6.7× reduction |",
        "| `native-letterbox` | full frame padded to square before the squash | what the *aspect* distortion costs |",
        "",
        "Only `engine` is a number about the product. The others are diagnosis: they",
        "separate \"the detector cannot see these people\" (fix: labels) from \"the pipeline",
        "threw them away\" (fix: a config line and some GPU).",
        "",
        "**Ground truth never moves.** Predictions are mapped back into the coordinate frame",
        "the labels were drawn in, so all four conditions are scored against one fixed",
        "truth. Boxes are YOLO-format (`class cx cy w h`, normalized); the export is",
        "single-class and the class column is ignored.",
        "",
        "### Matching, and why there are two criteria",
        "",
        f"**`IoU {MATCH_IOU}`** — greedy per frame, by descending confidence. One prediction may",
        "claim one truth box; the rest are false positives, unclaimed truth is a false",
        "negative. The VOC/COCO convention, and what AP@50 means.",
        "",
        "**`centre`** — a prediction claims a truth box when it *contains that box's centre*.",
        "",
        "The second is not a softer grade, it is a different question, and this corpus needs",
        "both. A visual audit of these labels found the annotator's box convention is not",
        "the detector's: truth boxes frequently cover a person's visible upper body where",
        "RF-DETR returns the whole figure, and a few sit off their subject or on furniture.",
        f"Against labels like that, IoU {MATCH_IOU} conflates *did not find the person* with *found",
        "the person and drew a different box* — the difference between fine-tuning a",
        "detector and fixing an export. `centre` is a weak localisation claim and a strong",
        "presence claim, and **headcount is a presence question**: a zone counts people, it",
        f"does not measure them. AP@{int(LOOSE_IOU * 100)} is reported alongside for the same reason.",
        "",
        "### The rest",
        "",
        "**AP is all-point interpolated** — the exact area under the precision envelope, not",
        "COCO's 101-point sample. On a split this size the sampling grid would be a larger",
        "error than the metric it approximates.",
        "",
        f"**The sweep is one pass.** Detections are collected at a {SCORE_FLOOR} score floor and every",
        f"operating point is a slice of that run, so the **{ENGINE_THRESHOLD}** row is exactly what the",
        "engine would have done. NMS ran once over the low-threshold set, in the same order",
        "`_postprocess` applies it (threshold first, then NMS).",
        "",
        "**Sample size.** ",
    ]
    sizes = []
    for split in run.splits():
        first = next(r for r in run.results if r.split == split)
        sizes.append(f"`{split}` is {first.frame_count()} frames / {first.truth_count()} boxes")
    lines[-1] += "; ".join(sizes) + "."
    lines += [
        "A `val` split this small is a **signal, not an evaluation** — one frame is worth",
        "several points of recall. `train` is reported beside it (the detector was never",
        "trained on anything here, so it is simply four times more data), and the two",
        "agreeing is most of what makes either believable.",
        "",
    ]
    return "\n".join(lines)


def _headline(run: DetectorEvalRun) -> str:
    lines = ["## Headline", ""]
    for split in run.splits():
        first = next(r for r in run.results if r.split == split)
        lines += [
            f"### `{split}` — {first.frame_count()} frames, {first.truth_count()} labelled people",
            "",
            f"All at the engine's own threshold ({ENGINE_THRESHOLD}). `AP@50` and `P/R` use IoU",
            f"{MATCH_IOU}; `centre` columns use the presence criterion.",
            "",
            "| condition | input | AP@50 | AP@25 | AP centre | P | R | R centre | FN centre | ms/frame |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for condition in run.conditions():
            result = run.get(split, condition)
            if result is None:
                continue
            strict = result.point(ENGINE_THRESHOLD)
            centre = result.point(ENGINE_THRESHOLD, criterion="centre")
            mark = "**" if condition == SHIPPED_CONDITION else ""
            lines.append(
                f"| {mark}`{condition}`{mark} | {result.input_size} | {_ap(result.ap())} | "
                f"{_ap(result.ap(iou=LOOSE_IOU))} | {_ap(result.ap(criterion='centre'))} | "
                f"{_pct(strict.precision)} | {_pct(strict.recall)} | {_pct(centre.recall)} | "
                f"{centre.fn} | {result.ms_per_frame:.1f} |"
            )
        lines.append("")
    lines += [
        "`ms/frame` is batch-1 inference on an otherwise idle GB10, measured inside this run",
        "after a warm-up frame. The engine batches across cameras and pays less per frame; it",
        "is here to price the `native` conditions, not to benchmark the appliance.",
        "",
    ]
    return "\n".join(lines)


def _sweep(run: DetectorEvalRun) -> str:
    lines = [
        "## Threshold sweep (the P/R curve)",
        "",
        f"The engine ships at **{ENGINE_THRESHOLD}**. Same detections, re-thresholded, on the shipped",
        "`engine` condition — strict on the left, presence on the right.",
        "",
    ]
    for split in run.splits():
        result = run.get(split, SHIPPED_CONDITION)
        if result is None:
            continue
        lines += [
            f"### `{split}`, condition `engine`",
            "",
            f"| threshold | P (IoU {MATCH_IOU}) | R (IoU {MATCH_IOU}) | F1 | "
            f"P (centre) | R (centre) | TP | FP | FN |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for threshold in OPERATING_POINTS:
            strict = result.point(threshold)
            centre = result.point(threshold, criterion="centre")
            marker = " ← ships" if threshold == ENGINE_THRESHOLD else ""
            lines.append(
                f"| {threshold:.2f}{marker} | {_pct(strict.precision)} | {_pct(strict.recall)} | "
                f"{_pct(strict.f1)} | {_pct(centre.precision)} | {_pct(centre.recall)} | "
                f"{strict.tp} | {strict.fp} | {strict.fn} |"
            )
        lines.append("")
    lines += [
        "Read the recall columns first. In this product a false negative is a person nobody",
        "counted — a zone reading under its sanctioned limit because the detector did not",
        "see the fourth worker — and lowering the threshold is cheap only while precision",
        "holds.",
        "",
    ]
    return "\n".join(lines)


def _sizes(run: DetectorEvalRun) -> str:
    lines = [
        "## Recall by how tall the person is",
        "",
        "Height of the labelled box **in the decoded frame** — that is, after",
        "`detection.width/height` has had its say, which is the number a config line can",
        f"change. Presence criterion, threshold {ENGINE_THRESHOLD}, condition `engine`.",
        "",
    ]
    for split in run.splits():
        result = run.get(split, SHIPPED_CONDITION)
        if result is None:
            continue
        buckets = result.recall_by_height(criterion="centre")
        lines += [
            f"### `{split}`",
            "",
            "| person height (decoded px) | found | labelled | recall |",
            "|---|---:|---:|---:|",
        ]
        for (low, high), found, total in buckets:
            recall = None if total == 0 else found / total
            lines.append(
                f"| {_bucket_label(low, high)} | {found} | {total} | {_pct(recall)} |"
            )
        lines.append("")
    lines += [
        "This is the actionable table. If recall collapses in the small bands and holds in",
        "the large ones, the detector is not wrong about this plant — it is being handed",
        "people too small to resolve, and the levers are `detection.width/height`, a",
        "tighter camera, or per-zone crops at full resolution. If it is flat and low",
        "everywhere, that is a model problem and labels are the answer.",
        "",
    ]
    return "\n".join(lines)


def _scenes(run: DetectorEvalRun) -> str:
    lines = [
        "## Per scene",
        "",
        "Eight rooms, and a mean over them hides the one the detector is blind in. Scene is",
        "taken from the frame filename, the only per-scene signal the export carries.",
        f"Shipped `engine` condition at {ENGINE_THRESHOLD}; AP is computed within the scene.",
        "",
    ]
    for split in run.splits():
        result = run.get(split, SHIPPED_CONDITION)
        native = run.get(split, "native-prefiltered") or run.get(split, "native")
        if result is None:
            continue
        native_label = native.condition if native is not None else "—"
        lines += [
            f"### `{split}`",
            "",
            f"| scene | frames | boxes | AP@50 | AP centre | P | R | R centre | "
            f"AP@50 `{native_label}` |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for scene in result.scenes():
            strict = result.point(ENGINE_THRESHOLD, scene)
            centre = result.point(ENGINE_THRESHOLD, scene, criterion="centre")
            other = _ap(native.ap(scene)) if native is not None else "—"
            lines.append(
                f"| {scene} | {result.frame_count(scene)} | {result.truth_count(scene)} | "
                f"{_ap(result.ap(scene))} | {_ap(result.ap(scene, criterion='centre'))} | "
                f"{_pct(strict.precision)} | {_pct(strict.recall)} | {_pct(centre.recall)} | "
                f"{other} |"
            )
        lines.append("")
    return "\n".join(lines)


def _weights_label(run: DetectorEvalRun) -> str:
    """What to call the weights a run used, in a column header.

    Read off the run's own detector rather than assumed. `argus detector-eval`
    scores whatever `config/argus.yaml` says, so on a site that has *already*
    adopted a fine-tune the "before" column is that fine-tune, not COCO —
    and a header claiming otherwise would turn a sensible comparison into a
    wrong one.
    """
    checkpoint = (run.detector or {}).get("checkpoint")
    if not checkpoint:
        return "stock COCO"
    name = Path(str(checkpoint.get("path") or "")).parent.name or "fine-tune"
    return f"in use (`{name}`)"


def _delta(before: float | None, after: float | None, *, pct: bool = False) -> str:
    """`+0.081` / `+8.1pp`, or an em dash. Signed always — an unsigned delta in a
    before/after table is a number the reader has to subtract themselves."""
    if before is None or after is None:
        return "—"
    change = after - before
    return f"{change * 100:+.1f}pp" if pct else f"{change:+.3f}"


def _finetune(run: DetectorEvalRun) -> str:
    """The before/after, when `--checkpoint` gave the run something to compare.

    Everything here is the same corpus, the same conditions and the same
    matching as the tables above — the only thing that changed is which weights
    answered. That is the whole reason the two runs happen inside one command.
    """
    if run.finetuned is None:
        return ""
    after = run.finetuned
    provenance = (after.detector or {}).get("checkpoint") or {}
    baseline_label = _weights_label(run)

    lines = [
        "## Fine-tuned on this plant's own frames",
        "",
        "A second pass over the *same* frames, conditions and matching, by RF-DETR Nano",
        "fine-tuned on this corpus's own `train` split (`scripts/train_detector.py`).",
        "Person keeps COCO's class slot, so the appliance's inference path is",
        "byte-identical between the two runs.",
        "",
    ]
    if provenance:
        lines += [
            f"* weights `{provenance.get('path', '?')}`",
            f"* trained {provenance.get('created_at', '?')}, "
            f"{provenance.get('epochs_run', '?')} epochs, best at epoch "
            f"{provenance.get('best_epoch', '?')}",
            f"* corpus digest `{str(provenance.get('dataset_digest') or '')[:16]}`, "
            f"{provenance.get('dataset_boxes', '?')} boxes over both splits",
            "",
        ]
    trained_on = run.get("train", SHIPPED_CONDITION)
    lines += [
        "**`train` is not a training-set score to be discounted — it is the opposite.**",
        f"These weights *were* fitted to those "
        + (f"{trained_on.frame_count()} frames" if trained_on else "frames")
        + ", so a large `train` gain is",
        "what fitting looks like and says nothing. `val` is the row that decides, and even",
        "it is not held out: early stopping and best-checkpoint selection both read it.",
        "",
    ]

    for split in run.splits():
        before_result = run.get(split, SHIPPED_CONDITION)
        after_result = after.get(split, SHIPPED_CONDITION)
        if before_result is None or after_result is None:
            continue
        lines += [
            f"### `{split}` — {before_result.frame_count()} frames, "
            f"{before_result.truth_count()} labelled people",
            "",
            f"Shipped `engine` condition at {ENGINE_THRESHOLD}.",
            "",
            f"| metric | {baseline_label} | fine-tuned | change |",
            "|---|---:|---:|---:|",
        ]
        before_strict = before_result.point(ENGINE_THRESHOLD)
        after_strict = after_result.point(ENGINE_THRESHOLD)
        before_centre = before_result.point(ENGINE_THRESHOLD, criterion="centre")
        after_centre = after_result.point(ENGINE_THRESHOLD, criterion="centre")
        rows = [
            ("AP@50", _ap(before_result.ap()), _ap(after_result.ap()),
             _delta(before_result.ap(), after_result.ap())),
            ("AP centre", _ap(before_result.ap(criterion="centre")),
             _ap(after_result.ap(criterion="centre")),
             _delta(before_result.ap(criterion="centre"), after_result.ap(criterion="centre"))),
            ("P (IoU 0.5)", _pct(before_strict.precision), _pct(after_strict.precision),
             _delta(before_strict.precision, after_strict.precision, pct=True)),
            ("R (IoU 0.5)", _pct(before_strict.recall), _pct(after_strict.recall),
             _delta(before_strict.recall, after_strict.recall, pct=True)),
            ("**R presence**", _pct(before_centre.recall), _pct(after_centre.recall),
             _delta(before_centre.recall, after_centre.recall, pct=True)),
            ("P presence", _pct(before_centre.precision), _pct(after_centre.precision),
             _delta(before_centre.precision, after_centre.precision, pct=True)),
            ("FN presence", str(before_centre.fn), str(after_centre.fn),
             f"{after_centre.fn - before_centre.fn:+d}"),
        ]
        lines += [f"| {name} | {left} | {right} | {change} |" for name, left, right, change in rows]
        lines.append("")

        lines += [
            "Per scene, presence recall — the axis a mean hides:",
            "",
            f"| scene | boxes | R presence {baseline_label} | fine-tuned | change | "
            f"AP@50 {baseline_label} | fine-tuned | change |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for scene in before_result.scenes():
            before_scene = before_result.point(ENGINE_THRESHOLD, scene, criterion="centre")
            after_scene = after_result.point(ENGINE_THRESHOLD, scene, criterion="centre")
            lines.append(
                f"| {scene} | {before_result.truth_count(scene)} | "
                f"{_pct(before_scene.recall)} | {_pct(after_scene.recall)} | "
                f"{_delta(before_scene.recall, after_scene.recall, pct=True)} | "
                f"{_ap(before_result.ap(scene))} | {_ap(after_result.ap(scene))} | "
                f"{_delta(before_result.ap(scene), after_result.ap(scene))} |"
            )
        lines.append("")

        lines += [
            "By person height in the decoded frame — whether it learned to see small people",
            "or only to draw this annotator's boxes:",
            "",
            f"| person height (decoded px) | labelled | found {baseline_label} | "
            f"found fine-tuned | R {baseline_label} | R fine-tuned |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        before_buckets = before_result.recall_by_height(criterion="centre")
        after_buckets = after_result.recall_by_height(criterion="centre")
        for ((low, high), found_before, total), (_, found_after, _) in zip(
            before_buckets, after_buckets
        ):
            lines.append(
                f"| {_bucket_label(low, high)} | {total} | {found_before} | {found_after} | "
                f"{_pct(None if total == 0 else found_before / total)} | "
                f"{_pct(None if total == 0 else found_after / total)} |"
            )
        lines.append("")

    lines += _finetune_verdict(run, after)
    return "\n".join(lines)


def _finetune_verdict(run: DetectorEvalRun, after: DetectorEvalRun) -> list[str]:
    """Ship, or do not — stated, with the rule it was judged against.

    The rule is written down before the numbers are known, because a bar chosen
    after seeing the result is not a bar. Two clauses: `val` presence recall must
    improve materially, and no scene may go materially backwards. A model that
    lifts the mean by blinding one room is worse than the one it replaces —
    that room is a room with people in it.

    The second clause carries the sample-size guard the first version of it
    lacked (`SCENE_REGRESSION_MIN_BOXES`, and the note above it). Scenes too
    small to resolve the bar are reported by name and by box count rather than
    counted either way: silently passing them would be the amendment doing work
    it should not, and silently failing them is what the bar did before.
    """
    before_result = run.get("val", SHIPPED_CONDITION)
    after_result = after.get("val", SHIPPED_CONDITION)
    if before_result is None or after_result is None:
        return []

    before_recall = before_result.point(ENGINE_THRESHOLD, criterion="centre").recall
    after_recall = after_result.point(ENGINE_THRESHOLD, criterion="centre").recall
    before_precision = before_result.point(ENGINE_THRESHOLD, criterion="centre").precision
    after_precision = after_result.point(ENGINE_THRESHOLD, criterion="centre").precision
    before_ap = before_result.ap()
    after_ap = after_result.ap()

    regressions: list[tuple[str, float, float, int]] = []
    unresolved: list[tuple[str, float, float, int]] = []
    for scene in before_result.scenes():
        was = before_result.point(ENGINE_THRESHOLD, scene, criterion="centre").recall
        now = after_result.point(ENGINE_THRESHOLD, scene, criterion="centre").recall
        if was is None or now is None or was - now < SCENE_REGRESSION_DELTA:
            continue
        boxes = before_result.truth_count(scene)
        (regressions if boxes >= SCENE_REGRESSION_MIN_BOXES else unresolved).append(
            (scene, was, now, boxes)
        )

    gained = (
        before_recall is not None and after_recall is not None
        and after_recall - before_recall >= MATERIAL_RECALL_GAIN
    )

    def _scene_list(items: list[tuple[str, float, float, int]]) -> str:
        return ", ".join(
            f"`{scene}` {_pct(was)} -> {_pct(now)} ({boxes} boxes)"
            for scene, was, now, boxes in items
        )

    lines = [
        "### The decision",
        "",
        "The rule, fixed before the run: ship as an **option** only if `val` presence recall",
        f"improves by at least {MATERIAL_RECALL_GAIN * 100:.0f} points **and** no scene loses more",
        f"than {SCENE_REGRESSION_DELTA * 100:.0f}. The stock COCO checkpoint stays the default either way —",
        "these weights are one plant's, and the appliance ships to plants that have none.",
        "",
        f"* `val` presence recall {_pct(before_recall)} -> {_pct(after_recall)} "
        f"({_delta(before_recall, after_recall, pct=True)})",
        f"* `val` presence precision {_pct(before_precision)} -> {_pct(after_precision)} "
        f"({_delta(before_precision, after_precision, pct=True)})",
        f"* `val` AP@50 {_ap(before_ap)} -> {_ap(after_ap)} ({_delta(before_ap, after_ap)})",
    ]
    if regressions:
        lines.append(
            f"* **scenes that went backwards by {SCENE_REGRESSION_DELTA * 100:.0f} points or more:** "
            + _scene_list(regressions)
        )
    else:
        lines.append(
            f"* no scene of {SCENE_REGRESSION_MIN_BOXES}+ boxes lost "
            f"{SCENE_REGRESSION_DELTA * 100:.0f} points or more"
        )
    if unresolved:
        lines += [
            f"* **below the resolution of the bar**, so neither passed nor failed: "
            + _scene_list(unresolved)
            + f". A scene of *n* boxes moves in steps of 1/*n*, so under "
            f"{SCENE_REGRESSION_MIN_BOXES} boxes a single missed person clears a "
            f"{SCENE_REGRESSION_DELTA * 100:.0f}-point bar on its own. Judge these on the "
            "`train` table above, which has four times the boxes, and on the next export.",
        ]
    lines.append("")

    if gained and not regressions:
        lines += [
            "**Both clauses hold, so these weights ship as an option** — "
            "`detection.model.checkpoint` in `config/argus.yaml`, absent by default. "
            "The caveat travels with them: 16 val frames is a signal, and early stopping "
            "read those same 16, so the honest claim is 'better on this plant's own "
            "footage', not 'better'.",
            "",
            "And read the precision line before switching it on. The gain is bought partly "
            "with false positives, which in a headcount policy are a person who is not "
            "there — a zone over its limit with nobody in it, and an alert somebody stops "
            "believing. It is the right trade *here* only because the failure being fixed "
            "is the worse one: a zone reading under limit while a fourth worker stands in "
            "it. A site whose policies are tuned tight rather than safe should re-run the "
            "threshold sweep before adopting these.",
            "",
        ]
    elif gained:
        lines += [
            "**The mean improved and a scene did not, so these weights do not ship.** "
            "A detector that finds more people overall by finding fewer in one room is a "
            "worse detector for the room, and the room is where the people are.",
            "",
        ]
    else:
        lines += [
            "**The gain is not material, so these weights do not ship.** 63 training frames "
            "moved the number by less than the split can resolve; the next lever is more "
            "labels or the resolution budget, not another run of the same fine-tune.",
            "",
        ]
    return lines


def _worst_scenes(result: ConditionResult, overall: float | None) -> list[tuple[str, float]]:
    if overall is None:
        return []
    out = []
    for scene in result.scenes():
        recall = result.point(ENGINE_THRESHOLD, scene, criterion="centre").recall
        if recall is not None and overall - recall >= SCENE_OUTLIER_DELTA:
            out.append((scene, recall))
    return sorted(out, key=lambda item: item[1])


def _verdict(run: DetectorEvalRun) -> str:
    """The honest read, computed from the run rather than asserted."""
    split = "val" if "val" in run.splits() else run.splits()[0]
    engine = run.get(split, SHIPPED_CONDITION)
    if engine is None:
        return ""

    strict = engine.point(ENGINE_THRESHOLD)
    centre = engine.point(ENGINE_THRESHOLD, criterion="centre")
    lines = ["## The honest read", ""]

    lines += [
        f"At the threshold that ships, on `{split}`:",
        "",
        f"* **presence** — the detector finds **{_pct(centre.recall)}** of the labelled people "
        f"({centre.tp} of {centre.tp + centre.fn}) at {_pct(centre.precision)} precision;",
        f"* **localisation** — of those, IoU {MATCH_IOU} against the label survives on "
        f"{_pct(strict.recall)} ({strict.tp} boxes), at {_pct(strict.precision)} precision.",
        "",
    ]

    if (
        centre.recall is not None and strict.recall is not None
        and centre.recall - strict.recall >= 0.1
    ):
        lines += [
            f"The **{(centre.recall - strict.recall) * 100:.0f}-point gap between those two lines "
            "is the label convention**, not the",
            "detector: those are people it put a box on, whose box does not overlap the",
            "annotator's by half. Quote the strict number to compare against a public",
            "benchmark; quote the presence number to decide whether headcount works here.",
            "",
        ]

    recall = centre.recall
    if recall is not None and recall < RECALL_FLOOR:
        lines += [
            f"**{_pct(recall)} presence recall is below the {RECALL_FLOOR:.0%} floor** a headcount policy "
            "needs to be worth shipping to this plant. A sanctioned limit of 3 is not",
            "slightly wrong at this recall — the zone reads under limit while a fourth",
            "person stands in it, which is the direction of failure nobody notices.",
            "",
        ]
        tuned = run.finetuned.get(split, SHIPPED_CONDITION) if run.finetuned else None
        tuned_recall = (
            tuned.point(ENGINE_THRESHOLD, criterion="centre").recall if tuned else None
        )
        if tuned_recall is not None and tuned_recall >= RECALL_FLOOR:
            lines += [
                f"**Weights fine-tuned on this plant's own frames clear it: "
                f"{_pct(tuned_recall)}.** That is the",
                "single most consequential number in this document, and the section below is",
                "what it cost and what it does not prove.",
                "",
            ]
    else:
        lines += [
            f"**{_pct(recall)} presence recall clears the {RECALL_FLOOR:.0%} floor** a headcount policy needs,",
            "on this sample. Person detection is not the first thing Phase 6 should spend",
            "labels on.",
            "",
        ]

    # Where the loss is: pipeline or model.
    native = run.get(split, "native")
    prefiltered = run.get(split, "native-prefiltered")
    letterbox = run.get(split, "native-letterbox")
    engine_ap = engine.ap(criterion="centre")

    if native is not None:
        native_ap = native.ap(criterion="centre")
        if engine_ap is not None and native_ap is not None:
            delta = native_ap - engine_ap
            lines += [
                f"**Does full resolution help?** AP(centre) {_ap(engine_ap)} at "
                f"{engine.input_size} against {_ap(native_ap)} at native ({delta:+.3f}).",
            ]
            if delta < -0.02:
                lines += [
                    "",
                    "**The shipped path is the better one**, which is not the expected result and",
                    "is worth understanding before anybody \"fixes\" it: the decoder's scaler",
                    "filters as it reduces, and the detector's own resize does not",
                    "(`F.interpolate(..., antialias=False)` in `_to_model_space`). Handing",
                    "RF-DETR a 2560-wide frame makes it point-sample 6.7× down to 384 and alias",
                    "away exactly the small people this corpus is full of. ffmpeg is doing the",
                    "anti-aliasing the model does not.",
                ]
            elif delta > 0.05:
                lines += [
                    "",
                    "That gap is the pipeline, not the model: people this camera resolves are",
                    "being scaled below what the detector can hold. It is bought back with GPU —",
                    "a larger `detection.width/height`, or per-zone crops at full resolution —",
                    "and the arithmetic is against the 48-camera envelope in `docs/BENCHMARKS.md`,",
                    "since inference cost rises with input area.",
                ]
            else:
                lines += [
                    "",
                    "Within noise. Whatever the detector is missing, it misses at full resolution",
                    "too, so a wider decode buys nothing.",
                ]
            lines.append("")

    if prefiltered is not None:
        pre_ap = prefiltered.ap(criterion="centre")
        if engine_ap is not None and pre_ap is not None:
            delta = pre_ap - engine_ap
            lines += [
                f"**Is there headroom above the shipped decode?** `native-prefiltered` — the same",
                "full-resolution frame, resampled to the model's input with a real filter —",
                f"scores AP(centre) {_ap(pre_ap)} against the shipped {_ap(engine_ap)} ({delta:+.3f}).",
            ]
            if delta > 0.02:
                lines += [
                    "",
                    "So the 960×544 stage is costing a little that a better resampler recovers —",
                    "a ceiling on what filtering alone can buy, and it is not much.",
                ]
            else:
                lines += [
                    "",
                    "So the decoded 960×544 frame is already carrying everything the model can",
                    "use, and a better resampler in front of it recovers nothing.",
                ]
            lines += ["", ANTIALIAS_PROBE, ""]

    if letterbox is not None and native is not None:
        lb_ap, native_ap = letterbox.ap(criterion="centre"), native.ap(criterion="centre")
        if lb_ap is not None and native_ap is not None:
            delta = lb_ap - native_ap
            verdict = (
                "worth taking" if delta >= 0.02
                else "noise" if abs(delta) < 0.02 else "actively worse"
            )
            lines += [
                f"**What the aspect squash costs:** AP(centre) {_ap(native_ap)} squashed against "
                f"{_ap(lb_ap)} letterboxed ({delta:+.3f}) — {verdict}. RF-DETR's own inference path",
                "squashes, so letterboxing is off-distribution for these weights; a fine-tune",
                "would be free to change that.",
                "",
            ]

    buckets = engine.recall_by_height(criterion="centre")
    small = [b for b in buckets if b[0][1] <= HEIGHT_BUCKETS[0][1] and b[2] > 0]
    large = [b for b in buckets if b[0][0] >= HEIGHT_BUCKETS[-1][0] and b[2] > 0]
    if small and large:
        small_recall = small[0][1] / small[0][2]
        large_recall = large[0][1] / large[0][2]
        lines += [
            f"**Where the misses are:** {_pct(small_recall)} recall on people under "
            f"{HEIGHT_BUCKETS[0][1]:g} px tall in the decoded frame "
            f"({small[0][2]} of them), against {_pct(large_recall)} on people over "
            f"{HEIGHT_BUCKETS[-1][0]:g} px ({large[0][2]}).",
            "",
        ]

    outliers = _worst_scenes(engine, recall)
    if outliers:
        listed = ", ".join(f"`{scene}` ({_pct(value)})" for scene, value in outliers)
        lines += [
            f"**Scenes more than {SCENE_OUTLIER_DELTA:.0%} below the split's own presence recall:** "
            f"{listed}.",
            "A fine-tune pays here first — check them by eye before assuming more labels of",
            "the easy rooms help.",
            "",
        ]

    other = [s for s in run.splits() if s != split]
    if other:
        second = run.get(other[0], SHIPPED_CONDITION)
        if second is not None:
            second_centre = second.point(ENGINE_THRESHOLD, criterion="centre")
            agrees = (
                second_centre.recall is not None and recall is not None
                and abs(second_centre.recall - recall) < 0.1
            )
            lines += [
                f"**Does `{other[0]}` agree?** presence recall {_pct(second_centre.recall)} at "
                f"{_pct(second_centre.precision)} precision over {second.truth_count()} boxes — "
                + (
                    "yes, and that is what makes the small split believable."
                    if agrees else
                    "no, and the disagreement is larger than the small split explains on its "
                    "own. Trust the larger one."
                ),
                "",
            ]

    lines += [
        "**What this is not.** It is not a benchmark of RF-DETR Nano — it is COCO weights on",
        "one plant's cameras, against one annotator's boxes. It is not a licence to quote a",
        "per-camera accuracy to a customer: a few hundred boxes in eight rooms of one",
        "factory is a direction, not a spec. And it says nothing about tracking, which is",
        "where a dropped frame is recovered or a false box becomes a phantom worker — the",
        "appliance counts tracks over a debounce window, not detections in a frame.",
        "",
    ]
    return "\n".join(lines)


def _operator_note(run: DetectorEvalRun) -> list[str]:
    """When a site should bother fine-tuning, and when it should not.

    Written for whoever installs the next appliance, not for whoever wrote the
    trainer — so it leads with the two conditions that make it pointless, and
    quotes this run's own wall clock rather than a general claim about cost.
    """
    tuned = run.finetuned
    provenance = ((tuned.detector if tuned else {}) or {}).get("checkpoint") or {}
    images = provenance.get("dataset_images")
    epochs = provenance.get("epochs_run")

    lines = [
        "## Fine-tuning this detector for a site",
        "",
        "The appliance ships COCO weights and that is the right default: they are the only",
        "weights that are not about somebody else's factory. `detection.model.checkpoint`",
        "loads site-specific ones instead, and it is empty until a site has earned it.",
        "",
        "**A steep recall-by-height table is not a reason to skip this**, and that sentence",
        "replaces the opposite one. The reading above — recall collapsing in the short bands,",
        "holding in the tall — was taken as evidence that the corpus could not help, because",
        "no amount of labelling changes the number of pixels a person occupies. On this plant",
        "that was **wrong**: the fine-tune took the 24–48 px band from 6.4% to 66.0% and the",
        "under-24 band from 0% to 46.7%, at exactly the same 384 px input. Those people were",
        "resolvable all along; COCO had simply never been asked to find someone that small in",
        "a room like this one. The resolution lever and the label lever are not ordered in",
        "advance — measure both, and measure this one first, because it is nine minutes.",
        "",
        "**Do not bother without labels from this site.** Not a similar plant: this one. The",
        "gain, where there is one, is largely the model learning these cameras' angles,",
        "lighting and this annotator's idea of where a person's box ends.",
        "",
        "**Do not bother if false alarms are the complaint.** This buys recall and spends",
        "precision. A site drowning in phantom workers wants the threshold sweep above, not",
        "a fine-tune.",
        "",
        "**Otherwise:**",
        "",
        "```",
        "./scripts/setup_detector_training.sh                     # once, builds .venv-train",
        "uv run python scripts/train_detector.py                  # writes models/finetuned/<date>/",
        "uv run argus detector-eval --checkpoint models/finetuned/<date>/checkpoint_best_total.pth",
        "```",
        "",
    ]
    if provenance:
        wall = provenance.get("wall_clock_sec")
        cost = (
            f"{images} labelled frames across both splits"
            if images else "the labelled frames on this box"
        )
        lines += [
            f"Measured here: {cost}, {epochs or '?'} epochs"
            + (f", {float(wall) / 60:.1f} minutes" if isinstance(wall, (int, float)) else "")
            + " on an idle",
            "GB10 — early-stopped on `val` mAP, inside the epoch box rather than at it. The",
            "training stack lives in its own virtualenv and is never imported by the engine;",
            "`scripts/license_audit.py --check` is unchanged by installing it.",
            "",
        ]
    lines += [
        "The third command is the one that decides. It scores both checkpoints over the same",
        "frames in one run and writes the before/after into this document, per scene — read",
        "the per-scene table, not the mean. Only then does `detection.model.checkpoint` get a",
        "value, and the weights are loaded only alongside the `provenance.json` their",
        "training run wrote beside them: an appliance that cannot say what a model was",
        "trained on will not run it.",
        "",
        "**And then re-set every sanctioned limit.** This is the step that is easy to miss and",
        "impossible to skip. A limit was chosen by an operator watching the *old* detector's",
        "counts, so it encodes that detector's blind spot. On this plant, switching the",
        "checkpoint on took one zone's live headcount from 6 people to 17 in the same",
        "shot — every box on a real worker, most of them ones the stock detector never",
        "reported. A site that adopts the weights without touching its limits does not get a",
        "better appliance, it gets an appliance that alerts constantly and is switched off.",
        "",
    ]
    return lines


def _reproduce(run: DetectorEvalRun) -> str:
    detector = run.detector
    described = ", ".join(
        f"`{key}={detector[key]}`"
        for key in ("model", "backend", "resolution", "device")
        if key in detector
    )
    return "\n".join([
        "## Reproduce",
        "",
        "```",
        "uv run argus detector-eval",
        "```",
        "",
        f"Detector as loaded: {described or '(unavailable)'}. The eval overrides only the",
        f"score threshold ({SCORE_FLOOR}, so one pass carries the sweep); variant, precision, NMS",
        "IoU and weights are the config's.",
        "",
        "`--dataset` points at another export, `--split` and `--condition` narrow the run,",
        "`--json` writes the per-frame dump. The corpus is gitignored, so on a fresh clone",
        "this exits 2 with the path it looked at.",
        "",
        *_operator_note(run),
    ])


__all__ = ["ANTIALIAS_PROBE", "RECALL_FLOOR", "SCENE_OUTLIER_DELTA", "render"]
