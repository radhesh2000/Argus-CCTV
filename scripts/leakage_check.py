#!/usr/bin/env python
"""Which evaluation frames are already in the training set, and how badly.

The hand-labelled corpus and the harvest were cut from the *same six clips*.
`scripts/harvest_frames.py` samples one native frame per second of `t < 40 s`
into `train`; the 79 hand-labelled frames were cut from those same clips at
unrecorded times. Nothing prevented a hand-labelled **val** frame from landing
half a second away from a harvested **train** frame, and if it did, the split
the student early-stops on is partly inside its own training set.

That is not a hypothetical and it is not visible anywhere else: the filenames
are `_frame_07` and `t018.jpg`, which share nothing, and the two corpora were
built by different scripts months apart. This measures it directly, in pixels.

**How it decides.** Every frame is reduced to a 96x54 greyscale thumbnail and
compared by RMSE. The threshold is not a constant — it is derived per scene
from the harvest itself: the distance between *consecutive* harvested frames,
one second apart, is what "the same shot a moment later" looks like in this
camera, in this lighting, with this much worker movement. A hand-labelled frame
closer to a training frame than the closest consecutive pair in its own scene
is, for training purposes, that frame.

    .venv/bin/python scripts/leakage_check.py
    .venv/bin/python scripts/leakage_check.py --split val --json report.json

**Fixing it, not only naming it.** `--write-exclusions` turns the measurement
into the list of harvested frames a training set must not contain: for every
hand-labelled frame of the checked split, the harvested second it matches best
becomes an *anchor*, and every harvested frame within `--exclude-window-sec` of
that anchor is excluded. The window is temporal rather than photometric because
the thing being excluded is a neighbourhood, not a match — a frame 1.5 s away
from a val frame is a different picture by RMSE and the same moment by any
standard a detector cares about.

The anchor is a guess for the four val frames the check calls `distinct`: their
true capture time is unrecorded, and nearest-by-RMSE is the only estimate
available. Excluding around a guess costs training frames and cannot create a
leak, so it is done anyway and counted.

    .venv/bin/python scripts/leakage_check.py --split val \
        --exclude-window-sec 2 --write-exclusions footage/leak-exclusions.json
    .venv/bin/python scripts/leakage_check.py --split val \
        --excluded footage/leak-exclusions.json      # re-check: is it clean now?

No GPU, no model. Nothing here writes to either corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "footage" / "anmol-samples" / "cctv_person_tracking_dataset"
HARVEST = ROOT / "footage" / "anmol-harvest"

# Small enough that JPEG noise and a worker's arm do not dominate, large enough
# that two different moments on a packing line are not the same picture.
THUMB = (96, 54)


def thumbnail(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("L").resize(THUMB, Image.BILINEAR), dtype=np.float32) / 255.0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def harvest_signatures(harvest: Path, split: str) -> dict[str, list[tuple[float, np.ndarray, str]]]:
    """Every harvested frame of ``split``, by scene, sorted in time."""
    manifest = json.loads((harvest / "manifest.json").read_text())
    by_scene: dict[str, list[tuple[float, np.ndarray, str]]] = {}
    for entry in manifest["frames"]:
        if entry["split"] != split:
            continue
        by_scene.setdefault(str(entry["scene"]), []).append(
            (float(entry["t_sec"]), thumbnail(harvest / str(entry["file"])), str(entry["file"]))
        )
    for frames in by_scene.values():
        frames.sort(key=lambda triple: triple[0])
    return by_scene


def without(signatures: dict[str, list[tuple[float, np.ndarray, str]]], excluded: frozenset[str]
            ) -> dict[str, list[tuple[float, np.ndarray, str]]]:
    """The reference set a *training set* was actually built from.

    Dropping the excluded frames here is what makes the re-check meaningful:
    the question the second run answers is not "was the harvest contaminated"
    — it was — but "is what the model was trained on contaminated".

    The per-scene baseline is deliberately **not** recomputed from this. What
    "one second later in this camera" costs in RMSE is a property of the
    camera; measuring it across a hole this function just punched would widen
    the threshold exactly where the check needs it not to move.
    """
    if not excluded:
        return signatures
    return {
        scene: [triple for triple in frames if triple[2] not in excluded]
        for scene, frames in signatures.items()
    }


def harvest_times(harvest: Path, split: str) -> dict[str, list[tuple[float, str]]]:
    """``scene -> [(t_sec, "scene/file.jpg")]``, without reading any pixels."""
    manifest = json.loads((harvest / "manifest.json").read_text())
    by_scene: dict[str, list[tuple[float, str]]] = {}
    for entry in manifest["frames"]:
        if entry["split"] != split:
            continue
        by_scene.setdefault(str(entry["scene"]), []).append(
            (float(entry["t_sec"]), str(entry["file"]))
        )
    for frames in by_scene.values():
        frames.sort(key=lambda pair: pair[0])
    return by_scene


def exclusions_for(rows: list[dict[str, Any]], times: dict[str, list[tuple[float, str]]],
                   window_sec: float) -> tuple[list[str], list[dict[str, Any]]]:
    """Harvested frames within ``window_sec`` of any checked frame's anchor.

    One entry per anchor is returned alongside the flat list, because "31 frames
    removed" is not reviewable and "these eight val frames each cost 3–5
    harvested seconds, here is which" is.
    """
    excluded: set[str] = set()
    anchors: list[dict[str, Any]] = []
    for row in rows:
        if row["nearest_t_sec"] is None:
            continue
        centre = float(row["nearest_t_sec"])
        hits = sorted(
            file for t, file in times.get(row["scene"], []) if abs(t - centre) <= window_sec
        )
        excluded.update(hits)
        anchors.append({
            "frame": row["frame"], "scene": row["scene"], "verdict": row["verdict"],
            "anchor_t_sec": centre, "rmse": row["rmse"], "excludes": hits,
        })
    return sorted(excluded), anchors


def scene_baseline(frames: list[tuple[float, np.ndarray, str]]) -> tuple[float, float]:
    """What "one second later in this scene" costs in RMSE: (min, median).

    Per scene because the scenes differ enormously. `Loading_Point` is four
    people in a large still room and its consecutive frames are far apart;
    `Line_No.3_Packing4_Bas` is twenty-five people at arm's length and its are
    close. One global threshold would call the first clean and the second
    contaminated, both times for the wrong reason.
    """
    gaps = [rmse(a[1], b[1]) for a, b in zip(frames, frames[1:])]
    return (float(np.min(gaps)), float(np.median(gaps))) if gaps else (0.0, 0.0)


def classify(distance: float, closest_consecutive: float, median_consecutive: float) -> str:
    if distance <= closest_consecutive:
        return "duplicate"
    if distance <= median_consecutive:
        return "near-duplicate"
    return "distinct"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--harvest", type=Path, default=HARVEST)
    parser.add_argument("--against", default="train", help="the harvest split that was trained on")
    parser.add_argument("--split", nargs="*", default=["val", "train"],
                        help="which hand-labelled splits to check")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--excluded", type=Path, default=None,
                        help="a --write-exclusions report; those harvested frames are dropped "
                             "from the reference set, so this checks the TRAINING SET rather "
                             "than the harvest")
    parser.add_argument("--exclude-window-sec", type=float, default=2.0,
                        help="half-width of the temporal neighbourhood --write-exclusions removes")
    parser.add_argument("--write-exclusions", type=Path, default=None,
                        help="write the harvested frames a clean training set must omit")
    args = parser.parse_args()

    from argus.eval.detector import scene_of

    excluded_in: frozenset[str] = frozenset()
    if args.excluded is not None:
        excluded_in = frozenset(json.loads(args.excluded.read_text())["excluded"])
        print(f"reference set excludes {len(excluded_in)} harvested {args.against} frames "
              f"({args.excluded})\n")

    everything = harvest_signatures(args.harvest, args.against)
    # The threshold comes from the whole harvest even when the reference set is
    # a subset of it — see `without`.
    baselines = {scene: scene_baseline(frames) for scene, frames in everything.items()}
    reference = without(everything, excluded_in)

    rows: list[dict[str, Any]] = []
    for split in args.split:
        for image in sorted((args.dataset / "images" / split).glob("*.jpg")):
            scene = scene_of(image.stem)
            if scene not in reference:
                rows.append({"frame": image.stem, "split": split, "scene": scene,
                             "verdict": "no-harvest", "rmse": None, "nearest_t_sec": None})
                continue
            signature = thumbnail(image)
            distance, when = min(
                (rmse(signature, other), t) for t, other, _ in reference[scene]
            )
            closest, median = baselines[scene]
            rows.append({
                "frame": image.stem, "split": split, "scene": scene,
                "rmse": round(distance, 4), "nearest_t_sec": when,
                "scene_consecutive_min": round(closest, 4),
                "scene_consecutive_median": round(median, 4),
                "verdict": classify(distance, closest, median),
            })

    width = max(len(r["frame"]) for r in rows)
    print(f"{'frame':<{width}} {'split':<6} {'rmse':>7} {'nearest':>8}  {'1s min':>7} {'1s med':>7}  verdict")
    for row in rows:
        if row["verdict"] == "no-harvest":
            print(f"{row['frame']:<{width}} {row['split']:<6} "
                  f"{'-':>7} {'-':>8}  {'-':>7} {'-':>7}  no harvest for this scene")
            continue
        print(f"{row['frame']:<{width}} {row['split']:<6} {row['rmse']:7.4f} "
              f"{row['nearest_t_sec']:7.0f}s  {row['scene_consecutive_min']:7.4f} "
              f"{row['scene_consecutive_median']:7.4f}  {row['verdict']}")

    print()
    for split in args.split:
        subset = [r for r in rows if r["split"] == split]
        counts = {v: sum(1 for r in subset if r["verdict"] == v) for v in
                  ("duplicate", "near-duplicate", "distinct", "no-harvest")}
        contaminated = counts["duplicate"] + counts["near-duplicate"]
        print(f"{split}: {len(subset)} frames — {contaminated} are a harvested "
              f"{args.against} frame or within one second of one "
              f"({counts['duplicate']} duplicate, {counts['near-duplicate']} near, "
              f"{counts['distinct']} distinct, {counts['no-harvest']} from scenes never harvested)")

    if args.write_exclusions:
        if excluded_in:
            raise SystemExit(
                "--write-exclusions on top of --excluded would compute a neighbourhood of a "
                "neighbourhood. Write the exclusions from the unfiltered harvest, once."
            )
        times = harvest_times(args.harvest, args.against)
        total = sum(len(frames) for frames in times.values())
        excluded, anchors = exclusions_for(rows, times, args.exclude_window_sec)
        print(f"\nexcluding every harvested {args.against} frame within "
              f"±{args.exclude_window_sec:g} s of a checked frame's nearest match:")
        for anchor in anchors:
            print(f"  {anchor['frame']:<{width}} t={anchor['anchor_t_sec']:>4.0f}s "
                  f"({anchor['verdict']:<14}) -> {len(anchor['excludes'])} frames")
        print(f"  {len(excluded)} of {total} harvested {args.against} frames removed "
              f"({len(excluded) / max(total, 1):.0%})")
        args.write_exclusions.write_text(json.dumps({
            "generator": "scripts/leakage_check.py --write-exclusions",
            "harvest_split": args.against,
            "checked_splits": list(args.split),
            "window_sec": args.exclude_window_sec,
            "harvest_frames_total": total,
            "excluded_count": len(excluded),
            "excluded": excluded,
            "anchors": anchors,
            "note": (
                "Anchors are nearest-by-RMSE, which is an estimate of a capture time the "
                "corpus never recorded. For a frame the check calls `distinct` the anchor may "
                "be the wrong second entirely; excluding around it costs training frames and "
                "cannot create a leak, so it is done anyway."
            ),
        }, indent=2) + "\n")
        print(f"wrote {args.write_exclusions}")

    if args.json:
        args.json.write_text(json.dumps({
            "generator": "scripts/leakage_check.py",
            "dataset": str(args.dataset.relative_to(ROOT)),
            "harvest_split": args.against,
            "thumbnail": list(THUMB),
            "excluded_from_reference": sorted(excluded_in),
            "rows": rows,
            "note": (
                "A hand-labelled val frame classed duplicate or near-duplicate is, for "
                "training purposes, inside the training set: the student saw that moment. "
                "The production fine-tune trained on 63 hand-labelled frames only and does "
                "not have this contamination, so a val comparison between them is not "
                "symmetric."
            ),
        }, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
