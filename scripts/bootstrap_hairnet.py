#!/usr/bin/env python3
"""Propose head-region crops and a first guess at hair covering, for a human to correct.

WHAT THIS IS NOT
================
This is **not a hairnet detector** and running it does not give Argus one. It is
the first half of the loop docs/REBUILD_PLAN.md calls the standing pattern for
every new class: auto-label with a slow zero-shot model, correct by hand, then
train a fast closed-set one. What comes out of here is a directory of crops
sorted into three buckets by a model that has never seen this plant, and every
single one of them is a *candidate* until somebody looks at it.

Nothing in ``argus/`` imports this file. ``argus run`` does not know it exists.
It is a lab tool that writes to ``datasets/hairnet-bootstrap/`` (gitignored) and
exits.

HOW IT DECIDES
==============
OWLv2 (Apache-2.0, ``google/owlv2-base-patch16-ensemble``) is an open-vocabulary
detector: it scores boxes against free-text phrases. One pass over each image
with a phrase list that spans both classes, then per surviving box:

  * the winning phrase's class decides the bucket — ``hairnet`` / ``no_hairnet``;
  * a box whose winner is the neutral anchor phrase ("a human head") goes to
    ``unsure``, because the model found a head and could not say more;
  * a box whose two best phrases are from *different* classes and within
    ``--margin`` of each other also goes to ``unsure``. A 0.31-vs-0.29 split is
    a coin toss wearing a decimal point, and the whole value of this pass is
    that the coin tosses are separated out for a human rather than mixed into
    the training set.

THE THING A REVIEWER MUST DECIDE FIRST
======================================
This plant's people cover their hair with several different things: blue
disposable nets, cloth caps, and coloured headscarves tied over the hair. Those
are three different objects and one compliance question, and which of them the
*site* accepts is not something a model can be asked. The buckets below put all
three in ``hairnet`` on the grounds that hair is covered; docs/HAIRNET_BOOTSTRAP.md
is where a reviewer records the site's actual rule and what to do with the ones
that do not match it.

USAGE
=====
    ./scripts/fetch_owlv2.sh
    uv run python scripts/bootstrap_hairnet.py --limit 40      # a look first
    uv run python scripts/bootstrap_hairnet.py                 # the whole corpus

    # then, by hand:
    #   drag files between datasets/hairnet-bootstrap/crops/{hairnet,no_hairnet,unsure}/
    #   delete anything that is not a head at all
    #   read docs/HAIRNET_BOOTSTRAP.md before starting
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"
DEFAULT_OUT = ROOT / "datasets" / "hairnet-bootstrap"

# Where the frames come from. Both are gitignored customer data; a missing one
# is skipped with a sentence rather than being an error, because a fresh clone
# has neither and must still be able to run --help and the tests.
DATASET_IMAGES = ROOT / "footage" / "anmol-samples" / "cctv_person_tracking_dataset" / "images"
VIDEO_DIR = ROOT / "footage" / "anmol-samples"
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".m4v")

# The phrase list, and the class each phrase votes for. Order is not
# significant; the text is. Two things were learned writing it and are worth
# keeping:
#
#   * OWLv2 scores a *phrase*, so "a hairnet" alone lands on the net as an
#     object and boxes the fabric rather than the head. Every phrase here is
#     therefore about a head, so every box is a head box and the crops are
#     comparable to each other.
#   * The uncovered phrases have to name the hair. "a bare head" scores highest
#     on the backs of heads whatever is on them; "with uncovered black hair"
#     is the phrase that actually separates.
PHRASES: tuple[tuple[str, str], ...] = (
    ("a factory worker's head wearing a disposable hairnet", "hairnet"),
    ("a head wearing a blue hairnet", "hairnet"),
    ("a head wearing a cloth cap", "hairnet"),
    ("a head wearing a headscarf tied over the hair", "hairnet"),
    ("a head with uncovered black hair", "no_hairnet"),
    ("a bare human head, no cap and no net", "no_hairnet"),
    ("a human head", "unsure"),
)

BUCKETS = ("hairnet", "no_hairnet", "unsure")

# Scenes where only part of the frame is the camera, keyed by the prefix
# :func:`scene_of` returns.
#
# `Broken_Biscuits` is a **screen recording of a video call** in which somebody
# was sharing the camera (config/seed.anmol.yaml says so at length). The shared
# picture occupies x 0.053-0.942, y 0.153-0.975; above it is a row of
# participant tiles, and on the smoke run of 2026-08-25 OWLv2 found the
# participant's face in six of thirteen candidates, filed the *same* face as
# `hairnet`, `no_hairnet` and `unsure` across three crops, and gave it the
# largest head boxes in the set — 192 px against 88 px for the real worker in
# the plant behind her.
#
# That is not only noise. It is a named person on a video call, harvested from a
# customer's file into a training set, which is the one thing this tool must not
# do quietly. Restricting the scan to the shared picture is the fix; it is a
# constant rather than a flag because forgetting the flag is the failure.
SCENE_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "Broken_Biscuits": (0.053, 0.153, 0.942, 0.975),
}


@dataclass(slots=True)
class Candidate:
    """One proposed head region, and every number that produced it."""

    crop: str                    # path relative to the output directory
    source: str                  # path relative to the repo root
    scene: str
    bucket: str
    score: float
    phrase: str
    runner_up: str = ""
    runner_up_score: float = 0.0
    box_px: tuple[int, int, int, int] = (0, 0, 0, 0)
    box_norm: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    crop_px: tuple[int, int, int, int] = (0, 0, 0, 0)
    head_px: int = 0             # the longer side of the head box, in source pixels

    def as_dict(self) -> dict[str, Any]:
        return {
            "crop": self.crop,
            "source": self.source,
            "scene": self.scene,
            "bucket": self.bucket,
            "score": round(self.score, 4),
            "phrase": self.phrase,
            "runner_up": self.runner_up,
            "runner_up_score": round(self.runner_up_score, 4),
            "box_px": list(self.box_px),
            "box_norm": [round(v, 5) for v in self.box_norm],
            "crop_px": list(self.crop_px),
            "head_px": self.head_px,
        }


@dataclass
class Run:
    started_at: str
    model: str
    images: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    duration_sec: float = 0.0

    def by_bucket(self) -> dict[str, int]:
        counts = {name: 0 for name in BUCKETS}
        for candidate in self.candidates:
            counts[candidate.bucket] = counts.get(candidate.bucket, 0) + 1
        return counts


def scene_of(path: Path) -> str:
    """The scene a frame belongs to, whichever of this corpus's three spellings.

    ``Loading_Point_frame_00.jpg``   the labelled dataset's stills
    ``Loading_Point_t000300.jpg``    stills this tool cut from the video
    ``Broken-Biscuits-frame-07.png`` the same dataset, hyphenated

    It is not only cosmetic. :data:`SCENE_REGIONS` is keyed by what this
    returns, so a spelling it does not strip means a scene silently loses its
    region — which for ``Broken_Biscuits`` means the video-call participant
    tiles come back. That is how the ``_t<centiseconds>`` case was found.
    """
    stem = path.stem
    for marker in ("_frame_", "-frame-"):
        if marker in stem:
            return stem.split(marker)[0]
    trimmed = re.sub(r"[_-]t\d+$", "", stem)
    return trimmed or stem


def iter_images(
    *, dataset_images: Path, video_dir: Path, every_sec: float, per_video: int,
    frames_dir: Path, log: Any,
) -> Iterable[Path]:
    """Dataset stills first, then stills cut from the videos.

    The labelled dataset comes first on purpose: those 79 frames are the ones
    whose person boxes already exist, so a hairnet label added to them lands
    beside a person box rather than needing one. The video frames are the
    *supplement* — they cover Loading_Point and Broken_Biscuits at timestamps
    the dataset skipped, and they are what stops the bootstrap set being 79
    pictures of four benches.
    """
    if dataset_images.is_dir():
        found = sorted(
            path for path in dataset_images.rglob("*")
            if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        log(f"images: {len(found)} from {dataset_images.relative_to(ROOT)}")
        yield from found
    else:
        log(f"images: {dataset_images.relative_to(ROOT)} is not on this box — skipped")

    if not video_dir.is_dir():
        log(f"images: {video_dir} is not on this box — skipped")
        return
    from argus.eval.frames import cut_frame, probe_duration, sample_offsets

    videos = sorted(
        path for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    frames_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        offsets = sample_offsets(probe_duration(video), every_sec, per_video)
        cut = 0
        for offset in offsets:
            out = frames_dir / f"{video.stem}_t{int(round(offset * 100)):06d}.jpg"
            if not out.is_file():
                try:
                    cut_frame(video, offset, out)
                except Exception as exc:  # noqa: BLE001 - one bad seek is not a failed run
                    log(f"images: {video.name} @ {offset}s — {exc}")
                    continue
            cut += 1
            yield out
        log(f"images: {cut} from {video.name}")


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_boxes(
    boxes: list[tuple[float, list[float], int]], *, iou: float,
) -> list[tuple[float, list[float], int, str, float]]:
    """Class-agnostic NMS that keeps the loser's score.

    Class-*agnostic* is the whole point: the phrases are seven ways of
    describing one head, so per-class NMS would emit the same head three times
    under three labels. Suppressing across classes and keeping the best
    suppressed score from a different class is what makes the ``--margin``
    abstention above possible — otherwise there is nothing to take a margin
    against.

    Returns ``(score, box, phrase_index, runner_up_phrase, runner_up_score)``.
    """
    ordered = sorted(boxes, key=lambda item: item[0], reverse=True)
    kept: list[tuple[float, list[float], int, str, float]] = []
    for score, box, index in ordered:
        hit = None
        for position, (_, other, _, _, _) in enumerate(kept):
            if _iou(box, other) >= iou:
                hit = position
                break
        if hit is None:
            kept.append((score, box, index, "", 0.0))
            continue
        best_score, best_box, best_index, runner, runner_score = kept[hit]
        # Only a phrase from a *different* class is informative as a runner-up:
        # two hairnet phrases agreeing says nothing about the abstention band.
        if PHRASES[index][1] != PHRASES[best_index][1] and score > runner_score:
            kept[hit] = (best_score, best_box, best_index, PHRASES[index][0], score)
    return kept


def bucket_for(
    phrase_index: int, score: float, runner_up_score: float, *, margin: float,
) -> str:
    """Which directory this crop lands in. See the module docstring."""
    votes = PHRASES[phrase_index][1]
    if votes == "unsure":
        return "unsure"
    if runner_up_score and (score - runner_up_score) < margin:
        return "unsure"
    return votes


def expand(
    box: Sequence[float], width: int, height: int, *, pad: float, min_px: int,
) -> tuple[int, int, int, int]:
    """Grow a head box into a crop a person can actually judge.

    A tight head box at this camera distance is 20-40 px on a side, and a human
    asked to say "hairnet or not" from a 24x24 thumbnail will say "unsure" to
    all of it. The padding buys shoulders and background for scale, and
    ``min_px`` stops the crop being smaller than the JPEG it is written to.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = max((x2 - x1) * (1.0 + pad) / 2.0, min_px / 2.0)
    half_h = max((y2 - y1) * (1.0 + pad) / 2.0, min_px / 2.0)
    left = int(max(0, round(cx - half_w)))
    top = int(max(0, round(cy - half_h)))
    right = int(min(width, round(cx + half_w)))
    bottom = int(min(height, round(cy + half_h)))
    return (left, top, max(left + 1, right), max(top + 1, bottom))


def run(args: argparse.Namespace, *, log: Any = print) -> Run:
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / "models" / args.model
    if not (model_path / "config.json").is_file():
        raise SystemExit(
            f"no OWLv2 checkpoint at {model_path}\n"
            f"    ./scripts/fetch_owlv2.sh {args.model}"
        )

    out_dir = Path(args.out)
    crops_dir = out_dir / "crops"
    for bucket in BUCKETS:
        (crops_dir / bucket).mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # float32 on CPU, float16 on the GB10: OWLv2's ViT is the whole cost here
    # and this is a batch job nobody is waiting on interactively.
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = Owlv2Processor.from_pretrained(str(model_path))
    model = Owlv2ForObjectDetection.from_pretrained(str(model_path), dtype=dtype).to(device)
    model.eval()

    started = time.time()
    result = Run(
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        model=args.model,
    )
    phrases = [text for text, _ in PHRASES]

    def note(*parts: Any) -> None:
        """Where an image came from, kept for provenance.json AND printed.

        The generator below yields lazily, so these arrive interleaved with the
        progress lines rather than as a header — which is the honest order: the
        video frames really are cut while the scan is running.
        """
        line = " ".join(str(part) for part in parts)
        result.notes.append(line)
        log(line)

    images = iter_images(
        dataset_images=Path(args.dataset_images), video_dir=Path(args.video_dir),
        every_sec=args.every_sec, per_video=args.per_video,
        frames_dir=out_dir / "frames", log=note,
    )

    for position, image_path in enumerate(images):
        if args.limit and position >= args.limit:
            break
        try:
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"skip {image_path.name}: {exc}")
            continue
        result.images += 1
        source_width, source_height = image.size
        scene = scene_of(image_path)
        region = SCENE_REGIONS.get(scene)
        origin_x = origin_y = 0
        if region is not None:
            box = (
                round(region[0] * image.width), round(region[1] * image.height),
                round(region[2] * image.width), round(region[3] * image.height),
            )
            origin_x, origin_y = box[0], box[1]
            image = image.crop(box)
        width, height = image.size

        # DETECT SMALL, CROP LARGE. Two reasons, and the first is a crash:
        # OWLv2's own preprocessor anti-aliases with a gaussian whose kernel is
        # sized from the downscale factor, and 2910x1682 -> 960 makes a kernel
        # torch's conv2d refuses ("RuntimeError: label is too far", on CPU).
        # The second is the point of the tool — the model needs 960 px and the
        # *human* needs every pixel the camera sent, so boxes are found on a
        # downscaled copy and mapped back onto the source before cropping.
        small = image
        if width > args.detect_width:
            small = image.resize(
                (args.detect_width, max(1, round(height * args.detect_width / width))),
                Image.LANCZOS,
            )
        scale = width / small.width

        inputs = processor(text=[phrases], images=small, return_tensors="pt").to(device)
        if dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            outputs = model(**inputs)
        post = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=torch.tensor([[small.height, small.width]], device=device),
            threshold=args.threshold,
        )[0]

        raw = [
            (float(score), [float(v) * scale for v in box], int(label))
            for score, box, label in zip(post["scores"], post["boxes"], post["labels"])
        ]
        for score, box, index, runner, runner_score in merge_boxes(raw, iou=args.iou):
            bucket = bucket_for(index, score, runner_score, margin=args.margin)
            crop_box = expand(box, width, height, pad=args.pad, min_px=args.min_crop)
            name = (
                f"{scene}_{image_path.stem}_"
                f"{int(box[0]):04d}x{int(box[1]):04d}_{int(score * 1000):03d}.jpg"
            )
            crop_path = crops_dir / bucket / name
            image.crop(crop_box).save(crop_path, quality=92)
            result.candidates.append(Candidate(
                crop=str(crop_path.relative_to(out_dir)),
                source=_relative(image_path),
                scene=scene,
                bucket=bucket,
                score=score,
                phrase=PHRASES[index][0],
                runner_up=runner,
                runner_up_score=runner_score,
                # Reported against the SOURCE image, not the region: a box a
                # human cannot draw back onto the original file is not
                # provenance, it is a number.
                box_px=(int(box[0]) + origin_x, int(box[1]) + origin_y,
                        int(box[2]) + origin_x, int(box[3]) + origin_y),
                box_norm=((box[0] + origin_x) / source_width,
                          (box[1] + origin_y) / source_height,
                          (box[2] + origin_x) / source_width,
                          (box[3] + origin_y) / source_height),
                crop_px=(crop_box[0] + origin_x, crop_box[1] + origin_y,
                         crop_box[2] + origin_x, crop_box[3] + origin_y),
                head_px=int(round(max(box[2] - box[0], box[3] - box[1]))),
            ))
        if result.images % 10 == 0:
            log(f"  {result.images} image(s), {len(result.candidates)} candidate(s)")

    result.duration_sec = time.time() - started
    _write(result, out_dir, args)
    return result


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write(result: Run, out_dir: Path, args: argparse.Namespace) -> None:
    """candidates.jsonl + provenance.json. Both matter and for different reasons.

    ``candidates.jsonl`` is the record of where every crop *started*, so after a
    human has dragged files between the three directories, the difference
    between this file and the directory tree is exactly the correction — which
    is both the training signal and the only honest measurement of how good the
    zero-shot pass was.

    ``provenance.json`` is the same contract ``scripts/train_detector.py``
    writes: what produced this, from what, with which knobs. A dataset without
    it is a directory of pictures.
    """
    (out_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(item.as_dict()) + "\n" for item in result.candidates)
    )
    provenance = {
        "tool": "scripts/bootstrap_hairnet.py",
        "started_at": result.started_at,
        "duration_sec": round(result.duration_sec, 1),
        "model": result.model,
        "model_sha256": _tree_digest(Path(args.model) if Path(args.model).is_absolute()
                                     else ROOT / "models" / args.model),
        "phrases": [{"text": text, "votes": votes} for text, votes in PHRASES],
        "scene_regions": {k: list(v) for k, v in SCENE_REGIONS.items()},
        "settings": {
            "threshold": args.threshold, "iou": args.iou, "margin": args.margin,
            "pad": args.pad, "min_crop": args.min_crop, "device": args.device,
            "detect_width": args.detect_width,
            "every_sec": args.every_sec, "per_video": args.per_video,
            "limit": args.limit,
        },
        "sources": {
            "dataset_images": _relative(Path(args.dataset_images)),
            "video_dir": _relative(Path(args.video_dir)),
        },
        "images": result.images,
        "candidates": len(result.candidates),
        "by_bucket": result.by_bucket(),
        "by_scene": _by_scene(result),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "notes": result.notes,
        "WARNING": (
            "These are CANDIDATE labels from a zero-shot model that has never seen "
            "this plant. Nothing here is a hairnet detector and nothing here is "
            "ground truth until a human has reviewed it — docs/HAIRNET_BOOTSTRAP.md."
        ),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def _by_scene(result: Run) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for candidate in result.candidates:
        bucket = out.setdefault(candidate.scene, {name: 0 for name in BUCKETS})
        bucket[candidate.bucket] = bucket.get(candidate.bucket, 0) + 1
    return dict(sorted(out.items()))


def _tree_digest(path: Path) -> str:
    """A digest of the checkpoint's weight files, so a re-run with different
    weights is visible rather than inferred from a directory name."""
    if not path.is_dir():
        return ""
    digest = hashlib.sha256()
    for weight in sorted(path.glob("*.safetensors")):
        digest.update(weight.name.encode())
        digest.update(str(weight.stat().st_size).encode())
    return digest.hexdigest()[:16]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="This produces CANDIDATES, not labels. Read docs/HAIRNET_BOOTSTRAP.md.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OWLv2 checkpoint under models/ (default: {DEFAULT_MODEL})")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"output directory (default: {DEFAULT_OUT.relative_to(ROOT)})")
    parser.add_argument("--dataset-images", default=str(DATASET_IMAGES),
                        help="directory of labelled stills to scan")
    parser.add_argument("--video-dir", default=str(VIDEO_DIR),
                        help="directory of source videos to sample stills from")
    parser.add_argument("--every-sec", type=float, default=6.0,
                        help="video sampling interval (default: 6)")
    parser.add_argument("--per-video", type=int, default=10,
                        help="max frames per video (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.12,
                        help="OWLv2 score floor (default: 0.12 — low on purpose; "
                             "recall now, precision from the human pass)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="class-agnostic NMS IoU (default: 0.5)")
    parser.add_argument("--margin", type=float, default=0.03,
                        help="a box whose best two phrases disagree by less than this "
                             "goes to unsure/ (default: 0.03)")
    parser.add_argument("--pad", type=float, default=1.4,
                        help="fraction of the head box added around the crop (default: 1.4)")
    parser.add_argument("--min-crop", type=int, default=96,
                        help="minimum crop side in source pixels (default: 96)")
    parser.add_argument("--detect-width", type=int, default=1280,
                        help="images are downscaled to this width for detection and "
                             "the boxes mapped back; crops still come from the source "
                             "pixels (default: 1280 — OWLv2 sees 960 either way)")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N images (0 = all)")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(f"\nimages:     {result.images}")
    print(f"candidates: {len(result.candidates)} in {result.duration_sec:.1f}s")
    for bucket, count in result.by_bucket().items():
        print(f"  {bucket:<12} {count}")
    print(f"\nwritten to {args.out}")
    print("These are CANDIDATES. Read docs/HAIRNET_BOOTSTRAP.md before using any of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
