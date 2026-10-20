#!/usr/bin/env python3
"""Ask the served VLM the same question at several decode resolutions.

WHY THIS EXISTS
===============
There is a resolution gap in the middle of this product and nothing measured it
until now. A prompt policy is scored two ways:

  * in ``argus eval``, on a frame cut from the **source video** at 2560x1440 and
    then cropped to the zone — so the crop can be 800 px across;
  * in ``argus run``, on the engine's **960x544 decode** and then cropped to the
    same zone — so the same crop is 300 px across, and ``zone_crop`` grows the
    short side to 320 rather than adding detail that is not there.

Those are not the same question. A policy that scores well in the harness and
answers "no" in service has not regressed; it has been shown a different
picture. This probe takes one prompt, one zone rectangle and a list of frames,
and asks it at each width in ``--at``, going through the **real**
``zone_crop`` and the **real** ``VlmClient`` so the pixels are the pixels the
appliance sends.

It is a lab tool. Nothing in ``argus/`` imports it.

USAGE
=====
    argus vlm start

    # what the engine's decode costs a plant policy
    uv run python scripts/vlm_resolution_probe.py \\
        --policy anmol_carton_loading_count \\
        --video footage/anmol-samples/Loading_Point.MOV \\
        --ts 3,15,27,39,51 --crop 0.28,0.0,0.60,0.50 --at 2560,960

    # can the model see a hairnet at all, and at what size
    uv run python scripts/vlm_resolution_probe.py \\
        --prompt-file /tmp/hairnet.txt \\
        --video footage/anmol-samples/Line_No.3_Packing3_Bas.MP4 \\
        --ts 21 --crop 0.50,0.20,0.62,0.34 --at 2560,1280,960
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.vlm.evidence import MIN_CROP_PX  # noqa: E402  (after the path fix)


@dataclass(slots=True)
class Probe:
    """One (frame, width) question and the answer to it."""

    frame: str
    width: int
    crop_px: tuple[int, int, int, int]
    cropped: bool
    triggered: bool
    confidence: float
    reason: str
    latency_sec: float
    expected: bool | None = None

    @property
    def agrees(self) -> bool | None:
        return None if self.expected is None else self.triggered == self.expected

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame, "width": self.width,
            "crop_px": list(self.crop_px), "cropped": self.cropped,
            "triggered": self.triggered, "confidence": round(self.confidence, 3),
            "reason": self.reason, "latency_sec": round(self.latency_sec, 2),
            "expected": self.expected, "agrees": self.agrees,
        }


@dataclass
class Result:
    prompt: str
    model: str = ""
    probes: list[Probe] = field(default_factory=list)

    def by_width(self) -> dict[int, list[Probe]]:
        out: dict[int, list[Probe]] = {}
        for probe in self.probes:
            out.setdefault(probe.width, []).append(probe)
        return dict(sorted(out.items(), reverse=True))


def rectangle(text: str) -> list[list[float]]:
    """``x1,y1,x2,y2`` in 0-1 fractions -> the polygon ``zone_crop`` wants."""
    try:
        x1, y1, x2, y2 = (float(value) for value in text.split(","))
    except ValueError as exc:
        raise SystemExit(f"--crop wants x1,y1,x2,y2 in 0-1 fractions: {exc}") from exc
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def frames_at(video: Path, offsets: Sequence[float], out_dir: Path) -> list[Path]:
    from argus.eval.frames import cut_frame

    out_dir.mkdir(parents=True, exist_ok=True)
    cut: list[Path] = []
    for offset in offsets:
        path = out_dir / f"{video.stem}_t{int(round(offset * 100)):06d}.jpg"
        if not path.is_file():
            cut_frame(video, offset, path)
        cut.append(path)
    return cut


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text().strip()
    if args.policy:
        from argus.config import load_config
        from argus.db import Database

        config = load_config(args.config)
        row = Database(config.db_path).get_policy(args.policy)
        if row is None or not row.get("prompt"):
            raise SystemExit(f"no policy {args.policy!r} with a prompt in {config.db_path}")
        return str(row["prompt"]).strip()
    raise SystemExit("one of --prompt / --prompt-file / --policy is required")


async def probe(args: argparse.Namespace) -> Result:
    from PIL import Image

    from argus.config import load_config
    from argus.vlm.client import VlmClient
    from argus.vlm.evidence import zone_crop

    config = load_config(args.config)
    client = VlmClient(config.vlm)
    health = await client.health()
    if not health.get("available"):
        raise SystemExit(f"no model at {client.base_url}: {health.get('reason')} "
                         "— start it with `argus vlm start`")

    prompt = resolve_prompt(args)
    polygon = rectangle(args.crop)
    offsets = [float(value) for value in args.ts.split(",") if value.strip()]
    expected = None if args.expect is None else args.expect
    result = Result(prompt=prompt, model=str(health.get("model") or ""))

    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    sources = frames_at(video, offsets, Path(args.work))

    for width in (int(value) for value in args.at.split(",")):
        for path in sources:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
                if image.width != width:
                    # The engine's ladder scales with a proper filter before the
                    # model ever sees it, and so does ffmpeg; using anything
                    # cheaper here would measure the resampler, not the model.
                    # (docs/DETECTOR_EVAL.md found exactly that on the detector:
                    # RF-DETR's own resize does not anti-alias and scores worse
                    # than ffmpeg's at the same size.)
                    height = max(1, round(image.height * width / image.width))
                    image = image.resize((width, height), Image.LANCZOS)
            found = zone_crop(image, polygon, min_px=args.min_crop_px)
            if found is None:
                view, cropped = image, False
                box = (0, 0, image.width, image.height)
            else:
                view, rect = found
                cropped = True
                box = (
                    round(rect[0] * image.width), round(rect[1] * image.height),
                    round(rect[2] * image.width), round(rect[3] * image.height),
                )
            verdict = await client.evaluate(prompt, view, timeout=args.timeout)
            result.probes.append(Probe(
                frame=path.name, width=width,
                crop_px=(box[2] - box[0], box[3] - box[1], view.width, view.height),
                cropped=cropped,
                triggered=verdict.triggered, confidence=verdict.confidence,
                reason=verdict.reason, latency_sec=verdict.latency_sec,
                expected=expected,
            ))
            print(f"  {width:>5}px  {path.name:<34} "
                  f"{'FIRE ' if verdict.triggered else 'quiet'} "
                  f"{verdict.confidence:.2f}  {verdict.reason[:96]}")
    return result


def report(result: Result) -> str:
    lines = [f"model: {result.model}", ""]
    lines.append(f"{'decode width':>12}  {'sent':>11}  {'fired':>5}  {'agree':>5}")
    for width, probes in result.by_width().items():
        sent = probes[0].crop_px
        fired = sum(1 for item in probes if item.triggered)
        agreed = [item.agrees for item in probes if item.agrees is not None]
        agreement = f"{sum(agreed)}/{len(agreed)}" if agreed else "—"
        lines.append(
            f"{width:>10}px  {sent[2]:>4}x{sent[3]:<4}  "
            f"{fired:>2}/{len(probes):<2}  {agreement:>5}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="the question, as a literal")
    source.add_argument("--prompt-file", help="a file holding the question")
    source.add_argument("--policy", help="read the prompt from this policy in the database")
    parser.add_argument("--video", required=True, help="source video to cut frames from")
    parser.add_argument("--ts", required=True, help="comma-separated offsets in seconds")
    parser.add_argument("--crop", required=True,
                        help="zone rectangle x1,y1,x2,y2 in 0-1 fractions")
    parser.add_argument("--at", default="2560,960",
                        help="comma-separated decode widths to simulate "
                             "(default: 2560,960 — the camera and the engine)")
    parser.add_argument("--expect", type=lambda v: v.lower() in ("1", "true", "yes"),
                        default=None, help="the answer you would give, for an agreement column")
    # Production grows any crop side under MIN_CROP_PX, because a 140x90 cut-out
    # of a 960x544 frame is fewer pixels of the zone than the wide shot already
    # had. That is right in service and wrong for *this* measurement: a
    # rectangle grown to 320 px at both widths shows more surrounding context at
    # the smaller one, so the two questions differ by more than resolution.
    # Pass 0 to see the rectangle itself and isolate the pixel count.
    parser.add_argument("--min-crop-px", type=int, default=MIN_CROP_PX,
                        help=f"grow crop sides to at least this many pixels "
                             f"(default: {MIN_CROP_PX}, what argus run does; "
                             f"0 = the rectangle exactly)")
    parser.add_argument("--work", default=str(ROOT / "data" / "live" / "probe"),
                        help="where to cut source frames")
    parser.add_argument("--json", default=None, help="write the raw probes here")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("-c", "--config", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = asyncio.run(probe(args))
    print()
    print(report(result))
    if args.json:
        Path(args.json).write_text(json.dumps({
            "model": result.model, "prompt": result.prompt,
            "probes": [item.as_dict() for item in result.probes],
        }, indent=2) + "\n")
        print(f"\njson: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
