"""Detector smoke test: does RF-DETR actually find the people in a real frame?

Marked ``gpu``/``slow`` and skipped when the model cannot be loaded (no rfdetr,
no cached weights and no network). Everything else in the suite runs without it
— the engine tests use a fake detector — so a machine without a GPU still gets a
meaningful test run.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

MIN_EXPECTED_PEOPLE = 3     # the demo frame has ~8; 3 is "the model is working"


@pytest.fixture(scope="module")
def frame(tmp_path_factory) -> np.ndarray:
    """One 960x544 BGR frame from the demo recording, as the decoder emits it."""
    from argus.config import ROOT

    video = ROOT / "packing_section_recording.mp4"
    if not video.exists():
        pytest.skip(f"sample video missing: {video}")
    raw = tmp_path_factory.mktemp("frame") / "frame.raw"
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "12", "-i", str(video),
         "-frames:v", "1", "-vf", "scale=960:544", "-pix_fmt", "bgr24", "-f", "rawvideo", str(raw)],
        capture_output=True, check=False,
    )
    if result.returncode != 0 or not raw.exists():
        pytest.skip("ffmpeg could not extract a test frame")
    return np.frombuffer(raw.read_bytes(), dtype=np.uint8).reshape(544, 960, 3)


@pytest.fixture(scope="module")
def detector():
    pytest.importorskip("rfdetr")
    pytest.importorskip("supervision")
    from argus.vision.detector import PersonDetector

    instance = PersonDetector(variant="nano", precision="fp16", threshold=0.5)
    try:
        instance.load()
    except Exception as exc:  # noqa: BLE001 - no weights cached and no network
        pytest.skip(f"could not load RF-DETR: {exc}")
    return instance


def test_finds_people_in_the_demo_frame(detector, frame):
    detections = detector.detect([frame])[0]
    assert len(detections) >= MIN_EXPECTED_PEOPLE
    assert (detections.confidence > 0.5).all()
    assert set(detections.class_id.tolist()) == {0}


def test_boxes_are_in_source_pixels_and_inside_the_frame(detector, frame):
    detections = detector.detect([frame])[0]
    height, width = frame.shape[:2]
    xyxy = detections.xyxy
    assert (xyxy[:, 0] >= 0).all() and (xyxy[:, 1] >= 0).all()
    assert (xyxy[:, 2] <= width).all() and (xyxy[:, 3] <= height).all()
    assert (xyxy[:, 2] > xyxy[:, 0]).all() and (xyxy[:, 3] > xyxy[:, 1]).all()
    # A person box should be a sane fraction of a 960x544 frame, not the whole
    # thing and not a few pixels.
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) / (width * height)
    assert (areas > 0.0005).all() and (areas < 0.6).all()


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return intersection / (area_a[:, None] + area_b[None, :] - intersection)


def _assert_same_people(left, right, *, min_confidence: float = 0.6, min_iou: float = 0.9) -> int:
    """Every person either result is confident about must appear in the other.

    Not an equality assertion, deliberately. FP16 reductions depend on the batch
    size and on whether the graph is traced, so scores move by ~0.003 between
    runs and a detection sitting on the threshold can flip in or out. Those are
    exactly the flickers ByteTrack's minimum-consecutive-frames rule absorbs
    before anything is counted. What must not move is the confident set.
    """
    confident = left.xyxy[left.confidence >= min_confidence]
    if len(confident) == 0:
        return 0
    overlaps = _iou(confident, right.xyxy)
    assert (overlaps.max(axis=1) >= min_iou).all(), (
        f"confident detections not reproduced: best IoU {overlaps.max(axis=1)}"
    )
    return len(confident)


def test_batching_is_equivalent_to_one_at_a_time(detector, frame):
    """The core claim of the batched design: batching changes speed, not answers."""
    single = detector.detect([frame])[0]
    batched = detector.detect([frame] * 4)
    assert len(batched) == 4
    for result in batched:
        assert _assert_same_people(single, result) >= MIN_EXPECTED_PEOPLE
        _assert_same_people(result, single)


def test_empty_input_is_not_an_error(detector):
    assert detector.detect([]) == []


def test_traced_and_eager_agree(frame):
    """The jit trace is an optimisation; it must not move a box."""
    pytest.importorskip("rfdetr")
    from argus.vision.detector import PersonDetector

    eager = PersonDetector(variant="nano", precision="fp16", trace=False)
    traced = PersonDetector(variant="nano", precision="fp16", trace=True)
    try:
        eager_result = eager.detect([frame])[0]
        traced_result = traced.detect([frame])[0]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not load RF-DETR: {exc}")
    assert _assert_same_people(eager_result, traced_result) >= MIN_EXPECTED_PEOPLE
    _assert_same_people(traced_result, eager_result)
    assert traced.describe()["backend"].endswith("traced")


def test_describe_reports_what_ran(detector, frame):
    detector.detect([frame])
    described = detector.describe()
    assert described["model"] == "rf-detr-nano"
    assert described["resolution"] > 0
    assert described["frames"] >= 1
    assert described["ms_per_frame"] > 0
