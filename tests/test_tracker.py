"""The tracker gets what ByteTrack is built on, and a flicker is not a person.

Until 2026-09-19 nothing in this suite touched ``argus/vision/tracker.py``, which
is how the engine ran for six weeks with ByteTrack's second association pass
receiving an empty list: the detector cut every box at 0.5 before the tracker saw
it. These tests pin the mechanism rather than supervision's internals — a worker
whose score dips keeps their identity *only if* the doubtful boxes arrive, and a
doubtful box on its own never becomes a person.

No GPU and no model: detections are synthetic, and the detector's floor logic is
exercised through ``_postprocess`` on CPU tensors.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from argus.config import ConfigError, DetectionConfig, ModelConfig, TrackerConfig
from argus.vision.detector import PersonDetector
from argus.vision.tracker import CameraTracker, TrackerSettings

FPS = 5.0


def person(score: float, x: float = 100.0) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[x, 100.0, x + 60.0, 280.0]], dtype=np.float32),
        confidence=np.array([score], dtype=np.float32),
        class_id=np.array([0]),
    )


def nobody() -> sv.Detections:
    return sv.Detections.empty()


def ids(tracked: sv.Detections) -> list[int]:
    return [] if tracked.tracker_id is None else [int(i) for i in tracked.tracker_id]


def run(tracker: CameraTracker, frames: list[sv.Detections]) -> list[list[int]]:
    return [ids(tracker.update(frame)) for frame in frames]


# --- what the defaults say ---------------------------------------------------


def test_defaults_confirm_a_track_before_reporting_it():
    assert TrackerSettings().min_consecutive_frames == 3
    assert TrackerConfig().min_consecutive_frames == 3
    assert 0.0 < ModelConfig().track_floor < ModelConfig().threshold


# --- a flicker is not a person ------------------------------------------------


def test_a_single_frame_box_is_never_reported():
    tracker = CameraTracker("cam", FPS)
    seen = run(tracker, [person(0.9), nobody(), nobody(), nobody()])
    assert seen == [[], [], [], []]


def test_a_person_is_reported_once_confirmed_and_keeps_one_id():
    tracker = CameraTracker("cam", FPS)
    seen = run(tracker, [person(0.9)] * 6)
    assert seen[0] == [] and seen[1] == []          # not yet: 0.6 s of evidence first
    confirmed = [frame for frame in seen if frame]
    assert confirmed, "a steady detection must become a track"
    assert len({tuple(frame) for frame in confirmed}) == 1


def test_at_one_frame_the_old_behaviour_reported_a_flicker_as_a_head():
    """The regression this file exists for, stated as the setting that caused it."""
    eager = CameraTracker("cam", FPS, TrackerSettings(min_consecutive_frames=1))
    # supervision's own first-frame rule aside, a two-frame blip is enough.
    seen = run(eager, [person(0.9), person(0.9), nobody()])
    assert any(seen), "min_consecutive_frames=1 reports a blip"
    patient = CameraTracker("cam", FPS)
    assert not any(run(patient, [person(0.9), person(0.9), nobody()]))


# --- a doubtful box continues a track, and never starts one ---------------------


def test_a_low_score_box_alone_never_starts_a_track():
    tracker = CameraTracker("cam", FPS)
    assert not any(run(tracker, [person(0.2)] * 10))


# Three seconds: longer than `lost_track_sec` (2.0), so the two tests below differ
# in exactly one thing — whether the doubtful boxes were allowed to arrive.
DIP = 15


def test_a_score_dip_keeps_the_identity_when_the_doubtful_boxes_arrive():
    """A worker turns away: 0.9 -> 0.2 for three seconds -> 0.9 again."""
    frames = [person(0.9)] * 5 + [person(0.2)] * DIP + [person(0.9)] * 5
    seen = run(CameraTracker("cam", FPS), frames)
    confirmed = [frame for frame in seen if frame]
    assert {tuple(frame) for frame in confirmed} == {tuple(confirmed[0])}, "one person, one id"
    # ...and they were *present* through the dip, which is what a headcount needs.
    assert all(seen[5:5 + DIP]), "the track must be reported while the score is low"


def test_the_same_dip_loses_the_person_when_the_detector_withholds_those_boxes():
    """What the engine did until 2026-09-19: everything <= 0.5 never arrived."""
    withheld = [person(0.9)] * 5 + [nobody()] * DIP + [person(0.9)] * 5
    seen = run(CameraTracker("cam", FPS), withheld)
    assert not any(seen[6:5 + DIP]), "nobody is counted during the dip"
    before = {i for frame in seen[:5] for i in frame}
    after = {i for frame in seen[5 + DIP:] for i in frame}
    # Past `lost_track_sec` the track is gone, so the worker comes back as
    # somebody new — which is how one camera issued 414,163 ids.
    assert before and after and before.isdisjoint(after)


# --- the detector's half ------------------------------------------------------


def _detector(**kwargs) -> PersonDetector:
    detector = PersonDetector(trace=False, **kwargs)
    detector.person_class_index = 0
    return detector


def _postprocess(detector: PersonDetector, scores: list[float]):
    torch = pytest.importorskip("torch")
    count = len(scores)
    # Well-separated boxes, so the NMS guard has nothing to say.
    centres = torch.linspace(0.1, 0.9, count)
    boxes = torch.stack([centres, torch.full((count,), 0.5), torch.full((count,), 0.05),
                         torch.full((count,), 0.3)], dim=-1)[None]
    logits = torch.logit(torch.tensor(scores, dtype=torch.float32))[None, :, None]
    (xyxy, confidence, _), = detector._postprocess(boxes, logits, [np.zeros((544, 960, 3), np.uint8)])
    return sorted(round(float(c), 2) for c in confidence)


def test_the_detector_emits_doubtful_boxes_only_when_asked():
    scores = [0.05, 0.2, 0.45, 0.6, 0.9]
    assert _postprocess(_detector(threshold=0.5), scores) == [0.6, 0.9]
    assert _postprocess(_detector(threshold=0.5, track_floor=0.15), scores) == [0.2, 0.45, 0.6, 0.9]


def test_a_floor_at_or_above_the_threshold_changes_nothing():
    assert _detector(threshold=0.5, track_floor=0.0).emit_floor == 0.5
    assert _detector(threshold=0.5, track_floor=0.5).emit_floor == 0.5
    assert _detector(threshold=0.5, track_floor=0.8).emit_floor == 0.5


def test_the_eval_harness_still_gets_the_floor_it_sets():
    """`argus detector-eval` lowers `threshold` to 0.01 to sweep a PR curve; a
    track floor above that must not quietly raise it back."""
    detector = _detector(threshold=0.5, track_floor=0.15)
    detector.threshold = 0.01
    assert detector.emit_floor == 0.01


def test_config_refuses_nonsense():
    with pytest.raises(ConfigError):
        DetectionConfig(model=ModelConfig(track_floor=1.5))
    with pytest.raises(ConfigError):
        DetectionConfig(tracker=TrackerConfig(min_consecutive_frames=0))
