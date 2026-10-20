"""Per-camera multi-object tracking (ByteTrack via ``supervision``, MIT).

Tracking is what turns "3 boxes" into "3 people, and that one has been in the
zone for 40 seconds". Everything downstream that reasons about *a person* rather
than *a detection* — dwell time, phone-use duration, inactivity — needs a stable
identity, so it is wired in now even though this phase only counts heads.

One tracker per camera, always: ByteTrack carries Kalman state and an id
counter, so sharing an instance across cameras would associate a worker on the
packing line with a worker on the loading bay the moment their boxes overlap in
image space. v1 got this right by accident (one YOLO instance per camera); here
it is explicit.

``lost_track_sec`` is expressed in *seconds*, not frames, because Argus runs
detection at 5 fps rather than supervision's assumed 30 — a buffer of "30
frames" would be six seconds on one camera and one on another.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass
from typing import Any

# supervision scales its lost-track buffer by frame_rate/30 internally, so a
# buffer expressed in seconds converts with this constant on both sides.
_SUPERVISION_REFERENCE_FPS = 30.0


@dataclass(slots=True)
class TrackerSettings:
    """Tuning for ByteTrack, in units that mean something at 5 fps."""

    # supervision starts a NEW track only at `activation_threshold + 0.1`, and
    # offers boxes scoring between 0.1 and `activation_threshold` to tracks the
    # confident boxes left unmatched — the "BYTE" in ByteTrack, and the reason a
    # worker who turns away and drops to 0.2 for a second keeps their identity.
    #
    # Until 2026-09-19 none of that ran here: the detector discarded everything
    # at or below 0.5 before this class saw it, so the second pass was handed an
    # empty list on every frame of every camera, and `min_consecutive_frames: 1`
    # reported a one-frame false box as a person. `anmol_packing_7` had issued
    # 414,163 track ids — about one a second for ~24 people. The detector now
    # emits down to `detection.model.track_floor` (argus/vision/detector.py).
    # Measured on two 60 s plant clips, same weights, 3 s-median headcount within
    # +/-2 of the hand count: Packing 4 89.5% -> 100%, Packing 7 46.6% -> 93.0%.
    # That is a hand count held constant over two clips — a direction, not a
    # product claim. docs/DETECTION_V2.md has the sweep.
    activation_threshold: float = 0.3   # new tracks need this + 0.1; below it, continue-only
    match_threshold: float = 0.8        # IoU distance for associating to an existing track
    lost_track_sec: float = 2.0         # how long a track survives an occlusion
    min_consecutive_frames: int = 3     # frames before a track is reported (0.6 s at 5 fps)

    def lost_track_buffer(self) -> int:
        return max(1, round(self.lost_track_sec * _SUPERVISION_REFERENCE_FPS))


class CameraTracker:
    """ByteTrack for one camera, with the frame rate it is actually fed."""

    def __init__(self, camera_id: str, fps: float, settings: TrackerSettings | None = None):
        import supervision as sv

        self.camera_id = camera_id
        self.fps = float(fps)
        self.settings = settings or TrackerSettings()
        with warnings.catch_warnings():
            # supervision deprecated ByteTrack in 0.28 and removes it in 0.31 in
            # favour of ByteTrackTracker in the separate `trackers` package —
            # whose only new hard dependency is OpenCV, which v2 otherwise does
            # not need. pyproject pins supervision <0.31 for exactly this
            # reason; the migration is `update_with_detections()` -> `update()`
            # and lives entirely inside this class. Warning suppressed so it is
            # not printed once per camera on every start.
            warnings.filterwarnings("ignore", message=".*ByteTrack.*", category=FutureWarning)
            self._tracker = sv.ByteTrack(
                track_activation_threshold=self.settings.activation_threshold,
                lost_track_buffer=self.settings.lost_track_buffer(),
                minimum_matching_threshold=self.settings.match_threshold,
                frame_rate=self.fps,
                minimum_consecutive_frames=self.settings.min_consecutive_frames,
            )
        self.updates = 0

    def update(self, detections: Any) -> Any:
        """Attach ``tracker_id`` to this frame's detections.

        Note that ByteTrack drops detections it will not track (below the
        activation threshold, or not yet confirmed), so the returned set can be
        smaller than the input — which is the desired behaviour for counting:
        a flickering half-detection should not add a person to a zone.
        """
        self.updates += 1
        return self._tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Forget all state — used when a camera reconnects after a long gap."""
        self._tracker.reset()


class TrackerPool:
    """Lazily-created trackers keyed by camera id."""

    def __init__(self, fps: float, settings: TrackerSettings | None = None):
        self.fps = fps
        self.settings = settings or TrackerSettings()
        self._trackers: dict[str, CameraTracker] = {}
        self._lock = threading.Lock()

    def for_camera(self, camera_id: str, fps: float | None = None) -> CameraTracker:
        with self._lock:
            tracker = self._trackers.get(camera_id)
            if tracker is None:
                tracker = CameraTracker(camera_id, fps or self.fps, self.settings)
                self._trackers[camera_id] = tracker
            return tracker

    def drop(self, camera_id: str) -> None:
        with self._lock:
            self._trackers.pop(camera_id, None)

    def reset(self, camera_id: str) -> None:
        with self._lock:
            tracker = self._trackers.get(camera_id)
        if tracker is not None:
            tracker.reset()

    def __len__(self) -> int:
        return len(self._trackers)


def tracker_settings_from_config(config: Any) -> TrackerSettings:
    tracker = config.detection.tracker
    return TrackerSettings(
        activation_threshold=tracker.activation_threshold,
        match_threshold=tracker.match_threshold,
        lost_track_sec=tracker.lost_track_sec,
        min_consecutive_frames=tracker.min_consecutive_frames,
    )


__all__ = ["CameraTracker", "TrackerPool", "TrackerSettings", "tracker_settings_from_config"]
