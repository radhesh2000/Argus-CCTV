"""The Argus engine process — decode, detect, track, judge, prove.

One process owns the whole live path:

    gateway (one connection per camera)
        -> decoder threads          (ffmpeg -> BGR frames at detection fps)
        -> vision loop              (ONE batched RF-DETR pass across cameras)
        -> analytics pool           (per-camera, ordered; see vision/analytics.py)
             -> per-camera ByteTrack     (stable ids)
             -> zone occupancy           (feet-in-polygon vs sanctioned limit)
             -> policy evaluation        (debounced, per policy cooldown)
             -> evidence                 (annotated snapshot now, ±8s clip shortly after)

Three structural choices are worth stating, because they are what separates this
from v1:

* **The vision loop is one thread, not one per camera.** Detection happens on
  whatever frames are fresh, in a single forward pass. v1 held a model per
  camera and ran them independently, which spent GPU launch overhead per camera
  and scaled linearly with the fleet; here 8 cameras cost 10 ms rather than 8 x
  3 ms, and adding cameras mostly adds decode.
* **Nothing blocks the decoders.** A decoder thread hands its newest frame to a
  slot and goes back to reading; if analytics fall behind, the *older* frame is
  dropped rather than the pipe backing up into ffmpeg. Falling behind therefore
  degrades sampling rate, never liveness.
* **Analytics can run beside detection — and measurably should not, here.**
  Everything downstream of the forward pass can be handed to a pool that
  preserves per-camera order (``detection.analytics_workers``), and on this box
  that is 9% *slower* than doing it inline, because 64 decoder threads already
  fill every GIL gap the forward pass opens. The pool ships off. The
  measurements, and the conditions under which it would start paying, are in
  ``argus/vision/analytics.py`` and docs/BENCHMARKS.md.

``argus run`` is this module.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from argus.config import ArgusConfig
from argus.db import Database, Event, decoded_meta, scope_allows
from argus.ingest.decoder import Frame, FrameSource, FrameSourceThread, source_for_camera
from argus.ingest.gateway import Gateway
from argus.live import LiveFrames
from argus.media.annotate import (
    SEVERITY_COLORS,
    annotate_zone_event,
    hex_to_rgb,
    save_snapshot,
)
from argus.media.ring import RingRecorder
from argus.notify import Notifier
from argus.redact import redact_source
from argus.retention import Retention
from argus.vision.analytics import AnalyticsPool
from argus.vision.debounce import Debouncer
from argus.vision.detector import Detector, detector_from_config
from argus.vision.tracker import TrackerPool, tracker_settings_from_config
from argus.vision.zone_state import OPEN, ZoneState, ZoneVerdict, describe_duration
from argus.vision.zones import ZoneSet

# The detector key this phase implements. VLM policies are Phase 2; PPE and the
# rest of the roadmap are Phase 6.
HEADCOUNT_DETECTOR_KEY = "manpower_headcount"


# How long `stop()` will wait, in total, for evidence clips already being cut.
# One NVENC pass on a 16 s 1440p clip measures 2.2 s here, so this is roughly
# "let the handful in flight finish" and not "cut the backlog". Clips whose
# timer had not fired are cancelled outright — they were always going to be,
# and their event row simply carries no clip_path, which is honest.
#
# It has to fit inside the unit's `TimeoutStopSec=45` (argus/services.py), and
# it matters *because* of systemd: with no `KillMode=` the default is
# control-group, so every ffmpeg the engine started is SIGTERMed with it. Run
# from a shell the child is merely orphaned and finishes anyway, which is why
# this was invisible until a clip took seconds instead of milliseconds.
CLIP_DRAIN_SEC = 12.0

# How long the vision loop lingers after the first camera reports a frame, to
# let its siblings' frames arrive and share the forward pass. Small enough to be
# invisible at 5 fps, big enough to coalesce cameras that are nearly in phase.
#
# It is waited out in slices and abandoned the moment enough cameras have landed
# to fill a batch. On a small fleet that changes nothing; on a large one it is
# the difference between a 25 ms pause per cycle and none, because at 64 cameras
# 320 frames arrive every second and the batch is full before the loop has
# finished asking. The flat wait cost roughly a third of the cycle there.
_BATCH_GRACE_SEC = 0.025
_BATCH_GRACE_SLICE_SEC = 0.004

# How often cameras, zones and policies are re-read from the database, so an
# operator editing a zone does not have to restart the engine.
_CONFIG_REFRESH_SEC = 15.0

# Smoothing for the per-camera capture-to-processed lag. Same 10-frame memory as
# the decoder's fps EWMA, so the two numbers describe the same window.
_LAG_ALPHA = 0.1

# How often the stats file is republished. Decoupled from the printed report
# because the file now carries the live wall's zone counts and track boxes:
# a console line every ten seconds is plenty, an overlay ten seconds behind the
# video is not. One small atomic write per tick.
_STATS_WRITE_SEC = 2.0


def normalized_boxes(
    xyxy: Any, tracker_ids: Any, confidence: Any, width: int, height: int
) -> list[list[float]]:
    """Detection boxes as ``[x1, y1, x2, y2, track_id, confidence]``, normalized.

    Normalized because nothing downstream sees the 960x544 the detector did:
    an evidence overlay lands on the full-resolution clip, and the live wall
    draws on a WebRTC stream at whatever the camera's own resolution is. A
    track the tracker has not named yet is ``-1`` rather than absent, so the
    row is always six numbers wide.
    """
    return [
        [round(float(x1) / width, 4), round(float(y1) / height, 4),
         round(float(x2) / width, 4), round(float(y2) / height, 4),
         int(tid) if tid is not None else -1, round(float(conf), 3)]
        for (x1, y1, x2, y2), tid, conf in zip(xyxy, tracker_ids, confidence)
    ]


def condition_key(policy_id: str, camera_id: str, zone_id: str) -> str:
    """The identity of a detector-origin ongoing condition.

    Prefixed ``det:`` because ``events.condition_key`` is one namespace shared
    with the VLM scheduler, whose keys are ``{policy}:{camera}[:{zone}]``. An
    unprefixed detector key would collide with a *zone-scoped prompt policy* of
    the same id on the same zone, and the two would then extend each other's
    rows — which is not a hypothetical, because the verification pass already
    points one kind of policy at the other's events.
    """
    return f"det:{policy_id}:{camera_id}:{zone_id}"


@dataclass
class OpenCondition:
    """What the engine remembers about a condition row it is keeping alive.

    Bookkeeping, not truth: the row in ``events`` is the condition, and every
    field here is a cache of it that exists so the steady state — five frames a
    second per zone, for hours — costs a dict lookup instead of a SELECT. It is
    rebuilt from the row whenever the cache is missing, which is what makes an
    engine restart adopt rather than duplicate.
    """

    event_id: int
    opened_ts: float
    snapshot_path: str | None
    #: Frame timestamp of the last in-place update, so the cadence can be kept.
    #: 0.0 on adoption, so the first frame after a restart writes at once and
    #: proves the row is being observed again.
    last_write_ts: float = 0.0
    #: The peak already written to the row; a rise beats the cadence.
    peak: int = 0
    observations: int = 1
    #: What was true when the condition started. Carried because `description`
    #: is overwritten with the newest sentence on every update, and "how bad was
    #: it when this began" is the first question asked of a long-running card.
    opened_count: int = 0
    opened_limit: int | None = None


@dataclass
class CameraRuntime:
    """Everything the engine keeps per camera."""

    camera: dict[str, Any]
    zones: ZoneSet
    source: FrameSource
    worker: FrameSourceThread | None = None
    ring: RingRecorder | None = None
    debouncer: Debouncer = field(default_factory=Debouncer)
    # Ongoing conditions (detection.continuity). One smoothed, hysteretic state
    # machine per (policy, zone) on this camera — see argus/vision/zone_state.py
    # — and, for the ones currently in breach, the row each is extending.
    zone_states: dict[str, ZoneState] = field(default_factory=dict)
    conditions: dict[str, OpenCondition] = field(default_factory=dict)
    status: str = "connecting"
    events: int = 0
    frames_processed: int = 0
    # Detected frames the analytics pool could not keep up with, so an older one
    # for this camera was discarded. Published because it is the difference
    # between "the detector never saw it" and "it was seen and then dropped".
    frames_dropped: int = 0
    last_frame_at: float | None = None
    last_status_write: float = 0.0
    # Seconds between a frame leaving the decoder and the analytics finishing
    # with it — smoothed, and the worst seen. This is the number that says
    # whether the box is keeping up: achieved fps can look healthy while the
    # vision loop works further and further behind the live edge.
    lag_ms: float = 0.0
    lag_ms_peak: float = 0.0
    # What the last analysed frame looked like: zone counts and normalized
    # track boxes. Published in the stats file so the live wall can draw the
    # overlay the operator would otherwise only ever see on an event snapshot.
    # Kept as the latest sample rather than a history — a wall showing what was
    # true ten frames ago is a wall nobody trusts.
    live_occupancy: list[dict[str, Any]] = field(default_factory=list)
    live_tracks: list[list[float]] = field(default_factory=list)
    live_ts: float | None = None
    # The pixels of that same frame, held as a reference for the live-frame
    # publisher (argus/live.py) — the API process cannot see this one's memory,
    # so a prompt test needs a JPEG on disk to look at. A reference, not a copy:
    # the array already exists and the vision loop is finished with it.
    live_image: Any | None = None
    _pending: Frame | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def id(self) -> str:
        return str(self.camera["id"])

    @property
    def name(self) -> str:
        return str(self.camera.get("name") or self.camera["id"])

    def offer(self, frame: Frame) -> None:
        """Publish the newest frame, discarding any the vision loop never took."""
        with self._lock:
            self._pending = frame
        self.last_frame_at = frame.ts

    def waiting(self) -> bool:
        """Whether a frame is sitting in the slot. Read without the lock on
        purpose: this only steers how long the loop lingers for stragglers, and
        one attribute read per camera per slice has to stay free."""
        return self._pending is not None

    def take(self) -> Frame | None:
        with self._lock:
            frame, self._pending = self._pending, None
        return frame

    def record_lag(self, seconds: float) -> None:
        milliseconds = seconds * 1000.0
        if self.lag_ms == 0.0:
            self.lag_ms = milliseconds
        else:
            self.lag_ms += _LAG_ALPHA * (milliseconds - self.lag_ms)
        self.lag_ms_peak = max(self.lag_ms_peak, milliseconds)


class Engine:
    def __init__(
        self,
        config: ArgusConfig,
        database: Database,
        *,
        use_gateway: bool = True,
        direct: bool = False,
        detector: Detector | None = None,
        record: bool | None = None,
        notify: bool = True,
        housekeeping: bool = True,
        vlm: bool | None = None,
        index: bool | None = None,
    ):
        self.config = config
        self.db = database
        self.use_gateway = use_gateway
        self.direct = direct
        self.record = config.evidence.enabled if record is None else record
        self.detector: Detector = detector or detector_from_config(config)
        self.trackers = TrackerPool(config.detection.fps, tracker_settings_from_config(config))
        # Alert delivery and housekeeping live inside `argus run` rather than in
        # a separate service: an appliance whose retention depends on somebody
        # remembering to install a cron job is an appliance that fills its disk.
        self.notifier = Notifier(config, database) if notify else None
        self.retention = Retention(config, database) if housekeeping else None
        # Phase 2: prompt policies judged by a local VLM, on their own thread.
        # Imported here rather than at module scope so `argus run` with the VLM
        # disabled never pays for the import.
        self.vlm: Any | None = None
        if config.vlm.enabled if vlm is None else vlm:
            from argus.vlm.scheduler import VlmRunner

            self.vlm = VlmRunner(
                config, database, notifier=self.notifier, on_event=self._schedule_clip_for,
            )

        # Phase 3: SigLIP 2 embeddings of the frames this loop already decoded,
        # written to the local vector store so they can be searched in English.
        # Imported here for the same reason the VLM is: a run with the index off
        # never pays for transformers.
        self.index: Any | None = None
        if config.index.enabled if index is None else index:
            from argus.index.runner import IndexRunner

            self.index = IndexRunner(config, log=print)

        # Everything downstream of the forward pass, optionally off the vision
        # thread. `analytics_workers: 0` — the shipped default, and what the
        # tests that call `_process` directly exercise — keeps it inline.
        detection = config.detection
        self.analytics: AnalyticsPool | None = None
        if detection.analytics_workers > 0:
            self.analytics = AnalyticsPool(
                detection.analytics_workers,
                self._process,
                queue_depth=detection.analytics_queue_depth,
                on_error=self._analytics_failed,
                name="analytics",
            )

        self.gateway: Gateway | None = None
        self.runtimes: dict[str, CameraRuntime] = {}
        self.policies: list[dict[str, Any]] = []
        self.started_at: float = 0.0

        self._stop = threading.Event()
        self._frame_ready = threading.Event()
        self._vision_thread: threading.Thread | None = None
        self._occupancy: dict[tuple[str, str], dict[str, Any]] = {}
        self._occupancy_lock = threading.Lock()
        self._last_occupancy_flush = 0.0
        self._last_refresh = 0.0
        self._last_report = 0.0
        self._last_stats_write = 0.0
        self._stats_write_failed = False
        # Ongoing conditions (detection.continuity). `conditions_opened` counts
        # incidents and `condition_observations` counts sightings; on a plant
        # where a zone stays busy the second climbs and the first does not,
        # which is the whole shape of this change and the number to watch after
        # it ships. `conditions_adopted` is how often a restart continued a row
        # instead of filing a duplicate.
        self.conditions_opened = 0
        self.conditions_closed = 0
        self.conditions_adopted = 0
        self.condition_observations = 0
        # Phase 5 (authoring): the frame Policy Studio's prompt test looks at.
        # Costs nothing until the API asks — see argus/live.py.
        self.live = LiveFrames.from_config(config)
        self._live_publish_failed = False
        self._stopped = False
        self._timers: set[threading.Timer] = set()
        self._timers_lock = threading.Lock()

    # --- setup ------------------------------------------------------------

    def _source_url(self, camera: dict[str, Any]) -> str:
        """Where this camera's pixels come from.

        Through the gateway by default: go2rtc holds the single connection to
        the physical camera and fans it out, so the decoder and the ring
        recorder together do not double the load on the camera.
        """
        if self.use_gateway and not self.direct:
            return self.config.rtsp_url(camera["id"])
        return str(camera["source"])

    def _build_runtime(self, camera: dict[str, Any], fleet: int = 1) -> CameraRuntime:
        url = self._source_url(camera)
        source = source_for_camera(
            camera, self.config, override_source=url,
            start_timeout_sec=self.config.detection.start_timeout_for(fleet),
        )
        zones = ZoneSet.from_rows(self.db.list_zones(camera["id"]))
        runtime = CameraRuntime(camera=camera, zones=zones, source=source)
        if self.record:
            evidence = self.config.evidence
            runtime.ring = RingRecorder(
                camera["id"], url, self.config.ring_dir / camera["id"],
                segment_sec=evidence.ring_segment_sec,
                retention_sec=evidence.ring_retention_sec,
                mode=evidence.ring_mode,
                clip_transcode=evidence.clip_transcode,
            )
        return runtime

    def start(self) -> None:
        self.config.ensure_dirs()
        cameras = self.db.list_cameras(enabled_only=True)
        if not cameras:
            raise RuntimeError("no enabled cameras — run `argus seed` first")

        self._refresh_policies()
        if not self.policies:
            print("policies: no active manpower_headcount policy — occupancy will be "
                  "recorded but no events will be raised")

        # The detector, before the first camera. It would otherwise build itself
        # inside the vision thread on the first batch, which is the wrong place
        # for it to fail: a daemon thread that dies leaves an engine that runs
        # forever, decodes every camera and finds nobody. Weights that will not
        # load should stop `argus run` in its first second and say why — which
        # matters most for the one case where they might not, a fine-tuned
        # checkpoint whose provenance no longer matches it.
        #
        # It is also printed unconditionally, stock or not. The risk of an
        # optional checkpoint is a site running one nobody knows about, or
        # assuming one that was never switched on.
        loader = getattr(self.detector, "load", None)
        if callable(loader):
            loader()
        summary = getattr(self.detector, "weights_summary", None)
        if callable(summary):
            print(f"detector: {summary()}")

        if self.use_gateway:
            self.gateway = Gateway(self.config, cameras)
            if not self.gateway.available():
                raise FileNotFoundError(
                    f"go2rtc not found at {self.gateway.binary}. "
                    "Run scripts/fetch_gateway.sh (or pass --no-gateway)."
                )
            print(f"gateway: {self.gateway.version()} -> {self.config.gateway_url}")
            self.gateway.start()
            if self.gateway.wait_ready(timeout=20):
                print(f"gateway: ready, config at {self.gateway.config_path}")
            else:
                print("gateway: WARNING — API did not answer within 20s; continuing")

        if self.analytics is not None:
            self.analytics.start()
            print(f"analytics: {self.analytics.workers} worker(s), "
                  f"{self.analytics.queue_depth} frame(s) queued per camera")

        # Staggered, and the deadline widened for the fleet. Each camera costs
        # three ffmpeg processes (decoder, ring recorder, and the gateway's own
        # publisher when the first consumer connects), so starting 96 of them at
        # once is ~290 simultaneous process spawns and CUDA context creations.
        # That used to starve every decoder past its first-frame deadline, and
        # a starved hardware decode was read as proof the box had none.
        stagger = max(0.0, self.config.detection.start_stagger_sec)
        fleet = len(cameras)
        if stagger and fleet > 1:
            print(f"cameras: starting {fleet} with a {stagger * 1000:.0f} ms stagger "
                  f"({stagger * fleet:.1f}s), first frame due within "
                  f"{self.config.detection.start_timeout_for(fleet):.0f}s")
        for index, camera in enumerate(cameras):
            if index and stagger and self._stop.wait(stagger):
                break
            self.db.set_camera_status(camera["id"], "connecting")
            runtime = self._build_runtime(camera, fleet=fleet)
            runtime.worker = FrameSourceThread(runtime.source, on_frame=self._on_frame).start()
            if runtime.ring is not None:
                runtime.ring.start()
            self.runtimes[runtime.id] = runtime
            zones = len(runtime.zones)
            print(f"camera {runtime.id}: {runtime.source.kind} source, "
                  f"decoding from {redact_source(runtime.source.source)}, {zones} zone(s)"
                  + (", recording" if runtime.ring else ""))

        self.started_at = time.time()
        self._last_occupancy_flush = self.started_at
        self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True, name="vision")
        self._vision_thread.start()
        if self.vlm is not None:
            # Never fatal: a VLM that will not come up degrades the prompt
            # policies and leaves detection, tracking and alerting untouched.
            self.vlm.refresh()
            self.vlm.start()
            print(f"vlm: {len(self.vlm.schedules)} policy/camera schedule(s) against "
                  f"{self.config.vlm.endpoint}"
                  + (", verification pass on" if self.config.vlm.verify.enabled else ""))
        if self.index is not None:
            # Never fatal, exactly like the VLM: a missing checkpoint degrades
            # search and leaves detection, recording and alerting untouched.
            if self.index.start():
                described = self.index.stats()
                print(f"index: {described['model']} ({described['dim']}-d) on "
                      f"{described['device']} — sampling every "
                      f"{described['sample_interval_sec']}s per camera")
            else:
                print(f"index: NOT running — {self.index.reason}")
        if self.retention is not None:
            retention = self.config.retention
            # The evidence line first, and by tier when tiers are configured: an
            # operator reading this log to answer "what is this box doing with
            # my data" is asking about the clips, and "clips 14d" was a true
            # sentence about a config key rather than about the policy in force.
            evidence = retention.evidence
            if evidence.configured:
                windows = ", ".join(
                    f"{tier} {evidence.days(tier, retention.clips_days):g}d"
                    for tier in ("shadow", "reviewed", "auto_routed",
                                 "confirmed", "false_positive")
                )
                print(f"retention: evidence by stage — {windows}")
            else:
                print(f"retention: clips {retention.clips_days}d, "
                      f"snapshots {retention.snapshots_days}d")
            budgets = ", ".join(
                f"{kind} {retention.budget_gb(kind):g} GB"
                for kind in ("clips", "snapshots") if retention.budget_gb(kind) > 0
            )
            print(f"retention: budget {budgets}" if budgets
                  else "retention: NO evidence budget — ages and the disk guard only")
            print(f"retention: events {retention.events_days}d, "
                  f"occupancy {retention.occupancy_days}d, index {retention.index_days}d; "
                  f"disk guard at {retention.max_disk_percent}%")
            # The backup policy belongs in the same breath as the retention
            # policy: they run on one thread, and an operator reading this log
            # to answer "what is this box doing with my data" needs both.
            backup_config = self.config.backup
            if backup_config.enabled:
                print(f"backup:    every {backup_config.interval_hours}h, keeping "
                      f"{backup_config.keep}, in {self.config.backups_dir}")
            else:
                print("backup:    disabled by config — no database snapshots")
            self.retention.start()
        self._write_stats()

    # --- frame intake -----------------------------------------------------

    def _on_frame(self, frame: Frame) -> None:
        runtime = self.runtimes.get(frame.camera_id)
        if runtime is None:
            return
        runtime.offer(frame)
        if self.vlm is not None:
            # A second slot, not a second copy of the pipeline: the VLM samples
            # every few seconds and must not steal the frame the vision loop is
            # about to take. One dict assignment; see VlmRunner.offer.
            self.vlm.offer(frame)
        if self.index is not None:
            # A third reader of the same frame, and the cheapest: one dict
            # lookup and a float compare, taking the frame roughly once every
            # `index.sample_interval_sec`. This is what "reuse the frames the
            # engine already decodes" means in code — there is no second
            # ffmpeg anywhere in Phase 3.
            self.index.offer(frame)
        self._frame_ready.set()

    # --- vision loop ------------------------------------------------------

    def _wait_for_batch(self, max_batch: int) -> None:
        """Linger for stragglers, but only until the batch would be full.

        The grace period exists so cameras that are nearly in phase share one
        forward pass. Waiting it out unconditionally, as this did until
        2026-08-12, is dead time on any fleet big enough to have already filled
        the batch — which at 5 fps is anything above ``max_batch`` cameras.
        """
        deadline = time.monotonic() + _BATCH_GRACE_SEC
        runtimes = list(self.runtimes.values())
        while time.monotonic() < deadline:
            if sum(1 for runtime in runtimes if runtime.waiting()) >= max_batch:
                return
            if self._stop.wait(_BATCH_GRACE_SLICE_SEC):
                return

    def _vision_loop(self) -> None:
        interval = 1.0 / max(self.config.detection.fps, 0.1)
        max_batch = max(1, self.config.detection.model.max_batch)
        while not self._stop.is_set():
            woken = self._frame_ready.wait(timeout=interval)
            self._frame_ready.clear()
            if woken and len(self.runtimes) > 1:
                self._wait_for_batch(max_batch)
            batch = [(runtime, frame) for runtime in list(self.runtimes.values())
                     if (frame := runtime.take()) is not None]
            if not batch:
                continue
            for start in range(0, len(batch), max_batch):
                chunk = batch[start:start + max_batch]
                try:
                    results = self.detector.detect([frame.image for _, frame in chunk])
                except Exception as exc:  # noqa: BLE001 - one bad batch must not end the run
                    print(f"vision: detection failed ({type(exc).__name__}: {exc})", flush=True)
                    continue
                for (runtime, frame), detections in zip(chunk, results):
                    self._dispatch_analytics(runtime, frame, detections)
            self._maybe_flush_occupancy()

    def _dispatch_analytics(self, runtime: CameraRuntime, frame: Frame, detections: Any) -> None:
        """Hand one detected frame to the analytics pool, or run it here.

        The vision thread's next batch is worth more than this frame's tracking
        pass, so a full queue drops this camera's oldest pending frame rather
        than making the detector wait — the same rule the decoder's slot
        applies, one stage further down.
        """
        if self.analytics is None:
            try:
                self._process(runtime, frame, detections)
            except Exception as exc:  # noqa: BLE001
                self._analytics_failed(runtime.id, exc)
            return
        if not self.analytics.submit(runtime.id, runtime, frame, detections):
            runtime.frames_dropped += 1

    def _analytics_failed(self, camera_id: str, exc: BaseException) -> None:
        print(f"vision: {camera_id} analytics failed ({type(exc).__name__}: {exc})", flush=True)

    def _process(self, runtime: CameraRuntime, frame: Frame, detections: Any) -> None:
        """Track, count, judge and (if warranted) raise — for one frame."""
        tracked = self.trackers.for_camera(runtime.id, self.config.detection.fps).update(detections)
        runtime.frames_processed += 1
        runtime.record_lag(time.time() - frame.ts)
        height, width = frame.image.shape[:2]

        occupancy = runtime.zones.occupancy(tracked.xyxy, width, height)
        for zone_occupancy in occupancy:
            self._record_occupancy(runtime.id, zone_occupancy.zone.id, frame.ts, zone_occupancy.count)

        # The judgement is made first and the database work second, so the live
        # wall is published from the same frame even if a snapshot or a row
        # fails: advancing the state machines is pure arithmetic and cannot.
        judged = (
            self._advance_zone_states(runtime, frame, occupancy)
            if self.config.detection.continuity.enabled else None
        )
        self._record_live_state(runtime, frame, tracked, occupancy, width, height, judged)
        if judged is None:
            self._judge_per_frame(runtime, frame, tracked, occupancy)
            return
        for policy, zone_occupancy, verdict in judged:
            self._apply_zone_verdict(runtime, frame, tracked, zone_occupancy, policy, verdict)

    def _judge_per_frame(
        self, runtime: CameraRuntime, frame: Frame, tracked: Any, occupancy: Any
    ) -> None:
        """The pre-``detection.continuity`` path: one row per cooldown.

        Kept reachable behind the flag, and kept honest about what it does. It
        decides on a single frame's count, and it re-fires for as long as the
        zone stays over its limit — which on this appliance meant 112 151 rows.
        A site that wants one-alert-per-cooldown semantics can still have them;
        nobody should get them by accident.
        """
        for policy in self.policies:
            scope = policy.get("scope") or []
            for zone_occupancy in occupancy:
                if not zone_occupancy.over_limit:
                    continue
                # Where this policy applies (migration 5). An unscoped policy
                # has an empty list and this is always true, which is what makes
                # the feature invisible to an appliance that never used it.
                if not scope_allows(scope, runtime.id, zone_occupancy.zone.id):
                    continue
                key = f"{policy['id']}:{zone_occupancy.zone.id}"
                if not runtime.debouncer.should_emit(key, frame.ts, self._cooldown(policy)):
                    continue
                self._emit_zone_event(runtime, frame, tracked, zone_occupancy, policy)

    def _cooldown(self, policy: dict[str, Any]) -> float:
        cooldown = policy.get("cooldown_sec")
        return float(
            self.config.evidence.default_cooldown_sec if cooldown is None else cooldown
        )

    def _record_live_state(
        self, runtime: CameraRuntime, frame: Frame, tracked: Any, occupancy: Any,
        width: int, height: int,
        judged: list[tuple[dict[str, Any], Any, ZoneVerdict]] | None = None,
    ) -> None:
        """Keep the latest zone counts and boxes for the live wall.

        The alternative — having the UI poll the occupancy table — shows the
        peak of a ten-second flush window rather than what is on screen now,
        and has nothing at all to say about where the people are. This costs one
        list comprehension per analysed frame and rides out on the stats file
        the API already reads.

        **``over_limit`` is the smoothed answer, ``count`` is this frame's.**
        The tile is asking "is this zone over its limit", and the honest answer
        to that is not a property of one frame: with ~13% of boxes false
        positive and ~19% of people missed per frame (measured 2026-09-19), a
        raw comparison made every tile on a busy zone strobe amber. The median
        the policy is judged on cannot move on one bad frame, so that is what
        the border follows. The number printed on the tile stays instantaneous,
        because a wall that lags the video is a wall nobody trusts.

        A zone with a condition already open stays lit even if the median dips
        to the limit for a moment — the alert on the feed is still standing, and
        a tile that disagreed with the card beside it would be the worse of the
        two errors.
        """
        count = len(tracked.xyxy)
        confidence = tracked.confidence if tracked.confidence is not None else np.ones(count)
        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else [None] * count
        # A zone can be watched by more than one policy. The loudest verdict
        # wins the border: a tile that showed "clear" while one of its policies
        # was mid-alert would be contradicting the feed beside it.
        smoothed: dict[str, tuple[int, bool]] = {}
        for _policy, zone_occupancy, verdict in judged or ():
            zone_id = zone_occupancy.zone.id
            lit = verdict.over_limit or verdict.state == OPEN
            seen = smoothed.get(zone_id)
            smoothed[zone_id] = (
                (verdict.count, lit) if seen is None
                else (max(seen[0], verdict.count), seen[1] or lit)
            )
        runtime.live_occupancy = [
            {
                "zone_id": zone_occupancy.zone.id,
                "zone_name": zone_occupancy.zone.name,
                "count": int(zone_occupancy.count),
                # None when nothing is judging this zone (no policy in scope, or
                # the feature is off), which is different from "the median is 0".
                "smoothed_count": smoothed[zone_occupancy.zone.id][0]
                if zone_occupancy.zone.id in smoothed else None,
                "limit": zone_occupancy.zone.sanctioned_limit,
                "over_limit": bool(
                    smoothed[zone_occupancy.zone.id][1]
                    if zone_occupancy.zone.id in smoothed else zone_occupancy.over_limit
                ),
            }
            for zone_occupancy in occupancy
        ]
        runtime.live_tracks = normalized_boxes(tracked.xyxy, tracker_ids, confidence, width, height)
        runtime.live_ts = frame.ts
        runtime.live_image = frame.image

    # --- ongoing conditions (detection.continuity) -------------------------
    #
    # A zone over its sanctioned limit is a *state*, and the debounce module's
    # own docstring has always said so. What it did was rate-limit: a fresh row
    # every `cooldown_sec` for as long as the zone stayed busy. Measured on this
    # box on 2026-09-19 that was 10 958 events in 24 hours from one policy —
    # ~1 400 per camera per day, one a minute around the clock — and 112 151 in
    # total, against a review queue of 114 462 unreviewed rows.
    #
    # So the decision moves to `argus/vision/zone_state.py`, which smooths the
    # count over a short window and opens and closes on different clocks, and
    # the *row* becomes the condition: opened once, extended in place while it
    # lasts, closed when the zone clears. That is migration 7's machinery, used
    # for the origin its config comment claimed did not need it.
    #
    # `cooldown_sec` keeps the one job it was always right about: how soon a new
    # condition may open. It is stamped when a condition opens (the Debouncer's
    # own semantics) rather than when one closes, deliberately — the same
    # conservative reading the VLM scheduler documents, and here it costs even
    # less, because hysteresis already means a reopen needs `clear_sec` of quiet
    # followed by `sustain_sec` of breach before the cooldown is even consulted.

    def _advance_zone_states(
        self, runtime: CameraRuntime, frame: Frame, occupancy: Any
    ) -> list[tuple[dict[str, Any], Any, ZoneVerdict]]:
        """Feed this frame's counts to every (policy, zone) state machine here.

        Pure with respect to the database and the filesystem: it advances state
        and returns verdicts, and the caller decides what to write. That split
        is what lets the live wall be published before any I/O happens.
        """
        settings = self.config.detection.continuity
        judged: list[tuple[dict[str, Any], Any, ZoneVerdict]] = []
        live: set[str] = set()
        for policy in self.policies:
            scope = policy.get("scope") or []
            for zone_occupancy in occupancy:
                zone = zone_occupancy.zone
                if not scope_allows(scope, runtime.id, zone.id):
                    continue
                key = condition_key(str(policy["id"]), runtime.id, zone.id)
                live.add(key)
                state = runtime.zone_states.get(key)
                if state is None:
                    state = runtime.zone_states[key] = ZoneState(
                        window_sec=settings.window_sec,
                        sustain_sec=settings.sustain_sec,
                        clear_sec=settings.clear_sec,
                        release_margin=settings.release_margin,
                    )
                else:
                    # Config can be reloaded under a running engine; adopting the
                    # numbers without dropping the window means a retuned
                    # threshold does not also restart every condition on the box.
                    state.tune(
                        window_sec=settings.window_sec,
                        sustain_sec=settings.sustain_sec,
                        clear_sec=settings.clear_sec,
                        release_margin=settings.release_margin,
                    )
                judged.append((policy, zone_occupancy, state.observe(
                    frame.ts, zone_occupancy.count, zone.sanctioned_limit,
                )))
        # A zone deleted, or a policy that stopped applying here. Its machine is
        # dropped rather than left to resume a run from before the edit; any row
        # it had open is left to the stale sweep, which is the only thing that
        # can close a condition whose policy no longer exists.
        for gone in set(runtime.zone_states) - live:
            runtime.zone_states.pop(gone, None)
            runtime.conditions.pop(gone, None)
        return judged

    def _apply_zone_verdict(
        self, runtime: CameraRuntime, frame: Frame, tracked: Any, occupancy: Any,
        policy: dict[str, Any], verdict: ZoneVerdict,
    ) -> None:
        """Open, extend or close the row this verdict is about."""
        key = condition_key(str(policy["id"]), runtime.id, occupancy.zone.id)
        condition = runtime.conditions.get(key)
        if verdict.state == OPEN:
            if condition is None:
                # Either the condition opened on this frame, or this engine has
                # no memory of one it should have — a restart, or a cache
                # dropped by a policy edit. Both are the same question.
                condition = self._open_condition(
                    runtime, frame, tracked, occupancy, policy, verdict, key
                )
                if condition is not None:
                    runtime.conditions[key] = condition
            else:
                self._observe_zone_condition(
                    runtime, frame, tracked, occupancy, policy, verdict, key, condition
                )
        elif condition is not None:
            self._close_zone_condition(
                runtime, key, condition,
                closed_at=verdict.closed_at if verdict.closed_at is not None else frame.ts,
                reason="cleared",
            )

    def _open_condition(
        self, runtime: CameraRuntime, frame: Frame, tracked: Any, occupancy: Any,
        policy: dict[str, Any], verdict: ZoneVerdict, key: str,
    ) -> OpenCondition | None:
        """Adopt the row already on file for this key, or write a new one.

        **The lookup comes first, and that is the whole of restart adoption.**
        An appliance restarted mid-breach re-establishes the state five seconds
        later, finds its own open row under the same key and continues it — no
        second card on the feed, no second email, and a span that covers the
        outage rather than pretending the problem began when the process did.
        The same lookup is what makes the cache safe to drop for any reason.

        Returns None when the cooldown refuses a genuinely new condition. The
        state machine stays open, so this is retried on the next frame and the
        condition is filed the moment the cooldown allows — as opposed to being
        lost, which is what a `continue` in the old per-frame loop meant. Those
        retries repeat the lookup, at 5 fps, for as long as the cooldown lasts;
        that is one hit on the partial index over the handful of *open*
        conditions on the box (``idx_events_condition_open``), it is bounded by
        ``cooldown_sec``, and it is not the steady state — an established
        condition is a dict lookup and one UPDATE per ``update_sec``.
        """
        existing = self.db.open_condition(key)
        if existing is not None:
            meta = decoded_meta(existing) or {}
            opened = meta.get("opened") if isinstance(meta.get("opened"), dict) else {}
            adopted = OpenCondition(
                event_id=int(existing["id"]),
                opened_ts=float(existing["ts"]),
                snapshot_path=existing.get("snapshot_path"),
                # 0.0: write on the very next frame, so `condition_updated_at`
                # says "observed" again before the stale sweep next runs.
                last_write_ts=0.0,
                peak=max(int(meta.get("peak") or 0), verdict.peak),
                observations=int(existing.get("observations") or 1),
                opened_count=int(opened.get("count") or meta.get("count") or 0),
                opened_limit=opened.get("limit", meta.get("limit")),
            )
            self.conditions_adopted += 1
            print(f"condition {adopted.event_id}: {runtime.id} / {occupancy.zone.name} — "
                  f"adopted, still over limit after {adopted.observations} observation(s)",
                  flush=True)
            return adopted

        if not runtime.debouncer.should_emit(key, frame.ts, self._cooldown(policy)):
            return None
        event_id, snapshot_rel = self._emit_zone_event(
            runtime, frame, tracked, occupancy, policy, verdict=verdict, condition_key=key,
        )
        self.conditions_opened += 1
        return OpenCondition(
            event_id=event_id,
            opened_ts=frame.ts,
            snapshot_path=snapshot_rel,
            last_write_ts=frame.ts,
            peak=verdict.peak,
            observations=1,
            opened_count=verdict.count,
            opened_limit=verdict.limit,
        )

    def _observe_zone_condition(
        self, runtime: CameraRuntime, frame: Frame, tracked: Any, occupancy: Any,
        policy: dict[str, Any], verdict: ZoneVerdict, key: str, condition: OpenCondition,
    ) -> None:
        """One more sighting of a condition already on file.

        Three things are deliberately *not* done, for the reasons the VLM's
        ``_extend`` gives: no new row (the whole feature), no second alert (the
        recipient was told when it opened) and no second clip (the ±8 s clip is
        evidence of the onset, and re-cutting one every cadence would spend the
        ring buffer on the middle of a thing whose beginning is the part worth
        keeping). The snapshot *is* refreshed in place, over the same path, so
        one condition owns one JPEG however long it runs.

        Nor is it done every frame. At 5 fps that would be five UPDATEs a second
        per zone to restate a number that has not changed; the cadence is
        ``detection.continuity.update_sec``, and a rise in the peak jumps the
        queue because the peak is the one figure a reader cannot reconstruct
        from a later sample.
        """
        settings = self.config.detection.continuity
        elapsed = frame.ts - condition.last_write_ts
        # `not (0 <= elapsed < update_sec)` rather than `elapsed >= update_sec`:
        # looped demo footage and a camera whose clock restarts on reconnect
        # both hand us a frame older than the last write, and treating that as
        # "not due yet" would stop updating the row until the loop caught up —
        # by which time the stale sweep would have closed it.
        if 0.0 <= elapsed < settings.update_sec and verdict.peak <= condition.peak:
            return

        zone = occupancy.zone
        summary = self._condition_summary(zone.name, verdict, frame.ts - condition.opened_ts)
        if condition.snapshot_path:
            try:
                self._write_zone_snapshot(
                    runtime, frame, tracked, occupancy, policy, summary, condition.snapshot_path
                )
            except Exception as exc:  # noqa: BLE001 - the row is what matters
                print(f"condition {condition.event_id}: could not refresh evidence ({exc})",
                      flush=True)

        height, width = frame.image.shape[:2]
        confidence = (tracked.confidence if tracked.confidence is not None
                      else np.ones(len(occupancy.inside)))
        tracker_ids = (tracked.tracker_id if tracked.tracker_id is not None
                       else [None] * len(occupancy.inside))
        meta = self._event_meta(runtime, occupancy, tracked, tracker_ids, confidence,
                                width, height, verdict=verdict)
        meta["opened"] = {
            "ts": condition.opened_ts,
            "count": condition.opened_count,
            "limit": condition.opened_limit,
        }
        meta["ongoing_sec"] = round(max(0.0, frame.ts - condition.opened_ts), 1)

        row = self.db.observe_condition(
            condition.event_id,
            ts=frame.ts,
            description=summary,
            meta=meta,
            snapshot_path=condition.snapshot_path,
        )
        if row is None or row.get("condition_state") != "open":
            # Closed or deleted underneath us — a stale sweep with a short
            # window, a retention pass, an operator. Forget it rather than write
            # into a corpse; the next frame opens a fresh condition if the zone
            # is still over its limit.
            runtime.conditions.pop(key, None)
            return
        condition.last_write_ts = frame.ts
        condition.peak = verdict.peak
        # The count as the database now holds it: `observations = observations
        # + 1` happens in SQL, and a number computed here would be a guess at it.
        condition.observations = int(row.get("observations") or condition.observations + 1)
        self.condition_observations += 1

    def _close_zone_condition(
        self, runtime: CameraRuntime, key: str, condition: OpenCondition,
        *, closed_at: float, reason: str,
    ) -> None:
        runtime.conditions.pop(key, None)
        if self.db.close_condition(condition.event_id, closed_at=closed_at, reason=reason) == 0:
            return
        self.conditions_closed += 1
        print(f"condition {condition.event_id}: {runtime.id} closed ({reason}) after "
              f"{describe_duration(closed_at - condition.opened_ts)}, "
              f"{condition.observations} observation(s)", flush=True)

    def close_stale_conditions(self, now: float | None = None) -> list[int]:
        """End detector conditions the appliance has heard nothing about.

        Run from :meth:`_refresh`, which is the engine's 15 s cadence, so it
        also runs on the first tick after a start — the case that matters most.
        An appliance stopped with conditions open would otherwise come back
        claiming they are all still ongoing, and the span it eventually wrote
        would cover the hours it spent switched off.

        **This exists separately from the VLM scheduler's sweep because the VLM
        may not be running.** On this box today it is not: `vlm.enabled` gates
        the whole runner, and with it off — or the model down, or the service
        stopped — nothing swept anything. A detector condition whose camera died
        would have sat "ongoing" on the feed forever. The two sweeps stay off
        each other's rows through ``detector_type``; see
        :meth:`Database.stale_conditions` for why their windows differ.

        ``now`` is wall-clock and so is the column it is measured against. The
        span is still closed at ``last_seen_ts`` — the last moment the zone was
        known over its limit — because "when did it end" and "when did we stop
        looking" are different questions and only the first belongs on the card.
        """
        settings = self.config.detection.continuity
        if not settings.enabled:
            return []
        now = time.time() if now is None else now
        closed: list[int] = []
        for row in self.db.stale_conditions(
            now - settings.stale_after_sec, detector_type="detector"
        ):
            event_id = int(row["id"])
            last_seen = float(row.get("last_seen_ts") or row["ts"])
            if not self.db.close_condition(event_id, closed_at=last_seen, reason="stale"):
                continue
            closed.append(event_id)
            self.conditions_closed += 1
            print(f"condition {event_id}: closed (stale, nothing observed it for "
                  f"{describe_duration(now - last_seen)})", flush=True)
        if closed:
            stale = set(closed)
            for runtime in self.runtimes.values():
                for key, condition in list(runtime.conditions.items()):
                    if condition.event_id in stale:
                        runtime.conditions.pop(key, None)
        return closed

    @staticmethod
    def _condition_summary(zone_name: str, verdict: ZoneVerdict, ongoing_sec: float) -> str:
        """"Packing floor: 12 people · limit 8 · peak 14 · 6 min".

        A middle dot rather than an em dash, because this string is *drawn into
        the snapshot* and the bundled font has no em dash (argus/media/
        annotate.py). It is also the separator the console already uses.
        """
        limit = "no limit" if verdict.limit is None else f"limit {verdict.limit}"
        return (f"{zone_name}: {verdict.count} people · {limit} · "
                f"peak {verdict.peak} · {describe_duration(ongoing_sec)}")

    # --- events -----------------------------------------------------------

    def _write_zone_snapshot(
        self, runtime: CameraRuntime, frame: Frame, tracked: Any, occupancy: Any,
        policy: dict[str, Any], headline: str, snapshot_rel: str,
    ) -> Any:
        """Draw and save the evidence frame, returning the annotated image.

        Shared by the open and the extend paths so a refreshed snapshot is the
        same picture, drawn the same way, as the one the condition opened with —
        the rule ``VlmRunner._write_snapshot`` already follows for the same
        reason.
        """
        zone = occupancy.zone
        severity = str(policy.get("severity") or "medium")
        inside = occupancy.inside
        confidence = tracked.confidence if tracked.confidence is not None else np.ones(len(inside))
        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else [None] * len(inside)
        image = annotate_zone_event(
            frame.image,
            polygon_norm=zone.polygon,
            headline=headline,
            color=SEVERITY_COLORS.get(severity, SEVERITY_COLORS["high"]),
            zone_color=hex_to_rgb(zone.color),
        )
        save_snapshot(image, self.config.snapshots_dir / snapshot_rel,
                      quality=self.config.evidence.snapshot_quality)
        return image

    def _emit_zone_event(
        self, runtime: CameraRuntime, frame: Frame, tracked: Any, occupancy: Any,
        policy: dict[str, Any], *, verdict: ZoneVerdict | None = None,
        condition_key: str | None = None,
    ) -> tuple[int, str]:
        """Snapshot, row, clip — in that order, so evidence exists before the alert.

        ``verdict`` is the smoothed judgement this event is being raised on
        (``detection.continuity``); without it the numbers come straight off the
        frame, which is the pre-change behaviour. ``condition_key`` makes the row
        an ongoing condition rather than a one-off — migration 7's columns, set
        exactly as the VLM scheduler sets them.

        Returns the event id and the snapshot's path relative to
        ``snapshots_dir``, because a condition refreshes that same file for as
        long as it lasts and must not have to re-derive its name.
        """
        zone = occupancy.zone
        inside = occupancy.inside
        confidence = tracked.confidence if tracked.confidence is not None else np.ones(len(inside))
        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else [None] * len(inside)
        # The count the decision was made on: the median of the window when
        # there is one, this frame's count when there is not. Quoting the raw
        # count on a smoothed decision would put a number in the email that the
        # policy did not act on.
        count = occupancy.count if verdict is None else verdict.count
        headline = f"{zone.name}: {count} people - limit {zone.sanctioned_limit}"

        snapshot_rel = f"{runtime.id}/{zone.id}_{int(frame.ts)}.jpg"
        image = self._write_zone_snapshot(
            runtime, frame, tracked, occupancy, policy, headline, snapshot_rel
        )

        height, width = frame.image.shape[:2]
        inside_confidence = [float(c) for c, is_in in zip(confidence, inside) if is_in]

        event_id = self.db.insert_event(Event(
            camera_id=runtime.id,
            zone_id=zone.id,
            policy_id=policy["id"],
            detector_name=policy["id"],
            detector_type="detector",
            severity=str(policy.get("severity") or "medium"),
            # The count is exact; what is uncertain is the detector's belief in
            # the people that make it up, so the weakest of those is the honest
            # confidence for the event as a whole.
            confidence=round(min(inside_confidence), 4) if inside_confidence else 1.0,
            stage=str(policy.get("stage") or "shadow"),
            routed_to=policy.get("routed_to"),
            description=(
                f"Zone '{zone.name}' over sanctioned limit: {count} people "
                f"(limit {zone.sanctioned_limit})"
            ),
            ts=frame.ts,
            snapshot_path=snapshot_rel,
            meta=self._event_meta(runtime, occupancy, tracked, tracker_ids, confidence,
                                  width, height, verdict=verdict),
            condition_key=condition_key,
            condition_state="open" if condition_key else None,
        ))
        runtime.events += 1
        print(f"event {event_id}: {runtime.id} / {zone.name} — {count} people "
              f"(limit {zone.sanctioned_limit}), stage={policy.get('stage')}"
              + (", ongoing" if condition_key else ""), flush=True)
        self._schedule_clip(runtime, event_id, frame.ts)
        if self.index is not None:
            # `frame.image`, not `image`: the annotated snapshot is what a
            # search *result* shows, but what gets embedded is the scene as the
            # camera saw it. Embedding our own boxes and banner would make
            # "zone overlay" a searchable feature of the footage.
            self.index.offer_event(runtime.id, event_id, frame.image, frame.ts, snapshot_rel)
        if self.notifier is not None:
            # Sends only for auto_routed events, on its own thread, with the
            # snapshot that already exists attached. The clip is still being
            # cut; the mail does not wait for it, because an alert that arrives
            # 13 seconds late is 13 seconds of somebody standing in the zone.
            if self.vlm is not None:
                # Phase 2: with the verification pass enabled, the VLM sees this
                # annotated frame first and may veto the alert. Returns
                # immediately either way — the check runs on the VLM thread.
                self.vlm.dispatch(event_id, image, self.notifier)
            else:
                self.notifier.dispatch(event_id)
        return event_id, snapshot_rel

    def _event_meta(
        self, runtime: CameraRuntime, occupancy: Any, tracked: Any, tracker_ids: Any,
        confidence: Any, width: int, height: int, *, verdict: ZoneVerdict | None = None,
    ) -> dict[str, Any]:
        """Everything a reviewer or the UI needs to redraw this event.

        Boxes are stored normalized so the overlay lands correctly on the
        full-resolution clip and on the WebRTC live view, neither of which is
        the 960x544 the detector saw.

        ``count`` is the number the *decision* was made on — the smoothed one
        when there is a verdict — and ``frame_count`` is what this particular
        frame held. Both, because a reviewer looking at a snapshot showing 11
        boxes under a banner reading 12 is entitled to know which one the
        appliance acted on and why they differ.
        """
        meta: dict[str, Any] = {
            "count": occupancy.count if verdict is None else verdict.count,
            "limit": occupancy.zone.sanctioned_limit,
            "excess": occupancy.excess if verdict is None else verdict.excess,
            "zone_name": occupancy.zone.name,
            "camera_name": runtime.name,
            "track_ids": [
                int(t) for t, is_in in zip(tracker_ids, occupancy.inside) if is_in and t is not None
            ],
            "boxes": normalized_boxes(tracked.xyxy, tracker_ids, confidence, width, height),
            "inside": [bool(v) for v in occupancy.inside],
            "frame": {"width": width, "height": height},
            "detector": self._detector_info().get("model"),
        }
        if verdict is not None:
            settings = self.config.detection.continuity
            meta.update({
                "frame_count": occupancy.count,
                "peak": verdict.peak,
                # When the breach actually began, which is `sustain_sec` before
                # the row's own `ts`: the condition existed before it was
                # confirmed, and a duration measured from the row alone is short
                # by exactly that much.
                "over_since": verdict.over_since,
                "smoothing": {
                    "window_sec": settings.window_sec,
                    "sustain_sec": settings.sustain_sec,
                    "clear_sec": settings.clear_sec,
                    "samples": verdict.samples,
                },
            })
        return meta

    def _schedule_clip_for(
        self, camera_id: str, event_id: int, event_ts: float,
        image: Any | None = None, snapshot_path: str | None = None,
    ) -> None:
        """Evidence for an event raised outside the vision loop: clip, index.

        The VLM runner has an event id and a camera id but no ``CameraRuntime``
        — the ring buffers and the index runner belong to the engine. This is
        the whole of the engine's downward interface to it, which is why the
        raw frame travels along with the id: it is the thing both consumers
        need and the one thing the runner cannot look up.
        """
        runtime = self.runtimes.get(camera_id)
        if runtime is not None:
            self._schedule_clip(runtime, event_id, event_ts)
        if self.index is not None and image is not None:
            self.index.offer_event(camera_id, event_id, image, event_ts, snapshot_path)

    def _schedule_clip(self, runtime: CameraRuntime, event_id: int, event_ts: float) -> None:
        """Cut the ±clip_pad clip once the *after* footage is on disk.

        The cut is deferred rather than skipped: at the moment of the event the
        following seconds do not exist yet, and a clip that stops at the event
        is the one thing a reviewer cannot use.
        """
        if runtime.ring is None:
            return
        evidence = self.config.evidence

        def cut() -> None:
            with self._timers_lock:
                self._timers.discard(timer)
            try:
                relative = f"{runtime.id}/event_{event_id}.mp4"
                clip = runtime.ring.cut(
                    event_ts - evidence.clip_pad_sec,
                    event_ts + evidence.clip_pad_sec,
                    self.config.clips_dir / relative,
                )
                if clip is None:
                    print(f"clip: event {event_id} has no ring footage "
                          f"({runtime.ring.stats.last_error})", flush=True)
                    return
                self.db.update_event(event_id, clip_path=relative, clip_start=clip.start)
            except Exception as exc:  # noqa: BLE001 - a failed clip must not kill the timer thread
                print(f"clip: event {event_id} failed ({type(exc).__name__}: {exc})", flush=True)

        timer = threading.Timer(evidence.clip_delay_sec, cut)
        timer.daemon = True
        with self._timers_lock:
            self._timers.add(timer)
        timer.start()

    # --- occupancy --------------------------------------------------------

    def _record_occupancy(self, camera_id: str, zone_id: str, ts: float, count: int) -> None:
        """Aggregate to one sample per zone per flush window.

        Storing every sampled frame would be 5 rows per zone per second forever;
        what the occupancy chart and the capacity argument actually need is the
        *peak* in each window, which is also the number a safety officer asks
        about.
        """
        with self._occupancy_lock:
            sample = self._occupancy.get((camera_id, zone_id))
            if sample is None:
                self._occupancy[(camera_id, zone_id)] = {
                    "camera_id": camera_id, "zone_id": zone_id, "ts": ts, "count": count,
                }
            else:
                sample["count"] = max(sample["count"], count)
                sample["ts"] = ts

    def _maybe_flush_occupancy(self, force: bool = False) -> int:
        now = time.time()
        if not force and now - self._last_occupancy_flush < self.config.evidence.occupancy_flush_sec:
            return 0
        with self._occupancy_lock:
            samples, self._occupancy = list(self._occupancy.values()), {}
        self._last_occupancy_flush = now
        if not samples:
            return 0
        return self.db.insert_occupancy(samples)

    # --- database refresh -------------------------------------------------

    def _refresh_policies(self) -> None:
        self.policies = [
            policy for policy in self.db.list_policies(
                kind="detector", enabled_only=True, active_only=True
            )
            if policy["detector_key"] == HEADCOUNT_DETECTOR_KEY
        ]

    def _refresh(self) -> None:
        """Pick up zone and policy edits without a restart."""
        self._refresh_policies()
        for runtime in self.runtimes.values():
            runtime.zones = ZoneSet.from_rows(self.db.list_zones(runtime.id))
        # On the same cadence, and here rather than in the VLM scheduler's tick,
        # because the VLM is optional and on this box is not running at all. A
        # condition goes stale precisely because nothing is observing it, so the
        # sweep has to live somewhere that is always up.
        try:
            self.close_stale_conditions()
        except Exception as exc:  # noqa: BLE001 - housekeeping must not end a run
            print(f"engine: stale-condition sweep failed "
                  f"({type(exc).__name__}: {exc})", flush=True)

    # --- camera health ----------------------------------------------------

    def _refresh_camera_status(self) -> None:
        """Keep the cameras table honest about what is actually decoding.

        v1 (and stage 1) set 'live' once and never revisited it, so a camera that
        died at 02:00 still read 'live' at breakfast. Status is derived here from
        the decoder's own clock every report tick.
        """
        now = time.time()
        stall = self.config.detection.stall_timeout_sec
        for runtime in self.runtimes.values():
            stats = runtime.source.stats
            if stats.frames == 0:
                elapsed = now - (stats.started_at or now)
                status = "connecting" if elapsed < self.config.detection.start_timeout_sec else "error"
            elif runtime.last_frame_at is not None and now - runtime.last_frame_at > stall:
                status = "error"
            else:
                status = "live"
            detail = stats.last_error if status == "error" else None
            # Write on change, or every 30s while live so last_seen_at stays fresh.
            if status != runtime.status or (status == "live" and now - runtime.last_status_write > 30):
                try:
                    self.db.set_camera_status(runtime.id, status, detail)
                except Exception:  # noqa: BLE001 - health reporting must never crash the run
                    pass
                runtime.status = status
                runtime.last_status_write = now

    # --- reporting --------------------------------------------------------

    def _detector_info(self) -> dict[str, Any]:
        """``describe()`` is optional on the Detector protocol — a fake in a test
        need not implement it."""
        describe = getattr(self.detector, "describe", None)
        return describe() if callable(describe) else {}

    def stats(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at or None,
            "uptime_sec": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
            # A clean shutdown says so, rather than leaving the API to infer it
            # from a stats file that will not go stale for another 45 seconds.
            "stopped": self._stopped,
            "detector": self._detector_info(),
            "analytics": self.analytics.stats() if self.analytics is not None else None,
            "gateway": {
                "enabled": self.gateway is not None,
                "running": self.gateway.is_running() if self.gateway else False,
                "url": self.config.gateway_url,
                "last_error": self.gateway.last_error if self.gateway else None,
            },
            "retention": self.retention.last_result.as_dict()
            if self.retention and self.retention.last_result else None,
            # Ongoing conditions for detector policies. `opened` counts
            # incidents, `observations` counts how much watching each of them
            # stands for — before this change the two were the same number, and
            # it was 112 151.
            "continuity": {
                "enabled": self.config.detection.continuity.enabled,
                "window_sec": self.config.detection.continuity.window_sec,
                "sustain_sec": self.config.detection.continuity.sustain_sec,
                "clear_sec": self.config.detection.continuity.clear_sec,
                "opened": self.conditions_opened,
                "closed": self.conditions_closed,
                "adopted": self.conditions_adopted,
                "observations": self.condition_observations,
                "open": sum(len(runtime.conditions) for runtime in self.runtimes.values()),
            },
            "vlm": self.vlm.stats() if self.vlm is not None else None,
            "index": self.index.stats() if self.index is not None else None,
            "cameras": [
                {
                    **runtime.source.stats.as_dict(),
                    "status": runtime.status,
                    "processed": runtime.frames_processed,
                    "dropped": runtime.frames_dropped,
                    "lag_ms": round(runtime.lag_ms, 1),
                    "lag_ms_peak": round(runtime.lag_ms_peak, 1),
                    "events": runtime.events,
                    "zones": len(runtime.zones),
                    # `status()` rather than `stats.as_dict()`: it adds what is
                    # actually inside the segments and whether an evidence clip
                    # from them has to be re-encoded before a browser will play
                    # it. Both are per-camera facts an operator needs when a
                    # clip costs 2.2 s and 3x the bytes on some cameras and
                    # nothing on others.
                    "ring": runtime.ring.status() if runtime.ring else None,
                    # What the live wall draws: counts per zone and the boxes
                    # from the most recently analysed frame.
                    "occupancy": runtime.live_occupancy,
                    "tracks": runtime.live_tracks,
                    "live_ts": runtime.live_ts,
                }
                for runtime in self.runtimes.values()
            ],
        }

    def _publish_live_frames(self) -> None:
        """Answer any outstanding request for a current frame (argus/live.py).

        On the reporting tick rather than the vision loop, and gated on a
        request rather than continuous: the steady-state cost of this method is
        one ``stat`` of a JSON file, and a JPEG encode happens only while
        somebody has Policy Studio open with the test panel. Never raises —
        publishing a picture for a button must not be able to interrupt a stats
        write, let alone a camera.
        """
        try:
            wanted = self.live.wanted()
            if not wanted:
                return
            images = {
                runtime.id: runtime.live_image
                for runtime in self.runtimes.values()
                if runtime.id in wanted and runtime.live_image is not None
            }
            if images:
                self.live.publish(images, quality=self.config.evidence.snapshot_quality)
        except Exception as exc:  # noqa: BLE001 - reporting must never end a run
            if not self._live_publish_failed:
                self._live_publish_failed = True
                print(f"engine: could not publish a live frame ({exc})", flush=True)

    def _write_stats(self) -> None:
        """Publish ``stats()`` for the API process to read.

        Written to a temporary file and renamed, so a reader never sees a
        half-written document — rename is atomic within a filesystem, and this
        is cheaper and simpler than a lock two processes would have to agree
        on. Failure here is logged once and otherwise ignored: statistics are
        not worth stopping a camera for.
        """
        path = self.config.engine_stats_path
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(self.stats(), default=str))
            temporary.replace(path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            if not self._stats_write_failed:
                self._stats_write_failed = True
                print(f"engine: could not publish stats to {path} ({exc})", flush=True)

    def report(self) -> str:
        lines = []
        for camera_id, runtime in sorted(self.runtimes.items()):
            stats = runtime.source.stats
            ring = runtime.ring.stats if runtime.ring else None
            ring_text = f" ring={ring.mode}:{ring.segments}" if ring else ""
            detail = f" ({stats.last_error})" if runtime.status == "error" and stats.last_error else ""
            lines.append(
                f"  {camera_id:<26} {runtime.status:<10} mode={stats.mode:<5} "
                f"fps={stats.fps:5.2f} seen={stats.frames:<6} done={runtime.frames_processed:<6} "
                f"lag={runtime.lag_ms:6.0f}ms events={runtime.events}{ring_text}{detail}"
            )
        described = self._detector_info()
        if described:
            lines.append(
                f"  detector: {described.get('model')} {described.get('backend')} "
                f"@{described.get('resolution')} — {described.get('ms_per_frame')} ms/frame, "
                f"batches={described.get('batches')} last_batch={described.get('last_batch_size')}"
            )
        if self.analytics is not None:
            pool = self.analytics.stats()
            lines.append(
                f"  analytics: {pool['workers']} worker(s) — {pool['processed']} processed, "
                f"{pool['dropped']} dropped, queued={pool['queued']}"
            )
        if self.vlm is not None:
            vlm = self.vlm.stats()
            inference = vlm["inference"]
            state = "ready" if vlm["available"] else f"DEGRADED ({vlm.get('reason')})"
            lines.append(
                f"  vlm:      {vlm['model']} {state} — {vlm['evaluations']} evals, "
                f"{vlm['events']} events, {inference['avg_latency_sec']}s avg, "
                f"queue={inference['queue_depth']}"
            )
        if self.index is not None:
            index = self.index.stats()
            state = "ready" if index["ready"] else f"OFF ({index.get('reason')})"
            lines.append(
                f"  index:    {index['model']} {state} — {index['indexed']} frames "
                f"({index['events_indexed']} events), {index['dropped']} dropped, "
                f"{index['ms_per_frame']} ms/frame, queue={index['queued']}"
            )
        return "\n".join(lines)

    # --- run --------------------------------------------------------------

    def run(self, duration: float | None = None, report_every: float = 10.0) -> int:
        self.start()
        self._install_signal_handlers()
        deadline = (time.time() + duration) if duration else None
        try:
            while not self._stop.is_set():
                if self._stop.wait(min(report_every, _STATS_WRITE_SEC)):
                    break
                now = time.time()
                if now - self._last_refresh >= _CONFIG_REFRESH_SEC:
                    self._last_refresh = now
                    self._refresh()
                self._refresh_camera_status()
                # The API reads this file for /api/health and /api/live/state and
                # treats it as stale after ENGINE_STALE_SEC.
                if now - self._last_stats_write >= _STATS_WRITE_SEC:
                    self._last_stats_write = now
                    self._write_stats()
                    self._publish_live_frames()
                if now - self._last_report >= report_every:
                    self._last_report = now
                    print(f"[{time.strftime('%H:%M:%S')}] engine:\n{self.report()}", flush=True)
                    if self.gateway is not None and not self.gateway.is_running():
                        print(f"gateway: not running ({self.gateway.last_error})", flush=True)
                if deadline and time.time() >= deadline:
                    break
        finally:
            self.stop()
        return 0

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):  # pragma: no cover - signal path
            print(f"\nreceived signal {signum}; shutting down", flush=True)
            self._stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except ValueError:  # not on the main thread (tests)
                pass

    def stop(self) -> None:
        self._stop.set()
        self._stopped = True
        self._frame_ready.set()
        # Vision thread first, then its pool: a frame the detector has already
        # decided on may still owe an event row, a snapshot and a clip, and both
        # the VLM's verification pass and the notifier are still up to take them.
        if self._vision_thread is not None:
            self._vision_thread.join(timeout=10)
            self._vision_thread = None
        if self.analytics is not None:
            self.analytics.drain(timeout=5.0)
            self.analytics.stop(timeout=5.0)
        if self.vlm is not None:
            # It may have an in-flight evaluation that still wants to write an
            # event row and cut a clip.
            self.vlm.stop(timeout=15)
        if self.index is not None:
            # After the VLM, because a last-moment VLM event is one more frame
            # to index; the runner drains its queue rather than dropping it.
            self.index.stop(timeout=15)
        if self.retention is not None:
            self.retention.stop(timeout=5)
        with self._timers_lock:
            timers, self._timers = list(self._timers), set()
        for timer in timers:
            timer.cancel()
        # `cancel()` stops a timer that has not fired; it does nothing to one
        # already inside `cut()`, and the join is what covers that. It did not
        # used to matter — a stream-copied clip is a 10 ms remux, so the window
        # was never open. A transcoded one is an NVENC pass measured at 2.2 s on
        # a 16 s 1440p clip, and without this the interpreter exits underneath
        # it and leaves a partial mp4 with no moov atom on disk. Bounded by one
        # shared deadline rather than per timer, because a fleet that raised
        # forty events in its last second must not turn shutdown into a minute.
        deadline = time.monotonic() + CLIP_DRAIN_SEC
        for timer in timers:
            timer.join(timeout=max(0.0, deadline - time.monotonic()))
        for camera_id, runtime in self.runtimes.items():
            if runtime.worker is not None:
                runtime.worker.stop(timeout=5)
            if runtime.ring is not None:
                runtime.ring.stop(timeout=5)
            try:
                self.db.set_camera_status(camera_id, "offline")
            except Exception:  # noqa: BLE001 - shutdown must not fail on db errors
                pass
        self._maybe_flush_occupancy(force=True)
        if self.gateway is not None:
            self.gateway.stop()
            self.gateway = None
        # Written after the cameras are marked offline and before the runtimes
        # are dropped, so the last published snapshot matches the database.
        self._write_stats()
        # A published frame outlives the process that published it, and the API
        # has no way to tell "current" from "the last thing this box saw before
        # it was shut down last Tuesday" except by its age. Removing them makes
        # the answer "the engine is not running", which is the true one.
        self.live.clear()
        self.runtimes.clear()


__all__ = [
    "CameraRuntime",
    "Engine",
    "HEADCOUNT_DETECTOR_KEY",
    "OpenCondition",
    "condition_key",
]
