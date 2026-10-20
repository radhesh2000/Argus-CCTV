"""Typed configuration for the Argus appliance.

Loading order, lowest priority first:

1. the dataclass defaults in this module (so Argus runs with no config file),
2. ``config/argus.yaml`` (the checked-in, deployment-editable source of truth),
3. environment variables named after the dotted path of a scalar setting,
   upper-cased and prefixed with ``ARGUS_`` (``detection.fps`` ->
   ``ARGUS_DETECTION_FPS``).

The env layer is deliberately generic rather than a hand-maintained list: it is
derived by walking the dataclass tree, so a new scalar field is overridable the
moment it is declared. Secrets never live in the YAML — ``smtp.password_env``
names the variable that holds the password instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "argus.yaml"
ENV_PREFIX = "ARGUS_"


class ConfigError(ValueError):
    """Raised when a config file or environment override cannot be applied."""


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SiteConfig:
    """Who this appliance belongs to, and how its day is divided.

    The plant name is masthead furniture, but the shifts are not: they label
    every event in analytics and decide the columns of the zone x shift
    heatmap, so they belong in deployment config rather than in the UI's
    JavaScript where a plant running 12-hour shifts could not reach them.

    A shift is only a start hour; it runs until the next one begins, wrapping
    midnight. That makes the common "06-14 / 14-22 / 22-06" rota three numbers
    instead of six, and makes an overlapping or gapped rota unrepresentable.
    """

    name: str = "Argus Appliance"
    shifts: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"id": "A", "start_hour": 6},
            {"id": "B", "start_hour": 14},
            {"id": "C", "start_hour": 22},
        ]
    )

    def __post_init__(self) -> None:
        if not isinstance(self.shifts, (list, tuple)) or not self.shifts:
            raise ConfigError("site.shifts must be a non-empty list")
        cleaned: list[dict[str, Any]] = []
        for entry in self.shifts:
            if not isinstance(entry, dict):
                raise ConfigError("site.shifts entries must be mappings with id and start_hour")
            try:
                hour = int(entry["start_hour"])
                shift_id = str(entry["id"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigError(f"site.shifts: bad entry {entry!r} (need id, start_hour)") from exc
            if not shift_id:
                raise ConfigError("site.shifts: id must not be empty")
            if not 0 <= hour <= 23:
                raise ConfigError(f"site.shifts: start_hour must be 0-23 (got {hour})")
            cleaned.append({"id": shift_id, "start_hour": hour})
        cleaned.sort(key=lambda s: s["start_hour"])
        hours = [s["start_hour"] for s in cleaned]
        if len(set(hours)) != len(hours):
            raise ConfigError("site.shifts: two shifts cannot start in the same hour")
        self.shifts = cleaned

    @property
    def shift_windows(self) -> list[dict[str, Any]]:
        """Shifts with their end hour filled in — the last one wraps midnight."""
        return [
            {**shift, "end_hour": self.shifts[(index + 1) % len(self.shifts)]["start_hour"]}
            for index, shift in enumerate(self.shifts)
        ]

    def shift_for_hour(self, hour: int) -> dict[str, Any]:
        """Which shift owns a wall-clock hour (0-23)."""
        hour = int(hour) % 24
        current = self.shift_windows[-1]  # before the first start hour == the wrapping shift
        for window in self.shift_windows:
            if hour >= window["start_hour"]:
                current = window
        return current


@dataclass(slots=True)
class ModelConfig:
    """The person detector. See argus/vision/detector.py for why torch FP16."""

    variant: str = "nano"          # rf-detr nano | small | medium | base | large
    precision: str = "fp16"        # fp16 | fp32
    device: str = "auto"           # auto | cuda | cpu | cuda:N
    trace: bool = True             # cache a torch.jit trace per batch size
    threshold: float = 0.5         # the detector's own operating point (eval, stand-alone use)
    # Boxes scoring between this and `threshold` still reach the tracker, where
    # they may only CONTINUE a track that already exists — never start one. It
    # is the half of ByteTrack the engine had been withholding; see
    # argus/vision/tracker.py. 0, or anything >= `threshold`, turns it off.
    track_floor: float = 0.15
    nms_iou: float = 0.8           # duplicate-box guard; 1.0 disables it
    resolution: int = 0            # 0 = the variant's native resolution
    max_batch: int = 8             # cameras coalesced into one forward pass
    # Weights trained on *this site's* labelled frames by
    # `scripts/train_detector.py`. Empty — the default, and what every appliance
    # ships with — means the stock COCO checkpoint. A site turns this on only
    # after `argus detector-eval --checkpoint <path>` has shown a gain on its own
    # footage; see docs/DETECTOR_EVAL.md. The file is loaded only alongside the
    # `provenance.json` its training run wrote beside it.
    checkpoint: str = ""


@dataclass(slots=True)
class TrackerConfig:
    """ByteTrack, in seconds rather than supervision's 30-fps frame counts."""

    activation_threshold: float = 0.3
    match_threshold: float = 0.8
    lost_track_sec: float = 2.0
    min_consecutive_frames: int = 3   # 0.6 s at 5 fps before a track is a person


@dataclass(slots=True)
class DetectionContinuityConfig:
    """Ongoing conditions for the *detector*: one row per breach, not per minute.

    The counterpart of ``vlm.continuity``, and it exists because the claim that
    section used to make about this one was wrong. It said "detector-origin
    events are untouched: their debounce already does this". Measured on this
    appliance on 2026-09-19, ``manpower_headcount`` had written **10 958 events
    in 24 hours** — ~1 400 per camera per day, one a minute around the clock,
    **112 151** in total. The debounce was not holding a state across frames; it
    was re-firing a brand-new row every ``cooldown_sec`` for as long as a zone
    stayed busy, which on eight cameras is 8 x 1 440 = 11.5k rows a day.

    With this on, a zone occupancy policy opens **one** event when the breach is
    established and extends that row in place while it lasts — the same
    machinery migration 7 gave VLM conditions, and the console already renders
    it as "ongoing since · N observations". ``cooldown_sec`` keeps its meaning
    for the case it was always right about: how soon a *new* condition may open.

    The smoothing and the hysteresis are in ``argus/vision/zone_state.py``;
    these are its knobs. Set ``enabled: false`` for the pre-change behaviour,
    one row per cooldown — the flag exists so the new behaviour is a decision
    rather than something that happened to a site on upgrade.
    """

    enabled: bool = True

    # Seconds of per-frame counts the working count is the median of. At the
    # shipped 5 fps that is ~15 samples. Chosen against the detector's measured
    # error on this plant's own frames (2026-09-19, hand labels): ~13% of boxes
    # are false positives and ~19% of people are missed per frame, so a single
    # frame's count flickers either side of the limit and one spurious box opens
    # an alert. A median over three seconds needs a real change to move.
    window_sec: float = 3.0

    # Seconds the smoothed count must be continuously over the limit before a
    # condition opens. Five seconds of a genuine over-crowd is still a prompt
    # alert; five seconds of detector noise does not exist.
    sustain_sec: float = 5.0

    # Seconds the smoothed count must be continuously at or under the limit
    # before the condition closes. Deliberately longer than ``sustain_sec``:
    # this is the hysteresis, and it is what makes a count hovering at the limit
    # one condition rather than a dozen. Symmetric thresholds would re-create
    # the storm in a quieter key.
    clear_sec: float = 10.0

    # How far under its limit an OPEN zone must fall before it starts to clear,
    # as a fraction of the limit and never less than one person: limit 20
    # releases at 18, limit 8 at 7. ``clear_sec`` is hysteresis in time and does
    # nothing for a count that sits on the limit; this is the other half.
    # 0 restores the single threshold. See ZoneState.release_level.
    release_margin: float = 0.10

    # How often an open condition's row is rewritten in place while it lasts.
    # Not every frame — that would be five UPDATEs a second per zone for a
    # number that has not changed — and not never, because ``last_seen_ts`` is
    # what the console's "ongoing since" is measured from and what the stale
    # sweep below reads. A rise in the peak is written immediately regardless.
    update_sec: float = 15.0

    # Seconds an open detector condition may go unobserved before it is closed
    # as stale. It is the answer to "the engine stopped, the camera dropped, the
    # zone was deleted" — the cases where nothing will ever close the row from
    # the live path — and it doubles as the window in which a restarted engine
    # can adopt the row it left open instead of filing a duplicate.
    #
    # Must be comfortably longer than ``update_sec``, or a perfectly healthy
    # condition is closed between its own updates and immediately reopened; the
    # VLM side learned that the hard way (see ``Database.stale_conditions``) and
    # ``__post_init__`` refuses the configuration here. 120 s is eight missed
    # updates, and well past ``detection.stall_timeout_sec``, so a camera that
    # is merely stuttering does not end a real condition.
    stale_after_sec: float = 120.0

    def __post_init__(self) -> None:
        if self.window_sec < 0:
            raise ConfigError("detection.continuity.window_sec must be >= 0")
        if self.sustain_sec < 0 or self.clear_sec < 0:
            raise ConfigError("detection.continuity.sustain_sec/clear_sec must be >= 0")
        if not 0.0 <= self.release_margin < 1.0:
            raise ConfigError("detection.continuity.release_margin must be in [0, 1)")
        if self.update_sec <= 0:
            raise ConfigError("detection.continuity.update_sec must be > 0")
        if self.stale_after_sec <= self.update_sec:
            raise ConfigError(
                "detection.continuity.stale_after_sec must be greater than update_sec, "
                "or an observed condition is closed between its own updates"
            )


@dataclass(slots=True)
class DetectionConfig:
    fps: float = 5.0
    width: int = 960
    height: int = 544
    hwaccel: bool = True
    start_timeout_sec: float = 20.0
    # A cold start is the worst moment of a camera's life: 96 cameras means ~290
    # ffmpeg processes spawning at once (decoder, ring recorder and gateway
    # publisher each), and at that scale nobody's first frame arrives inside a
    # flat 20 s. Two knobs answer that: the engine leaves this long between
    # starting one camera and the next, and the first-frame deadline grows with
    # the fleet instead of staying at the single-camera value.
    start_stagger_sec: float = 0.15
    start_timeout_per_camera_sec: float = 0.25
    start_timeout_max_sec: float = 120.0
    reconnect_backoff_sec: float = 2.0
    reconnect_backoff_max_sec: float = 30.0
    # How long a camera that was starved of its first frame on hardware decode
    # stays on software before trying hardware again, doubling each time it is
    # starved again (capped). Not "forever": see argus/ingest/decoder.py.
    hwaccel_retry_sec: float = 120.0
    hwaccel_retry_max_sec: float = 3600.0
    # Seconds without a frame before a camera is reported as unhealthy.
    stall_timeout_sec: float = 10.0
    # Threads consuming detector output (tracking, zones, occupancy, events).
    # 0 runs them inline on the vision thread, and 0 is what ships.
    #
    # In isolation a pool looks like a clear win — one worker overlaps with the
    # ~26% of `detect()` that runs GIL-free and measures 1.27x. In the running
    # engine it is a 9% *loss* (4.78 -> 4.37 fps at 64 cameras), because the
    # vision thread is not alone: 64 decoder threads and their stderr drains
    # already fill every GIL gap detection opens, and one more GIL-hungry
    # thread just adds handoffs. The pool is kept because it is where this work
    # belongs the moment the tracker stops holding the GIL — free-threaded
    # Python, or a native tracker — but it is off by default and the number
    # that justifies that is in docs/BENCHMARKS.md.
    analytics_workers: int = 0
    # Frames per camera the pool will hold before dropping the oldest one for
    # that camera. Bounded for the same reason the decoder's slot is: falling
    # behind must cost sampling rate, never liveness or memory.
    analytics_queue_depth: int = 3
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    continuity: DetectionContinuityConfig = field(default_factory=DetectionContinuityConfig)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ConfigError("detection.fps must be > 0")
        if self.width <= 0 or self.height <= 0:
            raise ConfigError("detection.width/height must be > 0")
        if self.analytics_workers < 0:
            raise ConfigError("detection.analytics_workers must be >= 0")
        if self.analytics_queue_depth < 1:
            raise ConfigError("detection.analytics_queue_depth must be >= 1")
        if not 0.0 <= self.model.track_floor <= 1.0:
            raise ConfigError("detection.model.track_floor must be between 0 and 1")
        if self.tracker.min_consecutive_frames < 1:
            raise ConfigError("detection.tracker.min_consecutive_frames must be >= 1")

    def start_timeout_for(self, fleet: int) -> float:
        """First-frame deadline for a fleet of ``fleet`` cameras."""
        return min(
            self.start_timeout_sec + max(0, fleet) * self.start_timeout_per_camera_sec,
            self.start_timeout_max_sec,
        )

    @property
    def frame_bytes(self) -> int:
        """Size of one BGR24 frame — the decoder reads exactly this per frame."""
        return self.width * self.height * 3


@dataclass(slots=True)
class RolloutConfig:
    """The bar a policy clears before it may notify anybody unattended.

    These numbers were in the console's JavaScript and nowhere else, which meant
    the staged rollout — the thing this product is actually selling — was a
    disabled button rather than a rule. `POST /api/policies/{id}/promote`
    enforces them now, and the console reads them from `/api/config` so there is
    one copy.

    `min_reviews` is the one people argue about, and 200 is not arbitrary: at
    195 confirmed out of 200 the 95% lower bound on precision is still about
    94%, so a smaller sample cannot distinguish "precise" from "lucky". A site
    with a rarer violation can lower it — and is then choosing to promote on
    weaker evidence, which is a decision worth having to write down in a file.
    """

    min_precision: float = 0.95
    min_reviews: int = 200
    # Whether an operator may promote anyway. Left ON because the alternative is
    # an appliance that cannot be made to work on a site whose evidence lives
    # somewhere else — but every override is refused unless it names a reason,
    # and the reason is stored on the policy row.
    allow_override: bool = True


@dataclass(slots=True)
class EvidenceConfig:
    """Ring buffer, evidence clips and snapshots."""

    enabled: bool = True
    clip_pad_sec: float = 8.0        # seconds kept either side of an event
    ring_segment_sec: float = 2.0    # ring segment length
    ring_retention_sec: float = 300.0  # rolling footage kept per camera
    ring_mode: str = "auto"          # auto | copy | encode
    # Whether an evidence clip is re-encoded to H.264 when the ring holds
    # something a browser will not play (H.265, MPEG-4 Part 2). `auto` decides
    # per camera from what is actually in the segments; `never` restores the
    # pre-Phase-6 behaviour of always stream-copying, which is correct only for
    # a site that reviews evidence somewhere other than this console.
    clip_transcode: str = "auto"     # auto | never
    snapshot_quality: int = 85
    occupancy_flush_sec: float = 10.0
    # Cooldown applied to a policy whose own cooldown_sec is unset. A policy
    # that explicitly sets 0 is honoured: it alerts on every sampled frame.
    default_cooldown_sec: float = 60.0

    @property
    def clip_delay_sec(self) -> float:
        """How long to wait before cutting a clip.

        The event's ``+clip_pad`` footage has to exist *and* the segment holding
        it has to be closed — which happens when the following segment starts —
        so the wait is the pad plus two segments.
        """
        return self.clip_pad_sec + 2 * self.ring_segment_sec + 1.0


@dataclass(slots=True)
class EvidenceRetentionConfig:
    """How long a clip or snapshot lives, **by what the event turned out to be**.

    A flat ``clips_days`` treats every event as worth the same disk, and no
    site's events are. On this appliance, 47 881 events in two and a half days
    were almost entirely shadow-stage repeats of the same looped test footage,
    and every one of them cut a 19.4 MB clip that the flat 14-day window kept
    for a fortnight. Meanwhile the handful a human actually confirmed — the only
    events anybody will ever ask to see again — had exactly the same deadline.

    So the window is keyed on the event, resolved verdict-first:

    * a review verdict of ``confirmed`` or ``false_positive`` wins, because a
      human looked and said what this was;
    * failing that, the event's stage — ``shadow`` (nobody was told),
      ``reviewed`` (it reached the triage queue), ``auto_routed`` (it emailed
      somebody with no human in the loop).

    ``false_positive_days`` is deliberately **not** the shortest. A confirmed
    false positive is a labelled negative, and labelled negatives are the one
    training and eval input a deployment produces for free —
    ``argus.eval.dataset.triage_cases`` reads exactly these snapshots. There are
    164 reviewed events on this box against 47 875 unreviewed ones; keeping them
    costs nothing and throwing them away costs the eval set.

    **Every field defaults to None, meaning "inherit the flat window"**
    (``retention.clips_days`` / ``retention.snapshots_days``). A config that
    does not write this block therefore behaves exactly as it did before it
    existed — no upgrade quietly starts deleting evidence a site was keeping.
    """

    shadow_days: float | None = None
    reviewed_days: float | None = None
    #: The production stage — an alert that went out with nobody in the loop.
    #: Defaults to ``reviewed_days`` rather than being folded into it silently:
    #: on a mature site *every* alert is auto_routed, and an operator who set
    #: ``reviewed_days: 2`` thinking of the triage queue would otherwise have
    #: given their live alerting a two-day evidence window without being asked.
    auto_routed_days: float | None = None
    confirmed_days: float | None = None
    false_positive_days: float | None = None

    def __post_init__(self) -> None:
        for name in ("shadow_days", "reviewed_days", "auto_routed_days",
                     "confirmed_days", "false_positive_days"):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if value < 0:
                raise ConfigError(f"retention.evidence.{name} must be >= 0")
            setattr(self, name, value)

    @property
    def configured(self) -> bool:
        """Whether the operator wrote any per-stage window at all."""
        return any(getattr(self, f.name) is not None for f in fields(self))

    def days(self, tier: str, fallback: float) -> float:
        """The window for one tier, falling back through auto_routed -> reviewed
        -> the media kind's flat window."""
        value = getattr(self, f"{tier}_days", None)
        if value is None and tier == "auto_routed":
            value = self.reviewed_days
        return float(fallback if value is None else value)


@dataclass(slots=True)
class RetentionConfig:
    """Bounded storage. v1 had none and the demo loop produced 186 GB.

    Three mechanisms now, in ascending order of bluntness:

    * the **age sweep** — the ``*_days`` windows, refined per event tier by
      ``evidence`` above;
    * the **evidence budget** — ``max_clips_gb`` / ``max_snapshots_gb``, a hard
      ceiling enforced by evicting the least valuable evidence first. Ages bound
      how *long* a site keeps things; only a budget bounds how *much*, and a
      pathological site (or a demo loop that never clears a violation) is a
      volume problem, not an age problem;
    * the **disk guard** — ``max_disk_percent``, oldest-first across every
      directory Argus owns, for when the other two were configured wrong or
      something else on the box filled the same filesystem.
    """

    enabled: bool = True
    clips_days: int = 14
    snapshots_days: int = 30
    events_days: int = 180
    occupancy_days: int = 7
    # Index sample thumbnails and their vectors. **The shortest window here, and
    # the arithmetic is why** — every figure below measured on this box's own
    # output, not estimated: a periodic sample at the default 10 s is 8 640
    # frames per camera per day, at 15.1 KB a thumbnail and 3.33 KB a stored
    # vector, so 159 MB per camera-day. Across 48 cameras that is 7.6 GB a day;
    # 14 days of it is **107 GB**, which a 3.7 TB appliance carries comfortably
    # (3%), and 180 days would be 1.4 TB, which it would not. Event frames are
    # pruned with their events, not by this.
    index_days: int = 14
    # Per-stage refinement of clips_days/snapshots_days. Empty = flat windows.
    evidence: EvidenceRetentionConfig = field(default_factory=EvidenceRetentionConfig)
    # --- the hard budget -------------------------------------------------
    #
    # Two numbers rather than one combined "evidence" budget, because clips and
    # snapshots are not the same commodity. A clip on this box is 19.4 MB after
    # the H.265 -> H.264 transcode; a snapshot is 154 KB — 126x smaller and the
    # thing that actually renders in the triage list and the search results. A
    # single pooled budget would let clips crowd snapshots out completely, and
    # an eviction pass working against it would delete a hundred stills to buy
    # back the space of one clip. Sized separately, each number is also a
    # sentence somebody can plan a disk against: "100 GB of clips" is
    # procurement input; "120 GB of evidence" is not.
    #
    # 0 (the dataclass default) means no budget — ages and the disk guard only,
    # i.e. what this appliance did before budgets existed. config/argus.yaml
    # ships real numbers; an unedited *programmatic* config does not invent a
    # ceiling for a disk it has never seen.
    max_clips_gb: float = 0.0
    max_snapshots_gb: float = 0.0
    # How often the housekeeping thread inside `argus run` sweeps.
    interval_hours: float = 6.0
    # Filesystem fullness (whole device holding data_dir) that triggers an
    # early oldest-first reclaim, regardless of the age windows above.
    max_disk_percent: float = 85.0
    # Floor for that emergency pass: evidence younger than this is never
    # deleted, so an alert raised minutes ago always still has its clip. The
    # budget eviction honours it too, for the same reason.
    min_keep_hours: float = 6.0

    def __post_init__(self) -> None:
        if not 1.0 <= self.max_disk_percent <= 100.0:
            raise ConfigError("retention.max_disk_percent must be between 1 and 100")
        if self.interval_hours <= 0:
            raise ConfigError("retention.interval_hours must be > 0")
        for name in ("max_clips_gb", "max_snapshots_gb"):
            if float(getattr(self, name)) < 0:
                raise ConfigError(f"retention.{name} must be >= 0 (0 = no budget)")

    def budget_gb(self, kind: str) -> float:
        """The budget for a media kind, or 0 when it has none."""
        return float({"clips": self.max_clips_gb,
                      "snapshots": self.max_snapshots_gb}.get(kind, 0.0))


@dataclass(slots=True)
class BackupConfig:
    """Nightly SQLite snapshots. The other half of data safety.

    Retention bounds what the appliance keeps; this is what happens when the
    thing it kept goes bad — a mistaken bulk delete, a half-applied migration,
    a corrupted page. Runs on the same housekeeping thread as retention, for
    the same reason: an appliance whose backups depend on somebody adding a
    cron entry is an appliance without backups.
    """

    enabled: bool = True
    # Hours between scheduled backups. Age-based rather than "at 03:00": a box
    # powered down overnight silently skips a wall-clock window and nothing
    # says so, whereas age is owed until it is paid. See argus/backup.py:due.
    interval_hours: float = 24.0
    # Complete backups retained. Seven puts a week between "somebody deleted
    # the wrong camera" and "nobody noticed for a week", which is the realistic
    # detection time on an unattended appliance.
    keep: int = 7
    # Empty = data_dir/backups. Point this at a mounted share and the copy also
    # survives the NVMe failing; on the default it does not, and docs/INSTALL.md
    # says so in those words rather than implying otherwise.
    directory: str = ""
    # The vector index is copied too. It is NOT derived data: `argus index
    # backfill` rebuilds vectors for event snapshots, but the periodic samples
    # that make an ordinary minute searchable exist nowhere else. It is also
    # the file that grows with the fleet, so a storage-constrained site has a
    # switch rather than a fork of this module.
    include_index: bool = True
    # Refuse to start a backup above this fullness. A backup that fills the
    # disk takes the appliance down to protect data that was not yet at risk.
    max_disk_percent: float = 90.0

    def __post_init__(self) -> None:
        if self.interval_hours <= 0:
            raise ConfigError("backup.interval_hours must be > 0")
        if self.keep < 1:
            raise ConfigError("backup.keep must be >= 1")
        if not 1.0 <= self.max_disk_percent <= 100.0:
            raise ConfigError("backup.max_disk_percent must be between 1 and 100")


@dataclass(slots=True)
class LogsConfig:
    """Rotation for the two logs that are not on journald.

    ``argus run`` and ``argus api`` write to stdout, which under
    ``argus install-services`` is journald — already rotated by systemd, and
    duplicating that policy here would give one appliance two of them. What is
    left is the output of the two *child* processes Argus supervises, go2rtc
    and the vLLM server, which are redirected to files: measured on this box,
    ``vlm-server.log`` reached 14.7 MB in a day of use.
    """

    rotate: bool = True
    # Rotate a log once it passes this. 32 MB is roughly two days of a chatty
    # vLLM at the observed rate, and small enough to open over SSH.
    max_mb: float = 32.0
    # Generations kept (`go2rtc.log.1` … `.3`). Bounds each log at keep+1 files.
    keep: int = 3

    def __post_init__(self) -> None:
        if self.max_mb <= 0:
            raise ConfigError("logs.max_mb must be > 0")
        if self.keep < 1:
            raise ConfigError("logs.keep must be >= 1")


@dataclass(slots=True)
class SmtpConfig:
    host: str = "localhost"
    port: int = 587
    username: str = ""
    from_address: str = "Argus <argus@localhost>"
    starttls: bool = True
    password_env: str = "ARGUS_SMTP_PASSWORD"
    recipients: dict[str, list[str]] = field(default_factory=dict)

    @property
    def password(self) -> str:
        """Resolved from the environment, never from the config file."""
        return os.environ.get(self.password_env, "")

    def recipients_for(self, role: str | None) -> list[str]:
        if not role:
            return []
        return list(self.recipients.get(role, ()))


def _clean_webrtc_candidates(value: Any) -> list[str]:
    """Normalise and validate ``gateway.webrtc_candidates``.

    Accepts the YAML form (a list) and the env form
    (``ARGUS_GATEWAY_WEBRTC_CANDIDATES="a:8555,b:8555"``), because the env layer
    sets fields directly and would otherwise leave a bare string where go2rtc
    expects a list — which does not fail here, it fails as a gateway that will
    not parse its own config.

    A typo is rejected at load rather than at 3 a.m.: go2rtc silently ignores a
    candidate it cannot parse, so the symptom of ``100.113.208.36-8555`` is a
    console whose tiles never connect and a log that says nothing at all.
    """
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = [str(item) for item in value]
    else:
        raise ConfigError(
            "gateway.webrtc_candidates: expected a list of \"host:port\" strings, "
            f"got {type(value).__name__}"
        )

    cleaned: list[str] = []
    for item in items:
        candidate = item.strip()
        if not candidate:
            continue
        host, sep, port = candidate.rpartition(":")
        # go2rtc allows a transport suffix on the port ("8555/tcp").
        port = port.split("/", 1)[0]
        if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
            raise ConfigError(
                f"gateway.webrtc_candidates: expected \"host:port\", got {item!r}"
            )
        cleaned.append(candidate)
    return cleaned


@dataclass(slots=True)
class GatewayConfig:
    binary: str = "bin/go2rtc"
    api_port: int = 1984
    rtsp_port: int = 8554
    webrtc_port: int = 8555
    listen_host: str = "0.0.0.0"
    # Phase 5. go2rtc's HTTP API — snapshots, stream listing, WebRTC signalling,
    # and its own unauthenticated web UI — binds to loopback only. The browser
    # reaches it through `/api/gateway/*` on the authenticated API instead, so
    # there is one door with a lock on it rather than two, one of which had
    # `origin: "*"` on it. Widening this back to 0.0.0.0 re-opens exactly the
    # hole Phase 5 closed; the field exists so that is a decision somebody has
    # to write down, not a default.
    api_host: str = "127.0.0.1"
    # Extra ICE host candidates go2rtc advertises, ``"host:port"`` each. Emitted
    # as ``webrtc.candidates`` in the generated go2rtc.yaml and trickled to the
    # browser ahead of the ones go2rtc gathers for itself. Empty by default,
    # which is right for a LAN-only site.
    #
    # What it is for is a viewer who is not on that LAN. Signalling for them
    # already works — it rides the authenticated relay on the API port — but the
    # ICE candidates in that exchange are whatever go2rtc could enumerate from
    # this box's own interfaces. If the address that viewer reaches the
    # appliance on is not one of them — a port-forwarded public IP, a VPN
    # terminating on another host, a tailscaled in userspace mode with no TUN
    # device — the browser is offered nothing it can route to: the console
    # loads, the camera list is right, and every tile stays black with nothing
    # in any log to say why.
    #
    # Measured here, so it is not overclaimed: an overlay address that *is* a
    # local interface (this appliance's tailscale0) is already gathered with
    # this unset, so naming it is belt and braces — it pins the address rather
    # than trusting interface enumeration, and offers it first. An address that
    # is not a local interface can only be advertised this way.
    #
    # This *advertises* an address, it does not open one: the port is already
    # bound on all interfaces (see listen_host's note and docs/SECURITY.md).
    webrtc_candidates: list[str] = field(default_factory=list)
    log_level: str = "info"
    restart_backoff_sec: float = 3.0

    def __post_init__(self) -> None:
        _clean_webrtc_candidates(self.webrtc_candidates)

    @property
    def webrtc_candidate_list(self) -> list[str]:
        """Validated candidates, whatever shape they were configured in.

        Read through this rather than the field: an ``ARGUS_``-prefixed override
        bypasses ``__post_init__`` (it is a ``setattr`` on a built object), so
        this is the only checkpoint both paths pass through.
        """
        return _clean_webrtc_candidates(self.webrtc_candidates)


@dataclass(slots=True)
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    # Seconds between database polls on the SSE event stream.
    sse_poll_sec: float = 2.0
    # Where an operator reaches this appliance from a browser. Used only to put
    # a working link in alert emails; empty means "send no link" rather than
    # guess a hostname the recipient cannot resolve.
    public_url: str = ""
    # Comma-separated browser origins allowed to call the API cross-site. Empty
    # (the default) installs no CORS middleware at all: the Phase-4 SPA is
    # served by this same process, and only a separate dev server needs this.
    cors_origins: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def event_url(self, event_id: int) -> str | None:
        base = self.public_url.rstrip("/")
        return f"{base}/api/events/{event_id}" if base else None


@dataclass(slots=True)
class AuthConfig:
    """Accounts, sessions and the login rate limit (Phase 5).

    There is no "auth: off" switch, and that absence is the decision. A flag
    that turns authentication off is a flag that ships turned off — and the
    thing being protected here is footage of people at work. A fresh appliance
    has no accounts and therefore refuses every request until an installer runs
    ``argus user add``; that is the bootstrap, and it is one command.
    """

    cookie_name: str = "argus_session"
    # Absolute lifetime of a session, from login. A shift plus a handover: an
    # operator should not be asked to log in mid-shift, and a browser left on a
    # control-room desk should not still be authenticated tomorrow.
    session_ttl_hours: float = 12.0
    # Log a session out after this long with no request. 0 disables it, which
    # is what a permanently-displayed wall panel wants — see
    # argus/auth.py:idle_deadline_passed. Default 8 h: long enough that it never
    # fires during a shift, short enough that an unattended browser is not a key.
    session_idle_minutes: float = 480.0
    # `last_seen_at` is only rewritten when it is this stale, so the SSE poll and
    # the 2 s live-state poll do not turn every session into a write per request.
    session_touch_sec: float = 60.0
    # Set true when the appliance is behind TLS. Left false by default because
    # the overwhelmingly common install is plain HTTP on a segmented plant LAN,
    # and a `Secure` cookie there is not "more secure", it is "cannot log in".
    # docs/SECURITY.md says this in the sentence next to the TLS advice.
    cookie_secure: bool = False
    # Lax rather than Strict: `api.public_url` puts links to individual events in
    # alert emails, and Strict withholds the cookie on that first navigation, so
    # every alert link would land on a login screen. Lax still refuses the
    # cross-site POST, which is where CSRF actually lives.
    cookie_samesite: str = "lax"
    # Failed logins allowed in `login_window_sec`, counted per username and per
    # source address independently. Six is generous for a person and useless for
    # a script: at 5/300 s a 10 000-word list takes nine months.
    login_max_attempts: int = 6
    login_window_sec: float = 300.0
    # How long the lockout lasts once tripped. Equal to the window by
    # construction — the counter simply ages out — and named separately so the
    # 429's Retry-After is a promise rather than an estimate.
    login_lockout_sec: float = 300.0
    # Attempt rows older than this are swept by the retention job. Long enough
    # to answer "who has been trying to get in this week".
    attempt_retention_days: float = 7.0


@dataclass(slots=True)
class VlmServerConfig:
    """How ``argus vlm start`` brings up the local inference server.

    The server runs in its **own virtualenv**, not in the argus environment.
    That is a licensing boundary as much as a dependency one: vLLM drags in a
    large tree (and, on this box, its own torch build) that has no business in
    the appliance's runtime dependency graph. Argus talks to it over HTTP like
    any other OpenAI-compatible endpoint, so the server is swappable without
    touching a line of argus.
    """

    venv: Path = Path(".venv-vlm")     # created by scripts/setup_vlm_server.sh
    models_dir: Path = Path("models")  # where scripts/fetch_vlm.sh put the weights
    host: str = "127.0.0.1"            # loopback: nothing about this is for the network
    port: int = 8000
    # Fraction of the unified 128 GB vLLM may claim. GB10 has no separate VRAM,
    # so this is carved out of the same pool the detector, the decoders and the
    # page cache live in — 0.30 is ~38 GB, comfortably more than an 8B bf16
    # checkpoint plus its KV cache, and it leaves the vision plane untouched.
    gpu_memory_utilization: float = 0.30
    # A policy prompt plus one 1024px frame is ~1.5k tokens. 8192 is roomy.
    max_model_len: int = 8192
    # The Spark is a small-batch box; past ~4 concurrent sequences per-token
    # bandwidth dominates and latency degrades faster than throughput improves.
    max_num_seqs: int = 4
    # Weights load and CUDA graphs capture on first start; a cold boot of an 8B
    # model from page cache is ~2 min, from cold disk longer.
    startup_timeout_sec: float = 900.0
    # Appended verbatim to the `vllm serve` command line, for anything this
    # config does not name. Split on whitespace.
    extra_args: str = ""

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


@dataclass(slots=True)
class VlmVerifyConfig:
    """The alert-fatigue reducer: detector events double-checked before they mail.

    Off by default, and deliberately so. It puts a second model in the path
    between a detected violation and the person who needs to know about it, and
    that is a trade a site should opt into once it trusts the verifier — not
    something that silently starts vetoing alerts on upgrade.
    """

    enabled: bool = False
    # Only events at these stages are checked. Comma-separated. The default is
    # the only stage that mails anybody: shadow and reviewed events already have
    # a human in the loop, so a veto there would suppress nothing and cost an
    # inference per event.
    stages: str = "auto_routed"
    # A veto needs the model to be at least this sure. Below it the alert goes
    # out: the failure mode we accept is a false alarm, never a missed one.
    min_confidence: float = 0.6
    # Seconds to wait for the verdict before mailing anyway. An alert that is
    # late is worse than an alert that was not double-checked.
    timeout_sec: float = 20.0

    @property
    def stage_list(self) -> list[str]:
        return [s.strip() for s in self.stages.split(",") if s.strip()]


@dataclass(slots=True)
class VlmContinuityConfig:
    """Ongoing conditions: one row per *condition*, not per sampled verdict.

    A prompt policy asks about a **state** — "is there loose laminate on the
    wrapper apron" — and a state that is true at 20:22 is usually still true at
    20:37. Before this, every cooldown expiry raised a new row: measured on this
    appliance on 2026-08-26, ``anmol_packing_floor_waste`` had written **140**
    near-identical events about one apron, 40 of them overnight at the policy's
    ~15-minute cooldown. The feed was unreadable and the analytics counted a
    single mess as forty incidents.

    So a triggered verdict that continues the condition already on file
    **extends** that row instead of inserting one. The cooldown still governs
    when a *new* condition may be opened; it no longer governs how often the
    same one is re-reported.

    **Only VLM-origin events.** This used to say the detector's debounce
    already did the job for counts in a polygon; it did not, and 112 151 rows
    on this box said so. The detector's own version of the feature is
    :class:`DetectionContinuityConfig`, and it runs in the engine rather than
    here — a detector event still never enters this module.
    """

    enabled: bool = True

    # Grounding-box agreement, for a policy with no zone.
    #
    # Two observations of the same condition should be in the same place, and
    # the model's own ``box_2d`` is the only evidence of where. IoU is the test,
    # and this is the bar it must clear.
    #
    # **It is applied only to unscoped policies, and that is a measurement, not
    # a shortcut.** A zone-scoped policy is already anchored to a polygon: the
    # model is shown the zone crop, the question is about the zone, and waste
    # that moves from one end of the apron to the other is the same waste
    # problem seen again. Comparing boxes there would split it — on this
    # appliance's own 139 consecutive laminate observations the *median* IoU
    # between one sighting and the next is **0.000**, with the box centres a
    # median 0.11 of the frame apart against a box only 0.077 wide. An unscoped
    # policy has no polygon to stand in, so the box has to, and this is the bar.
    min_iou: float = 0.1

    # How long a condition may go unobserved before it is closed as stale.
    #
    # Generous on purpose, and asymmetric with the "cleared" close: an
    # *untriggered* verdict is positive evidence the condition ended and closes
    # it at once, whereas silence is only evidence that nobody looked — a
    # camera dropped, the model went degraded, an operator edited the policy.
    # Closing early on silence would write a span claiming the problem ended at
    # a moment nobody observed. Fifteen minutes is ten missed samples of the
    # slowest plant policy shipped (90 s) and, by coincidence worth naming, the
    # cooldown that produced the forty rows this feature replaces.
    #
    # **It must be longer than the policy's own ``interval_sec``**, or a healthy
    # condition is closed between samples and immediately reopened. Measured on
    # this box: a 20 s policy under a 30 s window churned once per sample. The
    # window is compared against when the verdict was *written*, not when the
    # frame was captured, so the model's latency does not eat into it — but the
    # interval still does, and no default can know a policy's interval.
    stale_after_sec: float = 900.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_iou <= 1.0:
            raise ConfigError("vlm.continuity.min_iou must be between 0 and 1")
        if self.stale_after_sec <= 0:
            raise ConfigError("vlm.continuity.stale_after_sec must be > 0")


@dataclass(slots=True)
class VlmConfig:
    """The VLM policy engine (Phase 2).

    ``enabled`` gates the scheduler, not the client: with it off, ``kind='vlm'``
    policies are simply not evaluated and the engine behaves exactly as it did
    in Phase 1.
    """

    enabled: bool = True
    # Where the OpenAI-compatible endpoint lives. Empty means "derive it from
    # the server section", which is what a single-box appliance always wants.
    base_url: str = ""
    # Served model id. Cosmos-Reason2-8B is the rebuild plan's choice but its HF
    # repo is gated; this is its Apache-2.0 base model. Same architecture, same
    # code path — see scripts/fetch_vlm.sh.
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    # vLLM ignores the key unless started with --api-key, but the OpenAI wire
    # format wants an Authorization header and some proxies insist on one.
    api_key: str = "argus-local"
    timeout_sec: float = 90.0
    # Concurrent in-flight inferences. Frames are sampled every 6-8 s per
    # policy, so this is a burst limit, not a throughput target.
    max_concurrency: int = 2
    # Frames are downscaled before they are sent: a VLM's image token cost grows
    # with area, and 1024px is past the point where more pixels change the
    # answer for "is someone holding a phone".
    max_image_width: int = 1024
    jpeg_quality: int = 85
    # Applied to a vlm policy whose own interval_sec is unset.
    default_interval_sec: float = 8.0
    temperature: float = 0.0
    max_tokens: int = 320
    # A `triggered` below this is treated as not triggered. The schema forces
    # the model to answer with a confidence; this is where that answer is used.
    min_confidence: float = 0.5
    # Reconnect backoff after the server goes away. Same shape as the decoder's.
    retry_backoff_sec: float = 5.0
    retry_backoff_max_sec: float = 120.0
    verify: VlmVerifyConfig = field(default_factory=VlmVerifyConfig)
    continuity: VlmContinuityConfig = field(default_factory=VlmContinuityConfig)
    server: VlmServerConfig = field(default_factory=VlmServerConfig)

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ConfigError("vlm.max_concurrency must be >= 1")
        if self.default_interval_sec <= 0:
            raise ConfigError("vlm.default_interval_sec must be > 0")

    @property
    def endpoint(self) -> str:
        """The base URL actually used, with the server section as the default."""
        return (self.base_url or self.server.base_url).rstrip("/")


@dataclass(slots=True)
class IndexConfig:
    """The semantic index (Phase 3): SigLIP 2 embeddings + a local vector store.

    ``enabled`` gates the *writer* — the periodic sampler and the event hook
    inside ``argus run``. The search API and ``argus index backfill`` work
    against whatever is already in the store either way, because an operator
    who turns indexing off to reclaim GPU should not also lose the ability to
    search what was already indexed.
    """

    enabled: bool = True
    # HuggingFace id of the SigLIP 2 checkpoint, resolved under models/ (never
    # fetched at runtime — scripts/fetch_siglip.sh puts it there). The literal
    # string "fake" selects the deterministic stub in embedder.py, which is how
    # the whole path is exercised on a box with no weights.
    model: str = "google/siglip2-base-patch16-384"
    device: str = "auto"        # auto | cuda | cpu | cuda:N
    precision: str = "fp16"     # fp16 | fp32 (fp32 is forced on CPU)
    # Seconds between periodic samples of each camera's live frame. These reuse
    # the frames the engine has already decoded — there is no second decode
    # path — so the only cost is the embedding and the thumbnail.
    sample_interval_sec: float = 10.0
    # Frames coalesced into one forward pass. 8 matches the detector's batch and
    # is where the vision tower stops getting cheaper per frame on this GB10.
    batch_size: int = 8
    # Frames the queue holds before the *oldest* is dropped. Bounded for the
    # same reason every other stage is: falling behind must cost index density,
    # never liveness or memory. 64 x 1.5 MB is 96 MB worst case.
    queue_depth: int = 64
    # Longest the worker waits for a batch to fill before embedding what it has.
    batch_wait_sec: float = 0.5
    # Width of the JPEG written for a sampled frame, and its quality. This is
    # the only new *storage* the index adds, and it is the number to turn down
    # first: see docs/PHASE3_NOTES.md for the per-camera-day arithmetic.
    thumb_width: int = 320
    thumb_quality: int = 70
    # Where the search API embeds a query. 'cpu' by default and deliberately:
    # the text tower is ~110M parameters, a query is one 64-token sequence, and
    # keeping it off the GPU means a search costs the vision engine nothing at
    # all. Set to 'cuda' only if measured query latency becomes a complaint.
    query_device: str = "cpu"
    # Similarity floor for a result to be returned at all. SigLIP scores are
    # small absolute numbers (a good match on this footage is ~0.10-0.17), so
    # this is deliberately low: it exists to drop noise, not to rank.
    min_score: float = 0.02
    default_limit: int = 24

    def __post_init__(self) -> None:
        if self.sample_interval_sec <= 0:
            raise ConfigError("index.sample_interval_sec must be > 0")
        if self.batch_size < 1:
            raise ConfigError("index.batch_size must be >= 1")
        if self.queue_depth < 1:
            raise ConfigError("index.queue_depth must be >= 1")
        if self.thumb_width < 32:
            raise ConfigError("index.thumb_width must be >= 32")


@dataclass(slots=True)
class EvalConfig:
    """``argus eval`` — the dev-time harness that measures candidate VLMs.

    **This section is data, not behaviour.** It names a third-party base URL and
    an environment variable, and naming them is not calling them: the cloud
    runner lives in ``argus/eval/openrouter.py``, is imported only when the
    operator passes ``--cloud`` *and* the key is present, and nothing in the
    runtime product tree imports it (``tests/test_eval.py`` asserts that by
    importing the engine in a subprocess and checking ``sys.modules``). Keeping
    the candidate list here rather than hard-coded is what lets a reviewer
    change which models get measured without editing Python.
    """

    # Where the label file and the extracted frames live. `frames_dir` is
    # derived data and gitignored: labels.yaml records each frame's source video
    # and timestamp, so a fresh checkout re-extracts what it needs. Human labels
    # are precious and versioned; JPEGs of a video already in the repo are not.
    labels_path: Path = Path("evals/labels.yaml")
    frames_dir: Path = Path("evals/frames")
    results_dir: Path = Path("evals")
    report_path: Path = Path("docs/VLM_EVAL.md")
    # Frame sampling for `--build-labels`: one frame every N seconds of source,
    # capped per video so a long NVR export cannot produce a label file nobody
    # will ever finish filling in.
    sample_every_sec: float = 20.0
    max_frames_per_video: int = 12
    # Sequential by default, and deliberately: the harness reports latency, and
    # a latency measured at concurrency 4 describes a load test rather than the
    # per-evaluation cost the policy budget in docs/PHASE2_NOTES.md is built on.
    concurrency: int = 1
    max_tokens: int = 192
    # The cloud candidates get their own, far larger budget, and it is not a
    # generosity: OpenRouter counts a reasoning model's hidden chain of thought
    # against `max_tokens`, and every candidate below is a reasoning model. At
    # 192 the 2026-08-25 run spent the entire budget thinking and returned
    # `content: null` on 91% of calls. Measured ceiling across the four is ~470
    # completion tokens with reasoning bounded; 1536 leaves room for a model
    # that thinks harder about a hard frame. The local runner keeps 192, which
    # is what the appliance actually grants it.
    cloud_max_tokens: int = 1536
    timeout_sec: float = 120.0
    # Cloud candidates, comma-separated (same idiom as api.cors_origins). All
    # four verified vision-capable on the OpenRouter catalog 2026-08-15.
    # z-ai/glm-5.2 is deliberately absent: it is text-only there.
    cloud_models: str = (
        "qwen/qwen3.8-27b,qwen/qwen3.8-max,google/gemini-3.7-flash,moonshotai/kimi-k3"
    )
    cloud_base_url: str = "https://openrouter.ai/api/v1"
    cloud_key_env: str = "OPENROUTER_API_KEY"

    @property
    def cloud_model_list(self) -> list[str]:
        return [model.strip() for model in self.cloud_models.split(",") if model.strip()]


@dataclass(slots=True)
class ArgusConfig:
    data_dir: Path = Path("data")
    # Built SPA (vite build output). Served at / by `argus api` when it exists;
    # a source checkout with no build simply serves the JSON API alone.
    ui_dist: Path = Path("ui/dist")
    site: SiteConfig = field(default_factory=SiteConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    logs: LogsConfig = field(default_factory=LogsConfig)
    smtp: SmtpConfig = field(default_factory=SmtpConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    vlm: VlmConfig = field(default_factory=VlmConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Where this config was loaded from (None when built from defaults).
    source_path: Path | None = None

    # --- derived paths ----------------------------------------------------
    # Everything the appliance writes lives under data_dir so retention and
    # backup have exactly one directory to reason about.

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db" / "argus.sqlite3"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def tracks_dir(self) -> Path:
        return self.data_dir / "tracks"

    @property
    def ring_dir(self) -> Path:
        return self.data_dir / "ring"

    @property
    def gateway_dir(self) -> Path:
        return self.data_dir / "gateway"

    @property
    def index_dir(self) -> Path:
        """Thumbnails for the periodic index samples.

        Under data_dir because unlike the model weights these *are* appliance
        state: retention sweeps them, and a support bundle should see how big
        they got.
        """
        return self.data_dir / "index"

    @property
    def index_db_path(self) -> Path:
        """The vector store, beside the appliance database rather than inside
        it. See argus/index/store.py for why they are two files."""
        return self.data_dir / "db" / "index.sqlite3"

    @property
    def live_dir(self) -> Path:
        """The engine's request-gated frame publication (argus/live.py).

        Not evidence and not swept by retention: it holds at most one JPEG per
        camera, each overwritten in place, plus the request file. The engine
        empties it on shutdown.
        """
        return self.data_dir / "live"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def backups_dir(self) -> Path:
        """Nightly database snapshots (argus/backup.py).

        Under data_dir by default so one directory still answers "what is this
        appliance storing", and deliberately *not* a retention target — the
        backup job keeps its own generations, and an age sweep that deleted
        backups on the same schedule as evidence would delete the record of the
        evidence along with it. ``backup.directory`` moves it to another
        filesystem, which is the only way a copy survives this one failing.
        """
        configured = (self.backup.directory or "").strip()
        return _resolve_path(Path(configured)) if configured else self.data_dir / "backups"

    @property
    def models_dir(self) -> Path:
        """Detector weights. Under data_dir so backup/retention see them too."""
        return self.data_dir / "models"

    @property
    def detector_checkpoint_path(self) -> Path | None:
        """The fine-tuned detector weights, if this site has any.

        ``None`` is the shipped answer and the one every unconfigured appliance
        gives. A relative value resolves against the repo root, the same rule
        ``ui_dist`` and ``gateway.binary`` follow, so `models/finetuned/<date>/…`
        works from any working directory — including the systemd unit's.

        Not under ``data_dir``: like the VLM checkpoint, trained weights are an
        install artefact of a decision somebody made, and retention must never
        sweep the model the plant is currently counting people with.
        """
        configured = (self.detection.model.checkpoint or "").strip()
        return _resolve_path(Path(configured).expanduser()) if configured else None

    @property
    def engine_stats_path(self) -> Path:
        """Where ``argus run`` publishes its live per-camera statistics.

        The engine and the API are separate processes; the database is the bus
        for anything durable, but achieved fps and decoder state are neither
        durable nor worth a schema. One atomically-replaced JSON file per report
        tick is enough, is trivially inspectable with `cat`, and its mtime is
        how the API knows whether the engine is still alive.
        """
        return self.data_dir / "engine_stats.json"

    @property
    def ui_dist_dir(self) -> Path:
        """Absolute path to the built SPA. Relative values resolve against the
        repo root so `argus api` finds it from any working directory."""
        return _resolve_path(Path(self.ui_dist))

    @property
    def gateway_binary(self) -> Path:
        binary = Path(self.gateway.binary).expanduser()
        return binary if binary.is_absolute() else ROOT / binary

    # --- VLM paths --------------------------------------------------------
    # These live outside data_dir on purpose. A 17 GB checkpoint and a 12 GB
    # virtualenv are install artefacts, not appliance state: retention must not
    # sweep them and a support bundle must not tar them up.

    @property
    def vlm_venv(self) -> Path:
        venv = Path(self.vlm.server.venv).expanduser()
        return venv if venv.is_absolute() else ROOT / venv

    @property
    def vlm_models_dir(self) -> Path:
        directory = Path(self.vlm.server.models_dir).expanduser()
        return directory if directory.is_absolute() else ROOT / directory

    @property
    def vlm_model_path(self) -> Path:
        """Where ``scripts/fetch_vlm.sh`` put the served checkpoint.

        vLLM is given this path rather than the bare HuggingFace id so that a
        running appliance never reaches for the network — the "fully local at
        runtime" claim is enforced here, not merely intended.
        """
        return self.vlm_models_dir / self.vlm.model

    @property
    def index_model_path(self) -> Path:
        """The SigLIP checkpoint, fetched by scripts/fetch_siglip.sh into the
        same ``models/`` tree as the VLM — an install artefact, not state."""
        return self.vlm_models_dir / self.index.model

    # --- eval paths -------------------------------------------------------
    # Outside data_dir, like the model weights: an eval set is a *development*
    # artefact that belongs to the repository and its reviewers, not appliance
    # state that retention should sweep or a support bundle should collect.

    @property
    def eval_labels_path(self) -> Path:
        return _resolve_path(Path(self.eval.labels_path))

    @property
    def eval_frames_dir(self) -> Path:
        return _resolve_path(Path(self.eval.frames_dir))

    @property
    def eval_results_dir(self) -> Path:
        return _resolve_path(Path(self.eval.results_dir))

    @property
    def eval_report_path(self) -> Path:
        return _resolve_path(Path(self.eval.report_path))

    @property
    def gateway_url(self) -> str:
        """Where *this process* reaches go2rtc's HTTP API.

        Since Phase 5 that is loopback by default and nothing else can reach it
        — the browser goes through `/api/gateway/*` on the authenticated API.
        A `0.0.0.0` here still resolves to 127.0.0.1 for our own client, because
        a bind address is not an address you connect to.
        """
        host = self.gateway.api_host
        return f"http://{'127.0.0.1' if host in ('0.0.0.0', '::', '') else host}:{self.gateway.api_port}"

    def rtsp_url(self, stream_name: str) -> str:
        """Internal RTSP URL the decoders pull from for a gateway stream."""
        host = "127.0.0.1" if self.gateway.listen_host in ("0.0.0.0", "::") else self.gateway.listen_host
        return f"rtsp://{host}:{self.gateway.rtsp_port}/{stream_name}"

    def ensure_dirs(self) -> None:
        for path in (
            self.db_path.parent,
            self.snapshots_dir,
            self.clips_dir,
            self.tracks_dir,
            self.ring_dir,
            self.gateway_dir,
            self.live_dir,
            self.logs_dir,
            self.models_dir,
            self.index_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}


def _coerce(value: Any, target: Any, where: str) -> Any:
    """Coerce a YAML/env value to the annotated type of a dataclass field."""
    # `float | None` (retention.evidence.*): an explicit YAML null means "no
    # per-stage window here, inherit the flat one", and anything else is coerced
    # as the underlying scalar. Without this branch the union fell through to
    # the structural passthrough at the bottom and `shadow_days: banana` became
    # the string "banana" — a typo that reached the sweeper instead of the
    # loader, which is the one place this config tree refuses to let typos go.
    if target in (float | None, int | None):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return _coerce(value, float if target == (float | None) else int, where)
    if target is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUTHY:
            return True
        if text in _FALSEY:
            return False
        raise ConfigError(f"{where}: expected a boolean, got {value!r}")
    if target is int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{where}: expected an integer, got {value!r}") from exc
    if target is float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{where}: expected a number, got {value!r}") from exc
    if target is str:
        return str(value)
    if target is Path or target == (Path | None):
        return _resolve_path(Path(str(value)).expanduser())
    # dict[str, list[str]] (smtp.recipients) and anything else structural
    if isinstance(value, dict):
        return {str(k): list(v) if isinstance(v, (list, tuple)) else [str(v)] for k, v in value.items()}
    return value


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _build(cls: type, data: dict[str, Any], prefix: str = "") -> Any:
    """Instantiate a config dataclass from a plain dict, rejecting typos."""
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        section = prefix.rstrip(".") or "config"
        raise ConfigError(f"unknown key(s) in {section}: {', '.join(sorted(unknown))}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        annotation = hints[f.name]
        where = f"{prefix}{f.name}"
        if is_dataclass(annotation):
            if not isinstance(value, dict):
                raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
            kwargs[f.name] = _build(annotation, value, prefix=f"{where}.")
        else:
            kwargs[f.name] = _coerce(value, annotation, where)
    return cls(**kwargs)


def _apply_env_overrides(obj: Any, env: dict[str, str], prefix: str = "") -> None:
    """Walk the dataclass tree and apply ARGUS_<DOTTED_PATH> overrides in place."""
    hints = get_type_hints(type(obj))
    for f in fields(obj):
        annotation = hints[f.name]
        path = f"{prefix}{f.name}"
        current = getattr(obj, f.name)
        if is_dataclass(annotation):
            _apply_env_overrides(current, env, prefix=f"{path}.")
            continue
        env_key = ENV_PREFIX + path.upper().replace(".", "_")
        if env_key in env:
            setattr(obj, f.name, _coerce(env[env_key], annotation, env_key))


def load_config(path: str | Path | None = None, env: dict[str, str] | None = None) -> ArgusConfig:
    """Load configuration from YAML + environment.

    ``path`` defaults to ``$ARGUS_CONFIG`` and then to ``config/argus.yaml``.
    A missing file is not an error — the dataclass defaults are a valid config.
    """
    env = dict(os.environ if env is None else env)

    # An explicitly requested file that is missing is a mistake worth failing on;
    # a missing *default* file just means "run on the built-in defaults".
    explicit = path is not None or "ARGUS_CONFIG" in env
    if path is None:
        path = env.get("ARGUS_CONFIG") or DEFAULT_CONFIG_PATH
    config_path = Path(path).expanduser()

    data: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{config_path}: top level must be a mapping")
        data = loaded
    elif explicit and config_path != DEFAULT_CONFIG_PATH:
        raise ConfigError(f"config file not found: {config_path}")

    # source_path is bookkeeping, not a user-settable key.
    data.pop("source_path", None)

    config = _build(ArgusConfig, data)
    _apply_env_overrides(config, env)
    config.data_dir = _resolve_path(Path(config.data_dir))
    config.source_path = config_path if config_path.exists() else None
    return config
