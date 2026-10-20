"""The VLM policy engine: sample, ask, judge, prove.

One object — :class:`VlmRunner` — owns everything Phase 2 adds to the live path:

* an asyncio loop on its own thread, so nothing here can stall the vision loop;
* one schedule per (``kind='vlm'`` policy x enabled camera), each firing on the
  policy's own ``interval_sec`` and honouring its own ``cooldown_sec``;
* the verification pass, where a detector event on its way to an auto_routed
  email is shown to the model first — cropped to the zone whose limit it claims
  was broken — and may be vetoed;
* degradation, which is the part that decides whether this is a feature or a
  liability.

**Why a tick loop rather than a task per policy.** ``_tick`` takes the current
time and evaluates whatever is due. That makes the entire schedule — intervals,
cooldowns, backoff — testable by calling it with synthetic timestamps, with no
sleeping and no GPU. A task-per-policy design would be marginally tidier at
runtime and untestable without wall-clock waits, which is the wrong trade for
the component whose job is to *not* misfire.

**Degradation is the contract.** If the server is unreachable, VLM policies mark
themselves degraded and keep the appliance running: detector policies are
untouched (they never enter this module), the engine keeps decoding, tracking,
counting and alerting, and ``/api/health`` says plainly that the VLM is down and
for how long. Retries back off. A verification pass that cannot reach the model
**lets the alert through** — the failure we accept is a false alarm, never a
missed one.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from argus.config import ArgusConfig
from argus.db import Database, Event, decoded_meta, scope_zones
from argus.media.annotate import save_snapshot
from argus.vision.debounce import Debouncer
from argus.vlm import continuity
from argus.vlm.client import VlmClient, VlmUnavailable
from argus.vlm.evidence import (
    annotate_policy_event,
    annotate_veto,
    box_in_frame,
    box_meta,
    crop_to_zone,
    zone_crop,
)
from argus.vlm.schema import PolicyVerdict, VlmResponseError, verification_prompt

# How often the loop wakes to look for due work. Well below the shortest
# sensible policy interval (6 s in the seed), so a policy fires within half a
# second of its due time without the loop spinning.
_TICK_SEC = 0.5

# Cameras, zones and policies are re-read on this cadence, matching the engine's
# own refresh so an operator editing a policy sees it take effect in one window
# rather than two.
_REFRESH_SEC = 15.0

# The detector key that marks a prompt policy, from config/seed.yaml.
VLM_DETECTOR_KEY = "vlm_prompt"


@dataclass
class Schedule:
    """One policy evaluated against one camera — or one zone on one camera.

    A policy scoped to a zone (migration 5, ``policy_scopes``) gets its own
    schedule per zone rather than one per camera, so its interval, its cooldown
    and its counters are per zone too. That is not bookkeeping: "no phones at
    the press" and "no phones at packing" on the same camera are two questions,
    and a shared cooldown would let the first answer silence the second.
    """

    policy_id: str
    camera_id: str
    label: str
    prompt: str
    interval_sec: float
    cooldown_sec: float
    severity: str
    stage: str
    routed_to: str | None
    # None = the whole camera view, which is what an unscoped policy gets.
    zone_id: str | None = None
    zone_name: str | None = None
    zone_polygon: list[list[float]] | None = None
    next_due: float = 0.0
    evaluations: int = 0
    events: int = 0
    failures: int = 0
    skipped_no_frame: int = 0
    # Samples dropped because the previous one had not come back yet. A healthy
    # number here is 0; a rising one means the model is slower than the policy's
    # interval, and the effective sampling rate is the model's, not the policy's.
    skipped_in_flight: int = 0
    in_flight: bool = False
    # Triggered verdicts folded into a condition already on file rather than
    # raised as a new row. `events` counts incidents; this counts sightings, and
    # on a policy watching something that does not move the second number is the
    # one that grows.
    observations: int = 0
    # Conditions this schedule has closed, and why the last one ended. An
    # operator asking "did the mess get cleaned up or did the camera drop?" is
    # asking about this field.
    closed: int = 0
    last_close_reason: str | None = None
    last_verdict: dict[str, Any] | None = None
    last_error: str | None = None
    # Whether the last evaluation actually got a crop. A zone can be scoped and
    # still be shown wide — `crop_to_zone` declines a zone covering most of the
    # frame — and the console has to be able to say which happened.
    last_cropped: bool | None = None

    @property
    def key(self) -> str:
        return (
            f"{self.policy_id}:{self.camera_id}"
            if self.zone_id is None
            else f"{self.policy_id}:{self.camera_id}:{self.zone_id}"
        )

    def as_dict(self, *, degraded: bool) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "cropped": self.last_cropped,
            "key": self.key,
            "label": self.label,
            "interval_sec": self.interval_sec,
            "cooldown_sec": self.cooldown_sec,
            "stage": self.stage,
            "evaluations": self.evaluations,
            "events": self.events,
            # Since migration 7 `events` is a count of *conditions* — one row
            # per incident, however long it lasted — and these two say how much
            # of the plant's day that one row is standing for.
            "observations": self.observations,
            "closed": self.closed,
            "last_close_reason": self.last_close_reason,
            "failures": self.failures,
            "skipped_no_frame": self.skipped_no_frame,
            "skipped_in_flight": self.skipped_in_flight,
            "last_verdict": self.last_verdict,
            "last_error": self.last_error,
            # A policy is degraded because its *server* is, not because of
            # anything about the policy — but this is where an operator looks.
            "degraded": degraded,
        }


class VlmRunner:
    """The Phase-2 addition to ``argus run``.

    The engine holds one of these and touches it in exactly three places: it
    offers frames, it routes auto-routed events through :meth:`dispatch`, and it
    folds :meth:`stats` into its published statistics. Everything else —
    threading, scheduling, HTTP, retries — is behind this class.
    """

    def __init__(
        self,
        config: ArgusConfig,
        database: Database,
        *,
        client: VlmClient | None = None,
        notifier: Any | None = None,
        # Called after an event row exists, with the frame it was raised on, so
        # the engine can cut its evidence clip and index it. The engine owns
        # both the ring buffers and the index runner, so it owns this; passing
        # a callable keeps `argus.vlm` free of any import from `argus.engine`
        # (and of any import from `argus.index`, which is the whole reason the
        # *frame* travels with the id rather than the runner reaching for it).
        on_event: Callable[[str, int, float, Any, str | None], None] | None = None,
        log: Callable[..., None] = print,
    ):
        self.config = config
        self.db = database
        # The `vlm:` config section. Named `settings` rather than `vlm` because
        # the engine's attribute of that name is *this object*, and two different
        # `.vlm` meanings one call apart is how the wrong one gets read.
        self.settings = config.vlm
        self.client = client or VlmClient(self.settings)
        self.notifier = notifier
        self.on_event = on_event
        self.log = log

        self.schedules: dict[str, Schedule] = {}
        self.debouncer = Debouncer(cooldown_sec=self.settings.default_interval_sec)
        # Ongoing conditions (migration 7). The last time the stale sweep ran;
        # it runs on the refresh cadence rather than every 0.5 s tick, because
        # it is a scan of open conditions and nothing can go stale inside one
        # tick that was not already stale in the last.
        self._last_stale_sweep = 0.0

        # Server state, as seen from here.
        self.available: bool = False
        self.model_name: str | None = None
        self.last_unavailable_reason: str | None = None
        self.degraded_since: float | None = None
        self._backoff_sec: float = self.settings.retry_backoff_sec
        self._retry_at: float = 0.0

        self.verifications = 0
        self.vetoes = 0
        self.verify_failures = 0

        self._frames: dict[str, Any] = {}
        self._frames_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._last_refresh = 0.0
        self._tasks: set[asyncio.Task] = set()

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Bring up the loop thread. Never raises: a VLM that will not start is
        a degraded appliance, not a dead one."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="vlm")
        self._thread.start()
        # Wait briefly for the loop to exist so `dispatch` from the vision
        # thread has somewhere to submit to; do not wait for the *server*.
        self._ready.wait(timeout=5.0)

    def _run_loop(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:  # noqa: BLE001 - the engine must survive this thread
            self.log(f"vlm: scheduler stopped ({type(exc).__name__}: {exc})")

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._ready.set()
        await self._probe()
        while not self._stopping.is_set():
            now = time.time()
            try:
                await self._tick(now)
            except Exception as exc:  # noqa: BLE001 - one bad tick must not end the run
                self.log(f"vlm: tick failed ({type(exc).__name__}: {exc})")
            await asyncio.sleep(_TICK_SEC)
        # Let in-flight evaluations finish so an event that was already decided
        # still gets its row and its snapshot.
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=10.0)

    def stop(self, timeout: float = 10.0) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._loop = None

    # --- frame intake -----------------------------------------------------

    def offer(self, frame: Any) -> None:
        """Publish a camera's newest frame. Called from the engine's decode path.

        Deliberately the cheapest thing that can work: one dict assignment under
        an uncontended lock, on a path that runs at ``fps x cameras``. The
        scheduler samples from here at its own leisurely interval, so the frames
        in between cost nothing beyond being overwritten.
        """
        with self._frames_lock:
            self._frames[frame.camera_id] = frame

    def latest_frame(self, camera_id: str) -> Any | None:
        with self._frames_lock:
            return self._frames.get(camera_id)

    # --- server state -----------------------------------------------------

    async def _probe(self) -> bool:
        """Check the server and update degradation state."""
        report = await self.client.health(timeout=5.0)
        if report.get("available"):
            if not self.available:
                self.log(f"vlm: {report.get('model')} at {self.client.base_url} — ready")
            self.available = True
            self.model_name = report.get("model")
            self.last_unavailable_reason = None
            self.degraded_since = None
            self._backoff_sec = self.settings.retry_backoff_sec
            return True
        self._mark_unavailable(str(report.get("reason") or "unreachable"))
        return False

    def _mark_unavailable(self, reason: str) -> None:
        first = self.available or self.degraded_since is None
        self.available = False
        self.last_unavailable_reason = reason[:300]
        if self.degraded_since is None:
            self.degraded_since = time.time()
        if first:
            self.log(f"vlm: degraded — {reason[:200]}")
        self._retry_at = time.time() + self._backoff_sec
        self._backoff_sec = min(self._backoff_sec * 2, self.settings.retry_backoff_max_sec)

    # --- schedule ---------------------------------------------------------

    def refresh(self, now: float | None = None) -> None:
        """Rebuild the (policy x camera x zone) schedule from the database.

        Existing schedules keep their counters and their next due time, so an
        unrelated policy edit does not reset every camera's sampling phase.

        **Scope is applied here and nowhere else.** A policy with no scope rows
        runs against every enabled camera exactly as it did before migration 5;
        a scoped one runs only where it was told, which is what turns "three
        camera-policy pairs at an 8 s interval" from a fleet-wide ceiling into a
        per-policy budget an author controls.
        """
        now = time.time() if now is None else now
        policies = [
            policy for policy in self.db.list_policies(
                kind="vlm", enabled_only=True, active_only=True
            )
            if policy.get("prompt")
        ]
        cameras = self.db.list_cameras(enabled_only=True)
        zones = {zone["id"]: zone for zone in self.db.list_zones()}
        wanted: dict[str, Schedule] = {}
        for policy in policies:
            interval = float(policy.get("interval_sec") or self.settings.default_interval_sec)
            cooldown = policy.get("cooldown_sec")
            cooldown = (
                self.config.evidence.default_cooldown_sec if cooldown is None else float(cooldown)
            )
            scope = policy.get("scope") or []
            for camera in cameras:
                for zone_id in scope_zones(scope, str(camera["id"])):
                    zone = zones.get(zone_id) if zone_id else None
                    schedule = Schedule(
                        policy_id=str(policy["id"]),
                        camera_id=str(camera["id"]),
                        label=str(policy.get("label") or policy["id"]),
                        prompt=str(policy["prompt"]),
                        interval_sec=interval,
                        cooldown_sec=cooldown,
                        severity=str(policy.get("severity") or "medium"),
                        stage=str(policy.get("stage") or "shadow"),
                        routed_to=policy.get("routed_to"),
                        zone_id=zone_id,
                        zone_name=str(zone["name"]) if zone else None,
                        zone_polygon=list(zone["polygon"]) if zone else None,
                        # Stagger the first evaluation across cameras so a fleet
                        # does not send every policy's first frame in one burst.
                        next_due=now + len(wanted) * 0.5,
                    )
                    existing = self.schedules.get(schedule.key)
                    if existing is not None:
                        # Pick up edits without losing the phase or the counters.
                        existing.prompt = schedule.prompt
                        existing.label = schedule.label
                        existing.interval_sec = interval
                        existing.cooldown_sec = cooldown
                        existing.severity = schedule.severity
                        existing.stage = schedule.stage
                        existing.routed_to = schedule.routed_to
                        existing.zone_name = schedule.zone_name
                        existing.zone_polygon = schedule.zone_polygon
                        schedule = existing
                    wanted[schedule.key] = schedule

        for gone in set(self.schedules) - set(wanted):
            self.debouncer.forget(gone)
        self.schedules = wanted

    # --- the tick ---------------------------------------------------------

    async def _tick(self, now: float) -> None:
        """Evaluate everything due at ``now``. The whole scheduler, in one place."""
        if now - self._last_refresh >= _REFRESH_SEC:
            self._last_refresh = now
            await asyncio.to_thread(self.refresh, now)

        # Before anything else, and before the degraded early-return below: a
        # condition goes stale precisely *because* nothing is observing it, and
        # the commonest reason nothing is observing it is that the model is
        # down. Sweeping after the `available` check would leave every open
        # condition on the feed indefinitely for the whole duration of an
        # outage, which is the one time the feed most needs to stop claiming
        # things are ongoing.
        if now - self._last_stale_sweep >= _REFRESH_SEC:
            self._last_stale_sweep = now
            await asyncio.to_thread(self.close_stale_conditions, now)

        if not self.available:
            if now < self._retry_at:
                return
            if not await self._probe():
                return

        for schedule in list(self.schedules.values()):
            if now < schedule.next_due:
                continue
            schedule.next_due = now + schedule.interval_sec
            if schedule.in_flight:
                # The previous sample has not been answered yet. Drop this one
                # rather than queue it — the same rule the decoder applies to
                # frames, and for the same reason: falling behind must degrade
                # the sampling rate, never build an unbounded backlog of
                # questions about frames that are already minutes old. Without
                # this, a policy whose interval is shorter than the model's
                # latency enqueues faster than the server drains, forever.
                schedule.skipped_in_flight += 1
                continue
            frame = self.latest_frame(schedule.camera_id)
            if frame is None:
                # The camera has not produced a frame yet (or at all). Not an
                # error and not a failure — just nothing to look at.
                schedule.skipped_no_frame += 1
                continue
            schedule.in_flight = True
            task = asyncio.create_task(self._evaluate(schedule, frame))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _evaluate(self, schedule: Schedule, frame: Any) -> None:
        """One policy, one frame: ask, then raise an event if it fires."""
        try:
            await self._evaluate_inner(schedule, frame)
        finally:
            schedule.in_flight = False

    def _view(self, schedule: Schedule, image: Any) -> tuple[Any, tuple[float, ...] | None]:
        """What the model is shown for this schedule, and where it came from.

        A zone-scoped policy is asked about the zone, for the reason the
        verification pass is (docs/VLM_EVAL.md): the same model, the same prompt
        and the same footage answered a counting question 12/12 on zone crops
        and 3/12 on the wide shots they were cut from. A zone that covers most
        of the frame is left alone — there is no sub-picture to find — and so is
        a polygon that cannot be read, because a policy that silently stopped
        being evaluated over a bad zone is worse than one evaluated wide.
        """
        if not schedule.zone_id or not schedule.zone_polygon:
            return image, None
        try:
            found = zone_crop(image, schedule.zone_polygon)
        except Exception as exc:  # noqa: BLE001 - a bad crop must not cost the evaluation
            self.log(f"vlm: could not crop {schedule.key} to its zone ({exc})")
            return image, None
        if found is None:
            return image, None
        return found

    async def _evaluate_inner(self, schedule: Schedule, frame: Any) -> None:
        schedule.evaluations += 1
        view, crop = await asyncio.to_thread(self._view, schedule, frame.image)
        schedule.last_cropped = crop is not None
        try:
            verdict = await self.client.evaluate(schedule.prompt, view)
        except VlmUnavailable as exc:
            schedule.failures += 1
            schedule.last_error = str(exc)[:300]
            self._mark_unavailable(str(exc))
            return
        except VlmResponseError as exc:
            # The server is fine, the answer was not. Counted and dropped —
            # the next sample is 6-8 seconds away.
            schedule.failures += 1
            schedule.last_error = f"unreadable answer: {exc}"[:300]
            self.log(f"vlm: {schedule.key} — {schedule.last_error}")
            return
        except Exception as exc:  # noqa: BLE001 - never kill the loop for one frame
            schedule.failures += 1
            schedule.last_error = f"{type(exc).__name__}: {exc}"[:300]
            self.log(f"vlm: {schedule.key} evaluation failed — {schedule.last_error}")
            return

        if crop is not None and verdict.box_2d is not None:
            # The model saw the crop, so its box is in the crop's coordinates.
            # Mapped back before anything stores or draws it, because the two
            # consumers downstream — the snapshot and `events.meta` — both treat
            # box_2d as a fraction of the whole frame.
            verdict.box_2d = box_in_frame(verdict.box_2d, crop)  # type: ignore[arg-type]

        schedule.last_error = None
        schedule.last_verdict = verdict.as_dict()
        if not verdict.fires(self.settings.min_confidence):
            # A verdict the model answered `false` to is positive evidence that
            # the condition ended, and it is the only evidence of that kind this
            # appliance ever gets — so it closes the open condition here.
            #
            # A verdict that is `true` but under `min_confidence` does neither.
            # It is not enough to raise on and it is not a claim that the mess
            # was cleared; treating "the model is unsure" as "it is over" would
            # end a real condition on the model's hesitation, and the reader
            # would see a span that closed while the laminate was still there.
            if not verdict.triggered:
                await asyncio.to_thread(self._close_condition, schedule, frame.ts, "cleared")
            return
        try:
            await asyncio.to_thread(self._record, schedule, frame, verdict)
        except Exception as exc:  # noqa: BLE001
            schedule.failures += 1
            schedule.last_error = f"emit failed: {type(exc).__name__}: {exc}"[:300]
            self.log(f"vlm: {schedule.key} — {schedule.last_error}")

    # --- ongoing conditions ------------------------------------------------
    #
    # A prompt policy asks about a state, and a state persists: "loose laminate
    # on the wrapper apron" was true at 20:22 and still true at 20:37. Until
    # migration 7 the only thing between that and a new row was `cooldown_sec`,
    # so one apron wrote 140 events on this box — 40 of them in a night. The
    # cooldown was rate-limiting a question about *identity*.
    #
    # So a triggered verdict either continues the condition already open for its
    # (policy, camera, zone) — extending the row, its `last_seen_ts` and its
    # `observations` — or opens a new one. The cooldown still governs the second
    # case and no longer touches the first, which is why a persistent mess is now
    # one card that grows rather than a column of near-identical ones.
    #
    # **The debouncer is not consulted on an extension, and is not reset by a
    # close.** Its clock therefore runs from the last time a condition *opened*,
    # not from the last time one was reported. That is the conservative reading
    # and it is chosen deliberately: forgetting the key on close would leave a
    # flapping policy — fires, clears, fires, clears at its own interval — with
    # no rate limit at all, which is the alert storm `cooldown_sec` exists to
    # prevent. The price is that a genuinely new incident arriving inside the
    # previous one's cooldown waits for it, exactly as it did before this
    # feature; nothing about that case got worse.
    #
    # The identity test itself is in `argus/vlm/continuity.py`, in pure functions
    # over plain values, so it can be argued with in a test.

    def _record(self, schedule: Schedule, frame: Any, verdict: PolicyVerdict) -> int | None:
        """One triggered verdict: extend the condition on file, or open one.

        Returns the event id it landed on, or None when nothing was written —
        which happens when the verdict belongs to a *different* condition from
        the open one and the cooldown has not yet allowed a second to be opened.
        """
        settings = self.config.vlm.continuity
        if not settings.enabled:
            # The pre-migration-7 behaviour, kept reachable: a row per cooldown,
            # no condition columns, nothing to close. A site that wants its
            # events to stay one-per-alert can have that, and the flag is what
            # makes the new behaviour a decision rather than an upgrade.
            if not self.debouncer.should_emit(schedule.key, frame.ts, schedule.cooldown_sec):
                return None
            return self._emit(schedule, frame, verdict, condition=False)

        open_row = self.db.open_condition(schedule.key)
        if open_row is not None:
            match = continuity.continues(
                last_seen_ts=float(open_row.get("last_seen_ts") or open_row["ts"]),
                ts=frame.ts,
                stale_after_sec=settings.stale_after_sec,
                open_box=continuity.box_of({"meta": decoded_meta(open_row)}),
                new_box=verdict.box_2d,
                zone_scoped=schedule.zone_id is not None,
                min_iou=settings.min_iou,
            )
            if match:
                return self._extend(schedule, frame, verdict, open_row)
            # The condition on file is not this one. Do not close it *until*
            # this verdict can actually become a row: closing first and then
            # being refused by the cooldown would retire the card on the feed
            # and put nothing in its place, so the reader would watch a live
            # problem disappear.
            if not self.debouncer.should_emit(schedule.key, frame.ts, schedule.cooldown_sec):
                return None
            reason = "stale" if match.reason == continuity.STALE else "moved"
            self._close_row(schedule, int(open_row["id"]),
                            float(open_row.get("last_seen_ts") or open_row["ts"]), reason)
            return self._emit(schedule, frame, verdict, condition=True)

        if not self.debouncer.should_emit(schedule.key, frame.ts, schedule.cooldown_sec):
            return None
        return self._emit(schedule, frame, verdict, condition=True)

    def _extend(self, schedule: Schedule, frame: Any, verdict: PolicyVerdict,
                open_row: dict[str, Any]) -> int:
        """Fold one more sighting into the condition already on file.

        Three things are deliberately *not* done here.

        **No new row**, which is the whole feature.

        **No second alert.** ``notifier.dispatch`` is not called: the condition
        was already routed when it opened, and an email every cooldown for a
        problem the recipient has already been told about is the fatigue this
        replaces.

        **No second clip.** ``on_event`` is not called either. The ±8 s clip is
        evidence of the *onset* — the moment the apron went from clean to not —
        and re-cutting one every sample would spend the ring buffer on the
        middle of a thing whose beginning is the part worth keeping.

        The snapshot *is* refreshed, in place, over the same path. That keeps
        one evidence file per condition rather than one per sighting (which
        would have been forty JPEGs for one offcut) and it is why the console
        cache-busts the image with `last_seen_ts`.
        """
        event_id = int(open_row["id"])
        snapshot_rel = open_row.get("snapshot_path")
        if snapshot_rel:
            try:
                self._write_snapshot(schedule, frame, verdict, str(snapshot_rel))
            except Exception as exc:  # noqa: BLE001 - the row is what matters
                self.log(f"vlm: could not refresh evidence for condition {event_id} ({exc})")

        meta = decoded_meta(open_row) or {}
        height, width = frame.image.shape[:2]
        # The opening sentence is kept once, here, because `description` is
        # about to be overwritten with the newest one and "what did the model
        # say when this started" is the first question asked of a long-running
        # condition.
        meta.setdefault("opened", {
            "ts": float(open_row["ts"]),
            "reason": open_row.get("description"),
            "confidence": open_row.get("confidence"),
        })
        meta.update({
            "reason": verdict.reason,
            "latency_sec": round(verdict.latency_sec, 3),
            "prompt_tokens": verdict.prompt_tokens,
            "completion_tokens": verdict.completion_tokens,
            "frame": {"width": width, "height": height},
            "zone_cropped": bool(schedule.last_cropped),
            **(box_meta(verdict) or {"box_2d": None}),
        })
        row = self.db.observe_condition(
            event_id,
            ts=frame.ts,
            description=verdict.reason or schedule.label,
            confidence=round(verdict.confidence, 4),
            meta=meta,
            snapshot_path=str(snapshot_rel) if snapshot_rel else None,
        )
        schedule.observations += 1
        # The count as the *database* now holds it, not as this process
        # calculated it: `observations = observations + 1` is done in SQL, and
        # logging a number computed here would be logging a guess about it.
        observations = int((row or {}).get("observations") or 0)
        self.log(
            f"condition {event_id}: {schedule.camera_id} / {schedule.label} still true — "
            f"observation {observations}, "
            f"ongoing {frame.ts - float(open_row['ts']):.0f}s"
        )
        return event_id

    def _close_condition(self, schedule: Schedule, ts: float, reason: str) -> None:
        """Close whatever condition this schedule has open, if it has one."""
        if not self.config.vlm.continuity.enabled:
            return
        open_row = self.db.open_condition(schedule.key)
        if open_row is None:
            return
        self._close_row(schedule, int(open_row["id"]), ts, reason)

    def _close_row(self, schedule: Schedule, event_id: int, closed_at: float,
                   reason: str) -> None:
        if self.db.close_condition(event_id, closed_at=closed_at, reason=reason) == 0:
            return
        schedule.closed += 1
        schedule.last_close_reason = reason
        self.log(f"condition {event_id}: {schedule.camera_id} / {schedule.label} closed ({reason})")

    def close_stale_conditions(self, now: float) -> list[int]:
        """End conditions the appliance has heard nothing about for too long.

        Called on the refresh cadence, and on the first tick after a restart —
        which is the case that matters most. An appliance that is stopped with
        conditions open would otherwise come back claiming they are all still
        ongoing, and the span it eventually wrote would cover the hours it spent
        switched off.

        ``now`` is wall-clock, and so is the column this is measured against;
        see :meth:`Database.stale_conditions` for why it is not the frame clock.
        The span is still closed at ``last_seen_ts`` — the last moment the
        condition was known *true* — because "when did the problem end" and
        "when did we stop hearing about it" are different questions and only the
        first belongs on the card.

        Returns the ids it closed so a caller (and the test) can see them; the
        console hears about them over the SSE stream's condition cursor.
        """
        if not self.config.vlm.continuity.enabled:
            return []
        cutoff = now - self.config.vlm.continuity.stale_after_sec
        closed: list[int] = []
        for row in self.db.stale_conditions(cutoff):
            key = str(row.get("condition_key") or "")
            schedule = self.schedules.get(key)
            last_seen = float(row.get("last_seen_ts") or row["ts"])
            if schedule is not None:
                self._close_row(schedule, int(row["id"]), last_seen, "stale")
            elif self.db.close_condition(int(row["id"]), closed_at=last_seen, reason="stale"):
                # The policy or the camera it belonged to is gone. Still closed,
                # and still logged: an unclosable condition would sit "ongoing"
                # on the feed forever with nothing left that could ever end it.
                self.log(f"condition {row['id']}: closed (stale, schedule {key or '?'} gone)")
            closed.append(int(row["id"]))
        return closed

    # --- events -----------------------------------------------------------

    def _banner(self, schedule: Schedule) -> str:
        # A middle dot, not an em dash: this string is *drawn into the
        # snapshot*, and the bundled font has no em dash (argus/media/
        # annotate.py, `drawable`). The dot is also the separator the console
        # and the masthead already use.
        return (schedule.label if schedule.zone_name is None
                else f"{schedule.label} · {schedule.zone_name}")

    def _write_snapshot(self, schedule: Schedule, frame: Any, verdict: PolicyVerdict,
                        snapshot_rel: str) -> str:
        """Draw and save the evidence frame. Shared by the open and extend paths
        so a refreshed snapshot is the same picture, drawn the same way, as the
        one the condition opened with."""
        # The whole frame, not the crop the model was shown: the box has already
        # been mapped back into frame coordinates, and a reviewer judging "is
        # that person on a phone" needs to see where in the plant they are.
        image = annotate_policy_event(
            frame.image, verdict, label=self._banner(schedule), severity=schedule.severity
        )
        save_snapshot(image, self.config.snapshots_dir / snapshot_rel,
                      quality=self.config.evidence.snapshot_quality)
        return snapshot_rel

    def _emit(self, schedule: Schedule, frame: Any, verdict: PolicyVerdict,
              *, condition: bool = True) -> int:
        """Snapshot, row, clip, alert — the same order, and the same pipeline,
        as a detector event. Evidence exists before anybody is told."""
        label = self._banner(schedule)
        stem = schedule.zone_id or schedule.policy_id
        snapshot_rel = f"{schedule.camera_id}/{stem}_{schedule.policy_id}_{int(frame.ts)}.jpg"
        self._write_snapshot(schedule, frame, verdict, snapshot_rel)

        height, width = frame.image.shape[:2]
        event_id = self.db.insert_event(Event(
            camera_id=schedule.camera_id,
            # Set when the policy is zone-scoped, so triage can draw the polygon
            # and the analytics can group by zone exactly as they do for the
            # detector's events.
            zone_id=schedule.zone_id,
            policy_id=schedule.policy_id,
            detector_name=schedule.policy_id,
            detector_type="vlm",
            severity=schedule.severity,
            confidence=round(verdict.confidence, 4),
            stage=schedule.stage,
            routed_to=schedule.routed_to,
            description=verdict.reason or schedule.label,
            ts=frame.ts,
            snapshot_path=snapshot_rel,
            meta={
                "policy_label": schedule.label,
                "reason": verdict.reason,
                "prompt": schedule.prompt,
                "model": self.model_name or self.settings.model,
                "latency_sec": round(verdict.latency_sec, 3),
                "prompt_tokens": verdict.prompt_tokens,
                "completion_tokens": verdict.completion_tokens,
                "frame": {"width": width, "height": height},
                # What the model was actually shown. A reviewer looking at a
                # wide snapshot and a zone-scoped policy needs to know whether
                # the answer came from the crop or from the whole view.
                "zone_name": schedule.zone_name,
                "zone_cropped": bool(schedule.last_cropped),
                **(box_meta(verdict) or {}),
            },
            # The condition this row *is*. Keyed on the schedule, so a policy
            # scoped to two zones on one camera keeps two independent
            # conditions — the same reason the schedule itself is per zone.
            condition_key=schedule.key if condition else None,
            condition_state="open" if condition else None,
        ))
        schedule.events += 1
        self.log(
            f"event {event_id}: {schedule.camera_id} / {label} — "
            f"{verdict.reason[:80]} (conf {verdict.confidence:.2f}, "
            f"{verdict.latency_sec:.2f}s), stage={schedule.stage}"
        )
        if self.on_event is not None:
            # The *unannotated* frame: the snapshot written above has the policy
            # banner and the model's grounding box drawn on it, and an index
            # that embeds those learns Argus's own overlay rather than the scene.
            self.on_event(schedule.camera_id, event_id, frame.ts, frame.image, snapshot_rel)
        if self.notifier is not None:
            self.notifier.dispatch(event_id)
        return event_id

    # --- verification pass ------------------------------------------------

    def will_verify(self, event: dict[str, Any]) -> bool:
        """Whether this event is a candidate for the double-check."""
        verify = self.settings.verify
        return bool(
            verify.enabled
            and self.settings.enabled
            # Verifying a VLM event with the same VLM is circular; only
            # detector-origin events are double-checked.
            and event.get("detector_type") == "detector"
            and str(event.get("stage")) in verify.stage_list
        )

    def dispatch(self, event_id: int, image: Any, notifier: Any) -> None:
        """Route a detector event to its alert, through the verifier if enabled.

        Called from the vision thread, and must not block it: the verification
        round trip is submitted to the loop thread and the caller returns
        immediately. When verification is off, or this event is not a candidate,
        or the loop is not running, this is exactly ``notifier.dispatch``.
        """
        event = self.db.get_event(event_id)
        if event is None:
            return
        loop = self._loop
        if loop is None or not self.will_verify(event):
            notifier.dispatch(event_id)
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._verify_then_notify(event, image, notifier), loop
            )
        except RuntimeError:
            # Loop shutting down — the alert still has to go out.
            notifier.dispatch(event_id)

    async def _verify_then_notify(self, event: dict[str, Any], image: Any, notifier: Any) -> None:
        event_id = int(event["id"])
        verdict = await self._verify(event, image)
        if verdict is None:
            # Could not ask. Mail it: an unverified alert beats a silent one.
            notifier.dispatch(event_id)
            return

        upheld = verdict.triggered or verdict.confidence < self.settings.verify.min_confidence
        model = f"vlm:{self.model_name or self.settings.model}"
        self.verifications += 1
        if upheld:
            await asyncio.to_thread(
                self.db.record_verification, event_id,
                verified_by=model, confidence=verdict.confidence, suppressed=False,
            )
            notifier.dispatch(event_id)
            return

        self.vetoes += 1
        reason = verdict.reason or "the verifier did not see the violation"
        await asyncio.to_thread(self._suppress, event, verdict, reason, model)
        self.log(
            f"vlm: event {event_id} suppressed — {reason[:100]} "
            f"(conf {verdict.confidence:.2f}); alert not sent"
        )

    async def _zone_view(self, event: dict[str, Any], image: Any) -> Any:
        """The picture the verifier is shown: the zone, when the event has one.

        The claim being audited is about a zone — *"Zone 'Packing floor' over
        sanctioned limit: 5 people (limit 3)"* — and until now the question was
        asked over the whole camera view, which is how it went unanswered: the
        harness measured the same model vetoing 12/12 of those over-claims on a
        zone crop and 3/12 on the wide shot the crop came from
        (docs/VLM_EVAL.md). Crowded wide shots are not countable by inspection,
        by a model or by the person labelling the ground truth.

        Never raises and never returns nothing: every path that cannot produce a
        crop returns the frame it was given, because the alternative — an
        auto-routed alert that stops being verified because a zone was deleted or
        drawn badly — is a failure this pass is supposed to prevent.
        """
        zone_id = event.get("zone_id")
        if not zone_id:
            return image
        try:
            zone = await asyncio.to_thread(self.db.get_zone, str(zone_id))
            if zone is None:
                return image
            cropped = crop_to_zone(image, zone.get("polygon") or [])
        except Exception as exc:  # noqa: BLE001 - a bad crop must not cost the check
            self.log(f"vlm: could not crop event {event.get('id')} to its zone ({exc})")
            return image
        if cropped is None:
            return image
        self.log(
            f"vlm: verifying event {event.get('id')} on the '{zone.get('name')}' "
            f"zone crop ({cropped.width}x{cropped.height})"
        )
        return cropped

    async def _verify(self, event: dict[str, Any], image: Any) -> PolicyVerdict | None:
        """Ask the model whether the detector was right. None = could not ask."""
        prompt = verification_prompt(str(event.get("description") or ""))
        # Any grounding box in the answer would be in the crop's coordinates and
        # not the frame's. Nothing downstream reads one — the prompt asks for
        # box_2d null and a verdict is used for its boolean, its confidence and
        # its sentence — but that is the reason it must stay that way.
        image = await self._zone_view(event, image)
        try:
            return await self.client.evaluate(
                prompt, image, timeout=self.settings.verify.timeout_sec, max_tokens=192,
                # A person is waiting on the other end of this one.
                priority=True,
            )
        except VlmUnavailable as exc:
            self.verify_failures += 1
            self._mark_unavailable(str(exc))
            return None
        except Exception as exc:  # noqa: BLE001
            self.verify_failures += 1
            self.log(f"vlm: verification failed ({type(exc).__name__}: {exc})")
            return None

    def _suppress(
        self, event: dict[str, Any], verdict: PolicyVerdict, reason: str, model: str
    ) -> None:
        """Mark the event suppressed and re-banner its snapshot.

        The row, the clip and the annotated frame all survive. A suppressed
        event is a reviewable claim about the *verifier*, and deleting it would
        make the verifier unauditable — which is how a de-noising feature
        quietly becomes a way to miss things.
        """
        event_id = int(event["id"])
        self.db.record_verification(
            event_id, verified_by=model, confidence=verdict.confidence,
            suppressed=True, reason=reason,
        )
        snapshot = event.get("snapshot_path")
        if not snapshot:
            return
        path = self.config.snapshots_dir / str(snapshot)
        if not path.is_file():
            return
        try:
            from PIL import Image as PilImage

            with PilImage.open(path) as original:
                marked = annotate_veto(original.copy(), headline="", reason=reason)
            save_snapshot(marked, path, quality=self.config.evidence.snapshot_quality)
        except Exception as exc:  # noqa: BLE001 - the DB row is what matters
            self.log(f"vlm: could not re-banner snapshot for event {event_id} ({exc})")

    # --- reporting --------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """What ``/api/health`` and the engine report show."""
        degraded = not self.available
        return {
            "enabled": self.settings.enabled,
            "available": self.available,
            "endpoint": self.client.base_url,
            "model": self.model_name or self.settings.model,
            "reason": self.last_unavailable_reason,
            "degraded_for_sec": (
                round(time.time() - self.degraded_since, 1) if self.degraded_since else None
            ),
            "policies": len({s.policy_id for s in self.schedules.values()}),
            "schedules": [s.as_dict(degraded=degraded) for s in self.schedules.values()],
            "events": sum(s.events for s in self.schedules.values()),
            "evaluations": sum(s.evaluations for s in self.schedules.values()),
            # Ongoing conditions (migration 7). `events` counts incidents
            # opened; these count how much watching each of them is standing
            # for. On a plant where a mess persists, `observations` climbs and
            # `events` does not — which is the whole shape of the change.
            "continuity": {
                "enabled": self.config.vlm.continuity.enabled,
                "min_iou": self.config.vlm.continuity.min_iou,
                "stale_after_sec": self.config.vlm.continuity.stale_after_sec,
                "observations": sum(s.observations for s in self.schedules.values()),
                "closed": sum(s.closed for s in self.schedules.values()),
            },
            "verification": {
                "enabled": self.settings.verify.enabled,
                "stages": self.settings.verify.stage_list,
                "checked": self.verifications,
                "suppressed": self.vetoes,
                "failures": self.verify_failures,
            },
            "inference": self.client.stats.as_dict(),
        }


__all__ = ["VLM_DETECTOR_KEY", "Schedule", "VlmRunner"]
