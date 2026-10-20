"""Is this zone over its limit — as a *state*, not as a frame.

``argus/vision/debounce.py`` opens by saying "a violation is a state, not an
event", and then rate-limits events. Measured on this appliance on 2026-09-19,
that is the distance between a sentence and a behaviour: ``manpower_headcount``
had written **10 958 events in 24 hours** — ~1 400 per camera per day, one a
minute around the clock, **112 151** in total, against a review queue holding
114 462 unreviewed rows. Eight cameras x 1 440 minutes a day is 11.5k, which is
exactly the shape of a 60 s cooldown re-firing for as long as a zone stays busy.
Nothing was wrong with the detector. The row was answering *"is it still true?"*
when the only question worth a row is *"did it start?"*.

Two decisions were being made per frame, and both are wrong per frame:

**Whether the zone is over its limit.** Measured against hand labels on this
plant's own footage the same day, ~13% of the detector's boxes are false
positives and ~19% of the people present are missed. A single frame's count
therefore flickers either side of the limit and one spurious box is enough to
open an alert. So this module works from the **median** of a short rolling
window rather than from the newest count: a median survives a bad frame in three
and needs a real, persistent change to move.

**When the state changes.** A count hovering at the limit crosses it repeatedly,
and a symmetric test turns one situation into a column of incidents. So opening
and closing run on different clocks — ``sustain_sec`` of continuous breach to
open, ``clear_sec`` of continuous non-breach to close. That hysteresis is what
makes "12 people in a zone limited to 8, 09:14 to 09:41" one thing.

Everything here is driven by ``(ts, count, limit)`` and holds no reference to a
frame, a camera, a policy or the database, so the whole of the judgement can be
argued with in a test that runs in microseconds — which is the point, because
what is being changed is when this appliance is allowed to wake somebody up.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
from statistics import median_low

#: The two states. A zone is either in a condition or it is not; "about to be"
#: is a timer, not a state, and naming it would invite somebody to report it.
CLEAR = "clear"
OPEN = "open"

#: What changed on this observation, for the caller to act on. ``None`` — the
#: overwhelmingly common answer, five times a second per zone — means the state
#: held and there is nothing to write.
OPENED = "opened"
CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ZoneVerdict:
    """One observation's answer, and everything a caller needs to act on it."""

    #: ``CLEAR`` or ``OPEN`` — the state *after* this observation.
    state: str
    #: ``OPENED``, ``CLOSED``, or None when the state held.
    changed: str | None
    #: The working count: the median of the window. This is the number that was
    #: compared against the limit, so it is the number an alert should quote.
    count: int
    #: This frame's own count, unsmoothed. Published to the live wall beside the
    #: smoothed one, because an operator looking at a tile is looking at *now*.
    raw_count: int
    #: The limit in force at this observation; None for a zone that has none.
    limit: int | None
    #: Highest smoothed count seen since the condition opened (0 while clear).
    peak: int
    #: When the smoothed count first went over the limit in this run — earlier
    #: than the open by ``sustain_sec``, because the condition began before it
    #: was confirmed. None when the zone is not currently in breach.
    over_since: float | None
    #: Frame timestamp of the observation that opened the condition.
    opened_at: float | None
    #: On a ``CLOSED`` verdict: the last moment the condition was known true,
    #: which is when the count came down — not ``clear_sec`` later, when we
    #: accepted that it had stayed down. The same distinction
    #: ``Database.stale_conditions`` draws between the plant's clock and ours.
    closed_at: float | None
    #: How many samples the median was taken over. 1 means the window had just
    #: been emptied (a restart, a seek, a reconnect, a long gap) and there is
    #: nothing to smooth yet.
    samples: int

    @property
    def over_limit(self) -> bool:
        """Whether the *smoothed* zone is over its limit right now.

        Not the same as ``state == OPEN``: this is true from the first smoothed
        breach, while the state waits out ``sustain_sec``.
        """
        return self.limit is not None and self.count > self.limit

    @property
    def excess(self) -> int:
        return 0 if self.limit is None else max(0, self.count - self.limit)


class ZoneState:
    """The smoothed, hysteretic occupancy state of one (camera, zone, policy).

    One instance per triple, held by the caller. Not thread-safe and does not
    need to be: the engine's analytics pool preserves per-camera order and never
    runs two frames of one camera at once.
    """

    __slots__ = (
        "window_sec", "sustain_sec", "clear_sec", "release_margin",
        "state", "peak", "over_since", "under_since", "opened_at",
        "over_for", "under_for",
        "_samples", "_last_ts",
    )

    def __init__(
        self,
        *,
        window_sec: float = 3.0,
        sustain_sec: float = 5.0,
        clear_sec: float = 10.0,
        release_margin: float = 0.0,
    ):
        self.window_sec = max(0.0, float(window_sec))
        self.sustain_sec = max(0.0, float(sustain_sec))
        self.clear_sec = max(0.0, float(clear_sec))
        self.release_margin = max(0.0, float(release_margin))

        self.state: str = CLEAR
        self.peak: int = 0
        #: Start of the current run of smoothed breach / non-breach, for display.
        self.over_since: float | None = None
        self.under_since: float | None = None
        self.opened_at: float | None = None
        #: Seconds of breach / non-breach actually *observed* in the current
        #: run. See ``observe`` for why this is not simply ``ts - over_since``.
        self.over_for: float = 0.0
        self.under_for: float = 0.0

        self._samples: deque[tuple[float, int]] = deque()
        self._last_ts: float | None = None

    # --- observation -------------------------------------------------------

    def observe(self, ts: float, count: int, limit: int | None) -> ZoneVerdict:
        """Fold in one frame's count and say what the zone is now.

        ``limit`` is passed every time rather than held, because an operator can
        edit a zone's ``sanctioned_limit`` while a condition is open and the
        engine re-reads zones every 15 s. A limit raised above the current count
        starts the clear run; a limit removed entirely (``None``) does the same,
        so a zone that stops being judged ends its condition through the same
        door as one that empties — rather than leaving a row open forever with
        nothing left in the world that could close it.

        **Continuity is credited from what was observed, not from the clock.**
        Each observation adds the time since the previous one, capped at the
        window. A steady 5 fps stream therefore credits real time to the tenth
        of a second, while a camera that went away for an hour and came back
        over its limit credits ``window_sec`` for that hour and not an hour —
        silence is evidence that nobody looked, never evidence of a breach.
        Capping rather than resetting is deliberate: a flat "a gap resets the
        run" rule would leave any source slower than one frame per
        ``window_sec`` permanently unable to raise an alert at all, which is a
        silent alerting outage in the band between a sluggish camera and one
        ``detection.stall_timeout_sec`` is willing to call dead.
        """
        count = int(count)
        ts = float(ts)

        if self._last_ts is None:
            step = 0.0
        elif ts < self._last_ts:
            # Time went backwards: looped demo footage, a seek, a camera whose
            # clock restarted on reconnect. The samples on hand describe a
            # different stretch of video from this one and the runs were
            # measured across a discontinuity, so both are discarded. The
            # *state* is kept — see ``_reset_window``.
            self._reset_window()
            step = 0.0
        else:
            step = ts - self._last_ts
            if self.window_sec > 0:
                step = min(step, self.window_sec)
        self._last_ts = ts

        self._samples.append((ts, count))
        cutoff = ts - self.window_sec
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()

        smoothed = median_low([value for _, value in self._samples])
        # Two thresholds, not one. A zone opens above its limit and, once open,
        # is held until the count is *clearly* back under it — a Schmitt trigger.
        # `clear_sec` alone is hysteresis in time, and time does not help a count
        # that sits ON the limit: measured live on 2026-09-20, the morning after
        # this class shipped, `anmol_packing_7_benches` (limit 20, median count
        # 21, over the limit in 53% of frames) dipped to 20 for ten seconds once
        # a minute, closed, and reopened the moment `cooldown_sec` allowed — ten
        # conditions in ten minutes about one crowded bench. Replaying the same
        # detections with a release level one person lower opens one.
        threshold = self.release_level(limit) if self.state == OPEN else limit
        over = threshold is not None and smoothed > threshold

        if over:
            self.over_for = (self.over_for + step) if self.over_since is not None else 0.0
            if self.over_since is None:
                self.over_since = ts
            self.under_since, self.under_for = None, 0.0
        else:
            self.under_for = (self.under_for + step) if self.under_since is not None else 0.0
            if self.under_since is None:
                self.under_since = ts
            self.over_since, self.over_for = None, 0.0

        changed: str | None = None
        closed_at: float | None = None
        if self.state == CLEAR:
            if over and self.over_for >= self.sustain_sec:
                self.state = OPEN
                self.opened_at = ts
                self.peak = smoothed
                changed = OPENED
        else:
            self.peak = max(self.peak, smoothed)
            if not over and self.under_for >= self.clear_sec:
                closed_at = self.under_since
                self.state = CLEAR
                changed = CLOSED

        verdict = ZoneVerdict(
            state=self.state,
            changed=changed,
            count=smoothed,
            raw_count=count,
            limit=limit,
            peak=self.peak,
            over_since=self.over_since,
            opened_at=self.opened_at,
            closed_at=closed_at,
            samples=len(self._samples),
        )
        if changed == CLOSED:
            self.peak = 0
            self.opened_at = None
        return verdict

    # --- housekeeping ------------------------------------------------------

    def _reset_window(self) -> None:
        """Forget the samples and both runs, keeping the state itself.

        The state survives on purpose. A camera that reconnects mid-breach has
        not stopped being over its limit, and dropping the condition here would
        file a second one the moment the stream came back — which is the failure
        this module exists to remove. What is discarded is only the evidence
        that cannot speak about the new footage: the samples, and the runs of
        breach or non-breach measured across the discontinuity. An open
        condition therefore needs a fresh ``clear_sec`` of observed quiet to
        close, and a clear zone a fresh ``sustain_sec`` of observed breach to
        open.
        """
        self._samples.clear()
        self.over_since = self.under_since = None
        self.over_for = self.under_for = 0.0

    def reset(self) -> None:
        """Forget everything, state included — for a deleted zone, or a policy
        that stopped applying here."""
        self._reset_window()
        self._last_ts = None
        self.state = CLEAR
        self.peak = 0
        self.opened_at = None

    def release_level(self, limit: int | None) -> int | None:
        """The count an OPEN condition must fall to before it starts to clear.

        ``release_margin`` is a fraction of the limit and never less than one
        person — a flat "one below" is noise on a fifty-person hall and the
        whole zone on a two-person booth. Clamped at zero so a no-go zone
        (limit 0) still clears when it empties, rather than waiting for a
        negative headcount.
        """
        if limit is None:
            return None
        if self.release_margin <= 0:
            return limit
        return max(0, limit - max(1, ceil(self.release_margin * limit)))

    def tune(self, *, window_sec: float, sustain_sec: float, clear_sec: float,
             release_margin: float | None = None) -> None:
        """Adopt edited settings without losing the state or the window."""
        self.window_sec = max(0.0, float(window_sec))
        self.sustain_sec = max(0.0, float(sustain_sec))
        self.clear_sec = max(0.0, float(clear_sec))
        if release_margin is not None:
            self.release_margin = max(0.0, float(release_margin))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ZoneState({self.state}, samples={len(self._samples)}, peak={self.peak}, "
            f"window={self.window_sec}s, sustain={self.sustain_sec}s, clear={self.clear_sec}s)"
        )


def describe_duration(seconds: float) -> str:
    """"45 s", "6 min", "1 h 12 min" — a span as a card would show it.

    Lives beside the state machine because it is the other half of the headline
    its numbers go into, and because a span that reads "0 min" for the first
    minute of an alert is a bug somebody has to notice twice before finding it.
    """
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{int(round(seconds))} s"
    minutes = int(round(seconds / 60.0))
    if minutes < 90:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h" if minutes == 0 else f"{hours} h {minutes} min"


__all__ = [
    "CLEAR",
    "CLOSED",
    "OPEN",
    "OPENED",
    "ZoneState",
    "ZoneVerdict",
    "describe_duration",
]
