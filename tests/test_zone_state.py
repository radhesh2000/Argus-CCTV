"""When is a zone over its limit — the arithmetic, with no engine anywhere.

The bug this file is about was measured, not imagined. On 2026-09-19 this
appliance's ``manpower_headcount`` policy had written **10 958 events in 24
hours** — ~1 400 per camera per day, one a minute around the clock — and 112 151
in total, against a review queue of 114 462 unreviewed rows. Two per-frame
decisions were behind it: the count was compared with the limit on a single
frame, and the row was re-fired every ``cooldown_sec`` for as long as the answer
stayed yes.

``argus/vision/zone_state.py`` is the replacement, and it is a pure function of
``(ts, count, limit)`` precisely so the part that decides whether somebody gets
woken up can be argued with here rather than in a live run. Every test below is
a sentence about the plant: a spike is not an incident, a wobble at the limit is
one incident, and silence is not evidence of anything.
"""

from __future__ import annotations

import pytest

from argus.vision.zone_state import (
    CLEAR,
    CLOSED,
    OPEN,
    OPENED,
    ZoneState,
    describe_duration,
)

#: 5 fps, the shipped design point, so a "tick" below is one sampled frame.
FPS = 5.0
STEP = 1.0 / FPS


def feed(state: ZoneState, counts, limit=2, *, start=1_000_000.0, step=STEP):
    """Drive the machine with a sequence of per-frame counts, 1/`step` per second.

    Returns every verdict, so a test can assert on the whole run rather than on
    the last answer — which matters, because the failures being prevented are
    about *how many* transitions happened, not about where it ended up.
    """
    return [
        state.observe(start + index * step, count, limit)
        for index, count in enumerate(counts)
    ]


def transitions(verdicts):
    """Just the moments something changed — usually the whole assertion."""
    return [verdict.changed for verdict in verdicts if verdict.changed]


def over(seconds: float, count: int = 5, *, step: float = STEP) -> list[int]:
    return [count] * int(round(seconds / step))


# --------------------------------------------------------------------------
# smoothing: one bad frame is not a fact
# --------------------------------------------------------------------------


def test_a_single_frame_spike_opens_nothing():
    """Measured on this plant's own frames on 2026-09-19, ~13% of the detector's
    boxes are false positives. One of them landing in a zone at its limit used
    to be an alert; the median of a three-second window cannot move for it."""
    state = ZoneState()
    verdicts = feed(state, [2, 2, 2, 9, 2, 2, 2] + [2] * 100, limit=2)
    assert transitions(verdicts) == []
    assert state.state == CLEAR
    # The spike is still *reported*, it is just not believed: the wall shows
    # what the camera saw, the policy acts on the median.
    assert verdicts[3].raw_count == 9
    assert verdicts[3].count == 2


def test_a_single_frame_dropout_does_not_end_a_condition():
    """The other half of the same measurement: ~19% of the people present are
    missed per frame, so a genuine over-crowd reads under-limit now and then. A
    condition that ended on one of those would be a column of incidents."""
    state = ZoneState()
    feed(state, over(20.0, 5), limit=2)
    assert state.state == OPEN
    verdicts = feed(state, [5, 5, 1, 5, 5, 5], limit=2, start=1_000_100.0)
    assert transitions(verdicts) == []
    assert state.state == OPEN


def test_the_working_count_is_the_median_and_it_is_an_observed_count():
    """`median_low`, so the number quoted in an alert is one the detector
    actually produced rather than an average of two frames it did not."""
    state = ZoneState(window_sec=1.0)
    verdicts = feed(state, [4, 8, 4, 9], limit=99, step=0.2)
    assert [verdict.count for verdict in verdicts] == [4, 4, 4, 4]


# --------------------------------------------------------------------------
# opening: sustained, not instantaneous
# --------------------------------------------------------------------------


def test_a_sustained_breach_opens_exactly_once_and_then_says_nothing():
    state = ZoneState(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0)
    verdicts = feed(state, over(600.0, 5), limit=2)          # ten minutes at 5 fps
    assert transitions(verdicts) == [OPENED]
    assert state.state == OPEN
    # 3 s of window before the median moves, then 5 s of sustained breach.
    opened = next(v for v in verdicts if v.changed == OPENED)
    assert opened.opened_at == pytest.approx(1_000_000.0 + 5.0, abs=0.4)


def test_nothing_opens_before_sustain_sec_has_actually_elapsed():
    state = ZoneState(window_sec=1.0, sustain_sec=5.0)
    assert transitions(feed(state, over(4.8, 5), limit=2)) == []
    assert state.state == CLEAR


def test_at_the_limit_is_not_over_it():
    """`Occupancy.over_limit` is strictly greater, and so is this — the
    sanctioned limit is a number of people the zone is allowed to hold."""
    state = ZoneState()
    assert transitions(feed(state, over(60.0, 2), limit=2)) == []


def test_a_zone_with_no_limit_is_counted_and_never_alerts():
    state = ZoneState()
    verdicts = feed(state, over(60.0, 40), limit=None)
    assert transitions(verdicts) == []
    assert verdicts[-1].count == 40 and verdicts[-1].over_limit is False


def test_sustain_zero_opens_on_the_first_smoothed_breach():
    """A site that wants the old promptness can have it, and the knob means what
    it says rather than being clamped to something safer behind its back."""
    state = ZoneState(window_sec=0.0, sustain_sec=0.0)
    assert feed(state, [7], limit=2)[0].changed == OPENED


# --------------------------------------------------------------------------
# hysteresis: a wobble at the limit is one thing
# --------------------------------------------------------------------------


def test_a_count_flickering_around_the_limit_is_one_condition():
    """The case this module exists for. A zone hovering at its limit crosses it
    over and over; with a symmetric test that is an incident every time, which
    is how one busy afternoon became 1 400 rows."""
    state = ZoneState(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0)
    feed(state, over(10.0, 5), limit=2)
    assert state.state == OPEN

    # Twelve minutes of 3-2-3-2..., three seconds either side of the limit.
    flicker: list[int] = []
    for _ in range(120):
        flicker += over(3.0, 3) + over(3.0, 1)
    verdicts = feed(state, flicker, limit=2, start=1_000_100.0)

    assert transitions(verdicts) == [], "a wobble at the limit split the condition"
    assert state.state == OPEN


def test_it_closes_only_after_clear_sec_of_quiet():
    state = ZoneState(window_sec=1.0, sustain_sec=5.0, clear_sec=10.0)
    feed(state, over(20.0, 5), limit=2)
    assert state.state == OPEN

    quiet = feed(state, over(9.0, 0), limit=2, start=1_000_100.0)
    assert transitions(quiet) == [], "closed before clear_sec had elapsed"
    more = feed(state, over(4.0, 0), limit=2, start=1_000_109.2)
    assert transitions(more) == [CLOSED]
    assert state.state == CLEAR


def test_the_close_is_stamped_when_the_count_came_down_not_when_we_accepted_it():
    """"When did the problem end" is a fact about the plant; "when were we
    satisfied" is a fact about the appliance. Only the first belongs on a card —
    the same distinction `Database.stale_conditions` draws."""
    state = ZoneState(window_sec=0.0, sustain_sec=1.0, clear_sec=10.0)
    feed(state, over(5.0, 5), limit=2)
    quiet_from = 1_000_100.0
    verdicts = feed(state, over(12.0, 0), limit=2, start=quiet_from)
    closed = next(v for v in verdicts if v.changed == CLOSED)
    assert closed.closed_at == pytest.approx(quiet_from)


def test_a_new_breach_after_a_close_is_a_second_condition():
    """Hysteresis is not suppression: a zone that clears and fills again has had
    two incidents, and the second one is somebody's afternoon going wrong."""
    state = ZoneState(window_sec=1.0, sustain_sec=5.0, clear_sec=10.0)
    feed(state, over(20.0, 5), limit=2)
    feed(state, over(20.0, 0), limit=2, start=1_000_100.0)
    assert state.state == CLEAR
    assert transitions(feed(state, over(20.0, 5), limit=2, start=1_000_200.0)) == [OPENED]


# --------------------------------------------------------------------------
# the peak
# --------------------------------------------------------------------------


def test_the_peak_is_the_worst_the_condition_reached():
    state = ZoneState(window_sec=1.0, sustain_sec=2.0, clear_sec=5.0)
    feed(state, over(4.0, 5), limit=2)
    feed(state, over(3.0, 11), limit=2, start=1_000_010.0)
    verdicts = feed(state, over(3.0, 6), limit=2, start=1_000_020.0)
    assert verdicts[-1].peak == 11
    assert verdicts[-1].count == 6


def test_the_peak_resets_with_the_condition():
    state = ZoneState(window_sec=1.0, sustain_sec=2.0, clear_sec=5.0)
    feed(state, over(5.0, 12), limit=2)
    feed(state, over(10.0, 0), limit=2, start=1_000_020.0)
    verdicts = feed(state, over(5.0, 4), limit=2, start=1_000_040.0)
    assert verdicts[-1].peak == 4, "the new condition inherited the old one's peak"


# --------------------------------------------------------------------------
# a limit edited under a running condition
# --------------------------------------------------------------------------


def test_raising_the_limit_mid_condition_closes_it_through_the_normal_door():
    """An operator who decides the zone was always allowed 12 people has not
    said the last ten minutes did not happen — the condition ends, it is not
    erased, and it ends `clear_sec` after the edit like any other."""
    state = ZoneState(window_sec=1.0, sustain_sec=2.0, clear_sec=10.0)
    feed(state, over(10.0, 5), limit=2)
    assert state.state == OPEN
    verdicts = feed(state, over(20.0, 5), limit=12, start=1_000_020.0)
    assert transitions(verdicts) == [CLOSED]


def test_lowering_the_limit_opens_on_a_count_that_was_always_there():
    state = ZoneState(window_sec=1.0, sustain_sec=5.0)
    assert transitions(feed(state, over(20.0, 4), limit=9)) == []
    assert transitions(feed(state, over(20.0, 4), limit=2, start=1_000_100.0)) == [OPENED]


def test_removing_the_limit_altogether_ends_the_condition():
    """A zone whose `sanctioned_limit` is cleared stops being judged. The row has
    to close through the same door as one that emptied, or it sits open forever
    with nothing left in the world that could ever end it."""
    state = ZoneState(window_sec=1.0, sustain_sec=2.0, clear_sec=10.0)
    feed(state, over(10.0, 8), limit=2)
    assert transitions(feed(state, over(20.0, 8), limit=None, start=1_000_020.0)) == [CLOSED]


# --------------------------------------------------------------------------
# clocks that misbehave
# --------------------------------------------------------------------------


def test_a_timestamp_that_jumps_backwards_does_not_open_or_close_anything():
    """Looped demo footage and a camera whose clock restarts on reconnect both
    do this. The samples on hand describe different video, so they go; the state
    stays, because a stream coming back has not stopped being over its limit."""
    state = ZoneState(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0)
    feed(state, over(30.0, 6), limit=2)
    assert state.state == OPEN

    rewound = feed(state, over(4.0, 6), limit=2, start=1_000_000.0)   # back to the top
    assert transitions(rewound) == []
    assert state.state == OPEN
    assert rewound[0].samples == 1, "the window survived a discontinuity"


def test_a_rewind_does_not_let_a_spike_open_a_condition():
    state = ZoneState(window_sec=3.0, sustain_sec=5.0)
    feed(state, over(30.0, 1), limit=2)
    # One over-limit frame, timestamped an hour before everything so far.
    verdict = state.observe(1_000_000.0 - 3600.0, 9, 2)
    assert verdict.changed is None and verdict.state == CLEAR


def test_an_hour_of_silence_is_not_an_hour_of_sustained_breach():
    """The gap case, and the reason continuity is credited from observations
    rather than from the clock: `ts - over_since` would have read 3 600 s of
    breach on the second frame the camera ever delivered."""
    state = ZoneState(window_sec=3.0, sustain_sec=5.0)
    assert state.observe(1_000_000.0, 9, 2).changed is None
    assert state.observe(1_000_000.0 + 3600.0, 9, 2).changed is None
    assert state.state == CLEAR
    # And the window itself holds only the new sample — the old count says
    # nothing about footage an hour later.
    assert state.observe(1_000_000.0 + 3600.2, 9, 2).samples == 2


def test_a_gap_does_not_close_an_open_condition_either():
    """Silence is evidence that nobody looked, never that the zone emptied. A
    condition that went unobserved is closed as *stale* by the engine, with a
    reason on the row — not quietly reported as cleared."""
    state = ZoneState(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0)
    feed(state, over(30.0, 6), limit=2)
    assert state.state == OPEN
    assert state.observe(1_000_100.0, 0, 2).changed is None
    assert state.state == OPEN


def test_a_camera_slower_than_the_window_can_still_raise_an_alert():
    """A flat "a gap resets the run" rule would leave any source delivering
    frames further apart than `window_sec` permanently unable to open anything —
    a silent alerting outage in the band between a sluggish camera and one
    `detection.stall_timeout_sec` is willing to call dead. Each observation
    credits at most one window, so a slow camera is slow, not mute."""
    state = ZoneState(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0)
    verdicts = feed(state, over(60.0, 7, step=4.0), limit=2, step=4.0)   # one frame / 4 s
    assert transitions(verdicts) == [OPENED]


def test_two_observations_of_one_frame_each_cannot_sustain_a_breach():
    """The same cap from the other side: `window_sec` of credit per observation
    means a 5 s sustain needs at least two windows' worth of looking."""
    state = ZoneState(window_sec=3.0, sustain_sec=5.0)
    assert state.observe(1_000_000.0, 9, 2).changed is None
    assert state.observe(1_000_030.0, 9, 2).changed is None    # credits 3 s, not 30
    assert state.observe(1_000_060.0, 9, 2).changed == OPENED  # 6 s observed


# --------------------------------------------------------------------------
# housekeeping
# --------------------------------------------------------------------------


def test_reset_forgets_the_state_as_well_as_the_window():
    state = ZoneState(window_sec=1.0, sustain_sec=2.0)
    feed(state, over(10.0, 6), limit=2)
    assert state.state == OPEN
    state.reset()
    assert state.state == CLEAR and state.peak == 0
    assert state.observe(1_000_000.0, 6, 2).changed is None


def test_retuning_keeps_the_state_and_the_window():
    """Config is re-read under a running engine. A retuned threshold must not
    also restart every condition on the box."""
    state = ZoneState(window_sec=1.0, sustain_sec=2.0, clear_sec=10.0)
    feed(state, over(10.0, 6), limit=2)
    state.tune(window_sec=2.0, sustain_sec=9.0, clear_sec=30.0)
    assert state.state == OPEN
    assert state.observe(1_000_010.2, 6, 2).changed is None


@pytest.mark.parametrize(("seconds", "text"), [
    (0.0, "0 s"),
    (44.6, "45 s"),
    (89.0, "89 s"),
    (90.0, "2 min"),
    (360.0, "6 min"),
    (5_400.0, "1 h 30 min"),
    (7_200.0, "2 h"),
])
def test_a_span_reads_the_way_a_card_should_say_it(seconds: float, text: str):
    assert describe_duration(seconds) == text


# --- two thresholds: a count that sits ON the limit is one condition ----------
#
# `clear_sec` is hysteresis in time. Measured live on 2026-09-20, the morning
# after this class shipped: `anmol_packing_7_benches` (limit 20, median count 21)
# dipped to exactly 20 for ten seconds once a minute, closed, and reopened as
# soon as the policy's cooldown allowed — ten conditions in ten minutes about one
# crowded bench. Time cannot fix that; a second, lower threshold does.


def _straddle(limit: int, minutes: int = 10) -> list[int]:
    """A zone living on its limit: 40 s one over, 20 s exactly at it, repeated."""
    return ([limit + 1] * int(40 * FPS) + [limit] * int(20 * FPS)) * minutes


def _opened(verdicts) -> int:
    return sum(v.changed == OPENED for v in verdicts)


def test_a_count_sitting_on_the_limit_flaps_without_a_release_margin():
    """The deployed behaviour, pinned so the reason for the margin stays legible."""
    single = ZoneState(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0)
    assert _opened(feed(single, _straddle(20), limit=20)) == 10


def test_with_a_release_margin_it_is_one_condition():
    held = ZoneState(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0, release_margin=0.10)
    verdicts = feed(held, _straddle(20), limit=20)
    assert _opened(verdicts) == 1
    assert verdicts[-1].state == OPEN


def test_it_still_closes_when_the_zone_is_clearly_under_its_limit():
    held = ZoneState(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0, release_margin=0.10)
    # Limit 20 releases at 18: thirty seconds at 18 is "clearly back under".
    verdicts = feed(held, [22] * int(20 * FPS) + [18] * int(30 * FPS), limit=20)
    assert [v.changed for v in verdicts if v.changed] == [OPENED, CLOSED]
    # ...and thirty seconds at 19 is not.
    held = ZoneState(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0, release_margin=0.10)
    verdicts = feed(held, [22] * int(20 * FPS) + [19] * int(30 * FPS), limit=20)
    assert [v.changed for v in verdicts if v.changed] == [OPENED]


def test_the_margin_never_makes_a_zone_easier_to_open():
    """The lower threshold holds an open condition; it must not open one."""
    held = ZoneState(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0, release_margin=0.10)
    assert _opened(feed(held, [20] * int(60 * FPS), limit=20)) == 0     # at the limit: fine
    assert _opened(feed(held, [19] * int(60 * FPS), limit=20)) == 0


@pytest.mark.parametrize("limit, margin, level", [
    (20, 0.10, 18),     # two people
    (8, 0.10, 7),       # a tenth of 8 is less than a person: one person
    (50, 0.10, 45),
    (11, 0.10, 9),      # ceil(1.1) = 2
    (1, 0.10, 0),
    (0, 0.10, 0),       # a no-go zone clears when it empties, not at -1
    (20, 0.0, 20),      # 0 restores the single threshold
])
def test_release_level(limit, margin, level):
    assert ZoneState(release_margin=margin).release_level(limit) == level
    assert ZoneState(release_margin=margin).release_level(None) is None


def test_a_no_go_zone_still_opens_and_still_clears():
    held = ZoneState(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0, release_margin=0.10)
    verdicts = feed(held, [1] * int(20 * FPS) + [0] * int(30 * FPS), limit=0)
    assert [v.changed for v in verdicts if v.changed] == [OPENED, CLOSED]


def test_tune_adopts_a_new_margin_without_dropping_the_condition():
    state = ZoneState(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0)
    feed(state, [22] * int(20 * FPS), limit=20)
    assert state.state == OPEN
    state.tune(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0, release_margin=0.10)
    assert state.state == OPEN and state.release_level(20) == 18
    # An older caller that does not know about the margin leaves it alone.
    state.tune(window_sec=3.0, sustain_sec=5.0, clear_sec=10.0)
    assert state.release_level(20) == 18
