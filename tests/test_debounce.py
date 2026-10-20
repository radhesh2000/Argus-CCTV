"""Debouncing: one alert per incident, not one per frame."""

from __future__ import annotations

from argus.vision.debounce import Debouncer


def test_first_occurrence_always_emits():
    assert Debouncer(cooldown_sec=60).should_emit("zone", 1000.0) is True


def test_repeat_inside_the_window_is_suppressed():
    debouncer = Debouncer(cooldown_sec=60)
    assert debouncer.should_emit("zone", 1000.0) is True
    assert debouncer.should_emit("zone", 1030.0) is False
    assert debouncer.should_emit("zone", 1059.9) is False


def test_the_window_is_inclusive_at_its_edge():
    debouncer = Debouncer(cooldown_sec=60)
    debouncer.should_emit("zone", 1000.0)
    assert debouncer.should_emit("zone", 1060.0) is True


def test_a_continuous_violation_at_5fps_alerts_once_a_minute():
    """The behaviour the class exists for: 5 fps of over-limit frames for five
    minutes is five alerts, not fifteen hundred."""
    debouncer = Debouncer(cooldown_sec=60)
    emitted = sum(debouncer.should_emit("zone", 1000.0 + tick * 0.2) for tick in range(5 * 60 * 5))
    assert emitted == 5


def test_keys_are_independent():
    debouncer = Debouncer(cooldown_sec=60)
    assert debouncer.should_emit("zone_a", 1000.0) is True
    assert debouncer.should_emit("zone_b", 1000.0) is True
    assert debouncer.should_emit("zone_a", 1001.0) is False


def test_per_call_cooldown_overrides_the_default():
    debouncer = Debouncer(cooldown_sec=60)
    debouncer.should_emit("zone", 1000.0, cooldown_sec=5)
    assert debouncer.should_emit("zone", 1004.0, cooldown_sec=5) is False
    assert debouncer.should_emit("zone", 1005.0, cooldown_sec=5) is True


def test_zero_cooldown_emits_every_time():
    debouncer = Debouncer(cooldown_sec=0)
    assert all(debouncer.should_emit("zone", 1000.0 + tick * 0.2) for tick in range(10))


def test_remaining_counts_down_and_clamps_at_zero():
    debouncer = Debouncer(cooldown_sec=60)
    debouncer.should_emit("zone", 1000.0)
    assert debouncer.remaining("zone", 1010.0) == 50.0
    assert debouncer.remaining("zone", 9999.0) == 0.0
    assert debouncer.remaining("never-seen", 1000.0) == 0.0


def test_forget_and_clear_reset_history():
    debouncer = Debouncer(cooldown_sec=60)
    debouncer.should_emit("a", 1000.0)
    debouncer.should_emit("b", 1000.0)
    debouncer.forget("a")
    assert debouncer.should_emit("a", 1001.0) is True
    assert debouncer.should_emit("b", 1001.0) is False
    debouncer.clear()
    assert len(debouncer) == 0
    assert debouncer.should_emit("b", 1002.0) is True


def test_last_emitted_reports_the_recorded_timestamp():
    debouncer = Debouncer(cooldown_sec=60)
    debouncer.should_emit("zone", 1234.5)
    assert debouncer.last_emitted("zone") == 1234.5
    assert debouncer.last_emitted("other") is None
