"""Alert debouncing.

A violation is a *state*, not an event: someone standing in an over-crowded zone
is over the limit in every sampled frame. Without a cooldown the appliance would
emit one alert per frame — at 5 fps that is 300 alerts a minute for a single
incident, which is precisely the alert fatigue the product exists to remove.

Ported from v1 (``pipeline/debounce.py``, ``v1-archive``) with the same semantics (first
occurrence emits, repeats inside the window are suppressed) plus two additions
the v2 policy engine needs: per-key cooldowns (each policy carries its own
``cooldown_sec``) and introspection so the API can show "next alert in Ns".

**What it does not do, despite the opening sentence.** Rate-limiting is not
holding a state: this emits again every cooldown for as long as the caller keeps
asking, so "a violation is a state" was a sentence about the problem and never a
description of the fix. Measured on this appliance on 2026-09-19,
``manpower_headcount`` had written 10 958 events in 24 hours and 112 151 in
total, one a minute per camera around the clock, by doing exactly what is
written here. Deciding whether the zone is *in* a violation now belongs to
``argus/vision/zone_state.py``, and this is left with the one job it can do:
saying how soon the next **new** condition may open.
"""

from __future__ import annotations


class Debouncer:
    """Suppress repeat alerts for the same key within a cooldown window.

    ``cooldown_sec <= 0`` disables suppression entirely — every call emits. That
    is a real configuration (a policy that must fire on every sampled frame), so
    it is honoured rather than clamped, and it is why the seeded
    ``manpower_headcount`` policy carries an explicit non-zero cooldown.
    """

    __slots__ = ("cooldown_sec", "_last_emitted")

    def __init__(self, cooldown_sec: float = 60.0):
        self.cooldown_sec = float(cooldown_sec)
        self._last_emitted: dict[str, float] = {}

    def should_emit(self, key: str, ts_seconds: float, cooldown_sec: float | None = None) -> bool:
        """Record and allow an alert for ``key``, or refuse it inside the cooldown.

        ``cooldown_sec`` overrides the instance default for this key — the engine
        passes the policy's own cooldown so one Debouncer can serve every policy
        on a camera.
        """
        window = self.cooldown_sec if cooldown_sec is None else float(cooldown_sec)
        last = self._last_emitted.get(key)
        if last is None or window <= 0 or (ts_seconds - last) >= window:
            self._last_emitted[key] = ts_seconds
            return True
        return False

    def remaining(self, key: str, ts_seconds: float, cooldown_sec: float | None = None) -> float:
        """Seconds until ``key`` may fire again (0.0 when it may fire now)."""
        window = self.cooldown_sec if cooldown_sec is None else float(cooldown_sec)
        last = self._last_emitted.get(key)
        if last is None or window <= 0:
            return 0.0
        return max(0.0, window - (ts_seconds - last))

    def last_emitted(self, key: str) -> float | None:
        return self._last_emitted.get(key)

    def forget(self, key: str) -> None:
        """Drop a key's history — used when a zone or policy is deleted."""
        self._last_emitted.pop(key, None)

    def clear(self) -> None:
        self._last_emitted.clear()

    def __len__(self) -> int:
        return len(self._last_emitted)


__all__ = ["Debouncer"]
