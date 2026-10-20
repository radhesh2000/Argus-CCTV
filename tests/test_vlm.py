"""The VLM policy engine, tested without a GPU.

Everything here runs against :class:`FakeVlmServer` — a real HTTP server on
loopback speaking the OpenAI chat-completions shape. That is a deliberate choice
over monkeypatching the client: the thing most likely to break in this
subsystem is the wire contract (does the payload carry the image the way the
server expects, is the schema accepted, is a 503 handled), and a mocked client
cannot fail that way. The server is ~60 lines and needs no model.

The scheduler is driven by calling ``_tick(now)`` with synthetic timestamps
rather than by sleeping, so interval and cooldown behaviour is asserted exactly
and the whole file runs in about a second.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from argus.config import ArgusConfig
from argus.db import Database, Camera, Event, Policy
from argus.db import Zone as DbZone
from argus.vlm.client import VlmClient, VlmUnavailable, encode_frame
from argus.vlm.evidence import MIN_CROP_PX, crop_to_zone
from argus.vlm.schema import (
    PolicyVerdict,
    VlmResponseError,
    normalize_box,
    parse_verdict,
    verification_prompt,
)
from argus.vlm.scheduler import VlmRunner

MODEL = "test/vlm-1b"


# --------------------------------------------------------------------------
# a fake OpenAI-compatible server
# --------------------------------------------------------------------------


class FakeVlmServer:
    """An OpenAI-compatible endpoint that answers from a scripted queue.

    ``replies`` is a list of dicts (the verdict JSON) or ints (an HTTP status to
    fail with). The last entry repeats once exhausted, so a test that does not
    care how many calls happen does not have to count them.

    ``delay_sec`` makes a completion take measurable time. Only the tests about
    *concurrency* set it, and they need it: a server that answers in
    microseconds cannot demonstrate which request waited for which, so an
    assertion about queueing order against an instant server is a coin toss.
    """

    def __init__(self, replies=None, *, model: str = MODEL, delay_sec: float = 0.0):
        self.replies = list(replies or [{"triggered": False, "confidence": 0.9,
                                         "reason": "nothing to report", "box_2d": None}])
        self.model = model
        self.delay_sec = delay_sec
        self.requests: list[dict] = []
        self.status = 200
        self._index = 0
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):  # keep pytest output clean
                pass

            def _json(self, payload, status=200):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if server.status != 200:
                    self._json({"error": "down"}, status=server.status)
                    return
                if self.path.endswith("/models"):
                    self._json({"object": "list", "data": [{"id": server.model}]})
                else:
                    self._json({"error": "not found"}, status=404)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                with server._lock:
                    server.requests.append(payload)
                    reply = server._next()
                if isinstance(reply, int):
                    self._json({"error": {"message": "scripted failure"}}, status=reply)
                    return
                if server.delay_sec:
                    time.sleep(server.delay_sec)
                content = reply if isinstance(reply, str) else json.dumps(reply)
                self._json({
                    "id": "chatcmpl-test",
                    "model": server.model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1234, "completion_tokens": 42},
                })

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def _next(self):
        reply = self.replies[min(self._index, len(self.replies) - 1)]
        self._index += 1
        return reply

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def calls(self) -> int:
        return len(self.requests)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


class FakeFrame:
    """Duck-types argus.ingest.decoder.Frame for the scheduler's purposes."""

    def __init__(self, camera_id: str, ts: float, image: np.ndarray | None = None):
        self.camera_id = camera_id
        self.ts = ts
        self.index = 0
        self.image = np.zeros((544, 960, 3), dtype=np.uint8) if image is None else image


class RecordingNotifier:
    def __init__(self):
        self.dispatched: list[int] = []

    def dispatch(self, event_id: int):
        self.dispatched.append(event_id)
        return None


@pytest.fixture
def server():
    with FakeVlmServer() as running:
        yield running


@pytest.fixture
def vlm_config(config: ArgusConfig, server: FakeVlmServer) -> ArgusConfig:
    config.vlm.base_url = server.base_url
    config.vlm.model = MODEL
    config.vlm.enabled = True
    config.vlm.retry_backoff_sec = 0.01
    config.vlm.timeout_sec = 10.0
    return config


def seed_policy(database: Database, **overrides) -> str:
    fields = {
        "id": "phone_usage_on_line", "kind": "vlm", "detector_key": "vlm_prompt",
        "label": "Phone usage on the line", "enabled": True, "stage": "reviewed",
        "lifecycle": "production", "severity": "medium", "routed_to": "line_supervisor",
        "prompt": "Is any worker visibly using a mobile phone?",
        "interval_sec": 6.0, "cooldown_sec": 45.0,
    }
    fields.update(overrides)
    return database.upsert_policy(Policy(**fields))


def seed_camera(database: Database, camera_id: str = "packing_line_demo") -> str:
    return database.upsert_camera(
        Camera(id=camera_id, name=camera_id.replace("_", " ").title(), source="/tmp/x.mp4")
    )


def make_runner(vlm_config, database, server, **kwargs) -> VlmRunner:
    runner = VlmRunner(vlm_config, database, log=lambda *_a, **_k: None, **kwargs)
    runner.available = True
    runner.model_name = server.model
    return runner


def tick(runner: VlmRunner, now: float) -> None:
    """Run one scheduler tick and let its evaluation tasks finish."""

    async def once():
        await runner._tick(now)
        if runner._tasks:
            await asyncio.wait(set(runner._tasks), timeout=10)

    asyncio.run(once())


# --------------------------------------------------------------------------
# schema parsing
# --------------------------------------------------------------------------


def test_parse_verdict_reads_the_documented_shape():
    verdict = parse_verdict(
        '{"triggered": true, "confidence": 0.82, "reason": "worker on phone", '
        '"box_2d": [100, 200, 300, 400]}',
        width=1000, height=1000,
    )
    assert verdict.triggered is True
    assert verdict.confidence == pytest.approx(0.82)
    assert verdict.reason == "worker on phone"
    assert verdict.box_2d == (100.0, 200.0, 300.0, 400.0)


def test_parse_verdict_tolerates_a_json_fence():
    verdict = parse_verdict(
        '```json\n{"triggered": false, "confidence": 0.4, "reason": "clear", "box_2d": null}\n```',
        width=960, height=544,
    )
    assert verdict.triggered is False
    assert verdict.reason == "clear"


def test_parse_verdict_rejects_non_json():
    with pytest.raises(VlmResponseError):
        parse_verdict("I think the worker might be on a phone.", width=960, height=544)


def test_parse_verdict_clamps_a_wild_confidence():
    verdict = parse_verdict(
        '{"triggered": true, "confidence": 7.5, "reason": "x", "box_2d": null}',
        width=960, height=544,
    )
    assert verdict.confidence == 1.0


def test_parse_verdict_survives_a_non_numeric_confidence():
    """Schema-constrained decoding should prevent this; the parser must not
    depend on that, because a fallback server may have no constraint at all."""
    verdict = parse_verdict(
        '{"triggered": true, "confidence": "high", "reason": "x", "box_2d": null}',
        width=960, height=544,
    )
    assert verdict.confidence == 0.0
    assert verdict.fires(min_confidence=0.5) is False


def test_box_is_dropped_when_not_triggered():
    verdict = parse_verdict(
        '{"triggered": false, "confidence": 0.9, "reason": "no", "box_2d": [1, 2, 900, 900]}',
        width=960, height=544,
    )
    assert verdict.box_2d is None


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ([0.1, 0.2, 0.5, 0.6], (96.0, 108.8, 480.0, 326.4)),      # 0-1 fractions
        ([100, 200, 500, 600], (96.0, 108.8, 480.0, 326.4)),      # 0-1000 grid
        ([500, 600, 100, 200], (96.0, 108.8, 480.0, 326.4)),      # corners swapped
    ],
)
def test_normalize_box_accepts_each_coordinate_convention(box, expected):
    assert normalize_box(box, width=960, height=544) == pytest.approx(expected)


def test_normalize_box_rejects_degenerate_and_malformed():
    assert normalize_box([10, 10, 11, 11], width=960, height=544) is None  # sub-4px
    assert normalize_box([1, 2, 3], width=960, height=544) is None
    assert normalize_box("somewhere", width=960, height=544) is None
    assert normalize_box(["a", "b", "c", "d"], width=960, height=544) is None


def test_normalize_box_clamps_to_the_frame():
    box = normalize_box([-50, -50, 2000, 2000], width=960, height=544)
    assert box == (0.0, 0.0, 960.0, 544.0)


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------


def test_encode_frame_downscales_and_produces_a_data_uri():
    encoded = encode_frame(np.zeros((1080, 1920, 3), dtype=np.uint8), max_width=1024)
    assert encoded.data_uri.startswith("data:image/jpeg;base64,")
    assert encoded.width == 1024
    assert encoded.height == 576


def test_encode_frame_leaves_a_smaller_frame_alone():
    encoded = encode_frame(np.zeros((544, 960, 3), dtype=np.uint8), max_width=1024)
    assert (encoded.width, encoded.height) == (960, 544)


def test_client_sends_the_image_and_the_schema(vlm_config, server):
    server.replies = [{"triggered": True, "confidence": 0.77, "reason": "phone", "box_2d": None}]
    client = VlmClient(vlm_config.vlm)
    verdict = asyncio.run(client.evaluate("prompt?", np.zeros((544, 960, 3), dtype=np.uint8)))

    assert verdict.triggered is True
    assert verdict.confidence == pytest.approx(0.77)
    assert verdict.prompt_tokens == 1234

    sent = server.requests[0]
    assert sent["model"] == MODEL
    assert sent["temperature"] == 0.0
    kinds = [part["type"] for part in sent["messages"][0]["content"]]
    assert kinds == ["text", "image_url"]
    assert sent["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    schema = sent["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert set(schema["schema"]["required"]) == {"triggered", "confidence", "reason", "box_2d"}


def test_client_returns_the_box_as_a_fraction_of_the_frame(vlm_config, server):
    """The model sees a downscaled copy; nothing downstream should have to know
    which resolution that was."""
    server.replies = [{"triggered": True, "confidence": 0.9, "reason": "there",
                       "box_2d": [250, 250, 750, 750]}]
    client = VlmClient(vlm_config.vlm)
    verdict = asyncio.run(client.evaluate("p", np.zeros((1080, 1920, 3), dtype=np.uint8)))
    assert verdict.box_2d == pytest.approx((0.25, 0.25, 0.75, 0.75), abs=0.01)


def test_client_raises_unavailable_on_server_error(vlm_config, server):
    server.replies = [503]
    client = VlmClient(vlm_config.vlm)
    with pytest.raises(VlmUnavailable):
        asyncio.run(client.evaluate("p", np.zeros((544, 960, 3), dtype=np.uint8)))
    assert client.stats.failures == 1


def test_client_raises_unavailable_when_nothing_is_listening(config):
    config.vlm.base_url = "http://127.0.0.1:9/v1"   # discard port
    config.vlm.timeout_sec = 1.0
    client = VlmClient(config.vlm)
    with pytest.raises(VlmUnavailable):
        asyncio.run(client.evaluate("p", np.zeros((544, 960, 3), dtype=np.uint8)))


def test_client_health_reports_the_served_model(vlm_config, server):
    client = VlmClient(vlm_config.vlm)
    report = asyncio.run(client.health())
    assert report["available"] is True
    assert report["model"] == MODEL


def test_client_health_reports_a_dead_server_without_raising(config):
    config.vlm.base_url = "http://127.0.0.1:9/v1"
    client = VlmClient(config.vlm)
    report = asyncio.run(client.health(timeout=1.0))
    assert report["available"] is False
    assert report["reason"]


def test_client_counters_do_not_leak_under_concurrency(vlm_config, server):
    client = VlmClient(vlm_config.vlm)

    async def many():
        frame = np.zeros((544, 960, 3), dtype=np.uint8)
        await asyncio.gather(*(client.evaluate("p", frame) for _ in range(6)))

    asyncio.run(many())
    assert client.stats.requests == 6
    assert client.stats.in_flight == 0
    assert client.stats.queued == 0


# --------------------------------------------------------------------------
# scheduling: intervals and cooldowns
# --------------------------------------------------------------------------


def test_refresh_builds_one_schedule_per_policy_and_camera(vlm_config, database, server):
    seed_policy(database)
    seed_policy(database, id="emergency_exit_blocked", label="Exit blocked", interval_sec=8.0)
    seed_camera(database, "cam_a")
    seed_camera(database, "cam_b")

    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    assert len(runner.schedules) == 4
    assert {s.policy_id for s in runner.schedules.values()} == {
        "phone_usage_on_line", "emergency_exit_blocked"
    }
    assert runner.schedules["phone_usage_on_line:cam_a"].interval_sec == 6.0
    assert runner.schedules["emergency_exit_blocked:cam_a"].interval_sec == 8.0


def test_draft_and_disabled_policies_are_never_scheduled(vlm_config, database, server):
    seed_policy(database, id="draft_policy", lifecycle="draft")
    seed_policy(database, id="off_policy", enabled=False)
    seed_policy(database, id="promptless", prompt=None)
    seed_camera(database)

    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    assert runner.schedules == {}


def test_refresh_preserves_counters_and_phase_across_an_edit(vlm_config, database, server):
    seed_policy(database)
    seed_camera(database, "cam_a")
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    schedule = runner.schedules["phone_usage_on_line:cam_a"]
    schedule.evaluations = 7
    schedule.next_due = 1234.0

    database.update_policy("phone_usage_on_line", interval_sec=20.0, severity="high")
    runner.refresh(now=100.0)

    same = runner.schedules["phone_usage_on_line:cam_a"]
    assert same.evaluations == 7          # counters survive
    assert same.next_due == 1234.0        # sampling phase survives
    assert same.interval_sec == 20.0      # the edit lands
    assert same.severity == "high"


def test_policy_is_evaluated_on_its_own_interval(vlm_config, database, server):
    seed_policy(database, interval_sec=6.0)
    seed_camera(database, "cam_a")
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.schedules["phone_usage_on_line:cam_a"].next_due = 0.0
    runner.offer(FakeFrame("cam_a", ts=0.0))

    tick(runner, now=0.0)
    assert server.calls == 1
    tick(runner, now=3.0)      # inside the interval
    assert server.calls == 1
    tick(runner, now=6.0)      # due again
    assert server.calls == 2


def test_no_frame_yet_is_skipped_not_failed(vlm_config, database, server):
    seed_policy(database)
    seed_camera(database, "cam_a")
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.schedules["phone_usage_on_line:cam_a"].next_due = 0.0

    tick(runner, now=0.0)
    schedule = runner.schedules["phone_usage_on_line:cam_a"]
    assert server.calls == 0
    assert schedule.skipped_no_frame == 1
    assert schedule.failures == 0


def test_a_persistent_condition_stays_one_row_past_its_cooldown(vlm_config, database, server):
    """The violation is a state, not an event, and the row is the state.

    This test used to assert the opposite of its second half — that a fifth
    sample *past* the cooldown wrote a second row. That was the shipped
    behaviour and it was wrong in service: with a ~15-minute cooldown,
    ``anmol_packing_floor_waste`` filed **140** near-identical events about one
    laminate offcut on one apron, 40 of them in a single night, and the feed
    became unreadable while the analytics counted one mess as forty incidents.

    So the cooldown now governs only whether a *new* condition may be opened.
    A triggered verdict that continues the one already on file extends it.
    """
    server.replies = [{"triggered": True, "confidence": 0.9, "reason": "phone", "box_2d": None}]
    seed_policy(database, interval_sec=6.0, cooldown_sec=45.0)
    seed_camera(database, "cam_a")
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.schedules["phone_usage_on_line:cam_a"].next_due = 0.0

    for now in (0.0, 6.0, 12.0, 18.0):
        runner.offer(FakeFrame("cam_a", ts=now))
        tick(runner, now=now)

    assert server.calls == 4                      # asked every interval
    assert database.count_events() == 1           # and told the feed once

    runner.offer(FakeFrame("cam_a", ts=50.0))     # past the 45 s cooldown
    tick(runner, now=50.0)

    assert database.count_events() == 1           # still one incident, not two
    event = database.recent_events()[0]
    assert event["condition_state"] == "open"
    assert event["observations"] == 5             # every sighting, counted
    assert event["ts"] == 0.0                     # ongoing since the first
    assert event["last_seen_ts"] == 50.0          # and last seen at the newest
    assert runner.schedules["phone_usage_on_line:cam_a"].observations == 4


def test_a_slow_model_degrades_sampling_rate_instead_of_queueing(vlm_config, database, server):
    """The bug this pins was found on real hardware: with a 6 s interval and a
    ~5.7 s model, the scheduler enqueued faster than the server drained and the
    queue grew without bound (8 -> 16 -> 25 -> 33 over one run), so every verdict
    described a steadily older frame. Falling behind must cost sampling rate,
    never build a backlog — the same rule the decoder applies to frames."""
    seed_policy(database, interval_sec=6.0)
    seed_camera(database, "cam_a")
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    schedule = runner.schedules["phone_usage_on_line:cam_a"]
    schedule.next_due = 0.0
    schedule.in_flight = True                      # a previous sample is outstanding
    runner.offer(FakeFrame("cam_a", ts=0.0))

    for now in (0.0, 6.0, 12.0, 18.0, 24.0):
        tick(runner, now=now)

    assert server.calls == 0                       # nothing was queued behind it
    assert schedule.skipped_in_flight == 5
    assert runner.client.stats.queued == 0

    schedule.in_flight = False                     # the answer lands
    tick(runner, now=30.0)
    assert server.calls == 1


def test_in_flight_is_cleared_even_when_the_evaluation_fails(vlm_config, database, server):
    """Otherwise one failure would silence that policy permanently."""
    server.replies = [503]
    seed_policy(database)
    seed_camera(database, "cam_a")
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    schedule = runner.schedules["phone_usage_on_line:cam_a"]
    schedule.next_due = 0.0
    runner.offer(FakeFrame("cam_a", ts=0.0))

    tick(runner, now=0.0)
    assert schedule.failures == 1
    assert schedule.in_flight is False


def test_verification_does_not_queue_behind_routine_sampling(vlm_config, database, server):
    """A verification sits in the path of an alert; a scheduled sample does not.
    The reserved slot is what keeps a person's alert off the sampling queue."""
    vlm_config.vlm.max_concurrency = 1
    # An inference has to take long enough for "waited behind" to mean
    # something: against an instant server all four finish in the order the
    # kernel happens to wake them, and this asserts nothing.
    server.delay_sec = 0.1
    client = VlmClient(vlm_config.vlm)
    order: list[str] = []

    async def race():
        async def sample(name):
            await client.evaluate("routine", np.zeros((544, 960, 3), dtype=np.uint8))
            order.append(name)

        async def verify():
            await client.evaluate("urgent", np.zeros((544, 960, 3), dtype=np.uint8),
                                  priority=True)
            order.append("verify")

        # Three routine samples saturate the single shared slot; the priority
        # request must not wait behind them.
        await asyncio.gather(sample("a"), sample("b"), sample("c"), verify())

    asyncio.run(race())
    assert order.index("verify") < 3
    assert client.stats.queued == 0


def test_low_confidence_trigger_raises_nothing(vlm_config, database, server):
    server.replies = [{"triggered": True, "confidence": 0.2, "reason": "maybe", "box_2d": None}]
    vlm_config.vlm.min_confidence = 0.5
    seed_policy(database)
    seed_camera(database, "cam_a")
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.schedules["phone_usage_on_line:cam_a"].next_due = 0.0
    runner.offer(FakeFrame("cam_a", ts=0.0))

    tick(runner, now=0.0)
    assert server.calls == 1
    assert database.count_events() == 0


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


def test_a_fired_policy_writes_a_full_event_with_evidence(vlm_config, database, server):
    server.replies = [{"triggered": True, "confidence": 0.88,
                       "reason": "worker holding a phone at the belt",
                       "box_2d": [250, 250, 750, 750]}]
    seed_policy(database, stage="reviewed", severity="medium", routed_to="line_supervisor")
    seed_camera(database, "cam_a")
    clips: list[tuple] = []
    notifier = RecordingNotifier()
    runner = make_runner(vlm_config, database, server,
                         notifier=notifier, on_event=lambda *a: clips.append(a))
    runner.refresh(now=0.0)
    runner.schedules["phone_usage_on_line:cam_a"].next_due = 0.0
    runner.offer(FakeFrame("cam_a", ts=1000.0))

    tick(runner, now=0.0)

    events = database.recent_events()
    assert len(events) == 1
    event = events[0]
    assert event["detector_type"] == "vlm"
    assert event["detector_name"] == "phone_usage_on_line"
    assert event["policy_id"] == "phone_usage_on_line"
    assert event["stage"] == "reviewed"
    assert event["routed_to"] == "line_supervisor"
    assert event["severity"] == "medium"
    assert event["confidence"] == pytest.approx(0.88)
    assert event["description"] == "worker holding a phone at the belt"
    assert event["ts"] == 1000.0

    # Evidence exists on disk, with the grounding box recorded normalized.
    snapshot = Path(vlm_config.snapshots_dir) / event["snapshot_path"]
    assert snapshot.is_file() and snapshot.stat().st_size > 0
    meta = json.loads(event["meta"])
    assert meta["box_2d"] == pytest.approx([0.25, 0.25, 0.75, 0.75], abs=0.01)
    assert meta["coords"] == "normalized"
    assert meta["model"] == MODEL

    # ...and it went through the same downstream pipeline as a detector event.
    # The callback carries the *raw* frame and the snapshot path as well as the
    # id, because the engine behind it owes this event two things: a ±8 s clip
    # cut from the ring, and a row in the semantic index. The index embeds the
    # frame and displays the snapshot, so both have to travel (Phase 3).
    assert len(clips) == 1
    camera_id, clip_event_id, clip_ts, frame_image, snapshot_rel = clips[0]
    assert (camera_id, clip_event_id, clip_ts) == ("cam_a", event["id"], 1000.0)
    assert snapshot_rel == event["snapshot_path"]
    # Unannotated: the banner and the grounding box belong on the evidence
    # file, never in the vector.
    assert frame_image.shape == (544, 960, 3)
    assert notifier.dispatched == [event["id"]]


def test_an_unreadable_answer_is_counted_and_dropped(vlm_config, database, server):
    server.replies = ["the worker seems fine to me"]
    seed_policy(database)
    seed_camera(database, "cam_a")
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.schedules["phone_usage_on_line:cam_a"].next_due = 0.0
    runner.offer(FakeFrame("cam_a", ts=0.0))

    tick(runner, now=0.0)
    schedule = runner.schedules["phone_usage_on_line:cam_a"]
    assert schedule.failures == 1
    assert "unreadable" in schedule.last_error
    assert database.count_events() == 0
    assert runner.available is True   # the server is fine; the answer was not


# --------------------------------------------------------------------------
# degraded mode
# --------------------------------------------------------------------------


def test_scheduler_degrades_when_the_server_dies(vlm_config, database, server):
    server.replies = [503]
    seed_policy(database)
    seed_camera(database, "cam_a")
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.schedules["phone_usage_on_line:cam_a"].next_due = 0.0
    runner.offer(FakeFrame("cam_a", ts=0.0))

    tick(runner, now=0.0)

    assert runner.available is False
    assert runner.degraded_since is not None
    assert "503" in (runner.last_unavailable_reason or "")
    stats = runner.stats()
    assert stats["available"] is False
    assert all(s["degraded"] for s in stats["schedules"])
    assert database.count_events() == 0   # degraded, not inventing alerts


def test_a_degraded_scheduler_backs_off_then_recovers(vlm_config, database, server):
    seed_policy(database)
    seed_camera(database, "cam_a")
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.offer(FakeFrame("cam_a", ts=0.0))
    runner._mark_unavailable("connection refused")
    runner._retry_at = 100.0

    tick(runner, now=50.0)          # inside the backoff window: no traffic at all
    assert server.calls == 0

    tick(runner, now=150.0)         # past it: probes, recovers, evaluates
    assert runner.available is True
    assert runner.last_unavailable_reason is None
    assert runner.degraded_since is None


def test_engine_start_does_not_fail_when_the_vlm_is_down(config, database):
    """The whole point of degradation: a missing VLM must cost prompt policies
    and nothing else."""
    config.vlm.base_url = "http://127.0.0.1:9/v1"
    config.vlm.retry_backoff_sec = 0.01
    seed_policy(database)
    seed_camera(database, "cam_a")

    runner = VlmRunner(config, database, log=lambda *_a, **_k: None)
    runner.refresh()
    runner.start()
    try:
        assert len(runner.schedules) == 1
        stats = runner.stats()
        assert stats["enabled"] is True
        assert stats["available"] is False
    finally:
        runner.stop(timeout=5)


# --------------------------------------------------------------------------
# the verification pass
# --------------------------------------------------------------------------


def detector_event(database: Database, *, stage: str = "auto_routed") -> int:
    seed_camera(database, "cam_a")
    return database.insert_event(Event(
        camera_id="cam_a", detector_name="manpower_headcount", detector_type="detector",
        severity="medium", description="Zone 'Packing floor' over sanctioned limit: 5 people",
        ts=1000.0, stage=stage, routed_to="safety_manager", confidence=0.9,
    ))


def test_the_verification_prompt_makes_the_count_part_of_the_claim():
    """Argus's one shipped detector claims a number, and until 2026-08-15 the
    verifier was never asked to check it — it upheld 16, 20, 25, 30 and 40 people
    on a frame holding sixteen, with the same sentence as its reason each time.
    This clause took over-claims from 15/24 to 24/24 vetoed (docs/VLM_EVAL.md)."""
    prompt = verification_prompt("Zone 'Packing floor' over sanctioned limit: 30 people (limit 3)")

    assert "First count the people you can actually see in the image." in prompt
    assert "The number in the claim is part of the claim" in prompt
    assert "triggered=false if the count is wrong" in prompt
    assert "30 people (limit 3)" in prompt        # the claim, verbatim


def test_verification_is_off_by_default(vlm_config, database, server):
    event_id = detector_event(database)
    notifier = RecordingNotifier()
    runner = make_runner(vlm_config, database, server)

    runner.dispatch(event_id, np.zeros((544, 960, 3), dtype=np.uint8), notifier)

    assert notifier.dispatched == [event_id]
    assert server.calls == 0
    assert database.get_event(event_id)["suppressed"] == 0


def test_a_veto_suppresses_the_event_and_stops_the_alert(vlm_config, database, server):
    server.replies = [{"triggered": False, "confidence": 0.93,
                       "reason": "only two people are inside the marked zone", "box_2d": None}]
    vlm_config.vlm.verify.enabled = True
    vlm_config.vlm.verify.min_confidence = 0.6
    event_id = detector_event(database)
    notifier = RecordingNotifier()
    runner = make_runner(vlm_config, database, server, notifier=notifier)

    asyncio.run(runner._verify_then_notify(
        database.get_event(event_id), np.zeros((544, 960, 3), dtype=np.uint8), notifier
    ))

    row = database.get_event(event_id)
    assert row["suppressed"] == 1
    assert row["suppressed_reason"] == "only two people are inside the marked zone"
    assert row["verified_by"] == f"vlm:{MODEL}"
    assert row["verify_confidence"] == pytest.approx(0.93)
    assert row["verified_at"]
    assert notifier.dispatched == []          # the mail did not go out
    assert runner.vetoes == 1


def test_a_vetoed_event_stays_visible_and_is_counted(vlm_config, database, server):
    """Suppressed is not deleted and not downgraded — the whole feature depends
    on a reviewer being able to audit what was held back."""
    server.replies = [{"triggered": False, "confidence": 0.9, "reason": "false alarm",
                       "box_2d": None}]
    vlm_config.vlm.verify.enabled = True
    event_id = detector_event(database)
    notifier = RecordingNotifier()
    runner = make_runner(vlm_config, database, server, notifier=notifier)

    asyncio.run(runner._verify_then_notify(
        database.get_event(event_id), np.zeros((544, 960, 3), dtype=np.uint8), notifier
    ))

    assert database.count_events() == 1
    assert [e["id"] for e in database.recent_events(suppressed=True)] == [event_id]
    assert database.recent_events(suppressed=False) == []
    assert database.get_event(event_id)["stage"] == "auto_routed"   # stage untouched

    accuracy = database.detector_accuracy("manpower_headcount")[0]
    assert accuracy["suppressed"] == 1
    assert accuracy["verified"] == 1
    assert accuracy["suppression_rate"] == pytest.approx(1.0)


def test_an_upheld_event_is_recorded_and_still_alerts(vlm_config, database, server):
    server.replies = [{"triggered": True, "confidence": 0.95, "reason": "five people visible",
                       "box_2d": None}]
    vlm_config.vlm.verify.enabled = True
    event_id = detector_event(database)
    notifier = RecordingNotifier()
    runner = make_runner(vlm_config, database, server, notifier=notifier)

    asyncio.run(runner._verify_then_notify(
        database.get_event(event_id), np.zeros((544, 960, 3), dtype=np.uint8), notifier
    ))

    row = database.get_event(event_id)
    assert row["suppressed"] == 0
    assert row["verified_by"] == f"vlm:{MODEL}"   # checked-and-upheld is recorded
    assert notifier.dispatched == [event_id]
    assert runner.vetoes == 0
    assert runner.verifications == 1


def test_an_unconfident_veto_is_not_a_veto(vlm_config, database, server):
    """The failure we accept is a false alarm, never a missed one."""
    server.replies = [{"triggered": False, "confidence": 0.3, "reason": "not sure", "box_2d": None}]
    vlm_config.vlm.verify.enabled = True
    vlm_config.vlm.verify.min_confidence = 0.6
    event_id = detector_event(database)
    notifier = RecordingNotifier()
    runner = make_runner(vlm_config, database, server, notifier=notifier)

    asyncio.run(runner._verify_then_notify(
        database.get_event(event_id), np.zeros((544, 960, 3), dtype=np.uint8), notifier
    ))

    assert database.get_event(event_id)["suppressed"] == 0
    assert notifier.dispatched == [event_id]


def test_an_unreachable_verifier_lets_the_alert_through(config, database):
    config.vlm.verify.enabled = True
    config.vlm.base_url = "http://127.0.0.1:9/v1"
    config.vlm.timeout_sec = 1.0
    config.vlm.verify.timeout_sec = 1.0
    event_id = detector_event(database)
    notifier = RecordingNotifier()
    runner = VlmRunner(config, database, log=lambda *_a, **_k: None, notifier=notifier)

    asyncio.run(runner._verify_then_notify(
        database.get_event(event_id), np.zeros((544, 960, 3), dtype=np.uint8), notifier
    ))

    assert notifier.dispatched == [event_id]
    assert database.get_event(event_id)["suppressed"] == 0
    assert runner.verify_failures == 1


def test_only_configured_stages_are_verified(vlm_config, database, server):
    vlm_config.vlm.verify.enabled = True
    vlm_config.vlm.verify.stages = "auto_routed"
    runner = make_runner(vlm_config, database, server)

    shadow = database.get_event(detector_event(database, stage="shadow"))
    routed = database.get_event(detector_event(database, stage="auto_routed"))
    assert runner.will_verify(shadow) is False
    assert runner.will_verify(routed) is True


def test_vlm_events_are_never_verified_by_the_vlm(vlm_config, database, server):
    """Circular, and it would double the inference cost of every prompt policy."""
    vlm_config.vlm.verify.enabled = True
    seed_camera(database, "cam_a")
    event_id = database.insert_event(Event(
        camera_id="cam_a", detector_name="phone_usage_on_line", detector_type="vlm",
        severity="medium", description="phone", ts=1.0, stage="auto_routed",
    ))
    runner = make_runner(vlm_config, database, server)
    assert runner.will_verify(database.get_event(event_id)) is False


# --------------------------------------------------------------------------
# what the verifier is shown: the zone, not the camera view
# --------------------------------------------------------------------------

# The demo box's own two zones, verbatim from config/seed.yaml. One of them is
# most of the picture and one is a corner of it, which is the whole reason
# cropping has to be a decision rather than an unconditional step.
PACKING_FLOOR = [[0.05, 0.2], [0.95, 0.2], [0.95, 0.95], [0.05, 0.95]]
CONVEYOR_SIDE = [[0.5, 0.5], [0.9, 0.5], [0.9, 0.9], [0.5, 0.9]]


def sent_image(server: FakeVlmServer) -> Image.Image:
    """The picture the last request actually carried, decoded off the wire.

    Asserted on the encoded bytes rather than on a mock's arguments because the
    thing worth checking is that a smaller picture reached the model, and only
    the wire knows that.
    """
    url = server.requests[-1]["messages"][0]["content"][1]["image_url"]["url"]
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def zoned_event(database: Database, polygon, *, name: str = "Conveyor side") -> int:
    seed_camera(database, "cam_a")
    zone_id = database.upsert_zone(DbZone(
        id="zone_a", camera_id="cam_a", name=name, polygon=polygon, sanctioned_limit=3,
    ))
    return database.insert_event(Event(
        camera_id="cam_a", zone_id=zone_id, detector_name="manpower_headcount",
        detector_type="detector", severity="medium",
        description="Zone 'Conveyor side' over sanctioned limit: 5 people (limit 3)",
        ts=1000.0, stage="auto_routed", routed_to="safety_manager", confidence=0.9,
    ))


def test_the_verifier_is_shown_the_zone_and_not_the_camera_view(vlm_config, database, server):
    """The claim is about a zone. Asked over the wide shot, the same model
    vetoed 3/12 of the over-claims it vetoed 12/12 of on the crop."""
    vlm_config.vlm.verify.enabled = True
    event_id = zoned_event(database, CONVEYOR_SIDE)
    runner = make_runner(vlm_config, database, server, notifier=RecordingNotifier())

    asyncio.run(runner._verify_then_notify(
        database.get_event(event_id), np.zeros((544, 960, 3), dtype=np.uint8),
        RecordingNotifier(),
    ))

    picture = sent_image(server)
    assert picture.size == (460, 320)                    # zone + pad, not 960x544
    assert picture.width * picture.height < 0.3 * 960 * 544


def test_a_zone_that_is_most_of_the_view_is_left_alone(vlm_config, database, server):
    """`Packing floor` spans 90% of the frame by 75%. Cropping it would be a
    gesture, and a 3% crop that claims to be a zone view is worse than none."""
    vlm_config.vlm.verify.enabled = True
    event_id = zoned_event(database, PACKING_FLOOR, name="Packing floor")
    runner = make_runner(vlm_config, database, server, notifier=RecordingNotifier())

    asyncio.run(runner._verify_then_notify(
        database.get_event(event_id), np.zeros((544, 960, 3), dtype=np.uint8),
        RecordingNotifier(),
    ))

    assert sent_image(server).size == (960, 544)


def test_an_event_without_a_zone_is_verified_on_the_whole_frame(vlm_config, database, server):
    vlm_config.vlm.verify.enabled = True
    event_id = detector_event(database)                  # no zone_id
    runner = make_runner(vlm_config, database, server, notifier=RecordingNotifier())

    asyncio.run(runner._verify_then_notify(
        database.get_event(event_id), np.zeros((544, 960, 3), dtype=np.uint8),
        RecordingNotifier(),
    ))

    assert sent_image(server).size == (960, 544)


def test_a_zone_deleted_after_its_event_still_gets_verified(vlm_config, database, server):
    """A missing zone must cost the crop and nothing else. The failure mode
    being avoided is an auto-routed alert that quietly stops being checked."""
    server.replies = [{"triggered": True, "confidence": 0.9, "reason": "five people",
                       "box_2d": None}]
    vlm_config.vlm.verify.enabled = True
    event_id = zoned_event(database, CONVEYOR_SIDE)
    database.delete_zone("zone_a")
    notifier = RecordingNotifier()
    runner = make_runner(vlm_config, database, server, notifier=notifier)

    asyncio.run(runner._verify_then_notify(
        database.get_event(event_id), np.zeros((544, 960, 3), dtype=np.uint8), notifier
    ))

    assert server.calls == 1
    assert sent_image(server).size == (960, 544)
    assert notifier.dispatched == [event_id]


def test_a_small_zone_is_grown_rather_than_sent_as_a_postage_stamp():
    """A vision encoder sees a grid of patches; a 130x140 cut-out is fewer
    pixels of the zone than the wide shot already had."""
    frame = np.zeros((544, 960, 3), dtype=np.uint8)
    corner = [[0.0, 0.6], [0.1, 0.6], [0.1, 0.7], [0.0, 0.7]]

    crop = crop_to_zone(frame, corner)

    assert crop is not None
    assert crop.size == (MIN_CROP_PX, MIN_CROP_PX)


def test_a_polygon_that_cannot_be_read_declines_to_crop():
    frame = np.zeros((544, 960, 3), dtype=np.uint8)
    assert crop_to_zone(frame, [[0.1, 0.1], [0.2, 0.2]]) is None      # not a polygon
    assert crop_to_zone(frame, [["x", 0.1], [0.2, 0.2], [0.3, 0.3]]) is None
    assert crop_to_zone(frame, []) is None


def test_the_crop_keeps_the_zone_inside_it():
    """The point of the pad is that people standing on the zone's edges survive
    it — a crop that clips the polygon is measuring a different question."""
    frame = np.zeros((544, 960, 3), dtype=np.uint8)
    crop = crop_to_zone(frame, CONVEYOR_SIDE)

    assert crop is not None
    left, top = round(0.5 * 960), round(0.5 * 544)
    right, bottom = round(0.9 * 960), round(0.9 * 544)
    # The crop's own origin, recovered from the padded bounds it was cut at.
    origin_x, origin_y = round(0.46 * 960), 199
    assert origin_x <= left and origin_y <= top
    assert origin_x + crop.width >= right and origin_y + crop.height >= bottom


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def test_stats_expose_model_queue_depth_and_last_inference_age(vlm_config, database, server):
    seed_policy(database)
    seed_camera(database, "cam_a")
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.schedules["phone_usage_on_line:cam_a"].next_due = 0.0
    runner.offer(FakeFrame("cam_a", ts=0.0))
    tick(runner, now=0.0)

    stats = runner.stats()
    assert stats["model"] == MODEL
    assert stats["evaluations"] == 1
    assert stats["endpoint"] == server.base_url
    inference = stats["inference"]
    assert inference["requests"] == 1
    assert inference["queue_depth"] == 0
    assert inference["last_inference_age_sec"] is not None
    assert inference["avg_latency_sec"] > 0
    assert stats["verification"]["enabled"] is False


def test_health_report_includes_the_vlm_section(config, database):
    from argus.health import health_report

    config.vlm.base_url = "http://127.0.0.1:9/v1"
    report = health_report(config, database)
    assert report["vlm"]["available"] is False
    assert report["vlm"]["source"] == "probe"
    assert report["vlm"]["model"] == config.vlm.model
