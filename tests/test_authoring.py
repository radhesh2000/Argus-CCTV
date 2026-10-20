"""Authoring: where a policy applies, and the live prompt test.

Three seams, and the reason each is tested here rather than beside its module:

* **Scope semantics** are one rule (``argus.db.scope_allows``) obeyed by two
  engines that share no code — the detector loop in ``argus/engine.py`` and the
  VLM scheduler in ``argus/vlm/scheduler.py``. A rule with two independent
  implementations is a rule that drifts, so both are driven from one file
  against the same fixture.
* **The live-frame protocol** (``argus/live.py``) has two ends in two
  processes. Here they are two objects in one test, which is the only place the
  agreement between them is written down as an assertion.
* **The prompt test's degraded modes** are the point of the feature, not an
  edge of it: an authoring panel that answers "no engine" and "model down"
  clearly is the difference between a button an operator trusts and one they
  stop pressing.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from argus.api.app import create_app
from argus.config import ArgusConfig
from argus.db import (
    Camera,
    Database,
    Policy,
    PolicyScope,
    Zone,
    scope_allows,
    scope_cameras,
    scope_zones,
)
from argus.live import LiveFrames, UnsafeCameraId

pytest.importorskip("supervision")

LEFT = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
RIGHT = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
WIDTH, HEIGHT = 320, 180


# --------------------------------------------------------------------------
# the rule itself
# --------------------------------------------------------------------------


def test_an_empty_scope_is_every_camera():
    """The whole compatibility story of migration 5 is this assertion: an
    appliance that has never scoped anything must behave exactly as before."""
    assert scope_cameras([]) is None
    assert scope_zones([], "anything") == [None]
    assert scope_allows([], "any_camera", "any_zone") is True
    assert scope_allows([], "any_camera", None) is True


def test_a_scoped_policy_applies_only_where_it_was_put():
    scope = [{"camera_id": "line_a", "zone_id": None}]
    assert scope_allows(scope, "line_a", "any_zone") is True
    assert scope_allows(scope, "line_b", None) is False


def test_a_zone_scope_narrows_to_that_zone():
    scope = [{"camera_id": "line_a", "zone_id": "press"}]
    assert scope_allows(scope, "line_a", "press") is True
    assert scope_allows(scope, "line_a", "packing") is False
    # A zone-scoped policy has nothing to say about the camera as a whole.
    assert scope_allows(scope, "line_a", None) is False


def test_a_whole_camera_row_beats_a_zone_row_beside_it():
    """"This camera, and also this zone on it" is the camera. Otherwise an
    operator who ticked both would silently get the narrower of the two."""
    scope = [{"camera_id": "line_a", "zone_id": None},
             {"camera_id": "line_a", "zone_id": "press"}]
    assert scope_zones(scope, "line_a") == [None]
    assert scope_allows(scope, "line_a", "packing") is True


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------


@pytest.fixture
def fleet(database: Database) -> Database:
    database.upsert_camera(Camera(id="line_a", name="Line A", source="rtsp://cam/a"))
    database.upsert_camera(Camera(id="line_b", name="Line B", source="rtsp://cam/b"))
    database.upsert_zone(Zone(id="a_left", camera_id="line_a", name="Left",
                              polygon=LEFT, sanctioned_limit=1))
    database.upsert_zone(Zone(id="a_right", camera_id="line_a", name="Right",
                              polygon=RIGHT, sanctioned_limit=1))
    database.upsert_zone(Zone(id="b_left", camera_id="line_b", name="Left",
                              polygon=LEFT, sanctioned_limit=1))
    return database


def test_scope_round_trips_and_is_attached_to_every_read(fleet: Database):
    fleet.upsert_policy(Policy(id="p", kind="vlm", detector_key="vlm_prompt", label="P",
                               prompt="anything", scope=[PolicyScope("line_a", "a_left")]))
    assert fleet.get_policy("p")["scope"] == [{"camera_id": "line_a", "zone_id": "a_left"}]
    # Attached by list_policies too — both engines read policies that way, and a
    # scope they had to fetch separately is a scope one of them would forget.
    assert fleet.list_policies()[0]["scope"] == [{"camera_id": "line_a", "zone_id": "a_left"}]


def test_an_upsert_that_says_nothing_about_scope_leaves_it_alone(fleet: Database):
    """`Policy.scope=None` is not `[]`. Anything that upserts a policy for an
    unrelated reason must not silently unscope it."""
    fleet.upsert_policy(Policy(id="p", kind="detector", detector_key="manpower_headcount",
                               label="P", scope=["line_a"]))
    fleet.upsert_policy(Policy(id="p", kind="detector", detector_key="manpower_headcount",
                               label="Renamed"))
    assert fleet.get_policy("p")["scope"] == [{"camera_id": "line_a", "zone_id": None}]
    fleet.upsert_policy(Policy(id="p", kind="detector", detector_key="manpower_headcount",
                               label="Renamed", scope=[]))
    assert fleet.get_policy("p")["scope"] == []


def test_a_scope_naming_nothing_is_refused(fleet: Database):
    fleet.upsert_policy(Policy(id="p", kind="detector", detector_key="manpower_headcount",
                               label="P"))
    with pytest.raises(ValueError, match="no such camera"):
        fleet.set_policy_scope("p", ["nope"])
    with pytest.raises(ValueError, match="no such zone"):
        fleet.set_policy_scope("p", [{"camera_id": "line_a", "zone_id": "nope"}])
    # A zone that exists but on another camera is the mistake worth naming:
    # it produces a policy that is scoped to nowhere at all.
    with pytest.raises(ValueError, match="belongs to camera"):
        fleet.set_policy_scope("p", [{"camera_id": "line_b", "zone_id": "a_left"}])
    assert fleet.get_policy("p")["scope"] == []


def test_deleting_a_zone_widens_the_policy_back_to_its_camera(fleet: Database):
    fleet.upsert_policy(Policy(id="p", kind="detector", detector_key="manpower_headcount",
                               label="P", scope=[PolicyScope("line_a", "a_left"),
                                                 PolicyScope("line_a", "a_right")]))
    fleet.delete_zone("a_left")
    assert fleet.get_policy("p")["scope"] == [{"camera_id": "line_a", "zone_id": "a_right"}]
    fleet.delete_zone("a_right")
    # No rows left: the policy means "everywhere" again. Visible on the card
    # rather than silent, which is why this is acceptable — see migration 5.
    assert fleet.get_policy("p")["scope"] == []


# --------------------------------------------------------------------------
# the detector loop honours it
# --------------------------------------------------------------------------


class FakeDetector:
    """Two people in the left half of the frame, on every camera."""

    def detect(self, images):
        import supervision as sv

        boxes = np.array([[10, 40, 40, 170], [60, 40, 90, 170]], dtype=np.float32)
        return [
            sv.Detections(
                xyxy=boxes.copy(),
                confidence=np.full(2, 0.9, dtype=np.float32),
                class_id=np.zeros(2, dtype=int),
            )
            for _ in images
        ]


def _frame(camera_id: str, ts: float):
    from argus.ingest.decoder import Frame

    return Frame(camera_id=camera_id, index=1, ts=ts,
                 image=np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8))


def _run_breach(config: ArgusConfig, database: Database, camera_id: str) -> list[dict]:
    """Ten seconds of analysed frames through the real ``Engine._process``.

    Ten and not one, because since ``detection.continuity`` a zone has to be
    *seen* over its limit for ``sustain_sec`` before anything is written — a
    single frame is a measurement, not an incident. What is being asserted here
    is unchanged: whether the policy applies to this camera and this zone at
    all.
    """
    from argus.engine import Engine

    detector = FakeDetector()
    engine = Engine(config, database, use_gateway=False, detector=detector, record=False,
                    notify=False, housekeeping=False, vlm=False, index=False)
    engine._refresh_policies()
    runtime = engine._build_runtime(database.get_camera(camera_id))
    detections = detector.detect([None])[0]
    for tick in range(51):                          # 10 s at the shipped 5 fps
        engine._process(runtime, _frame(camera_id, 1_000_000.0 + tick * 0.2), detections)
    return database.recent_events()


def test_a_scoped_detector_policy_fires_only_on_its_camera(config, fleet: Database):
    fleet.upsert_policy(Policy(
        id="manpower_headcount", kind="detector", detector_key="manpower_headcount",
        label="Manpower", stage="reviewed", lifecycle="production",
        scope=[PolicyScope("line_a")],
    ))
    assert [e["camera_id"] for e in _run_breach(config, fleet, "line_a")] == ["line_a"]
    # Same two people, same zone geometry, same over-limit condition — and on
    # line_b the policy does not apply, so nothing is raised.
    _run_breach(config, fleet, "line_b")
    assert [e["camera_id"] for e in fleet.recent_events()] == ["line_a"]


def test_an_unscoped_detector_policy_still_fires_everywhere(config, fleet: Database):
    fleet.upsert_policy(Policy(
        id="manpower_headcount", kind="detector", detector_key="manpower_headcount",
        label="Manpower", stage="reviewed", lifecycle="production",
    ))
    _run_breach(config, fleet, "line_a")
    _run_breach(config, fleet, "line_b")
    assert sorted({e["camera_id"] for e in fleet.recent_events()}) == ["line_a", "line_b"]


def test_a_zone_scoped_detector_policy_fires_only_in_that_zone(config, fleet: Database):
    """Both zones on line_a are over their limit of one — the people are in the
    left half, and `Right` counts nobody, so only `Left` can fire anyway. The
    assertion that matters is the mirror: scoping to `Right` fires nothing."""
    fleet.upsert_policy(Policy(
        id="manpower_headcount", kind="detector", detector_key="manpower_headcount",
        label="Manpower", stage="reviewed", lifecycle="production",
        scope=[PolicyScope("line_a", "a_left")],
    ))
    assert [e["zone_id"] for e in _run_breach(config, fleet, "line_a")] == ["a_left"]

    fleet.set_policy_scope("manpower_headcount", [PolicyScope("line_a", "a_right")])
    before = len(fleet.recent_events())
    _run_breach(config, fleet, "line_a")
    assert len(fleet.recent_events()) == before


# --------------------------------------------------------------------------
# the VLM scheduler honours it
# --------------------------------------------------------------------------


def _runner(config: ArgusConfig, database: Database):
    from argus.vlm.scheduler import VlmRunner

    config.vlm.enabled = True
    return VlmRunner(config, database, log=lambda *_a, **_k: None)


def test_the_scheduler_builds_one_schedule_per_scoped_place(config, fleet: Database):
    fleet.upsert_policy(Policy(
        id="phones", kind="vlm", detector_key="vlm_prompt", label="Phones",
        stage="reviewed", lifecycle="production", prompt="anyone on a phone?",
        scope=[PolicyScope("line_a", "a_left"), PolicyScope("line_a", "a_right")],
    ))
    runner = _runner(config, fleet)
    runner.refresh(now=0.0)
    assert sorted(runner.schedules) == ["phones:line_a:a_left", "phones:line_a:a_right"]
    assert {s.camera_id for s in runner.schedules.values()} == {"line_a"}
    assert {s.zone_name for s in runner.schedules.values()} == {"Left", "Right"}


def test_an_unscoped_prompt_policy_still_runs_on_every_camera(config, fleet: Database):
    fleet.upsert_policy(Policy(
        id="phones", kind="vlm", detector_key="vlm_prompt", label="Phones",
        stage="reviewed", lifecycle="production", prompt="anyone on a phone?",
    ))
    runner = _runner(config, fleet)
    runner.refresh(now=0.0)
    assert sorted(runner.schedules) == ["phones:line_a", "phones:line_b"]


def test_scoping_a_policy_drops_the_schedules_it_no_longer_covers(config, fleet: Database):
    fleet.upsert_policy(Policy(
        id="phones", kind="vlm", detector_key="vlm_prompt", label="Phones",
        stage="reviewed", lifecycle="production", prompt="anyone on a phone?",
    ))
    runner = _runner(config, fleet)
    runner.refresh(now=0.0)
    schedule = runner.schedules["phones:line_a"]
    schedule.evaluations = 7

    fleet.set_policy_scope("phones", [PolicyScope("line_a")])
    runner.refresh(now=100.0)
    assert sorted(runner.schedules) == ["phones:line_a"]
    # The surviving schedule keeps its phase and its counters: editing scope
    # must not reset the sampling clock of a camera that was already running.
    assert runner.schedules["phones:line_a"].evaluations == 7


def test_a_zone_scoped_schedule_is_shown_the_crop(config, fleet: Database):
    """The measured reason this feature exists: the same model, prompt and
    footage answered 12/12 on zone crops and 3/12 on the wide shots (see
    docs/VLM_EVAL.md). ``_view`` is where that happens."""
    fleet.upsert_policy(Policy(
        id="phones", kind="vlm", detector_key="vlm_prompt", label="Phones",
        stage="reviewed", lifecycle="production", prompt="anyone on a phone?",
        scope=[PolicyScope("line_a", "a_left")],
    ))
    runner = _runner(config, fleet)
    runner.refresh(now=0.0)
    schedule = runner.schedules["phones:line_a:a_left"]

    image = np.zeros((544, 960, 3), dtype=np.uint8)
    view, crop = runner._view(schedule, image)
    assert crop is not None
    # The left half plus `polygon_bounds`' padding — 0.04 on the free side, the
    # rest clamped at the frame edge.
    assert view.size == (518, 544)
    assert crop == pytest.approx((0.0, 0.0, 518 / 960, 1.0))


def test_a_zone_covering_the_whole_view_is_asked_wide(config, fleet: Database):
    """A crop of 100% of the frame is not a closer look — `crop_to_zone`
    declines it, and the schedule has to survive that rather than skip."""
    fleet.upsert_zone(Zone(id="a_all", camera_id="line_a", name="All",
                           polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    fleet.upsert_policy(Policy(
        id="phones", kind="vlm", detector_key="vlm_prompt", label="Phones",
        stage="reviewed", lifecycle="production", prompt="anyone on a phone?",
        scope=[PolicyScope("line_a", "a_all")],
    ))
    runner = _runner(config, fleet)
    runner.refresh(now=0.0)
    schedule = runner.schedules["phones:line_a:a_all"]
    image = np.zeros((544, 960, 3), dtype=np.uint8)
    view, crop = runner._view(schedule, image)
    assert crop is None and view is image


def test_a_box_from_a_crop_is_mapped_back_onto_the_frame():
    """A model shown the right half answers in the right half's coordinates.
    Stored unmapped, every grounded box would sit in the top-left corner."""
    from argus.vlm.evidence import box_in_frame

    assert box_in_frame((0.0, 0.0, 1.0, 1.0), (0.5, 0.0, 1.0, 1.0)) == (0.5, 0.0, 1.0, 1.0)
    assert box_in_frame((0.5, 0.5, 1.0, 1.0), (0.0, 0.0, 0.5, 0.5)) == (0.25, 0.25, 0.5, 0.5)


# --------------------------------------------------------------------------
# the live-frame protocol
# --------------------------------------------------------------------------


def test_a_frame_is_published_only_after_it_is_asked_for(tmp_path: Path):
    """The whole cost argument for this design. No request, no encode."""
    live = LiveFrames(tmp_path / "live")
    image = np.full((180, 320, 3), 60, dtype=np.uint8)

    assert live.wanted() == set()
    assert live.read("cam") is None

    live.request("cam")
    assert live.wanted() == {"cam"}
    assert live.publish({"cam": image}) == ["cam"]

    found = live.read("cam")
    assert found is not None
    payload, written = found
    assert payload[:2] == b"\xff\xd8"                   # a JPEG, at decode size
    assert time.time() - written < 5


def test_an_expired_request_is_dropped(tmp_path: Path):
    live = LiveFrames(tmp_path / "live")
    live.request("cam", ttl_sec=1.0, now=time.time() - 3600)
    assert live.wanted() == set()


def test_waiting_ignores_a_frame_from_before_the_request(tmp_path: Path):
    """The file survives a restart of the engine. Answering a prompt test with
    yesterday's frame is worse than answering "nothing is publishing"."""
    live = LiveFrames(tmp_path / "live")
    live.publish({"cam": np.zeros((32, 32, 3), dtype=np.uint8)})
    assert live.wait_for("cam", newer_than=time.time() + 60, timeout=0.2) is None
    assert live.wait_for("cam", newer_than=0, timeout=0.2) is not None


def test_a_camera_id_that_is_a_path_is_refused(tmp_path: Path):
    live = LiveFrames(tmp_path / "live")
    for bad in ("../../etc/passwd", "a/b", "", "."):
        with pytest.raises(UnsafeCameraId):
            live.frame_path(bad)


def test_stopping_the_engine_clears_what_it_published(tmp_path: Path):
    live = LiveFrames(tmp_path / "live")
    live.request("cam")
    live.publish({"cam": np.zeros((32, 32, 3), dtype=np.uint8)})
    live.clear()
    assert live.read("cam") is None and live.wanted() == set()


def test_the_engine_publishes_the_frame_it_analysed(config, fleet: Database):
    """End to end across the two ends of the protocol, in one process: the
    picture the API gets back is the one the detector was shown."""
    from argus.engine import Engine

    detector = FakeDetector()
    engine = Engine(config, fleet, use_gateway=False, detector=detector, record=False,
                    notify=False, housekeeping=False, vlm=False, index=False)
    engine._refresh_policies()
    runtime = engine._build_runtime(fleet.get_camera("line_a"))
    engine.runtimes["line_a"] = runtime

    engine._publish_live_frames()                       # nobody asked yet
    assert engine.live.read("line_a") is None

    frame = _frame("line_a", 1_000_000.0)
    engine._process(runtime, frame, detector.detect([None])[0])
    LiveFrames.from_config(config).request("line_a")
    engine._publish_live_frames()

    from PIL import Image as PilImage
    import io

    payload, _written = engine.live.read("line_a")
    with PilImage.open(io.BytesIO(payload)) as published:
        assert published.size == (WIDTH, HEIGHT)


# --------------------------------------------------------------------------
# the API
# --------------------------------------------------------------------------


@pytest.fixture
def client(config: ArgusConfig, fleet: Database, sign_in) -> TestClient:
    config.vlm.enabled = True
    # Point the VLM at a port nothing listens on. Without this these tests pass
    # or fail depending on whether somebody has `argus vlm start` running on the
    # development box — which is exactly the kind of test that is green on the
    # machine that wrote it.
    config.vlm.base_url = "http://127.0.0.1:9/v1"
    client = TestClient(create_app(config, fleet))
    sign_in(client, "RESPOND")
    return client


def sse(response) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    frames = []
    for block in response.text.split("\n\n"):
        if not block.strip():
            continue
        name = block.split("event: ", 1)[1].split("\n", 1)[0]
        data = block.split("data: ", 1)[1]
        frames.append((name, json.loads(data)))
    return frames


def test_a_policy_can_be_authored_with_a_scope(client: TestClient, fleet: Database):
    created = client.post("/api/policies", json={
        "kind": "vlm", "detector_key": "vlm_prompt", "label": "No phones at the press",
        "prompt": "Is anyone using a phone?", "interval_sec": 10, "cooldown_sec": 60,
        "severity": "high", "routed_to": "line_supervisor",
        "scope": [{"camera_id": "line_a", "zone_id": "a_left"}],
    })
    assert created.status_code == 201, created.text
    policy = created.json()["policy"]
    assert policy["id"] == "no_phones_at_the_press"
    # Born draft/shadow whatever the body said: the staged rollout is the product.
    assert (policy["lifecycle"], policy["stage"]) == ("draft", "shadow")
    assert policy["scope"] == [{"camera_id": "line_a", "zone_id": "a_left"}]


def test_a_create_with_a_bad_scope_leaves_no_policy_behind(client: TestClient, fleet: Database):
    failed = client.post("/api/policies", json={
        "kind": "vlm", "detector_key": "vlm_prompt", "label": "Ghost",
        "prompt": "anything", "scope": [{"camera_id": "line_z"}],
    })
    assert failed.status_code == 422
    assert "no such camera" in failed.json()["detail"]
    assert fleet.get_policy("ghost") is None


def test_editing_scope_and_prompt_together_is_all_or_nothing(client: TestClient, fleet: Database):
    fleet.upsert_policy(Policy(id="p", kind="vlm", detector_key="vlm_prompt", label="P",
                               prompt="original"))
    failed = client.patch("/api/policies/p", json={
        "prompt": "rewritten", "scope": [{"camera_id": "line_z"}],
    })
    assert failed.status_code == 422
    assert fleet.get_policy("p")["prompt"] == "original"


def test_widening_a_policy_back_to_every_camera(client: TestClient, fleet: Database):
    fleet.upsert_policy(Policy(id="p", kind="vlm", detector_key="vlm_prompt", label="P",
                               prompt="x", scope=[PolicyScope("line_a")]))
    assert client.patch("/api/policies/p", json={"scope": []}).status_code == 200
    assert fleet.get_policy("p")["scope"] == []
    # Omitting the field is not the same as sending []; it leaves scope alone.
    fleet.set_policy_scope("p", [PolicyScope("line_a")])
    assert client.patch("/api/policies/p", json={"label": "Renamed"}).status_code == 200
    assert fleet.get_policy("p")["scope"] == [{"camera_id": "line_a", "zone_id": None}]


def test_a_camera_source_that_would_run_a_program_is_refused(client: TestClient):
    """`exec:` is a go2rtc source type and a file source lands on an ffmpeg
    command line. Authoring cameras from the console is what made this a door."""
    for source in ("exec:sh -c id", "ffmpeg:rtsp://cam/1", "/tmp/x.mp4 -f null -",
                   "rtsp://cam/1 -y /etc/passwd", "relative/path.mp4"):
        refused = client.post("/api/cameras", json={"name": "Bad", "source": source})
        assert refused.status_code == 422, f"{source} was accepted"


def test_a_real_camera_url_is_accepted(client: TestClient):
    ok = client.post("/api/cameras", json={"name": "Dock 3", "source": "rtsp://10.0.0.9:554/h264"})
    assert ok.status_code == 201, ok.text
    assert ok.json()["camera"]["kind"] == "rtsp"


def test_probing_needs_exactly_one_of_source_or_camera(client: TestClient):
    assert client.post("/api/cameras/probe", json={}).status_code == 422
    both = client.post("/api/cameras/probe",
                       json={"source": "rtsp://cam/1", "camera_id": "line_a"})
    assert both.status_code == 422


def test_a_probe_falls_back_to_ffprobe_when_the_gateway_is_not_running(client: TestClient):
    """go2rtc lives inside `argus run`, so with only the API up there is no
    stream to ask about. The probe then opens the source itself and says which
    route answered — never a 502 the console would render as a broken camera."""
    answer = client.post("/api/cameras/probe", json={"camera_id": "line_a"})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["route"] == "ffprobe"
    assert body["ok"] is False              # rtsp://cam/a resolves to nothing


def test_a_probe_opens_a_real_file(client: TestClient, sample_video):
    answer = client.post("/api/cameras/probe", json={"source": str(sample_video)})
    body = answer.json()
    assert body["ok"] is True and body["route"] == "ffprobe", body
    assert body["medias"] and "h264" in body["medias"][0]


def test_the_prompt_test_says_when_no_engine_is_publishing(client: TestClient):
    answer = client.post("/api/policies/test", json={
        "camera_id": "line_a", "prompt": "Is anyone using a phone?", "wait_sec": 1,
    })
    assert answer.status_code == 200
    frames = sse(answer)
    assert [name for name, _ in frames] == ["error"]
    assert frames[0][1]["code"] == "no_frame"
    assert "argus run" in frames[0][1]["detail"]


def test_the_prompt_test_shows_the_frame_before_it_asks(client: TestClient, config):
    """With the model down, the panel still gets a picture. That ordering is
    the whole reason this streams: half the author's questions are answered by
    seeing the frame, and they should not be blocked on an inference."""
    live = LiveFrames.from_config(config)
    live.publish({"line_a": np.full((180, 320, 3), 90, dtype=np.uint8)})

    answer = client.post("/api/policies/test", json={
        "camera_id": "line_a", "prompt": "Is anyone using a phone?", "wait_sec": 2,
    })
    names = [name for name, _ in sse(answer)]
    assert names[0] == "frame"
    assert names[-1] == "error"

    frame = sse(answer)[0][1]
    assert frame["camera_id"] == "line_a" and frame["cropped"] is False
    assert (frame["width"], frame["height"]) == (320, 180)
    assert base64.b64decode(frame["image"].split(",", 1)[1])[:2] == b"\xff\xd8"

    failure = sse(answer)[-1][1]
    # No vLLM in this test process, so the endpoint is refused rather than slow.
    assert failure["code"] in ("vlm_unavailable", "failed")


def test_the_prompt_test_says_when_prompt_policies_are_off(client: TestClient, config):
    config.vlm.enabled = False
    LiveFrames.from_config(config).publish(
        {"line_a": np.full((180, 320, 3), 90, dtype=np.uint8)}
    )
    frames = sse(client.post("/api/policies/test", json={
        "camera_id": "line_a", "prompt": "anything", "wait_sec": 2,
    }))
    assert [name for name, _ in frames] == ["frame", "error"]
    assert frames[-1][1]["code"] == "vlm_disabled"


def test_the_prompt_test_crops_to_a_zone(client: TestClient, config):
    LiveFrames.from_config(config).publish(
        {"line_a": np.full((544, 960, 3), 90, dtype=np.uint8)}
    )
    frames = sse(client.post("/api/policies/test", json={
        "camera_id": "line_a", "prompt": "anything", "zone_id": "a_left", "wait_sec": 2,
    }))
    frame = dict(frames)["frame"]
    assert frame["cropped"] is True
    assert (frame["width"], frame["height"]) == (518, 544)     # padded left half
    assert (frame["frame_width"], frame["frame_height"]) == (960, 544)
    assert frame["zone_name"] == "Left"


def test_the_prompt_test_refuses_a_zone_from_another_camera(client: TestClient):
    refused = client.post("/api/policies/test", json={
        "camera_id": "line_b", "prompt": "anything", "zone_id": "a_left",
    })
    assert refused.status_code == 422
    assert "not on camera" in refused.json()["detail"]


def test_the_prompt_test_needs_something_to_ask(client: TestClient, fleet: Database):
    fleet.upsert_policy(Policy(id="counting", kind="detector",
                               detector_key="manpower_headcount", label="Counting"))
    empty = client.post("/api/policies/test",
                        json={"camera_id": "line_a", "policy_id": "counting"})
    assert empty.status_code == 422
    assert "prompt" in empty.json()["detail"]


def test_the_prompt_test_creates_nothing(client: TestClient, fleet: Database, config):
    """A test is a question. A panel that filed events while an author tuned a
    sentence would poison the precision statistics the promotion gate reads."""
    LiveFrames.from_config(config).publish(
        {"line_a": np.full((180, 320, 3), 90, dtype=np.uint8)}
    )
    client.post("/api/policies/test",
                json={"camera_id": "line_a", "prompt": "anything", "wait_sec": 2})
    assert fleet.recent_events() == []
    assert list(config.snapshots_dir.rglob("*.jpg")) == []


# --------------------------------------------------------------------------
# the seed file
# --------------------------------------------------------------------------


def test_seed_reads_both_spellings_of_a_scope(database: Database, tmp_path: Path):
    from argus.seed import SeedError, apply_seed

    data = {
        "cameras": [{"id": "line_a", "name": "A", "source": "rtsp://cam/a"},
                    {"id": "line_b", "name": "B", "source": "rtsp://cam/b"}],
        "zones": [{"id": "a_left", "camera_id": "line_a", "name": "Left", "polygon": LEFT}],
        "policies": [
            {"id": "p1", "kind": "vlm", "detector_key": "vlm_prompt", "label": "P1",
             "prompt": "x", "cameras": ["line_a", {"camera": "line_b"}]},
            {"id": "p2", "kind": "vlm", "detector_key": "vlm_prompt", "label": "P2",
             "prompt": "x", "cameras": [{"camera": "line_a", "zone": "a_left"}]},
            {"id": "p3", "kind": "vlm", "detector_key": "vlm_prompt", "label": "P3",
             "prompt": "x"},
        ],
    }
    apply_seed(database, data)
    assert database.get_policy("p1")["scope"] == [
        {"camera_id": "line_a", "zone_id": None}, {"camera_id": "line_b", "zone_id": None},
    ]
    assert database.get_policy("p2")["scope"] == [{"camera_id": "line_a", "zone_id": "a_left"}]
    assert database.get_policy("p3")["scope"] == []

    # Declarative: dropping the key from the file unscopes the policy, exactly
    # as editing `prompt:` there overrides an edit made in the console.
    data["policies"][0].pop("cameras")
    apply_seed(database, data)
    assert database.get_policy("p1")["scope"] == []

    data["policies"][0]["cameras"] = ["line_z"]
    with pytest.raises(SeedError, match=r"policies\[0\].*no such camera"):
        apply_seed(database, data)


def test_the_shipped_seed_file_still_loads(database: Database):
    from argus.seed import load_seed, apply_seed

    result = apply_seed(database, load_seed())
    assert result.policies >= 4
    assert database.get_policy("phone_usage_on_line")["scope"] != []
