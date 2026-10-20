"""API tests.

Everything runs against a real FastAPI app wired to a real (temporary) SQLite
file — no mocked database. The point of these tests is the seam between HTTP
and storage, and a fake on either side would test the fake.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from argus.api.app import create_app
from argus.config import ArgusConfig
from argus.db import Camera, Database, Event, Policy, Zone
from tests.conftest import TEST_PASSWORD

SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


@pytest.fixture
def anon(config: ArgusConfig, database: Database) -> TestClient:
    """A client with no session. Used by the handful of tests that are about
    being logged out; everything else uses `client`, which is signed in."""
    config.api.sse_poll_sec = 0.05
    return TestClient(create_app(config, database))


@pytest.fixture
def client(anon: TestClient, sign_in) -> TestClient:
    """Signed in as ADMIN.

    Phase 5 put a session in front of every route, and these tests are about
    what the routes *do*. Making the default client an administrator keeps them
    that way; who may call what is tested once, as a matrix, in test_auth.py.
    """
    sign_in(anon, "ADMIN")
    return anon


@pytest.fixture
def fleet(database: Database) -> dict[str, str]:
    """One camera, one zone, one promoted policy — the minimum that produces
    an event worth reviewing."""
    database.upsert_camera(Camera(id="line_a", name="Line A", source="rtsp://cam/line-a"))
    database.upsert_zone(Zone(id="line_a_pack", camera_id="line_a", name="Packing",
                              polygon=SQUARE, sanctioned_limit=3))
    database.upsert_policy(Policy(
        id="headcount_line_a", kind="detector", detector_key="manpower_headcount",
        label="Line A headcount", stage="reviewed", lifecycle="production",
        severity="high", routed_to="safety_manager",
    ))
    return {"camera": "line_a", "zone": "line_a_pack", "policy": "headcount_line_a"}


def make_event(database: Database, fleet: dict[str, str], *, stage: str = "reviewed",
               severity: str = "high", ts: float | None = None, **overrides) -> int:
    return database.insert_event(Event(
        camera_id=fleet["camera"], zone_id=fleet["zone"], policy_id=fleet["policy"],
        detector_name=fleet["policy"], detector_type="detector", severity=severity,
        description="Zone 'Packing' over sanctioned limit: 5 people (limit 3)",
        ts=ts if ts is not None else time.time(), stage=stage, confidence=0.91,
        routed_to="safety_manager", meta={"count": 5, "limit": 3, "zone_name": "Packing"},
        **overrides,
    ))


def make_event_with_boxes(
    database: Database, fleet: dict[str, str], boxes: list[list[float]], **overrides
) -> int:
    """Like `make_event`, but with real `meta.boxes` — for box-correction tests,
    which need a known track id to correct."""
    overrides.setdefault("stage", "reviewed")
    overrides.setdefault("severity", "high")
    return database.insert_event(Event(
        camera_id=fleet["camera"], zone_id=fleet["zone"], policy_id=fleet["policy"],
        detector_name=fleet["policy"], detector_type="detector",
        description="Zone 'Packing' over sanctioned limit: 5 people (limit 3)",
        ts=time.time(), confidence=0.91, routed_to="safety_manager",
        meta={"count": len(boxes), "limit": 3, "zone_name": "Packing", "boxes": boxes},
        **overrides,
    ))


# --------------------------------------------------------------------------
# cameras & zones
# --------------------------------------------------------------------------


def test_camera_crud_round_trip(client: TestClient):
    created = client.post("/api/cameras", json={"name": "Packing Line 2", "source": "rtsp://cam/2"})
    assert created.status_code == 201
    camera = created.json()["camera"]
    assert camera["id"] == "packing_line_2"
    assert camera["kind"] == "rtsp"
    assert camera["status"] == "offline"

    listed = client.get("/api/cameras").json()["cameras"]
    assert [c["id"] for c in listed] == ["packing_line_2"]
    assert listed[0]["zones"] == []

    patched = client.patch("/api/cameras/packing_line_2", json={"enabled": False})
    assert patched.json()["camera"]["enabled"] == 0

    assert client.delete("/api/cameras/packing_line_2").status_code == 200
    assert client.get("/api/cameras/packing_line_2").status_code == 404


def test_changing_a_source_rederives_its_kind(client: TestClient, tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really video, but it exists")
    client.post("/api/cameras", json={"id": "c1", "name": "C1", "source": "rtsp://cam/1"})
    patched = client.patch("/api/cameras/c1", json={"source": str(clip)})
    assert patched.json()["camera"]["kind"] == "file"


def test_duplicate_camera_id_is_a_conflict(client: TestClient):
    client.post("/api/cameras", json={"id": "c1", "name": "C1", "source": "rtsp://cam/1"})
    clash = client.post("/api/cameras", json={"id": "c1", "name": "Other", "source": "rtsp://cam/2"})
    assert clash.status_code == 409


def test_unknown_field_in_a_patch_is_rejected(client: TestClient):
    """A typo must fail loudly: silently changing nothing is the worst outcome."""
    client.post("/api/cameras", json={"id": "c1", "name": "C1", "source": "rtsp://cam/1"})
    assert client.patch("/api/cameras/c1", json={"nmae": "typo"}).status_code == 422


def test_zone_lifecycle_and_polygon_validation(client: TestClient):
    client.post("/api/cameras", json={"id": "c1", "name": "C1", "source": "rtsp://cam/1"})

    created = client.post("/api/cameras/c1/zones",
                          json={"name": "Packing", "polygon": SQUARE, "sanctioned_limit": 4})
    assert created.status_code == 201
    zone = created.json()["zone"]
    assert zone["polygon"] == SQUARE
    assert zone["sanctioned_limit"] == 4

    patched = client.patch(f"/api/zones/{zone['id']}", json={"sanctioned_limit": 2})
    assert patched.json()["zone"]["sanctioned_limit"] == 2

    assert client.delete(f"/api/zones/{zone['id']}").status_code == 200
    assert client.get(f"/api/zones/{zone['id']}").status_code == 404


def test_a_pixel_coordinate_polygon_is_refused(client: TestClient):
    """Zones are normalized 0-1. A polygon in pixels would be accepted and then
    silently never contain anybody, which is the worst kind of bug to ship."""
    client.post("/api/cameras", json={"id": "c1", "name": "C1", "source": "rtsp://cam/1"})
    response = client.post("/api/cameras/c1/zones",
                           json={"name": "Bad", "polygon": [[10, 10], [900, 10], [900, 500]]})
    assert response.status_code == 422
    assert "normalized" in response.text


def test_a_polygon_needs_three_points(client: TestClient):
    client.post("/api/cameras", json={"id": "c1", "name": "C1", "source": "rtsp://cam/1"})
    response = client.post("/api/cameras/c1/zones",
                           json={"name": "Line", "polygon": [[0.1, 0.1], [0.5, 0.5]]})
    assert response.status_code == 422


def test_zones_on_an_unknown_camera_are_404(client: TestClient):
    response = client.post("/api/cameras/ghost/zones", json={"name": "Z", "polygon": SQUARE})
    assert response.status_code == 404


# --------------------------------------------------------------------------
# camera credentials
# --------------------------------------------------------------------------
#
# A camera source is `rtsp://user:password@host/...`, and the API used to hand
# the cameras row back as it was stored — to every signed-in role. The password
# is write-only now (argus/redact.py): it goes in through POST and PATCH and
# comes back out of nothing.

#: Distinctive on purpose, so "it is nowhere in this response" is a substring
#: test and cannot pass by accident. The host is a documentation address (RFC
#: 5737) rather than anything on this box: nothing here dials it, and if a probe
#: ever stopped being faked it must not offer a wrong password to a real NVR.
CAMERA_PASSWORD = "s3cr3t-nvr-pass"
NVR_SOURCE = f"rtsp://test:{CAMERA_PASSWORD}@192.0.2.10:554/Streaming/Channels/101"
NVR_MASKED = "rtsp://test:***@192.0.2.10:554/Streaming/Channels/101"


@pytest.fixture
def nvr(config: ArgusConfig, database: Database, monkeypatch) -> dict[str, str]:
    """A camera with a password, and that password in every place it has leaked
    from: the row, the decoder's last error (stored and published), the ring
    recorder's, and go2rtc's stream table."""
    database.upsert_camera(Camera(id="nvr_101", name="NVR 101", source=NVR_SOURCE))
    database.upsert_zone(Zone(id="nvr_101_pack", camera_id="nvr_101", name="Packing",
                              polygon=SQUARE, sanctioned_limit=3))
    database.upsert_policy(Policy(
        id="headcount_nvr", kind="detector", detector_key="manpower_headcount",
        label="NVR headcount", stage="reviewed", lifecycle="production",
        severity="high", routed_to="safety_manager",
    ))
    fleet = {"camera": "nvr_101", "zone": "nvr_101_pack", "policy": "headcount_nvr"}
    event_id = make_event(database, fleet)

    # What an engine from before the decoder scrubbed its stderr left behind.
    quoted = f"[rtsp @ 0x55d0] {NVR_SOURCE}: Connection refused"
    database.set_camera_status("nvr_101", "error", quoted)
    write_engine_stats(config, [{
        "camera_id": "nvr_101", "fps": 0.0, "mode": "cuda",
        "last_error": quoted, "ring": {"running": False, "last_error": quoted},
    }])

    # go2rtc names the URL it dialled on every producer. Only that one call is
    # faked; everything else the API opens still goes where it would have gone.
    import io
    import urllib.request

    real_urlopen = urllib.request.urlopen

    def urlopen(url, *args, **kwargs):
        if str(url).endswith("/api/streams"):
            return io.BytesIO(json.dumps(
                {"nvr_101": {"producers": [{"url": NVR_SOURCE, "medias": ["video, H264"]}]}}
            ).encode())
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    # The poster route waits this long for an engine that is not there.
    monkeypatch.setattr("argus.api.routers.gateway.ENGINE_FRAME_WAIT_SEC", 0.05)
    return {**fleet, "event": str(event_id)}


def every_get(app, ids: dict[str, str]) -> list[str]:
    """Every GET the app documents, with its parameters filled in.

    Read from the OpenAPI document rather than listed here, so a route added
    next year is walked without anybody remembering to add it. A parameter this
    does not know how to fill is a KeyError naming it — louder than a skip.
    """
    values = {
        "camera_id": ids["camera"], "zone_id": ids["zone"], "policy_id": ids["policy"],
        "event_id": ids["event"], "user_id": "1", "q": "a person", "src": ids["camera"],
    }
    urls = []
    for path, operations in app.openapi()["paths"].items():
        # The SSE stream never ends under a TestClient; its frames are event
        # rows, which have no source column.
        if "get" not in operations or path == "/api/events/stream":
            continue
        required = [p for p in operations["get"].get("parameters", []) if p.get("required")]
        query = "&".join(f"{p['name']}={values[p['name']]}" for p in required if p["in"] == "query")
        url = path
        for parameter in required:
            if parameter["in"] == "path":
                url = url.replace("{" + parameter["name"] + "}", values[parameter["name"]])
        urls.append(f"{url}?{query}" if query else url)
    return urls


@pytest.mark.parametrize("role", ["REVIEW", "RESPOND", "ADMIN", "SYSTEM"])
def test_no_session_is_ever_sent_a_camera_password(anon: TestClient, sign_in, nvr, role: str):
    """Every GET route, as every role. REVIEW is the one that was reported — a
    read-only account reading every camera's password off `GET /api/cameras` —
    but nobody needs it back, so nobody gets it: an edit round-trips the mask.
    """
    sign_in(anon, role)
    urls = every_get(anon.app, nvr)
    assert {"/api/cameras", "/api/cameras/nvr_101", f"/api/events/{nvr['event']}",
            "/api/live/state", "/api/health", "/api/gateway/streams"} <= set(urls)

    answered = {}
    for url in urls:
        response = anon.get(url)
        answered[url] = response.status_code
        assert CAMERA_PASSWORD.encode() not in response.content, f"{role} GET {url}"

    if role == "SYSTEM":
        return
    # ...and not because the routes refused or the fixture missed: each of the
    # places the password used to be answered 200 with the mask in its place.
    listed = anon.get("/api/cameras").json()["cameras"]
    assert [camera["source"] for camera in listed] == [NVR_MASKED]
    assert "test:***@192.0.2.10" in listed[0]["status_detail"]
    assert anon.get("/api/cameras/nvr_101").json()["camera"]["source"] == NVR_MASKED
    assert anon.get(f"/api/events/{nvr['event']}").json()["camera"]["source"] == NVR_MASKED
    assert "test:***@" in anon.get("/api/live/state").json()["cameras"][0]["status_detail"]
    health = anon.get("/api/health").json()["cameras"][0]
    assert "test:***@" in health["last_error"] and "test:***@" in health["ring"]["last_error"]
    producer = anon.get("/api/gateway/streams").json()["streams"]["nvr_101"]["producers"][0]
    assert producer["url"] == NVR_MASKED
    assert answered["/api/cameras"] == 200


def test_enrolling_a_camera_does_not_echo_its_password(client: TestClient, database: Database):
    created = client.post("/api/cameras", json={"name": "NVR 101", "source": NVR_SOURCE})
    assert created.status_code == 201, created.text
    assert created.json()["camera"]["source"] == NVR_MASKED
    assert CAMERA_PASSWORD not in created.text
    assert database.get_camera("nvr_101")["source"] == NVR_SOURCE      # stored whole


def test_editing_a_camera_without_retyping_the_password_keeps_it(
    client: TestClient, database: Database, nvr
):
    """The edit form is filled from the masked source and sends it back. The
    `***` it never replaced must come out as the password that was stored."""
    shown = client.get("/api/cameras/nvr_101").json()["camera"]["source"]
    assert shown == NVR_MASKED

    renamed = client.patch("/api/cameras/nvr_101", json={"name": "Dock 1", "source": shown})
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["camera"]["name"] == "Dock 1"
    assert renamed.json()["camera"]["source"] == NVR_MASKED
    assert database.get_camera("nvr_101")["source"] == NVR_SOURCE

    # The common real edit: main stream to sub-stream, same login.
    moved = client.patch("/api/cameras/nvr_101",
                         json={"source": shown.replace("Channels/101", "Channels/102")})
    assert moved.status_code == 200, moved.text
    assert CAMERA_PASSWORD not in moved.text
    assert database.get_camera("nvr_101")["source"] == NVR_SOURCE.replace("/101", "/102")

    # A PATCH that does not mention the source leaves it alone.
    client.patch("/api/cameras/nvr_101", json={"enabled": False})
    assert database.get_camera("nvr_101")["source"] == NVR_SOURCE.replace("/101", "/102")

    # Typing over the mask is how a password is changed.
    retyped = client.patch("/api/cameras/nvr_101",
                           json={"source": shown.replace("***", "rotated-pass")})
    assert retyped.status_code == 200, retyped.text
    assert retyped.json()["camera"]["source"] == NVR_MASKED
    assert database.get_camera("nvr_101")["source"] == NVR_SOURCE.replace(
        CAMERA_PASSWORD, "rotated-pass")


@pytest.mark.parametrize("source", [
    "rtsp://test:***@198.51.100.7:554/Streaming/Channels/101",     # another host
    "rtsp://test:***@192.0.2.10:8554/Streaming/Channels/101",      # another port
    "rtsp://admin:***@192.0.2.10:554/Streaming/Channels/101",      # another user
    "http://test:***@192.0.2.10:554/Streaming/Channels/101",       # another scheme
])
def test_a_kept_password_cannot_be_pointed_somewhere_else(
    client: TestClient, database: Database, nvr, source: str
):
    """Otherwise masking would be decoration: change the host to a machine of
    your own, leave the `***`, and the appliance delivers the password it would
    not show you. The stored one is kept for the same login on the same host
    and for nothing else."""
    refused = client.patch("/api/cameras/nvr_101", json={"source": source})
    assert refused.status_code == 422, refused.text
    assert "type the password" in refused.json()["detail"]
    assert CAMERA_PASSWORD not in refused.text
    assert database.get_camera("nvr_101")["source"] == NVR_SOURCE


def test_a_mask_with_no_stored_password_behind_it_is_refused(client: TestClient, fleet):
    # A new camera has nothing stored to keep...
    new = client.post("/api/cameras", json={"name": "New", "source": NVR_MASKED})
    assert new.status_code == 422
    # ...and neither does one enrolled without a password.
    bare = client.patch("/api/cameras/line_a", json={"source": "rtsp://test:***@cam/line-a"})
    assert bare.status_code == 422
    # The rest of a masked source is still checked like any other.
    unsafe = client.patch("/api/cameras/line_a", json={"source": "rtsp://test:***@cam/a -y /etc/passwd"})
    assert unsafe.status_code == 422


def test_testing_an_edit_opens_the_camera_with_the_stored_password(
    client: TestClient, nvr, monkeypatch
):
    """"Test connection" on the edit form holds only the masked source. The
    probe fills it from the camera being edited — and ffprobe's complaint,
    which quotes the URL it was given, goes back without the password."""
    opened: list[str] = []

    def ffprobe(source: str, *, timeout: float = 30.0) -> dict:
        opened.append(source)
        raise RuntimeError(f"ffprobe failed for {source}: [rtsp @ 0x1] {source}: 401 Unauthorized")

    monkeypatch.setattr("argus.ingest.decoder.probe_source", ffprobe)

    edited = NVR_MASKED.replace("Channels/101", "Channels/102")
    answer = client.post("/api/cameras/probe", json={"source": edited, "password_from": "nvr_101"})
    assert answer.status_code == 200, answer.text
    assert opened == [NVR_SOURCE.replace("/101", "/102")]
    assert CAMERA_PASSWORD not in answer.text
    assert "test:***@192.0.2.10" in answer.json()["detail"]

    # The same guard as an edit: the stored password is not sent to another host.
    elsewhere = client.post("/api/cameras/probe", json={
        "source": "rtsp://test:***@198.51.100.7/Streaming/Channels/101", "password_from": "nvr_101",
    })
    assert elsewhere.status_code == 422
    # And a mask with no camera named has nothing to fill it.
    assert client.post("/api/cameras/probe", json={"source": NVR_MASKED}).status_code == 422
    assert len(opened) == 1

    # The stored camera's own re-test falls through to ffprobe with the gateway
    # down, and that answer is scrubbed too.
    stored = client.post("/api/cameras/probe", json={"camera_id": "nvr_101"})
    assert stored.status_code == 200 and CAMERA_PASSWORD not in stored.text


# --------------------------------------------------------------------------
# policies & the staged rollout
# --------------------------------------------------------------------------


def test_a_new_policy_always_starts_in_shadow_and_draft(client: TestClient):
    """The staged rollout is the product; it must not be skippable by POST."""
    created = client.post("/api/policies", json={
        "kind": "detector", "detector_key": "manpower_headcount",
        "label": "Line B headcount", "severity": "high", "routed_to": "safety_manager",
    })
    assert created.status_code == 201
    policy = created.json()["policy"]
    assert (policy["stage"], policy["lifecycle"]) == ("shadow", "draft")


def test_lifecycle_pilot_then_promote(client: TestClient):
    client.post("/api/policies", json={"id": "p1", "kind": "detector",
                                       "detector_key": "manpower_headcount", "label": "P1"})
    started = client.post("/api/policies/p1/pilot")
    assert started.status_code == 200
    policy = started.json()["policy"]
    assert policy["lifecycle"] == "pilot"
    assert policy["pilot"]["days_left"] == pytest.approx(2.0, abs=0.01)

    # Starting a pilot twice is a conflict, not a silent no-op.
    assert client.post("/api/policies/p1/pilot").status_code == 409

    extended = client.post("/api/policies/p1/pilot/extend", json={"extra_days": 3})
    assert extended.json()["policy"]["pilot_duration_days"] == 5.0

    # Promotion to a stage that still has a human in front of it needs no
    # evidence — that stage is how the evidence is collected.
    reviewed = client.post("/api/policies/p1/promote", json={"stage": "reviewed"})
    assert reviewed.status_code == 200
    assert reviewed.json()["policy"]["lifecycle"] == "production"
    assert reviewed.json()["policy"]["stage"] == "reviewed"

    # auto_routed is gated: no verdicts, no unattended email.
    refused = client.post("/api/policies/p1/promote", json={"routed_to": "safety_manager"})
    assert refused.status_code == 409
    detail = refused.json()["detail"]
    assert detail["code"] == "not_eligible"
    assert detail["reviewed"] == 0 and detail["precision"] is None
    assert detail["required"] == {"precision": 0.95, "reviews": 200}

    promoted = client.post("/api/policies/p1/promote", json={
        "routed_to": "safety_manager", "override": True,
        "override_reason": "validated on the same model at the Pune plant",
    })
    assert promoted.status_code == 200
    assert promoted.json()["policy"]["stage"] == "auto_routed"
    # The reason is on the row, with the numbers as they were at the time.
    note = promoted.json()["policy"]["promotion_note"]
    assert "Pune plant" in note and "no verdicts" in note


def test_an_override_without_a_reason_is_refused(client: TestClient):
    """The override exists so a site can say why. One that does not say why is
    just the gate switched off."""
    client.post("/api/policies", json={"id": "p1", "kind": "detector",
                                       "detector_key": "manpower_headcount", "label": "P1",
                                       "routed_to": "safety_manager"})
    silent = client.post("/api/policies/p1/promote", json={"override": True})
    assert silent.status_code == 422
    assert "reason" in silent.json()["detail"]
    assert client.get("/api/policies/p1").json()["policy"]["stage"] == "shadow"


def test_the_gate_is_the_appliances_own_bar(client: TestClient, config: ArgusConfig):
    """`rollout:` is config, and the console reads it from /api/config so there
    is one copy of the number rather than one per client."""
    config.rollout.min_reviews = 0
    config.rollout.min_precision = 0.0
    client.post("/api/policies", json={"id": "p1", "kind": "detector",
                                       "detector_key": "manpower_headcount", "label": "P1",
                                       "routed_to": "safety_manager"})
    assert client.post("/api/policies/p1/promote", json={}).status_code == 200
    assert client.get("/api/config").json()["rollout"]["min_reviews"] == 0


def test_config_reports_the_resolved_evidence_windows_and_budget(
    client: TestClient, config: ArgusConfig
):
    """The console must not reimplement the fallback chain in TypeScript and get
    it subtly different, so the appliance resolves it and sends the numbers."""
    config.retention.evidence.shadow_days = 1.0
    config.retention.evidence.reviewed_days = 2.0
    config.retention.evidence.confirmed_days = 30.0
    config.retention.max_clips_gb = 100.0

    retention = client.get("/api/config").json()["retention"]

    assert retention["evidence_tiered"] is True
    assert retention["evidence_days"]["shadow"] == 1.0
    # auto_routed inherits reviewed_days; false_positive falls back to the flat
    # clips window. Both are resolved here rather than left to be guessed.
    assert retention["evidence_days"]["auto_routed"] == 2.0
    assert retention["evidence_days"]["false_positive"] == config.retention.clips_days
    assert retention["max_clips_gb"] == 100.0
    assert retention["max_snapshots_gb"] is None       # 0 means no ceiling


def test_promoting_to_auto_routed_without_a_recipient_role_is_refused(client: TestClient):
    """An auto-routed policy with nobody to route to fires into the void."""
    client.post("/api/policies", json={"id": "p1", "kind": "detector",
                                       "detector_key": "manpower_headcount", "label": "P1"})
    response = client.post("/api/policies/p1/promote", json={})
    assert response.status_code == 422
    assert "routed_to" in response.text


def test_a_vlm_policy_needs_a_prompt(client: TestClient):
    response = client.post("/api/policies", json={
        "kind": "vlm", "detector_key": "vlm_prompt", "label": "Spills",
    })
    assert response.status_code == 422


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


def test_event_list_filters_and_pagination(client: TestClient, database: Database, fleet):
    now = time.time()
    ids = {
        "old_low": make_event(database, fleet, severity="low", ts=now - 7200),
        "shadow": make_event(database, fleet, stage="shadow", ts=now - 60),
        "recent": make_event(database, fleet, severity="high", ts=now),
    }

    everything = client.get("/api/events").json()
    assert everything["total"] == 3
    assert [e["id"] for e in everything["events"]] == sorted(ids.values(), reverse=True)
    assert everything["detectors"] == [fleet["policy"]]
    # meta comes back as an object, not a string of JSON.
    assert everything["events"][0]["meta"]["zone_name"] == "Packing"

    assert client.get("/api/events", params={"severity": "low"}).json()["total"] == 1
    assert client.get("/api/events", params={"stage": "shadow"}).json()["total"] == 1
    assert client.get("/api/events", params={"camera_id": "nope"}).json()["total"] == 0
    assert client.get("/api/events", params={"detector_type": "vlm"}).json()["total"] == 0

    windowed = client.get("/api/events", params={"since_ts": now - 120}).json()
    assert windowed["total"] == 2

    page = client.get("/api/events", params={"limit": 1, "offset": 1}).json()
    assert len(page["events"]) == 1
    # The total is the unpaginated count, or a paginator never finds the end.
    assert page["total"] == 3


def test_event_detail_carries_its_evidence_context(client: TestClient, database: Database, fleet):
    event_id = make_event(database, fleet, snapshot_path="line_a/z_1.jpg")
    payload = client.get(f"/api/events/{event_id}").json()
    assert payload["event"]["snapshot_path"] == "line_a/z_1.jpg"
    assert payload["camera"]["name"] == "Line A"
    assert payload["zone"]["name"] == "Packing"
    assert payload["review"] is None
    assert payload["notifications"] == []


def test_review_records_a_verdict_and_returns_fresh_accuracy(
    client: TestClient, database: Database, fleet
):
    first = make_event(database, fleet)
    second = make_event(database, fleet)

    confirmed = client.post(f"/api/events/{first}/review",
                            json={"verdict": "confirmed", "reviewer": "aq"})
    assert confirmed.status_code == 200
    body = confirmed.json()
    assert body["review"]["verdict"] == "confirmed"
    # The session names the reviewer. `reviewer: "aq"` in the body is accepted
    # for compatibility and ignored — a name the client chooses is not a person,
    # and this column is what answers "who cleared that alert" a year later.
    assert body["review"]["reviewer"] == "admin"
    assert body["accuracy"]["precision"] == 1.0

    false_positive = client.post(f"/api/events/{second}/review",
                                 json={"verdict": "false_positive", "note": "reflection"})
    assert false_positive.json()["accuracy"]["precision"] == 0.5
    assert false_positive.json()["accuracy"]["false_positive"] == 1

    # The verdict is visible on the list without a second request.
    listed = client.get("/api/events", params={"verdict": "false_positive"}).json()
    assert listed["total"] == 1
    assert listed["events"][0]["id"] == second


def test_re_reviewing_replaces_the_verdict(client: TestClient, database: Database, fleet):
    event_id = make_event(database, fleet)
    client.post(f"/api/events/{event_id}/review", json={"verdict": "confirmed"})
    client.post(f"/api/events/{event_id}/review", json={"verdict": "false_positive"})
    accuracy = client.get("/api/accuracy").json()
    assert accuracy["total_reviewed"] == 1
    assert accuracy["detectors"][0]["false_positive"] == 1


def test_shadow_events_are_not_reviewable(client: TestClient, database: Database, fleet):
    """Shadow output is the detector talking to itself. Letting it into the
    precision statistics would corrupt the number promotion depends on."""
    event_id = make_event(database, fleet, stage="shadow")
    response = client.post(f"/api/events/{event_id}/review", json={"verdict": "confirmed"})
    assert response.status_code == 409


def test_an_invalid_verdict_is_rejected(client: TestClient, database: Database, fleet):
    event_id = make_event(database, fleet)
    assert client.post(f"/api/events/{event_id}/review", json={"verdict": "maybe"}).status_code == 422


def test_review_of_a_missing_event_is_404(client: TestClient):
    assert client.post("/api/events/999/review", json={"verdict": "confirmed"}).status_code == 404


def test_box_correction_is_recorded_and_visible_on_the_event(
    client: TestClient, database: Database, fleet
):
    """A box correction judges one detection, not the whole alert — distinct
    from /review, and both can coexist on the same event."""
    event_id = make_event_with_boxes(
        database, fleet, [[0.1, 0.1, 0.2, 0.2, 3, 0.91], [0.5, 0.5, 0.6, 0.6, 7, 0.84]],
    )

    response = client.post(
        f"/api/events/{event_id}/boxes/3/correction",
        json={"track_id": 3, "verdict": "false_positive", "note": "reflection, not a person"},
    )
    assert response.status_code == 200
    corrections = response.json()["box_corrections"]
    assert len(corrections) == 1
    assert corrections[0]["track_id"] == 3
    assert corrections[0]["verdict"] == "false_positive"
    assert corrections[0]["reviewer"] == "admin"

    # It rides along on the event detail without a second request.
    detail = client.get(f"/api/events/{event_id}").json()
    assert len(detail["box_corrections"]) == 1
    assert detail["box_corrections"][0]["track_id"] == 3


def test_re_correcting_the_same_box_replaces_the_verdict(
    client: TestClient, database: Database, fleet
):
    event_id = make_event_with_boxes(database, fleet, [[0.1, 0.1, 0.2, 0.2, 3, 0.91]])
    client.post(f"/api/events/{event_id}/boxes/3/correction",
               json={"track_id": 3, "verdict": "false_positive"})
    client.post(f"/api/events/{event_id}/boxes/3/correction",
               json={"track_id": 3, "verdict": "true_positive"})
    corrections = client.get(f"/api/events/{event_id}").json()["box_corrections"]
    assert len(corrections) == 1
    assert corrections[0]["verdict"] == "true_positive"


def test_correcting_a_track_not_in_the_event_is_422(
    client: TestClient, database: Database, fleet
):
    event_id = make_event_with_boxes(database, fleet, [[0.1, 0.1, 0.2, 0.2, 3, 0.91]])
    response = client.post(f"/api/events/{event_id}/boxes/99/correction",
                           json={"track_id": 99, "verdict": "false_positive"})
    assert response.status_code == 422


def test_correction_path_and_body_track_id_must_agree(
    client: TestClient, database: Database, fleet
):
    event_id = make_event_with_boxes(database, fleet, [[0.1, 0.1, 0.2, 0.2, 3, 0.91]])
    response = client.post(f"/api/events/{event_id}/boxes/3/correction",
                           json={"track_id": 4, "verdict": "false_positive"})
    assert response.status_code == 422


def test_box_correction_of_a_missing_event_is_404(client: TestClient):
    response = client.post("/api/events/999/boxes/3/correction",
                           json={"track_id": 3, "verdict": "false_positive"})
    assert response.status_code == 404


def test_triage_queue_holds_only_unreviewed_non_shadow_events(
    client: TestClient, database: Database, fleet
):
    reviewed = make_event(database, fleet)
    pending = make_event(database, fleet)
    make_event(database, fleet, stage="shadow")
    client.post(f"/api/events/{reviewed}/review", json={"verdict": "confirmed"})

    queue = client.get("/api/triage").json()["events"]
    assert [event["id"] for event in queue] == [pending]


def test_triage_reports_the_whole_backlog_not_the_page(
    client: TestClient, database: Database, fleet
):
    """The nav badge read `events.length`, so it said 200 on a box with 7 682
    outstanding — measured 2026-08-25. `total` is the number an operator needs
    and `limit=1` is how the badge should ask for it."""
    ids = [make_event(database, fleet) for _ in range(5)]
    payload = client.get("/api/triage", params={"limit": 2}).json()
    assert len(payload["events"]) == 2
    assert payload["total"] == len(ids)


def test_triage_can_be_worked_from_the_newest_end(
    client: TestClient, database: Database, fleet
):
    """Oldest-first is right for draining a queue and wrong for reaching today:
    a page of 200 on a backlog of thousands never contains this shift."""
    ids = [make_event(database, fleet) for _ in range(4)]
    oldest = client.get("/api/triage", params={"limit": 2}).json()
    newest = client.get("/api/triage", params={"limit": 2, "order": "newest"}).json()
    assert [event["id"] for event in oldest["events"]] == ids[:2]
    assert [event["id"] for event in newest["events"]] == ids[::-1][:2]
    assert newest["order"] == "newest"
    assert oldest["total"] == newest["total"] == len(ids)


def test_triage_refuses_an_order_it_does_not_know(client: TestClient, fleet):
    assert client.get("/api/triage", params={"order": "sideways"}).status_code == 422


# --------------------------------------------------------------------------
# live stream
# --------------------------------------------------------------------------


# These run against a real uvicorn server rather than TestClient. Starlette's
# TestClient calls the app to completion and buffers the body (testclient.py:
# `portal.call(self.app, scope, receive, send)`), so an endless SSE response
# hangs it forever — and an endless response is precisely what is under test.


@pytest.fixture
def live_server(config: ArgusConfig, database: Database, make_user):
    import threading

    import uvicorn

    # The stream is behind the same session gate as everything else — an alert
    # is a description of who was where. `authed()` below is how these tests
    # hold one.
    make_user("ADMIN", "admin")
    config.api.sse_poll_sec = 0.05
    server = uvicorn.Server(uvicorn.Config(
        create_app(config, database), host="127.0.0.1", port=0, log_level="error",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    if not server.started:  # pragma: no cover - only when the port cannot bind
        pytest.fail("uvicorn did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@contextmanager
def authed(live_server: str):
    """An httpx client signed in against the live server.

    The SSE tests cannot use `client`: TestClient buffers a response to
    completion and an endless stream is the thing under test. So they carry
    their own client, and since Phase 5 it has to carry a cookie.
    """
    import httpx

    with httpx.Client(timeout=20, base_url=live_server) as http:
        response = http.post("/api/auth/login",
                             json={"username": "admin", "password": TEST_PASSWORD})
        assert response.status_code == 200, response.text
        yield http


def sse_frames(response):
    """Yield SSE frames (blank-line separated), skipping keepalive comments.

    One generator per response: httpx's ``iter_lines`` is a single-use
    iterator over the stream, so this must be created once and pulled from.
    A stalled stream surfaces as the client's read timeout, not a hang.
    """
    buffer = ""
    for line in response.iter_lines():
        buffer += line + "\n"
        if line == "":
            if buffer.strip() and not buffer.startswith(":"):
                yield buffer
            buffer = ""


def sse_data(frame: str) -> dict:
    return json.loads(frame.split("data: ", 1)[1])


def sse_frames_next(frames) -> dict:
    """The next frame, split into the three things a test asks about."""
    raw = next(frames)
    name = raw.split("event: ", 1)[1].split("\n", 1)[0]
    return {"raw": raw, "name": name, "data": sse_data(raw)}


def test_sse_stream_delivers_new_events(live_server: str, database: Database, fleet):
    """The route also has to *exist*: `/events/stream` is declared before
    `/events/{event_id}`, and if that order is ever lost this 422s."""
    import httpx

    with authed(live_server) as http:
        with http.stream("GET", f"{live_server}/api/events/stream") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers["x-accel-buffering"] == "no"
            frames = sse_frames(response)

            hello = next(frames)
            assert "event: hello" in hello
            assert sse_data(hello)["last_id"] == 0

            event_id = make_event(database, fleet)
            frame = next(frames)
            assert f"id: {event_id}" in frame
            assert "event: event" in frame
            payload = sse_data(frame)
            assert payload["id"] == event_id
            # meta arrives as an object here too, not a string of JSON.
            assert payload["meta"]["count"] == 5


def test_sse_resumes_from_a_cursor(live_server: str, database: Database, fleet):
    """A client that reconnects must not lose the events it missed. Browsers
    send Last-Event-ID automatically; an explicit ?since_id= does the same."""
    import httpx

    first = make_event(database, fleet)
    second = make_event(database, fleet)

    with authed(live_server) as http:
        with http.stream("GET", f"{live_server}/api/events/stream",
                         params={"since_id": first - 1}) as response:
            frames = sse_frames(response)
            next(frames)                                   # hello
            assert [sse_data(next(frames))["id"] for _ in range(2)] == [first, second]

        with http.stream("GET", f"{live_server}/api/events/stream",
                         headers={"Last-Event-ID": str(first)}) as response:
            frames = sse_frames(response)
            next(frames)
            assert sse_data(next(frames))["id"] == second


def test_sse_sends_the_clip_when_it_lands(live_server: str, database: Database, fleet):
    """An alert is streamed ~8 s before its clip exists, because the clip covers
    seconds that have not happened yet. A client reading only the stream would
    otherwise believe every live alert has no video, forever."""
    import httpx

    with authed(live_server) as http:
        with http.stream("GET", f"{live_server}/api/events/stream") as response:
            frames = sse_frames(response)
            next(frames)                                       # hello

            event_id = make_event(database, fleet)
            first = sse_frames_next(frames)
            assert first["name"] == "event"
            assert first["data"]["clip_path"] is None

            # What the engine's deferred cut does, once the footage exists.
            database.update_event(event_id, clip_path="cam/event_1.mp4", clip_start=100.0)

            update = sse_frames_next(frames)
            assert update["name"] == "update"
            assert update["data"]["id"] == event_id
            assert update["data"]["clip_path"] == "cam/event_1.mp4"
            # No `id:` field: an update must not drag a reconnecting client's
            # Last-Event-ID cursor backwards over events it has already seen.
            assert not update["raw"].startswith("id:")


def test_sse_stops_asking_about_a_clip_that_never_arrives(
    live_server: str, database: Database, fleet, monkeypatch
):
    """A clip can fail to cut — no ring footage, evidence disabled. The stream
    must give up on it rather than re-query that row for the life of the
    connection."""
    import httpx

    from argus.api.routers import events as events_router

    monkeypatch.setattr(events_router, "_CLIP_GRACE_SEC", 0.0)
    with authed(live_server) as http:
        with http.stream("GET", f"{live_server}/api/events/stream") as response:
            frames = sse_frames(response)
            next(frames)
            event_id = make_event(database, fleet)
            assert sse_frames_next(frames)["data"]["id"] == event_id

            database.update_event(event_id, clip_path="cam/event_1.mp4")
            # The id was dropped on the poll that streamed it, so the completion
            # is never sent; the next frame is the next *event*.
            second = make_event(database, fleet)
            following = sse_frames_next(frames)
            assert following["name"] == "event"
            assert following["data"]["id"] == second


# --------------------------------------------------------------------------
# derived views & health
# --------------------------------------------------------------------------


def test_occupancy_series(client: TestClient, database: Database, fleet):
    now = time.time()
    database.insert_occupancy([
        {"camera_id": "line_a", "zone_id": "line_a_pack", "ts": now - 30, "count": 2},
        {"camera_id": "line_a", "zone_id": "line_a_pack", "ts": now, "count": 5},
    ])
    series = client.get("/api/occupancy", params={"zone_id": "line_a_pack"}).json()["series"]
    assert [row["count"] for row in series] == [2, 5]


def test_policy_analytics_reports_the_false_positive_rate(
    client: TestClient, database: Database, fleet
):
    confirmed = make_event(database, fleet)
    rejected = make_event(database, fleet)
    client.post(f"/api/events/{confirmed}/review", json={"verdict": "confirmed"})
    client.post(f"/api/events/{rejected}/review", json={"verdict": "false_positive"})

    payload = client.get(f"/api/policies/{fleet['policy']}/analytics").json()
    assert payload["analytics"]["total"] == 2
    assert payload["analytics"]["false_positive_rate"] == 0.5


def test_health_reports_gpu_disk_and_database(client: TestClient, fleet):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"

    # The engine is not running in a test, and the report says so plainly
    # rather than failing.
    assert payload["engine"]["running"] is False

    assert payload["disk"]["total_gb"] > 0
    assert 0 <= payload["disk"]["percent"] <= 100
    assert set(payload["disk"]["by_directory_mb"]) >= {"clips", "snapshots", "ring"}

    # Evidence against its budget, which is the number that moves. A 3.7 TB
    # appliance read 36% full while data/clips grew 250 GB a day.
    evidence = payload["disk"]["evidence"]
    assert set(evidence["kinds"]) == {"clips", "snapshots"}
    assert evidence["kinds"]["clips"]["used_gb"] >= 0
    assert evidence["over_budget"] is False

    assert "available" in payload["gpu"]
    if payload["gpu"]["available"]:
        gpu = payload["gpu"]["gpus"][0]
        assert gpu["utilization_percent"] is not None
        # GB10 has no separate VRAM; memory then comes from the unified pool.
        assert gpu["memory_total_mb"] is not None
        assert payload["gpu"]["memory_source"] in ("device", "unified")

    assert payload["database"]["schema_version"] >= 1
    assert payload["database"]["cameras"] == 1


def test_health_reports_per_camera_fps_from_the_engine(
    client: TestClient, config: ArgusConfig, fleet
):
    """The engine and the API are separate processes; the stats file is the
    bridge. A fresh file means live numbers."""
    config.engine_stats_path.write_text(json.dumps({
        "started_at": time.time(), "uptime_sec": 120.0,
        "detector": {"model": "rf-detr-nano"},
        "retention": {"mb_deleted": 12.5},
        "cameras": [{"camera_id": "line_a", "fps": 4.97, "fps_avg": 5.0,
                     "frames": 600, "mode": "cuda", "restarts": 0}],
    }))
    payload = client.get("/api/health").json()
    assert payload["engine"]["running"] is True
    assert payload["engine"]["detector"]["model"] == "rf-detr-nano"
    assert payload["engine"]["retention"]["mb_deleted"] == 12.5
    camera = payload["cameras"][0]
    assert camera["id"] == "line_a"
    assert camera["fps"] == 4.97
    assert camera["decode_mode"] == "cuda"
    # The achieved rate is meaningless without the rate that was asked for,
    # and that answer lives in config, not in the engine's report.
    assert camera["target_fps"] == config.detection.fps


def test_health_calls_a_stale_stats_file_stopped(client: TestClient, config: ArgusConfig, fleet):
    import os

    config.engine_stats_path.write_text(json.dumps({"cameras": []}))
    stale = time.time() - 600
    os.utime(config.engine_stats_path, (stale, stale))
    payload = client.get("/api/health").json()
    assert payload["engine"]["running"] is False
    assert "no engine report" in payload["engine"]["reason"]


def test_health_vlm_section_keeps_one_shape_whichever_source_answers(
    client: TestClient, config: ArgusConfig, fleet
):
    """The console's VLM panel reads one set of keys. When the engine is up they
    are measured; when it is not they are null — never a fabricated zero, which
    would draw an idle queue on a box with no model at all."""
    config.vlm.base_url = "http://127.0.0.1:9/v1"        # discard port: nothing listens
    config.engine_stats_path.write_text(json.dumps({
        "started_at": time.time(), "uptime_sec": 120.0, "cameras": [],
        "vlm": {
            "enabled": True, "available": True, "model": "Qwen/Qwen3-VL-8B-Instruct",
            "endpoint": "http://127.0.0.1:8000/v1", "reason": None,
            "degraded_for_sec": None, "policies": 2, "evaluations": 31, "events": 4,
            "schedules": [{"policy_id": "phone_usage_on_line", "camera_id": "line_a",
                           "label": "Phone usage", "evaluations": 31, "events": 4,
                           "skipped_in_flight": 2, "degraded": False}],
            "verification": {"enabled": False, "stages": ["auto_routed"],
                             "checked": 0, "suppressed": 0, "failures": 0},
            "inference": {"requests": 31, "queue_depth": 1, "in_flight": 1,
                          "avg_latency_sec": 4.49, "last_inference_age_sec": 2.3},
        },
    }))
    live = client.get("/api/health").json()["vlm"]
    assert live["source"] == "engine"
    assert live["model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert live["inference"]["queue_depth"] == 1
    assert live["inference"]["last_inference_age_sec"] == 2.3
    assert live["schedules"][0]["skipped_in_flight"] == 2
    # Install facts come from disk on both paths, so a degraded panel can say
    # which half is missing without the engine's help.
    assert set(live["install"]) >= {"venv_ready", "model_ready", "model_path", "pid"}

    config.engine_stats_path.unlink()
    down = client.get("/api/health").json()["vlm"]
    assert down["source"] == "probe"
    assert down["available"] is False
    # Same keys, and the ones only the scheduler can count are unknown.
    assert set(down) >= set(live) - {"stopped"}
    assert down["inference"] is None
    assert down["schedules"] is None
    assert down["evaluations"] is None
    assert down["verification"]["checked"] is None
    # The configured id is still reported: "which model would this box serve"
    # is answerable with the server switched off.
    assert down["model"] == config.vlm.model
    assert "127.0.0.1:9" in down["reason"]


def test_health_says_a_disabled_vlm_is_disabled_not_broken(
    client: TestClient, config: ArgusConfig
):
    """An appliance sold without prompt policies must not read as a fault."""
    config.vlm.enabled = False
    config.vlm.base_url = "http://127.0.0.1:9/v1"
    payload = client.get("/api/health").json()["vlm"]
    assert payload["enabled"] is False
    assert "vlm.enabled is false" in payload["reason"]


def test_config_advertises_the_vlm_engine_from_the_appliance_config(
    client: TestClient, config: ArgusConfig
):
    """Phase 2 landed, so this flag is read from the config rather than pinned
    to False — and it answers "does this box run prompt policies", not "is the
    model up", which is what /api/health is for."""
    assert client.get("/api/config").json()["features"]["vlm"] is True
    assert client.get("/api/config").json()["features"]["vlm_verification"] is False

    config.vlm.enabled = False
    assert client.get("/api/config").json()["features"]["vlm"] is False


def test_config_endpoint_never_leaks_smtp_credentials(client: TestClient, config: ArgusConfig):
    """Recipients are the product (the settings screen shows who gets told);
    the account Argus sends *as* and its password are not."""
    config.smtp.username = "alerts@plant.example.com"
    config.smtp.recipients = {"safety_manager": ["safety@plant.example.com"]}
    payload = client.get("/api/config").json()
    body = client.get("/api/config").text
    assert payload["notify_roles"] == ["safety_manager"]
    assert payload["notify_recipients"] == {"safety_manager": ["safety@plant.example.com"]}
    assert "alerts@plant.example.com" not in body        # the SMTP login
    # Not a blanket "password" ban: `auth.min_password_length` is a UI hint and
    # says so. What must never appear is a credential or the name of the
    # variable holding one.
    assert config.smtp.password_env not in body
    assert "password_env" not in body
    assert "password_hash" not in body


def test_config_endpoint_describes_the_site_and_its_shifts(client: TestClient, config: ArgusConfig):
    config.site.name = "Veltra Stamping OH-1"
    payload = client.get("/api/config").json()
    assert payload["site"]["name"] == "Veltra Stamping OH-1"
    # Each shift knows where it ends, and the last one wraps midnight.
    assert payload["site"]["shifts"] == [
        {"id": "A", "start_hour": 6, "end_hour": 14},
        {"id": "B", "start_hour": 14, "end_hour": 22},
        {"id": "C", "start_hour": 22, "end_hour": 6},
    ]
    # Phase 5 landed, and there is no switch that turns it off.
    assert payload["features"]["auth"] is True
    assert payload["features"]["retraining_exports"] is False
    assert payload["features"]["channels"] == {"email": True, "whatsapp": False, "slack": False}
    # Phase 3 landed: the flag now tracks the config rather than a constant, and
    # says "this appliance does semantic search at all" — not "there is
    # something indexed right now", which /api/search/status answers.
    assert payload["features"]["search"] is True
    assert payload["index"]["model"]
    assert payload["retention"]["index_days"] > 0


# --------------------------------------------------------------------------
# live state (the wall)
# --------------------------------------------------------------------------


def write_engine_stats(config: ArgusConfig, cameras: list[dict]) -> None:
    config.engine_stats_path.write_text(json.dumps({
        "started_at": time.time(), "uptime_sec": 30.0,
        "gateway": {"enabled": True, "running": True, "url": "http://127.0.0.1:1984"},
        "cameras": cameras,
    }))


def test_live_state_joins_zone_geometry_to_engine_counts(
    client: TestClient, config: ArgusConfig, fleet
):
    write_engine_stats(config, [{
        "camera_id": "line_a", "fps": 4.98, "mode": "cuda", "lag_ms": 82.0,
        "occupancy": [{"zone_id": "line_a_pack", "zone_name": "Packing", "count": 5,
                       "limit": 3, "over_limit": True}],
        "tracks": [[0.1, 0.2, 0.2, 0.6, 7, 0.93]],
        "live_ts": time.time(),
    }])
    payload = client.get("/api/live/state").json()
    assert payload["engine"]["running"] is True
    camera = payload["cameras"][0]
    assert camera["stream"] == "line_a"          # the go2rtc stream name
    assert camera["fps"] == 4.98
    assert camera["people"] == 1
    zone = camera["zones"][0]
    # Geometry from the database, the count from the engine.
    assert zone["polygon"] == SQUARE
    assert zone["count"] == 5
    assert zone["over_limit"] is True
    assert payload["shift"]["id"] in {"A", "B", "C"}


def test_live_state_says_unknown_rather_than_zero_when_the_engine_is_down(
    client: TestClient, fleet
):
    """A wall that renders 0/3 for a camera nobody is watching is a wall that
    lies. With no engine, counts are null and the zone still draws."""
    payload = client.get("/api/live/state").json()
    assert payload["engine"]["running"] is False
    camera = payload["cameras"][0]
    assert camera["tracks"] is None
    assert camera["people"] is None
    assert camera["zones"][0]["count"] is None
    assert camera["zones"][0]["over_limit"] is False
    assert camera["zones"][0]["polygon"] == SQUARE


def test_live_state_lists_cameras_the_engine_never_picked_up(
    client: TestClient, config: ArgusConfig, database: Database, fleet
):
    database.upsert_camera(Camera(id="line_b", name="Line B", source="rtsp://cam/line-b"))
    write_engine_stats(config, [{"camera_id": "line_a", "fps": 5.0, "occupancy": [], "tracks": []}])
    cameras = {c["id"]: c for c in client.get("/api/live/state").json()["cameras"]}
    assert set(cameras) == {"line_a", "line_b"}
    assert cameras["line_b"]["fps"] is None
    assert cameras["line_b"]["zones"] == []


# --------------------------------------------------------------------------
# analytics
# --------------------------------------------------------------------------


def test_analytics_summary_counts_raw_and_verified_per_day(
    client: TestClient, database: Database, fleet
):
    now = time.time()
    confirmed_id = make_event(database, fleet, ts=now)
    make_event(database, fleet, ts=now)
    make_event(database, fleet, ts=now - 3 * 86400)
    database.record_review(confirmed_id, "confirmed", reviewer="tester")

    payload = client.get("/api/analytics/summary?days=7").json()
    assert payload["window"]["days"] == 7
    # Dense series: seven bars whether or not anything happened on a given day.
    assert len(payload["by_day"]) == 7
    today = payload["by_day"][-1]
    assert (today["raw"], today["confirmed"]) == (2, 1)
    assert sum(day["raw"] for day in payload["by_day"]) == 3

    totals = payload["totals"]
    assert totals["events"] == 3
    assert totals["confirmed"] == 1
    assert totals["reviewed"] == 1
    assert totals["precision"] == 1.0
    # Two unreviewed high-severity events past shadow are still open.
    assert totals["open_high"] == 2
    assert totals["median_response_sec"] is not None


def test_analytics_summary_groups_by_type_and_zone_shift(
    client: TestClient, database: Database, fleet
):
    from datetime import datetime, timedelta

    # 09:00 local is shift A, 15:00 is shift B under the default rota. Seeded
    # *yesterday* rather than today because the window ends at `now`: an event
    # stamped 15:00 is in the future for most of the working day, and a suite
    # that only passes after 3 p.m. is worse than no suite.
    yesterday = (datetime.now() - timedelta(days=1)).replace(minute=0, second=0, microsecond=0)
    make_event(database, fleet, ts=yesterday.replace(hour=9).timestamp())
    make_event(database, fleet, ts=yesterday.replace(hour=9).timestamp())
    make_event(database, fleet, ts=yesterday.replace(hour=15).timestamp())

    payload = client.get("/api/analytics/summary?days=2").json()
    by_type = payload["by_type"]
    assert len(by_type) == 1
    assert by_type[0]["key"] == "headcount_line_a"
    assert by_type[0]["label"] == "Line A headcount"   # the policy's own label
    assert by_type[0]["raw"] == 3

    row = payload["by_zone_shift"][0]
    assert row["zone_name"] == "Packing"
    assert row["raw"]["A"] == 2
    assert row["raw"]["B"] == 1
    assert row["total"] == 3


def test_analytics_summary_is_empty_but_well_formed_on_a_fresh_appliance(client: TestClient):
    payload = client.get("/api/analytics/summary?days=30").json()
    assert payload["totals"]["events"] == 0
    assert payload["totals"]["precision"] is None
    assert payload["by_type"] == []
    assert payload["by_zone_shift"] == []
    assert len(payload["by_day"]) == 30
    assert {day["raw"] for day in payload["by_day"]} == {0}


# --------------------------------------------------------------------------
# SPA
# --------------------------------------------------------------------------


def test_the_spa_is_served_at_the_root_with_a_history_fallback(
    config: ArgusConfig, database: Database, tmp_path: Path
):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Argus</title>")
    (dist / "assets" / "app.js").write_text("console.log('argus')")
    config.ui_dist = dist
    spa_client = TestClient(create_app(config, database))

    assert spa_client.get("/").text.startswith("<!doctype html>")
    # A deep link reloads into index.html; the SPA router takes it from there.
    assert spa_client.get("/alerts").text.startswith("<!doctype html>")
    assert spa_client.get("/assets/app.js").status_code == 200
    # A missing *asset* must still 404 — serving HTML as JavaScript is worse.
    assert spa_client.get("/assets/missing.js").status_code == 404
    # And the API keeps its routes.
    assert spa_client.get("/api/health").json()["status"] == "ok"
    assert spa_client.get("/api").json()["docs"] == "/api/docs"


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------


def test_snapshots_are_served_from_the_static_mount(client: TestClient, config: ArgusConfig):
    snapshot = config.snapshots_dir / "line_a" / "z_1.jpg"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    response = client.get("/snapshots/line_a/z_1.jpg")
    assert response.status_code == 200
    assert response.content.startswith(b"\xff\xd8")
    assert client.get("/snapshots/line_a/missing.jpg").status_code == 404


def test_root_points_at_the_docs(client: TestClient):
    body = client.get("/").json()
    assert body["docs"] == "/api/docs"
    assert client.get("/api/openapi.json").status_code == 200
