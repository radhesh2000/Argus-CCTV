"""Alert delivery tests.

``smtplib.SMTP`` is replaced with a recorder, so ``Notifier._connect`` itself
still runs — STARTTLS, login and the implicit-TLS branch are all exercised.
Everything below that is a socket, and a socket is not what these tests are
about.
"""

from __future__ import annotations

import smtplib
import time

import pytest

from argus.config import ArgusConfig
from argus.db import Camera, Database, Event
from argus.notify import Notifier, build_message, format_alert


class FakeSMTP:
    """Records what a real server would have been asked to do."""

    instances: list["FakeSMTP"] = []
    fail_on: set[str] = set()          # recipients whose send raises
    connect_error: Exception | None = None

    def __init__(self, host, port, timeout=None):
        if type(self).connect_error is not None:
            raise type(self).connect_error
        self.host, self.port, self.timeout = host, port, timeout
        self.started_tls = False
        self.login_args: tuple | None = None
        self.sent: list = []
        self.quit_called = False
        self.implicit_tls = False
        type(self).instances.append(self)

    def starttls(self):
        self.started_tls = True

    def login(self, username, password):
        self.login_args = (username, password)

    def send_message(self, message):
        recipient = message["To"]
        if recipient in type(self).fail_on:
            raise smtplib.SMTPRecipientsRefused({recipient: (550, b"no such user")})
        self.sent.append(message)

    def quit(self):
        self.quit_called = True


class FakeSMTPSSL(FakeSMTP):
    def __init__(self, host, port, timeout=None):
        super().__init__(host, port, timeout)
        self.implicit_tls = True


@pytest.fixture(autouse=True)
def fake_smtp(monkeypatch: pytest.MonkeyPatch):
    FakeSMTP.instances = []
    FakeSMTP.fail_on = set()
    FakeSMTP.connect_error = None
    monkeypatch.setattr("argus.notify.smtplib.SMTP", FakeSMTP)
    monkeypatch.setattr("argus.notify.smtplib.SMTP_SSL", FakeSMTPSSL)
    return FakeSMTP


@pytest.fixture
def wired(config: ArgusConfig, database: Database) -> ArgusConfig:
    config.smtp.host = "smtp.plant.local"
    config.smtp.username = "alerts@plant.example.com"
    config.smtp.from_address = "Argus <alerts@plant.example.com>"
    config.smtp.recipients = {
        "safety_manager": ["safety@plant.example.com", "hod@plant.example.com"],
        "line_supervisor": ["supervisor@plant.example.com"],
    }
    database.upsert_camera(Camera(id="line_a", name="Packing Line A", source="rtsp://cam/a"))
    return config


def make_event(database: Database, *, stage: str = "auto_routed",
               routed_to: str | None = "safety_manager", **overrides) -> int:
    return database.insert_event(Event(
        camera_id="line_a", detector_name="headcount_line_a", detector_type="detector",
        severity="high", description="Zone 'Packing' over sanctioned limit: 5 people (limit 3)",
        ts=time.time(), stage=stage, confidence=0.91, routed_to=routed_to,
        meta={"count": 5, "limit": 3, "zone_name": "Packing"}, **overrides,
    ))


# --------------------------------------------------------------------------
# the auto_routed gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["shadow", "reviewed"])
def test_only_auto_routed_events_are_emailed(wired: ArgusConfig, database: Database, stage: str):
    """Shadow events are being validated in silence; reviewed events belong to
    a human in the triage queue. Mailing either is the alert fatigue the staged
    rollout exists to prevent."""
    event_id = make_event(database, stage=stage)
    result = Notifier(wired, database, log=lambda _m: None).deliver(database.get_event(event_id))

    assert result.sent == []
    assert result.skipped is not None
    assert FakeSMTP.instances == []
    # Not an attempt, so not an attempt-shaped row either.
    assert database.notifications_for_event(event_id) == []


def test_an_auto_routed_event_reaches_every_recipient_of_its_role(
    wired: ArgusConfig, database: Database
):
    event_id = make_event(database)
    result = Notifier(wired, database, log=lambda _m: None).deliver(database.get_event(event_id))

    assert result.sent == ["safety@plant.example.com", "hod@plant.example.com"]
    assert result.failed == []

    # One connection, two messages: a slow SMTP handshake is paid once.
    assert len(FakeSMTP.instances) == 1
    server = FakeSMTP.instances[0]
    assert (server.host, server.port) == ("smtp.plant.local", 587)
    assert server.started_tls
    assert server.login_args == ("alerts@plant.example.com", "")
    assert server.quit_called
    assert len(server.sent) == 2

    rows = database.notifications_for_event(event_id)
    assert [row["status"] for row in rows] == ["sent", "sent"]
    assert {row["target"] for row in rows} == set(result.sent)
    assert {row["channel"] for row in rows} == {"smtp"}


def test_a_role_with_no_recipients_is_logged_as_a_failure(
    wired: ArgusConfig, database: Database
):
    """A promoted policy pointing at a role nobody configured is a
    misconfiguration a human has to see, not a silent no-op."""
    event_id = make_event(database, routed_to="maintenance_head")
    result = Notifier(wired, database, log=lambda _m: None).deliver(database.get_event(event_id))

    assert result.sent == []
    assert "no recipients configured" in result.skipped
    rows = database.notifications_for_event(event_id)
    assert len(rows) == 1
    assert (rows[0]["status"], rows[0]["channel"]) == ("failed", "none")
    assert "maintenance_head" in rows[0]["detail"]


def test_an_auto_routed_event_with_no_role_is_logged_as_a_failure(
    wired: ArgusConfig, database: Database
):
    event_id = make_event(database, routed_to=None)
    result = Notifier(wired, database, log=lambda _m: None).deliver(database.get_event(event_id))
    assert "no routed_to" in result.skipped
    assert database.notifications_for_event(event_id)[0]["status"] == "failed"


# --------------------------------------------------------------------------
# failure paths
# --------------------------------------------------------------------------


def test_a_dead_smtp_server_records_a_failure_per_recipient(
    wired: ArgusConfig, database: Database
):
    FakeSMTP.connect_error = OSError("connection refused")
    event_id = make_event(database)
    result = Notifier(wired, database, log=lambda _m: None).deliver(database.get_event(event_id))

    assert result.sent == []
    assert len(result.failed) == 2
    rows = database.notifications_for_event(event_id)
    assert [row["status"] for row in rows] == ["failed", "failed"]
    assert "connection refused" in rows[0]["detail"]


def test_one_bad_address_does_not_sink_the_batch(wired: ArgusConfig, database: Database):
    FakeSMTP.fail_on = {"hod@plant.example.com"}
    event_id = make_event(database)
    result = Notifier(wired, database, log=lambda _m: None).deliver(database.get_event(event_id))

    assert result.sent == ["safety@plant.example.com"]
    assert result.failed == ["hod@plant.example.com"]
    rows = {row["target"]: row["status"] for row in database.notifications_for_event(event_id)}
    assert rows == {"safety@plant.example.com": "sent", "hod@plant.example.com": "failed"}


def test_port_465_uses_implicit_tls_and_never_calls_starttls(
    wired: ArgusConfig, database: Database
):
    wired.smtp.port = 465
    event_id = make_event(database)
    Notifier(wired, database, log=lambda _m: None).deliver(database.get_event(event_id))
    server = FakeSMTP.instances[0]
    assert server.implicit_tls
    assert not server.started_tls


def test_starttls_can_be_turned_off(wired: ArgusConfig, database: Database):
    wired.smtp.starttls = False
    event_id = make_event(database)
    Notifier(wired, database, log=lambda _m: None).deliver(database.get_event(event_id))
    assert not FakeSMTP.instances[0].started_tls


def test_an_unauthenticated_relay_skips_login(wired: ArgusConfig, database: Database):
    wired.smtp.username = ""
    event_id = make_event(database)
    Notifier(wired, database, log=lambda _m: None).deliver(database.get_event(event_id))
    assert FakeSMTP.instances[0].login_args is None


# --------------------------------------------------------------------------
# the message itself
# --------------------------------------------------------------------------


def test_the_message_says_what_where_and_how_sure(wired: ArgusConfig, database: Database):
    event_id = make_event(database)
    Notifier(wired, database, log=lambda _m: None).deliver(database.get_event(event_id))
    message = FakeSMTP.instances[0].sent[0]

    assert message["Subject"] == "[Argus/HIGH] headcount line a — Packing Line A"
    assert message["From"] == "Argus <alerts@plant.example.com>"
    body = message.get_body(preferencelist=("plain",)).get_content()
    assert "over sanctioned limit" in body
    assert "Packing Line A / Packing" in body     # camera name and zone name
    assert "Confidence: 0.91" in body
    assert "safety_manager" in body
    assert f"Event id:   {event_id}" in body


def test_the_snapshot_rides_along(wired: ArgusConfig, database: Database):
    """An alert without its evidence gets argued with."""
    snapshot = wired.snapshots_dir / "line_a" / "zone_1.jpg"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 512)

    event_id = make_event(database, snapshot_path="line_a/zone_1.jpg")
    Notifier(wired, database, log=lambda _m: None).deliver(database.get_event(event_id))

    attachments = list(FakeSMTP.instances[0].sent[0].iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_filename() == "zone_1.jpg"
    assert attachments[0].get_content_type() == "image/jpeg"


def test_a_missing_snapshot_never_stops_the_alert(wired: ArgusConfig, database: Database):
    """Retention may have reaped the file, or the disk may be full. The text of
    the message is the part that matters."""
    event_id = make_event(database, snapshot_path="line_a/gone.jpg")
    result = Notifier(wired, database, log=lambda _m: None).deliver(database.get_event(event_id))
    assert len(result.sent) == 2
    assert list(FakeSMTP.instances[0].sent[0].iter_attachments()) == []


def test_an_oversized_snapshot_is_left_off(wired: ArgusConfig, database: Database,
                                           monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("argus.notify.MAX_ATTACHMENT_BYTES", 100)
    snapshot = wired.snapshots_dir / "big.jpg"
    snapshot.write_bytes(b"0" * 5000)
    event_id = make_event(database, snapshot_path="big.jpg")
    Notifier(wired, database, log=lambda _m: None).deliver(database.get_event(event_id))
    assert list(FakeSMTP.instances[0].sent[0].iter_attachments()) == []


def test_a_deep_link_appears_only_when_the_appliance_has_a_reachable_url(
    wired: ArgusConfig, database: Database
):
    event_id = make_event(database)
    notifier = Notifier(wired, database, log=lambda _m: None)
    notifier.deliver(database.get_event(event_id))
    assert "Evidence:" not in FakeSMTP.instances[0].sent[0].get_content()

    wired.api.public_url = "http://argus.plant.local:8080/"
    FakeSMTP.instances = []
    notifier.deliver(database.get_event(event_id))
    body = FakeSMTP.instances[0].sent[0].get_content()
    assert f"http://argus.plant.local:8080/api/events/{event_id}" in body


def test_format_alert_survives_meta_arriving_as_json_text():
    """``get_event`` hands back the raw column; the formatter must cope."""
    subject, body = format_alert(
        {"id": 1, "severity": "low", "detector_name": "spill_check", "camera_id": "cam1",
         "description": "spill", "created_at": "2026-08-09T10:00:00+00:00", "confidence": 0.5,
         "meta": '{"zone_name": "Bay 3"}'},
        camera_name="Bay Camera",
    )
    assert subject == "[Argus/LOW] spill check — Bay Camera"
    assert "Bay Camera / Bay 3" in body


def test_build_message_tolerates_broken_meta():
    message = build_message(
        {"id": 1, "severity": "low", "detector_name": "d", "camera_id": "c",
         "description": "x", "created_at": "", "confidence": 0, "meta": "not json"},
        "someone@example.com", from_address="argus@localhost",
    )
    assert message["To"] == "someone@example.com"


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def test_dispatch_does_not_spawn_a_thread_for_a_shadow_event(
    wired: ArgusConfig, database: Database
):
    """Most events are shadow. A thread per shadow event is pure waste on the
    vision loop's hot path."""
    event_id = make_event(database, stage="shadow")
    assert Notifier(wired, database, log=lambda _m: None).dispatch(event_id) is None


def test_dispatch_delivers_off_the_calling_thread(wired: ArgusConfig, database: Database):
    event_id = make_event(database)
    thread = Notifier(wired, database, log=lambda _m: None).dispatch(event_id)
    assert thread is not None
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert len(database.notifications_for_event(event_id)) == 2


def test_dispatch_of_a_missing_event_is_a_no_op(wired: ArgusConfig, database: Database):
    assert Notifier(wired, database, log=lambda _m: None).dispatch(9999) is None
