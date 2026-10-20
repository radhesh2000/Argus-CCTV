"""The appliance pause switch: the durable flag, its boot gate, the endpoint.

Everything here runs without systemd and without a GPU. The one thing a test on
a machine with no user manager *cannot* prove — that systemd accepts the
``ExecCondition`` and skips a paused unit — is asserted at the seam instead: the
rendered unit carries the gate, and the gate (``argus processing gate``) returns
the exit code that makes systemd skip. The full stop+start round trip is
verified by running it on the box (docs/INSTALL.md, and the deploy that shipped
this).
"""

from __future__ import annotations

import json
import subprocess

import pytest
from fastapi.testclient import TestClient

from argus import processing, services
from argus.api.app import create_app
from argus.cli import main
from argus.config import ArgusConfig
from argus.db import Database


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def config_file(config: ArgusConfig, tmp_path):
    """A YAML config pointing at the same data dir the ``config`` fixture uses,
    so the CLI gate reads the very flag the test wrote."""
    path = tmp_path / "argus.yaml"
    path.write_text(f"data_dir: {config.data_dir}\n")
    return path


def _client(config: ArgusConfig, database: Database) -> TestClient:
    return TestClient(create_app(config, database))


class _Recorder:
    """A services runner that records argv and reports success, so actuation can
    be asserted without a systemctl on the box."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, "", "")


@pytest.fixture
def has_systemctl(monkeypatch):
    """Make ``systemctl_available()`` true regardless of the box, by patching the
    ``shutil.which`` that backs it — one function object, shared by services and
    processing, so both see it."""
    monkeypatch.setattr(
        services.shutil, "which",
        lambda name: "/usr/bin/systemctl" if name == "systemctl" else None,
    )


@pytest.fixture
def no_systemctl(monkeypatch):
    """The other side: no user manager. The flag still persists; nothing is
    actuated. Used by the API tests so a POST never touches a real unit."""
    monkeypatch.setattr(services.shutil, "which", lambda name: None)


# --------------------------------------------------------------------------
# the durable flag
# --------------------------------------------------------------------------


def test_flag_defaults_to_enabled_when_the_file_is_absent(config: ArgusConfig):
    # A fresh, never-configured appliance runs — the switch is opt-in.
    assert not processing.state_path(config).exists()
    assert processing.read_state(config).enabled is True
    assert processing.is_enabled(config) is True


def test_write_then_read_round_trips_and_records_who(config: ArgusConfig):
    state = processing.write_state(config, False, actor="ops", note="quiet the box")
    assert state.enabled is False
    again = processing.read_state(config)
    assert again.enabled is False
    assert again.updated_by == "ops"
    assert again.note == "quiet the box"
    assert again.updated_at  # timestamped, so the audit line has a when


def test_write_is_atomic_and_leaves_valid_json_with_no_temp(config: ArgusConfig):
    processing.write_state(config, False)
    path = processing.state_path(config)
    assert path.is_file()
    # The tmp the atomic write used is gone; a reader never sees a half-file.
    assert not path.with_suffix(".json.tmp").exists()
    payload = json.loads(path.read_text())
    assert payload[processing.FLAG_KEY] is False


def test_a_corrupt_flag_reads_as_enabled_not_paused(config: ArgusConfig):
    # The bias is toward watching the plant: a mangled byte must not silently
    # stop a safety appliance. Pausing leaves a valid document (atomic write).
    processing.state_path(config).write_text("{ this is not json")
    assert processing.is_enabled(config) is True


def test_the_flag_lives_on_disk_and_survives_a_fresh_process(config: ArgusConfig):
    # "Survives reboot" in miniature: the desired state is a file, not in-memory,
    # so a brand-new config object built after the write still reads it.
    processing.write_state(config, False, actor="ops")
    reloaded = ArgusConfig(data_dir=config.data_dir)
    assert processing.read_state(reloaded).enabled is False


# --------------------------------------------------------------------------
# the boot gate — a rebooted-but-paused appliance comes up paused
# --------------------------------------------------------------------------


def test_the_engine_and_vlm_units_carry_the_pause_gate(config: ArgusConfig):
    units = {u.name: u.text for u in services.render(config, command=["/opt/argus/.venv/bin/argus"])}
    gate = "ExecCondition=/opt/argus/.venv/bin/argus processing gate"
    engine = units[services.ENGINE_UNIT]
    assert gate in engine
    assert gate in units[services.VLM_UNIT]
    # The condition must precede the start it guards.
    assert engine.index("ExecCondition") < engine.index("ExecStart=/opt/argus/.venv/bin/argus run")
    # The control plane is never gated — the console stays up while paused.
    assert "processing gate" not in units[services.API_UNIT]
    assert "processing gate" not in units[services.TARGET_UNIT]


def test_the_gate_returns_skip_when_paused_and_proceed_when_enabled(
    config: ArgusConfig, config_file
):
    # This is the reboot contract: systemd runs `argus processing gate` as the
    # unit's ExecCondition; exit 1 skips the start (paused stays paused), exit 0
    # proceeds. Written flag -> gate exit code, exactly as the boot path reads it.
    processing.write_state(config, False)
    assert main(["-c", str(config_file), "processing", "gate"]) == 1

    processing.write_state(config, True)
    assert main(["-c", str(config_file), "processing", "gate"]) == 0


def test_the_gate_proceeds_when_the_config_is_unreadable(tmp_path):
    # A broken config must not masquerade as a pause: the gate proceeds (exit 0)
    # and lets the real unit fail loudly instead of being silently skipped.
    assert main(["-c", str(tmp_path / "nope.yaml"), "processing", "gate"]) == 0


# --------------------------------------------------------------------------
# actuation — start/stop the right units, never block, always persist
# --------------------------------------------------------------------------


def test_resume_starts_engine_plus_vlm_non_blocking(config: ArgusConfig, monkeypatch, has_systemctl):
    monkeypatch.setattr(processing, "vlm_should_run", lambda cfg: True)
    rec = _Recorder()
    result = processing.set_processing(config, True, actor="ops", runner=rec)

    assert result["actuated"] is True
    assert processing.read_state(config).enabled is True
    assert rec.calls == [[
        "systemctl", "--user", "start", "--no-block",
        services.ENGINE_UNIT, services.VLM_UNIT,
    ]]


def test_resume_without_an_installed_vlm_starts_only_the_engine(
    config: ArgusConfig, monkeypatch, has_systemctl
):
    monkeypatch.setattr(processing, "vlm_should_run", lambda cfg: False)
    rec = _Recorder()
    processing.set_processing(config, True, runner=rec)
    assert rec.calls == [["systemctl", "--user", "start", "--no-block", services.ENGINE_UNIT]]


def test_pause_stops_both_units(config: ArgusConfig, has_systemctl):
    rec = _Recorder()
    result = processing.set_processing(config, False, actor="ops", runner=rec)
    assert result["actuated"] is True
    assert processing.read_state(config).enabled is False
    assert rec.calls == [[
        "systemctl", "--user", "stop", "--no-block",
        services.ENGINE_UNIT, services.VLM_UNIT,
    ]]


def test_the_flag_persists_even_with_no_user_systemd(config: ArgusConfig, no_systemctl):
    # The desired state must survive even where nothing here can enforce it.
    result = processing.set_processing(config, False, actor="ops")
    assert result["actuated"] is False
    assert "systemctl" in (result["detail"] or "")
    assert processing.read_state(config).enabled is False


def test_control_builds_the_expected_systemctl_line(has_systemctl):
    rec = _Recorder()
    services.control("start", [services.ENGINE_UNIT], runner=rec)
    assert rec.calls == [["systemctl", "--user", "start", "--no-block", services.ENGINE_UNIT]]
    services.control("stop", [services.ENGINE_UNIT], no_block=False, runner=rec)
    assert rec.calls[-1] == ["systemctl", "--user", "stop", services.ENGINE_UNIT]


def test_control_refuses_an_action_it_does_not_know(has_systemctl):
    with pytest.raises(services.ServiceError):
        services.control("frobnicate", [services.ENGINE_UNIT], runner=_Recorder())


# --------------------------------------------------------------------------
# the derived status the console renders
# --------------------------------------------------------------------------


def test_view_reads_paused_when_the_flag_is_off(config: ArgusConfig):
    processing.write_state(config, False)
    view = processing.processing_view(
        config, engine_running=False, engine_uptime_sec=None, vlm_available=False,
    )
    assert view["status"] == "paused"
    assert view[processing.FLAG_KEY] is False


def test_view_reads_running_when_no_vlm_is_required(config: ArgusConfig, monkeypatch):
    monkeypatch.setattr(processing, "vlm_should_run", lambda cfg: False)
    processing.write_state(config, True)
    view = processing.processing_view(
        config, engine_running=True, engine_uptime_sec=30, vlm_available=False,
    )
    assert view["status"] == "running"
    assert view["vlm"] == "disabled"


def test_view_reads_starting_while_the_vlm_warms_then_running(config: ArgusConfig, monkeypatch):
    monkeypatch.setattr(processing, "vlm_should_run", lambda cfg: True)
    processing.write_state(config, True)
    warming = processing.processing_view(
        config, engine_running=True, engine_uptime_sec=5, vlm_available=False,
    )
    assert warming["status"] == "starting"
    assert warming["vlm"] == "starting"

    ready = processing.processing_view(
        config, engine_running=True, engine_uptime_sec=30, vlm_available=True,
    )
    assert ready["status"] == "running"
    assert ready["vlm"] == "ready"


def test_a_silent_vlm_past_its_grace_is_degraded_not_warming(config: ArgusConfig, monkeypatch):
    # The banner must stop crying "warming up" once the model has had its whole
    # cold-start budget: past the grace, a silent VLM is a health problem shown
    # on Model Health, and the box reads "running", not "starting", forever.
    monkeypatch.setattr(processing, "vlm_should_run", lambda cfg: True)
    monkeypatch.setattr(processing, "warmup_grace_sec", lambda cfg: 100.0)
    processing.write_state(config, True)
    view = processing.processing_view(
        config, engine_running=True, engine_uptime_sec=500, vlm_available=False,
    )
    assert view["status"] == "running"
    assert view["vlm"] == "down"


# --------------------------------------------------------------------------
# the API — health reflects state, and the write is ADMIN-only and audited
# --------------------------------------------------------------------------


def test_health_reflects_the_paused_flag(config: ArgusConfig, database: Database, sign_in):
    processing.write_state(config, False, actor="ops")
    client = _client(config, database)
    sign_in(client, "ADMIN")
    payload = client.get("/api/health").json()
    assert payload["processing"][processing.FLAG_KEY] is False
    assert payload["processing"]["status"] == "paused"


def test_live_state_carries_processing(config: ArgusConfig, database: Database, sign_in):
    processing.write_state(config, True)
    client = _client(config, database)
    sign_in(client, "REVIEW")
    block = client.get("/api/live/state").json()["processing"]
    assert block[processing.FLAG_KEY] is True
    assert "status" in block


def test_get_processing_is_readable_by_every_role(
    config: ArgusConfig, database: Database, sign_in, no_systemctl
):
    for role in ("ADMIN", "RESPOND", "REVIEW", "SYSTEM"):
        client = _client(config, database)
        sign_in(client, role, username=role.lower())
        response = client.get("/api/system/processing")
        assert response.status_code == 200, (role, response.text)
        assert processing.FLAG_KEY in response.json()


def test_post_processing_is_admin_only(
    config: ArgusConfig, database: Database, sign_in, no_systemctl
):
    # A plant-wide switch belongs to the role that also owns accounts.
    for role, allowed in (("REVIEW", False), ("RESPOND", False),
                          ("SYSTEM", False), ("ADMIN", True)):
        client = _client(config, database)
        sign_in(client, role, username=role.lower())
        response = client.post("/api/system/processing", json={"enabled": False})
        if allowed:
            assert response.status_code == 200, response.text
        else:
            assert response.status_code == 403, (role, response.status_code)


def test_post_processing_flips_the_flag_and_health_follows(
    config: ArgusConfig, database: Database, sign_in, no_systemctl
):
    processing.write_state(config, True)
    client = _client(config, database)
    sign_in(client, "ADMIN")  # username 'admin'

    response = client.post("/api/system/processing", json={"enabled": False, "note": "overnight"})
    assert response.status_code == 200
    body = response.json()
    assert body[processing.FLAG_KEY] is False
    # systemctl patched away, so it persisted but did not actuate — and said so.
    assert body["actuated"] is False

    stored = processing.read_state(config)
    assert stored.enabled is False
    assert stored.updated_by == "admin"      # the audited actor is the caller
    assert stored.note == "overnight"

    # And the health surface the console polls now reads paused.
    assert client.get("/api/health").json()["processing"]["status"] == "paused"


def test_post_processing_requires_a_boolean_enabled(
    config: ArgusConfig, database: Database, sign_in, no_systemctl
):
    client = _client(config, database)
    sign_in(client, "ADMIN")
    # Missing field: a switch with no position must not be read as either.
    assert client.post("/api/system/processing", json={"note": "x"}).status_code == 422
    # A misspelled field is refused loudly (extra='forbid').
    assert client.post(
        "/api/system/processing", json={"enabled": True, "enabledd": True}
    ).status_code == 422


def test_an_unauthenticated_caller_cannot_read_or_write_processing(
    config: ArgusConfig, database: Database
):
    client = _client(config, database)
    assert client.get("/api/system/processing").status_code == 401
    assert client.post("/api/system/processing", json={"enabled": False}).status_code == 401


# --------------------------------------------------------------------------
# the doctor reads a paused box calmly, not as a fault
# --------------------------------------------------------------------------


def test_doctor_calls_a_paused_engine_ok_and_a_stopped_one_a_warning(
    config: ArgusConfig, database: Database
):
    from argus import doctor

    # Paused: the engine is *meant* to be down, so it is OK — not a WARN an
    # operator learns to ignore, which would also hide a real crash.
    processing.write_state(config, False)
    engine = next(c for c in doctor.check_engine(config, database) if c.name == "running")
    assert engine.status == doctor.OK
    assert "paused" in engine.detail.lower()

    # Enabled but not running is the ordinary "you have not started it" warning.
    processing.write_state(config, True)
    engine = next(c for c in doctor.check_engine(config, database) if c.name == "running")
    assert engine.status == doctor.WARN
