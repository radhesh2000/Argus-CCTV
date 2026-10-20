"""Packaging: systemd units, the installer scripts, backups, log rotation, doctor.

Everything here runs without systemd, without a GPU and without the network —
which is the point. A unit file is a pure function of config, a backup is two
SQLite files and a manifest, and rotation is a copy plus a truncate; none of
those need the thing that will eventually run them, so all of them can be
proven in CI on a machine that has none of it.

What is *not* covered here, said plainly so nobody reads green as more than it
is: that systemd accepts these units and brings the appliance up. That needs a
box with a user manager and is verified by running it — see docs/INSTALL.md.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from argus import backup, doctor, services
from argus.config import ROOT, ArgusConfig
from argus.db import SCHEMA_VERSION, Camera, Database
from argus.retention import Retention

SCRIPTS = ROOT / "scripts"


# --------------------------------------------------------------------------
# systemd units
# --------------------------------------------------------------------------


@pytest.fixture
def units(config: ArgusConfig) -> dict[str, services.Unit]:
    rendered = services.render(config, root=ROOT, command=["/opt/argus/.venv/bin/argus"])
    return {unit.name: unit for unit in rendered}


def test_renders_every_unit(units):
    assert set(units) == set(services.UNIT_NAMES)
    for unit in units.values():
        assert unit.text.startswith(services.GENERATED_MARKER)
        assert unit.text.endswith("\n")


def test_engine_unit_runs_argus_run_and_restarts(units):
    text = units[services.ENGINE_UNIT].text
    assert "ExecStart=/opt/argus/.venv/bin/argus run" in text
    assert "Restart=always" in text
    assert "StandardOutput=journal" in text
    assert "SyslogIdentifier=argus-engine" in text
    # The engine stops ffmpeg children and the gateway on SIGTERM; a stop
    # timeout shorter than that turns a clean shutdown into a SIGKILL.
    assert "TimeoutStopSec=45" in text
    assert f"WantedBy={services.TARGET_UNIT}" in text


def test_api_is_ordered_after_the_engine_but_does_not_pull_it_in(units):
    text = units[services.API_UNIT].text
    assert f"After={services.ENGINE_UNIT}" in text
    # No Wants/Requires: the console is designed to start without the engine
    # and say so. Ordering exists only so the engine wins the race to migrate.
    assert "Wants=" not in text
    assert "Requires=" not in text


def test_vlm_unit_is_written_but_not_enabled_by_default(units):
    unit = units[services.VLM_UNIT]
    assert unit.enable_by_default is False
    assert "Type=forking" in unit.text
    assert "PIDFile=" in unit.text
    assert "ExecStart=/opt/argus/.venv/bin/argus vlm start" in unit.text
    assert "ExecStop=/opt/argus/.venv/bin/argus vlm stop" in unit.text
    # Model load is minutes; the default 90 s start timeout would kill it.
    assert "TimeoutStartSec=900" in unit.text
    assert "StartLimitBurst=3" in unit.text


def test_units_carry_this_appliance_config_and_offline_environment(config, units):
    config.source_path = Path("/etc/argus/argus.yaml")
    rendered = {u.name: u.text for u in
                services.render(config, root=ROOT, command=["/x/argus"])}
    for name in (services.ENGINE_UNIT, services.API_UNIT, services.VLM_UNIT):
        assert "Environment=ARGUS_CONFIG=/etc/argus/argus.yaml" in rendered[name]
        # The "fully local at runtime" claim, enforced at the process boundary.
        assert "Environment=HF_HUB_OFFLINE=1" in rendered[name]
        assert f"Environment=RF_HOME={config.models_dir}" in rendered[name]


def test_target_is_wanted_by_default_target(units):
    assert "WantedBy=default.target" in units[services.TARGET_UNIT].text


def test_argus_command_prefers_the_checkout_venv(tmp_path):
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    executable = venv_bin / "argus"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    assert services.argus_command(tmp_path) == [str(executable)]


def test_unit_dir_follows_xdg(tmp_path):
    env = {"XDG_CONFIG_HOME": str(tmp_path / "conf"), "HOME": str(tmp_path)}
    assert services.unit_dir(env) == tmp_path / "conf" / "systemd" / "user"
    assert services.unit_dir({"HOME": str(tmp_path)}) == \
        tmp_path / ".config" / "systemd" / "user"


class FakeSystemctl:
    """Records argv instead of running it. `systemctl --user` in CI is a lie
    either way — better a lie we can assert on."""

    def __init__(self, returncode: int = 0):
        self.calls: list[list[str]] = []
        self.returncode = returncode

    def __call__(self, argv):
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(list(argv), self.returncode, "Linger=yes\n", "")


def test_install_writes_enables_and_is_idempotent(config, tmp_path, monkeypatch):
    monkeypatch.setattr(services, "systemctl_available", lambda: True)
    runner = FakeSystemctl()
    target = tmp_path / "units"

    first = services.install(config, directory=target, runner=runner,
                             command=["/x/argus"])
    assert {u["action"] for u in first["units"]} == {"written"}
    for name in services.UNIT_NAMES:
        assert (target / name).is_file()

    assert runner.calls[0] == ["systemctl", "--user", "daemon-reload"]
    enable = next(c for c in runner.calls if "enable" in c)
    assert services.VLM_UNIT not in enable          # optional equipment
    assert services.ENGINE_UNIT in enable and services.API_UNIT in enable

    second = services.install(config, directory=target, runner=FakeSystemctl(),
                              command=["/x/argus"])
    assert {u["action"] for u in second["units"]} == {"unchanged"}


def test_install_with_vlm_enables_it(config, tmp_path, monkeypatch):
    monkeypatch.setattr(services, "systemctl_available", lambda: True)
    runner = FakeSystemctl()
    services.install(config, directory=tmp_path, runner=runner, with_vlm=True,
                     command=["/x/argus"])
    enable = next(c for c in runner.calls if "enable" in c)
    assert services.VLM_UNIT in enable


def test_install_refuses_to_clobber_a_hand_written_unit(config, tmp_path, monkeypatch):
    monkeypatch.setattr(services, "systemctl_available", lambda: True)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / services.ENGINE_UNIT).write_text("[Service]\nExecStart=/usr/bin/true\n")
    with pytest.raises(services.ServiceError, match="not written by argus"):
        services.install(config, directory=tmp_path, runner=FakeSystemctl(),
                         command=["/x/argus"])


def test_dry_run_writes_nothing(config, tmp_path, monkeypatch):
    monkeypatch.setattr(services, "systemctl_available", lambda: True)
    runner = FakeSystemctl()
    target = tmp_path / "units"
    result = services.install(config, directory=target, runner=runner,
                              dry_run=True, command=["/x/argus"])
    assert not target.exists()
    assert not [call for call in runner.calls if call[0] == "systemctl"]
    assert result["commands"]


def test_uninstall_removes_only_generated_units(config, tmp_path, monkeypatch):
    monkeypatch.setattr(services, "systemctl_available", lambda: True)
    services.install(config, directory=tmp_path, runner=FakeSystemctl(),
                     command=["/x/argus"])
    mine = tmp_path / services.API_UNIT
    mine.write_text("[Service]\nExecStart=/usr/bin/true\n")  # somebody edited it

    result = services.uninstall(directory=tmp_path, runner=FakeSystemctl())
    assert str(mine) in result["kept"]
    assert mine.is_file()
    assert not (tmp_path / services.ENGINE_UNIT).exists()


def test_states_reports_a_missing_unit_as_not_found(monkeypatch):
    monkeypatch.setattr(services, "systemctl_available", lambda: True)

    def runner(argv):
        return subprocess.CompletedProcess(
            list(argv), 0,
            "LoadState=not-found\nActiveState=inactive\nSubState=dead\n"
            "UnitFileState=\nNRestarts=0\n", "")

    state = services.states([services.ENGINE_UNIT], runner=runner)
    assert state[services.ENGINE_UNIT]["load_state"] == "not-found"


# --------------------------------------------------------------------------
# backups
# --------------------------------------------------------------------------


def _camera(database: Database, camera_id: str) -> None:
    database.upsert_camera(Camera(id=camera_id, name=camera_id, source="file:///x.mp4",
                                  kind="file"))


def _index_store(config: ArgusConfig) -> None:
    """A stand-in vector store. The real one needs sqlite-vec; a backup copies
    pages and does not care, which is exactly the property worth pinning."""
    config.index_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.index_db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS frames (id INTEGER PRIMARY KEY, path TEXT)")
    conn.execute("INSERT INTO frames (path) VALUES ('a.jpg')")
    conn.commit()
    conn.close()


def test_backup_copies_both_databases_with_a_manifest(config, database):
    _camera(database, "cam-a")
    _index_store(config)

    entry = backup.run(config, reason="test")

    assert entry.path.is_dir()
    assert {f.name for f in entry.files} == {"argus.sqlite3", "index.sqlite3"}
    manifest = json.loads((entry.path / backup.MANIFEST_NAME).read_text())
    assert manifest["reason"] == "test"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert backup.verify(entry) == []


def test_backup_of_a_live_wal_database_is_checkpointed(config, database):
    """The engine is mid-write while this runs. The copy must be a consistent,
    fully checkpointed snapshot — a `cp` here would be the bug."""
    _camera(database, "cam-a")
    live = database.connect()                       # an open writer, as in production
    try:
        live.execute(
            "INSERT INTO cameras (id, name, source, kind, enabled, created_at, updated_at) "
            "VALUES ('cam-b', 'b', 'file:///b.mp4', 'file', 1, '2026-01-01', '2026-01-01')"
        )
        live.commit()
        assert Path(str(config.db_path) + "-wal").exists(), "expected a live WAL"

        entry = backup.run(config, reason="test")
        copied = entry.path / "argus.sqlite3"
        assert not Path(str(copied) + "-wal").exists()

        read = sqlite3.connect(f"file:{copied}?mode=ro", uri=True)
        try:
            assert read.execute("SELECT COUNT(*) FROM cameras").fetchone()[0] == 2
        finally:
            read.close()
    finally:
        live.close()


def test_index_is_optional(config, database):
    _index_store(config)
    config.backup.include_index = False
    entry = backup.run(config)
    assert {f.name for f in entry.files} == {"argus.sqlite3"}


def test_an_interrupted_backup_is_not_a_backup(config, database, monkeypatch):
    original = backup._copy_database
    calls = {"n": 0}

    def explode(source, destination):
        calls["n"] += 1
        original(source, destination)
        raise OSError("disk went away")

    monkeypatch.setattr(backup, "_copy_database", explode)
    with pytest.raises(OSError):
        backup.run(config)

    assert calls["n"] == 1
    # Nothing was committed: the staging directory is gone and nothing lists.
    assert backup.list_backups(config) == []
    assert not any(p.name.startswith(".incoming-")
                   for p in config.backups_dir.iterdir())


def test_prune_keeps_the_newest_and_drops_incomplete_first(config, database):
    stamps = []
    for hour in range(4):
        entry = backup.run(config, when=_at(hour))
        stamps.append(entry.stamp)
    half_done = config.backups_dir / "20260101T000000Z"
    half_done.mkdir()
    (half_done / "argus.sqlite3").write_bytes(b"partial")

    config.backup.keep = 2
    removed = backup.prune(config)

    kept = [b.stamp for b in backup.list_backups(config)]
    assert kept == stamps[-1:-3:-1]                 # the two newest, newest first
    assert "20260101T000000Z" in removed


def _at(hour: int):
    from datetime import datetime, timezone

    return datetime(2026, 5, 1, hour, 0, 0, tzinfo=timezone.utc)


def test_due_is_age_based_not_wall_clock(config, database):
    assert backup.due(config) is True             # nothing yet: owed immediately
    entry = backup.run(config, when=_at(3))
    moment = backup.parse_stamp(entry.stamp).timestamp()

    assert backup.due(config, now=moment + 3600) is False
    assert backup.due(config, now=moment + 25 * 3600) is True

    config.backup.enabled = False
    assert backup.due(config, now=moment + 25 * 3600) is False


def test_restore_puts_the_database_back_and_keeps_the_bad_one(config, database):
    _camera(database, "cam-a")
    _index_store(config)
    entry = backup.run(config)

    _camera(database, "cam-b")                     # the mistake
    assert len(database.list_cameras()) == 2

    result = backup.restore(config, entry.stamp)

    assert [c["id"] for c in database.list_cameras()] == ["cam-a"]
    assert result["schema_version"] == SCHEMA_VERSION
    # Nothing was deleted — the database being replaced is still on disk.
    aside = list(config.db_path.parent.glob("argus.sqlite3.pre-restore-*"))
    assert aside, "the replaced database must be kept, not deleted"


def test_restore_moves_the_wal_aside_too(config, database):
    """A leftover -wal replayed over a restored file is not a failed restore,
    it is a database that never existed. A killed engine leaves exactly that."""
    _camera(database, "cam-a")
    entry = backup.run(config)

    killed = database.connect()                     # the engine, about to be killed
    try:
        killed.execute("DELETE FROM cameras")
        killed.commit()
        wal = Path(str(config.db_path) + "-wal")
        assert wal.stat().st_size > 0

        result = backup.restore(config, entry.stamp, force=True)
    finally:
        killed.close()

    assert any(name.endswith(f"-wal{result['pre_restore_suffix']}")
               for item in result["restored"] for name in item["kept_aside"])
    assert not Path(str(config.db_path) + "-wal").exists()
    assert [c["id"] for c in Database(config.db_path).list_cameras()] == ["cam-a"]


def test_restore_refuses_a_damaged_backup(config, database):
    _camera(database, "cam-a")
    entry = backup.run(config)
    (entry.path / "argus.sqlite3").write_bytes(b"not a database at all")

    with pytest.raises(backup.BackupError, match="damaged"):
        backup.restore(config, entry.stamp)
    # And the live database is untouched.
    assert len(database.list_cameras()) == 1


def test_restore_refuses_while_the_engine_is_running(config, database, monkeypatch):
    _camera(database, "cam-a")
    entry = backup.run(config)
    monkeypatch.setattr(backup, "read_engine_stats", lambda _cfg: {"running": True})

    with pytest.raises(backup.BackupError, match="engine is running"):
        backup.restore(config, entry.stamp)

    result = backup.restore(config, entry.stamp, force=True)
    assert result["stamp"] == entry.stamp


def test_restore_accepts_latest_and_a_prefix(config, database):
    first = backup.run(config, when=_at(1))
    second = backup.run(config, when=_at(2))
    assert backup.find(config, "latest").stamp == second.stamp
    assert backup.find(config, first.stamp[:11]).stamp == first.stamp
    with pytest.raises(backup.BackupError, match="matches 2"):
        backup.find(config, "2026")


def test_two_backups_in_one_second_do_not_collide_or_look_overdue(config, database):
    first = backup.run(config, when=_at(4))
    second = backup.run(config, when=_at(4))
    assert first.stamp != second.stamp and second.stamp.endswith("-1")
    # The suffix must not defeat the age check, or the box backs up in a loop.
    moment = backup.parse_stamp(second.stamp).timestamp()
    assert backup.due(config, now=moment + 3600) is False


def test_a_failed_restore_leaves_the_live_database_in_place(config, database,
                                                            monkeypatch):
    _camera(database, "cam-a")
    entry = backup.run(config)
    _camera(database, "cam-b")

    def explode(source, destination, **_kw):
        raise OSError("no space left on device")

    monkeypatch.setattr(backup.shutil, "copy2", explode)
    with pytest.raises(OSError):
        backup.restore(config, entry.stamp)

    # Neither moved aside nor replaced: the appliance is exactly as it was.
    assert config.db_path.is_file()
    assert {c["id"] for c in Database(config.db_path).list_cameras()} == {"cam-a", "cam-b"}
    assert not list(config.db_path.parent.glob("*.pre-restore-*"))
    assert not list(config.db_path.parent.glob(".restoring-*"))


def test_backup_refuses_a_full_disk(config, database, monkeypatch):
    monkeypatch.setattr(backup, "disk_usage",
                        lambda _p: {"percent": 97.0, "free_gb": 100.0})
    with pytest.raises(backup.BackupError, match="max_disk_percent"):
        backup.run(config)


def test_backup_survives_an_older_schema(config, database):
    """A backup taken before a migration restores and is then migrated."""
    entry = backup.run(config)
    manifest_path = entry.path / backup.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest))

    result = backup.restore(config, entry.stamp)
    assert result["from_schema"] == 1
    assert result["schema_version"] == SCHEMA_VERSION


# --------------------------------------------------------------------------
# log rotation
# --------------------------------------------------------------------------


def test_rotation_keeps_the_inode_so_an_open_writer_survives(config):
    """The whole reason this is copy-and-truncate rather than rename."""
    log = config.logs_dir / "go2rtc.log"
    config.logs.max_mb = 0.001
    handle = log.open("ab")                        # the supervisor's fd
    try:
        handle.write(b"x" * 4096)
        handle.flush()
        before = log.stat().st_ino

        rotated = backup.rotate_logs(config)

        assert rotated and rotated[0]["rotated_to"] == "go2rtc.log.1"
        assert log.stat().st_ino == before, "rename would break the open writer"
        assert log.stat().st_size == 0
        assert (config.logs_dir / "go2rtc.log.1").stat().st_size == 4096

        handle.write(b"after\n")                   # still writing to the live file
        handle.flush()
        assert log.read_bytes().endswith(b"after\n")
    finally:
        handle.close()


def test_rotation_leaves_small_logs_alone(config):
    log = config.logs_dir / "vlm-server.log"
    log.write_bytes(b"tiny")
    assert backup.rotate_logs(config) == []
    assert log.read_bytes() == b"tiny"


def test_rotation_bounds_the_generations(config):
    log = config.logs_dir / "go2rtc.log"
    config.logs.max_mb = 0.001
    config.logs.keep = 2
    for _ in range(4):
        log.write_bytes(b"y" * 4096)
        backup.rotate_logs(config)
    assert sorted(p.name for p in config.logs_dir.iterdir()) == [
        "go2rtc.log", "go2rtc.log.1", "go2rtc.log.2",
    ]


def test_rotation_can_be_turned_off(config):
    log = config.logs_dir / "go2rtc.log"
    log.write_bytes(b"z" * 4096)
    config.logs.max_mb = 0.001
    config.logs.rotate = False
    assert backup.rotate_logs(config) == []
    assert log.stat().st_size == 4096


# --------------------------------------------------------------------------
# the housekeeping thread that carries all three
# --------------------------------------------------------------------------


def test_housekeeping_sweeps_rotates_and_backs_up_in_one_tick(config, database):
    log = config.logs_dir / "go2rtc.log"
    log.write_bytes(b"q" * 4096)
    config.logs.max_mb = 0.001
    lines: list[str] = []

    Retention(config, database, log=lines.append).housekeeping()

    assert any(line.startswith("retention:") for line in lines)
    assert any("rotated go2rtc.log" in line for line in lines)
    assert any(line.startswith("backup:") for line in lines)
    assert len(backup.list_backups(config)) == 1


def test_a_failing_backup_does_not_stop_the_sweep(config, database, monkeypatch):
    """Reclaiming disk keeps the appliance up; a backup is insurance. The
    insurance must never be able to take the appliance down."""
    monkeypatch.setattr(backup, "run_if_due",
                        lambda _cfg, **_kw: (_ for _ in ()).throw(
                            backup.BackupError("destination is read-only")))
    lines: list[str] = []

    result = Retention(config, database, log=lines.append).housekeeping()

    assert result is not None
    assert any("backup: skipped — destination is read-only" in line for line in lines)
    assert any(line.startswith("retention:") for line in lines)


def test_disabling_retention_does_not_disable_backups(config, database):
    """Two policies, two switches. A site that manages storage elsewhere has
    not asked for its database to stop being backed up."""
    config.retention.enabled = False
    lines: list[str] = []

    Retention(config, database, log=lines.append).housekeeping()

    assert not any(line.startswith("retention: reclaimed") for line in lines)
    assert len(backup.list_backups(config)) == 1


def test_backups_are_not_a_retention_target(config, database):
    """The age sweep must never delete a backup: the backup job keeps its own
    generations, and a sweep that aged them out on the evidence schedule would
    delete the record along with the thing it records."""
    entry = backup.run(config)
    os.utime(entry.path / "argus.sqlite3", (0, 0))          # ancient
    os.utime(entry.path / backup.MANIFEST_NAME, (0, 0))

    Retention(config, database).sweep()

    assert backup.list_backups(config)[0].stamp == entry.stamp


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def test_doctor_runs_every_check_without_crashing(config, database):
    checks, elapsed = doctor.report(config, database)
    assert checks and elapsed >= 0
    groups = {c.group for c in checks}
    assert {"host", "gpu", "binaries", "models", "database", "storage",
            "backup", "logs", "services", "ports", "engine"} <= groups
    # Every failing or warning check must carry the command that fixes it —
    # this is a support tool, and a finding without a fix is a phone call.
    for check in checks:
        if check.status in (doctor.FAIL, doctor.WARN) and check.group != "doctor":
            assert check.fix, f"{check.group}/{check.name} has no fix"


def test_doctor_fails_on_a_fresh_appliance_with_no_accounts(config, database):
    checks = doctor.check_accounts(config, database)
    assert checks[0].status == doctor.FAIL
    assert doctor.exit_code(checks) == 1


def test_doctor_fails_when_the_detector_weights_are_missing(config, database, tmp_path):
    # Point the model trees at empty directories: this test is about what the
    # doctor says when weights are absent, not about what happens to be
    # installed on the box running the suite.
    config.vlm.server.venv = str(tmp_path / "no-venv")
    config.vlm.server.models_dir = str(tmp_path / "no-models")

    checks = {c.name: c for c in doctor.check_models(config)}
    assert checks["detector"].status == doctor.FAIL
    assert "fetch_detector.sh" in checks["detector"].fix
    # The VLM is optional equipment: missing is a warning, never a failure.
    assert checks["vlm"].status == doctor.WARN


def test_doctor_reports_a_schema_from_the_future_as_a_failure(config, database,
                                                              monkeypatch):
    monkeypatch.setattr(doctor, "SCHEMA_VERSION", SCHEMA_VERSION - 1)
    schema = next(c for c in doctor.check_database(config, database) if c.name == "schema")
    assert schema.status == doctor.FAIL
    assert "newer" in schema.detail


def test_doctor_fails_when_a_backup_is_missing_then_passes(config, database):
    assert any(c.status == doctor.FAIL for c in doctor.check_backups(config))
    backup.run(config)
    latest = next(c for c in doctor.check_backups(config) if c.name == "latest")
    assert latest.status == doctor.OK
    # Same-filesystem backups are always worth a warning.
    assert any(c.name == "destination" and c.status == doctor.WARN
               for c in doctor.check_backups(config))


def test_doctor_separates_a_stopped_service_from_a_dead_one(config, tmp_path,
                                                             monkeypatch):
    """`NRestarts` remembers forever; `Result` is what says whether the last
    exit was deliberate. Reporting a deliberately stopped unit as broken is how
    a support tool trains people to ignore it."""
    units = tmp_path / "units"
    units.mkdir()
    for name in (services.ENGINE_UNIT, services.API_UNIT):
        (units / name).write_text(services.GENERATED_MARKER + "\n")
    monkeypatch.setattr(services, "unit_dir", lambda *_a, **_k: units)
    monkeypatch.setattr(services, "linger_enabled", lambda *_a, **_k: True)
    monkeypatch.setattr(services, "states", lambda names, **_kw: {
        services.ENGINE_UNIT: {"active_state": "inactive", "sub_state": "dead",
                               "unit_file_state": "enabled", "result": "success",
                               "restarts": "3"},
        services.API_UNIT: {"active_state": "failed", "sub_state": "failed",
                            "unit_file_state": "enabled", "result": "exit-code",
                            "restarts": "6"},
    })

    checks = {c.name: c for c in doctor.check_services(config)}
    assert checks[services.ENGINE_UNIT].status == doctor.WARN
    assert "not running" in checks[services.ENGINE_UNIT].detail
    assert checks[services.API_UNIT].status == doctor.FAIL
    assert "exit-code" in checks[services.API_UNIT].detail


# --- memory, and who else is using it --------------------------------------
#
# The numbers in these tests are the ones measured on the appliance on
# 2026-09-19, when a second user's llama-server had left the VLM 13 GiB of the
# 36.5 it needs and the doctor said `[ok] vlm` for twenty days.

_LLAMA = {"pid": 225484, "mib": 63098.0, "owner": "rahul", "command": "llama-server",
          "foreign": True}
_ENGINE = {"pid": 3428647, "mib": 1444.0, "owner": "arjunvs", "command": "argus",
           "foreign": False}


def test_doctor_names_whoever_is_holding_the_memory_the_vlm_needs(config):
    config.vlm.enabled = True
    config.vlm.server.gpu_memory_utilization = 0.30
    (memory,) = doctor.check_memory(config, tenants=[_ENGINE, _LLAMA],
                                    memory=(13.3, 121.69), vlm_listening=False)
    assert memory.status == doctor.FAIL
    assert "36.5 GiB" in memory.detail and "13.3 GiB" in memory.detail
    assert "llama-server" in memory.detail and "rahul" in memory.detail
    # The appliance's own engine is a tenant too, and is not the news.
    assert "argus" not in memory.detail.split("held by")[1]
    # It must not suggest killing a colleague's process: this account cannot.
    assert "Ask rahul" in memory.fix and "reset-failed" in memory.fix


def test_doctor_does_not_fail_memory_for_a_vlm_that_is_already_up(config):
    """A running VLM has its memory; 13 GiB free beside it is a healthy box."""
    config.vlm.enabled = True
    (memory,) = doctor.check_memory(config, tenants=[_ENGINE], memory=(13.3, 121.69),
                                    vlm_listening=True)
    assert memory.status == doctor.OK


def test_doctor_warns_about_a_large_foreign_tenant_even_when_everything_fits(config):
    config.vlm.enabled = False
    (memory,) = doctor.check_memory(config, tenants=[_ENGINE, _LLAMA], memory=(40.0, 121.69),
                                    vlm_listening=False)
    assert memory.status == doctor.WARN
    assert "61.6 GiB" in memory.detail


def test_a_crashed_vlm_the_site_depends_on_is_a_failure_not_a_warning(config, tmp_path,
                                                                     monkeypatch):
    """`disabled` is how the unit file always reads — argus.target starts it —
    and that used to demote a dead server to a WARN."""
    units = tmp_path / "units"
    units.mkdir()
    (units / services.VLM_UNIT).write_text(services.GENERATED_MARKER + "\n")
    monkeypatch.setattr(services, "unit_dir", lambda *_a, **_k: units)
    monkeypatch.setattr(services, "linger_enabled", lambda *_a, **_k: True)
    monkeypatch.setattr(services, "states", lambda names, **_kw: {
        services.VLM_UNIT: {"active_state": "failed", "sub_state": "failed",
                            "unit_file_state": "disabled", "result": "exit-code",
                            "restarts": "3"},
    })
    config.vlm.enabled = True
    checks = {c.name: c for c in doctor.check_services(config)}
    assert checks[services.VLM_UNIT].status == doctor.FAIL

    config.vlm.enabled = False          # a site that never asked for it
    checks = {c.name: c for c in doctor.check_services(config)}
    assert checks[services.VLM_UNIT].status != doctor.FAIL


def test_doctor_says_what_a_dead_vlm_costs_in_policies(config, database):
    from argus.db import Policy

    config.vlm.enabled = True
    database.upsert_policy(Policy(id="floor_waste", kind="vlm", detector_key="vlm_prompt",
                                  label="Floor waste", prompt="Is there waste on the floor?"))
    database.upsert_policy(Policy(id="switched_off", kind="vlm", detector_key="vlm_prompt",
                                  label="Off", prompt="?", enabled=False))
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    (config.logs_dir / "vlm-server.log").write_text(
        "(EngineCore pid=1) ValueError: Free memory on device cuda:0 (13.3/121.69 GiB) on "
        "startup is less than desired GPU memory utilization (0.3, 36.51 GiB).\n")

    (dark,) = doctor.check_vlm_policies(config, database, vlm_listening=False)
    assert dark.status == doctor.FAIL
    assert "1 enabled prompt policy is not being evaluated" in dark.detail
    assert "floor_waste" in dark.detail and "switched_off" not in dark.detail
    assert "Free memory on device" in dark.detail     # vLLM's own reason, quoted

    assert doctor.check_vlm_policies(config, database, vlm_listening=True) == []


def test_doctor_treats_linger_as_a_failure(config, tmp_path, monkeypatch):
    """An appliance that stops when its operator logs out is not an appliance."""
    units = tmp_path / "units"
    units.mkdir()
    (units / services.ENGINE_UNIT).write_text(services.GENERATED_MARKER + "\n")
    monkeypatch.setattr(services, "unit_dir", lambda *_a, **_k: units)
    monkeypatch.setattr(services, "states", lambda names, **_kw: {})
    monkeypatch.setattr(services, "linger_enabled", lambda *_a, **_k: False)

    linger = next(c for c in doctor.check_services(config) if c.name == "linger")
    assert linger.status == doctor.FAIL
    assert "enable-linger" in linger.fix


def test_doctor_flags_a_gateway_api_exposed_to_the_network(config, monkeypatch):
    """The Phase-5 hole, as a check: config says loopback, the socket decides."""
    monkeypatch.setattr(doctor, "local_address", lambda: "10.0.0.5")
    monkeypatch.setattr(doctor, "_listening",
                        lambda host, port: port == config.gateway.api_port)

    checks = {c.name: c for c in doctor.check_ports(config)}
    assert checks["gateway api"].status == doctor.FAIL
    assert "gateway.api_host" in checks["gateway api"].fix


def test_doctor_exit_code_separates_warnings_from_failures():
    warn = [doctor.Check("g", "n", doctor.WARN, "d", fix="f")]
    assert doctor.exit_code(warn) == 0
    assert doctor.exit_code(warn, strict=True) == 1
    assert doctor.exit_code(warn + [doctor.Check("g", "m", doctor.FAIL, "d")]) == 1


def test_doctor_json_is_machine_readable(config, database, capsys):
    from argus.cli import main

    code = main(["-c", str(_write_config(config)), "doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(code, int)
    assert payload["summary"]["fail"] >= 1          # no accounts on a fresh box
    assert all({"group", "name", "status", "detail"} <= set(c)
               for c in payload["checks"])


def test_backup_cli_round_trips_through_main(config, database, capsys):
    """The commands an operator actually types, including the ones that must
    fail politely rather than with a traceback."""
    from argus.cli import main

    path = _write_config(config)
    Database(config.db_path).migrate()

    assert main(["-c", str(path), "backup", "verify"]) == 1        # nothing yet
    assert "no complete backups" in capsys.readouterr().err

    assert main(["-c", str(path), "backup", "now", "--reason", "drill"]) == 0
    assert main(["-c", str(path), "backup", "list"]) == 0
    assert "argus.sqlite3" in capsys.readouterr().out
    assert main(["-c", str(path), "backup", "verify"]) == 0
    assert "sha256 verified" in capsys.readouterr().out
    assert main(["-c", str(path), "backup", "restore", "latest"]) == 0
    assert "nothing was deleted" in capsys.readouterr().out

    assert main(["-c", str(path), "backup", "restore", "20990101T000000Z"]) == 1
    assert "no backup" in capsys.readouterr().err


def _write_config(config: ArgusConfig) -> Path:
    """A minimal argus.yaml pointing at this test's data dir."""
    path = config.data_dir.parent / "argus.yaml"
    path.write_text(
        f"data_dir: {config.data_dir}\n"
        f"ui_dist: {config.data_dir / 'no-ui'}\n"
        "index:\n  enabled: false\n"
        "vlm:\n  enabled: false\n"
    )
    return path


# --------------------------------------------------------------------------
# the shell scripts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("*.sh")))
def test_every_script_is_syntactically_valid_bash(script):
    result = subprocess.run(["bash", "-n", str(SCRIPTS / script)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("*.sh")))
def test_every_script_is_executable_and_strict(script):
    path = SCRIPTS / script
    assert os.access(path, os.X_OK), f"{script} is not executable"
    # An installer step that fails silently is worse than one that stops.
    assert "set -euo pipefail" in path.read_text()


def test_installer_inventory_runs_offline(tmp_path):
    """`--list` must work on any box: it is the first thing support asks for."""
    result = subprocess.run(["bash", str(SCRIPTS / "install.sh"), "--list"],
                            capture_output=True, text=True, check=False,
                            cwd=str(ROOT), timeout=60)
    assert result.returncode == 0, result.stderr
    for expected in ("python environment", "go2rtc", "detector weights",
                     "built console", "systemd units"):
        assert expected in result.stdout


def test_installer_offline_refuses_to_fetch(tmp_path):
    """--offline on a checkout with nothing staged must fail loudly at the
    first missing artefact rather than quietly reaching for the network."""
    fake_root = tmp_path / "checkout"
    (fake_root / "scripts").mkdir(parents=True)
    (fake_root / "config").mkdir()
    (fake_root / "ui").mkdir()
    for name in ("install.sh",):
        (fake_root / "scripts" / name).write_text((SCRIPTS / name).read_text())
        (fake_root / "scripts" / name).chmod(0o755)
    (fake_root / "config" / "argus.yaml").write_text("data_dir: data\n")

    result = subprocess.run(["bash", str(fake_root / "scripts" / "install.sh"),
                             "--offline"],
                            capture_output=True, text=True, check=False,
                            cwd=str(fake_root), timeout=120)
    assert result.returncode == 1
    assert "no .venv" in (result.stdout + result.stderr)


def test_installer_help_does_not_need_the_appliance():
    result = subprocess.run(["bash", str(SCRIPTS / "install.sh"), "--help"],
                            capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0
    assert "--offline" in result.stdout


def test_python_scripts_compile():
    for script in sorted(SCRIPTS.glob("*.py")):
        result = subprocess.run([sys.executable, "-m", "py_compile", str(script)],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"{script.name}: {result.stderr}"


# --------------------------------------------------------------------------
# doctor: the fleet the engine is not actually running
# --------------------------------------------------------------------------
#
# Found by running the appliance rather than by reading it. A camera enrolled
# from the console probes OPENED, is stored enabled, and is then decoded by
# nothing until `argus run` restarts — the engine reads the camera table at
# startup only. Every surface was quiet about it: the console's row said
# `offline` (so does a dead camera), and doctor said "2/2 cameras live",
# because two was all the engine had ever been told about.


def _engine_stats(config: ArgusConfig, camera_ids: list[str]) -> None:
    """Publish an engine stats file naming exactly these cameras as live."""
    config.engine_stats_path.parent.mkdir(parents=True, exist_ok=True)
    config.engine_stats_path.write_text(json.dumps({
        "started_at": 0.0, "uptime_sec": 120.0, "stopped": False,
        "cameras": [{"camera_id": cid, "status": "live", "fps": 5.0}
                    for cid in camera_ids],
    }))


def test_doctor_names_a_camera_that_is_enrolled_but_not_being_decoded(config, database):
    database.upsert_camera(Camera(id="cam_a", name="A", source="/tmp/a.mp4"))
    database.upsert_camera(Camera(id="cam_b", name="B", source="/tmp/b.mp4"))
    # The engine only ever knew about cam_a — cam_b was enrolled after it started.
    _engine_stats(config, ["cam_a"])

    checks = {c.name: c for c in doctor.check_engine(config, database)}
    assert checks["running"].status == doctor.OK
    fleet = checks["fleet"]
    assert fleet.status == doctor.WARN
    assert "cam_b" in fleet.detail and "cam_a" not in fleet.detail
    assert fleet.data["missing"] == ["cam_b"]
    # A support tool that finds something must say what to do about it.
    assert "restart" in fleet.fix.lower()


def test_doctor_is_quiet_when_the_engine_runs_the_whole_fleet(config, database):
    database.upsert_camera(Camera(id="cam_a", name="A", source="/tmp/a.mp4"))
    database.upsert_camera(Camera(id="cam_b", name="B", source="/tmp/b.mp4"))
    _engine_stats(config, ["cam_a", "cam_b"])

    names = {c.name for c in doctor.check_engine(config, database)}
    assert names == {"running"}


def test_doctor_ignores_a_disabled_camera_the_engine_never_started(config, database):
    """A camera an operator switched off is not a fault to report."""
    database.upsert_camera(Camera(id="cam_a", name="A", source="/tmp/a.mp4"))
    database.upsert_camera(Camera(id="cam_off", name="Off", source="/tmp/off.mp4",
                                  enabled=False))
    _engine_stats(config, ["cam_a"])

    names = {c.name for c in doctor.check_engine(config, database)}
    assert names == {"running"}


def test_doctor_does_not_claim_a_missing_fleet_while_the_engine_is_down(config, database):
    """With no engine there is nothing to be out of step with, and the fix is
    'start the engine' rather than 'restart it'."""
    database.upsert_camera(Camera(id="cam_a", name="A", source="/tmp/a.mp4"))
    checks = doctor.check_engine(config, database)
    assert [c.name for c in checks] == ["running"]
    assert checks[0].status == doctor.WARN
