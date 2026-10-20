from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path

import pytest

from argus.cli import build_parser, main
from argus.db import SCHEMA_VERSION, Database


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """A config pointing at a throwaway data dir, so the CLI never touches the
    real database."""
    path = tmp_path / "argus.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            data_dir: {tmp_path / "data"}
            gateway:
              api_port: 11985
              rtsp_port: 18556
              webrtc_port: 18557
              listen_host: 127.0.0.1
            """
        )
    )
    return path


def test_parser_exposes_the_documented_subcommands():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert set(actions[0].choices) == {
        "seed", "run", "api", "retention", "vlm", "processing", "index", "eval",
        "detector-eval", "user", "status", "backup", "doctor", "install-services",
    }


def test_no_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        main([])


def test_seed_creates_and_populates_the_database(config_file: Path, capsys):
    assert main(["-c", str(config_file), "seed"]) == 0
    output = capsys.readouterr().out
    assert "2 camera(s), 2 zone(s), 4 policy(ies)" in output

    db_path = config_file.parent / "data" / "db" / "argus.sqlite3"
    assert db_path.exists()
    database = Database(db_path)
    assert database.schema_version() >= 1
    assert database.counts()["cameras"] == 2


def test_seed_twice_is_idempotent(config_file: Path, capsys):
    assert main(["-c", str(config_file), "seed"]) == 0
    assert main(["-c", str(config_file), "seed"]) == 0
    database = Database(config_file.parent / "data" / "db" / "argus.sqlite3")
    assert database.counts()["cameras"] == 2


def test_seed_reports_a_bad_file_without_a_traceback(config_file: Path, tmp_path: Path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("cameras:\n  - id: c\n")           # missing name/source
    assert main(["-c", str(config_file), "seed", "-f", str(bad)]) == 1
    assert "seed failed" in capsys.readouterr().err


def test_status_before_and_after_seeding(config_file: Path, capsys):
    assert main(["-c", str(config_file), "status"]) == 0
    first = capsys.readouterr().out
    assert f"schema v{SCHEMA_VERSION}" in first
    assert "cameras=0" in first

    main(["-c", str(config_file), "seed"])
    capsys.readouterr()
    assert main(["-c", str(config_file), "status"]) == 0
    output = capsys.readouterr().out
    assert "packing_line_demo" in output
    assert "manpower_headcount" in output


def test_status_does_not_print_a_camera_password(config_file: Path, capsys):
    """`argus status` is the command whose output gets pasted into a ticket."""
    main(["-c", str(config_file), "seed"])
    database = Database(config_file.parent / "data" / "db" / "argus.sqlite3")
    database.create_camera(
        "NVR 101", "rtsp://test:hunter2-pw@192.0.2.10:554/Streaming/Channels/101"
    )
    capsys.readouterr()
    assert main(["-c", str(config_file), "status"]) == 0
    captured = capsys.readouterr()
    assert "hunter2-pw" not in captured.out + captured.err
    assert "rtsp://test:***@192.0.2.10:554/Streaming/Channels/101" in captured.out


def test_api_serves_on_the_configured_address(config_file: Path, capsys, monkeypatch):
    """`argus api` hands the loaded config and the migrated database to uvicorn.

    The server itself is not started here — it never returns — but the wiring
    that decides *where* it listens and *which* database it serves is exactly
    what a config mistake breaks.
    """
    captured = {}

    def fake_run(config, database=None, *, host=None, port=None, **kwargs):
        captured.update(config=config, database=database, host=host, port=port)

    monkeypatch.setattr("argus.api.app.run", fake_run)
    assert main(["-c", str(config_file), "api", "--port", "18099"]) == 0
    assert (captured["host"], captured["port"]) == ("0.0.0.0", 18099)
    assert captured["database"].path == config_file.parent / "data" / "db" / "argus.sqlite3"
    assert "18099" in capsys.readouterr().out


def test_retention_dry_run_reports_without_deleting(config_file: Path, tmp_path: Path, capsys):
    main(["-c", str(config_file), "seed"])
    capsys.readouterr()
    clip = tmp_path / "data" / "clips" / "cam" / "old.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"x" * 2048)
    stale = time.time() - 90 * 86400
    os.utime(clip, (stale, stale))

    assert main(["-c", str(config_file), "retention", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "disk:" in output
    assert "clips" in output
    assert clip.exists()                      # a dry run deletes nothing

    assert main(["-c", str(config_file), "retention"]) == 0
    assert "reclaimed" in capsys.readouterr().out
    assert not clip.exists()


def test_run_without_cameras_fails_clearly(config_file: Path, capsys):
    assert main(["-c", str(config_file), "run", "--duration", "1"]) == 1
    assert "argus seed" in capsys.readouterr().err


def test_bad_config_is_reported_as_a_config_error(tmp_path: Path, capsys):
    path = tmp_path / "argus.yaml"
    path.write_text("detection:\n  fps: banana\n")
    assert main(["-c", str(path), "status"]) == 1
    assert "config error" in capsys.readouterr().err


def test_missing_config_file_is_reported(tmp_path: Path, capsys):
    assert main(["-c", str(tmp_path / "nope.yaml"), "status"]) == 1
    assert "not found" in capsys.readouterr().err
