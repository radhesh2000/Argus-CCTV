"""Shared fixtures. Every test runs against a temporary data dir — nothing here
touches the real ``data/`` tree or the checked-in database."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from argus.config import ROOT, ArgusConfig
from argus.db import Database

SAMPLE_VIDEO = ROOT / "packing_section_recording.mp4"


@pytest.fixture
def config(tmp_path: Path) -> ArgusConfig:
    """A config rooted in tmp_path, with ports well away from a running Argus."""
    cfg = ArgusConfig(data_dir=tmp_path / "data")
    # Point the SPA mount at a directory that does not exist, so tests describe
    # the API alone and do not start passing or failing depending on whether
    # somebody has run `npm run build` in ui/.
    cfg.ui_dist = tmp_path / "no-ui"
    cfg.gateway.api_port = 11984
    cfg.gateway.rtsp_port = 18554
    cfg.gateway.webrtc_port = 18555
    cfg.gateway.listen_host = "127.0.0.1"
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def database(config: ArgusConfig) -> Database:
    db = Database(config.db_path)
    db.migrate()
    return db


# --------------------------------------------------------------------------
# accounts
# --------------------------------------------------------------------------
#
# Since Phase 5 every API route needs a session, so a test that is *about*
# something else (events, search, health) needs one incidentally. `sign_in`
# gives it to them in one line and keeps the authorisation matrix itself in
# tests/test_auth.py, where a reviewer can read it as one table rather than
# reconstructing it from four hundred assertions.

#: Long enough to pass `argus.auth.MIN_PASSWORD_LENGTH` everywhere.
TEST_PASSWORD = "correct-horse-battery-staple"


@pytest.fixture
def make_user(database: Database):
    """Create an account and return its row. `make_user('ADMIN')` is the usual
    call; the username defaults to the role, lower-cased."""
    from argus import accounts

    def _make(role: str = "ADMIN", username: str | None = None,
              password: str = TEST_PASSWORD, **kwargs):
        return accounts.create_user(
            database, username or role.lower(), password, role=role, **kwargs
        )

    return _make


@pytest.fixture
def sign_in(make_user):
    """Log a client in as a role, creating the account if it does not exist.

    Returns the created user's row, and leaves the session cookie on the client
    — so a test reads ``sign_in(client, 'REVIEW')`` and then simply makes
    requests.
    """
    def _sign_in(client, role: str = "ADMIN", username: str | None = None):
        name = username or role.lower()
        user = make_user(role, name)
        response = client.post(
            "/api/auth/login", json={"username": name, "password": TEST_PASSWORD}
        )
        assert response.status_code == 200, response.text
        return user

    return _sign_in


@pytest.fixture(scope="session")
def sample_video() -> Path:
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"sample video missing: {SAMPLE_VIDEO}")
    return SAMPLE_VIDEO


#: An H.265 source, which the checked-in sample is not. The Anmol export is the
#: only HEVC footage on this box and it is gitignored (customer plant), so
#: anything that needs it skips rather than fails on a fresh clone — the same
#: bargain `config/seed.anmol.yaml` strikes with `optional:`.
HEVC_VIDEO = ROOT / "footage" / "anmol-samples" / "Line_No.3_Packing3_Bas.MP4"


@pytest.fixture(scope="session")
def hevc_video() -> Path:
    if not HEVC_VIDEO.exists():
        pytest.skip(f"no HEVC sample on this box: {HEVC_VIDEO} (gitignored footage)")
    return HEVC_VIDEO


@pytest.fixture(scope="session")
def ffmpeg() -> str:
    """The ffmpeg *Argus* will use, which is not necessarily the one on PATH.

    ``argus.ingest.decoder`` prefers the vendored ``bin/ffmpeg`` built by
    ``scripts/build_ffmpeg.sh`` (LGPL, with NVDEC). Resolving this fixture with
    ``shutil.which`` instead would hand tests a different binary from the one
    under test — and a test asking "does hardware decode work here?" would then
    answer for the wrong build.
    """
    from argus.ingest.decoder import FFMPEG

    if not (Path(FFMPEG).is_file() or shutil.which(FFMPEG)):
        pytest.skip("ffmpeg not available")
    return FFMPEG
