from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from argus.config import DEFAULT_CONFIG_PATH, ROOT, ArgusConfig, ConfigError, load_config


def test_defaults_are_usable_without_a_file(monkeypatch, tmp_path: Path):
    """A missing default config is not an error — the dataclasses are a valid
    appliance configuration on their own."""
    monkeypatch.setattr("argus.config.DEFAULT_CONFIG_PATH", tmp_path / "absent.yaml")
    cfg = load_config(None, env={})
    assert cfg.source_path is None
    assert cfg.detection.fps > 0
    assert cfg.retention.clips_days > 0


def test_checked_in_config_loads_and_matches_spec():
    cfg = load_config(DEFAULT_CONFIG_PATH, env={})
    assert cfg.source_path == DEFAULT_CONFIG_PATH
    assert cfg.detection.fps == 5
    assert (cfg.detection.width, cfg.detection.height) == (960, 544)
    assert cfg.retention.clips_days == 14
    assert cfg.retention.snapshots_days == 30
    assert cfg.retention.events_days == 180
    assert cfg.retention.occupancy_days == 7
    assert cfg.gateway.api_port == 1984
    assert set(cfg.smtp.recipients) >= {"safety_manager", "line_supervisor"}


def test_data_dir_and_derived_paths_are_absolute():
    cfg = load_config(DEFAULT_CONFIG_PATH, env={})
    assert cfg.data_dir.is_absolute()
    assert cfg.data_dir == ROOT / "data"
    assert cfg.db_path == ROOT / "data" / "db" / "argus.sqlite3"
    assert cfg.gateway_binary == ROOT / "bin" / "go2rtc"


def test_yaml_overrides_defaults(tmp_path: Path):
    path = tmp_path / "argus.yaml"
    path.write_text(
        textwrap.dedent(
            """
            data_dir: /srv/argus
            detection:
              fps: 10
              width: 640
              height: 384
            gateway:
              api_port: 2984
            """
        )
    )
    cfg = load_config(path, env={})
    assert cfg.data_dir == Path("/srv/argus")
    assert cfg.detection.fps == 10
    assert cfg.detection.width == 640
    assert cfg.gateway.api_port == 2984
    # untouched sections keep their defaults
    assert cfg.retention.events_days == 180


def test_env_overrides_beat_yaml(tmp_path: Path):
    path = tmp_path / "argus.yaml"
    path.write_text("detection:\n  fps: 10\n  hwaccel: true\n")
    cfg = load_config(
        path,
        env={
            "ARGUS_DETECTION_FPS": "2.5",
            "ARGUS_DETECTION_HWACCEL": "false",
            "ARGUS_GATEWAY_API_PORT": "9999",
            "ARGUS_RETENTION_CLIPS_DAYS": "3",
        },
    )
    assert cfg.detection.fps == 2.5
    assert cfg.detection.hwaccel is False
    assert cfg.gateway.api_port == 9999
    assert cfg.retention.clips_days == 3


def test_unknown_key_is_rejected(tmp_path: Path):
    path = tmp_path / "argus.yaml"
    path.write_text("detection:\n  fpz: 5\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path, env={})


def test_bad_type_is_rejected(tmp_path: Path):
    path = tmp_path / "argus.yaml"
    path.write_text("detection:\n  fps: not-a-number\n")
    with pytest.raises(ConfigError, match="expected a number"):
        load_config(path, env={})


def test_missing_explicit_config_is_an_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", env={})


def test_invalid_detection_values_rejected(tmp_path: Path):
    path = tmp_path / "argus.yaml"
    path.write_text("detection:\n  fps: 0\n")
    with pytest.raises(ConfigError, match="fps must be"):
        load_config(path, env={})


# --------------------------------------------------------------------------
# detector continuity (ongoing conditions for zone occupancy)
# --------------------------------------------------------------------------


def test_the_shipped_config_smooths_the_detectors_zone_decision():
    """On by default, and the defaults are the ones the comments argue for. This
    is the block that turned 10 958 headcount events a day into one per breach;
    a site that has it off has chosen that."""
    cfg = load_config(DEFAULT_CONFIG_PATH, env={})
    continuity = cfg.detection.continuity
    assert continuity.enabled is True
    assert continuity.window_sec == 3.0            # ~15 samples at 5 fps
    assert continuity.sustain_sec == 5.0
    # Hysteresis: closing must be slower than opening, or a count hovering at
    # the limit is a dozen conditions instead of one.
    assert continuity.clear_sec > continuity.sustain_sec


def test_a_stale_window_inside_the_update_cadence_is_refused(tmp_path: Path):
    """The VLM side learned this on a live run: a window shorter than the
    interval that feeds it closes a perfectly healthy condition between two of
    its own observations and immediately reopens one. Here the arithmetic is
    known at load time, so it is refused at load time."""
    path = tmp_path / "argus.yaml"
    path.write_text("detection:\n  continuity:\n    update_sec: 60\n    stale_after_sec: 30\n")
    with pytest.raises(ConfigError, match="stale_after_sec must be greater"):
        load_config(path, env={})


def test_the_continuity_knobs_can_be_tuned_per_site(tmp_path: Path):
    path = tmp_path / "argus.yaml"
    path.write_text(
        "detection:\n  continuity:\n    enabled: false\n"
        "    window_sec: 6\n    sustain_sec: 20\n    clear_sec: 45\n"
    )
    continuity = load_config(path, env={}).detection.continuity
    assert continuity.enabled is False
    assert (continuity.window_sec, continuity.sustain_sec, continuity.clear_sec) == (6.0, 20.0, 45.0)


def test_the_release_margin_ships_on_and_refuses_nonsense(tmp_path: Path):
    """A count sitting on its limit was ten conditions in ten minutes without it."""
    assert load_config(DEFAULT_CONFIG_PATH, env={}).detection.continuity.release_margin == 0.10
    path = tmp_path / "argus.yaml"
    path.write_text("detection:\n  continuity:\n    release_margin: 0\n")
    assert load_config(path, env={}).detection.continuity.release_margin == 0.0
    for nonsense in ("1.0", "-0.1"):
        path.write_text(f"detection:\n  continuity:\n    release_margin: {nonsense}\n")
        with pytest.raises(ConfigError):
            load_config(path, env={})


def test_smtp_password_comes_from_env_not_the_file(monkeypatch):
    cfg = load_config(DEFAULT_CONFIG_PATH, env={})
    monkeypatch.delenv(cfg.smtp.password_env, raising=False)
    assert cfg.smtp.password == ""
    monkeypatch.setenv(cfg.smtp.password_env, "hunter2")
    assert cfg.smtp.password == "hunter2"
    assert "hunter2" not in DEFAULT_CONFIG_PATH.read_text()


def test_recipients_lookup():
    cfg = load_config(DEFAULT_CONFIG_PATH, env={})
    assert cfg.smtp.recipients_for("safety_manager")
    assert cfg.smtp.recipients_for("nobody") == []
    assert cfg.smtp.recipients_for(None) == []


def test_ensure_dirs_creates_the_tree(tmp_path: Path):
    cfg = ArgusConfig(data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    for path in (cfg.snapshots_dir, cfg.clips_dir, cfg.ring_dir, cfg.gateway_dir,
                 cfg.logs_dir, cfg.db_path.parent):
        assert path.is_dir()


def test_rtsp_url_binds_to_loopback_when_listening_on_all():
    cfg = ArgusConfig()
    cfg.gateway.listen_host = "0.0.0.0"
    assert cfg.rtsp_url("cam1") == f"rtsp://127.0.0.1:{cfg.gateway.rtsp_port}/cam1"
    assert cfg.gateway_url.startswith("http://127.0.0.1:")


def test_frame_bytes_matches_resolution():
    cfg = load_config(DEFAULT_CONFIG_PATH, env={})
    assert cfg.detection.frame_bytes == 960 * 544 * 3


# --------------------------------------------------------------------------
# gateway.webrtc_candidates
# --------------------------------------------------------------------------


def test_no_webrtc_candidates_are_advertised_by_default():
    """A LAN-only site names nothing: go2rtc's own interface discovery is
    already correct there, and an empty list keeps the generated go2rtc.yaml
    byte-identical to what it has always been."""
    assert ArgusConfig().gateway.webrtc_candidates == []
    assert ArgusConfig().gateway.webrtc_candidate_list == []


def test_webrtc_candidates_are_read_as_a_list(tmp_path: Path):
    path = tmp_path / "argus.yaml"
    path.write_text(
        textwrap.dedent(
            """
            gateway:
              webrtc_candidates:
                - "100.113.208.36:8555"
                - "vpn.plant.example:8555"
            """
        )
    )
    cfg = load_config(path, env={})
    assert cfg.gateway.webrtc_candidate_list == [
        "100.113.208.36:8555", "vpn.plant.example:8555",
    ]


def test_webrtc_candidates_can_come_from_the_environment(tmp_path: Path):
    """The ARGUS_ layer sets fields with setattr, so a list arrives as the
    comma-separated string an operator can actually type into a unit file.
    Without this it would silently become a *string* in the generated YAML,
    which go2rtc rejects at parse time — a gateway that crash-loops rather than
    a setting that did not apply."""
    path = tmp_path / "argus.yaml"
    path.write_text("gateway:\n  webrtc_candidates: []\n")
    cfg = load_config(path, env={"ARGUS_GATEWAY_WEBRTC_CANDIDATES": "10.0.0.9:8555, 10.0.0.10:8555"})
    assert cfg.gateway.webrtc_candidate_list == ["10.0.0.9:8555", "10.0.0.10:8555"]


def test_an_empty_environment_override_advertises_nothing(tmp_path: Path):
    path = tmp_path / "argus.yaml"
    path.write_text('gateway:\n  webrtc_candidates: ["100.113.208.36:8555"]\n')
    cfg = load_config(path, env={"ARGUS_GATEWAY_WEBRTC_CANDIDATES": ""})
    assert cfg.gateway.webrtc_candidate_list == []


@pytest.mark.parametrize("value", ["100.113.208.36", "100.113.208.36:", ":8555",
                                   "100.113.208.36:https", "100.113.208.36:0",
                                   "100.113.208.36:70000"])
def test_a_malformed_webrtc_candidate_is_rejected(tmp_path: Path, value: str):
    """go2rtc ignores a candidate it cannot parse and logs nothing, so the
    symptom of a typo here is tiles that never connect for reasons no operator
    can see. Fail at load instead."""
    path = tmp_path / "argus.yaml"
    path.write_text(f'gateway:\n  webrtc_candidates: ["{value}"]\n')
    with pytest.raises(ConfigError, match="webrtc_candidates"):
        load_config(path, env={})


def test_this_appliance_advertises_candidates_on_its_own_webrtc_port():
    """Whatever a site names, every entry has to be host:port on the port
    go2rtc is actually listening on — a candidate pointing at the wrong port is
    a candidate every browser will time out on."""
    cfg = load_config(DEFAULT_CONFIG_PATH, env={})
    for candidate in cfg.gateway.webrtc_candidate_list:
        host, _, port = candidate.rpartition(":")
        assert host
        assert int(port.split("/")[0]) == cfg.gateway.webrtc_port


# --------------------------------------------------------------------------
# site & shifts
# --------------------------------------------------------------------------


def test_shifts_default_to_the_usual_three_and_wrap_midnight():
    cfg = ArgusConfig()
    assert [s["id"] for s in cfg.site.shifts] == ["A", "B", "C"]
    windows = cfg.site.shift_windows
    assert windows[0] == {"id": "A", "start_hour": 6, "end_hour": 14}
    # C runs 22:00 -> 06:00: the wrap is what makes a night shift expressible.
    assert windows[-1] == {"id": "C", "start_hour": 22, "end_hour": 6}


@pytest.mark.parametrize(
    ("hour", "expected"),
    [(0, "C"), (5, "C"), (6, "A"), (13, "A"), (14, "B"), (21, "B"), (22, "C"), (23, "C")],
)
def test_every_hour_of_the_day_belongs_to_exactly_one_shift(hour: int, expected: str):
    assert ArgusConfig().site.shift_for_hour(hour)["id"] == expected


def test_a_plant_can_declare_its_own_rota(tmp_path: Path):
    path = tmp_path / "argus.yaml"
    path.write_text(textwrap.dedent("""
        site:
          name: Veltra Stamping OH-1
          shifts:
            - {id: NIGHT, start_hour: 20}
            - {id: DAY, start_hour: 8}
    """))
    cfg = load_config(path, env={})
    assert cfg.site.name == "Veltra Stamping OH-1"
    # Declared out of order, stored in clock order.
    assert [s["id"] for s in cfg.site.shifts] == ["DAY", "NIGHT"]
    assert cfg.site.shift_for_hour(9)["id"] == "DAY"
    assert cfg.site.shift_for_hour(3)["id"] == "NIGHT"


@pytest.mark.parametrize("shifts", [
    "[]",
    "[{id: A, start_hour: 6}, {id: B, start_hour: 6}]",   # two shifts, same hour
    "[{id: A, start_hour: 26}]",                          # not an hour
    "[{id: '', start_hour: 6}]",                          # nameless
])
def test_an_impossible_rota_is_rejected(tmp_path: Path, shifts: str):
    path = tmp_path / "argus.yaml"
    path.write_text(f"site:\n  shifts: {shifts}\n")
    with pytest.raises(ConfigError):
        load_config(path, env={})


def test_ui_dist_resolves_against_the_repo_root():
    cfg = ArgusConfig()
    assert cfg.ui_dist_dir == ROOT / "ui" / "dist"


# --------------------------------------------------------------------------
# retention: per-stage evidence windows and the hard budget
# --------------------------------------------------------------------------


def test_the_shipped_config_bounds_evidence_by_stage_and_by_size():
    """This file is what an appliance ships with, and after 2026-08-29 an
    appliance that ships without a volume ceiling is a bug. 578 GB of clips in
    2.5 days, under a flat 14-day window and a disk guard 3.5 TB away."""
    cfg = load_config(ROOT / "config" / "argus.yaml", env={})
    evidence = cfg.retention.evidence
    assert evidence.configured
    # Shortest for what nobody was told about, longest for what a human judged.
    assert evidence.shadow_days < evidence.reviewed_days < evidence.confirmed_days
    # A labelled negative is an eval input the deployment produced for free.
    assert evidence.false_positive_days == evidence.confirmed_days
    assert cfg.retention.max_clips_gb > 0 and cfg.retention.max_snapshots_gb > 0


def test_a_config_with_no_evidence_block_keeps_every_flat_window(tmp_path: Path):
    """The upgrade contract, asserted at the loader: writing nothing new must
    not change what an existing appliance deletes."""
    path = tmp_path / "argus.yaml"
    path.write_text("retention:\n  clips_days: 21\n")
    cfg = load_config(path, env={})
    assert not cfg.retention.evidence.configured
    assert cfg.retention.budget_gb("clips") == 0.0
    for tier in ("shadow", "reviewed", "auto_routed", "confirmed", "false_positive"):
        assert cfg.retention.evidence.days(tier, cfg.retention.clips_days) == 21


def test_a_partly_written_evidence_block_inherits_the_rest(tmp_path: Path):
    path = tmp_path / "argus.yaml"
    path.write_text(textwrap.dedent("""
        retention:
          clips_days: 14
          evidence:
            shadow_days: 1
            confirmed_days: 90
    """))
    evidence = load_config(path, env={}).retention.evidence
    assert evidence.days("shadow", 14) == 1
    assert evidence.days("confirmed", 14) == 90
    assert evidence.days("reviewed", 14) == 14          # unset = the flat window
    assert evidence.days("auto_routed", 14) == 14       # and so does its parent


def test_auto_routed_follows_reviewed_days_when_only_that_is_set(tmp_path: Path):
    path = tmp_path / "argus.yaml"
    path.write_text("retention:\n  evidence:\n    reviewed_days: 3\n")
    evidence = load_config(path, env={}).retention.evidence
    assert evidence.days("auto_routed", 14) == 3


@pytest.mark.parametrize("block", [
    "retention:\n  evidence:\n    shadow_days: -1\n",
    "retention:\n  max_clips_gb: -5\n",
    "retention:\n  max_snapshots_gb: -0.5\n",
])
def test_a_negative_window_or_budget_is_rejected(tmp_path: Path, block: str):
    path = tmp_path / "argus.yaml"
    path.write_text(block)
    with pytest.raises(ConfigError):
        load_config(path, env={})


def test_a_typo_in_an_evidence_window_is_caught_by_the_loader(tmp_path: Path):
    """`float | None` used to fall through the coercion table untouched, so
    `shadow_days: banana` reached the sweeper as a string. The loader is the one
    place this config tree refuses to let typos past."""
    path = tmp_path / "argus.yaml"
    path.write_text("retention:\n  evidence:\n    shadow_days: banana\n")
    with pytest.raises(ConfigError, match="shadow_days"):
        load_config(path, env={})

    path.write_text("retention:\n  evidence:\n    shadwo_days: 1\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path, env={})


def test_an_explicit_null_window_means_inherit(tmp_path: Path):
    path = tmp_path / "argus.yaml"
    path.write_text("retention:\n  evidence:\n    shadow_days: null\n    confirmed_days: 30\n")
    evidence = load_config(path, env={}).retention.evidence
    assert evidence.shadow_days is None
    assert evidence.days("shadow", 14) == 14
    assert evidence.configured                     # confirmed_days still counts


def test_the_budget_can_be_overridden_from_the_environment(tmp_path: Path):
    """Every scalar in this tree carries an ARGUS_ override, and an operator
    firefighting a full disk at 3 a.m. should not have to edit YAML first."""
    path = tmp_path / "argus.yaml"
    path.write_text("retention:\n  max_clips_gb: 100\n")
    cfg = load_config(path, env={"ARGUS_RETENTION_MAX_CLIPS_GB": "40",
                                 "ARGUS_RETENTION_EVIDENCE_SHADOW_DAYS": "0.5"})
    assert cfg.retention.max_clips_gb == 40.0
    assert cfg.retention.evidence.shadow_days == 0.5
