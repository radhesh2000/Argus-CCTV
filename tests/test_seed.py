from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from argus.db import Camera, Database
from argus.seed import DEFAULT_SEED_PATH, SeedError, apply_seed, load_seed, seed

CHECKED_IN_SEED = DEFAULT_SEED_PATH


def test_checked_in_seed_file_exists():
    assert CHECKED_IN_SEED.exists(), "config/seed.yaml is the declarative fleet definition"


def test_seed_imports_the_checked_in_file(database: Database):
    result = seed(database)
    assert result.cameras == 2
    assert result.zones == 2
    assert result.policies == 4

    cameras = {c["id"]: c for c in database.list_cameras()}
    assert set(cameras) == {"packing_line_demo", "packing_line_west_demo"}
    assert cameras["packing_line_demo"]["name"] == "Packing line (demo)"
    assert cameras["packing_line_demo"]["kind"] == "file"


def test_seed_is_idempotent(database: Database):
    """Running `argus seed` twice must be a no-op — this is the property that
    makes it safe to wire into every install and upgrade."""
    seed(database)
    first = _snapshot(database)
    seed(database)
    assert _snapshot(database) == first
    assert database.counts()["cameras"] == 2


def _snapshot(database: Database) -> dict:
    def strip(rows, *drop):
        return [{k: v for k, v in row.items() if k not in drop} for row in rows]

    return {
        "cameras": strip(database.list_cameras(), "updated_at", "created_at"),
        "zones": strip(database.list_zones(), "updated_at", "created_at"),
        "policies": strip(database.list_policies(), "updated_at", "created_at"),
    }


def test_seed_round_trips_every_field(database: Database):
    raw = load_seed()
    seed(database)

    for entry in raw["cameras"]:
        row = database.get_camera(entry["id"])
        assert row["name"] == entry["name"]
        assert bool(row["enabled"]) == bool(entry.get("enabled", True))

    for entry in raw["zones"]:
        row = database.get_zone(entry["id"])
        assert row["camera_id"] == entry["camera_id"]
        assert row["name"] == entry["name"]
        assert row["polygon"] == entry["polygon"]
        assert row["sanctioned_limit"] == entry.get("sanctioned_limit")
        assert row["color"] == entry.get("color", "#22d3ee")

    for entry in raw["policies"]:
        row = database.get_policy(entry["id"])
        assert row["kind"] == entry["kind"]
        assert row["detector_key"] == entry["detector_key"]
        assert row["label"] == entry["label"]
        assert bool(row["enabled"]) == bool(entry.get("enabled", True))
        assert row["stage"] == entry.get("stage", "shadow")
        assert row["lifecycle"] == entry.get("lifecycle", "draft")
        assert row["severity"] == entry.get("severity", "medium")
        assert row["routed_to"] == entry.get("routed_to")
        assert row["cooldown_sec"] == entry.get("cooldown_sec", 30.0)
        assert row["threshold"] == entry.get("threshold")
        assert row["interval_sec"] == entry.get("interval_sec")
        if entry.get("prompt"):
            assert row["prompt"] == entry["prompt"].strip()


def test_v1_semantics_survive_the_import(database: Database):
    """The staged-rollout and lifecycle state that the product is built on."""
    seed(database)
    headcount = database.get_policy("manpower_headcount")
    assert headcount["kind"] == "detector"
    assert (headcount["stage"], headcount["lifecycle"]) == ("reviewed", "production")
    assert headcount["threshold"] == 4.0

    # The AGPL-backed PPE detector ships disabled until the Apache replacement lands.
    assert database.get_policy("hardhat_compliance")["enabled"] == 0

    exit_policy = database.get_policy("emergency_exit_blocked")
    assert exit_policy["kind"] == "vlm"
    assert exit_policy["severity"] == "high"
    assert exit_policy["routed_to"] == "safety_manager"
    assert exit_policy["prompt"].startswith("Look at this factory floor camera frame.")
    assert exit_policy["interval_sec"] == 8.0


def test_file_sources_become_absolute_paths(database: Database):
    """The gateway and decoder are started from a supervisor with an unknown
    working directory; a relative source would break there and only there."""
    seed(database)
    for camera in database.list_cameras():
        assert Path(camera["source"]).is_absolute()
        assert Path(camera["source"]).exists()


def test_seed_preserves_rows_dropped_from_the_yaml(database: Database, tmp_path: Path):
    seed(database)
    trimmed = tmp_path / "seed.yaml"
    data = load_seed()
    data["cameras"] = data["cameras"][:1]
    data["zones"] = data["zones"][:1]
    trimmed.write_text(yaml.safe_dump(data))

    seed(database, trimmed)
    # Removing a camera from the file must not delete it (it owns evidence).
    assert database.counts()["cameras"] == 2


def test_seed_updates_changed_values(database: Database, tmp_path: Path):
    seed(database)
    data = load_seed()
    data["cameras"][0]["name"] = "Renamed line"
    data["cameras"][0]["enabled"] = False
    data["zones"][0]["sanctioned_limit"] = 99
    path = tmp_path / "seed.yaml"
    path.write_text(yaml.safe_dump(data))

    seed(database, path)
    camera = database.get_camera(data["cameras"][0]["id"])
    assert camera["name"] == "Renamed line" and camera["enabled"] == 0
    assert database.get_zone(data["zones"][0]["id"])["sanctioned_limit"] == 99


# --- validation -----------------------------------------------------------


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "seed.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_missing_file_is_an_error(tmp_path: Path):
    with pytest.raises(SeedError, match="not found"):
        load_seed(tmp_path / "absent.yaml")


def test_unknown_top_level_key_rejected(tmp_path: Path):
    with pytest.raises(SeedError, match="unknown top-level"):
        load_seed(_write(tmp_path, {"camaras": []}))


def test_unknown_camera_key_rejected(database: Database, tmp_path: Path):
    payload = {"cameras": [{"id": "c", "name": "n", "source": "/a.mp4", "rtsp_url": "x"}]}
    with pytest.raises(SeedError, match="unknown key"):
        apply_seed(database, load_seed(_write(tmp_path, payload)))


def test_missing_required_camera_field_rejected(database: Database, tmp_path: Path):
    with pytest.raises(SeedError, match="missing 'source'"):
        apply_seed(database, load_seed(_write(tmp_path, {"cameras": [{"id": "c", "name": "n"}]})))


def test_zone_referencing_unknown_camera_rejected(database: Database, tmp_path: Path):
    payload = {"zones": [{"id": "z", "camera_id": "ghost", "name": "n",
                          "polygon": [[0, 0], [1, 0], [1, 1]]}]}
    with pytest.raises(SeedError, match="unknown camera_id"):
        apply_seed(database, load_seed(_write(tmp_path, payload)))


def test_degenerate_polygon_rejected(database: Database, tmp_path: Path):
    database.upsert_camera(Camera(id="c", name="n", source="/a.mp4"))
    payload = {"zones": [{"id": "z", "camera_id": "c", "name": "n", "polygon": [[0, 0], [1, 1]]}]}
    with pytest.raises(SeedError, match="at least 3 points"):
        apply_seed(database, load_seed(_write(tmp_path, payload)))


def test_seed_yaml_roles_exist_in_argus_yaml():
    """A policy routed to a role with no recipients is an alert that silently
    goes nowhere — catch the mismatch at test time, not at 2am."""
    from argus.config import DEFAULT_CONFIG_PATH, load_config

    config = load_config(DEFAULT_CONFIG_PATH, env={})
    routed = {
        entry.get("routed_to")
        for entry in (load_seed().get("policies") or [])
        if entry.get("routed_to")
    }
    assert routed <= set(config.smtp.recipients), (
        f"seed.yaml routes to roles with no recipients: {routed - set(config.smtp.recipients)}"
    )


# --- optional cameras: a seed file whose footage may not be here -----------
#
# `config/seed.anmol.yaml` enrols a customer's video files from a gitignored
# directory. On the box holding them it is a six-camera plant; on a fresh clone
# it has to be a sentence, not a traceback — and above all not six cameras
# pointing at nothing, which is an offline tile and a decoder restarting for
# ever behind it.

ANMOL_SEED = DEFAULT_SEED_PATH.parent / "seed.anmol.yaml"


def _optional_fleet(tmp_path: Path, *, present: bool) -> Path:
    """A two-camera optional seed, one zone and one policy scoped to it."""
    source = tmp_path / "plant.mp4"
    if present:
        source.write_bytes(b"not really a video, but it is on disk")
    return _write(tmp_path, {
        "cameras": [
            {"id": "plant_1", "name": "Plant 1", "source": str(source), "optional": True},
        ],
        "zones": [
            {"id": "plant_1_floor", "camera_id": "plant_1", "name": "Floor",
             "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]], "sanctioned_limit": 3},
        ],
        "policies": [
            {"id": "plant_headcount", "kind": "detector", "detector_key": "manpower_headcount",
             "label": "Headcount", "cameras": ["plant_1"]},
        ],
    })


def test_checked_in_anmol_seed_exists_and_parses():
    assert ANMOL_SEED.exists(), "config/seed.anmol.yaml enrols the Anmol plant footage"
    data = load_seed(ANMOL_SEED)
    assert len(data["cameras"]) == 6
    assert len(data["zones"]) == 10
    assert all(camera["optional"] for camera in data["cameras"])
    assert all(camera["source"].startswith("./footage/") for camera in data["cameras"])


def test_every_anmol_prompt_policy_names_one_camera_and_one_zone():
    """The rule that makes four new detection classes affordable at all.

    An unscoped prompt policy runs on every enabled camera, so one of these six
    written without a `cameras:` key would be six schedules rather than one and
    would take the whole VLM budget on its own. Scope is not a style choice here
    and a reviewer adding a fifth policy should trip on this test, not on a
    `skipped_in_flight` counter three weeks into a pilot.
    """
    data = load_seed(ANMOL_SEED)
    policies = data["policies"]
    assert len(policies) == 4
    cameras = {camera["id"] for camera in data["cameras"]}
    zones = {zone["id"]: zone["camera_id"] for zone in data["zones"]}
    for policy in policies:
        assert policy["kind"] == "vlm" and policy["detector_key"] == "vlm_prompt"
        scope = policy["cameras"]
        assert len(scope) == 1, f"{policy['id']} is scoped to {len(scope)} cameras"
        entry = scope[0]
        assert entry["camera"] in cameras
        # A zone scope is also the crop the model is shown, so a prompt policy
        # without one is asked over the whole 2560x1440 view.
        assert zones[entry["zone"]] == entry["camera"], (
            f"{policy['id']} is scoped to a zone on a different camera"
        )


def test_anmol_prompt_policies_fit_the_measured_vlm_budget():
    """The arithmetic written into the file's header, as an assertion.

    Capacity is a *rate*: 6.1 s per evaluation at `max_concurrency: 2` measured
    with the vision engine running is 0.345 evaluations/s (docs/BENCHMARKS.md),
    and what a policy costs is 1/interval — never 1. Four classes at 20/30/45/90
    seconds come to 0.117/s, a third of the box. The margin exists so that the
    demo fleet's own two prompt policies (0.292/s) can also be loaded on a
    developer box without the schedule collapsing, which is exactly the state
    this repository is usually in.
    """
    data = load_seed(ANMOL_SEED)
    demand = sum(1.0 / float(policy["interval_sec"]) for policy in data["policies"])
    assert demand == pytest.approx(0.1167, abs=0.001)
    assert demand < 0.345 / 2, "the plant fleet alone should not need half the box"
    # Measured 2026-08-25 over two live runs (600 s and 200 s, six cameras, the
    # engine running): 87 evaluations, 0 skipped_in_flight. See docs/VLM_EVAL.md.
    assert demand * 800 == pytest.approx(93, abs=15)


def test_anmol_prompt_policies_ship_in_shadow_and_pilot():
    """Shadow so nobody is emailed, pilot so they actually run.

    `draft` would be the intuitive "not ready yet" value and it is the wrong
    one: `Database.list_policies(active_only=True)` filters to pilot and
    production, so a draft policy is never evaluated, produces no evidence, and
    can never earn the promotion the staged rollout exists to grant.
    """
    for policy in load_seed(ANMOL_SEED)["policies"]:
        assert policy["stage"] == "shadow", policy["id"]
        assert policy["lifecycle"] == "pilot", policy["id"]
        assert policy.get("routed_to") is None, f"{policy['id']} routes to somebody in shadow"


def test_zones_that_only_frame_a_prompt_carry_no_sanctioned_limit():
    """A zone with a limit is a headcount alert; these four are crops.

    `manpower_headcount` is unscoped, so it lands on every zone that has a
    limit. Four new zones drawn to frame a VLM question would otherwise be four
    new ways for a detector policy to fire, on rectangles nobody chose for
    counting people.
    """
    zones = {zone["id"]: zone for zone in load_seed(ANMOL_SEED)["zones"]}
    framing = {
        "anmol_broken_biscuits_trays", "anmol_packing_4_apron",
        "anmol_packing_3_gangway", "anmol_loading_point_bay",
    }
    assert framing <= set(zones)
    for zone_id in framing:
        assert zones[zone_id].get("sanctioned_limit") is None, zone_id
    scoped = {
        entry["zone"] for policy in load_seed(ANMOL_SEED)["policies"]
        for entry in policy["cameras"]
    }
    assert scoped == framing


def test_anmol_prompt_policies_import_with_their_zone_scope(database: Database):
    seed(database)
    result = seed(database, ANMOL_SEED)
    if result.skipped_cameras:
        pytest.skip("the Anmol footage is not on this box (footage/ is gitignored)")
    assert result.policies == 4
    for policy in database.list_policies(kind="vlm", enabled_only=True, active_only=True):
        if not policy["id"].startswith("anmol_"):
            continue
        scope = policy["scope"]
        assert len(scope) == 1 and scope[0]["zone_id"], policy["id"]


def test_anmol_seed_never_touches_the_demo_cameras(database: Database):
    """Both files are imported on this box, in either order, and the demo fleet
    has to survive it — `argus seed` upserts and never deletes."""
    seed(database)
    before = {c["id"]: dict(c) for c in database.list_cameras()}
    seed(database, ANMOL_SEED)
    after = {c["id"]: dict(c) for c in database.list_cameras()}
    for camera_id, row in before.items():
        assert after[camera_id] == row


def test_optional_camera_present_is_imported_normally(database: Database, tmp_path: Path):
    result = seed(database, _optional_fleet(tmp_path, present=True))
    assert (result.cameras, result.zones, result.policies) == (1, 1, 1)
    assert result.skipped == 0
    assert result.notes == []


def test_optional_camera_absent_is_skipped_with_its_zone(database: Database, tmp_path: Path):
    result = seed(database, _optional_fleet(tmp_path, present=False))
    assert (result.cameras, result.zones) == (0, 0)
    assert (result.skipped_cameras, result.skipped_zones) == (1, 1)
    assert database.list_cameras() == []
    # The message names the file, because "skipped 6 cameras" without a path is
    # a support call.
    assert len(result.notes) == 2
    assert "plant_1" in result.notes[0] and "plant.mp4" in result.notes[0]
    assert "footage not on this box" in str(result)


def test_a_policy_left_with_no_scope_is_skipped_not_widened(database: Database, tmp_path: Path):
    """The dangerous case. An empty `cameras:` means *every* camera everywhere
    else in this file, so a scope that is emptied by skipping would silently
    point a plant's policy at whatever else is enrolled — here, the demo fleet.
    """
    seed(database)
    result = seed(database, _optional_fleet(tmp_path, present=False))
    assert result.policies == 0
    assert result.skipped_policies == 1
    assert "plant_headcount" not in {p["id"] for p in database.list_policies()}


def test_a_partly_present_scope_keeps_the_cameras_that_are_here(
    database: Database, tmp_path: Path
):
    here = tmp_path / "here.mp4"
    here.write_bytes(b"video")
    path = _write(tmp_path, {
        "cameras": [
            {"id": "cam_here", "name": "Here", "source": str(here), "optional": True},
            {"id": "cam_gone", "name": "Gone", "source": str(tmp_path / "gone.mp4"),
             "optional": True},
        ],
        "policies": [
            {"id": "mixed", "kind": "detector", "detector_key": "manpower_headcount",
             "label": "Headcount", "cameras": ["cam_here", {"camera": "cam_gone"}]},
        ],
    })
    result = seed(database, path)
    assert (result.cameras, result.policies) == (1, 1)
    assert result.skipped_cameras == 1
    assert database.policy_scope("mixed") == [{"camera_id": "cam_here", "zone_id": None}]


def test_optional_is_only_about_files_not_unreachable_cameras(
    database: Database, tmp_path: Path
):
    """Proving an RTSP camera absent means opening a socket to it. A seed that
    pays a 30 s ffprobe timeout per unreachable camera is a seed nobody runs."""
    path = _write(tmp_path, {
        "cameras": [{"id": "rtsp_cam", "name": "RTSP", "optional": True,
                     "source": "rtsp://192.0.2.1/stream"}],
    })
    result = seed(database, path)
    assert result.cameras == 1 and result.skipped_cameras == 0


def test_a_camera_without_optional_still_fails_loudly(database: Database, tmp_path: Path):
    """`optional:` is opt-in. A normal camera whose file is missing is imported
    exactly as before — the operator meant it, and the console will say offline.
    """
    path = _write(tmp_path, {
        "cameras": [{"id": "cam", "name": "Cam", "source": str(tmp_path / "nope.mp4")}],
    })
    result = seed(database, path)
    assert result.cameras == 1 and result.skipped == 0
