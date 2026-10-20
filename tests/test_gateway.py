from __future__ import annotations

import itertools
import time
from pathlib import Path

import pytest
import yaml

from argus.config import ArgusConfig
from argus.db import Camera, Database
from argus.ingest.decoder import FrameSource, FrameSourceThread
from argus.ingest.gateway import (
    Gateway,
    GatewayConfigError,
    build_config,
    stream_name,
    write_config,
)


def _cameras() -> list[dict]:
    return [
        {"id": "file_cam", "source": "/videos/demo.mp4", "kind": "file", "enabled": 1},
        {"id": "rtsp_cam", "source": "rtsp://cam.local/sub", "kind": "rtsp", "enabled": 1},
        {"id": "off_cam", "source": "rtsp://cam.local/off", "kind": "rtsp", "enabled": 0},
    ]


# --- config generation ----------------------------------------------------


def test_config_layout(config: ArgusConfig):
    rendered = build_config(_cameras(), config)
    assert rendered["api"]["listen"] == f"127.0.0.1:{config.gateway.api_port}"
    assert rendered["rtsp"]["listen"] == f"127.0.0.1:{config.gateway.rtsp_port}"
    assert rendered["webrtc"]["listen"] == f":{config.gateway.webrtc_port}"
    assert rendered["log"]["level"] == config.gateway.log_level


def test_the_gateway_api_is_loopback_only_with_no_origin_wildcard(config: ArgusConfig):
    """Phase 5 closed the second door.

    go2rtc's HTTP API used to bind the LAN with `origin: "*"` so the SPA — on a
    different port — could sign WebRTC to it. That made snapshots of every
    camera, the stream table and go2rtc's own admin UI reachable by anyone who
    could route to the box. The console now goes through `/api/gateway/*` on the
    authenticated API, which is same-origin from go2rtc's point of view, so both
    can go: the bind is loopback and the wildcard is gone.
    """
    api = build_config(_cameras(), config)["api"]
    assert api["listen"].startswith("127.0.0.1:")
    assert "origin" not in api


def test_the_webrtc_media_port_is_still_on_the_lan(config: ArgusConfig):
    """And this is the honest part: ICE connects the browser straight to
    `webrtc.listen`, so relaying signalling does not move the media. Asserted so
    that nobody reads the test above as "the gateway is off the network".
    See docs/SECURITY.md."""
    rendered = build_config(_cameras(), config)
    assert rendered["webrtc"]["listen"] == f":{config.gateway.webrtc_port}"


def test_no_webrtc_candidates_are_emitted_when_none_are_configured(config: ArgusConfig):
    """The default site names nothing and the key is absent entirely, not
    present-and-empty: go2rtc has been parsing this file since Phase 1 and an
    unconfigured appliance should still get the file it always got."""
    assert config.gateway.webrtc_candidates == []
    assert build_config(_cameras(), config)["webrtc"] == {
        "listen": f":{config.gateway.webrtc_port}"
    }


def test_configured_webrtc_candidates_reach_go2rtc(config: ArgusConfig):
    """`webrtc.candidates` is go2rtc 1.9.14's own key for "advertise these ICE
    host candidates as well as the ones you discovered".

    Verified live on this appliance, not just here: with the entry below in
    place go2rtc trickles `candidate:... 100.113.208.36 8555 typ host` (UDP and
    TCP) *ahead* of the addresses it gathers for itself, through the
    authenticated relay at /api/gateway/ws. Which matters at a site whose
    remote-access address is not one of this box's own interfaces — a
    port-forwarded public IP, or a VPN that terminates elsewhere — because
    there is nothing there for go2rtc to discover.
    """
    config.gateway.webrtc_candidates = ["100.113.208.36:8555"]
    webrtc = build_config(_cameras(), config)["webrtc"]
    assert webrtc["candidates"] == ["100.113.208.36:8555"]
    # The listen line is untouched: this advertises an address, it does not
    # bind a new one.
    assert webrtc["listen"] == f":{config.gateway.webrtc_port}"


def test_webrtc_candidates_survive_the_yaml_go2rtc_actually_reads(config: ArgusConfig):
    config.gateway.webrtc_candidates = ["100.113.208.36:8555", "vpn.plant.example:8555/tcp"]
    written = yaml.safe_load(write_config(_cameras(), config).read_text())
    assert written["webrtc"]["candidates"] == [
        "100.113.208.36:8555", "vpn.plant.example:8555/tcp",
    ]


def test_disabled_cameras_are_not_published(config: ArgusConfig):
    streams = build_config(_cameras(), config)["streams"]
    assert set(streams) == {"file_cam", "rtsp_cam"}


def test_file_camera_is_looped_forever(config: ArgusConfig):
    """A recording has to behave like a camera. go2rtc's own `#input=file`
    template plays a file exactly once, so we drive ffmpeg ourselves."""
    entry = build_config(_cameras(), config)["streams"]["file_cam"][0]
    assert entry.startswith("exec:")
    assert "-stream_loop -1" in entry
    assert "-re" in entry
    assert "{output}" in entry
    assert "-c:v copy" in entry          # no re-encode: leaves the single NVENC free
    assert "/videos/demo.mp4" in entry


def test_file_camera_publishes_over_tcp(config: ArgusConfig):
    """Publishing to go2rtc's internal RTSP defaults to UDP, which it rejects
    with `461 Unsupported transport`."""
    entry = build_config(_cameras(), config)["streams"]["file_cam"][0]
    assert "-rtsp_transport tcp" in entry
    assert entry.index("-rtsp_transport tcp") > entry.index("-f rtsp")


def test_file_camera_regenerates_timestamps(config: ArgusConfig):
    """-stream_loop restarts the file's PTS at each seam; without +genpts the
    discontinuity makes the decoder's fps filter duplicate frames in bursts."""
    assert "-fflags +genpts" in build_config(_cameras(), config)["streams"]["file_cam"][0]


def test_rtsp_camera_is_handed_to_go2rtc_directly(config: ArgusConfig):
    """No ffmpeg in the path for real cameras — go2rtc has its own RTSP client."""
    entry = build_config(_cameras(), config)["streams"]["rtsp_cam"][0]
    assert entry == "rtsp://cam.local/sub"


def test_every_camera_gets_an_h264_fallback_source(config: ArgusConfig):
    """The second source is what a browser without H.265 actually gets.

    go2rtc uses the first source whose codec the consumer accepts, so an H.264
    camera never reaches this one and an H.265 camera has nothing else to reach.
    Without it go2rtc answers `codecs not matched: video:H265 => video:H264` and
    the tile is black — measured against a customer's five HEVC cameras.
    """
    rendered = build_config(_cameras(), config)
    streams = rendered["streams"]
    for name in ("file_cam", "rtsp_cam"):
        assert len(streams[name]) == 2
        assert streams[name][1] == f"ffmpeg:{name}#video=h264/nvenc"
    # ...and the template it names has to be in the file, or go2rtc silently
    # falls back to an x264 this appliance's LGPL ffmpeg does not have.
    assert rendered["ffmpeg"]["h264/nvenc"].startswith("-c:v h264_nvenc")


def test_the_fallback_names_its_encoder_rather_than_asking_for_hardware(
    config: ArgusConfig,
):
    """go2rtc 1.9.14 probes only v4l2m2m and rkmpp on linux/arm64 — the SBC
    encoders — so `#hardware` finds nothing on a DGX Spark, falls through to
    software libx264, and the transcode dies with `Unknown encoder 'libx264'`.
    Measured 2026-08-25, in that order."""
    rendered = build_config(_cameras(), config)
    for sources in rendered["streams"].values():
        assert "#hardware" not in sources[1]
    args = rendered["ffmpeg"]["h264/nvenc"]
    assert "-bf 0" in args            # B-frames cost a frame of latency
    assert "-pix_fmt" not in args     # inserts a CPU scaler on a hw chain
    # The one every go2rtc transcode example on the internet carries, because
    # every one of them is about a 1080p camera: H.264 level 4.1 cannot encode
    # 2560x1440 and NVENC refuses to open at all.
    assert "-level" not in args


def test_the_fallback_never_comes_first(config: ArgusConfig):
    """Order is the whole mechanism: put the transcode first and every camera,
    H.264 included, would be re-encoded for every viewer."""
    for sources in build_config(_cameras(), config)["streams"].values():
        assert not sources[0].startswith("ffmpeg:") or "#video=copy" in sources[0]
        assert "#video=h264" in sources[1]


def test_two_cameras_on_one_recording_get_distinct_sources(config: ArgusConfig):
    """The demo fleet points two cameras at the same file. go2rtc keys producers
    by source string, so identical commands make the second stream 404 and
    corrupt the first — measured, not theoretical."""
    cameras = [
        {"id": "cam_a", "source": "/videos/demo.mp4", "kind": "file", "enabled": 1},
        {"id": "cam_b", "source": "/videos/demo.mp4", "kind": "file", "enabled": 1},
    ]
    streams = build_config(cameras, config)["streams"]
    assert streams["cam_a"] != streams["cam_b"]
    assert "-metadata title=cam_a" in streams["cam_a"][0]
    assert "-metadata title=cam_b" in streams["cam_b"][0]


def test_duplicate_rtsp_urls_are_rejected(config: ArgusConfig):
    """Two cameras on one URL cannot both be served; fail at config generation
    rather than leaving an operator to decode a decoder error counter."""
    cameras = [
        {"id": "cam_a", "source": "rtsp://cam.local/sub", "kind": "rtsp", "enabled": 1},
        {"id": "cam_b", "source": "rtsp://cam.local/sub", "kind": "rtsp", "enabled": 1},
    ]
    with pytest.raises(GatewayConfigError, match="same gateway source"):
        build_config(cameras, config)


def test_stream_names_match_camera_ids():
    assert stream_name("packing_line_demo") == "packing_line_demo"


def test_write_config_emits_loadable_yaml_with_a_warning(config: ArgusConfig):
    path = write_config(_cameras(), config)
    assert path == config.gateway_dir / "go2rtc.yaml"
    text = path.read_text()
    assert "GENERATED BY ARGUS" in text
    assert set(yaml.safe_load(text)["streams"]) == {"file_cam", "rtsp_cam"}


def test_config_is_generated_from_the_database(config: ArgusConfig, database: Database):
    database.upsert_camera(Camera(id="cam_a", name="A", source="/videos/a.mp4"))
    database.upsert_camera(Camera(id="cam_b", name="B", source="rtsp://x/y"))
    database.upsert_camera(Camera(id="cam_c", name="C", source="/videos/c.mp4", enabled=False))

    streams = build_config(database.list_cameras(), config)["streams"]
    assert set(streams) == {"cam_a", "cam_b"}
    assert streams["cam_a"][0].startswith("exec:")
    assert streams["cam_b"][0] == "rtsp://x/y"


def test_urls_for_the_ui_and_decoders(config: ArgusConfig):
    gateway = Gateway(config, _cameras())
    assert gateway.rtsp_url("file_cam") == f"rtsp://127.0.0.1:{config.gateway.rtsp_port}/file_cam"
    assert gateway.webrtc_url("file_cam").startswith(f"http://127.0.0.1:{config.gateway.api_port}")
    assert "src=file_cam" in gateway.webrtc_url("file_cam")


def test_status_reports_missing_binary(config: ArgusConfig, tmp_path: Path):
    config.gateway.binary = str(tmp_path / "absent" / "go2rtc")
    gateway = Gateway(config, _cameras())
    assert gateway.available() is False
    assert gateway.version() is None
    assert gateway.status()["available"] is False
    with pytest.raises(FileNotFoundError, match="fetch_gateway"):
        gateway.start()


# --- credentials ----------------------------------------------------------
#
# The gateway is the one place a camera password *must* go: go2rtc cannot open
# the camera without it, so it is in the generated config. Everything else the
# gateway produces for a person to read goes through argus/redact.py.

NVR = "rtsp://test:hunter2-pw@192.0.2.10:554/Streaming/Channels/101"
NVR_MASKED = "rtsp://test:***@192.0.2.10:554/Streaming/Channels/101"


def _nvr_cameras() -> list[dict]:
    return [{"id": "nvr_101", "source": NVR, "kind": "rtsp", "enabled": 1}]


def _fake_go2rtc(tmp_path: Path, body: str) -> Path:
    """A stand-in binary: the supervisor only needs something that writes lines
    and exits, and the real one cannot be made to drop a camera on demand."""
    script = tmp_path / "go2rtc"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(0o755)
    return script


def _wait_for(condition, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_the_generated_config_still_carries_the_real_password(config: ArgusConfig):
    """Redaction is for readers. go2rtc has to dial the camera."""
    assert build_config(_nvr_cameras(), config)["streams"]["nvr_101"][0] == NVR


def test_a_config_error_does_not_quote_a_camera_password(config: ArgusConfig):
    """Two cameras on one URL stops `argus run` with a message that names the
    URL — on the terminal and in the journal."""
    twice = _nvr_cameras() + [{"id": "nvr_dup", "source": NVR, "kind": "rtsp", "enabled": 1}]
    with pytest.raises(GatewayConfigError) as failure:
        build_config(twice, config)
    assert "hunter2-pw" not in str(failure.value)
    assert NVR_MASKED in str(failure.value)


def test_go2rtc_log_lines_are_written_without_camera_passwords(config: ArgusConfig, tmp_path: Path):
    """go2rtc reports every dropped camera as `WRN ... error=EOF url=<source>`,
    and it used to own the log file. Measured on a customer feed: 198 such lines
    in one day's `go2rtc.log`, each with the NVR password in it."""
    config.gateway.binary = str(_fake_go2rtc(
        tmp_path,
        f"echo '23:14:46.990 WRN streams/producer.go:170 > error=EOF url={NVR}'\n"
        "echo 'second line' >&2\n"
        "exec sleep 30\n",
    ))
    gateway = Gateway(config, _nvr_cameras())
    gateway.start()
    try:
        assert _wait_for(lambda: gateway.log_path.exists()
                         and "second line" in gateway.log_path.read_text())
    finally:
        gateway.stop()
    log = gateway.log_path.read_text()
    assert "hunter2-pw" not in log
    assert f"error=EOF url={NVR_MASKED}" in log
    assert "--- argus start" in log


def test_passwords_already_in_the_logs_are_masked_when_the_gateway_starts(
    config: ArgusConfig, tmp_path: Path
):
    """Everything go2rtc wrote before its output was redacted is still on disk,
    in the current file and the rotated ones. Cleaned in place — same inode, so
    rotation and anybody tailing the file are undisturbed."""
    old = f"22:08:26.556 WRN producer.go:170 > error=EOF url={NVR}\n22:08:27 INF kept as it was\n"
    current, rotated = config.logs_dir / "go2rtc.log", config.logs_dir / "go2rtc.log.1"
    current.write_text(old)
    rotated.write_text(old * 3)
    other = config.logs_dir / "vlm-server.log"
    other.write_text("not the gateway's file\n")
    inode = current.stat().st_ino

    config.gateway.binary = str(_fake_go2rtc(tmp_path, "echo started\nexec sleep 30\n"))
    gateway = Gateway(config, _nvr_cameras())
    gateway.start()
    try:
        assert _wait_for(lambda: "started" in current.read_text())
    finally:
        gateway.stop()

    for path in (current, rotated):
        assert "hunter2-pw" not in path.read_text()
        assert f"url={NVR_MASKED}" in path.read_text()
        assert "INF kept as it was" in path.read_text()
    assert current.stat().st_ino == inode
    assert other.read_text() == "not the gateway's file\n"


def test_a_child_holding_the_log_pipe_does_not_stop_the_gateway_restarting(
    config: ArgusConfig, tmp_path: Path
):
    """The price of reading go2rtc's output is a pipe, and its ffmpeg children
    inherit the other end. One that outlives a crashed go2rtc keeps it open, so
    the supervisor waits on the process and never on end-of-file."""
    config.gateway.restart_backoff_sec = 0.05
    config.gateway.binary = str(_fake_go2rtc(tmp_path, "sleep 5 &\necho crashed\nexit 1\n"))
    gateway = Gateway(config, _nvr_cameras())
    gateway.start()
    try:
        assert _wait_for(lambda: gateway.restarts >= 2, timeout=4.0), gateway.restarts
    finally:
        gateway.stop(timeout=1.0)


# --- live process ---------------------------------------------------------


@pytest.fixture
def installed_gateway(config: ArgusConfig) -> Path:
    binary = ArgusConfig().gateway_binary          # bin/go2rtc at the repo root
    if not binary.exists():
        pytest.skip("go2rtc not installed — run scripts/fetch_gateway.sh")
    config.gateway.binary = str(binary)
    return binary


@pytest.mark.slow
def test_version_reports_go2rtc(config: ArgusConfig, installed_gateway: Path):
    version = Gateway(config, []).version()
    assert version and "go2rtc" in version.lower()


@pytest.mark.slow
def test_gateway_serves_a_looped_file_over_rtsp(
    config: ArgusConfig, database: Database, installed_gateway: Path, sample_video: Path, ffmpeg: str
):
    """End-to-end video plane: cameras table -> go2rtc.yaml -> running gateway ->
    decoder pulling BGR frames back out of its RTSP endpoint."""
    database.upsert_camera(Camera(id="demo_cam", name="Demo", source=str(sample_video)))
    gateway = Gateway(config, database.list_cameras(enabled_only=True))
    gateway.start()
    try:
        assert gateway.wait_ready(timeout=30), "gateway API never answered"
        assert gateway.is_running()
        assert "demo_cam" in gateway.streams()

        source = FrameSource(
            "demo_cam", gateway.rtsp_url("demo_cam"), fps=5, width=320, height=192,
            start_timeout_sec=30.0,
        )
        frames = list(itertools.islice(source.frames(), 5))
        source.stop()

        assert len(frames) == 5
        assert frames[0].image.shape == (192, 320, 3)
        assert frames[0].image.std() > 1.0
        assert source.stats.restarts == 0
    finally:
        gateway.stop()

    assert not gateway.is_running()


@pytest.mark.slow
def test_two_cameras_share_one_recording_concurrently(
    config: ArgusConfig, database: Database, installed_gateway: Path, sample_video: Path, ffmpeg: str
):
    """Regression: the demo fleet is two cameras on one file. With identical
    exec commands go2rtc served one and 404'd the other, and the survivor's
    frame rate blew up to ~1800 fps as the failing decoder retried into the
    shared session. Both must decode cleanly, side by side."""
    database.upsert_camera(Camera(id="cam_a", name="A", source=str(sample_video)))
    database.upsert_camera(Camera(id="cam_b", name="B", source=str(sample_video)))
    gateway = Gateway(config, database.list_cameras(enabled_only=True))
    gateway.start()
    workers = []
    try:
        assert gateway.wait_ready(timeout=30)
        for camera_id in ("cam_a", "cam_b"):
            source = FrameSource(camera_id, gateway.rtsp_url(camera_id), fps=5,
                                 width=320, height=192, start_timeout_sec=30.0)
            workers.append(FrameSourceThread(source).start())

        deadline = time.time() + 45
        while time.time() < deadline and any(w.stats.frames < 15 for w in workers):
            time.sleep(0.2)

        for worker in workers:
            stats = worker.stats
            assert stats.frames >= 15, f"{stats.camera_id} only got {stats.frames} frames"
            assert stats.restarts == 0, f"{stats.camera_id} restarted: {stats.last_error}"
            # The corruption showed up as a wildly inflated rate, not as an error.
            assert 2.0 < stats.fps_avg < 15.0, f"{stats.camera_id} fps_avg={stats.fps_avg}"
    finally:
        for worker in workers:
            worker.stop(timeout=10)
        gateway.stop()


@pytest.mark.slow
def test_gateway_restarts_after_a_crash(config: ArgusConfig, installed_gateway: Path):
    """A gateway that dies must come back on its own — it is the single point of
    contact for every camera."""
    gateway = Gateway(config, [])
    gateway.start()
    try:
        assert gateway.wait_ready(timeout=30)
        gateway._proc.kill()                       # simulate a crash

        deadline = time.time() + 30
        while gateway.restarts == 0 and time.time() < deadline:
            time.sleep(0.2)
        assert gateway.restarts >= 1
        assert gateway.wait_ready(timeout=30)
        assert gateway.is_running()
    finally:
        gateway.stop()
