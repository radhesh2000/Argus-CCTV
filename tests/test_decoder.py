"""Decoder tests.

Kept fast by decoding with ``realtime=False`` (no ``-re``), so ffmpeg runs at
full speed and a handful of frames arrive in well under a second. The NVDEC path
is exercised only when ffmpeg advertises it; the fallback that matters on a
plain ffmpeg build is tested unconditionally.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pytest

from argus.config import ArgusConfig
from argus.ingest.decoder import (
    DECODE_LADDER,
    FrameSource,
    FrameSourceThread,
    hwaccel_available,
    is_hardware_mode,
    probe_source,
    source_for_camera,
)

WIDTH, HEIGHT = 320, 192          # small: the shape assertion is the point, not the pixels


def _source(video: Path, **overrides) -> FrameSource:
    kwargs = dict(
        camera_id="test_cam", source=str(video), fps=5, width=WIDTH, height=HEIGHT,
        realtime=False, start_timeout_sec=30.0,
    )
    kwargs.update(overrides)
    return FrameSource(**kwargs)


# --- command construction (no subprocess) ---------------------------------


def test_file_source_is_looped_and_paced():
    src = FrameSource("cam", "/videos/a.mp4", fps=5, width=960, height=544)
    cmd = src.build_command(mode="cpu")
    assert src.kind == "file" and src.loop is True
    assert "-stream_loop" in cmd and cmd[cmd.index("-stream_loop") + 1] == "-1"
    assert "-re" in cmd                                   # real-time pacing
    assert "fps=5,scale=960:544" in cmd
    assert cmd[-3:] == ["-f", "rawvideo", "-"]
    assert "-pix_fmt" in cmd and cmd[cmd.index("-pix_fmt") + 1] == "bgr24"


def test_rtsp_source_uses_tcp_and_is_not_looped():
    src = FrameSource("cam", "rtsp://cam.local/sub", fps=5)
    cmd = src.build_command(mode="cpu")
    assert src.kind == "rtsp" and src.loop is False
    assert "-stream_loop" not in cmd and "-re" not in cmd
    assert cmd[cmd.index("-rtsp_transport") + 1] == "tcp"


def test_the_best_rung_keeps_frames_on_the_device_until_after_fps():
    """The whole point of the top rung: `fps` runs before `hwdownload`, so four
    frames in five are dropped on the GPU and never crossed to system memory.
    Measured 10.5x on 1080p; see docs/BENCHMARKS.md."""
    src = FrameSource("cam", "rtsp://cam.local/sub", fps=5, width=960, height=544)
    cmd = src.build_command(mode="nvdec")
    assert cmd[cmd.index("-hwaccel") + 1] == "cuda"
    assert cmd[cmd.index("-hwaccel_output_format") + 1] == "cuda"
    assert cmd.index("-hwaccel") < cmd.index("-i")
    filters = cmd[cmd.index("-vf") + 1]
    assert filters == "fps=5,hwdownload,format=nv12,scale=960:544"
    assert filters.index("fps=5") < filters.index("hwdownload")


def test_the_middle_rung_is_the_old_chain_and_the_bottom_one_has_no_hwaccel():
    src = FrameSource("cam", "rtsp://cam.local/sub", fps=5, width=960, height=544)
    middle = src.build_command(mode="nvdec-download")
    assert middle[middle.index("-hwaccel") + 1] == "cuda"
    assert "-hwaccel_output_format" not in middle
    assert middle[middle.index("-vf") + 1] == "fps=5,scale=960:544"

    cpu = src.build_command(mode="cpu")
    assert "-hwaccel" not in cpu
    assert cpu[cpu.index("-vf") + 1] == "fps=5,scale=960:544"


def test_frame_bytes_matches_geometry():
    src = FrameSource("cam", "/a.mp4", width=WIDTH, height=HEIGHT)
    assert src.frame_bytes == WIDTH * HEIGHT * 3


def test_loop_can_be_forced_off():
    src = FrameSource("cam", "/a.mp4", loop=False)
    assert "-stream_loop" not in src.build_command(mode="cpu")


# --- real decode ----------------------------------------------------------


@pytest.mark.slow
def test_frames_arrive_with_the_expected_shape(sample_video: Path, ffmpeg: str):
    src = _source(sample_video)
    frames = list(itertools.islice(src.frames(), 10))
    src.stop()

    assert len(frames) == 10
    for index, frame in enumerate(frames, start=1):
        assert frame.image.shape == (HEIGHT, WIDTH, 3)
        assert frame.image.dtype == np.uint8
        assert frame.camera_id == "test_cam"
        assert frame.index == index
        assert frame.ts > 0

    # Real video, not a black or constant buffer.
    assert frames[0].image.std() > 1.0
    assert not np.array_equal(frames[0].image, frames[-1].image)


@pytest.mark.slow
def test_achieved_fps_stats_are_reported(sample_video: Path, ffmpeg: str):
    src = _source(sample_video)
    for _ in itertools.islice(src.frames(), 12):
        pass
    src.stop()

    stats = src.stats
    assert stats.frames == 12
    assert stats.mode in DECODE_LADDER
    assert stats.fps > 0 and stats.fps_avg > 0
    assert stats.last_frame_at is not None
    assert stats.as_dict()["camera_id"] == "test_cam"


def test_one_close_pair_of_frames_does_not_blow_up_the_reported_fps():
    """A steady camera must not report 133 fps because two frames touched.

    Measured on the 2026-08-25 soak: `anmol_packing_7` advanced its frame
    counter by a rock-steady 70-81 frames every 15 s for 45 minutes and told the
    console **133.39, 126.48 and 96.11 fps** on eight samples, because the stat
    averaged `1/delta` and one 7 ms gap is 143 of those. Smoothing the gap and
    reporting its reciprocal is bounded; smoothing the rate is not.
    """
    src = _source(Path("/nonexistent.mp4"))     # no decoding here: stats arithmetic only
    now = 1000.0
    for _ in range(40):                          # settle at a clean 5 fps
        src._record_frame(now)
        now += 0.2
    steady = src.stats.fps
    assert 4.8 < steady < 5.2

    src._record_frame(now)                       # ...and now one 7 ms pair
    now += 0.007
    src._record_frame(now)
    assert src.stats.fps < 6.0, f"one close pair reported {src.stats.fps:.1f} fps"

    for _ in range(20):                          # and it comes back on its own
        now += 0.2
        src._record_frame(now)
    assert 4.8 < src.stats.fps < 5.2


def test_a_camera_that_really_slows_down_still_shows_it():
    """The other half: smoothing must not hide a stalling camera."""
    src = _source(Path("/nonexistent.mp4"))
    now = 1000.0
    for _ in range(40):
        src._record_frame(now)
        now += 0.2
    for _ in range(40):
        now += 1.0                               # 1 fps
        src._record_frame(now)
    assert src.stats.fps < 2.0


@pytest.mark.slow
def test_decoder_falls_back_to_cpu_when_hwaccel_is_unusable(sample_video: Path, ffmpeg: str):
    """The appliance must keep running on an ffmpeg without CUDA. Asking for
    hwaccel always yields frames — via NVDEC where available, software where not."""
    src = _source(sample_video, hwaccel=True)
    frames = list(itertools.islice(src.frames(), 3))
    src.stop()

    assert len(frames) == 3
    if hwaccel_available(ffmpeg):
        assert is_hardware_mode(src.stats.mode)
    else:
        assert src.stats.mode == "cpu"


@pytest.mark.slow
def test_realtime_pacing_respects_the_configured_fps(sample_video: Path, ffmpeg: str):
    """With -re, 6 frames at 5 fps take about a second — proof the fps filter,
    not the reader, is setting the cadence."""
    src = _source(sample_video, realtime=True, fps=5)
    started = time.time()
    frames = list(itertools.islice(src.frames(), 6))
    elapsed = time.time() - started
    src.stop()

    assert len(frames) == 6
    assert 0.5 < elapsed < 6.0
    assert src.stats.fps_avg == pytest.approx(5, abs=3)


@pytest.mark.slow
def test_file_source_loops_past_the_end_of_the_video(sample_video: Path, ffmpeg: str):
    """The 62s demo file has to behave like a camera that never stops. At 5 fps
    a single pass is ~310 frames, so 340 can only come from a loop."""
    src = _source(sample_video, fps=5)
    frames = list(itertools.islice(src.frames(), 340))
    src.stop()

    assert len(frames) == 340
    assert src.stats.restarts == 0          # looped inside one ffmpeg, not restarted
    assert frames[-1].image.shape == (HEIGHT, WIDTH, 3)


@pytest.mark.slow
def test_dead_source_restarts_with_backoff_then_gives_up(tmp_path: Path, ffmpeg: str):
    """A source that never works must not spin: bounded restarts, and the error
    is captured for the camera-status display."""
    src = FrameSource(
        "broken", str(tmp_path / "missing.mp4"), fps=5, width=WIDTH, height=HEIGHT,
        realtime=False, backoff_sec=0.05, backoff_max_sec=0.1, max_restarts=2,
        start_timeout_sec=5.0,
    )
    started = time.time()
    frames = list(src.frames())
    elapsed = time.time() - started

    assert frames == []
    assert src.stats.restarts == 2
    assert elapsed < 10
    assert src.stats.last_error
    assert not src.stats.running


@pytest.mark.slow
def test_stop_ends_the_iterator_and_reaps_ffmpeg(sample_video: Path, ffmpeg: str):
    src = _source(sample_video, realtime=True)
    iterator = src.frames()
    next(iterator)
    src.stop()

    assert list(iterator) == []
    assert src._proc is None


# --- threaded wrapper -----------------------------------------------------


@pytest.mark.slow
def test_frame_source_thread_tracks_latest_frame(sample_video: Path, ffmpeg: str):
    seen: list[int] = []
    worker = FrameSourceThread(_source(sample_video), on_frame=lambda f: seen.append(f.index))
    worker.start()
    deadline = time.time() + 20
    while worker.stats.frames < 5 and time.time() < deadline:
        time.sleep(0.05)
    latest = worker.latest()
    worker.stop(timeout=10)

    assert latest is not None
    assert latest.image.shape == (HEIGHT, WIDTH, 3)
    assert len(seen) >= 5
    assert not worker.is_alive()


# --- the decode ladder ----------------------------------------------------
#
# The narrative these pin is in docs/BENCHMARKS.md: at 80+ cameras every decoder
# silently reverted to software, because a hwaccel attempt that produced no
# frame during a cold-start stampede was taken as proof the build had no NVDEC,
# and that verdict latched for the life of the process.


def _hw_source(**overrides) -> FrameSource:
    """A source with the ladder forced open, so no ffmpeg is ever spawned."""
    src = FrameSource("cam", "rtsp://cam.local/sub", **overrides)
    src._ladder = DECODE_LADDER
    return src


def test_a_build_without_cuda_never_attempts_hardware():
    """Known before a single process is spawned — `hwaccel_available` answers it
    — so a plain ffmpeg build costs no failed starts at all."""
    src = FrameSource("cam", "rtsp://cam.local/sub", hwaccel=False)
    assert src.ladder() == ("cpu",)
    assert src.current_mode() == "cpu"


def test_a_refused_chain_is_disabled_but_the_next_one_is_tried():
    src = _hw_source()
    assert src.current_mode() == "nvdec"
    src._demote("nvdec", starved=False, now=1000.0)
    assert src.current_mode(now=1000.0) == "nvdec-download"
    src._demote("nvdec-download", starved=False, now=1000.0)
    assert src.current_mode(now=1_000_000.0) == "cpu"      # permanent: no re-arm
    assert src.promotion_due(now=1_000_000.0) is False


def test_a_starved_chain_goes_straight_to_software_and_climbs_back():
    """A stampede starves the *download* rung harder than the one above it, so
    demoting one step would just buy another start_timeout to fail the same
    way. Software now, hardware again in `hwaccel_retry_sec`."""
    src = _hw_source(hwaccel_retry_sec=120.0)
    src._demote("nvdec", starved=True, now=1000.0)

    assert src.current_mode(now=1000.0) == "cpu"
    assert src.promotion_due(now=1000.0 + 119) is False
    assert src.promotion_due(now=1000.0 + 121) is True
    assert src.current_mode(now=1000.0 + 121) == "nvdec"


def test_repeated_starvation_backs_off_instead_of_flapping():
    src = _hw_source(hwaccel_retry_sec=100.0, hwaccel_retry_max_sec=400.0)
    for expected in (100.0, 200.0, 400.0, 400.0):
        src._rung = 0
        src._demote("nvdec", starved=True, now=1000.0)
        assert src.promotion_due(now=1000.0 + expected - 1) is False
        assert src.promotion_due(now=1000.0 + expected + 1) is True


@pytest.mark.slow
def test_a_good_run_on_the_best_rung_forgets_the_starvation_history(
    sample_video: Path, ffmpeg: str
):
    """Otherwise the backoff only ever grows: a box that had one bad start in
    January would be waiting an hour between hardware attempts by March."""
    src = _source(sample_video, hwaccel=True)
    if not hwaccel_available(ffmpeg):
        pytest.skip("no cuda hwaccel in this ffmpeg")
    src._demote("nvdec", starved=True)
    assert src._starved == 1 and src.current_mode() == "cpu"
    src._retry_best_at = time.time() - 1              # the climb back has come due

    for _ in itertools.islice(src.frames(), 3):
        pass
    src.stop()

    assert src.stats.mode == "nvdec"
    assert src._starved == 0 and src._retry_best_at == 0.0


def test_a_disabled_rung_is_skipped_when_climbing_back():
    src = _hw_source(hwaccel_retry_sec=100.0)
    src._demote("nvdec", starved=False, now=1000.0)   # nvdec refused outright
    src._demote("nvdec-download", starved=True, now=1000.0)
    assert src.current_mode(now=1000.0) == "cpu"
    # The re-arm must not send it back to a chain ffmpeg has already refused.
    assert src.current_mode(now=1200.0) == "nvdec-download"


# --- config wiring --------------------------------------------------------


def test_start_timeout_grows_with_the_fleet():
    """20s is right for one camera and wrong for eighty: a cold start of 80
    cameras is ~240 ffmpeg spawns, and none of them delivers a first frame
    inside a single-camera deadline."""
    config = ArgusConfig()
    assert config.detection.start_timeout_for(1) == pytest.approx(20.25)
    assert config.detection.start_timeout_for(96) == pytest.approx(44.0)
    # Capped, so a mistyped fleet size cannot make a dead camera take an hour
    # to be reported as dead.
    assert config.detection.start_timeout_for(100_000) == config.detection.start_timeout_max_sec


def test_source_for_camera_carries_the_fleet_timeout():
    config = ArgusConfig()
    camera = {"id": "cam1", "source": "/videos/a.mp4", "detect_fps": None}
    assert source_for_camera(camera, config).start_timeout_sec == 20.0
    assert source_for_camera(camera, config, start_timeout_sec=44.0).start_timeout_sec == 44.0


def test_source_for_camera_uses_config_geometry():
    config = ArgusConfig()
    config.detection.fps = 7
    config.detection.width, config.detection.height = 640, 384
    camera = {"id": "cam1", "source": "/videos/a.mp4", "detect_fps": None}

    src = source_for_camera(camera, config)
    assert (src.fps, src.width, src.height) == (7, 640, 384)
    assert src.source == "/videos/a.mp4"


def test_per_camera_fps_override_wins():
    config = ArgusConfig()
    config.detection.fps = 5
    src = source_for_camera({"id": "c", "source": "/a.mp4", "detect_fps": 2}, config)
    assert src.fps == 2


def test_override_source_points_at_the_gateway():
    config = ArgusConfig()
    src = source_for_camera(
        {"id": "cam1", "source": "/videos/a.mp4", "detect_fps": None},
        config,
        override_source=config.rtsp_url("cam1"),
    )
    assert src.source.startswith("rtsp://")
    assert src.kind == "rtsp"


# --- probe ----------------------------------------------------------------


def test_probe_source_reads_video_metadata(sample_video: Path):
    info = probe_source(str(sample_video))
    assert int(info["width"]) == 1024
    assert int(info["height"]) == 576
    assert info["codec_name"] == "h264"


def test_probe_source_raises_on_a_bad_path(tmp_path: Path):
    with pytest.raises(RuntimeError, match="ffprobe failed"):
        probe_source(str(tmp_path / "nope.mp4"))


def _refused_camera() -> str:
    """An RTSP URL with a password, at a port nothing is listening on."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"rtsp://test:hunter2-pw@127.0.0.1:{port}/Streaming/Channels/101"


def test_probe_source_does_not_quote_the_password_back():
    """ffprobe reports a camera it cannot open by printing the URL it was given,
    credentials included. This message is the enrolment form's "Test
    connection" answer, so it is redacted — and before it is cut to length."""
    url = _refused_camera()
    with pytest.raises(RuntimeError, match="ffprobe failed") as failure:
        probe_source(url, timeout=5)
    assert "hunter2-pw" not in str(failure.value)
    assert "rtsp://test:***@127.0.0.1" in str(failure.value)


def test_a_decoder_error_does_not_carry_the_camera_password():
    """`last_error` is ffmpeg's stderr. It becomes the camera's `status_detail`
    in the database, a field of /api/health and /api/live/state, and a line of
    `argus run` output — for a camera decoded directly, with its password."""
    src = FrameSource(
        "nvr", _refused_camera(), fps=5, width=WIDTH, height=HEIGHT,
        backoff_sec=0.05, backoff_max_sec=0.1, max_restarts=1, start_timeout_sec=5.0,
    )
    assert list(src.frames()) == []
    assert src.stats.last_error and "hunter2-pw" not in src.stats.last_error
    assert "test:***@127.0.0.1" in src.stats.last_error


def test_hwaccel_probe_returns_a_bool(ffmpeg: str):
    assert isinstance(hwaccel_available(ffmpeg), bool)
    assert hwaccel_available("/nonexistent/ffmpeg") is False
