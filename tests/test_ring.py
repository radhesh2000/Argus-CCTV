"""Ring buffer: continuous recording that a clip can be cut out of."""

from __future__ import annotations

import subprocess
import time

import pytest

from argus.media.ring import (
    RingRecorder,
    Segment,
    clip_commands,
    concat_segments,
    probe_codec,
)


def _recorder(tmp_path, source="rtsp://camera/stream", **kwargs) -> RingRecorder:
    return RingRecorder("cam", source, tmp_path / "ring", **kwargs)


def _touch_segments(recorder: RingRecorder, epochs) -> None:
    recorder.directory.mkdir(parents=True, exist_ok=True)
    for epoch in epochs:
        (recorder.directory / f"{int(epoch)}.mp4").write_bytes(b"x" * 16)


# --- command construction --------------------------------------------------


def test_copy_mode_never_touches_the_encoder(tmp_path):
    command = _recorder(tmp_path).build_command("copy")
    assert "-c:v" in command and command[command.index("-c:v") + 1] == "copy"
    assert not any(arg.endswith("nvenc") or arg == "libx264" for arg in command)
    assert command[command.index("-segment_time") + 1] == "2"


def test_encode_mode_falls_back_to_nvenc_not_a_gpl_encoder(tmp_path):
    """The vendored ffmpeg is built LGPL and has no libx264 — asking for it
    would fail outright — and re-encoding on the Grace cores is what this
    module exists to avoid."""
    command = _recorder(tmp_path).build_command("encode")
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert "libx264" not in command


def test_rtsp_sources_are_pulled_over_tcp(tmp_path):
    assert "-rtsp_transport" in _recorder(tmp_path).build_command("copy")


def test_file_sources_loop_with_regenerated_timestamps(tmp_path):
    command = _recorder(tmp_path, source=str(tmp_path / "demo.mp4")).build_command("copy")
    assert "-stream_loop" in command and command[command.index("-stream_loop") + 1] == "-1"
    assert "-re" in command
    # Without +genpts the loop seam reaches the segmenter as a timestamp jump.
    assert command[command.index("-fflags") + 1] == "+genpts"


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _recorder(tmp_path, mode="turbo")


# --- segment bookkeeping ---------------------------------------------------


def test_the_segment_being_written_is_excluded(tmp_path):
    recorder = _recorder(tmp_path)
    _touch_segments(recorder, [1000, 1002, 1004])
    assert [s.start for s in recorder.segments()] == [1000, 1002]
    assert [s.start for s in recorder.segments(include_active=True)] == [1000, 1002, 1004]


def test_segment_spans_come_from_the_next_segment(tmp_path):
    """Stream-copied segments split at keyframes, so their real durations vary;
    the following segment's start is the honest end of this one."""
    recorder = _recorder(tmp_path)
    _touch_segments(recorder, [1000, 1003, 1005, 1007])
    spans = [(s.start, s.end) for s in recorder.segments()]
    assert spans == [(1000, 1003), (1003, 1005), (1005, 1007)]


def test_segments_covering_selects_the_overlapping_window(tmp_path):
    recorder = _recorder(tmp_path)
    _touch_segments(recorder, range(1000, 1030, 2))
    covering = recorder.segments_covering(1009, 1015)
    assert [s.start for s in covering] == [1008, 1010, 1012, 1014]


def test_non_epoch_filenames_are_ignored(tmp_path):
    recorder = _recorder(tmp_path)
    _touch_segments(recorder, [1000, 1002, 1004])
    (recorder.directory / "notes.mp4").write_bytes(b"x")
    assert [s.start for s in recorder.segments()] == [1000, 1002]


def test_prune_drops_footage_older_than_the_window(tmp_path):
    recorder = _recorder(tmp_path, retention_sec=10)
    now = time.time()
    _touch_segments(recorder, [now - 60, now - 30, now - 5, now - 1])
    assert recorder.prune() == 2
    assert len(recorder.segments(include_active=True)) == 2
    assert recorder.stats.bytes_on_disk > 0


def test_cut_without_footage_reports_why(tmp_path):
    recorder = _recorder(tmp_path)
    assert recorder.cut(1000, 1016, tmp_path / "clip.mp4") is None
    assert "no ring segments" in (recorder.stats.last_error or "")


def test_concat_of_nothing_is_none(tmp_path):
    assert concat_segments([], tmp_path / "clip.mp4") is None


def test_concat_reports_the_window_it_covers(tmp_path, monkeypatch):
    segments = [Segment(path=tmp_path / f"{t}.mp4", start=t, end=t + 2) for t in (1000, 1002)]
    for segment in segments:
        segment.path.write_bytes(b"x")

    def fake_run(cmd, **kwargs):
        # Stand in for ffmpeg: the arguments are what we care about here.
        assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"
        (tmp_path / "clip.mp4").write_bytes(b"video")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    clip = concat_segments(segments, tmp_path / "clip.mp4")
    assert clip is not None
    assert clip.start == 1000 and clip.duration == 4
    assert not (tmp_path / "clip.mp4.concat.txt").exists()      # list file cleaned up


# --- against real ffmpeg ---------------------------------------------------


@pytest.mark.slow
def test_records_and_cuts_a_real_clip(tmp_path, sample_video, ffmpeg):
    """The whole point, proven: record a file source, then cut evidence out of it
    without re-encoding."""
    assert probe_codec(str(sample_video)) == "h264"

    recorder = RingRecorder("cam", str(sample_video), tmp_path / "ring",
                            segment_sec=2.0, retention_sec=120)
    recorder.start()
    try:
        deadline = time.time() + 40
        while time.time() < deadline and len(recorder.segments()) < 4:
            time.sleep(0.5)
        segments = recorder.segments()
        assert len(segments) >= 4, "the ring recorder produced no segments"
        assert recorder.stats.mode == "copy"    # h264 source must not be re-encoded

        middle = segments[len(segments) // 2].start
        clip = recorder.cut(middle - 2, middle + 2, tmp_path / "clips" / "event_1.mp4")
    finally:
        recorder.stop()

    assert clip is not None and clip.path.exists()
    assert clip.path.stat().st_size > 1000
    probed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(clip.path)],
        capture_output=True, text=True, check=True,
    )
    assert float(probed.stdout.strip()) > 2.0


# --- evidence clips a browser can actually play ----------------------------
#
# The gap this section guards: `_COPYABLE_CODECS` and `PLAYABLE_CODECS` are not
# the same set, and the pipeline was silently shipping the difference. An H.265
# camera recorded and remuxed perfectly, and Chromium showed a black player with
# no error anybody would see.


@pytest.mark.parametrize(
    "codec, transcode",
    [
        ("h264", False),      # the demo camera: nothing to do, and it stays free
        ("av1", False),       # dav1d ships in every Chromium
        ("hevc", True),       # the Anmol cameras — remuxes fine, will not play
        ("h265", True),       # the same codec under ffprobe's other spelling
        ("mpeg4", True),      # copyable and unplayable: why the sets differ
        (None, True),         # probe failed, so playability cannot be proven
    ],
)
def test_clip_transcode_is_decided_by_what_the_ring_holds(tmp_path, codec, transcode):
    recorder = _recorder(tmp_path)
    recorder.stats.mode = "copy"
    recorder.stats.codec = codec
    assert recorder.segment_codec() == codec
    assert recorder.clip_needs_transcode() is transcode


def test_a_reencoded_ring_is_not_reencoded_a_second_time(tmp_path):
    """`encode` mode already produced H.264, whatever the camera sends."""
    recorder = _recorder(tmp_path, mode="encode")
    recorder.stats.mode = "encode"
    recorder.stats.codec = "hevc"
    assert recorder.segment_codec() == "h264"
    assert recorder.clip_needs_transcode() is False


def test_encode_mode_does_not_probe_the_source(tmp_path, monkeypatch):
    """Nothing a probe returns could change the answer, and a probe of an
    unreachable camera costs the ffprobe timeout at every start."""
    def explode(source):  # pragma: no cover - the point is that it is not called
        raise AssertionError(f"probed {source}")

    monkeypatch.setattr("argus.media.ring.probe_codec", explode)
    assert _recorder(tmp_path, mode="encode")._decide_mode() == "encode"


def test_never_keeps_the_old_always_copy_behaviour(tmp_path):
    recorder = _recorder(tmp_path, clip_transcode="never")
    recorder.stats.mode = "copy"
    recorder.stats.codec = "hevc"
    assert recorder.clip_needs_transcode() is False


def test_unknown_clip_transcode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _recorder(tmp_path, clip_transcode="sometimes")


def test_status_reports_both_codec_facts(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.stats.mode = "copy"
    recorder.stats.codec = "hevc"
    status = recorder.status()
    assert status["codec"] == "hevc"           # what the camera sends
    assert status["segment_codec"] == "hevc"   # what is on disk
    assert status["clip_transcode"] is True    # and whether a reviewer can watch it


def test_transcoding_clip_command_keeps_the_frames_on_the_gpu(tmp_path):
    commands = clip_commands(tmp_path / "list.txt", tmp_path / "clip.mp4", transcode=True)
    assert len(commands) == 2, "hardware first, software decode as the fallback"
    hardware, software = commands

    assert hardware[hardware.index("-hwaccel") + 1] == "cuda"
    assert hardware[hardware.index("-hwaccel_output_format") + 1] == "cuda"
    assert hardware[hardware.index("-c:v") + 1] == "h264_nvenc"
    # `-pix_fmt yuv420p` on this chain inserts a CPU scaler that cannot take a
    # cuda frame, and ffmpeg dies with "Impossible to convert between the
    # formats supported by the filter". NVENC writes 8-bit 4:2:0 anyway.
    assert "-pix_fmt" not in hardware

    assert "-hwaccel" not in software
    assert software[software.index("-c:v") + 1] == "h264_nvenc"
    assert software[software.index("-pix_fmt") + 1] == "yuv420p"
    assert "libx264" not in software           # the vendored ffmpeg is LGPL


def test_copying_clip_command_is_one_remux(tmp_path):
    commands = clip_commands(tmp_path / "list.txt", tmp_path / "clip.mp4", transcode=False)
    assert len(commands) == 1
    assert commands[0][commands[0].index("-c") + 1] == "copy"
    assert not any(arg.endswith("nvenc") for arg in commands[0])


def test_concat_falls_back_when_the_hardware_chain_fails(tmp_path, monkeypatch):
    """NVDEC does not decode every codec the ring can copy — mpeg4 is the live
    example — so a failed first chain must not mean a lost clip."""
    segments = [Segment(path=tmp_path / f"{t}.mp4", start=t, end=t + 2) for t in (1000, 1002)]
    for segment in segments:
        segment.path.write_bytes(b"x")
    attempts = []

    def fake_run(cmd, **kwargs):
        attempts.append(cmd)
        if "-hwaccel" in cmd:
            raise subprocess.CalledProcessError(1, cmd, b"", b"Impossible to convert")
        (tmp_path / "clip.mp4").write_bytes(b"video")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    clip = concat_segments(segments, tmp_path / "clip.mp4", transcode=True)
    assert clip is not None and clip.start == 1000
    assert len(attempts) == 2
    assert not (tmp_path / "clip.mp4.concat.txt").exists()


@pytest.mark.slow
def test_an_hevc_camera_yields_a_clip_a_browser_can_play(tmp_path, hevc_video, ffmpeg):
    """End to end on real HEVC: record with `-c copy`, cut, get H.264 out.

    This is the defect Phase 6 found, as a test: before the transcode existed
    this produced a perfectly valid `hvc1` mp4 that Chromium refused to decode.
    """
    assert probe_codec(str(hevc_video)) == "hevc"

    recorder = RingRecorder("cam", str(hevc_video), tmp_path / "ring",
                            segment_sec=2.0, retention_sec=120)
    recorder.start()
    try:
        deadline = time.time() + 60
        while time.time() < deadline and len(recorder.segments()) < 4:
            time.sleep(0.5)
        segments = recorder.segments()
        assert len(segments) >= 4, "recorder produced no segments"
        # The ring itself still copies: re-encoding every camera continuously is
        # the v1 mistake, and it is not what this change undoes.
        assert recorder.stats.mode == "copy"
        assert recorder.segment_codec() == "hevc"
        assert recorder.clip_needs_transcode() is True

        clip = recorder.cut(segments[0].start, segments[-1].end, tmp_path / "clip.mp4")
    finally:
        recorder.stop()

    assert clip is not None and clip.path.exists()
    probe = subprocess.run(
        [ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(clip.path)],
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == "h264"



def test_the_recorders_last_error_does_not_carry_the_camera_password(tmp_path):
    """Same binary as the decoder and the same habit: ffmpeg names the URL it
    could not open. The line is kept to 200 characters, so it is redacted first
    — cutting first can end the string inside the password."""
    import io
    import types

    url = "rtsp://test:hunter2-pw@10.0.0.9:554/Streaming/Channels/101"
    recorder = _recorder(tmp_path, source=url)
    padding = "x" * (200 - len("rtsp://test:hunt"))
    recorder._drain_stderr(types.SimpleNamespace(
        stderr=io.BytesIO(f"{padding}{url}: Connection refused\n".encode())
    ))
    assert "hunt" not in recorder.stats.last_error
    assert recorder.stats.last_error.endswith("rtsp://test:***@")
    assert len(recorder.stats.last_error) == 200
