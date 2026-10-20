"""The one place a camera password is taken out of something a person reads.

Everything that shows a source — the API, the console, `argus status`, the
engine's start-up lines, go2rtc's log — goes through ``argus/redact.py``. These
are the helper's own rules; that each of those callers actually uses it is
tested where the caller is (test_api.py, test_cli.py, test_gateway.py,
test_decoder.py).
"""

from __future__ import annotations

import pytest

from argus.ingest.gateway import UnsafeSource, assert_safe_source
from argus.redact import (
    MASK,
    PasswordNotKept,
    is_masked,
    redact_camera,
    redact_source,
    redact_text,
    redact_tree,
    restore_password,
)

NVR = "rtsp://test:hunter2@127.0.0.1:6554/Streaming/Channels/101"


@pytest.mark.parametrize(("source", "shown"), [
    (NVR, "rtsp://test:***@127.0.0.1:6554/Streaming/Channels/101"),
    # The username is kept: it is how an installer tells which account is in use.
    ("http://admin:pw@10.0.0.9/mjpeg", "http://admin:***@10.0.0.9/mjpeg"),
    # The userinfo ends at the *last* @, as it does for ffmpeg and go2rtc.
    ("rtsp://u:p@ss@host:554/x", "rtsp://u:***@host:554/x"),
    ("rtsp://u:p%40ss@[fe80::1]:554/x", "rtsp://u:***@[fe80::1]:554/x"),
    # Nothing to hide, so nothing changes — a mangled source would be a new bug.
    ("rtsp://cam.local/sub", "rtsp://cam.local/sub"),
    ("rtsp://10.0.0.9:554/h264", "rtsp://10.0.0.9:554/h264"),
    ("rtsp://viewer@cam.local/sub", "rtsp://viewer@cam.local/sub"),
    ("rtsp://[::1]:554/live@2x", "rtsp://[::1]:554/live@2x"),
    ("/srv/footage/line_3.mp4", "/srv/footage/line_3.mp4"),
    ("/srv/footage/a@b:c.mp4", "/srv/footage/a@b:c.mp4"),
])
def test_a_source_is_shown_without_its_password(source: str, shown: str):
    assert redact_source(source) == shown
    assert redact_source(shown) == shown                     # idempotent


def test_a_password_with_a_raw_slash_is_still_hidden():
    """`rtsp://user:pa/ss@host` is not a URL anything can open — it parses as
    host `user`, port `pa` — but it is a real password somebody typed, and it
    is accepted and stored. Hiding too much of it is the safe direction."""
    assert redact_source("rtsp://user:pa/ss@host/path") == "rtsp://user:***@host/path"


def test_text_that_quotes_a_url_loses_the_password():
    line = f"[rtsp @ 0x55d0] {NVR}: Connection refused; see http://docs.local:8080/a for b@c"
    assert redact_text(line) == (
        "[rtsp @ 0x55d0] rtsp://test:***@127.0.0.1:6554/Streaming/Channels/101: "
        "Connection refused; see http://docs.local:8080/a for b@c"
    )
    assert redact_text(None) is None and redact_text("") == ""
    # go2rtc's own warning, which is what filled data/logs/go2rtc.log.
    assert "hunter2" not in redact_text(f"WRN streams/producer.go:170 > error=EOF url={NVR}")


def test_a_known_source_is_replaced_even_when_it_does_not_parse():
    odd = "rtsp://user:pa/ss@host/path"
    assert redact_text(f"{odd}: Invalid argument", odd) == "rtsp://user:***@host/path: Invalid argument"


def test_redact_before_truncating_not_after():
    """`text[:n]` first can cut the URL inside its password, and then there is
    no `@` left for anything to recognise."""
    line = f"ffprobe failed for {NVR}: timed out"
    cut = line[: line.index("hunter2") + 4]
    assert "hunt" in redact_text(cut)                         # the trap
    assert "hunt" not in redact_text(line)[: len(cut)]        # the order that works


def test_a_decoded_document_is_scrubbed_wherever_the_url_sits():
    table = {"cam": {"producers": [{"url": NVR, "medias": ["video, H264"], "id": 3}],
                     "consumers": None}}
    clean = redact_tree(table)
    assert clean["cam"]["producers"][0] == {
        "url": "rtsp://test:***@127.0.0.1:6554/Streaming/Channels/101",
        "medias": ["video, H264"], "id": 3,
    }
    assert table["cam"]["producers"][0]["url"] == NVR        # the input is not edited


def test_a_camera_row_is_shown_with_source_and_error_scrubbed():
    row = {"id": "c1", "source": NVR, "status_detail": f"{NVR}: 401 Unauthorized", "enabled": 1}
    shown = redact_camera(row)
    assert shown == {
        "id": "c1", "enabled": 1,
        "source": "rtsp://test:***@127.0.0.1:6554/Streaming/Channels/101",
        "status_detail": "rtsp://test:***@127.0.0.1:6554/Streaming/Channels/101: 401 Unauthorized",
    }
    assert row["source"] == NVR                               # the engine's copy is untouched
    assert redact_camera(None) is None


# --- the edit round trip ---------------------------------------------------


def test_the_mask_can_never_be_a_real_password():
    """`restore_password` treats `***` as "keep what is stored". That is only
    safe because no stored source can contain one: `*` is shell punctuation to
    the gateway's own check, at the API door and again in `build_config`."""
    with pytest.raises(UnsafeSource):
        assert_safe_source(f"rtsp://test:{MASK}@127.0.0.1/x")


def test_an_unchanged_mask_gets_the_stored_password_back():
    assert is_masked(redact_source(NVR)) and not is_masked(NVR)
    assert restore_password(redact_source(NVR), NVR) == NVR
    # The path is the camera's business; the login and the host are not.
    assert restore_password("rtsp://test:***@127.0.0.1:6554/Streaming/Channels/102", NVR) == (
        "rtsp://test:hunter2@127.0.0.1:6554/Streaming/Channels/102")
    # Byte for byte: a percent-encoded password is not decoded and re-encoded.
    encoded = "rtsp://test:p%40ss%2Fword@cam.local/1"
    assert restore_password("rtsp://test:***@cam.local/2", encoded) == (
        "rtsp://test:p%40ss%2Fword@cam.local/2")


def test_a_source_that_is_not_masked_is_left_alone():
    assert restore_password("rtsp://test:new-pass@127.0.0.1:6554/x", NVR) == (
        "rtsp://test:new-pass@127.0.0.1:6554/x")
    assert restore_password("/srv/footage/line_3.mp4", NVR) == "/srv/footage/line_3.mp4"
    assert restore_password("rtsp://cam.local/sub", None) == "rtsp://cam.local/sub"


@pytest.mark.parametrize("submitted", [
    "rtsp://test:***@10.9.9.9:6554/Streaming/Channels/101",
    "rtsp://test:***@127.0.0.1:554/Streaming/Channels/101",
    "rtsp://test:***@127.0.0.1/Streaming/Channels/101",
    "rtsp://other:***@127.0.0.1:6554/Streaming/Channels/101",
    "rtsps://test:***@127.0.0.1:6554/Streaming/Channels/101",
])
def test_the_stored_password_does_not_follow_the_url_to_another_login(submitted: str):
    with pytest.raises(PasswordNotKept, match="type the password"):
        restore_password(submitted, NVR)


def test_a_second_at_sign_does_not_smuggle_the_password_to_another_host():
    """`...:***@127.0.0.1:6554@10.9.9.9/` is dialled as 10.9.9.9 by everything
    that opens URLs, with `***@127.0.0.1:6554` as the password. It is read the
    same way here — so it is not a mask, nothing stored is put into it, and the
    gateway's own check refuses the `*` it still contains."""
    smuggled = "rtsp://test:***@127.0.0.1:6554@10.9.9.9/Streaming/Channels/101"
    assert not is_masked(smuggled)
    assert restore_password(smuggled, NVR) == smuggled
    with pytest.raises(UnsafeSource):
        assert_safe_source(smuggled)


@pytest.mark.parametrize("stored", [None, "rtsp://cam.local/sub", "/srv/footage/a.mp4"])
def test_a_mask_needs_a_stored_password_behind_it(stored):
    with pytest.raises(PasswordNotKept, match="no stored password"):
        restore_password("rtsp://test:***@cam.local/sub", stored)
