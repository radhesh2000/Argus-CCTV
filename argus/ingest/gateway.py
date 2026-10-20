"""go2rtc stream gateway: RTSP/ONVIF in, WebRTC out.

go2rtc (MIT) replaces v1's OpenCV-decode-and-re-encode-as-MJPEG live wall. It
owns every connection to a camera exactly once and fans that single pull out to
however many consumers want it — the decoders, the browser over WebRTC, and
later the recorder — which matters on a box with one NVDEC and one NVENC.

The binary is vendored, not packaged: ``scripts/fetch_gateway.sh`` pins a
version and a sha256 into ``bin/``. This module owns two things:

* generating ``go2rtc.yaml`` from the cameras table, so the fleet is described
  in one place (the database) rather than in a second hand-edited file, and
* supervising the process — start, restart on crash with backoff, stop.

File-backed cameras are published through an ``exec:`` source with
``-stream_loop -1``. go2rtc's own ``#input=file`` template plays a file exactly
once (measured: it expands to ``-re -i``, no loop), which is not a camera.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import yaml

# The same binary the decoders use — the vendored LGPL/NVDEC build when it
# exists. go2rtc is told about it explicitly (`ffmpeg: bin:`) so its own
# `ffmpeg:` sources and our `exec:` sources cannot end up on different builds.
from argus.ingest.decoder import FFMPEG
from argus.redact import redact_source, redact_text


def stream_name(camera_id: str) -> str:
    """go2rtc stream name for a camera. Kept identical to the camera id so URLs
    are guessable from the database."""
    return camera_id


def _exec_source(source: str, camera_id: str, *, loop: bool) -> str:
    """ffmpeg command publishing a local file into go2rtc's internal RTSP.

    Details that are not optional, each measured on this box:

    * ``-rtsp_transport tcp`` on the *output*: publishing over RTSP defaults to
      UDP, which go2rtc's internal endpoint answers with ``461 Unsupported
      transport``; the stream then never starts.
    * ``-metadata title=<camera_id>``: go2rtc keys producers by the source
      string, so two cameras sharing one recording would otherwise generate
      byte-identical commands and collide — the second stream 404s while the
      first gets its RTP session corrupted. The title makes each command unique
      and lands in the SDP, which is useful anyway. See
      :func:`assert_unique_sources`.
    * ``-fflags +genpts``: ``-stream_loop`` restarts the file's timestamps at
      every seam. Without regenerated PTS the discontinuity reaches the decoder's
      ``fps`` filter, which fills the apparent gap with duplicated frames.
    * ``-c:v copy``: the file is already H.264, so the gateway remuxes rather
      than re-encodes, leaving the box's single NVENC free.
    """
    loop_args = "-stream_loop -1 " if loop else ""
    return (
        f"exec:{FFMPEG} -hide_banner -loglevel error -nostdin "
        f"-fflags +genpts -re {loop_args}-i {source} "
        f"-an -c:v copy -metadata title={camera_id} "
        f"-f rtsp -rtsp_transport tcp {{output}}"
    )


#: The name of the go2rtc ffmpeg template :func:`browser_fallback_source` asks
#: for, and the arguments written into the generated config under it.
#:
#: Every flag is about a *browser watching a live tile*, not about a file:
#: `-bf 0` because B-frames add a frame of latency and WebRTC decoders dislike
#: them; `-g 50` for a keyframe every two seconds at 25 fps, so a tile that
#: joins mid-stream draws within that; `-tune:v ll` for NVENC's low-latency
#: rate control; `-preset:v p4` as the balanced point.
#:
#: **Two flags are absent on purpose, and both were measured getting it wrong.**
#: `-pix_fmt` inserts a CPU scaler that cannot take a cuda frame, and NVENC
#: writes 8-bit 4:2:0 regardless (docs/PRODUCTION_READINESS.md §7.2). And
#: `-level:v 4.1` — which is in every go2rtc transcode example on the internet,
#: because every one of them is about a 1080p camera — **cannot encode 1440p**:
#: NVENC answers `Error while opening encoder - maybe incorrect parameters` and
#: the tile stays black. A customer's cameras are 2560x1440. Let the encoder
#: choose its own level.
NVENC_TEMPLATE_NAME = "h264/nvenc"
NVENC_TEMPLATE_ARGS = (
    "-c:v h264_nvenc -g 50 -bf 0 -profile:v high -preset:v p4 -tune:v ll"
)


def _primary_source(camera: dict[str, Any]) -> str:
    """The source go2rtc should prefer: the camera's own bytes, never re-encoded."""
    kind = camera.get("kind") or "rtsp"
    source = assert_safe_source(camera["source"])
    if kind == "file":
        return _exec_source(source, camera["id"], loop=True)
    if kind == "rtsp":
        # go2rtc's native RTSP client: no ffmpeg in the path at all.
        return source
    return f"ffmpeg:{source}#video=copy#audio=copy"


def browser_fallback_source(camera_id: str) -> str:
    """The second source: the same stream, re-encoded to H.264 for a browser.

    **Why every camera gets one.** go2rtc negotiates a codec with each consumer
    and uses the *first* source whose codec the consumer accepts. An H.264
    camera therefore never touches this: the browser matches source one and the
    transcoder is never started. An H.265 camera has nothing a browser can take,
    and before this existed go2rtc simply refused —

        streams: codecs not matched: video:H265 => video:H264, video:VP8

    — which is a black tile with no picture at all, for the *only* codec half of
    a modern NVR ships. Measured on this appliance 2026-08-25 against every
    camera in a customer's plant: all five H.265 feeds refused an H.264-only
    offer, all three H.264 feeds served it (docs/PRODUCTION_READINESS.md §10.3.1).

    So the branch is unconditional and costs nothing to have: no probe at
    startup, no per-codec table to keep current, and nothing spawned until a
    viewer actually needs it. It is also the right answer for codecs nobody
    listed — MPEG-4, MJPEG, an H.265 sub-stream on a camera enrolled as H.264.

    **Why it names the encoder instead of saying `#hardware`.** `#hardware` is
    go2rtc's own "use whatever this box has", and on this box it produces
    nothing: go2rtc 1.9.14's hardware probe on `linux/arm64` offers **v4l2m2m
    and rkmpp only** — the Raspberry Pi and Rockchip encoders — because arm64 is
    assumed to be an SBC. A DGX Spark is an arm64 machine with a datacentre
    NVENC, so the probe finds nothing, falls back to software `libx264`, and the
    transcode dies with `Unknown encoder 'libx264'` — this appliance's ffmpeg is
    built LGPL on purpose and x264 is GPL. Measured, in exactly that order, on
    2026-08-25.

    So the encoder is named: :data:`NVENC_TEMPLATE`, written into the generated
    config as a `h264/nvenc` template and selected here. On a box with no NVENC
    the transcode fails and the tile falls back to stills — which is the same
    outcome as before this existed, not a regression.

    It runs **per viewer**, not per camera: two operators watching the same
    H.265 tile are two encodes on the box's single NVENC. That is the capacity
    number this buys, and it is in docs/BENCHMARKS.md rather than assumed away.
    """
    return f"ffmpeg:{stream_name(camera_id)}#video={NVENC_TEMPLATE_NAME}"


def _stream_entry(camera: dict[str, Any]) -> list[str]:
    """One go2rtc `streams:` value for a cameras row: preferred first."""
    return [_primary_source(camera), browser_fallback_source(str(camera["id"]))]


class GatewayConfigError(ValueError):
    """Raised when the generated gateway config would not work."""


class UnsafeSource(GatewayConfigError):
    """A camera source that must not reach go2rtc."""


#: Schemes a camera may name. go2rtc understands more — ``exec:``, ``ffmpeg:``,
#: ``echo:``, ``expr:`` — and every one of them is a way to run a program.
SAFE_SCHEMES: tuple[str, ...] = ("rtsp://", "rtsps://", "rtmp://", "rtmps://",
                                 "http://", "https://")

#: Characters that must never appear in a source. A file source is interpolated
#: into the ``exec:`` command line above; go2rtc splits that on whitespace, so a
#: space in a path is not a quoting bug, it is a second argument to ffmpeg.
_SHELL_UNSAFE = set(" \t\r\n\"'`$;&|<>()*?[]{}!#\\")


def assert_safe_source(source: str, *, must_exist: bool = False) -> str:
    """Refuse a camera source that could become something other than a camera.

    **This is a real hole, not a hypothetical one.** ``_exec_source`` above
    interpolates a file source into an ffmpeg command line, and until the
    console could author cameras, the only way to write that column was
    ``config/seed.yaml`` — a file an operator with shell access owns anyway.
    Phase 5 gave RESPOND accounts ``POST /api/cameras``, which turns "the
    source column" into "an argument list a supervisor can type", and a source
    of ``exec:sh -c ...`` or ``/tmp/x -y http://elsewhere`` would be honoured by
    a component whose entire job is to run ffmpeg.

    So: a URL in one of :data:`SAFE_SCHEMES` with a host on it, or an absolute
    path to a file that exists, with no character in it that go2rtc's command
    splitter or ffmpeg's argument parser would read as punctuation. Called both
    at the API door (where the message reaches a person) and in
    :func:`build_config` (where it stops a row that arrived another way).

    ``must_exist`` is on at the door and off in ``build_config``, deliberately:
    a typed path that is not there is a typo somebody should be told about
    immediately, but a recording that was moved after enrolment must not stop
    the *other* forty-seven cameras from being published.
    """
    candidate = str(source).strip()
    if not candidate:
        raise UnsafeSource("a camera needs a source")
    if any(ord(char) < 32 for char in candidate):
        raise UnsafeSource("a camera source may not contain control characters")

    lowered = candidate.lower()
    if "://" in lowered:
        if not lowered.startswith(SAFE_SCHEMES):
            scheme = lowered.split("://", 1)[0]
            raise UnsafeSource(
                f"unsupported source scheme {scheme!r}: a camera may be "
                f"{', '.join(s.rstrip(':/') for s in SAFE_SCHEMES)} or a local file path"
            )
        parsed = urllib.parse.urlparse(candidate)
        if not parsed.hostname:
            raise UnsafeSource(f"no host in camera source: {redact_source(candidate)!r}")
        if any(char in _SHELL_UNSAFE for char in candidate):
            raise UnsafeSource("a camera URL may not contain whitespace or shell punctuation")
        return candidate

    # A local file, which is how the demo fleet and any NVR export is played.
    if candidate.startswith("-"):
        raise UnsafeSource("a file source may not start with '-' (ffmpeg would read it as a flag)")
    if any(char in _SHELL_UNSAFE for char in candidate):
        raise UnsafeSource("a file source may not contain whitespace or shell punctuation")
    path = Path(candidate)
    if not path.is_absolute():
        raise UnsafeSource(f"a file source must be an absolute path: {candidate!r}")
    if must_exist and not path.is_file():
        raise UnsafeSource(f"no such file: {candidate}")
    return candidate


def assert_unique_sources(streams: dict[str, list[str]]) -> None:
    """Reject two streams whose *preferred* source is the same string.

    go2rtc collides on identical sources: one stream serves, the other answers
    404, and the survivor's RTP session gets corrupted by the loser's retries.
    Two cameras genuinely sharing one RTSP URL is a configuration mistake worth
    failing loudly on rather than debugging from a decoder's error counter.

    Only the first source of each stream is compared. The second is the
    per-stream H.264 fallback (:func:`browser_fallback_source`), which names its
    own stream and so is distinct by construction.
    """
    seen: dict[str, str] = {}
    for name, sources in streams.items():
        source = sources[0]
        if source in seen:
            raise GatewayConfigError(
                f"cameras {seen[source]!r} and {name!r} resolve to the same gateway source "
                f"({redact_source(source)!r}); go2rtc cannot serve both. Give them distinct sources."
            )
        seen[source] = name


#: How long a probe waits. An RTSP handshake to a camera on the same LAN is
#: tens of milliseconds; a wrong IP is a TCP timeout, and this is long enough to
#: tell "slow" from "not there" without the enrolment form hanging. The gateway
#: half is a loopback HTTP call and needs far less than the ffprobe half, which
#: is opening a camera on a plant network.
PROBE_TIMEOUT_SEC = 4.0
PROBE_OPEN_TIMEOUT_SEC = 12.0


def describe_medias(payload: Any) -> list[str]:
    """The tracks go2rtc found, flattened for a human. ``["video, recvonly, H264"]``."""
    medias: list[str] = []
    if not isinstance(payload, dict):
        return medias
    for producer in payload.get("producers") or []:
        if isinstance(producer, dict):
            medias.extend(str(media) for media in (producer.get("medias") or []))
    return medias


def _probe_via_gateway(config, camera: dict[str, Any]) -> dict[str, Any] | None:
    """What go2rtc says about a camera it is already serving, or None.

    ``GET /api/streams?src=<stream name>`` reports the live producer: its
    transport, its tracks and its consumers. That is the *authoritative* answer
    for an enrolled camera, because it describes the connection the appliance is
    actually holding rather than a second one opened to ask.

    None means "the gateway had nothing to say" — it is not running, or it is
    running and does not know this stream — and the caller falls through to a
    direct probe rather than reporting a non-answer as a failure.
    """
    url = f"{config.gateway_url}/api/streams?{urllib.parse.urlencode({'src': stream_name(camera['id'])})}"
    try:
        with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT_SEC) as response:  # noqa: S310 - loopback, host from config
            payload = json.loads(response.read().decode() or "null")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError,
            TimeoutError, ValueError):
        return None
    medias = describe_medias(payload)
    if not medias:
        return None
    producers = payload.get("producers") or []
    protocol = str(producers[0].get("protocol") or "") if producers else ""
    return {
        "ok": True,
        "gateway": True,
        "route": "gateway",
        "kind": camera.get("kind"),
        "medias": medias,
        "detail": f"go2rtc is serving this stream over {protocol or 'its own transport'}: "
                  f"{', '.join(medias)}",
    }


def _probe_direct(camera: dict[str, Any]) -> dict[str, Any]:
    """Open the source with the appliance's own ffprobe.

    Used for a URL that is not enrolled yet — the enrolment form's test button.
    It is the same LGPL/NVDEC binary the decoders use and the one go2rtc is
    pointed at (``ffmpeg: bin:``), so "ffprobe can open it" and "the gateway
    will be able to open it" are as close as this appliance can get without
    saving the camera first.

    **Why not ask the gateway.** go2rtc 1.9.14 exposes no probe endpoint —
    ``/api/probe`` is a 404, ``/api/streams?src=`` resolves a stream *name* and
    not a source string, and ``POST /api/streams`` answered 400 on this build.
    Making it work would mean adding a temporary stream to a running gateway and
    remembering to remove it, which mutates the live video plane of a plant to
    answer a form. Measured, not assumed: both attempts are in this session's
    notes.
    """
    from argus.ingest.decoder import probe_source as ffprobe_source

    try:
        found = ffprobe_source(str(camera["source"]), timeout=PROBE_OPEN_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return {"ok": False, "gateway": True, "route": "ffprobe", "kind": camera.get("kind"),
                "detail": f"nothing answered within {PROBE_OPEN_TIMEOUT_SEC:.0f}s"}
    except Exception as exc:  # noqa: BLE001 - "it did not open" is the result
        return {"ok": False, "gateway": True, "route": "ffprobe", "kind": camera.get("kind"),
                "detail": (redact_text(str(exc), camera["source"]) or "")[:300]}
    codec = found.get("codec_name")
    if not codec:
        return {"ok": False, "gateway": True, "route": "ffprobe", "kind": camera.get("kind"),
                "detail": "opened, but there is no video track on it"}
    size = f"{found.get('width', '?')}x{found.get('height', '?')}"
    return {
        "ok": True,
        "gateway": True,
        "route": "ffprobe",
        "kind": camera.get("kind"),
        "medias": [f"video, {codec}, {size}"],
        "detail": f"opened: {codec} {size} at {found.get('r_frame_rate', '?')} fps",
    }


def probe_camera(config, camera: dict[str, Any], *, prefer_gateway: bool) -> dict[str, Any]:
    """Can this camera be opened, and by whom.

    Two routes, and the response says which one answered, because they answer
    different questions. ``gateway`` means go2rtc is holding this stream right
    now — the strongest possible answer, and only available for a camera that is
    already enrolled and running. ``ffprobe`` means the appliance opened the
    source itself, which is the only thing available for a URL an operator has
    just typed.

    Never raises. "Unreachable" is the result of a connection test, not an
    error in it — the console renders the two differently and an installer
    working through a wrong password needs the sentence, not a 500.
    """
    started = time.monotonic()
    answer = _probe_via_gateway(config, camera) if prefer_gateway else None
    if answer is None:
        answer = _probe_direct(camera)
    return {**answer, "elapsed_sec": round(time.monotonic() - started, 3)}


def build_config(cameras: Sequence[dict[str, Any]], config) -> dict[str, Any]:
    """Render the full go2rtc configuration for a set of cameras."""
    gateway = config.gateway
    host = gateway.listen_host
    streams = {
        stream_name(camera["id"]): _stream_entry(camera)
        for camera in cameras
        if camera.get("enabled", 1)
    }
    assert_unique_sources(streams)

    # `webrtc.candidates` (go2rtc 1.9.14's own key): addresses to advertise as
    # ICE host candidates *in addition to* the ones go2rtc enumerates from this
    # box's interfaces. Measured through the relay, each entry becomes a UDP and
    # a TCP `typ host` candidate trickled ahead of the gathered set. That is how
    # a site whose remote-access address is not a local interface — a
    # port-forward, a VPN terminating elsewhere — reaches a browser that would
    # otherwise be offered only addresses it cannot route to. See
    # `gateway.webrtc_candidates` in argus/config.py.
    #
    # Omitted entirely when unconfigured, rather than written as an empty list:
    # an unconfigured appliance keeps getting the file go2rtc has been parsing
    # since Phase 1.
    webrtc: dict[str, Any] = {"listen": f":{gateway.webrtc_port}"}
    candidates = gateway.webrtc_candidate_list
    if candidates:
        webrtc["candidates"] = candidates
    return {
        "api": {
            # Loopback (`gateway.api_host`, default 127.0.0.1). Phase 5 took
            # this off the LAN: go2rtc's HTTP API is snapshots of every camera,
            # the stream table, the WebRTC signalling *and* its own
            # unauthenticated admin UI, and it had no credentials on it. The
            # browser now reaches signalling and posters through
            # `/api/gateway/*` on `argus api`, which has a session cookie on it.
            #
            # `origin: "*"` is gone with it, and this is why it can be. It was
            # there because the SPA was served from another port, so every
            # WebRTC handshake was cross-origin and go2rtc's Upgrader rejected
            # it ("request origin not allowed"). Through the relay the handshake
            # is same-origin — go2rtc sees a request from this process on
            # loopback, and there is no browser origin to allow.
            "listen": f"{gateway.api_host}:{gateway.api_port}",
        },
        # RTSP stays on `listen_host` (0.0.0.0 by default). It is how the
        # decoders pull, which is loopback, but it is also the documented way a
        # site's existing VMS or NVR re-uses these streams — a real deployment
        # need, and one an operator can close by setting `listen_host`.
        "rtsp": {"listen": f"{host}:{gateway.rtsp_port}"},
        # The media path, and it cannot be relayed away: ICE connects the
        # browser to this port directly. See docs/SECURITY.md — the payload is
        # per-session SRTP, and a full TURN-style media relay is out of scope.
        "webrtc": webrtc,
        "srtp": {"listen": ""},
        # `bin` is the vendored LGPL/NVDEC build the decoders use. The extra
        # template is the H.264 encoder `browser_fallback_source` asks for by
        # name — go2rtc's own `#hardware` probe does not know about NVENC on
        # arm64 and would fall through to an x264 this build deliberately lacks.
        "ffmpeg": {"bin": FFMPEG, NVENC_TEMPLATE_NAME: NVENC_TEMPLATE_ARGS},
        "log": {"level": gateway.log_level},
        "streams": streams,
    }


def write_config(cameras: Sequence[dict[str, Any]], config, path: Path | None = None) -> Path:
    """Write go2rtc.yaml under the data dir. Returns the path written."""
    target = path or (config.gateway_dir / "go2rtc.yaml")
    target.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# GENERATED BY ARGUS — do not edit.\n"
        "# Rendered from the cameras table by argus/ingest/gateway.py on every start.\n"
    )
    target.write_text(header + yaml.safe_dump(build_config(cameras, config), sort_keys=False))
    return target


class Gateway:
    """Supervised go2rtc process.

    ``start()`` returns immediately; a daemon thread keeps the process alive and
    restarts it with backoff if it dies. Restarts are counted so a gateway that
    is crash-looping is visible in ``argus run`` output rather than silent.
    """

    def __init__(self, config, cameras: Sequence[dict[str, Any]]):
        self.config = config
        self.cameras = list(cameras)
        self.binary = config.gateway_binary
        self.config_path = config.gateway_dir / "go2rtc.yaml"
        self.log_path = config.logs_dir / "go2rtc.log"
        self.restarts = 0
        self.last_error: str | None = None
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- lifecycle --------------------------------------------------------

    def available(self) -> bool:
        return self.binary.exists() and self.binary.is_file()

    def version(self) -> str | None:
        if not self.available():
            return None
        try:
            proc = subprocess.run(
                [str(self.binary), "--version"], capture_output=True, text=True,
                timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return (proc.stdout or proc.stderr).strip().splitlines()[0] if (proc.stdout or proc.stderr) else None

    def prepare(self) -> Path:
        self.config.ensure_dirs()
        return write_config(self.cameras, self.config, self.config_path)

    def start(self) -> "Gateway":
        if not self.available():
            raise FileNotFoundError(
                f"go2rtc binary not found at {self.binary} — run scripts/fetch_gateway.sh"
            )
        self.prepare()
        self._stop.clear()
        self._thread = threading.Thread(target=self._supervise, daemon=True, name="gateway")
        self._thread.start()
        return self

    def _supervise(self) -> None:
        backoff = self.config.gateway.restart_backoff_sec
        first = True
        while not self._stop.is_set():
            if not first:
                self.restarts += 1
            first = False
            started = time.time()
            try:
                if self.restarts == 0:
                    self._scrub_logs()
                with self.log_path.open("ab") as log:
                    log.write(f"\n--- argus start {time.strftime('%Y-%m-%dT%H:%M:%S')} ---\n".encode())
                self._proc = subprocess.Popen(
                    [str(self.binary), "-config", str(self.config_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                )
                threading.Thread(
                    target=self._copy_log, args=(self._proc,), daemon=True, name="gateway-log",
                ).start()
                self._proc.wait()
            except OSError as exc:
                self.last_error = str(exc)
            finally:
                self._proc = None

            if self._stop.is_set():
                return
            self.last_error = f"go2rtc exited after {time.time() - started:.1f}s"
            # A process that survived a while was healthy; don't punish it with
            # an escalated backoff on its first crash.
            if time.time() - started > 30:
                backoff = self.config.gateway.restart_backoff_sec
            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2, 60.0)

    def _copy_log(self, proc: subprocess.Popen) -> None:
        """go2rtc's output into ``go2rtc.log``, minus the camera passwords.

        go2rtc used to write the file itself, and it reports a dropped camera as
        ``WRN ... error=EOF url=rtsp://user:password@host/...`` — once per drop,
        per camera, for as long as the appliance runs. So its output comes
        through here instead and is redacted a line at a time.

        Its own thread, with its own handle on the log. The supervisor must keep
        waiting on the *process*: an ffmpeg child that outlives go2rtc holds this
        pipe open, and a supervisor reading to end-of-file would wait on it
        instead of restarting the gateway. The handle is opened for append, so
        copy-and-truncate rotation (argus/backup.py) works as it did.

        Nothing in the loop may stop the reading. A pipe nobody drains fills,
        go2rtc blocks on its next log line, and the video plane stops with it —
        so a line that cannot be written (a full disk) is dropped, not raised.
        """
        stream = proc.stdout
        if stream is None:
            return
        sources = [camera.get("source") for camera in self.cameras]
        try:
            log = self.log_path.open("ab")
        except OSError:
            log = None
        try:
            for raw in iter(stream.readline, b""):
                if log is None:
                    continue
                try:
                    line = redact_text(raw.decode("utf-8", "replace"), *sources) or ""
                    log.write(line.encode("utf-8", "replace"))
                    log.flush()
                except Exception:  # noqa: BLE001, S112 - draining matters more than the line
                    continue
        finally:
            stream.close()
            if log is not None:
                log.close()

    def _scrub_logs(self) -> None:
        """Mask the passwords go2rtc already wrote, before it writes any more.

        Everything logged before :meth:`_copy_log` existed is still on disk,
        current file and rotated generations alike. Rewritten in place (same
        inode, like rotation) and only when there is something to mask. Never
        fatal: a log that cannot be cleaned must not keep the gateway down.
        """
        sources = [camera.get("source") for camera in self.cameras]
        for path in sorted(self.log_path.parent.glob(f"{self.log_path.name}*")):
            try:
                # surrogateescape: whatever bytes are in there come back out.
                text = path.read_text(encoding="utf-8", errors="surrogateescape")
                clean = redact_text(text, *sources) or ""
                if clean != text:
                    with path.open("r+", encoding="utf-8", errors="surrogateescape") as log:
                        log.write(clean)
                        log.truncate()
            except OSError:
                continue

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    # --- introspection ----------------------------------------------------

    def api_url(self, path: str = "") -> str:
        return f"{self.config.gateway_url}{path}"

    def wait_ready(self, timeout: float = 20.0) -> bool:
        """Block until the HTTP API answers (or the timeout expires)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._stop.is_set():
                return False
            try:
                with urllib.request.urlopen(self.api_url("/api"), timeout=2) as resp:
                    if resp.status < 400:
                        return True
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.4)
        return False

    def streams(self) -> dict[str, Any]:
        """Live stream table from the gateway API (producers/consumers)."""
        with urllib.request.urlopen(self.api_url("/api/streams"), timeout=5) as resp:
            return json.loads(resp.read().decode())

    def rtsp_url(self, camera_id: str) -> str:
        return self.config.rtsp_url(stream_name(camera_id))

    def webrtc_url(self, camera_id: str) -> str:
        """go2rtc's own player page for this camera — loopback only since Phase
        5, so this is a debugging URL for somebody on the appliance, not
        something the console links to. The console uses the authenticated
        relay at ``/api/gateway/ws``."""
        return f"{self.config.gateway_url}/webrtc.html?src={stream_name(camera_id)}"

    def status(self) -> dict[str, Any]:
        return {
            "binary": str(self.binary),
            "available": self.available(),
            "running": self.is_running(),
            "restarts": self.restarts,
            "last_error": self.last_error,
            "config_path": str(self.config_path),
            "api_url": self.config.gateway_url,
            "streams": [stream_name(c["id"]) for c in self.cameras if c.get("enabled", 1)],
        }
