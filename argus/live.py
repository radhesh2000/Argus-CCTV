"""The newest analysed frame, published on request, for the process that cannot see it.

Policy Studio's live prompt test needs one thing the API process does not have:
pixels. Live frames exist only inside ``argus run`` — in the decoder's slot, in
``VlmRunner._frames``, in ``CameraRuntime.live_image`` — and the bridge between
the two processes is the engine stats file, which carries counters and boxes.

Three ways to close that were considered and two rejected:

* **A second decoder in the API process.** Rejected outright. The measured
  ceiling on this box is the vision loop and the ~130 Python threads already
  contending for the GIL (docs/BENCHMARKS.md); a second ffmpeg per camera to
  answer a button is the one thing the whole ingest design exists to avoid.
* **The index runner's thumbnails.** Free — they are already written every
  ``index.sample_interval_sec`` — but they are 320 px at quality 70, and the
  scheduler sends the model 960 px at quality 85. A test panel that showed the
  model a quarter of the pixels production uses would answer a different
  question from the one the author is asking, in the direction that quietly
  teaches them to over-write the prompt.
* **What this module does: the engine publishes the frame it just analysed,
  and only when somebody asked.** Same frame the detector saw and the same
  frame the scheduler would have sampled, at full decode resolution.

**The protocol is two files and no ports.** The API writes a request into
``data/live/requests.json`` (camera id -> expiry epoch); the engine reads it on
its ordinary stats tick and encodes a JPEG to ``data/live/<camera>.jpg`` for
each unexpired name. Steady state is therefore one ``stat`` per tick and *zero*
encodes — which is the point, on a box where an always-on publisher would cost
one JPEG per camera per tick forever so that a button could be quick twice a
week.

The failure modes are all visible and none of them are silent: the engine is
not running (no file appears, and the stats file already says so), the camera
produced no frame (nothing to encode, and the request expires), two API workers
race on the request file (last writer wins; the loser's camera is simply not
published this tick and the caller retries). Nothing here blocks the vision
loop and nothing here can stop a camera.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

#: How long one request stays live. Long enough for the engine's 2 s stats tick
#: plus a slow encode, short enough that a closed browser tab stops costing
#: anything within one interval.
DEFAULT_TTL_SEC = 20.0

#: What the API waits for a frame before saying the engine is not answering.
#: Two stats ticks and a little: one to notice the request, one to publish.
DEFAULT_WAIT_SEC = 6.0

#: Camera ids reach the filesystem here, so they are checked rather than
#: trusted. The API's own schema is stricter (`^[a-z0-9][a-z0-9_-]*$`) but seed
#: files are hand-written and this is the layer where a `../` would matter.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class UnsafeCameraId(ValueError):
    """A camera id that cannot be used as a filename."""


def safe_id(camera_id: str) -> str:
    candidate = str(camera_id)
    if not _SAFE_ID.match(candidate) or ".." in candidate:
        raise UnsafeCameraId(f"unusable camera id: {camera_id!r}")
    return candidate


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write via a temporary sibling and rename, so a reader never sees half.

    The pid is in the temporary name because the engine and the API both write
    into this directory, and two processes sharing one ``.tmp`` would eventually
    truncate each other's write rather than one simply losing the race.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


class LiveFrames:
    """Both ends of the request/publish protocol. One class, two callers.

    Written as one object rather than a pair because the two halves are a single
    agreement about a directory layout, and splitting them across the API and
    the engine is how the two ends of a file protocol drift apart.
    """

    def __init__(self, directory: Path | str):
        self.dir = Path(directory)

    @classmethod
    def from_config(cls, config: Any) -> LiveFrames:
        return cls(config.live_dir)

    # --- layout -----------------------------------------------------------

    @property
    def requests_path(self) -> Path:
        return self.dir / "requests.json"

    def frame_path(self, camera_id: str) -> Path:
        return self.dir / f"{safe_id(camera_id)}.jpg"

    # --- the API side -----------------------------------------------------

    def request(self, camera_id: str, ttl_sec: float = DEFAULT_TTL_SEC,
                now: float | None = None) -> float:
        """Ask the engine to publish this camera's next analysed frame.

        Merges into whatever is already there and drops expired names on the
        way, so the file stays the size of the interest that currently exists
        rather than the size of every camera ever tested.
        """
        camera_id = safe_id(camera_id)
        now = time.time() if now is None else now
        expiry = now + max(1.0, float(ttl_sec))
        wanted = {name: until for name, until in self._read_requests().items() if until > now}
        wanted[camera_id] = max(expiry, wanted.get(camera_id, 0.0))
        _write_atomic(self.requests_path, json.dumps(wanted).encode())
        return expiry

    def read(self, camera_id: str) -> tuple[bytes, float] | None:
        """The published JPEG and the wall-clock time it was written, or None."""
        path = self.frame_path(camera_id)
        try:
            payload = path.read_bytes()
            written = path.stat().st_mtime
        except OSError:
            return None
        return (payload, written) if payload else None

    def wait_for(
        self, camera_id: str, *, newer_than: float, timeout: float = DEFAULT_WAIT_SEC,
        poll_sec: float = 0.15,
    ) -> tuple[bytes, float] | None:
        """Block until a frame newer than ``newer_than`` appears, or give up.

        ``newer_than`` rather than "any frame": the file may be left over from a
        previous test or a previous run of the engine, and a prompt test that
        silently answered about a frame from yesterday is worse than one that
        says the engine is not publishing.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            found = self.read(camera_id)
            if found is not None and found[1] > newer_than:
                return found
            if time.monotonic() >= deadline:
                return None
            time.sleep(poll_sec)

    # --- the engine side --------------------------------------------------

    def wanted(self, now: float | None = None) -> set[str]:
        """Camera ids with a live request. Cheap enough for every stats tick."""
        now = time.time() if now is None else now
        return {name for name, until in self._read_requests().items() if until > now}

    def publish(self, images: Mapping[str, Any], *, quality: int = 85) -> list[str]:
        """Encode and write the frames named by ``images``. Returns what landed.

        Never raises: this runs on the engine's reporting path, and a camera
        that cannot be encoded must cost a missing test frame rather than a
        missed stats write.
        """
        written: list[str] = []
        for camera_id, image in images.items():
            if image is None:
                continue
            try:
                _write_atomic(self.frame_path(camera_id), _encode(image, quality))
                written.append(camera_id)
            except (OSError, UnsafeCameraId, ValueError):
                continue
        return written

    def clear(self) -> None:
        """Drop every published frame and every request.

        Called when the engine stops. A frame left on disk is a frame the API
        would happily serve as current for as long as the appliance is up, and
        ``wait_for``'s freshness check is the belt to this brace.
        """
        try:
            for path in self.dir.glob("*.jpg"):
                path.unlink(missing_ok=True)
            self.requests_path.unlink(missing_ok=True)
        except OSError:
            pass

    # --- internals --------------------------------------------------------

    def _read_requests(self) -> dict[str, float]:
        try:
            payload = json.loads(self.requests_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        wanted: dict[str, float] = {}
        for name, until in payload.items():
            try:
                wanted[safe_id(name)] = float(until)
            except (UnsafeCameraId, TypeError, ValueError):
                continue
        return wanted


def _encode(image: Any, quality: int) -> bytes:
    """BGR array (or PIL image) -> JPEG bytes, at the decode resolution.

    Deliberately *not* downscaled. The whole argument for this path over the
    index thumbnails is that the model sees what production sees; the client
    applies ``vlm.max_image_width`` on the way out, exactly as it does for a
    scheduled evaluation.
    """
    import io

    from argus.media.annotate import to_image

    buffer = io.BytesIO()
    to_image(image).convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


__all__ = [
    "DEFAULT_TTL_SEC",
    "DEFAULT_WAIT_SEC",
    "LiveFrames",
    "UnsafeCameraId",
    "safe_id",
]
