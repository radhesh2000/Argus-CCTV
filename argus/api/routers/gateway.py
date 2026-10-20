"""The one authenticated door to go2rtc.

Before this, the console talked to go2rtc directly on ``:1984`` and go2rtc was
told ``api.origin: "*"`` to let it. That meant two listeners on the plant LAN —
one with a session cookie on it, one with nothing at all, serving a snapshot of
every camera, the stream table, and its own admin UI, to anybody who could
reach the port.

go2rtc's HTTP API now binds to ``127.0.0.1`` (``gateway.api_host``) and the
browser comes through here instead: same origin, same cookie, same permission
check as every other route. Two endpoints, because that is everything the
console needs.

* ``GET /api/gateway/ws?src=<camera>`` — the WebRTC **signalling** channel,
  relayed frame-for-frame to go2rtc's ``/api/ws``. Relaying rather than
  reimplementing keeps go2rtc's own protocol (trickle ICE over a WebSocket)
  authoritative, so ``ui/src/api/gateway.ts`` did not have to change its
  handshake — only its URL.
* ``GET /api/gateway/frame.jpeg?src=<camera>`` — one still, used as the poster
  while WebRTC negotiates.

**What this does not secure: the media.** ICE connects the browser straight to
``webrtc.listen`` (``:8555``) and no amount of relaying signalling changes that
— a media relay is a TURN server, which is a second network service with its
own credentials and its own bandwidth bill through the appliance. Out of scope,
and written down as such in ``docs/SECURITY.md``: the residual exposure is that
a host on the plant LAN which can reach ``:8555`` can attempt an ICE handshake.
The payload is per-session SRTP whose keys were exchanged over *this*
authenticated channel, so it is not readable footage; it is a port that
answers.

``src`` is validated against the cameras table on every call. Without that
check, an authenticated REVIEW account could pass any string go2rtc understands
— including its ``exec:`` and ``ffmpeg:`` source syntax — and the relay would
become a way to run go2rtc's ffmpeg with arbitrary arguments.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import websockets
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from argus.api.auth import VIEW, ViewDep, websocket_principal
from argus.api.deps import ConfigDep, DbDep
from argus.config import ArgusConfig
from argus.db import Database
from argus.ingest.gateway import stream_name
from argus.live import LiveFrames
from argus.redact import redact_text, redact_tree

router = APIRouter(prefix="/gateway", tags=["gateway"])

#: A poster still is ~30 KB of JPEG from a process on loopback. Two seconds is
#: generous; longer would mean a wall of twelve tiles hanging on one dead camera.
FRAME_TIMEOUT_SEC = 4.0

#: The engine-frame fallback, in three numbers. The request has to outlive the
#: engine's 2 s stats tick (`ttl`); the wait is two ticks and a little, the same
#: budget Policy Studio's live test uses; and a frame already on disk within
#: `stale` is served immediately rather than waiting for a fresh encode, because
#: a wall of tiles that each block for two seconds is a wall that takes half a
#: minute to draw. Any frame served this way carries its age in a header.
ENGINE_FRAME_TTL_SEC = 20.0
ENGINE_FRAME_WAIT_SEC = 5.0
ENGINE_FRAME_STALE_SEC = 4.0

#: Close codes the console can tell apart. 1008 is "policy violation", which is
#: the closest a WebSocket has to 401/403; 1011 is a server-side failure, which
#: for us means go2rtc is not answering.
WS_UNAUTHORIZED = 1008
WS_UPSTREAM_DOWN = 1011


def resolve_stream(db: Database, src: str) -> str:
    """``src`` -> the go2rtc stream name, or 404.

    The allowlist is the cameras table, which is also the list go2rtc's config
    was generated from — so "a stream exists" and "a camera exists" cannot drift
    apart. See the module docstring for what this is defending.
    """
    camera = db.get_camera(src)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"camera not found: {src}")
    if not camera.get("enabled", 1):
        # A disabled camera has no stream in the generated go2rtc config, so the
        # upstream would answer 404 anyway — this just makes it say why.
        raise HTTPException(status_code=409, detail=f"camera is disabled: {src}")
    return stream_name(str(camera["id"]))


def _upstream(config: ArgusConfig, path: str, params: dict[str, str]) -> str:
    return f"{config.gateway_url}{path}?{urllib.parse.urlencode(params)}"


def _gateway_still(config: ArgusConfig, stream: str) -> tuple[bytes, str]:
    """go2rtc's own still, or raise. One JPEG from loopback, read into memory."""
    url = _upstream(config, "/api/frame.jpeg", {"src": stream})
    with urllib.request.urlopen(url, timeout=FRAME_TIMEOUT_SEC) as response:  # noqa: S310 - loopback, host from config
        return response.read(), response.headers.get("Content-Type", "image/jpeg")


def _engine_still(config: ArgusConfig, camera_id: str) -> tuple[bytes, float] | None:
    """The last frame the *engine* analysed, published on request (argus/live.py).

    Cheaper than it looks: the pixels already exist in the decoder's slot, so
    this costs one JPEG encode on the engine's ordinary stats tick and nothing
    at all when no tile is asking.
    """
    live = LiveFrames.from_config(config)
    started = time.time()
    try:
        live.request(camera_id, ttl_sec=ENGINE_FRAME_TTL_SEC)
        return live.wait_for(camera_id, newer_than=started - ENGINE_FRAME_STALE_SEC,
                             timeout=ENGINE_FRAME_WAIT_SEC)
    except Exception:  # noqa: BLE001 - a poster must never 500 the wall
        return None


@router.get("/frame.jpeg", response_class=Response, responses={
    200: {"content": {"image/jpeg": {}}, "description": "One still from the stream"},
})
def frame(src: str, viewer: ViewDep, config: ConfigDep, db: DbDep) -> Response:
    """A single still of one camera: go2rtc's if it can make one, ours if not.

    **Why there is a fallback at all.** go2rtc renders this still by decoding
    the stream with ffmpeg, and on 2026-08-25 that was measured failing for
    *every H.265 camera on this appliance* — `HTTP 500: exit status 183` from
    go2rtc for all five of the plant's 1440p HEVC feeds, and a clean 200 for all
    three H.264 ones. A poster is the only picture a tile has while WebRTC
    negotiates, and — since a browser with no H.265 gets no WebRTC picture
    either (`docs/PRODUCTION_READINESS.md` §10.3.1) — often the only picture it
    will ever have. Six black rectangles is the wrong answer when the appliance
    is already holding a decoded frame of each of those six scenes.

    So: ask the gateway, and if the gateway cannot, serve the frame the engine
    itself last analysed. That frame is 960x544 rather than the camera's native
    size and it is up to a couple of seconds old, both of which the response
    states in headers so the console can label it rather than pass it off as
    live. What it is *not* is a second decoder: the pixels are already in the
    engine's hands (`argus/live.py`).

    `X-Argus-Frame-Source` is `gateway` or `engine`; `X-Argus-Frame-Age` is how
    many seconds old the picture is, and is absent for a gateway still because
    go2rtc renders those on demand.
    """
    stream = resolve_stream(db, src)
    try:
        body, content_type = _gateway_still(config, stream)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        found = _engine_still(config, src)
        if found is None:
            # 502, not 500: the appliance is fine, the thing behind it is not,
            # and the console renders this sentence rather than guessing. Both
            # halves are named — the gateway refused *and* the engine had
            # nothing — because those two need different fixes.
            raise HTTPException(
                status_code=502,
                detail=(f"no still for {src}: the gateway could not render one ({exc}) "
                        "and the engine is not publishing a frame for this camera"),
            ) from exc
        payload, written = found
        return Response(
            content=payload,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Argus-Frame-Source": "engine",
                "X-Argus-Frame-Age": f"{max(0.0, time.time() - written):.1f}",
            },
        )
    return Response(
        content=body,
        media_type=content_type,
        # A still is a moment. Caching it would show an operator the plant as it
        # was when the tile first rendered.
        headers={"Cache-Control": "no-store", "X-Argus-Frame-Source": "gateway"},
    )


@router.get("/streams")
def streams(viewer: ViewDep, config: ConfigDep, db: DbDep) -> dict[str, Any]:
    """go2rtc's own view of what it is serving — producers and consumers.

    Diagnostics for the health screen: "the camera is offline" and "the gateway
    never picked it up" look identical from a black tile.

    Not relayed verbatim: every producer carries the ``url`` go2rtc dialled,
    which is the camera's source with its password in it, and this route is
    open to anyone who may view. See ``argus/redact.py``.
    """
    try:
        with urllib.request.urlopen(  # noqa: S310 - loopback, host from config
            f"{config.gateway_url}/api/streams", timeout=FRAME_TIMEOUT_SEC
        ) as response:
            table = json.loads(response.read().decode())
        sources = [camera["source"] for camera in db.list_cameras()]
        return {"streams": redact_tree(table, *sources)}
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"gateway unreachable: {exc}") from exc


@router.websocket("/ws")
async def signalling(websocket: WebSocket, src: str = Query(...)) -> None:
    """Relay the WebRTC signalling WebSocket to go2rtc.

    Both directions are pumped concurrently and the first to end closes the
    other. Trickle ICE is bidirectional and asynchronous — the browser is still
    sending candidates while the answer is coming back — so a
    read-one-write-one loop would deadlock on the first candidate.

    Frames are forwarded unparsed (a camera password quoted in a go2rtc error
    is masked on the way out, and nothing else is touched). This deliberately
    does not parse the SDP:
    the protocol belongs to go2rtc, and a relay that understood it would be a
    second implementation to keep in step across go2rtc upgrades.
    """
    config: ArgusConfig = websocket.app.state.config
    db: Database = websocket.app.state.db

    # Accept first, then refuse with a code and a reason.
    #
    # A WebSocket closed *before* the handshake completes reaches the browser as
    # nothing but "connection failed" — no status, no message — which a tile
    # cannot tell from a dead gateway. Accepting and then closing with 1008 and
    # a sentence is the only way the console can say *why*. The cost is a
    # socket that lives for a millisecond; the caller is already authenticated
    # by the middleware, so this is not an unauthenticated connection.
    await websocket.accept()

    principal = websocket_principal(websocket, VIEW)
    if principal is None:
        # The middleware refuses an anonymous connection outright; this is the
        # authenticated-but-unprivileged case — a SYSTEM monitoring account.
        await websocket.close(code=WS_UNAUTHORIZED, reason="not permitted to view cameras")
        return

    camera = db.get_camera(src)
    if camera is None or not camera.get("enabled", 1):
        await websocket.close(code=WS_UNAUTHORIZED, reason=f"unknown camera: {src}")
        return
    stream = stream_name(str(camera["id"]))

    url = _upstream(config, "/api/ws", {"src": stream}).replace("http://", "ws://", 1)
    try:
        upstream = await websockets.connect(url, open_timeout=FRAME_TIMEOUT_SEC)
    except (OSError, websockets.exceptions.WebSocketException, TimeoutError) as exc:
        await websocket.close(code=WS_UPSTREAM_DOWN, reason=f"gateway unreachable: {exc}"[:120])
        return

    async def browser_to_gateway() -> None:
        while True:
            message = await websocket.receive_text()
            await upstream.send(message)

    async def gateway_to_browser() -> None:
        async for message in upstream:
            text = message if isinstance(message, str) else message.decode("utf-8", "replace")
            # Still not parsed. But go2rtc reports a camera it could not open in
            # an `error` frame, in its own words, and those can quote the URL
            # it dialled; an SDP or a candidate has nothing here to match.
            await websocket.send_text(redact_text(text, camera["source"]) or "")

    tasks = [asyncio.create_task(browser_to_gateway()),
             asyncio.create_task(gateway_to_browser())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        # Surface an upstream failure rather than swallowing it into a silent
        # close: a tile that says nothing is a tile an operator reads as "the
        # camera is dark".
        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(
                exc, (WebSocketDisconnect, websockets.exceptions.ConnectionClosed)
            ):
                raise exc
    except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed):
        pass
    finally:
        for task in tasks:
            task.cancel()
        await upstream.close()
        try:
            await websocket.close()
        except RuntimeError:
            # Already closed by the disconnect that ended the pump.
            pass
