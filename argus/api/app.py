"""The Argus HTTP API.

JSON only. v1 served server-rendered Jinja pages from the same process that
held the business logic, which meant every UI change was a Python change; the
Phase-4 SPA consumes this API and nothing else, so the two can move
independently and the API is testable without a browser.

The app is built by a factory rather than declared at import time. Two reasons:
tests need an app pointed at a temporary data directory, and the static mounts
need the config's paths before any route exists.

Evidence (snapshots, clips, track overlays) is served by ``StaticFiles`` mounts
rather than by a route that reads files: Starlette handles Range requests,
conditional GETs and ETags there, and a reviewer scrubbing an mp4 needs Range
to work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from argus import __version__
from argus.api.auth import AuthMiddleware
from argus.api.routers import (
    analytics,
    auth,
    cameras,
    events,
    gateway,
    live,
    policies,
    search,
    system,
    users,
)
from argus.config import ArgusConfig, load_config
from argus.db import Database

_DESCRIPTION = """\
Fully-local industrial safety video analytics.

Every route is JSON except the evidence mounts (`/snapshots`, `/clips`,
`/tracks`) and the gateway relay. Nothing here calls out to the internet.

**Authentication is required.** Sign in at `POST /api/auth/login` for a session
cookie; every route except that one, `GET /api/auth/session` and the `GET
/api/health` liveness probe needs it, and so do the evidence mounts. Four roles
— ADMIN, RESPOND, REVIEW, SYSTEM — decide what a session may do; a 403 names
the permission it lacked.

A freshly installed appliance has **no accounts** and refuses everything until
one is created on the box: `argus user add <name> --role ADMIN`.
"""


class SpaFiles(StaticFiles):
    """Static files with a single-page-app fallback.

    A SPA owns its own routes: reloading the browser on ``/alerts`` must serve
    ``index.html`` and let the router sort it out, not 404. Only *misses* fall
    back, so a genuinely absent asset (``/assets/main.js`` after a bad deploy)
    still 404s instead of silently returning HTML that the browser then fails to
    parse as JavaScript — the failure mode that makes this bug hard to find.
    """

    async def get_response(self, path: str, scope: Any) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and not Path(path).suffix:
                return await super().get_response("index.html", scope)
            raise


def create_app(config: ArgusConfig | None = None, database: Database | None = None) -> FastAPI:
    config = config or load_config()
    config.ensure_dirs()
    if database is None:
        database = Database(config.db_path)
        database.migrate()

    app = FastAPI(
        title="Argus",
        version=__version__,
        description=_DESCRIPTION,
        # Docs are part of the product on an air-gapped box: /docs is the only
        # way an integrator can discover the API without internet access.
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    app.state.config = config
    app.state.db = database

    # Deny-by-default, in front of routing *and* the static mounts below —
    # evidence is footage of people at work, and a FastAPI dependency cannot
    # reach a StaticFiles mount. See argus/api/auth.py.
    app.add_middleware(AuthMiddleware, config=config, database=database)

    origins = config.api.cors_origin_list
    if origins:
        # Installed only when configured. The SPA ships from this same process
        # in production, so the default (no middleware) is also the safe one.
        #
        # Added *after* the auth gate, which puts it *outside* it:
        # `add_middleware` inserts at the head of the stack, so the last one
        # added runs first. That ordering is load-bearing. A CORS pre-flight is
        # an OPTIONS request with no cookie on it — by specification — so an
        # auth gate in front would 401 it, and the browser would report "CORS
        # error" for what is really "please log in". CORSMiddleware answers the
        # pre-flight itself and never reaches the gate; the real request that
        # follows does.
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    for router in (system.router, auth.router, users.router, cameras.router,
                   policies.router, events.router, live.router, analytics.router,
                   search.router, gateway.router):
        app.include_router(router, prefix="/api")

    for path, directory in (
        ("/snapshots", config.snapshots_dir),
        ("/clips", config.clips_dir),
        ("/tracks", config.tracks_dir),
        # Phase 3: thumbnails for the periodic index samples. A separate mount
        # from /snapshots because these are not evidence — they are 15 KB
        # contact-sheet JPEGs with their own retention window, and an operator
        # reading the disk report should be able to tell the two apart.
        ("/index", config.index_dir),
    ):
        app.mount(path, StaticFiles(directory=str(directory)), name=path.strip("/"))

    info = {
        "name": "Argus",
        "version": __version__,
        "docs": "/api/docs",
        "health": "/api/health",
    }

    @app.get("/api", include_in_schema=False)
    def api_root() -> JSONResponse:
        return JSONResponse(info)

    # The SPA is mounted last so every route above wins: "/" as a mount would
    # otherwise swallow /api and the evidence directories. A checkout with no
    # `npm run build` behind it serves the JSON index instead, rather than
    # failing to start over a missing directory.
    spa = config.ui_dist_dir
    if (spa / "index.html").is_file():
        app.mount("/", SpaFiles(directory=str(spa), html=True), name="ui")
    else:
        @app.get("/", include_in_schema=False)
        def root() -> JSONResponse:
            return JSONResponse({**info, "ui": f"not built — run `npm run build` in {spa.parent}"})

    return app


def run(config: ArgusConfig | None = None, database: Database | None = None, *,
        host: str | None = None, port: int | None = None, **uvicorn_kwargs: Any) -> None:
    """Serve the API with uvicorn. This is what ``argus api`` calls."""
    import uvicorn

    config = config or load_config()
    uvicorn.run(
        create_app(config, database),
        host=host or config.api.host,
        port=port or config.api.port,
        access_log=False,
        **uvicorn_kwargs,
    )


__all__ = ["create_app", "run"]
