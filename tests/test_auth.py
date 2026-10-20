"""Authentication, roles, sessions and the one authenticated door.

The centrepiece is :data:`MATRIX` — every route the API exposes, with the
permission it demands, driven against all four roles. It is written as a table
on purpose: an authorisation model that has to be reconstructed by reading four
hundred assertions is an authorisation model nobody re-reads, and the failure
mode of RBAC is not "this test broke", it is "nobody noticed the new router had
no dependency on it". :func:`test_every_route_is_in_the_matrix` closes that
loop by walking the app's own route table.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from argus import accounts, auth as core
from argus.api.app import create_app
from argus.config import ArgusConfig
from argus.db import Camera, Database, Event, Policy, Zone
from tests.conftest import TEST_PASSWORD

SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


@pytest.fixture(autouse=True)
def _no_real_systemctl(monkeypatch):
    """`/api/system/processing` actuates the engine + VLM via systemctl. These
    tests are about *who may call it*, not the side effect, so keep the write
    from touching a real user unit: with systemctl reported unavailable the flag
    is persisted (to the tmp data dir) and nothing is started or stopped."""
    from argus import processing, services

    monkeypatch.setattr(processing, "systemctl_available", lambda: False)
    monkeypatch.setattr(services, "systemctl_available", lambda: False)


@pytest.fixture
def app(config: ArgusConfig, database: Database):
    config.api.sse_poll_sec = 0.05
    return create_app(config, database)


@pytest.fixture
def anon(app) -> TestClient:
    return TestClient(app)


@pytest.fixture
def admin(anon: TestClient, sign_in) -> TestClient:
    sign_in(anon, "ADMIN")
    return anon


@pytest.fixture
def seeded(database: Database) -> dict[str, int | str]:
    """One camera, one zone, one policy, one reviewable event, one saved search
    and one spare account — so a route under test answers 200 rather than 404,
    and a 403 is therefore unambiguous."""
    database.upsert_camera(Camera(id="line_a", name="Line A", source="rtsp://cam/a"))
    database.upsert_zone(Zone(id="line_a_pack", camera_id="line_a", name="Packing",
                              polygon=SQUARE, sanctioned_limit=3))
    database.upsert_policy(Policy(id="p1", kind="detector",
                                  detector_key="manpower_headcount", label="P1",
                                  stage="reviewed", lifecycle="production",
                                  routed_to="safety_manager"))
    event_id = database.insert_event(Event(
        camera_id="line_a", zone_id="line_a_pack", policy_id="p1", detector_name="p1",
        detector_type="detector", severity="high", description="over limit",
        ts=time.time(), stage="reviewed",
    ))
    search_id = database.create_saved_search("dock", "person on the dock")
    # A body for the account routes to act on that is not the caller: revoking
    # your own sessions mid-matrix would 401 every row after it.
    spare = accounts.create_user(database, "spare", TEST_PASSWORD, role="REVIEW")
    return {"event": event_id, "search": search_id, "user": int(spare["id"])}


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------
#
# (method, path, permission, body). `permission` is what the route demands;
# `None` means "a session, and nothing more". Paths are formatted against the
# `seeded` fixture's ids.

VIEW, TRIAGE, MANAGE, ADMIN, SYSTEM = (
    core.VIEW, core.TRIAGE, core.MANAGE, core.ADMIN, core.SYSTEM
)

MATRIX: list[tuple[str, str, str, dict | None]] = [
    # system — the only permission a SYSTEM monitor holds
    ("GET", "/api/config", SYSTEM, None),
    # The pause switch is readable by any signed-in role (a dark wall needs
    # explaining to whoever is watching it), but actuating it — a plant-wide
    # stop — is ADMIN, like the accounts it sits beside. Body pauses, never
    # resumes: with actuation stubbed out this only flips the tmp flag, and a
    # pause can never accidentally start a real unit.
    ("GET", "/api/system/processing", SYSTEM, None),
    ("POST", "/api/system/processing", ADMIN, {"enabled": False}),

    # fleet
    ("GET", "/api/cameras", VIEW, None),
    ("GET", "/api/cameras/line_a", VIEW, None),
    ("GET", "/api/cameras/line_a/zones", VIEW, None),
    ("GET", "/api/zones", VIEW, None),
    ("GET", "/api/zones/line_a_pack", VIEW, None),
    ("POST", "/api/cameras", MANAGE, {"name": "New", "source": "rtsp://cam/new"}),
    ("PATCH", "/api/cameras/line_a", MANAGE, {"name": "Renamed"}),
    ("POST", "/api/cameras/line_a/zones", MANAGE, {"name": "Z2", "polygon": SQUARE}),
    ("PATCH", "/api/zones/line_a_pack", MANAGE, {"sanctioned_limit": 4}),

    # policies
    ("GET", "/api/policies", VIEW, None),
    ("GET", "/api/policies/p1", VIEW, None),
    ("GET", "/api/policies/p1/analytics", VIEW, None),
    ("POST", "/api/policies", MANAGE,
     {"kind": "detector", "detector_key": "manpower_headcount", "label": "New"}),
    ("PATCH", "/api/policies/p1", MANAGE, {"label": "Renamed"}),
    ("POST", "/api/policies/p1/pilot", MANAGE, None),
    ("POST", "/api/policies/p1/pilot/extend", MANAGE, {"extra_days": 1}),
    # The row that was missing while the coverage walker below was blind, and
    # the most consequential write on the appliance: promotion is what starts
    # sending mail to a safety manager without a human in the loop.
    ("POST", "/api/policies/p1/promote", MANAGE, {"stage": "reviewed"}),
    ("DELETE", "/api/policies/p1", MANAGE, None),

    # events and evidence
    ("GET", "/api/events", VIEW, None),
    ("GET", "/api/events/{event}", VIEW, None),
    ("GET", "/api/triage", VIEW, None),
    ("GET", "/api/accuracy", VIEW, None),
    ("GET", "/api/occupancy", VIEW, None),
    ("GET", "/api/notifications", VIEW, None),
    ("POST", "/api/events/{event}/review", TRIAGE, {"verdict": "confirmed"}),
    # One detection inside an event, not the whole alert. The seeded event carries
    # no such track, so a permitted role gets a 422 — which is the point of the
    # matrix: the permission must be decided before the handler looks anything up.
    ("POST", "/api/events/{event}/boxes/4242/correction", TRIAGE,
     {"track_id": 4242, "verdict": "false_positive"}),

    # live and analytics
    ("GET", "/api/live/state", VIEW, None),
    ("GET", "/api/analytics/summary", VIEW, None),

    # search
    ("GET", "/api/search?q=a%20forklift", VIEW, None),
    ("GET", "/api/search/status", VIEW, None),
    ("GET", "/api/searches", VIEW, None),
    ("POST", "/api/searches", TRIAGE, {"query": "a forklift in the aisle"}),
    ("PATCH", "/api/searches/{search}", TRIAGE, {"label": "renamed"}),
    ("POST", "/api/searches/{search}/promote", MANAGE, None),
    ("DELETE", "/api/searches/{search}", TRIAGE, None),

    # accounts
    ("GET", "/api/users", ADMIN, None),
    ("GET", "/api/users/{user}", ADMIN, None),
    ("POST", "/api/users", ADMIN,
     {"username": "extra", "password": TEST_PASSWORD, "role": "REVIEW"}),
    ("PATCH", "/api/users/{user}", ADMIN, {"display_name": "Renamed"}),
    ("POST", "/api/users/{user}/sessions/revoke", ADMIN, None),

    # authoring (the policy scope, the enrolment probe, the prompt test)
    ("POST", "/api/cameras/probe", MANAGE, {"camera_id": "line_a"}),
    # The prompt test spends a GPU inference and creates a policy's wording:
    # MANAGE, not VIEW. It answers 200 with an SSE `error` frame when there is
    # no engine, which is why a role that may call it is not a 403 here.
    ("POST", "/api/policies/test", MANAGE,
     {"camera_id": "line_a", "prompt": "anything", "wait_sec": 1}),

    # the gateway relay: signalling and posters are as sensitive as the footage
    ("GET", "/api/gateway/frame.jpeg?src=line_a", VIEW, None),
    ("GET", "/api/gateway/streams", VIEW, None),

    # The fleet deletions go last: they cascade to the event and the zone, and a
    # row after them would be asking about something that no longer exists. A
    # 404 still satisfies "not 403", but a matrix that relies on that is a
    # matrix whose failures are hard to read.
    ("DELETE", "/api/zones/line_a_pack", MANAGE, None),
    ("DELETE", "/api/cameras/line_a", MANAGE, None),
]

ROLE_GRANTS = {
    "ADMIN": {VIEW, TRIAGE, MANAGE, ADMIN, SYSTEM},
    "RESPOND": {VIEW, TRIAGE, MANAGE, SYSTEM},
    "REVIEW": {VIEW, TRIAGE, SYSTEM},
    "SYSTEM": {SYSTEM},
}


#: MATRIX paths name the `seeded` fixture's rows so a route answers 200 rather
#: than 404; the app's own route table names its path parameters. This is the
#: dictionary between the two, and it is what makes
#: `test_every_route_is_in_the_matrix` compare like with like.
TEMPLATES = {
    "line_a": "{camera_id}",
    "line_a_pack": "{zone_id}",
    "p1": "{policy_id}",
    "{event}": "{event_id}",
    "4242": "{track_id}",
    "{search}": "{search_id}",
    "{user}": "{user_id}",
}


def as_template(path: str) -> str:
    segments = path.split("?")[0].split("/")
    return "/".join(TEMPLATES.get(segment, segment) for segment in segments)


def call(client: TestClient, method: str, path: str, body: dict | None, ids: dict):
    return client.request(method, path.format(**ids), json=body)


@pytest.mark.parametrize("role", sorted(ROLE_GRANTS))
def test_the_authorisation_matrix(app, sign_in, seeded, role: str):
    """Every route, against every role, in one table.

    A route the role may reach must not answer 403; a route it may not must
    answer exactly 403 — not 404, not 422, not 500. That distinction is the
    test: a permission check that runs *after* the handler has already looked
    something up leaks whether the thing exists.
    """
    permitted = ROLE_GRANTS[role]
    client = TestClient(app)
    sign_in(client, role, f"{role.lower()}_probe")
    for method, path, permission, body in MATRIX:
        response = call(client, method, path, body, seeded)
        allowed = permission is None or permission in permitted
        where = f"{role} {method} {path} -> {response.status_code}"
        if allowed:
            assert response.status_code != 403, where
            assert response.status_code != 401, where
        else:
            assert response.status_code == 403, f"{where} (expected 403)\n{response.text}"


def app_routes(app) -> list[tuple[str, str]]:
    """Every (method, path) the app will actually route, prefixes applied.

    ``app.routes`` used to be a flat list of leaf routes. Since FastAPI 0.139 an
    ``include_router`` leaves a lazy ``_IncludedRouter`` in that list instead,
    holding the original router and the prefix it was mounted under — so a
    walker that reads ``route.path`` sees ``None`` for every routed endpoint on
    the appliance and iterates nothing at all.

    That is not a cosmetic difference: it is exactly this repository's guard
    against a router arriving with no permission on it, and it was passing
    vacuously. Recursing through the wrapper restores it, and the fallback
    branch keeps this working on a FastAPI that flattens.
    """
    found: list[tuple[str, str]] = []

    def walk(routes, prefix: str = "") -> None:
        for route in routes:
            included = getattr(route, "original_router", None)
            if included is not None:
                context = getattr(route, "include_context", None)
                walk(included.routes, prefix + (getattr(context, "prefix", "") or ""))
                continue
            path = getattr(route, "path", None)
            if path is None:
                continue
            for method in sorted(getattr(route, "methods", None) or {"WEBSOCKET"}):
                found.append((method, prefix + path))

    walk(app.routes)
    return found


def test_the_route_walker_sees_the_whole_app(app):
    """Guard on the guard below. If this ever returns a handful of routes, the
    coverage test underneath it has stopped testing anything."""
    paths = {path for _method, path in app_routes(app)}
    assert {"/api/policies", "/api/policies/{policy_id}", "/api/cameras",
            "/api/events/stream", "/api/users"} <= paths
    assert len(paths) > 30, paths


def test_every_route_is_in_the_matrix(app):
    """The matrix must cover the app, not the other way round.

    This is the assertion that survives a Phase-6 router being added without a
    permission on it: a new path shows up here as a failure naming itself,
    rather than as an endpoint quietly reachable by a SYSTEM monitor account.
    """
    from argus.api.auth import PUBLIC_PATHS

    covered = {(method, as_template(path)) for method, path, _, _ in MATRIX}
    # Public by design (documented in argus/api/auth.py), plus the two routes
    # whose whole job is to be reachable without a session, plus FastAPI's own
    # documentation endpoints.
    exempt = PUBLIC_PATHS | {"/api/docs", "/api/docs/oauth2-redirect", "/api/openapi.json",
                             # Covered by their own tests: neither is a
                             # permission question. Logout and the self-service
                             # password change need a session and nothing else.
                             "/api/auth/logout", "/api/auth/password",
                             # A row here would hold the matrix open on a feed
                             # that never ends; `test_the_event_stream_needs_view`
                             # asserts its permission without reading it.
                             "/api/events/stream"}

    missing = []
    for method, path in app_routes(app):
        if not path.startswith("/api") or path in exempt:
            continue
        if method in ("HEAD", "OPTIONS"):
            continue
        if method == "WEBSOCKET":
            continue  # exercised by the relay tests below
        if (method, path) not in covered:
            missing.append(f"{method} {path}")
    assert not missing, (
        "routes with no row in MATRIX — add one, and give the route a "
        f"permission dependency if it has none: {missing}"
    )


def test_the_event_stream_needs_view(app, sign_in):
    """Asserted here rather than in the matrix, because a permitted call would
    hold the table open on a feed with no end. A SYSTEM monitor is refused
    before the response begins, so this one returns."""
    client = TestClient(app)
    sign_in(client, "SYSTEM", "system_stream")
    assert client.get("/api/events/stream").status_code == 403


@pytest.mark.parametrize("role", sorted(ROLE_GRANTS))
def test_every_role_can_end_its_own_session(app, sign_in, role: str):
    """Logout is the one authenticated route with no permission on it: a SYSTEM
    monitor that cannot read an event must still be able to sign out."""
    client = TestClient(app)
    sign_in(client, role)
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/session").status_code == 401


def test_a_role_is_told_what_it_lacks(app, sign_in, seeded):
    """A 403 that says "forbidden" produces a bug report; one that names the
    permission produces a request for the right role."""
    client = TestClient(app)
    sign_in(client, "REVIEW")
    response = client.post("/api/policies", json={
        "kind": "detector", "detector_key": "manpower_headcount", "label": "X",
    })
    assert response.status_code == 403
    assert "REVIEW" in response.json()["detail"]
    assert "manage" in response.json()["detail"]


def test_role_permissions_match_the_openapi_enum():
    """`schemas.Role` is spelled out so clients generate an enum; this is what
    stops it drifting from `argus.auth.ROLES`."""
    from argus.api.schemas import Role

    assert set(Role.__args__) == set(core.ROLES) == set(core.ROLE_PERMISSIONS)


# --------------------------------------------------------------------------
# what is reachable without a session
# --------------------------------------------------------------------------


def test_nothing_answers_without_a_session(anon: TestClient, make_user, seeded):
    make_user("ADMIN")
    for method, path, _permission, body in MATRIX:
        response = call(anon, method, path, body, seeded)
        assert response.status_code == 401, f"{method} {path} answered {response.status_code}"


def test_evidence_mounts_are_behind_the_session(anon: TestClient, config: ArgusConfig,
                                                make_user, sign_in):
    """The static mounts are the most sensitive bytes on the box and no FastAPI
    dependency can reach them — this is what the middleware is for."""
    (config.snapshots_dir / "line_a").mkdir(parents=True, exist_ok=True)
    (config.snapshots_dir / "line_a" / "evt.jpg").write_bytes(b"\xff\xd8\xff\xdb-not-really")
    for mount in ("/snapshots/line_a/evt.jpg", "/clips/x.mp4", "/tracks/x.json",
                  "/index/x.jpg"):
        assert anon.get(mount).status_code == 401, mount

    sign_in(anon, "ADMIN")
    assert anon.get("/snapshots/line_a/evt.jpg").status_code == 200
    # Still a real 404 for a file that is not there, rather than a 401 that
    # would tell a logged-in reviewer their session had died.
    assert anon.get("/clips/nope.mp4").status_code == 404


def test_a_system_account_cannot_download_the_evidence(app, config: ArgusConfig, sign_in):
    """Found by running it, not by reading it.

    The middleware asked only "is there a session" on the static mounts, so a
    SYSTEM monitoring account was refused every event route and could still
    fetch the JPEG those events pointed at. "Health and metrics only, no
    footage" has to be true of the footage.
    """
    (config.snapshots_dir / "line_a").mkdir(parents=True, exist_ok=True)
    (config.snapshots_dir / "line_a" / "evt.jpg").write_bytes(b"\xff\xd8\xff\xdb-not-really")

    monitor = TestClient(app)
    sign_in(monitor, "SYSTEM")
    assert monitor.get("/api/health").status_code == 200      # its whole job
    for mount in ("/snapshots/line_a/evt.jpg", "/clips/x.mp4", "/tracks/x.json",
                  "/index/x.jpg"):
        response = monitor.get(mount)
        assert response.status_code == 403, mount
        assert "SYSTEM" in response.json()["detail"]

    reviewer = TestClient(app)
    sign_in(reviewer, "REVIEW")
    assert reviewer.get("/snapshots/line_a/evt.jpg").status_code == 200


def test_health_is_liveness_only_until_you_sign_in(anon: TestClient, make_user, sign_in):
    """Monitoring can watch the process; it cannot enumerate the fleet."""
    anonymous = anon.get("/api/health")
    assert anonymous.status_code == 200
    assert anonymous.json()["status"] == "ok"
    assert anonymous.json()["authenticated"] is False
    assert "cameras" not in anonymous.json()
    assert "gpu" not in anonymous.json()

    sign_in(anon, "ADMIN")
    full = anon.get("/api/health").json()
    assert full["authenticated"] is True
    assert "cameras" in full and "gpu" in full and "disk" in full


def test_a_box_with_no_accounts_says_what_to_run(anon: TestClient):
    body = anon.get("/api/events").json()
    assert body["code"] == "no_accounts"
    assert "argus user add" in body["detail"]
    assert anon.get("/api/health").json()["accounts"] == 0


def test_the_spa_shell_is_public_so_there_is_a_login_screen(config: ArgusConfig,
                                                            database: Database):
    """There has to be something to serve the login form. The shell holds no
    plant data — every byte of that arrives over /api."""
    from argus.api.auth import needs_session

    assert not needs_session("/")
    assert not needs_session("/index.html")
    assert not needs_session("/assets/main-abc123.js")
    # ...but the sample-thumbnail mount is not the SPA's index.html.
    assert needs_session("/index/line_a/1.jpg")
    assert needs_session("/api/events")
    assert needs_session("/snapshots/a/b.jpg")


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------


def _backdate(database: Database, column: str, minutes: float) -> None:
    when = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    with database.conn() as conn:
        conn.execute(f"UPDATE sessions SET {column} = ?", (when,))  # noqa: S608 - test-local literal


def test_a_session_expires_at_its_absolute_deadline(anon: TestClient, database: Database,
                                                    sign_in):
    sign_in(anon, "ADMIN")
    assert anon.get("/api/events").status_code == 200

    with database.conn() as conn:
        conn.execute("UPDATE sessions SET expires_at = ?",
                     ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),))
    response = anon.get("/api/events")
    assert response.status_code == 401
    assert response.json()["code"] == "session_required"
    # The dead row is removed rather than left to accumulate for the life of
    # the appliance — nobody will ever present that cookie again.
    assert database.list_sessions_for_user(1) == []


def test_a_session_expires_after_sitting_idle(anon: TestClient, config: ArgusConfig,
                                              database: Database, sign_in):
    config.auth.session_idle_minutes = 30
    sign_in(anon, "ADMIN")
    _backdate(database, "last_seen_at", minutes=31)
    assert anon.get("/api/events").status_code == 401


def test_the_idle_timeout_can_be_turned_off_for_a_wall_display(
    anon: TestClient, config: ArgusConfig, database: Database, sign_in
):
    """A control-room panel nobody touches for a whole shift is a real
    deployment, and logging it out mid-shift is the worse failure."""
    config.auth.session_idle_minutes = 0
    sign_in(anon, "ADMIN")
    _backdate(database, "last_seen_at", minutes=60 * 24 * 30)
    assert anon.get("/api/events").status_code == 200


def test_activity_pushes_the_idle_deadline_out(anon: TestClient, config: ArgusConfig,
                                               database: Database, sign_in):
    config.auth.session_idle_minutes = 30
    config.auth.session_touch_sec = 1.0
    sign_in(anon, "ADMIN")
    _backdate(database, "last_seen_at", minutes=20)
    assert anon.get("/api/events").status_code == 200        # rolls last_seen_at forward
    _backdate(database, "last_seen_at", minutes=20)
    assert anon.get("/api/events").status_code == 200        # would have died without the roll


def test_the_cookie_value_is_never_stored(anon: TestClient, database: Database, sign_in):
    """A backup, a support bundle or an operator with sqlite3 must not be able
    to lift a live session out of the appliance database."""
    sign_in(anon, "ADMIN")
    token = anon.cookies["argus_session"]
    stored = [row["id"] for row in database.list_sessions_for_user(1)]
    assert stored and token not in stored
    assert stored[0] == core.token_digest(token)
    assert len(stored[0]) == 64                       # sha256 hex


def test_the_cookie_is_httponly(anon: TestClient, sign_in):
    """The token has to be unreachable from JavaScript: it is the difference
    between an XSS bug showing the wrong number and one exfiltrating a session
    that can download the plant's footage."""
    response = anon.post("/api/auth/login",
                         json={"username": "admin", "password": TEST_PASSWORD}) \
        if False else None
    from argus import accounts as acc

    acc.create_user(anon.app.state.db, "ops", TEST_PASSWORD, role="ADMIN")
    response = anon.post("/api/auth/login", json={"username": "ops", "password": TEST_PASSWORD})
    header = response.headers["set-cookie"].lower()
    assert "httponly" in header
    assert "samesite=lax" in header
    assert "path=/" in header


def test_logout_kills_the_session_server_side(anon: TestClient, database: Database, sign_in):
    """Clearing the cookie alone would leave a token that still works for
    whoever captured it — which is the population a logout button exists for."""
    sign_in(anon, "ADMIN")
    token = anon.cookies["argus_session"]
    assert anon.post("/api/auth/logout").status_code == 200
    assert database.get_session(core.token_digest(token)) is None
    anon.cookies.set("argus_session", token)
    assert anon.get("/api/events").status_code == 401


def test_disabling_an_account_cuts_its_sessions_at_once(anon: TestClient, app,
                                                        database: Database, sign_in):
    sign_in(anon, "ADMIN", "boss")
    reviewer = TestClient(app)
    sign_in(reviewer, "REVIEW", "rc")
    assert reviewer.get("/api/events").status_code == 200

    user = database.get_user_by_username("rc")
    assert anon.patch(f"/api/users/{user['id']}", json={"disabled": True}).status_code == 200
    assert reviewer.get("/api/events").status_code == 401
    assert database.list_sessions_for_user(int(user["id"])) == []


def test_changing_a_password_ends_every_session_for_it(anon: TestClient, app,
                                                       database: Database, sign_in):
    sign_in(anon, "ADMIN", "boss")
    victim = TestClient(app)
    sign_in(victim, "REVIEW", "rc")
    user = database.get_user_by_username("rc")

    anon.patch(f"/api/users/{user['id']}", json={"password": "a-brand-new-password"})
    assert victim.get("/api/events").status_code == 401


def test_a_role_change_takes_effect_without_a_new_login(anon: TestClient, app,
                                                        database: Database, sign_in):
    """The session row carries no role — it joins to `users` on every request —
    so a demotion cannot be outrun by holding a cookie."""
    sign_in(anon, "ADMIN", "boss")
    user = database.get_user_by_username("boss")
    accounts.create_user(database, "second", TEST_PASSWORD, role="ADMIN")
    anon.patch(f"/api/users/{user['id']}", json={"role": "REVIEW"})
    # Sessions are cut on a role change too, so this is a fresh login as REVIEW.
    anon.post("/api/auth/login", json={"username": "boss", "password": TEST_PASSWORD})
    assert anon.get("/api/auth/session").json()["user"]["role"] == "REVIEW"
    assert anon.get("/api/users").status_code == 403


# --------------------------------------------------------------------------
# login
# --------------------------------------------------------------------------


def test_login_failures_are_indistinguishable(anon: TestClient, make_user):
    """"No such user", "wrong password" and "account disabled" are one answer:
    the difference between them is a list of who works at this plant."""
    user = make_user("REVIEW", "rc")
    wrong_password = anon.post("/api/auth/login",
                               json={"username": "rc", "password": "not-the-password"})
    no_such_user = anon.post("/api/auth/login",
                             json={"username": "ghost", "password": "not-the-password"})
    accounts.set_disabled(anon.app.state.db, int(user["id"]), True)
    disabled = anon.post("/api/auth/login",
                         json={"username": "rc", "password": TEST_PASSWORD})

    bodies = {r.json()["detail"] for r in (wrong_password, no_such_user, disabled)}
    statuses = {r.status_code for r in (wrong_password, no_such_user, disabled)}
    assert bodies == {"invalid username or password"}
    assert statuses == {401}


def test_login_is_rate_limited_per_account(anon: TestClient, config: ArgusConfig,
                                           make_user):
    config.auth.login_max_attempts = 3
    make_user("REVIEW", "rc")
    for _ in range(3):
        assert anon.post("/api/auth/login",
                         json={"username": "rc", "password": "wrong"}).status_code == 401
    locked = anon.post("/api/auth/login", json={"username": "rc", "password": "wrong"})
    assert locked.status_code == 429
    assert "Retry-After" in locked.headers
    # And the *right* password is refused too, or the lockout is decoration.
    assert anon.post("/api/auth/login",
                     json={"username": "rc", "password": TEST_PASSWORD}).status_code == 429


def test_the_rate_limit_also_counts_the_source_address(anon: TestClient,
                                                       config: ArgusConfig, make_user):
    """Locking only on username lets one client walk the roster a few passwords
    at a time. Six wrong guesses against six different accounts is still six
    wrong guesses from one place."""
    config.auth.login_max_attempts = 3
    for name in ("aa", "bb", "cc"):
        make_user("REVIEW", name)
        anon.post("/api/auth/login", json={"username": name, "password": "wrong"})
    make_user("REVIEW", "dd")
    assert anon.post("/api/auth/login",
                     json={"username": "dd", "password": TEST_PASSWORD}).status_code == 429


def test_a_lockout_ages_out_of_the_window(anon: TestClient, config: ArgusConfig,
                                          database: Database, make_user):
    config.auth.login_max_attempts = 2
    config.auth.login_window_sec = 300
    make_user("REVIEW", "rc")
    for _ in range(2):
        anon.post("/api/auth/login", json={"username": "rc", "password": "wrong"})
    assert anon.post("/api/auth/login",
                     json={"username": "rc", "password": TEST_PASSWORD}).status_code == 429
    with database.conn() as conn:
        conn.execute("UPDATE login_attempts SET created_at = ?",
                     ((datetime.now(timezone.utc) - timedelta(seconds=400)).isoformat(),))
    assert anon.post("/api/auth/login",
                     json={"username": "rc", "password": TEST_PASSWORD}).status_code == 200


def test_a_successful_login_clears_the_counter(anon: TestClient, config: ArgusConfig,
                                               make_user):
    """The operator who fat-fingered their password four times must not stay
    one keystroke from a lockout after getting it right."""
    config.auth.login_max_attempts = 3
    make_user("REVIEW", "rc")
    for _ in range(2):
        anon.post("/api/auth/login", json={"username": "rc", "password": "wrong"})
    assert anon.post("/api/auth/login",
                     json={"username": "rc", "password": TEST_PASSWORD}).status_code == 200
    for _ in range(2):
        anon.post("/api/auth/login", json={"username": "rc", "password": "wrong"})
    assert anon.post("/api/auth/login",
                     json={"username": "rc", "password": TEST_PASSWORD}).status_code == 200


def test_login_attempts_are_recorded_both_ways(anon: TestClient, database: Database,
                                               make_user):
    """"Who logged in, from where, when" is the first question after an
    incident, and a table holding only the failures cannot answer it."""
    make_user("REVIEW", "rc")
    anon.post("/api/auth/login", json={"username": "rc", "password": "wrong"})
    anon.post("/api/auth/login", json={"username": "rc", "password": TEST_PASSWORD})
    attempts = database.recent_login_attempts()
    assert [row["ok"] for row in attempts][:1] == [1]
    assert all(row["username"] == "rc" for row in attempts)


def test_changing_your_own_password_needs_the_old_one(anon: TestClient, sign_in):
    sign_in(anon, "REVIEW", "rc")
    refused = anon.post("/api/auth/password",
                        json={"current_password": "guessing", "new_password": "another-long-one"})
    assert refused.status_code == 403
    accepted = anon.post("/api/auth/password",
                         json={"current_password": TEST_PASSWORD,
                               "new_password": "another-long-one"})
    assert accepted.status_code == 200
    # Signed out everywhere, including here — which is also how the operator
    # finds out it took effect.
    assert anon.get("/api/events").status_code == 401
    assert anon.post("/api/auth/login",
                     json={"username": "rc", "password": "another-long-one"}).status_code == 200


def test_a_short_password_is_refused_with_the_rule(anon: TestClient, sign_in):
    sign_in(anon, "ADMIN")
    response = anon.post("/api/users", json={"username": "weak", "password": "short",
                                             "role": "REVIEW"})
    assert response.status_code == 422


# --------------------------------------------------------------------------
# accounts
# --------------------------------------------------------------------------


def test_the_roster_never_carries_a_password_hash(anon: TestClient, sign_in):
    sign_in(anon, "ADMIN")
    body = anon.get("/api/users").text
    assert "password_hash" not in body
    assert "$argon2" not in body


def test_the_last_admin_cannot_lock_the_plant_out(anon: TestClient, database: Database,
                                                  sign_in):
    """An appliance whose only administrator disabled themselves needs a person
    with shell access and a sqlite3 prompt. That is a support call generated by
    a UI click."""
    sign_in(anon, "ADMIN", "boss")
    user = database.get_user_by_username("boss")
    disabled = anon.patch(f"/api/users/{user['id']}", json={"disabled": True})
    assert disabled.status_code == 409
    demoted = anon.patch(f"/api/users/{user['id']}", json={"role": "REVIEW"})
    assert demoted.status_code == 409

    # With a second administrator, both become allowed.
    accounts.create_user(database, "deputy", TEST_PASSWORD, role="ADMIN")
    assert anon.patch(f"/api/users/{user['id']}", json={"role": "REVIEW"}).status_code == 200


def test_usernames_are_case_insensitive(database: Database):
    """`J.Whitfield` and `j.whitfield` must be one account, not two with
    different permissions."""
    accounts.create_user(database, "J.Whitfield", TEST_PASSWORD, role="ADMIN")
    with pytest.raises(accounts.AccountError) as raised:
        accounts.create_user(database, "j.whitfield", TEST_PASSWORD, role="REVIEW")
    assert raised.value.status == 409


def test_a_verdict_is_attributed_to_the_session(anon: TestClient, database: Database,
                                                seeded, sign_in):
    sign_in(anon, "REVIEW", "rc")
    anon.post(f"/api/events/{seeded['event']}/review",
              json={"verdict": "confirmed", "reviewer": "somebody-else"})
    assert database.get_review(int(seeded["event"]))["reviewer"] == "rc"


def test_revoking_sessions_leaves_the_account_alone(anon: TestClient, app,
                                                    database: Database, sign_in):
    """What an admin wants when a laptop goes missing: the person keeps their
    account and their password, and every cookie in the wild stops working."""
    sign_in(anon, "ADMIN", "boss")
    reviewer = TestClient(app)
    sign_in(reviewer, "REVIEW", "rc")
    user = database.get_user_by_username("rc")

    anon.post(f"/api/users/{user['id']}/sessions/revoke")
    assert reviewer.get("/api/events").status_code == 401
    assert reviewer.post("/api/auth/login",
                         json={"username": "rc", "password": TEST_PASSWORD}).status_code == 200


def test_retention_sweeps_dead_sessions_and_old_attempts(config: ArgusConfig,
                                                         database: Database, make_user):
    """Neither table is bounded by a `retention.*_days` window on purpose — the
    deadlines are the auth settings — but both grow forever without this."""
    from argus.retention import Retention

    make_user("ADMIN")
    database.create_session("dead", 1, (datetime.now(timezone.utc)
                                        - timedelta(hours=1)).isoformat())
    database.create_session("live", 1, (datetime.now(timezone.utc)
                                        + timedelta(hours=1)).isoformat())
    database.record_login_attempt("rc", "10.0.0.9", False)
    with database.conn() as conn:
        conn.execute("UPDATE login_attempts SET created_at = ?",
                     ((datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),))

    result = Retention(config, database, log=lambda *_: None).sweep()
    assert result.by_kind["expired_sessions"] == 1
    assert result.by_kind["login_attempts"] == 1
    assert [row["id"] for row in database.list_sessions_for_user(1)] == ["live"]


# --------------------------------------------------------------------------
# the one authenticated door
# --------------------------------------------------------------------------
#
# A stand-in for go2rtc, served by uvicorn on a real port, with the two
# endpoints the console uses. Faking it rather than starting the vendored
# binary keeps this a test of the *relay* — the authorisation, the src
# allowlist, the two-way pump — instead of a test of go2rtc, which has its own.


#: What the fake saw, so a test can assert the relay passed the right `src`
#: through — and, more importantly, that it passed *nothing* through when the
#: src was not a camera.
GATEWAY_SEEN: list[str] = []


def build_fake_go2rtc() -> FastAPI:
    """The two endpoints the console uses, with go2rtc's own message shapes.

    Built at module scope rather than inside the fixture on purpose: this file
    has `from __future__ import annotations`, so every annotation is a string,
    and FastAPI resolves them against the *module* globals. A route function
    defined inside a fixture would have `WebSocket` resolve to nothing, FastAPI
    would take the parameter for a query field, and the handshake would be
    rejected 403 before the route ever ran — which is exactly what happened.
    """
    upstream = FastAPI()

    @upstream.get("/api/frame.jpeg")
    def frame(src: str) -> Response:
        GATEWAY_SEEN.append(src)
        return Response(content=b"JPEGBYTES:" + src.encode(), media_type="image/jpeg")

    @upstream.get("/api/streams")
    def streams() -> dict:
        return {"line_a": {"producers": 1}}

    @upstream.websocket("/api/ws")
    async def signalling(websocket: WebSocket, src: str = "") -> None:
        await websocket.accept()
        GATEWAY_SEEN.append(src)
        # go2rtc's shape: an offer comes in, an answer and a candidate go out.
        await websocket.receive_text()
        await websocket.send_text(f'{{"type":"webrtc/answer","value":"sdp-for-{src}"}}')
        await websocket.send_text('{"type":"webrtc/candidate","value":"candidate:1 1 udp"}')
        await websocket.receive_text()               # the browser's candidate

    return upstream


@pytest.fixture
def fake_go2rtc(config: ArgusConfig):
    """A minimal go2rtc on a real port, with the API pointed at it."""
    import threading

    import uvicorn

    GATEWAY_SEEN.clear()
    seen = GATEWAY_SEEN
    server = uvicorn.Server(uvicorn.Config(build_fake_go2rtc(), host="127.0.0.1", port=0,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "fake gateway did not start"
    config.gateway.api_host = "127.0.0.1"
    config.gateway.api_port = server.servers[0].sockets[0].getsockname()[1]
    yield seen
    server.should_exit = True
    thread.join(timeout=10)


def test_the_poster_still_comes_through_the_relay(anon: TestClient, database: Database,
                                                  sign_in, fake_go2rtc):
    database.upsert_camera(Camera(id="line_a", name="Line A", source="rtsp://cam/a"))
    sign_in(anon, "REVIEW")
    response = anon.get("/api/gateway/frame.jpeg", params={"src": "line_a"})
    assert response.status_code == 200
    assert response.content == b"JPEGBYTES:line_a"
    assert response.headers["cache-control"] == "no-store"
    assert fake_go2rtc == ["line_a"]


def test_the_relay_only_serves_cameras_this_appliance_knows(anon: TestClient,
                                                            database: Database,
                                                            sign_in, fake_go2rtc):
    """The `src` allowlist is the cameras table, and it is not decoration.

    go2rtc understands `exec:` and `ffmpeg:` in a source string. Without this
    check an authenticated REVIEW account could hand the relay one and get
    go2rtc to run ffmpeg with arguments of their choosing.
    """
    database.upsert_camera(Camera(id="line_a", name="Line A", source="rtsp://cam/a"))
    sign_in(anon, "REVIEW")
    assert anon.get("/api/gateway/frame.jpeg", params={"src": "ghost"}).status_code == 404
    assert anon.get("/api/gateway/frame.jpeg",
                    params={"src": "exec:ffmpeg -i /etc/passwd"}).status_code == 404
    assert fake_go2rtc == []


def test_a_disabled_camera_says_so_rather_than_404ing(anon: TestClient,
                                                      database: Database, sign_in,
                                                      fake_go2rtc):
    database.upsert_camera(Camera(id="off", name="Off", source="rtsp://cam/off",
                                  enabled=False))
    sign_in(anon, "REVIEW")
    response = anon.get("/api/gateway/frame.jpeg", params={"src": "off"})
    assert response.status_code == 409
    assert "disabled" in response.json()["detail"]


def test_an_unreachable_gateway_is_a_502_not_a_500(anon: TestClient, config: ArgusConfig,
                                                   database: Database, sign_in):
    """The appliance is fine; the thing behind it is not. The console renders
    this sentence rather than guessing at a 500 — and it names both halves,
    because "go2rtc refused" and "the engine has no frame" need different fixes.
    """
    database.upsert_camera(Camera(id="line_a", name="Line A", source="rtsp://cam/a"))
    config.gateway.api_port = 1            # nothing listens on port 1
    sign_in(anon, "REVIEW")
    response = anon.get("/api/gateway/frame.jpeg", params={"src": "line_a"})
    assert response.status_code == 502
    assert "engine is not publishing" in response.json()["detail"]


def test_a_still_the_gateway_cannot_render_comes_from_the_engine(
    anon: TestClient, config: ArgusConfig, database: Database, sign_in
):
    """go2rtc renders this still by decoding the stream, and on 2026-08-25 that
    was measured failing for **every H.265 camera** on this appliance —
    `HTTP 500: exit status 183` for all five of a customer's 1440p HEVC feeds,
    200 for all three H.264 ones. The engine already holds a decoded frame of
    each of those scenes; serving it is the difference between a live wall and
    six black rectangles.
    """
    from argus.live import LiveFrames

    database.upsert_camera(Camera(id="line_a", name="Line A", source="rtsp://cam/a"))
    config.gateway.api_port = 1            # the gateway cannot answer
    live = LiveFrames.from_config(config)
    live.frame_path("line_a").parent.mkdir(parents=True, exist_ok=True)
    live.frame_path("line_a").write_bytes(b"ENGINEJPEG")

    sign_in(anon, "REVIEW")
    response = anon.get("/api/gateway/frame.jpeg", params={"src": "line_a"})
    assert response.status_code == 200
    assert response.content == b"ENGINEJPEG"
    # Labelled, not passed off as live: a stale frame presented as current is
    # the one outcome worse than no frame.
    assert response.headers["x-argus-frame-source"] == "engine"
    assert float(response.headers["x-argus-frame-age"]) >= 0.0


def test_a_gateway_still_is_labelled_as_one(anon: TestClient, database: Database,
                                            sign_in, fake_go2rtc):
    """The console has to be able to tell the two apart, so both say which."""
    database.upsert_camera(Camera(id="line_a", name="Line A", source="rtsp://cam/a"))
    sign_in(anon, "REVIEW")
    response = anon.get("/api/gateway/frame.jpeg", params={"src": "line_a"})
    assert response.headers["x-argus-frame-source"] == "gateway"
    assert "x-argus-frame-age" not in response.headers


def test_webrtc_signalling_is_relayed_both_ways(anon: TestClient, database: Database,
                                                sign_in, fake_go2rtc):
    """The whole point of the relay: go2rtc's own handshake, unchanged, over an
    authenticated same-origin socket. Frames are forwarded verbatim — the relay
    does not parse SDP, so a go2rtc upgrade cannot break it."""
    database.upsert_camera(Camera(id="line_a", name="Line A", source="rtsp://cam/a"))
    sign_in(anon, "REVIEW")
    with anon.websocket_connect("/api/gateway/ws?src=line_a") as socket:
        socket.send_text('{"type":"webrtc/offer","value":"v=0..."}')
        answer = socket.receive_json()
        assert answer["type"] == "webrtc/answer"
        assert answer["value"] == "sdp-for-line_a"
        candidate = socket.receive_json()
        assert candidate["type"] == "webrtc/candidate"
        socket.send_text('{"type":"webrtc/candidate","value":"candidate:2 1 udp"}')
    assert fake_go2rtc == ["line_a"]


def test_signalling_without_a_session_is_refused(anon: TestClient, database: Database,
                                                 make_user, fake_go2rtc):
    """A WebSocket cannot answer 401, so the handshake closes with 1008 — the
    console can tell that apart from a dead gateway, which a bare connection
    failure would not."""
    database.upsert_camera(Camera(id="line_a", name="Line A", source="rtsp://cam/a"))
    make_user("ADMIN")
    with pytest.raises(WebSocketDisconnect) as raised:
        with anon.websocket_connect("/api/gateway/ws?src=line_a") as socket:
            socket.receive_text()
    assert raised.value.code == 1008
    assert fake_go2rtc == []


def test_a_system_account_cannot_watch_the_cameras(anon: TestClient, database: Database,
                                                   sign_in, fake_go2rtc):
    """SYSTEM is health and metrics. An IT monitoring account has no business
    holding a live video connection to the plant floor."""
    database.upsert_camera(Camera(id="line_a", name="Line A", source="rtsp://cam/a"))
    sign_in(anon, "SYSTEM")
    with pytest.raises(WebSocketDisconnect) as raised:
        with anon.websocket_connect("/api/gateway/ws?src=line_a") as socket:
            socket.receive_text()
    assert raised.value.code == 1008
    assert anon.get("/api/gateway/frame.jpeg", params={"src": "line_a"}).status_code == 403


def test_the_generated_gateway_config_keeps_the_api_on_loopback(config: ArgusConfig,
                                                                database: Database):
    """The other half of the door: relaying signalling only helps if go2rtc is
    not still listening on the LAN with `origin: "*"` beside it."""
    from argus.ingest.gateway import build_config

    database.upsert_camera(Camera(id="line_a", name="Line A", source="rtsp://cam/a"))
    rendered = build_config(database.list_cameras(), config)
    assert rendered["api"]["listen"] == f"127.0.0.1:{config.gateway.api_port}"
    assert "origin" not in rendered["api"]


# --------------------------------------------------------------------------
# the CLI bootstrap
# --------------------------------------------------------------------------


def test_argus_user_add_bootstraps_the_first_admin(config: ArgusConfig, database: Database,
                                                   monkeypatch, capsys, tmp_path):
    """The one command an installer runs. It has to work with no accounts, no
    session and no TTY — which is why the password comes from stdin."""
    import io

    from argus.cli import main

    config_file = tmp_path / "argus.yaml"
    config_file.write_text(f"data_dir: {config.data_dir}\n")
    monkeypatch.setattr("sys.stdin", io.StringIO(TEST_PASSWORD + "\n"))
    assert main(["-c", str(config_file), "user", "add", "ops", "--role", "ADMIN",
                 "--name", "J. Whitfield"]) == 0

    user = database.get_user_by_username("ops")
    assert user["role"] == "ADMIN"
    assert user["display_name"] == "J. Whitfield"
    assert core.verify_password(database.credentials_for("ops")["password_hash"], TEST_PASSWORD)


def test_argus_user_list_says_what_to_run_on_an_empty_box(config: ArgusConfig,
                                                          database: Database,
                                                          capsys, tmp_path):
    from argus.cli import main

    config_file = tmp_path / "argus.yaml"
    config_file.write_text(f"data_dir: {config.data_dir}\n")
    assert main(["-c", str(config_file), "user", "list"]) == 1
    assert "argus user add" in capsys.readouterr().out


def test_argus_user_disable_and_role(config: ArgusConfig, database: Database,
                                     monkeypatch, tmp_path):
    import io

    from argus.cli import main

    config_file = tmp_path / "argus.yaml"
    config_file.write_text(f"data_dir: {config.data_dir}\n")
    monkeypatch.setattr("sys.stdin", io.StringIO(TEST_PASSWORD + "\n"))
    main(["-c", str(config_file), "user", "add", "boss", "--role", "ADMIN"])
    monkeypatch.setattr("sys.stdin", io.StringIO(TEST_PASSWORD + "\n"))
    main(["-c", str(config_file), "user", "add", "rc", "--role", "REVIEW"])

    assert main(["-c", str(config_file), "user", "role", "rc", "--role", "RESPOND"]) == 0
    assert database.get_user_by_username("rc")["role"] == "RESPOND"
    assert main(["-c", str(config_file), "user", "disable", "rc"]) == 0
    assert database.get_user_by_username("rc")["disabled"] == 1
    assert main(["-c", str(config_file), "user", "enable", "rc"]) == 0
    assert database.get_user_by_username("rc")["disabled"] == 0
    # And the guard holds from the CLI too.
    assert main(["-c", str(config_file), "user", "disable", "boss"]) == 1
