"""The lock on the API door: session resolution, deny-by-default, permissions.

Two mechanisms, and the split is deliberate.

**A middleware decides whether the request may proceed at all.** It runs before
routing, so it also covers the four ``StaticFiles`` mounts — ``/snapshots``,
``/clips``, ``/tracks``, ``/index`` — which are the most sensitive bytes on the
appliance and which no FastAPI dependency can reach. It is written
deny-by-default over a short allowlist of public paths, so a router added in
Phase 6 is protected by existing to be routed to, rather than by somebody
remembering to decorate it. It also carries the *permission* for those four
mounts (:data:`MOUNT_PERMISSIONS`), because a mount has no route to hang a
dependency on.

**Dependencies decide what an authenticated caller may do.** ``Requires(MANAGE)``
on a route is a statement about that route, belongs next to it, and shows up in
the OpenAPI document. Putting the permission matrix in the middleware instead
would mean expressing "PATCH /api/policies/{id} needs manage but GET does not"
as path regexes, which is how authorisation bugs are written.

The public allowlist is four things and each one is load-bearing:

* ``GET /api/health`` — a liveness probe a plant's monitoring can point at
  without an account. It answers with *liveness only* when unauthenticated; the
  fleet's fps, the GPU and the disk need a session, because "which cameras are
  down" is reconnaissance.
* ``POST /api/auth/login`` — the door itself.
* ``GET /api/auth/session`` — how the console asks "am I logged in?", and it
  must be able to hear "no".
* the SPA bundle (``/``, ``/assets/*``) — there has to be a login screen to
  serve. It contains no plant data; every byte of that arrives over ``/api``.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable

from fastapi import Depends, HTTPException, Request, WebSocket
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from argus import auth as core
from argus.auth import Principal
from argus.config import ArgusConfig
from argus.db import Database

# --------------------------------------------------------------------------
# what is reachable without a session
# --------------------------------------------------------------------------

#: Everything under these prefixes needs a session. Anything else is the SPA
#: shell and its assets. Stated as "what is protected" rather than "what is
#: public" so that adding a router or a mount cannot accidentally publish it.
PROTECTED_PREFIXES: tuple[str, ...] = (
    "/api",
    "/snapshots",
    "/clips",
    "/tracks",
    "/index",
)

#: The four ``StaticFiles`` mounts, and the permission each one demands.
#:
#: These need a permission *here* rather than a dependency on a route, because
#: there is no route — Starlette serves them, which is exactly why they get
#: Range requests and conditional GETs for free, and exactly why a FastAPI
#: dependency cannot reach them.
#:
#: They are ``view`` rather than merely "a session", and that distinction was
#: found by running it: a SYSTEM monitoring account was refused every event
#: route and could still fetch the JPEG those events pointed at, because the
#: middleware only asked whether somebody was signed in. "Health and metrics
#: only, no footage" has to be true of the footage, not just of the routes that
#: describe it.
MOUNT_PERMISSIONS: dict[str, str] = {
    "/snapshots": core.VIEW,
    "/clips": core.VIEW,
    "/tracks": core.VIEW,
    "/index": core.VIEW,
}

#: Exact paths inside those prefixes that answer without one. See the module
#: docstring for why each is here.
PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/api",              # the tiny name/version index
        "/api/health",
        "/api/auth/login",
        "/api/auth/session",
    }
)


def _under(path: str, prefix: str) -> bool:
    """Prefix match on path *segments*. ``/index.html`` is not under ``/index``
    — the SPA's entry point and the sample-thumbnail mount differ by a dot."""
    return path == prefix or path.startswith(prefix + "/")


def needs_session(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return False
    return any(_under(path, prefix) for prefix in PROTECTED_PREFIXES)


def mount_permission(path: str) -> str | None:
    """The permission an evidence mount demands, or None if this is not one."""
    for prefix, permission in MOUNT_PERMISSIONS.items():
        if _under(path, prefix):
            return permission
    return None


# --------------------------------------------------------------------------
# resolving a cookie into a person
# --------------------------------------------------------------------------


def _cookie(headers: Headers, name: str) -> str | None:
    """Read one cookie without building a Request. The middleware runs on every
    static image on a triage screen, so it stays off the heavier machinery."""
    raw = headers.get("cookie")
    if not raw:
        return None
    for part in raw.split(";"):
        key, _, value = part.strip().partition("=")
        if key == name:
            return value or None
    return None


def resolve_principal(
    config: ArgusConfig, db: Database, token: str | None
) -> Principal | None:
    """A cookie value -> who it belongs to, or None.

    Every reason to say no is handled here rather than by the caller, because
    every one of them must look identical from outside: no cookie, an unknown
    session, an expired session, an idle session, a disabled account.

    An expired or idled session is *deleted*, not merely refused. The row has no
    further use, and leaving it means the sessions table grows for the lifetime
    of the appliance and a support bundle carries a year of dead logins.
    """
    if not token:
        return None
    row = db.get_session(core.token_digest(token))
    if row is None:
        return None
    session_id = str(row["session_id"])
    if core.is_expired(str(row["expires_at"])):
        db.delete_session(session_id)
        return None
    if core.idle_deadline_passed(str(row["last_seen_at"]), config.auth.session_idle_minutes):
        db.delete_session(session_id)
        return None
    if int(row["disabled"] or 0):
        # Disabling an account cuts its sessions (see `delete_sessions_for_user`),
        # so this is the belt to that braces — and the path that matters if a
        # row is ever flipped with sqlite3 rather than through the API.
        db.delete_sessions_for_user(int(row["user_id"]))
        return None

    # Roll `last_seen_at` forward, but not on every request: the console polls
    # live state every 2 s and holds an SSE connection that polls every 2 s, and
    # turning each of those into a write would put two writes a second on a
    # WAL database whose one writer is the vision engine.
    if core.idle_deadline_passed(
        str(row["last_seen_at"]), config.auth.session_touch_sec / 60.0
    ):
        db.touch_session(session_id)

    return Principal.from_row(row)


# --------------------------------------------------------------------------
# the middleware
# --------------------------------------------------------------------------


def _unauthenticated(detail: str, *, code: str) -> JSONResponse:
    """401 with a machine-readable reason.

    The console distinguishes these: ``no_accounts`` sends it to a bootstrap
    message naming the command to run, ``session_expired`` sends it to the login
    screen with "your session ended" rather than a bare form the operator will
    read as "it broke".
    """
    return JSONResponse(
        {"detail": detail, "code": code},
        status_code=401,
        headers={"WWW-Authenticate": 'Cookie realm="argus"'},
    )


class AuthMiddleware:
    """Deny-by-default gate in front of routing and the static mounts.

    Raw ASGI rather than ``BaseHTTPMiddleware``: the SSE event stream is a
    long-lived ``StreamingResponse``, and BaseHTTPMiddleware pumps such
    responses through an anyio memory stream, which changes the flush behaviour
    of a feed whose entire job is to arrive promptly. This wrapper touches the
    scope and gets out of the way.
    """

    def __init__(self, app: ASGIApp, *, config: ArgusConfig, database: Database):
        self.app = app
        self.config = config
        self.db = database

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        token = _cookie(headers, self.config.auth.cookie_name)
        principal = resolve_principal(self.config, self.db, token)

        # Stash on the scope's state dict, which is where Starlette's
        # `request.state` reads from. Set even when None, so a route that reads
        # it never has to guard against a missing key.
        state = scope.setdefault("state", {})
        state["principal"] = principal

        path = scope.get("path", "")
        if principal is None and needs_session(path):
            if scope["type"] == "websocket":
                # Closing before the handshake completes is what an ASGI server
                # turns into an HTTP 403, which is the right answer for an
                # unauthenticated caller: there is nothing to tell it beyond
                # "no". The console does not need the detail here — its own
                # session state already knows it is signed out, and the next
                # ordinary request routes it to the login screen. Refusals that
                # a *signed-in* operator needs explained are made after the
                # accept, in argus/api/routers/gateway.py.
                await send({"type": "websocket.close", "code": 1008,
                            "reason": "authentication required"})
                return
            has_accounts = self.db.count_users() > 0
            response = _unauthenticated(
                "authentication required"
                if has_accounts
                else "this appliance has no accounts yet — run `argus user add` on the box",
                code="session_required" if has_accounts else "no_accounts",
            )
            await response(scope, receive, send)
            return

        # Evidence mounts are served by Starlette, past every dependency, so
        # their permission is checked here or nowhere. See MOUNT_PERMISSIONS.
        required = mount_permission(path)
        if principal is not None and required is not None and not principal.can(required):
            response = JSONResponse(
                {"detail": f"role {principal.role} may not {required} evidence"},
                status_code=403,
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


# --------------------------------------------------------------------------
# dependencies
# --------------------------------------------------------------------------


def current_principal(request: Request) -> Principal | None:
    """Whoever the middleware resolved, or None on a public route."""
    return getattr(request.state, "principal", None)


def require_session(request: Request) -> Principal:
    principal = current_principal(request)
    if principal is None:
        # Only reachable if a route is added under a public path; the middleware
        # has already refused everything else.
        raise HTTPException(status_code=401, detail="authentication required")
    return principal


def Requires(permission: str) -> Callable[..., Principal]:  # noqa: N802 - reads as a marker
    """Dependency asserting one permission, e.g. ``Depends(Requires(MANAGE))``.

    The 403 names the permission and the role that was presented. An operator
    who is told "forbidden" files a bug; one who is told "role REVIEW cannot
    manage policies" asks an admin for RESPOND, which is the actual fix.
    """

    def dependency(request: Request) -> Principal:
        principal = require_session(request)
        if not principal.can(permission):
            raise HTTPException(
                status_code=403,
                detail=f"role {principal.role} may not {permission} "
                       f"(held: {', '.join(principal.permissions)})",
            )
        return principal

    return dependency


#: For a route that needs the caller — to attribute a verdict, to name the
#: reviewer, to know who to refuse.
ViewDep = Annotated[Principal, Depends(Requires(core.VIEW))]
TriageDep = Annotated[Principal, Depends(Requires(core.TRIAGE))]
ManageDep = Annotated[Principal, Depends(Requires(core.MANAGE))]
AdminDep = Annotated[Principal, Depends(Requires(core.ADMIN))]
SystemDep = Annotated[Principal, Depends(Requires(core.SYSTEM))]
SessionDep = Annotated[Principal, Depends(require_session)]

#: For a route (or a whole router) that only needs the *check*. Used as
#: ``APIRouter(dependencies=[NeedsView])`` for a router's floor and
#: ``@router.post(..., dependencies=[NeedsManage])`` to raise it on the writes —
#: which puts the read/write split on the decorator line, where a reviewer
#: reading the routes sees it without reading the bodies.
NeedsView = Depends(Requires(core.VIEW))
NeedsTriage = Depends(Requires(core.TRIAGE))
NeedsManage = Depends(Requires(core.MANAGE))
NeedsAdmin = Depends(Requires(core.ADMIN))
NeedsSystem = Depends(Requires(core.SYSTEM))


def websocket_principal(websocket: WebSocket, permission: str = core.VIEW) -> Principal | None:
    """The principal for a WebSocket route, permission-checked.

    Returns None instead of raising: a WebSocket cannot answer 403, so the
    caller closes the handshake with a policy-violation code and a reason the
    console can show.
    """
    principal = getattr(websocket.state, "principal", None)
    if principal is None or not principal.can(permission):
        return None
    return principal


# --------------------------------------------------------------------------
# cookies
# --------------------------------------------------------------------------


def set_session_cookie(response: Any, config: ArgusConfig, token: str) -> None:
    """Attach the session cookie.

    ``HttpOnly`` always: the token must be unreachable from JavaScript, which is
    the difference between "an XSS bug shows an operator the wrong number" and
    "an XSS bug exfiltrates a session that can download the plant's footage".
    ``max_age`` is deliberately *not* set — a session cookie dies with the
    browser, and the server-side ``expires_at`` is the authority on lifetime
    either way. See ``AuthConfig`` for ``secure`` and ``samesite``.
    """
    response.set_cookie(
        key=config.auth.cookie_name,
        value=token,
        httponly=True,
        secure=config.auth.cookie_secure,
        samesite=config.auth.cookie_samesite,
        path="/",
    )


def clear_session_cookie(response: Any, config: ArgusConfig) -> None:
    response.delete_cookie(
        key=config.auth.cookie_name,
        path="/",
        httponly=True,
        secure=config.auth.cookie_secure,
        samesite=config.auth.cookie_samesite,
    )


__all__ = [
    "AdminDep",
    "AuthMiddleware",
    "MOUNT_PERMISSIONS",
    "ManageDep",
    "NeedsAdmin",
    "NeedsManage",
    "NeedsSystem",
    "NeedsTriage",
    "NeedsView",
    "PROTECTED_PREFIXES",
    "PUBLIC_PATHS",
    "Requires",
    "SessionDep",
    "SystemDep",
    "TriageDep",
    "ViewDep",
    "clear_session_cookie",
    "current_principal",
    "mount_permission",
    "needs_session",
    "require_session",
    "resolve_principal",
    "set_session_cookie",
    "websocket_principal",
]


# Re-exported so routers import one module: `from argus.api.auth import MANAGE,
# ManageDep`. Keeps the permission name and the dependency that checks it in the
# same import line, where a mismatch is visible.
ADMIN = core.ADMIN
MANAGE = core.MANAGE
SYSTEM = core.SYSTEM
TRIAGE = core.TRIAGE
VIEW = core.VIEW
__all__ += ["ADMIN", "MANAGE", "SYSTEM", "TRIAGE", "VIEW"]
