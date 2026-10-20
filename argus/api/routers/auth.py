"""Sign in, sign out, and "who am I".

Three routes, two of which are on the public allowlist in
``argus/api/auth.py``. The third — logout — needs the session it is destroying.

``GET /api/auth/session`` answering **401** for a logged-out browser is a
deliberate contract, not an oversight: it is the one request the console makes
before it knows anything, and a 200 with ``{"user": null}`` would mean the SPA
had to distinguish "no session" from "the API is down" by inspecting a body.
The 401 carries a ``code`` so the login screen can say the right thing —
``no_accounts`` on a freshly installed appliance names the command to run.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from argus import accounts
from argus.api.auth import (
    SessionDep,
    clear_session_cookie,
    current_principal,
    set_session_cookie,
)
from argus.api.deps import ConfigDep, DbDep
from argus.api.schemas import LoginRequest, PasswordChange
from argus.auth import ROLE_DESCRIPTIONS, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request) -> str:
    """The address the rate limiter counts against.

    ``request.client`` and nothing else — deliberately not ``X-Forwarded-For``.
    The appliance serves the console itself, so a forwarded header here is
    attacker-supplied text, and honouring it would hand any client a fresh
    identity per attempt and turn the rate limit off. A deployment that really
    does put a reverse proxy in front needs a proxy-header middleware and a
    trusted-hop count, which is a decision with a blast radius, not a default.
    """
    return request.client.host if request.client else "unknown"


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response,
          config: ConfigDep, db: DbDep) -> dict[str, Any]:
    """Exchange a username and password for a session cookie."""
    try:
        result = accounts.login(
            config, db, body.username, body.password,
            ip=_client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )
    except accounts.AccountError as exc:
        headers = {"Retry-After": str(int(config.auth.login_lockout_sec))} if exc.status == 429 else None
        raise HTTPException(status_code=exc.status, detail=str(exc), headers=headers) from exc

    set_session_cookie(response, config, result.token)
    return {"user": result.principal.public()}


@router.get("/session")
def session(request: Request, db: DbDep) -> dict[str, Any]:
    """Who this browser is. 401 when it is nobody — see the module docstring."""
    principal = current_principal(request)
    if principal is None:
        hint = accounts.bootstrap_hint(db)
        raise HTTPException(
            status_code=401,
            detail=hint or "authentication required",
            headers={"X-Argus-Auth": "no_accounts" if hint else "session_required"},
        )
    return {"user": principal.public(), "roles": ROLE_DESCRIPTIONS}


@router.post("/logout")
def logout(principal: SessionDep, response: Response, config: ConfigDep,
           db: DbDep) -> dict[str, Any]:
    """End this session. Server-side first, then the cookie.

    Deleting the row is what makes this a logout: clearing the cookie alone
    would leave a token that still works to anyone who captured it, which is
    precisely the population a logout button exists for.
    """
    db.delete_session(principal.session_id)
    clear_session_cookie(response, config)
    return {"logged_out": True}


@router.post("/password")
def change_own_password(body: PasswordChange, principal: SessionDep, response: Response,
                        config: ConfigDep, db: DbDep) -> dict[str, Any]:
    """Change your own password. Requires the current one.

    Requiring the old password even though the caller is already authenticated
    is what stops an unattended browser from becoming a permanent account
    takeover. Verified directly rather than by calling
    :func:`argus.accounts.login`, which would mint a second session as a side
    effect of a password check.

    On success every session ends, this one included, so the operator signs back
    in with the new password — which is also how they find out it took effect.
    """
    record = db.credentials_for(principal.username)
    if record is None or not verify_password(str(record["password_hash"]), body.current_password):
        raise HTTPException(status_code=403, detail="current password is incorrect")
    try:
        accounts.set_password(db, principal.user_id, body.new_password)
    except accounts.AccountError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    clear_session_cookie(response, config)
    return {"changed": True, "signed_out": True}
