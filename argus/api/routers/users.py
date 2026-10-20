"""Accounts and roles — the Settings screen's users panel, made real.

ADMIN only, every route, including the list. A roster is a list of who works in
the control room and what each of them can reach; it is not something a REVIEW
account needs and not something an appliance should hand to a SYSTEM monitor.

There is no ``DELETE``. Disabling is the product operation, because
``reviews.reviewer`` names accounts for as long as the retention window holds
the events, and deleting the row that gives that name a person turns a year of
audit trail into initials nobody can resolve. ``Database.delete_user`` exists
for tests and for a support engineer with a prompt.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from argus import accounts
from argus.api.auth import AdminDep
from argus.api.deps import DbDep
from argus.api.schemas import UserCreate, UserUpdate
from argus.auth import ROLE_DESCRIPTIONS, ROLES

router = APIRouter(prefix="/users", tags=["users"])


def _fail(exc: accounts.AccountError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=str(exc))


@router.get("")
def list_users(admin: AdminDep, db: DbDep) -> dict[str, Any]:
    """The roster, plus the vocabulary the console needs to render it.

    ``roles`` and ``role_descriptions`` ride along so the role selector is
    populated by the appliance rather than by a hard-coded list in the SPA that
    can disagree with :mod:`argus.auth` after an upgrade.
    """
    return {
        "users": accounts.roster(db),
        "roles": list(ROLES),
        "role_descriptions": ROLE_DESCRIPTIONS,
        # So the console can grey out the control that would lock the plant out
        # of its own appliance, instead of offering it and then showing a 409.
        "admin_count": db.count_users(enabled_only=True, role="ADMIN"),
        "self": admin.username,
    }


@router.post("", status_code=201)
def create_user(body: UserCreate, admin: AdminDep, db: DbDep) -> dict[str, Any]:
    try:
        user = accounts.create_user(
            db, body.username, body.password,
            role=body.role, display_name=body.display_name,
        )
    except accounts.AccountError as exc:
        raise _fail(exc) from exc
    return {"user": user}


@router.get("/{user_id}")
def get_user(user_id: int, admin: AdminDep, db: DbDep) -> dict[str, Any]:
    user = db.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"user not found: {user_id}")
    # Where this account is currently signed in, so an admin investigating an
    # incident can see it — and so "I logged out everywhere" is checkable.
    return {"user": user, "sessions": db.list_sessions_for_user(user_id)}


@router.patch("/{user_id}")
def update_user(user_id: int, body: UserUpdate, admin: AdminDep,
                db: DbDep) -> dict[str, Any]:
    """Role, display name, enabled/disabled, password — one at a time or together.

    Applied in a fixed order with the destructive operations last, so a request
    that both renames and disables an account has the rename recorded even if
    the disable is refused for being the last administrator.
    """
    user = db.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"user not found: {user_id}")
    try:
        if body.display_name is not None:
            user = accounts.rename(db, user_id, body.display_name)
        if body.password is not None:
            user = accounts.set_password(db, user_id, body.password)
        if body.role is not None:
            user = accounts.set_role(db, user_id, body.role)
        if body.disabled is not None:
            user = accounts.set_disabled(db, user_id, body.disabled)
    except accounts.AccountError as exc:
        raise _fail(exc) from exc
    return {"user": user}


@router.post("/{user_id}/sessions/revoke")
def revoke_sessions(user_id: int, admin: AdminDep, db: DbDep) -> dict[str, Any]:
    """Sign an account out of every browser without changing its password.

    The operation an admin actually wants when a laptop goes missing: the person
    keeps their account and their password, and every cookie in the wild stops
    working.
    """
    if db.get_user(user_id) is None:
        raise HTTPException(status_code=404, detail=f"user not found: {user_id}")
    return {"revoked": db.delete_sessions_for_user(user_id)}
