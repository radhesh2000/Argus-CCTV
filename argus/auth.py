"""Local accounts, roles and session tokens.

No FastAPI here. The API layer (``argus/api/auth.py``) turns what this module
decides into 401s and 403s; this file owns the two things that must be right
whether or not anybody is holding an HTTP request: **what a role is allowed to
do**, and **what a password and a session token are worth on disk**.

Three decisions worth stating, because each has a cheaper wrong version:

* **Argon2id, via ``argon2-cffi`` (MIT).** bcrypt (Apache-2.0) was the other
  permissive candidate and would also have passed the licence audit; Argon2id
  wins because it is memory-hard — an appliance's password file is exactly the
  thing that walks out on a stolen disk, and a GPU-shaped attacker is exactly
  who this box attracts. bcrypt also silently truncates at 72 bytes, which is a
  footgun in a product that will one day accept passphrases.
* **The session cookie's value is never stored.** The ``sessions`` table is
  keyed by ``sha256(token)``. A read of the appliance database — a support
  bundle, a backup, an operator with ``sqlite3`` — therefore yields no usable
  cookie. Verifying a session is a single indexed lookup on the digest, so this
  costs nothing.
* **Roles are a fixed set, not a permission editor.** Four roles, named exactly
  as the console's Settings screen has always named them, each mapping to a
  frozen set of permissions. A plant that needs a fifth role needs a product
  decision, not a checkbox grid whose combinations nobody has tested.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Final, Mapping

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

# --------------------------------------------------------------------------
# roles and permissions
# --------------------------------------------------------------------------

#: Every permission the API checks. Deliberately coarse — one per *kind of
#: authority*, not one per route — because a matrix a reviewer cannot hold in
#: their head is a matrix nobody audits.
VIEW: Final = "view"        # events, evidence, cameras, zones, policies, search, live
TRIAGE: Final = "triage"    # record a verdict; save a search
MANAGE: Final = "manage"    # create/edit/delete policies, cameras, zones
ADMIN: Final = "admin"      # accounts and roles
SYSTEM: Final = "system"    # health, metrics, the appliance's own description

PERMISSIONS: Final[tuple[str, ...]] = (VIEW, TRIAGE, MANAGE, ADMIN, SYSTEM)

#: The four roles the Settings screen has shown since the design, and what each
#: one can actually do. Note what SYSTEM is *not* given: an IT/monitoring
#: account has no business watching the plant, so it gets health and metrics and
#: nothing that returns footage or a person's whereabouts.
ROLE_PERMISSIONS: Final[dict[str, frozenset[str]]] = {
    "ADMIN": frozenset({VIEW, TRIAGE, MANAGE, ADMIN, SYSTEM}),
    "RESPOND": frozenset({VIEW, TRIAGE, MANAGE, SYSTEM}),
    "REVIEW": frozenset({VIEW, TRIAGE, SYSTEM}),
    "SYSTEM": frozenset({SYSTEM}),
}

ROLES: Final[tuple[str, ...]] = ("ADMIN", "RESPOND", "REVIEW", "SYSTEM")

#: One line each, for the console and for `argus user list --help`.
ROLE_DESCRIPTIONS: Final[dict[str, str]] = {
    "ADMIN": "Everything, including accounts and roles.",
    "RESPOND": "Triage alerts, and manage policies, cameras and zones.",
    "REVIEW": "Triage alerts: confirm or reject, and search evidence.",
    "SYSTEM": "Health and metrics only — no footage, no events.",
}


def normalize_role(role: str) -> str:
    """Accept ``admin``/``Admin``/``ADMIN``; reject anything else loudly."""
    candidate = str(role).strip().upper()
    if candidate not in ROLE_PERMISSIONS:
        raise ValueError(f"unknown role {role!r} — expected one of {', '.join(ROLES)}")
    return candidate


def permissions_for(role: str) -> frozenset[str]:
    return ROLE_PERMISSIONS.get(str(role).upper(), frozenset())


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is making this request, resolved from a session cookie.

    Frozen because it is stashed on the ASGI scope and read by everything
    downstream: a route that could edit it could grant itself a permission.
    """

    user_id: int
    username: str
    display_name: str
    role: str
    session_id: str
    expires_at: str

    def can(self, permission: str) -> bool:
        return permission in permissions_for(self.role)

    @property
    def permissions(self) -> list[str]:
        return sorted(permissions_for(self.role))

    def public(self) -> dict[str, Any]:
        """What the console is told about itself. No session id: the SPA has no
        use for it and it is a bearer-adjacent value."""
        return {
            "id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "permissions": self.permissions,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Principal":
        return cls(
            user_id=int(row["user_id"]),
            username=str(row["username"]),
            display_name=str(row["display_name"]),
            role=str(row["role"]),
            session_id=str(row["session_id"]),
            expires_at=str(row["expires_at"]),
        )


# --------------------------------------------------------------------------
# passwords
# --------------------------------------------------------------------------

#: Defaults chosen by argon2-cffi's own RFC 9106 profile (64 MiB, t=3, p=4).
#: Measured on this GB10: ~55 ms per hash on one Grace core, which is the right
#: order for a login on an appliance — slow enough to matter offline, invisible
#: to an operator, and irrelevant to the 5 fps vision loop it shares a box with.
_HASHER = PasswordHasher()

#: Refusing an obviously weak password is the cheapest control available on a
#: box whose whole security story is "it is on your LAN". Length only: composition
#: rules push people to `Password1!` and NIST SP 800-63B stopped recommending them.
MIN_PASSWORD_LENGTH: Final = 10
#: Argon2 has no bcrypt-style truncation, but an unbounded field is a free
#: denial of service — 64 MiB of hashing per megabyte posted.
MAX_PASSWORD_LENGTH: Final = 1024


class PasswordError(ValueError):
    """A password that will not be accepted, with the reason a human needs."""


def validate_password(password: str) -> str:
    if not isinstance(password, str) or not password.strip():
        raise PasswordError("password must not be empty")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordError(
            f"password must be at least {MIN_PASSWORD_LENGTH} characters"
        )
    if len(password) > MAX_PASSWORD_LENGTH:
        raise PasswordError(f"password must be at most {MAX_PASSWORD_LENGTH} characters")
    return password


def hash_password(password: str) -> str:
    """Argon2id hash, self-describing (``$argon2id$v=19$m=...``) so a future
    parameter change can be detected per row rather than forcing a reset."""
    return _HASHER.hash(validate_password(password))


def verify_password(stored_hash: str, password: str) -> bool:
    """Constant-time-ish verify that never raises on bad input.

    A malformed or empty hash column returns False rather than exploding: the
    caller is an unauthenticated login endpoint, and "this row is corrupt" must
    look exactly like "wrong password" from outside.
    """
    if not stored_hash or not isinstance(password, str):
        return False
    try:
        return _HASHER.verify(stored_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(stored_hash: str) -> bool:
    try:
        return _HASHER.check_needs_rehash(stored_hash)
    except (InvalidHashError, VerificationError):
        return True


#: Hash of a password nobody has. Used to spend the same ~55 ms on a login for a
#: username that does not exist as on one that does, so the response time cannot
#: be used to enumerate accounts.
_DUMMY_HASH = _HASHER.hash("argus-timing-equaliser-not-a-real-password")


def waste_password_time() -> None:
    """Burn one verification's worth of CPU. See ``_DUMMY_HASH``."""
    verify_password(_DUMMY_HASH, "wrong")


# --------------------------------------------------------------------------
# session tokens
# --------------------------------------------------------------------------

#: 32 bytes of urandom, urlsafe-base64'd to 43 characters.
TOKEN_BYTES: Final = 32


def new_session_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def token_digest(token: str) -> str:
    """What actually goes in the ``sessions`` table. See the module docstring.

    Plain SHA-256 rather than a password hash on purpose: the input is 256 bits
    of urandom, so there is nothing to brute-force and no reason to make session
    validation cost 55 ms on every request.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.isoformat()


def expiry(ttl_hours: float, *, now: datetime | None = None) -> str:
    return iso((now or utcnow()) + timedelta(hours=ttl_hours))


def is_expired(expires_at: str, *, now: datetime | None = None) -> bool:
    """True when a stored ISO deadline has passed. An unparseable deadline is
    treated as expired — a session row we cannot read is not a session."""
    try:
        deadline = datetime.fromisoformat(expires_at)
    except (TypeError, ValueError):
        return True
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    return deadline <= (now or utcnow())


def idle_deadline_passed(last_seen_at: str, idle_minutes: float,
                         *, now: datetime | None = None) -> bool:
    """True when a session has sat untouched past the idle window.

    ``idle_minutes <= 0`` disables the check: a control-room wall display that
    nobody touches for a whole shift is a real deployment, and logging it out
    mid-shift is a worse failure than the one this defends against.
    """
    if idle_minutes <= 0:
        return False
    try:
        seen = datetime.fromisoformat(last_seen_at)
    except (TypeError, ValueError):
        return True
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=timezone.utc)
    return (now or utcnow()) - seen > timedelta(minutes=idle_minutes)


# --------------------------------------------------------------------------
# usernames
# --------------------------------------------------------------------------

MAX_USERNAME_LENGTH: Final = 64
_USERNAME_ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789._-")


def normalize_username(username: str) -> str:
    """Lower-cased, and restricted to a set that cannot surprise anything.

    Usernames end up in log lines, filenames of exports, and the ``reviewer``
    column of the audit trail. Restricting them here means none of those places
    has to think about it.
    """
    candidate = str(username).strip().lower()
    if not candidate:
        raise ValueError("username must not be empty")
    if len(candidate) > MAX_USERNAME_LENGTH:
        raise ValueError(f"username must be at most {MAX_USERNAME_LENGTH} characters")
    illegal = sorted(set(candidate) - _USERNAME_ALLOWED)
    if illegal:
        raise ValueError(
            "username may contain only letters, digits, dot, underscore and hyphen "
            f"(got {''.join(illegal)!r})"
        )
    return candidate


__all__ = [
    "ADMIN",
    "MANAGE",
    "MAX_PASSWORD_LENGTH",
    "MIN_PASSWORD_LENGTH",
    "PERMISSIONS",
    "ROLES",
    "ROLE_DESCRIPTIONS",
    "ROLE_PERMISSIONS",
    "SYSTEM",
    "TRIAGE",
    "VIEW",
    "PasswordError",
    "Principal",
    "expiry",
    "hash_password",
    "idle_deadline_passed",
    "is_expired",
    "iso",
    "needs_rehash",
    "new_session_token",
    "normalize_role",
    "normalize_username",
    "permissions_for",
    "token_digest",
    "tokens_equal",
    "utcnow",
    "validate_password",
    "verify_password",
    "waste_password_time",
]
