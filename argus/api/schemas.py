"""Request bodies for the Argus API.

Only *requests* are modelled. Responses are the database rows as they are — the
schema in ``argus/db.py`` is the single source of truth for what an event is,
and a second copy of it here would be one more thing to keep in sync and one
more place for the two to disagree. What pydantic is doing here is rejecting
bad input with a message that says which field and why, which is the part a
hand-written check does badly.

``model_config = ConfigDict(extra="forbid")`` throughout: a PATCH with a
misspelled field must fail loudly rather than silently change nothing.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from argus.auth import MAX_PASSWORD_LENGTH, MAX_USERNAME_LENGTH, MIN_PASSWORD_LENGTH
from argus.redact import MASK, is_masked, swap_password

Severity = Literal["low", "medium", "high"]
Stage = Literal["shadow", "reviewed", "auto_routed"]
Lifecycle = Literal["draft", "pilot", "production", "archived"]
Verdict = Literal["confirmed", "false_positive"]
PolicyKind = Literal["detector", "vlm"]
# Spelled out rather than built from argus.auth.ROLES, so the OpenAPI document
# names the four roles and a client generator produces an enum. test_auth.py
# asserts the two stay in step.
Role = Literal["ADMIN", "RESPOND", "REVIEW", "SYSTEM"]

# Zone polygons are normalized to the frame (see argus/vision/zones.py) so a
# zone drawn once survives a resolution change. A polygon in pixel coordinates
# would be accepted silently and then never contain anybody, so the range is
# enforced at the door.
Polygon = Annotated[list[tuple[float, float]], Field(min_length=3)]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_normalized(polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
    for x, y in polygon:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"polygon points must be normalized to 0-1 (got {x}, {y}); "
                "divide pixel coordinates by the frame width and height"
            )
    return polygon


# --------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------


def _check_source(source: str | None) -> str | None:
    """A camera source is an argument to a program, so it is checked here.

    ``argus.ingest.gateway`` interpolates a file source into an ``exec:``
    ffmpeg command line and hands a URL to go2rtc, which understands ``exec:``
    and ``ffmpeg:`` sources of its own. Before the console could enrol cameras
    this column was only ever written by ``config/seed.yaml``; now a RESPOND
    account can type into it, and the validation has to be at the door rather
    than in the operator's good judgement. The rule itself lives with the
    component that would run the thing — see ``assert_safe_source``.
    """
    from argus.ingest.gateway import UnsafeSource, assert_safe_source

    if source is None:
        return None
    if is_masked(source):
        raise ValueError(
            f"type the camera's password: {MASK} is what the console shows in place "
            "of a stored one, and a camera that is not enrolled yet has none"
        )
    try:
        return assert_safe_source(source, must_exist=True)
    except UnsafeSource as exc:
        raise ValueError(str(exc)) from exc


def _check_editable_source(source: str | None) -> str | None:
    """The same check, for a source that may be an edit of a stored one.

    The API never returns a camera's password (``argus/redact.py``), so the
    edit form sends back ``rtsp://user:***@host/...`` and the route puts the
    stored password where the placeholder is. Everything *but* the placeholder
    is still checked here; the masked string is returned untouched for the
    route to resolve, because only the route knows which camera it is for.
    """
    if source is None or not is_masked(source):
        return _check_source(source)
    _check_source(swap_password(source, "x"))
    return source


class CameraCreate(_Body):
    name: str = Field(min_length=1, max_length=120)
    source: str = Field(min_length=1, max_length=500)
    # Omit to have one derived from the name; supply it to make the fleet
    # reproducible from a seed file.
    id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=64)
    enabled: bool = True
    detect_fps: float | None = Field(default=None, gt=0, le=60)

    _safe = field_validator("source")(_check_source)


class CameraUpdate(_Body):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    source: str | None = Field(default=None, min_length=1, max_length=500)
    enabled: bool | None = None
    detect_fps: float | None = Field(default=None, gt=0, le=60)

    _safe = field_validator("source")(_check_editable_source)


class CameraProbe(_Body):
    """"Does this URL actually answer?" — asked before a camera is saved.

    Either an unsaved ``source`` (the enrolment form's test button) or the
    ``camera_id`` of one that already exists (the row's re-test). Never both:
    a probe that silently preferred one would let an operator believe they had
    tested the URL in front of them.

    ``password_from`` is for testing an *edit*: the form holds the masked source
    of an enrolled camera, and this names the camera whose stored password
    fills the ``***``. It does not choose what is probed — ``source`` does.
    """

    source: str | None = Field(default=None, min_length=1, max_length=500)
    camera_id: str | None = Field(default=None, max_length=64)
    password_from: str | None = Field(default=None, max_length=64)

    _safe = field_validator("source")(_check_editable_source)


# --------------------------------------------------------------------------
# zones
# --------------------------------------------------------------------------


class ZoneCreate(_Body):
    name: str = Field(min_length=1, max_length=120)
    polygon: Polygon
    id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=64)
    # None means "count people here but never alert" — a real and common setup.
    sanctioned_limit: int | None = Field(default=None, ge=0, le=1000)
    color: str = Field(default="#22d3ee", pattern=r"^#[0-9a-fA-F]{6}$")

    _normalized = field_validator("polygon")(_check_normalized)


class ZoneUpdate(_Body):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    polygon: Polygon | None = None
    sanctioned_limit: int | None = Field(default=None, ge=0, le=1000)
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")

    _normalized = field_validator("polygon")(_check_normalized)


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------


class PolicyScopeEntry(_Body):
    """One place a policy applies (migration 5, ``policy_scopes``).

    ``zone_id`` null means the whole camera. For a detector policy a zone
    restricts which zone may raise the event; for a VLM policy it is what the
    model is shown — the frame cropped to that polygon.
    """

    camera_id: str = Field(min_length=1, max_length=64)
    zone_id: str | None = Field(default=None, max_length=64)


#: A policy's whole scope. `[]` means every camera, which is what an appliance
#: that has never scoped anything has — so the field is always safe to omit.
Scope = Annotated[list[PolicyScopeEntry], Field(max_length=200)]


class PolicyCreate(_Body):
    kind: PolicyKind
    detector_key: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=200)
    id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=64)
    enabled: bool = True
    scope: Scope = Field(default_factory=list)
    # New policies start in shadow and in draft. A policy that could be created
    # straight into auto_routed would make the staged rollout optional, and the
    # staged rollout is the product; promotion goes through /promote.
    severity: Severity = "medium"
    routed_to: str | None = Field(default=None, max_length=64)
    prompt: str | None = None
    interval_sec: float | None = Field(default=None, gt=0)
    cooldown_sec: float = Field(default=30.0, ge=0)
    threshold: float | None = None
    pilot_duration_days: float = Field(default=2.0, gt=0)


class PolicyUpdate(_Body):
    label: str | None = Field(default=None, min_length=1, max_length=200)
    enabled: bool | None = None
    severity: Severity | None = None
    routed_to: str | None = Field(default=None, max_length=64)
    prompt: str | None = None
    interval_sec: float | None = Field(default=None, gt=0)
    cooldown_sec: float | None = Field(default=None, ge=0)
    threshold: float | None = None
    pilot_duration_days: float | None = Field(default=None, gt=0)
    # stage is editable (an operator may pull a noisy policy back to shadow
    # without demoting its lifecycle); lifecycle is not — it moves only through
    # the pilot/promote endpoints, which timestamp the transition.
    stage: Stage | None = None
    # Omitted leaves the scope alone; `[]` widens the policy back to every
    # camera. Both are meaningful, which is why this is not `| None`.
    scope: Scope | None = None


class PolicyPromote(_Body):
    stage: Stage = "auto_routed"
    routed_to: str | None = Field(default=None, max_length=64)
    # Promote to auto_routed on evidence that has not cleared `rollout:`. Two
    # fields rather than one, so an override is a deliberate sentence somebody
    # typed: the endpoint refuses `override: true` with no reason, and stores
    # the reason on the policy.
    override: bool = False
    override_reason: str | None = Field(default=None, max_length=500)


class PilotExtend(_Body):
    extra_days: float = Field(default=2.0, gt=0, le=365)


class PolicyTest(_Body):
    """One prompt, one camera, one frame, right now — Policy Studio's test panel.

    ``prompt`` is what is being tested and it is sent verbatim: the point of the
    panel is that the author sees the answer their *draft* wording produces, so
    taking the stored prompt when one is typed would test the wrong thing.
    ``policy_id`` only fills in a default and records what the test was about.
    """

    camera_id: str = Field(min_length=1, max_length=64)
    prompt: str | None = Field(default=None, min_length=1, max_length=4000)
    policy_id: str | None = Field(default=None, max_length=64)
    # Shows the model the zone crop, exactly as a zone-scoped policy would.
    zone_id: str | None = Field(default=None, max_length=64)
    # The engine publishes a frame on its 2 s stats tick; this is how long the
    # request waits for one before reporting that nothing is publishing.
    wait_sec: float = Field(default=6.0, gt=0, le=30)


# --------------------------------------------------------------------------
# reviews
# --------------------------------------------------------------------------


class ReviewCreate(_Body):
    verdict: Verdict
    reviewer: str | None = Field(default=None, max_length=120)
    note: str | None = Field(default=None, max_length=2000)


BoxVerdict = Literal["true_positive", "false_positive"]


class BoxCorrectionCreate(_Body):
    """One reviewer's judgement on a single detection within an event's
    `meta.boxes`, keyed by the track id the console already shows next to it
    ("Worker 3"). Distinct from `ReviewCreate`, which judges the whole alert —
    a reviewer can confirm the alert was a real over-limit zone and still mark
    one specific box in the crowd as not a person at all."""

    track_id: int
    verdict: BoxVerdict
    note: str | None = Field(default=None, max_length=2000)


# --------------------------------------------------------------------------
# saved searches (Phase 3)
# --------------------------------------------------------------------------


class SavedSearchCreate(_Body):
    query: str = Field(min_length=1, max_length=400)
    # Defaults to the query itself: the chip in the console shows the sentence,
    # and forcing an operator to invent a second name for it before they can
    # keep it is the kind of friction that means nobody saves anything.
    label: str | None = Field(default=None, min_length=1, max_length=200)
    camera_id: str | None = Field(default=None, max_length=64)
    # A *relative* window, in hours — see migration 3 for why not a timestamp.
    window_hours: float | None = Field(default=None, gt=0, le=24 * 365)


class SavedSearchUpdate(_Body):
    label: str | None = Field(default=None, min_length=1, max_length=200)
    query: str | None = Field(default=None, min_length=1, max_length=400)
    camera_id: str | None = Field(default=None, max_length=64)
    window_hours: float | None = Field(default=None, gt=0, le=24 * 365)


class SavedSearchPromote(_Body):
    """Search -> draft VLM policy. No `stage` or `lifecycle` field, on purpose:
    a promoted search is always born in shadow/draft and walks the same gate as
    every other policy."""

    label: str | None = Field(default=None, min_length=1, max_length=200)
    policy_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=64)
    severity: Severity = "medium"
    interval_sec: float | None = Field(default=None, gt=0)
    cooldown_sec: float | None = Field(default=None, ge=0)
    pilot_duration_days: float = Field(default=2.0, gt=0)


# --------------------------------------------------------------------------
# accounts (Phase 5)
# --------------------------------------------------------------------------
#
# Length bounds on the password fields are a denial-of-service control, not a
# password policy: Argon2id at 64 MiB per hash means an unbounded field lets one
# POST ask the appliance to allocate as much memory as the request body is long.
# The *minimum* is in argus/auth.py, with the reason it is length-only.


class LoginRequest(_Body):
    username: str = Field(min_length=1, max_length=MAX_USERNAME_LENGTH)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class PasswordChange(_Body):
    current_password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)
    new_password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)


class UserCreate(_Body):
    username: str = Field(min_length=1, max_length=MAX_USERNAME_LENGTH)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    role: Role
    # Defaults to the username. What the masthead and the audit trail show, so
    # it is free text — "M. Alvarez" is a name, `m.alvarez` is a login.
    display_name: str | None = Field(default=None, min_length=1, max_length=120)


class UserUpdate(_Body):
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    role: Role | None = None
    disabled: bool | None = None
    # An admin resetting a password does not need to know the old one — that is
    # what makes it a reset. It still ends every session the account holds.
    password: str | None = Field(default=None, min_length=MIN_PASSWORD_LENGTH,
                                 max_length=MAX_PASSWORD_LENGTH)


class ProcessingUpdate(_Body):
    """Flip the appliance's pause switch. ``enabled: false`` stops the whole
    GPU plane (engine + VLM) across every camera; ``true`` resumes it. Required
    and boolean — there is no third state, and an absent field must not be read
    as either."""

    enabled: bool
    # A short, optional line stored beside the flag: "quieting the box overnight",
    # "maintenance window". Ends up in the audit trail next to who set it.
    note: str | None = Field(default=None, max_length=200)


__all__ = [
    "CameraCreate",
    "CameraProbe",
    "CameraUpdate",
    "LoginRequest",
    "PasswordChange",
    "PilotExtend",
    "PolicyCreate",
    "PolicyPromote",
    "PolicyScopeEntry",
    "PolicyTest",
    "PolicyUpdate",
    "ProcessingUpdate",
    "ReviewCreate",
    "SavedSearchCreate",
    "SavedSearchPromote",
    "SavedSearchUpdate",
    "UserCreate",
    "UserUpdate",
    "ZoneCreate",
    "ZoneUpdate",
]
