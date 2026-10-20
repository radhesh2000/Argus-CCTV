"""Appliance health and configuration.

``/api/health`` is what a plant IT team will point their monitoring at, so it
answers with everything needed to decide "is this box doing its job?" in one
request: is the engine running, what fps is each camera actually achieving, how
loaded is the GPU, and how full is the disk.

It never raises. A probe that cannot answer says so in its own section and the
rest of the report still arrives — a health endpoint that 500s when nvidia-smi
is missing tells you nothing about the twelve cameras that are fine.
"""

from __future__ import annotations

import sys
from typing import Any

from fastapi import APIRouter, Request

from argus import __version__, processing
from argus.accounts import bootstrap_hint
from argus.api.auth import AdminDep, SystemDep, current_principal
from argus.api.deps import ConfigDep, DbDep
from argus.api.schemas import ProcessingUpdate
from argus.auth import MIN_PASSWORD_LENGTH, ROLE_DESCRIPTIONS, ROLES
from argus.health import health_report, read_engine_stats

router = APIRouter(tags=["system"])


def _processing_view(config, engine: dict[str, Any], vlm: dict[str, Any]) -> dict[str, Any]:
    """The light processing block that rides on ``/api/health`` and
    ``/api/live/state`` — built from figures those endpoints already have, so it
    adds no subprocess and no extra stats read."""
    return processing.processing_view(
        config,
        engine_running=bool(engine.get("running")),
        engine_uptime_sec=engine.get("uptime_sec"),
        vlm_available=bool(vlm.get("available")) if engine.get("running") else False,
    )


@router.get("/health")
def health(request: Request, config: ConfigDep, db: DbDep) -> dict[str, Any]:
    """Liveness for anyone; the full report for a session.

    This is the one route on the public allowlist that returns anything, and it
    is split rather than opened. A plant's monitoring needs to know the process
    is up without holding an account — but "which cameras are down, how full is
    the disk, what is the GPU doing" is reconnaissance, and handing it to an
    unauthenticated caller on the LAN would give away the fleet's weak points to
    exactly the person who should not have them.

    ``accounts`` is here for the console's very first request on a freshly
    imaged box: it is how the login screen knows to say "run `argus user add`"
    rather than presenting a form no password can satisfy.
    """
    principal = current_principal(request)
    if principal is None:
        return {
            "status": "ok",
            "version": __version__,
            "authenticated": False,
            "accounts": db.count_users(),
            "detail": bootstrap_hint(db),
        }
    report = health_report(config, db)
    # The desired-state pause switch, next to the actual state it governs. The
    # console reads this to draw a paused banner and to know a stopped engine is
    # a deliberate pause rather than a failure. Built from the report's own
    # figures — no extra probe. See argus/processing.py.
    report["processing"] = _processing_view(config, report["engine"], report["vlm"])
    return {"version": __version__, "authenticated": True, **report}


@router.get("/system/processing")
def get_processing(system: SystemDep, config: ConfigDep) -> dict[str, Any]:
    """The full pause-switch view: the durable flag, the derived status, and
    each toggled unit's actual systemd state.

    Readable by any signed-in role (all four hold ``system``), because a REVIEW
    operator looking at a dark wall needs to be able to see *why*. Actuating it
    is ADMIN-only — that is the ``POST`` below. This one shells out to
    ``systemctl show`` for the live unit states, so it is the on-demand detail
    view, not the two-second poll (that rides on ``/api/health``).
    """
    engine = read_engine_stats(config)
    vlm = engine.get("vlm") or {}
    return processing.processing_report(
        config,
        engine_running=bool(engine.get("running")),
        engine_uptime_sec=engine.get("uptime_sec"),
        vlm_available=bool(vlm.get("available")) if engine.get("running") else False,
    )


@router.post("/system/processing")
def set_processing(body: ProcessingUpdate, admin: AdminDep, config: ConfigDep) -> dict[str, Any]:
    """Pause or resume the whole GPU plane. **ADMIN only, and audited.**

    This stops (or starts) detection, indexing, evidence and the VLM across
    every camera on the appliance — a plant-wide switch, which is why it needs
    the role that also owns accounts, and why the actor is recorded both in the
    journal and in the flag file's own audit line.

    Returns immediately. Resuming does **not** wait for the VLM to load (minutes
    on a cold cache): the flag is flipped, the units are started ``--no-block``,
    and the caller learns readiness by polling ``/api/health`` — engine
    ``running`` fast, VLM ``starting`` until it answers. See
    argus/processing.py:set_processing.
    """
    verb = "resume" if body.enabled else "pause"
    # Audit line to the journal (this process's stdout is journald under
    # `argus install-services`). The flag file carries the durable copy.
    print(
        f"api: processing {verb} by {admin.username} (role {admin.role})"
        + (f" — {body.note}" if body.note else ""),
        file=sys.stderr,
        flush=True,
    )
    result = processing.set_processing(
        config, body.enabled, actor=admin.username, note=body.note,
    )
    # The fresh full view, so the console updates its banner and service panel
    # from the same answer that actuated the change.
    engine = read_engine_stats(config)
    vlm = engine.get("vlm") or {}
    report = processing.processing_report(
        config,
        engine_running=bool(engine.get("running")),
        engine_uptime_sec=engine.get("uptime_sec"),
        vlm_available=bool(vlm.get("available")) if engine.get("running") else False,
    )
    return {**report, "actuated": result["actuated"], "command": result["command"],
            "detail": result["detail"]}


@router.get("/config")
def configuration(system: SystemDep, config: ConfigDep) -> dict[str, Any]:
    """The settings a UI needs to render itself, and nothing else.

    Explicitly enumerated rather than dumped from the dataclass: the config
    tree holds ``smtp.username`` and the name of the variable holding the
    password, and an endpoint that serialises whatever it finds will leak the
    next secret somebody adds.
    """
    return {
        "version": __version__,
        "source": str(config.source_path) if config.source_path else None,
        "site": {
            "name": config.site.name,
            "shifts": config.site.shift_windows,
        },
        # What this appliance can actually do today. The UI renders every screen
        # the product has, and uses these to say "not enabled on this appliance"
        # where a phase has not landed — which is the honest alternative to
        # hiding the screen (the buyer asked about it) or faking the data.
        "features": {
            # Phase 3 landed. Like `vlm`, this answers "does this appliance do
            # semantic search at all", not "is there anything indexed right
            # now" — /api/search/status answers that, with the frame count and
            # the span, and the screen shows both.
            "search": config.index.enabled,
            # Phase 2 landed: `argus run` evaluates kind='vlm' policies when
            # this is on. It answers "does this appliance run prompt policies
            # at all", not "is the model up right now" — that is a health
            # question, and /api/health answers it with queue depth and a
            # reason. A console that conflated the two would call a box with a
            # restarting model "not equipped".
            "vlm": config.vlm.enabled,
            "vlm_verification": config.vlm.verify.enabled,
            # Phase 5 landed. Unconditionally true: there is no switch that
            # turns authentication off (see argus/config.py:AuthConfig), so a
            # console reading `false` here would be reading a build that
            # predates the lock, which is worth being able to tell.
            "auth": True,
            "retraining_exports": False,  # Phase 6: on-box retraining loop
            "channels": {"email": True, "whatsapp": False, "slack": False},
        },
        "auth": {
            "roles": list(ROLES),
            "role_descriptions": ROLE_DESCRIPTIONS,
            "session_ttl_hours": config.auth.session_ttl_hours,
            "session_idle_minutes": config.auth.session_idle_minutes,
            "min_password_length": MIN_PASSWORD_LENGTH,
        },
        "detection": {
            "fps": config.detection.fps,
            "width": config.detection.width,
            "height": config.detection.height,
            "model": config.detection.model.variant,
            "precision": config.detection.model.precision,
        },
        "evidence": {
            "clip_pad_sec": config.evidence.clip_pad_sec,
            "enabled": config.evidence.enabled,
        },
        # The promotion bar. Reported so Policy Studio draws the gate from the
        # appliance's own rule instead of a second copy in JavaScript — which is
        # what it was, and which is how the bar came to be enforced in the
        # console and nowhere else. See argus/config.py:RolloutConfig.
        "rollout": {
            "min_precision": config.rollout.min_precision,
            "min_reviews": config.rollout.min_reviews,
            "allow_override": config.rollout.allow_override,
        },
        "index": {
            "model": config.index.model,
            "sample_interval_sec": config.index.sample_interval_sec,
            "min_score": config.index.min_score,
            "default_limit": config.index.default_limit,
        },
        "retention": {
            "enabled": config.retention.enabled,
            "clips_days": config.retention.clips_days,
            "snapshots_days": config.retention.snapshots_days,
            "events_days": config.retention.events_days,
            "occupancy_days": config.retention.occupancy_days,
            "index_days": config.retention.index_days,
            "max_disk_percent": config.retention.max_disk_percent,
            # Per-stage evidence windows and the hard budget. Sent as the
            # resolved numbers, not the raw block: a null in the config means
            # "inherit the flat window", and resolving that on the appliance
            # keeps the console from having to reimplement the fallback chain
            # in TypeScript and get it subtly different.
            "evidence_days": {
                tier: config.retention.evidence.days(tier, config.retention.clips_days)
                for tier in ("shadow", "reviewed", "auto_routed",
                             "confirmed", "false_positive")
            },
            "evidence_tiered": config.retention.evidence.configured,
            "max_clips_gb": config.retention.max_clips_gb or None,
            "max_snapshots_gb": config.retention.max_snapshots_gb or None,
        },
        "gateway": {
            "url": config.gateway_url,
            # Reported so the Settings screen can state the posture, not so the
            # browser can dial them. Since Phase 5 the console reaches go2rtc
            # only through `/api/gateway/*` on this origin — `api_host` is what
            # says whether that is actually true on this box, because an
            # operator who widened it back to 0.0.0.0 should see that in the UI
            # rather than in a config file.
            "api_host": config.gateway.api_host,
            "api_port": config.gateway.api_port,
            # The exception, and the honest one: ICE connects the browser to
            # this port directly and no relay changes that. docs/SECURITY.md.
            "webrtc_port": config.gateway.webrtc_port,
            "relay": "/api/gateway",
        },
        "stages": ["shadow", "reviewed", "auto_routed"],
        "severities": ["low", "medium", "high"],
        "notify_roles": sorted(config.smtp.recipients),
        # The settings screen shows who a confirmed violation actually reaches.
        # Addresses, not credentials: smtp.username and the password variable
        # stay out of this response (see the docstring).
        "notify_recipients": {
            role: list(addresses) for role, addresses in sorted(config.smtp.recipients.items())
        },
        "smtp": {"host": config.smtp.host, "port": config.smtp.port,
                 "from_address": config.smtp.from_address},
    }
