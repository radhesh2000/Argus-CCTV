"""Policies and the staged rollout that governs them.

The lifecycle (draft -> pilot -> production) and the stage (shadow -> reviewed
-> auto_routed) are two different axes and both are deliberate:

* lifecycle answers "is this policy running against live video yet?"
* stage answers "who, if anyone, gets told when it fires?"

Lifecycle transitions are endpoints rather than PATCH fields because each one
timestamps itself (``pilot_started_at``, ``promoted_at``) — the evidence that a
policy served its validation window before anyone's phone started buzzing.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from argus.api.auth import NeedsManage, NeedsView
from argus.api.deps import ConfigDep, DbDep, changed, found
from argus.api.schemas import (
    PilotExtend,
    PolicyCreate,
    PolicyPromote,
    PolicyTest,
    PolicyUpdate,
)
from argus.db import Policy, slugify
from argus.live import LiveFrames
from argus.vlm.evidence import zone_crop

# Everyone with VIEW reads the policy list — a reviewer triaging an alert has
# to be able to see what raised it and at what stage. Every write, including
# the lifecycle transitions, needs MANAGE: promoting a policy to auto_routed
# is what starts sending mail to an HOD.
router = APIRouter(prefix="/policies", tags=["policies"], dependencies=[NeedsView])


def with_pilot_info(policy: dict[str, Any]) -> dict[str, Any]:
    """Attach the derived pilot window.

    Derived on read rather than stored: ``extend_pilot`` changes the duration,
    and a persisted end date would silently go stale the moment it did.
    """
    total = policy.get("pilot_duration_days") or 2.0
    info: dict[str, Any] = {
        "days_elapsed": None, "days_total": total, "days_left": None, "overdue": False,
    }
    started = policy.get("pilot_started_at")
    if started:
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds() / 86400
        info["days_elapsed"] = round(elapsed, 2)
        info["days_left"] = round(max(0.0, total - elapsed), 2)
        info["overdue"] = elapsed >= total
    return {**policy, "pilot": info}


@router.get("")
def list_policies(
    db: DbDep,
    kind: str | None = None,
    enabled_only: bool = False,
    active_only: bool = False,
) -> dict[str, Any]:
    policies = db.list_policies(kind=kind, enabled_only=enabled_only, active_only=active_only)
    return {"policies": [with_pilot_info(policy) for policy in policies]}


def _scope(entries: Any) -> list[dict[str, Any]]:
    return [{"camera_id": e.camera_id, "zone_id": e.zone_id} for e in entries]


def _bad_scope(exc: ValueError) -> HTTPException:
    """A scope naming a camera or a zone that is not there.

    422 rather than 404: the *policy* exists and the request is what is wrong,
    and a 404 here would read as "no such policy" to the console that just
    fetched it.
    """
    return HTTPException(status_code=422, detail=f"scope: {exc}")


@router.post("", status_code=201, dependencies=[NeedsManage])
def create_policy(body: PolicyCreate, db: DbDep) -> dict[str, Any]:
    policy_id = body.id or slugify(body.label)
    if db.get_policy(policy_id):
        raise HTTPException(status_code=409, detail=f"policy already exists: {policy_id}")
    if body.kind == "vlm" and not body.prompt:
        raise HTTPException(status_code=422, detail="a vlm policy needs a prompt")
    try:
        db.upsert_policy(Policy(
            id=policy_id,
            kind=body.kind,
            detector_key=body.detector_key,
            label=body.label,
            enabled=body.enabled,
            # Always born in shadow/draft — see schemas.PolicyCreate.
            stage="shadow",
            lifecycle="draft",
            severity=body.severity,
            routed_to=body.routed_to,
            prompt=body.prompt,
            interval_sec=body.interval_sec,
            cooldown_sec=body.cooldown_sec,
            threshold=body.threshold,
            pilot_duration_days=body.pilot_duration_days,
            scope=_scope(body.scope),
        ))
    except ValueError as exc:
        # The policy row is written before the scope, so a bad scope would leave
        # an unscoped policy behind. Removed rather than kept: a create that
        # answered 422 and created something anyway is the worst of both.
        db.delete_policy(policy_id)
        raise _bad_scope(exc) from exc
    return {"policy": with_pilot_info(db.get_policy(policy_id))}


@router.get("/{policy_id}")
def get_policy(policy_id: str, db: DbDep) -> dict[str, Any]:
    return {"policy": with_pilot_info(found(db.get_policy(policy_id), "policy", policy_id))}


@router.patch("/{policy_id}", dependencies=[NeedsManage])
def update_policy(policy_id: str, body: PolicyUpdate, db: DbDep) -> dict[str, Any]:
    policy = found(db.get_policy(policy_id), "policy", policy_id)
    fields = body.model_dump(exclude_unset=True)
    # `scope` is a table, not a column. Applied first so a request that both
    # rewrites the prompt and names a camera that does not exist changes
    # neither — the console sends one PATCH per Save and an operator should not
    # have to work out which half of it landed.
    if "scope" in fields:
        try:
            db.set_policy_scope(policy_id, _scope(body.scope or []))
        except ValueError as exc:
            raise _bad_scope(exc) from exc
        fields.pop("scope")
    if fields.get("prompt") is not None and policy["kind"] != "vlm":
        raise HTTPException(status_code=422, detail="only a vlm policy has a prompt")
    if fields:
        db.update_policy(policy_id, **fields)
    return {"policy": with_pilot_info(db.get_policy(policy_id))}


@router.delete("/{policy_id}", dependencies=[NeedsManage])
def delete_policy(policy_id: str, db: DbDep) -> dict[str, Any]:
    changed(db.delete_policy(policy_id), "policy", policy_id)
    return {"deleted": policy_id}


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


@router.post("/{policy_id}/pilot", dependencies=[NeedsManage])
def start_pilot(policy_id: str, db: DbDep) -> dict[str, Any]:
    policy = found(db.get_policy(policy_id), "policy", policy_id)
    if not db.start_pilot(policy_id):
        raise HTTPException(
            status_code=409,
            detail=f"only a draft policy can start a pilot (this one is {policy['lifecycle']})",
        )
    return {"policy": with_pilot_info(db.get_policy(policy_id))}


@router.post("/{policy_id}/pilot/extend", dependencies=[NeedsManage])
def extend_pilot(policy_id: str, body: PilotExtend, db: DbDep) -> dict[str, Any]:
    policy = found(db.get_policy(policy_id), "policy", policy_id)
    if not db.extend_pilot(policy_id, extra_days=body.extra_days):
        raise HTTPException(
            status_code=409,
            detail=f"only a pilot can be extended (this one is {policy['lifecycle']})",
        )
    return {"policy": with_pilot_info(db.get_policy(policy_id))}


def promotion_evidence(db: Any, policy_id: str) -> dict[str, Any]:
    """Precision and review count for the gate. Derived, never stored."""
    analytics = db.policy_analytics(policy_id)
    reviewed = int(analytics.get("reviewed_total") or 0)
    false_rate = analytics.get("false_positive_rate")
    precision = None if false_rate is None else 1.0 - float(false_rate)
    return {"precision": precision, "reviewed": reviewed}


@router.post("/{policy_id}/promote", dependencies=[NeedsManage])
def promote(policy_id: str, body: PolicyPromote, config: ConfigDep, db: DbDep) -> dict[str, Any]:
    """Pilot proved out -> production, and the gate is enforced *here*.

    Until now the bar — 95% precision over 200+ human reviews — lived in the
    console's JavaScript and nowhere else, so `curl` promoted a policy with zero
    verdicts to auto_routed and started mailing a safety manager. A staged
    rollout that a client can skip is a suggestion; this makes it a rule, with
    the numbers in `rollout:` so a site can set its own.

    **The override is deliberate and it is not a loophole.** A site whose
    validation happened elsewhere — a prior deployment, a labelled dataset, a
    consultant's report — has to be able to say so, and an appliance that
    refuses forever gets worked around by editing the database. So an override
    is allowed, must carry a reason, and the reason is written onto the policy
    row where the next person to ask "why is this live on 12 reviews?" will
    find it.
    """
    policy = found(db.get_policy(policy_id), "policy", policy_id)
    routed_to = body.routed_to or policy.get("routed_to")
    if body.stage == "auto_routed" and not routed_to:
        # Auto-routed with nobody to route to is a policy that fires into the
        # void; the notifier would log a failure for every event.
        raise HTTPException(
            status_code=422,
            detail="promoting to auto_routed needs routed_to (a role in smtp.recipients)",
        )

    rollout = config.rollout
    evidence = promotion_evidence(db, policy_id)
    # Only auto_routed is gated. Promoting to `reviewed` puts a policy in front
    # of humans, which is how the evidence gets collected in the first place.
    if body.stage == "auto_routed":
        # "No verdicts yet" is short only because the review count is short —
        # a site that sets `min_reviews: 0` has said it does not want evidence,
        # and a null precision must not then block it on a bar of zero.
        short = evidence["reviewed"] < rollout.min_reviews or (
            evidence["precision"] is not None
            and evidence["precision"] < rollout.min_precision
        )
        if short and not body.override:
            raise HTTPException(status_code=409, detail={
                "message": (
                    f"not eligible for auto_routed: needs at least "
                    f"{rollout.min_precision:.0%} precision over {rollout.min_reviews} human "
                    f"reviews, and this policy has "
                    + ("no verdicts yet" if evidence["precision"] is None
                       else f"{evidence['precision']:.1%} over {evidence['reviewed']}")
                ),
                "code": "not_eligible",
                **evidence,
                "required": {"precision": rollout.min_precision,
                             "reviews": rollout.min_reviews},
                "override_allowed": rollout.allow_override,
            })
        if short and body.override:
            if not rollout.allow_override:
                raise HTTPException(
                    status_code=409,
                    detail="overriding the promotion gate is disabled on this appliance "
                           "(rollout.allow_override)",
                )
            if not (body.override_reason or "").strip():
                raise HTTPException(
                    status_code=422,
                    detail="an override needs a reason — it is written onto the policy and is "
                           "the only answer the next person gets to 'why is this live?'",
                )

    changed(db.promote_policy(policy_id, stage=body.stage, routed_to=body.routed_to),
            "policy", policy_id)
    if body.stage == "auto_routed" and body.override and body.override_reason:
        db.note_promotion_override(policy_id, body.override_reason.strip(), evidence)
    return {"policy": with_pilot_info(db.get_policy(policy_id)), "evidence": evidence}


# --------------------------------------------------------------------------
# the live prompt test
# --------------------------------------------------------------------------


def _sse(name: str, payload: Any) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, default=str)}\n\n"


async def _run_test(
    config: Any, database: Any, *, camera_id: str, prompt: str, zone: dict[str, Any] | None,
    wait_sec: float,
) -> AsyncIterator[str]:
    """One prompt, one live frame, one answer — streamed as it happens.

    Streamed rather than returned because the three steps have three different
    latencies and three different ways to fail: waiting for the engine to
    publish a frame (up to one 2 s stats tick), cropping it, and the model
    itself (**4.98 s measured** on this box for the shipped 8B). A single JSON
    response would show a spinner for seven seconds and then either everything
    or nothing; this shows the frame the moment it exists, which is also the
    answer to half the questions an author has ("is that even the right
    camera?").

    **Every failure past the first frame is an SSE ``error`` event, not an HTTP
    status.** By then the response has begun, and — more to the point — "the
    model is down" is something the panel should say *underneath the frame it
    already fetched*, exactly as the rest of the console degrades.
    """
    live = LiveFrames.from_config(config)
    started = time.time()
    try:
        # Ask, then wait. The engine notices on its ordinary stats tick; nothing
        # here reaches into the vision loop. See argus/live.py.
        await asyncio.to_thread(live.request, camera_id)
        found = await asyncio.to_thread(
            live.wait_for, camera_id, newer_than=started - 1.0, timeout=wait_sec
        )
    except Exception as exc:  # noqa: BLE001 - a test panel must not 500
        yield _sse("error", {"code": "no_frame", "detail": str(exc)})
        return
    if found is None:
        yield _sse("error", {
            "code": "no_frame",
            "detail": (
                f"no frame from {camera_id} within {wait_sec:.0f}s. Live frames come from "
                "`argus run` — if the engine is stopped, or this camera is not decoding, "
                "there is nothing current to test against."
            ),
        })
        return

    payload, written = found
    image: Any = Image.open(io.BytesIO(payload))
    image.load()
    full_width, full_height = image.size
    cropped = False
    if zone is not None:
        crop = zone_crop(image, zone.get("polygon") or [])
        if crop is not None:
            image, _rect = crop
            cropped = True

    shown = payload if not cropped else _jpeg(image, config.evidence.snapshot_quality)
    yield _sse("frame", {
        "camera_id": camera_id,
        "zone_id": (zone or {}).get("id"),
        "zone_name": (zone or {}).get("name"),
        # False for a zone the crop declined — one covering most of the frame.
        # The panel says which, because "scoped to a zone" and "actually shown
        # the zone" are different facts and the second is the one that matters.
        "cropped": cropped,
        "width": image.size[0],
        "height": image.size[1],
        "frame_width": full_width,
        "frame_height": full_height,
        "age_sec": round(max(0.0, time.time() - written), 2),
        "image": "data:image/jpeg;base64," + base64.b64encode(shown).decode("ascii"),
    })

    if not config.vlm.enabled:
        yield _sse("error", {
            "code": "vlm_disabled",
            "detail": "prompt policies are switched off on this appliance (vlm.enabled)",
        })
        return

    from argus.vlm.client import VlmClient, VlmUnavailable
    from argus.vlm.schema import VlmResponseError

    # A client per request. The engine's own client lives in another process,
    # so `vlm.max_concurrency` was never a bound across both — what actually
    # serialises these is vLLM's scheduler, and the test path is one request a
    # human at a time.
    client = VlmClient(config.vlm, session_id="argus-test")
    yield _sse("asking", {"model": config.vlm.model, "endpoint": client.base_url})
    try:
        verdict = await client.evaluate(prompt, image, priority=True)
    except VlmUnavailable as exc:
        yield _sse("error", {"code": "vlm_unavailable", "detail": str(exc)})
        return
    except VlmResponseError as exc:
        yield _sse("error", {"code": "bad_answer", "detail": str(exc)})
        return
    except Exception as exc:  # noqa: BLE001 - never 500 into a stream
        yield _sse("error", {"code": "failed", "detail": f"{type(exc).__name__}: {exc}"})
        return

    yield _sse("verdict", {
        **verdict.as_dict(),
        "model": client._model_name or config.vlm.model,  # noqa: SLF001 - set by evaluate()
        "cropped": cropped,
    })


def _jpeg(image: Any, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


@router.post("/test", dependencies=[NeedsManage])
async def test_prompt(body: PolicyTest, config: ConfigDep, db: DbDep) -> StreamingResponse:
    """Run a prompt against this camera's freshest frame and stream the answer.

    MANAGE, not VIEW: this is an authoring tool that spends a GPU inference on
    the same device the detector is using, and it is the same permission that
    creates the policy the prompt is for.

    What it does *not* do is create anything. No event row, no snapshot, no
    notification — a test is a question, and a panel that quietly filed alerts
    while an author tuned a sentence would poison the very precision statistics
    the promotion gate reads.
    """
    camera = found(db.get_camera(body.camera_id), "camera", body.camera_id)
    prompt = body.prompt
    if prompt is None and body.policy_id:
        policy = found(db.get_policy(body.policy_id), "policy", body.policy_id)
        prompt = policy.get("prompt")
    if not prompt or not prompt.strip():
        raise HTTPException(
            status_code=422,
            detail="nothing to test: send a prompt, or a policy_id of a policy that has one",
        )
    zone = None
    if body.zone_id:
        zone = found(db.get_zone(body.zone_id), "zone", body.zone_id)
        if str(zone["camera_id"]) != str(camera["id"]):
            raise HTTPException(
                status_code=422,
                detail=f"zone {body.zone_id} is not on camera {camera['id']}",
            )
    return StreamingResponse(
        _run_test(config, db, camera_id=str(camera["id"]), prompt=prompt.strip(),
                  zone=zone, wait_sec=body.wait_sec),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{policy_id}/analytics")
def analytics(policy_id: str, db: DbDep) -> dict[str, Any]:
    """Volume, severity mix, daily trend and false-positive rate — the numbers
    the promotion decision is supposed to rest on."""
    policy = found(db.get_policy(policy_id), "policy", policy_id)
    return {"policy": with_pilot_info(policy), "analytics": db.policy_analytics(policy_id)}
