"""Natural-language search, saved searches, and search -> draft policy.

Three endpoints and one flow:

* ``GET /api/search`` — an English sentence through SigLIP's text tower, kNN
  against the frame index with camera and time filters, results carrying a
  thumbnail, a score, a camera, a timestamp and the event they belong to if
  they belong to one.
* ``/api/searches`` — the queries an operator keeps. CRUD.
* ``POST /api/searches/{id}/promote`` — the flow the design is actually about:
  the search that keeps finding the same violation becomes a **draft** VLM
  policy with that query as its prompt, and then walks the ordinary
  draft -> pilot -> production lifecycle like any other policy.

**Why promotion produces a draft and nothing more.** A search is evidence that
a phrase retrieves the right frames; it is not evidence that the phrase is a
good standing question for a model to answer every eight seconds. Creating the
policy in ``draft``/``shadow`` puts it in front of the same pilot-analytics gate
every other policy passes, which is the product's whole thesis about rollout.
Anything that skipped that would be a button that mails a plant's safety
manager based on a similarity score.

**The text tower runs on the CPU here.** ``index.query_device`` defaults to
``cpu``, so a search costs the vision engine and vLLM exactly nothing. Measured
on this box: **68 ms** median to embed one query on the Grace cores against
2.5 ms on the GPU, for 1.0 GB of RSS and no CUDA context in the API process.
68 ms is invisible next to a page load and 2.5 ms is not worth a share of the
SMs the detector is using; sites that disagree set ``query_device: cuda``.
The model is loaded lazily on the first query (2.8 s) and held for the life of
the process, so the first search after a restart pays for the load and the rest
do not.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from argus.api.auth import NeedsManage, NeedsTriage, NeedsView
from argus.api.deps import ConfigDep, DbDep, changed, found
from argus.api.schemas import SavedSearchCreate, SavedSearchPromote, SavedSearchUpdate
from argus.config import ArgusConfig
from argus.db import Database, Policy, slugify

# Searching is VIEW: it returns frames of the plant. *Keeping* a search is
# TRIAGE — a saved search is a shared artefact on the appliance, and the
# operators who work the queue are the ones who curate it. Promoting one to a
# draft policy is MANAGE, because it creates a policy.
router = APIRouter(tags=["search"], dependencies=[NeedsView])

# Guards the lazy build of the query embedder. Two simultaneous first searches
# would otherwise load the checkpoint twice.
_LOAD_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# lazily-built query side
# --------------------------------------------------------------------------


def _query_embedder(request: Request, config: ArgusConfig) -> Any:
    """The text tower, built once per API process and cached on app.state.

    Cached on the app rather than in a module global so two apps in one test
    process do not share one model — the same reason config and db live there.
    """
    state = request.app.state
    embedder = getattr(state, "index_embedder", None)
    if embedder is not None:
        return embedder
    with _LOAD_LOCK:
        embedder = getattr(state, "index_embedder", None)
        if embedder is not None:
            return embedder
        from argus.index.embedder import embedder_from_config

        # `query_device`, not `index.device`: the writer inside `argus run`
        # wants the GPU and the API's text tower deliberately does not.
        embedder = embedder_from_config(
            config, text_only=True, device=config.index.query_device
        )
        state.index_embedder = embedder
        return embedder


def _store(request: Request, config: ArgusConfig, embedder: Any) -> Any:
    store = getattr(request.app.state, "index_store", None)
    if store is None:
        from argus.index.store import VectorStore

        store = VectorStore(config.index_db_path, dim=embedder.dim, model=embedder.name)
        store.migrate()
        request.app.state.index_store = store
    return store


def _unavailable(exc: Exception) -> HTTPException:
    """503, not 500. The index being absent is a state of the appliance an
    operator can fix (``scripts/fetch_siglip.sh``, ``argus index backfill``),
    not a bug in the request — and the console renders the two differently."""
    return HTTPException(
        status_code=503,
        detail=f"the semantic index is not available on this appliance: {exc}",
    )


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------


def _thumb_url(row: dict[str, Any]) -> str:
    """Where the browser fetches this result's picture.

    Two mounts because there are two kinds of row: an event result points at
    the annotated evidence snapshot the operator already knows from the alert
    feed, and a sampled result points at the small JPEG the indexer wrote.
    Collapsing them into one directory would mean either copying evidence or
    losing the annotation.
    """
    prefix = "/snapshots" if row.get("thumb_kind") == "snapshot" else "/index"
    return f"{prefix}/{row['thumb_path']}"


@router.get("/search")
def search(
    request: Request,
    config: ConfigDep,
    db: DbDep,
    q: str = Query(min_length=1, max_length=400, description="what to look for, in plain English"),
    camera_id: str | None = None,
    since_ts: float | None = Query(default=None, description="unix epoch, inclusive"),
    until_ts: float | None = Query(default=None, description="unix epoch, inclusive"),
    kind: str | None = Query(default=None, description="event | sample"),
    limit: int = Query(default=24, ge=1, le=200),
    min_score: float | None = Query(default=None, ge=-1.0, le=1.0),
    saved_id: int | None = Query(default=None, description="count this run against a saved search"),
) -> dict[str, Any]:
    """Rank indexed frames against a sentence.

    Scores are cosine similarities between SigLIP embeddings, which on this
    footage put a good match around 0.10-0.17 and noise below 0.05. They are
    reported raw rather than stretched into a percentage: an operator deciding
    whether to trust a result is better served by a number that means the same
    thing every day than by one rescaled per query to always fill the bar.
    """
    from argus.index.embedder import QUERY_TEMPLATE

    started = time.perf_counter()
    try:
        embedder = _query_embedder(request, config)
        store = _store(request, config, embedder)
    except Exception as exc:  # noqa: BLE001 - a missing model is a 503, not a stack trace
        raise _unavailable(exc) from exc

    text = q.strip()
    if not text:
        raise HTTPException(status_code=422, detail="an empty query matches nothing")
    embed_started = time.perf_counter()
    vector = embedder.embed_texts([QUERY_TEMPLATE.format(text)])[0]
    embed_ms = (time.perf_counter() - embed_started) * 1000.0

    try:
        rows = store.search(
            vector, limit=limit, camera_id=camera_id,
            since_ts=since_ts, until_ts=until_ts, kind=kind,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise _unavailable(exc) from exc

    floor = config.index.min_score if min_score is None else min_score
    rows = [row for row in rows if row["score"] >= floor]

    # Resolve the linked events in one query rather than one per hit: a result
    # strip is 24 tiles and 24 round trips to SQLite for rows the caller wants
    # to render as a group is the kind of thing that only shows up under load.
    event_ids = [int(row["event_id"]) for row in rows if row.get("event_id")]
    events = {
        int(event["id"]): event
        for event in (db.get_event(event_id) for event_id in event_ids)
        if event is not None
    }
    cameras = {camera["id"]: camera for camera in db.list_cameras()}

    results = []
    for row in rows:
        event = events.get(int(row["event_id"])) if row.get("event_id") else None
        camera = cameras.get(row["camera_id"])
        results.append({
            "id": int(row["id"]),
            "camera_id": row["camera_id"],
            "camera_name": (camera or {}).get("name") or row["camera_id"],
            "ts": float(row["ts"]),
            "kind": row["kind"],
            "score": row["score"],
            "thumb_url": _thumb_url(row),
            "width": row.get("width"),
            "height": row.get("height"),
            "event": None if event is None else {
                "id": int(event["id"]),
                "description": event["description"],
                "severity": event["severity"],
                "detector_type": event["detector_type"],
                "detector_name": event["detector_name"],
                "stage": event["stage"],
                "confidence": event["confidence"],
                "clip_path": event["clip_path"],
                "clip_start": event["clip_start"],
                "suppressed": event.get("suppressed"),
                "review_verdict": event.get("review_verdict"),
            },
        })

    if saved_id is not None:
        db.note_saved_search_run(saved_id)

    coverage = store.stats()
    return {
        "query": text,
        "results": results,
        "count": len(results),
        "took_ms": round((time.perf_counter() - started) * 1000.0, 1),
        "embed_ms": round(embed_ms, 1),
        "min_score": floor,
        "filters": {
            "camera_id": camera_id, "since_ts": since_ts, "until_ts": until_ts, "kind": kind,
        },
        # What was actually searched. The design's strip says "SEARCHED 214 H OF
        # FOOTAGE IN 3.1 S", and the only honest version of that sentence is one
        # backed by the index's own span — not by how long the cameras have run.
        "index": {
            "frames": coverage["frames"],
            "events": coverage["events"],
            "samples": coverage["samples"],
            "cameras": coverage["cameras"],
            "span_hours": coverage["span_hours"],
            "oldest_ts": coverage["oldest_ts"],
            "newest_ts": coverage["newest_ts"],
            "model": coverage["model"],
        },
    }


@router.get("/search/status")
def search_status(request: Request, config: ConfigDep) -> dict[str, Any]:
    """Is search usable here, and over what.

    Answers without loading the model: the console asks this on mount, and a
    750 MB checkpoint load triggered by opening a tab is not a page load.
    """
    from argus.index.store import VectorStore

    settings = config.index
    model_ready = settings.model == "fake" or (config.index_model_path / "config.json").is_file()
    loaded = getattr(request.app.state, "index_embedder", None)
    store: Any = getattr(request.app.state, "index_store", None)
    stats: dict[str, Any] | None = None
    error: str | None = None
    if store is None and config.index_db_path.exists():
        # Dim and model are read back from the file's own metadata, so this
        # works before anything has told us which model wrote it.
        try:
            probe = VectorStore(config.index_db_path, dim=0, model="")
            meta = probe.meta()
            store = VectorStore(
                config.index_db_path, dim=int(meta.get("dim", 0) or 0),
                model=str(meta.get("model", "")),
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
    if store is not None:
        try:
            stats = store.stats()
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
    return {
        "enabled": settings.enabled,
        "model": settings.model,
        "model_path": str(config.index_model_path),
        "model_ready": model_ready,
        "query_device": settings.query_device,
        "query_model_loaded": loaded is not None,
        "store_path": str(config.index_db_path),
        "store_exists": config.index_db_path.exists(),
        "index": stats,
        "error": error,
    }


# --------------------------------------------------------------------------
# saved searches
# --------------------------------------------------------------------------


@router.get("/searches")
def list_searches(db: DbDep) -> dict[str, Any]:
    return {"searches": db.list_saved_searches()}


@router.post("/searches", status_code=201, dependencies=[NeedsTriage])
def create_search(body: SavedSearchCreate, db: DbDep) -> dict[str, Any]:
    if body.camera_id and not db.get_camera(body.camera_id):
        raise HTTPException(status_code=404, detail=f"camera not found: {body.camera_id}")
    search_id = db.create_saved_search(
        body.label or body.query, body.query,
        camera_id=body.camera_id, window_hours=body.window_hours,
    )
    return {"search": db.get_saved_search(search_id)}


@router.patch("/searches/{search_id}", dependencies=[NeedsTriage])
def update_search(search_id: int, body: SavedSearchUpdate, db: DbDep) -> dict[str, Any]:
    found(db.get_saved_search(search_id), "saved search", search_id)
    fields = body.model_dump(exclude_unset=True)
    if fields:
        db.update_saved_search(search_id, **fields)
    return {"search": db.get_saved_search(search_id)}


@router.delete("/searches/{search_id}", dependencies=[NeedsTriage])
def delete_search(search_id: int, db: DbDep) -> dict[str, Any]:
    changed(db.delete_saved_search(search_id), "saved search", search_id)
    return {"deleted": search_id}


@router.post("/searches/{search_id}/promote", status_code=201, dependencies=[NeedsManage])
def promote_search(
    search_id: int, body: SavedSearchPromote, db: DbDep, config: ConfigDep
) -> dict[str, Any]:
    """Turn a saved search into a draft VLM policy.

    The prompt is the operator's own sentence turned into the question the
    scheduler asks of a frame. Their words are kept verbatim inside it — a
    rewrite would mean the policy that fires is not the search that was
    validated, which is exactly the link this flow exists to preserve.
    """
    search = found(db.get_saved_search(search_id), "saved search", search_id)
    if search.get("policy_id") and db.get_policy(str(search["policy_id"])):
        raise HTTPException(
            status_code=409,
            detail=f"already promoted to policy {search['policy_id']}",
        )
    label = body.label or str(search["label"])
    policy_id = body.policy_id or db.unique_id("policies", slugify(label))
    if db.get_policy(policy_id):
        raise HTTPException(status_code=409, detail=f"policy already exists: {policy_id}")

    from argus.vlm.scheduler import VLM_DETECTOR_KEY

    db.upsert_policy(Policy(
        id=policy_id,
        kind="vlm",
        detector_key=VLM_DETECTOR_KEY,
        label=label,
        enabled=True,
        # Born in draft/shadow, always. See the module docstring.
        stage="shadow",
        lifecycle="draft",
        severity=body.severity,
        routed_to=None,
        prompt=policy_prompt(str(search["query"])),
        interval_sec=body.interval_sec or config.vlm.default_interval_sec,
        cooldown_sec=body.cooldown_sec or config.evidence.default_cooldown_sec,
        threshold=None,
        pilot_duration_days=body.pilot_duration_days,
    ))
    db.update_saved_search(search_id, policy_id=policy_id)
    return {
        "policy": db.get_policy(policy_id),
        "search": db.get_saved_search(search_id),
    }


def policy_prompt(query: str) -> str:
    """The search phrase as a yes/no question about one frame.

    A retrieval query and a policy prompt are different genres — "person near
    conveyor without a hairnet" ranks frames, but a model asked that flatly
    will describe the frame rather than judge it. The wrapper supplies the
    judgement contract; the quoted phrase stays exactly as it was written and
    exactly as it was validated in the search screen.
    """
    return (
        f'Look at this camera frame and decide whether it shows: "{query.strip()}".\n\n'
        "Answer triggered=true only if what you can see in the frame matches that "
        "description. State in one sentence what you see that supports your answer. "
        "This policy was created from a saved search, so the description is an "
        "operator's own words — judge the frame against them literally."
    )


__all__ = ["policy_prompt", "router"]
