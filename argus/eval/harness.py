"""Running the whole thing: which models, on which cases, and what comes out.

The orchestration is deliberately dull — build a set, run each model over it,
score, write two files. The two things in here worth arguing about are both
about restraint.

**The cloud gate is a function that returns a sentence.** :func:`cloud_gate`
answers "may this run reach the internet?" with a boolean and a one-line reason,
and it does that *without importing* ``argus.eval.openrouter``. That ordering is
the whole isolation guarantee: the module that can make a network request is not
loaded into the process until after a human has typed ``--cloud`` and a key is
present. Putting the check inside the cloud module would have meant importing it
to find out whether to import it.

**Models are run one after another, not interleaved.** They share this box.
Running the local vLLM service concurrently with four cloud models would make
every local latency in the report a measurement of contention, and latency is
half of what this harness is for.

**A baseline is re-scored, never copied.** ``--baseline`` folds a previous run's
per-case answers into this report so two local models can appear in one table —
which matters because vLLM serves one model per process, so comparing two of
them on this box is necessarily two runs. What is loaded is the *answers*
(verdict, reason, latency, tokens); the scoring is redone against **this** run's
eval set, so a label corrected since then moves the old model's score too. The
label carries the baseline's date for the one column that cannot be re-derived:
its latency was measured on a different day, under whatever else was running.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from argus.config import ROOT, ArgusConfig
from argus.db import Database
from argus.eval.cases import EvalSet
from argus.eval.dataset import build_eval_set
from argus.eval.metrics import Disagreement, ModelReport, disagreements, score
from argus.eval.runners import LocalRunner, Runner, RunnerResult, run_cases


# What a merged row's label ends with, so a reader can never mistake a stored
# latency for one measured in this run.
BASELINE_TAG = " baseline"

# How a carried-forward note says which run actually made the calls. Also the
# marker that stops it being said twice — see `load_baseline`.
MEASURED_IN = "(measured in "


def emit(*parts: Any) -> None:
    """``print``, flushed.

    The default ``print`` is block-buffered when stdout is a file, so a run
    redirected to a log — which is how anybody watches a 20-minute eval — shows
    nothing at all until it finishes. Progress you cannot see is not progress.
    """
    print(*parts, flush=True)


@dataclass
class EvalRun:
    """One invocation of ``argus eval``, in full."""

    started_at: str
    duration_sec: float = 0.0
    eval_set: EvalSet = field(default_factory=EvalSet)
    reports: list[ModelReport] = field(default_factory=list)
    results: dict[str, list[RunnerResult]] = field(default_factory=dict)
    disagreements: list[Disagreement] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    # Labels in ``reports``/``results`` that came from a previous run's JSON
    # rather than from a model answering today. The ship/hold section needs to
    # know which row is the incumbent, and inferring it from the label text
    # would make a naming convention load-bearing for a decision.
    baseline_labels: list[str] = field(default_factory=list)
    # label -> how many of its cases were asked a *different question* than this
    # run asks. A baseline holds answers, and an answer is only comparable to the
    # question it was given: when ``verification_prompt`` changes, every merged
    # row becomes a before rather than a rival, and the report has to say which.
    prompt_drift: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "duration_sec": round(self.duration_sec, 2),
            "context": self.context,
            "baseline_labels": self.baseline_labels,
            "prompt_drift": self.prompt_drift,
            "notes": self.notes,
            "eval_set": {
                "summary": self.eval_set.summary(),
                "cases": [case.as_dict() for case in self.eval_set.cases],
            },
            "models": [report.as_dict() for report in self.reports],
            "results": {
                model: [result.as_dict() for result in results]
                for model, results in self.results.items()
            },
            "disagreements": [item.as_dict() for item in self.disagreements],
        }


def cloud_gate(
    config: ArgusConfig, *, cloud: bool, env: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """May this run send frames to a third party? Returns (allowed, one-line reason).

    Both conditions are required and the message names the one that is missing,
    because "cloud eval skipped" without a reason is the kind of line somebody
    reads three times before checking their shell history.
    """
    environment = os.environ if env is None else env
    variable = config.eval.cloud_key_env
    has_key = bool(str(environment.get(variable, "")).strip())
    if not cloud and not has_key:
        return False, (
            f"cloud eval skipped: --cloud was not passed and {variable} is not set "
            "(local-only run)"
        )
    if not cloud:
        return False, (
            f"cloud eval skipped: {variable} is set but --cloud was not passed — "
            "the flag is required, a key alone is not consent"
        )
    if not has_key:
        return False, (
            f"cloud eval skipped: --cloud was passed but {variable} is not set in "
            "this environment"
        )
    return True, "cloud eval enabled: --cloud passed and the API key is present"


def _local_runner_or_note(
    config: ArgusConfig, model: str | None, concurrency: int,
) -> tuple[LocalRunner | None, str]:
    """Build the local runner and confirm something is actually serving.

    Probed before any case runs, because the alternative is 60 identical
    connection errors and a report full of zeroes that looks like a bad model.
    """
    runner = LocalRunner.from_config(config, model=model)
    runner.max_concurrency = max(1, concurrency)
    health = asyncio.run(runner.health())
    if not health.get("available"):
        return None, (
            f"local eval skipped: no model at {runner.base_url} "
            f"({health.get('reason')}) — start it with `argus vlm start`"
        )
    served = health.get("model")
    if served and served != runner.name:
        # The endpoint is the contract; the id in config is a guess about it.
        runner.name = str(served)
        runner.label = f"{served} (local)"
    return runner, f"local eval: {runner.name} at {runner.base_url}"


def load_baseline(run: EvalRun, path: Path, *, log: Any = emit) -> list[str]:
    """Fold a previous run's answers into ``run``, re-scored against today's cases.

    vLLM serves one model per process, so two local candidates are two runs by
    construction. This is how they end up in one table without pretending they
    were measured side by side: every score is recomputed from the stored
    verdicts against the current eval set, and the model's label is suffixed
    with the baseline's date so the latency column — the one number that is a
    property of *when* it ran, not of the labels — is never read as fresh.
    """
    payload = json.loads(path.read_text())
    when = str(payload.get("started_at", ""))[:10] or path.stem
    case_ids = {case.id for case in run.eval_set.cases}
    # A baseline file may itself have been produced with --baseline, and those
    # rows already carry the date of the run that measured them. Re-tagging them
    # with *this* file's date would attribute a model's latency to a run it was
    # not measured in — so a label that arrived tagged keeps its tag, and a
    # chain of comparisons stays readable.
    # ``baseline_labels`` is the authority; the suffix check is the fallback for
    # a results file written before that field existed, where the tag is the only
    # record that the row was already someone else's baseline.
    already_tagged = set(payload.get("baseline_labels") or [])
    notes: list[str] = []

    # Per-token rates the stored run measured its cost from, so a cloud model
    # replayed here keeps its cost column. Recomputed rather than copied: the
    # baseline is re-scored against *this* case set, and a different set of
    # frames is a different token count and so a different bill. A model whose
    # file predates these fields simply has no cost column, which reads the same
    # as a local model — hence `pricing_from` below, which is what tells a
    # reader whether "local" means local or means "we lost the price".
    stored_pricing: dict[str, tuple[float, float]] = {}
    for entry in payload.get("models") or []:
        prompt_usd, completion_usd = entry.get("prompt_usd"), entry.get("completion_usd")
        if prompt_usd is None or completion_usd is None:
            continue
        try:
            stored_pricing[str(entry.get("label"))] = (float(prompt_usd), float(completion_usd))
        except (TypeError, ValueError):
            continue

    # Did that run ask the same questions? A results file stores its eval set,
    # prompts and all, so this is answerable rather than assumed — and it has to
    # be asked, because the one thing a stored answer cannot tell you is what it
    # was an answer to. On 2026-08-15 the count clause moved into
    # `verification_prompt` and every counting row measured before it became a
    # *before*: same case ids, same frames, different question.
    stored_prompts = {
        str(case.get("id")): str(case.get("prompt") or "")
        for case in ((payload.get("eval_set") or {}).get("cases") or [])
    }
    asked_now = {case.id: case.prompt for case in run.eval_set.cases}
    shared = [case_id for case_id in stored_prompts if case_id in asked_now]
    reworded = [case_id for case_id in shared if stored_prompts[case_id] != asked_now[case_id]]

    for label, rows in (payload.get("results") or {}).items():
        results = [
            RunnerResult(
                case_id=str(row.get("case_id")),
                model=str(row.get("model") or label),
                triggered=row.get("triggered"),
                confidence=float(row.get("confidence") or 0.0),
                reason=str(row.get("reason") or ""),
                latency_sec=float(row.get("latency_sec") or 0.0),
                prompt_tokens=row.get("prompt_tokens"),
                completion_tokens=row.get("completion_tokens"),
                error=row.get("error"),
            )
            for row in rows
        ]
        matched = sum(1 for result in results if result.case_id in case_ids)
        model = results[0].model if results else label
        carried = label in already_tagged or label.endswith(BASELINE_TAG)
        tagged = label if carried else f"{label} · {when}{BASELINE_TAG}"
        # The same model measured twice on the same day collides here, and that
        # is a run worth keeping rather than an accident: two runs of one model
        # are this harness's own noise floor, which is the only thing that says
        # whether a gap between two models is real.
        if tagged in run.results:
            index = 2
            while f"{tagged} #{index}" in run.results:
                index += 1
            tagged = f"{tagged} #{index}"
        run.results[tagged] = results
        run.reports.append(score(
            model, tagged, run.eval_set.cases, results,
            pricing=stored_pricing.get(label),
        ))
        run.baseline_labels.append(tagged)
        run.prompt_drift[tagged] = len(reworded)
        note = (
            f"baseline: {label} from {path.name} ({when}) — {matched} of "
            f"{len(results)} stored case(s) are in this eval set, re-scored "
            "against the current labels; its latency is from that run, not this one"
        )
        if matched < len(results):
            note += f". {len(results) - matched} stored case(s) no longer exist"
        if reworded:
            note += (
                f". It answered a DIFFERENT prompt on {len(reworded)} of "
                f"{len(shared)} case(s) — read it as a before, not as a rival"
            )
        notes.append(note)
        log(note)

    # Carry forward how the stored run had to ask its questions. "reasoning
    # cannot be disabled on this endpoint — bounded to effort=low" is a property
    # of the measurement, not of the run that happens to be displaying it: it
    # explains that model's latency and cost columns, and a reader of the merged
    # table needs it exactly as much as a reader of the original one did.
    for carried_note in payload.get("notes") or []:
        text = str(carried_note)
        if not text.startswith("cloud eval: ") or text in run.notes:
            continue
        # A note that already names the file it was measured in keeps that name.
        # Re-stamping it with each file it passes through on the way here would
        # grow a tail of provenance one link per re-render, and the link that
        # matters is the first one — the run that actually made the calls.
        notes.append(text if MEASURED_IN in text else f"{text} {MEASURED_IN}{path.name})")

    if not notes:
        note = f"baseline: {path.name} holds no per-case results — nothing merged"
        notes.append(note)
        log(note)
    run.notes.extend(notes)
    return notes


def run_eval(
    config: ArgusConfig,
    database: Database | None = None,
    *,
    cloud: bool = False,
    local: bool = True,
    local_model: str | None = None,
    cloud_models: list[str] | None = None,
    slices: tuple[str, ...] | None = None,
    crop_banner: bool | None = None,
    limit: int | None = None,
    concurrency: int | None = None,
    baseline: Path | None = None,
    env: Mapping[str, str] | None = None,
    log: Any = emit,
) -> EvalRun:
    """Build the eval set, run every enabled model over it, score the lot."""
    started = time.monotonic()
    run = EvalRun(started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    concurrency = max(1, concurrency or config.eval.concurrency)

    run.eval_set = build_eval_set(
        config, database, slices=slices, crop_banner=crop_banner, limit=limit,
    )
    run.notes.extend(run.eval_set.notes)
    log(str(run.eval_set))
    for note in run.eval_set.notes:
        log(f"  {note}")

    # Before the live runners, so the incumbent heads every table and the model
    # under test reads as the challenger.
    if baseline is not None:
        load_baseline(run, baseline, log=log)

    runners: list[Runner] = []
    if local:
        runner, note = _local_runner_or_note(config, local_model, concurrency)
        run.notes.append(note)
        log(note)
        if runner is not None:
            runners.append(runner)
    else:
        run.notes.append("local eval skipped: --no-local")

    allowed, reason = cloud_gate(config, cloud=cloud, env=env)
    run.notes.append(reason)
    log(reason)
    if allowed:
        runners.extend(_cloud_runners(config, cloud_models, run, log))

    run.context = {
        "site": config.site.name,
        "eval_set_size": len(run.eval_set),
        "labelled": len(run.eval_set.scored),
        "concurrency": concurrency,
        "max_tokens": config.eval.max_tokens,
        "cloud_max_tokens": config.eval.cloud_max_tokens if allowed else None,
        "labels_path": str(config.eval_labels_path),
        "models": [runner.label for runner in runners],
        "baseline": str(baseline) if baseline is not None else None,
        "cloud_enabled": allowed,
        # Properties of the appliance, not of whichever model is loaded — the
        # report turns a mean latency into camera-policy pairs with them, and
        # they must not move when `vlm.model` is overridden for a trial.
        "vlm_interval_sec": config.vlm.default_interval_sec,
        "vlm_max_concurrency": config.vlm.max_concurrency,
    }

    if not runners or not run.eval_set.cases:
        run.disagreements = disagreements(run.eval_set.cases, run.results)
        run.duration_sec = time.monotonic() - started
        return run

    for runner in runners:
        log(f"running {len(run.eval_set)} case(s) against {runner.label}…")
        results = asyncio.run(run_cases(
            runner, run.eval_set.cases, concurrency=concurrency, log=log,
        ))
        run.results[runner.label] = results
        report = score(
            runner.name, runner.label, run.eval_set.cases, results,
            pricing=runner.pricing(),
        )
        run.reports.append(report)
        log(f"  {_one_line(report)}")
        # A cloud runner negotiates its request options with the endpoint on
        # the first call that objects (`OpenRouterRunner._degrade`). What it
        # settled on changes what the numbers above mean — bounded reasoning is
        # billed and slower — so it belongs in the report, not in a log line
        # that scrolled past three hours ago.
        for note in getattr(runner, "notes", ()):
            run.notes.append(note)
            log(f"  {note}")
        if getattr(runner, "retried", 0):
            note = (
                f"cloud eval: {runner.name}: {runner.retried} unreadable reply/replies "
                "were re-asked once before being recorded as errors"
            )
            run.notes.append(note)
            log(f"  {note}")

    run.disagreements = disagreements(run.eval_set.cases, run.results)
    run.duration_sec = time.monotonic() - started
    return run


def _cloud_runners(
    config: ArgusConfig, models: list[str] | None, run: EvalRun, log: Any,
) -> list[Runner]:
    """Import and build the cloud runners. Only ever called behind ``cloud_gate``."""
    from argus.eval.openrouter import (  # noqa: PLC0415 - the isolation boundary
        CloudDisabled, attach_pricing, build_runners, cloud_notice,
    )

    candidates = models or config.eval.cloud_model_list
    if not candidates:
        run.notes.append("cloud eval: no candidate models configured (eval.cloud_models)")
        return []
    try:
        built = build_runners(
            candidates,
            base_url=config.eval.cloud_base_url,
            key_env=config.eval.cloud_key_env,
            timeout_sec=config.eval.timeout_sec,
            # Not `eval.max_tokens` — see `EvalConfig.cloud_max_tokens`.
            max_tokens=config.eval.cloud_max_tokens,
        )
    except CloudDisabled as exc:  # pragma: no cover - gate already checked the key
        run.notes.append(f"cloud eval skipped: {exc}")
        return []
    log(cloud_notice(candidates, len(run.eval_set), config.eval.cloud_base_url))
    for note in attach_pricing(built):
        run.notes.append(note)
        log(f"  {note}")
    return list(built)


def _one_line(report: ModelReport) -> str:
    table = report.overall
    parts = [f"n={table.n}"]
    for name, value in (("P", table.precision), ("R", table.recall), ("F1", table.f1)):
        parts.append(f"{name}={value:.3f}" if value is not None else f"{name}=-")
    mean = report.latency.mean
    parts.append(f"mean={mean:.2f}s" if mean is not None else "mean=-")
    if table.errors:
        parts.append(f"errors={table.errors}")
    return " ".join(parts)


def default_json_path(config: ArgusConfig, when: date | None = None) -> Path:
    return config.eval_results_dir / f"results-{(when or date.today()).isoformat()}.json"


def write_json(run: EvalRun, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.as_dict(), indent=2, sort_keys=False) + "\n")
    return path


def build_labels(
    config: ArgusConfig, database: Database | None = None, *, log: Any = emit,
) -> dict[str, Any]:
    """Sample frames out of the test footage and merge them into the label file.

    Policy prompts are seeded from the database's own enabled VLM policies, so
    the questions a human is asked to label are the questions the appliance
    actually asks. Merging never overwrites: see ``labels.py``.
    """
    from argus.eval import frames as frame_tools
    from argus.eval import labels as label_tools

    existing = label_tools.load(config.eval_labels_path)
    policies: dict[str, dict[str, Any]] = {}
    if database is not None:
        scenes = _camera_scenes(database)
        for policy in database.list_policies(kind="vlm", enabled_only=True):
            prompt = str(policy.get("prompt") or "").strip()
            if not prompt:
                continue
            spec: dict[str, Any] = {
                "label": str(policy.get("label") or policy["id"]),
                "prompt": prompt,
            }
            # A scoped policy is scoped in the label file too. The appliance
            # already knows this policy is about one camera (migration 5,
            # `policy_scopes`); carrying that across means the human labelling
            # is asked about the dispatch bay only for the dispatch bay's
            # frames, and the report is not padded with cases nobody can score.
            # An unscoped policy gets nothing, which still means every frame.
            patterns = _scene_patterns(policy.get("scope") or [], scenes)
            if patterns:
                spec["applies_to"] = patterns
            policies[str(policy["id"])] = spec

    videos = frame_tools.find_videos(ROOT)
    if not videos:
        log("labels: no test footage found (packing_section_recording.mp4, footage/)")
    sampled: list[dict[str, Any]] = []
    for video in videos:
        cut = frame_tools.sample_video(
            video, config.eval_frames_dir,
            every_sec=config.eval.sample_every_sec,
            limit=config.eval.max_frames_per_video,
            root=ROOT,
        )
        log(f"labels: {len(cut)} frame(s) from {video.name}")
        for frame in cut:
            sampled.append(label_tools.blank_frame_entry(
                frame.id, frame.source, frame.ts_sec, list(policies),
                sha256=_digest(frame.path),
            ))

    merged, added = label_tools.merge(existing, sampled, policies)
    label_tools.save(config.eval_labels_path, merged)
    counts = label_tools.counts(merged)
    log(f"labels: {config.eval_labels_path} — {counts['frames']} frame(s) "
        f"({added} new), {counts['headcounts']} with a headcount, "
        f"{counts['policy_verdicts']} policy verdict(s), {counts['policies']} prompt(s)")
    return counts


def _camera_scenes(database: Database) -> dict[str, str]:
    """camera id -> the frame-id stem of the video it plays.

    Only file-backed cameras can be answered: an RTSP camera has no video in
    ``footage/`` to have been sampled from, so it maps to nothing and any policy
    scoped to it keeps the "every frame" default rather than silently matching
    none. Deriving the stem through :func:`argus.eval.frames.frame_id` rather
    than re-deriving it here is what keeps the two spellings from parting.
    """
    from argus.eval.frames import frame_id

    scenes: dict[str, str] = {}
    for camera in database.list_cameras():
        source = str(camera.get("source") or "")
        if not source or source.lower().startswith(("rtsp://", "rtsps://", "rtmp://", "http")):
            continue
        # frame_id() appends "-<centiseconds>"; the stem is what precedes it.
        scenes[str(camera["id"])] = frame_id(Path(source), 0.0).rsplit("-", 1)[0]
    return scenes


def _scene_patterns(scope: Any, scenes: Mapping[str, str]) -> list[str]:
    """Glob patterns for the scenes a policy scope names, in file order.

    Returns ``[]`` for an unscoped policy *and* for one scoped entirely to
    cameras whose source is not a local file — both of which mean "do not narrow
    this", which is the only safe default: a pattern that matches nothing would
    turn a policy the appliance runs into a policy the harness never asks.
    """
    patterns: list[str] = []
    for entry in scope or []:
        camera = entry.get("camera_id") if isinstance(entry, dict) else entry
        stem = scenes.get(str(camera or ""))
        if stem and f"{stem}-*" not in patterns:
            patterns.append(f"{stem}-*")
    return patterns


def _digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


__all__ = [
    "EvalRun",
    "build_labels",
    "cloud_gate",
    "default_json_path",
    "load_baseline",
    "run_eval",
    "write_json",
]
