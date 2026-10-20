"""Assembling the eval set: three sources, one shape.

## triage — labels the deployment produced

Every event a human reviewed is a labelled example nobody had to author.
`confirmed` means the detector was right, so a verifier should uphold;
`false_positive` means it should veto. The prompt is
:func:`argus.vlm.schema.verification_prompt`, imported rather than retyped, and
the frame is the annotated snapshot — **banner and all**, because that is
literally what ``VlmRunner.dispatch`` hands the verifier in ``argus run``.

That means this slice is scored on an input containing our own claim in pixels.
That is not a flaw in the slice, it is a property of the deployed system, and
the report says so. It is also why the counting slice below does not do it.

## policy — labels a human authored

Frames sampled out of the test footage, judged against the seeded prompt
policies in ``evals/labels.yaml``. Nothing is drawn on these frames.

## counting — the failure this package exists for

Phase 2 measured Qwen3-VL-8B upholding *"over sanctioned limit: 30 people
(limit 3)"* on a frame holding five, because "over limit" is directionally
true. `vlm.verify.enabled` ships `false` because of that one result, and until
now that result was five hand-run prompts on one frame.

Given a frame whose ground-truth headcount a human wrote down, this generates
three claims — the true count, a wildly inflated one, and an understated one —
and asks them under **two prompts**:

* ``headcount_verify`` — the production verification prompt, verbatim, whatever
  it currently says.
* ``headcount_strict`` — a frozen copy of the count-aware wording.

The pair was the point: if a model fails the first and passes the second, the
fix is a prompt and not a procurement. It did, and the fix was — on 2026-08-15
the clause moved into the production prompt, which is why the two are now the
same text. What the pair does from here is guard that: two identical questions
must produce two identical columns, so a drift in the production wording shows
up as a gap in the report rather than as a quiet loss of the 15/24 -> 24/24 that
paid for the clause.

Both counting policies run on **clean** frames. A claim about five people asked
over a picture with "5 people (limit 3)" burned across the top is a reading-
comprehension test wearing a counting test's clothes.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from argus.config import ROOT, ArgusConfig
from argus.db import Database
from argus.eval.cases import EvalCase, EvalSet, FrameRef
from argus.eval.frames import ensure_frame
from argus.eval.labels import load as load_labels
from argus.vlm.schema import verification_prompt

# The zone name used in generated headcount claims. A real claim carries the
# zone it came from; a generated one needs *a* name, and the seeded zone is the
# one every other measurement in this repository was taken against.
CLAIM_ZONE = "Packing floor"

TRIAGE_POLICY = "detector_verify"
COUNT_VERIFY_POLICY = "headcount_verify"
COUNT_STRICT_POLICY = "headcount_strict"


def headcount_claim(count: int, limit: int) -> str:
    """The exact wording ``Engine._raise_occupancy_event`` writes as a description."""
    return f"Zone '{CLAIM_ZONE}' over sanctioned limit: {count} people (limit {limit})"


def strict_verification_prompt(claim: str) -> str:
    """The count-aware verification wording, frozen as a literal.

    This started as the *variant*: the production prompt plus one clause saying
    the number is part of the claim. It measured 24/24 against production's
    15/24 on over-claims at twice the plausible headcount, so on 2026-08-15 the
    clause moved into :func:`argus.vlm.schema.verification_prompt` and this text
    became what production says.

    **It is deliberately still a copy rather than a call.** Delegating to the
    production function would make this slice track whatever the prompt happens
    to say, which is exactly the property a regression test must not have. Frozen
    here, it is the wording that was proven: ``headcount_verify`` asks production
    and ``headcount_strict`` asks this, so today the two columns must agree, and
    the day somebody edits the production prompt they will separate — in the
    report, on the same page as the number that justified the clause.
    """
    return (
        "An automated safety detector has flagged this camera frame.\n\n"
        f"Detector claim: {claim}\n\n"
        "You are the second opinion before this alert is emailed to a safety "
        "officer. First count the people you can actually see in the image. "
        "The number in the claim is part of the claim: answer triggered=true "
        "only if the claim is supported by what you can see AND its count "
        "matches your own count. Answer triggered=false if the count is wrong, "
        "even when the situation it describes is broadly true. State your own "
        "count in your reason. box_2d must be null."
    )


def counting_claims(people: int, *, exact: bool = False) -> list[tuple[str, int, int, bool]]:
    """``(variant, claimed_count, limit, expected_uphold)`` for a frame of ``people``.

    Five variants, each a different way for a count to be wrong, and each with a
    threshold chosen so the expected answer survives a labelling error of about
    three people. That is not paranoia: the demo footage puts 14-20 people in a
    wide shot at the edge of resolution, and a benchmark whose ground truth only
    holds if the labeller was perfect is a benchmark that measures the labeller.

    * ``truthful`` — the claim is right. Generated **only** where the labeller
      marked the count exact, because this is the one variant an off-by-one
      label would silently invert. Without it the slice would reward a model
      that vetoes everything, so its absence is reported rather than hidden.
    * ``moderate`` — **the discriminating one.** Twice what could possibly be
      there. Measured on this box, Qwen3-VL-8B's tolerance for an over-claim
      scales with how busy the scene is: it vetoed a claim of 4 over a crop
      holding 2, and upheld 40 over a wide shot holding ~16, flipping only
      somewhere past 60 (docs/BENCHMARKS.md). A variant that only probes far
      past that boundary reports a pass and learns nothing, which is what the
      first version of this slice did.
    * ``inflated`` — 6x, the control at the other end: an exaggeration no model
      and no labelling error could defend. A model failing *this* is not
      miscounting, it is not looking.
    * ``understated`` — the same error with the sign flipped, because a model
      that has only learned "big number = suspicious" passes both of the above
      while understanding nothing. Note the asymmetry: when the count is too
      low the claim's *situation* is still true, so under the production wording
      upholding it is arguably defensible — which is exactly why every claim is
      also asked under the count-aware wording, where it is not.
    * ``phantom`` — nobody there at all. A presence checker should get this one,
      so a model failing it is failing at something easier than counting.
    """
    people = max(0, int(people))
    if people == 0:
        return [("phantom", 5, 3, False)]
    # The limit is always one below the true count, so the claim's *situation*
    # ("this zone is over its limit") is true in every variant and only the
    # number is in question. A variant that got the situation wrong too would
    # let a model score by noticing the easy half.
    limit = max(1, people - 1)
    # The most people that could plausibly be in frame given this label. An
    # approximate count on a crowded scene is worth about three either way, and
    # every wrong-count claim is measured from the generous end of that.
    ceiling = people + (0 if exact else 3)
    claims: list[tuple[str, int, int, bool]] = []
    if people >= 2 and exact:
        claims.append(("truthful", people, limit, True))
    claims.append(("moderate", max(2 * ceiling, ceiling + 2), limit, False))
    claims.append(("inflated", max(6 * ceiling, ceiling + 10), limit, False))
    if people >= 6:
        # "2 people" against a labelled 6 is wrong even if the labeller was
        # three out in the generous direction.
        claims.append(("understated", 2, 1, False))
    return claims


def policy_applies(spec: Any, frame_id: str) -> bool:
    """Does this label-file policy ask its question about this frame?

    Absent ``applies_to`` means yes, for every frame, which is what the file
    meant before this key existed and what a single-site appliance still means.

    It exists because the policy slice is a *cross product* — every prompt is
    asked of every frame — and a customer corpus makes that cross product mostly
    nonsense. Seven prompts over the seventy-two frames this repository now
    holds is 504 evaluations at ~5 s each, of which the ~380 asking "are the
    reject trolleys overflowing?" about a dispatch bay are half an hour of GPU
    spent producing unlabelled rows no report can score. Worse, they are *not*
    harmless: an unlabelled case still lands in the disagreements table, so the
    queue a human is told to review next fills with questions nobody asked.

    Patterns are ``fnmatch`` globs against the frame id, so a scene reads as one
    line — ``Line-No-3-Packing3-Bas-*`` — and a hand-added crop row reads as
    another — ``*-gangway``. A malformed value is treated as absent rather than
    as an error: one bad row must not cost a whole eval run, the same rule
    ``_region`` follows above.
    """
    if not isinstance(spec, dict):
        return True
    patterns = spec.get("applies_to")
    if patterns is None:
        return True
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, (list, tuple)) or not patterns:
        return True
    return any(fnmatch(frame_id, str(pattern)) for pattern in patterns)


def _region(value: Any) -> tuple[float, float, float, float] | None:
    """A hand-written ``crop:`` list, or None. Malformed is None, not an error —
    one bad row must not cost a whole eval run."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2)


def _snapshot_path(config: ArgusConfig, event: dict[str, Any]) -> Path | None:
    relative = event.get("snapshot_path")
    if not relative:
        return None
    path = config.snapshots_dir / str(relative)
    return path if path.is_file() else None


def triage_cases(config: ArgusConfig, database: Database) -> tuple[list[EvalCase], list[str]]:
    """Cases from reviewed events. Free labels, and the only ones a site produces."""
    cases: list[EvalCase] = []
    missing = 0
    reviewed = database.reviewed_events_for_export()
    for event in reviewed:
        path = _snapshot_path(config, event)
        if path is None:
            missing += 1
            continue
        verdict = str(event.get("verdict") or "")
        if verdict not in ("confirmed", "false_positive"):
            continue
        reviewer = str(event.get("reviewer") or "").strip() or "unattributed reviewer"
        cases.append(EvalCase(
            id=f"triage-{int(event['id'])}",
            slice="triage",
            policy=TRIAGE_POLICY,
            prompt=verification_prompt(str(event.get("description") or "")),
            # The banner stays on: this is the input the appliance sends.
            frame=FrameRef(path=path, origin=f"event {event['id']} snapshot",
                           crop_banner=False),
            expected=(verdict == "confirmed"),
            label_source=f"triage verdict '{verdict}' by {reviewer}",
            note=str(event.get("description") or ""),
        ))
    notes: list[str] = []
    if reviewed:
        notes.append(
            f"triage: {len(cases)} case(s) from {len(reviewed)} reviewed event(s)"
            + (f", {missing} without a snapshot on disk" if missing else "")
        )
        notes.extend(_reviewer_notes(cases))
    else:
        notes.append("triage: no reviewed events in this database — slice is empty")
    return cases, notes


# Above this share of one slice, a single reviewer is not a diligent operator,
# it is a script. The number is deliberately not a rejection threshold — see
# :func:`_reviewer_notes`.
LOPSIDED_REVIEWER = 0.6


def _reviewer_notes(cases: list[EvalCase]) -> list[str]:
    """Say who produced this slice's labels when one name produced most of them.

    This slice is the one place the harness takes labels it did not ask for, and
    that is its whole appeal — a site's own triage verdicts are free ground
    truth. It is also the failure this note exists for: on 2026-08-25 this
    database held 161 reviewed events, 158 of them filed by ``rc`` in an
    alternating confirmed/false_positive pattern across four distinct
    descriptions, which is a soak test's signature and not a person's judgement.
    Scored, it would have been 96% of the slice and would have moved every
    model's headline F1.

    It is reported and **not** filtered, because no rule here can tell a script
    from a busy supervisor without guessing, and a harness that silently dropped
    a site's labels would be worse than one that names them. The operator
    decides; `--slice` is how they act on it.
    """
    if not cases:
        return []
    tally: dict[str, int] = {}
    for case in cases:
        who = case.label_source.split(" by ", 1)[-1].strip() or "unattributed"
        tally[who] = tally.get(who, 0) + 1
    who, count = max(tally.items(), key=lambda item: item[1])
    if count / len(cases) < LOPSIDED_REVIEWER or len(cases) < 10:
        return []
    return [
        f"triage: {count} of {len(cases)} verdict(s) in this slice were filed by "
        f"{who!r}. One name on most of a slice is worth a look before trusting it — "
        f"a load test reviews faster than a person does. Drop the slice with "
        f"`--slice policy --slice counting` if that is what this is."
    ]


def label_cases(config: ArgusConfig, data: dict[str, Any]) -> tuple[list[EvalCase], list[str]]:
    """Cases from ``evals/labels.yaml``: the policy slice and the counting slice."""
    frames_dir = config.eval_frames_dir
    policies: dict[str, dict[str, str]] = data.get("policies") or {}
    cases: list[EvalCase] = []
    malformed: list[str] = []
    audited: list[str] = []
    unavailable = 0
    counted = 0

    for entry in data.get("frames") or []:
        # Re-cuts the frame from its source video when evals/frames/ is empty,
        # which is the state of every fresh checkout — see frames.py.
        path = ensure_frame(entry, frames_dir, ROOT)
        if path is None:
            unavailable += 1
            continue
        identifier = str(entry.get("id"))
        region = _region(entry.get("crop"))
        origin = f"{entry.get('source')} @ {entry.get('ts_sec')}s" + (
            f" crop {region}" if region else ""
        )
        who = str(entry.get("labelled_by") or "").strip() or "unattributed"
        frame = FrameRef(path=path, origin=origin, crop_banner=False, region=region)
        # `audit:` is one label a human has been asked to look at again, and why.
        # With one labeller the harness cannot tell "the model is wrong" from
        # "the label is wrong" on any single case, so the disagreements a strong
        # model produced are written back into the file as questions rather than
        # silently kept as scores. Every one of them is reported, every run.
        audits = entry.get("audit") if isinstance(entry.get("audit"), dict) else {}

        for key, spec in policies.items():
            prompt = str((spec or {}).get("prompt") or "").strip()
            if not prompt:
                continue
            if not policy_applies(spec, identifier):
                continue
            value = (entry.get("labels") or {}).get(key)
            source = "unlabelled" if value is None else f"labels.yaml, by {who}"
            # Collapsed to one line: these are written as YAML blocks for the
            # human editing the file and rendered as bullets in the report.
            flagged = " ".join(str(audits.get(key) or "").split())
            if flagged:
                source = f"{source} — AUDIT ME: {flagged}"
                audited.append(f"audit: `{identifier}` / {key} — {flagged}")
            cases.append(EvalCase(
                id=f"policy-{identifier}-{key}",
                slice="policy",
                policy=key,
                prompt=prompt,
                frame=frame,
                expected=None if value is None else bool(value),
                label_source=source,
                note=str((spec or {}).get("label") or key),
            ))

        try:
            people = None if entry.get("people") is None else int(entry["people"])
        except (TypeError, ValueError):
            # A hand-edited YAML will eventually contain `people: five`. That is
            # one unusable frame, not a failed eval run.
            malformed.append(identifier)
            people = None
        if people is None:
            continue
        counted += 1
        exact = bool(entry.get("people_exact"))
        for variant, count, limit, expected in counting_claims(people, exact=exact):
            claim = headcount_claim(count, limit)
            note = (
                f"{people} people{'' if exact else ' (approximate)'}; "
                f"claim says {count} (limit {limit})"
            )
            cases.append(EvalCase(
                id=f"count-{identifier}-{variant}",
                slice="counting",
                policy=COUNT_VERIFY_POLICY,
                prompt=verification_prompt(claim),
                frame=frame,
                expected=expected,
                label_source=f"headcount {people} in labels.yaml, by {who}",
                note=note,
            ))
            cases.append(EvalCase(
                id=f"count-{identifier}-{variant}-strict",
                slice="counting",
                policy=COUNT_STRICT_POLICY,
                prompt=strict_verification_prompt(claim),
                frame=frame,
                expected=expected,
                label_source=f"headcount {people} in labels.yaml, by {who}",
                note=note,
            ))

    notes: list[str] = []
    total_frames = len(data.get("frames") or [])
    exact_frames = sum(
        1 for entry in (data.get("frames") or [])
        if entry.get("people") is not None and bool(entry.get("people_exact"))
    )
    if total_frames:
        notes.append(
            f"labels: {total_frames} frame(s), {counted} with a headcount "
            f"({exact_frames} of them exact enough to generate a truthful claim), "
            f"{len(policies)} policy prompt(s)"
        )
        if counted and not exact_frames:
            notes.append(
                "counting: no frame is marked people_exact, so the slice has no "
                "positive control — a model that vetoes everything would score 1.0"
            )
    else:
        notes.append("labels: evals/labels.yaml has no frames — run `argus eval --build-labels`")
    if unavailable:
        notes.append(
            f"labels: {unavailable} frame(s) skipped — not on disk and their source "
            f"video is missing (footage/ is gitignored)"
        )
    if malformed:
        notes.append(
            f"labels: {len(malformed)} frame(s) have an unreadable `people:` value "
            f"and generated no counting cases ({', '.join(malformed[:5])})"
        )
    notes.extend(audited)
    return cases, notes


def build_eval_set(
    config: ArgusConfig,
    database: Database | None = None,
    *,
    slices: tuple[str, ...] | None = None,
    crop_banner: bool | None = None,
    limit: int | None = None,
) -> EvalSet:
    """Assemble every case this box can produce.

    ``crop_banner`` overrides each case's own choice, in both directions. It
    exists for one experiment: running the triage slice with and without the
    burned-in headline answers "how much of the verifier's accuracy is reading
    our own caption?", which is a number worth having before trusting any of it.
    """
    wanted = slices or ("triage", "policy", "counting")
    cases: list[EvalCase] = []
    notes: list[str] = []

    if database is not None and "triage" in wanted:
        found, said = triage_cases(config, database)
        cases.extend(found)
        notes.extend(said)

    if {"policy", "counting"} & set(wanted):
        data = load_labels(config.eval_labels_path)
        found, said = label_cases(config, data)
        cases.extend(case for case in found if case.slice in wanted)
        notes.extend(said)

    if crop_banner is not None:
        for case in cases:
            case.frame.crop_banner = crop_banner
        notes.append(f"frames: banner crop forced {'on' if crop_banner else 'off'} for every case")

    if limit is not None and limit > 0:
        # Labelled cases first, so a truncated run keeps the part that can be
        # scored. Within that, the original order.
        cases.sort(key=lambda case: not case.scored)
        cases = cases[:limit]
        notes.append(f"limit: truncated to {len(cases)} case(s), labelled first")

    return EvalSet(cases=cases, notes=notes)


__all__ = [
    "CLAIM_ZONE",
    "COUNT_STRICT_POLICY",
    "COUNT_VERIFY_POLICY",
    "TRIAGE_POLICY",
    "build_eval_set",
    "counting_claims",
    "headcount_claim",
    "label_cases",
    "policy_applies",
    "strict_verification_prompt",
    "triage_cases",
]
