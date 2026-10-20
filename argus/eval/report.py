"""``docs/VLM_EVAL.md`` — the artifact a model decision gets made from.

The report is generated rather than written, for the same reason
``docs/LICENSES.md`` is: a table that has to be updated by hand is a table that
is wrong by the second run. What is hand-written is the *framing* — the caveats
below are constants in this module, not something a future run can drop by
accident, because the numbers are only safe to read alongside them.

Three things this file will always say, whatever the numbers turn out to be:

* how many labels each score rests on, next to the score;
* that the triage slice is scored on frames carrying Argus's own burned-in
  claim, because that is what the appliance sends its verifier;
* why ``z-ai/glm-5.2`` is not in the candidate list.
"""

from __future__ import annotations

from typing import Any

from argus.eval.harness import EvalRun
from argus.eval.metrics import Confusion, ModelReport

# Checked against the OpenRouter catalog on 2026-08-15. Kept as a constant so
# the exclusion survives a run that never talks to the catalog: the reason a
# model is *absent* is the kind of thing that gets lost, and then re-discovered
# by somebody spending an afternoon on it.
EXCLUDED_CANDIDATES: dict[str, str] = {
    "z-ai/glm-5.2": (
        "TEXT-ONLY on OpenRouter — its input modalities do not include `image`, "
        "so it cannot be given a frame. Excluded by inspection of the catalog, "
        "not by a bad score."
    ),
}

# Phase 2 measured what sharing the GPU with the detector costs the VLM:
# 4.49 s isolated became 6.1 s with `argus run` alongside it (docs/BENCHMARKS.md,
# 2026-08-12). Every latency in this report is measured on an idle box, so the
# policy budget it implies is optimistic by roughly this factor unless it is
# applied. It is a constant here rather than a note because a budget quoted from
# an idle-box latency is the mistake this whole section exists to prevent.
#
# It is the *8B's* factor, and Phase 4 found the factor is model-dependent: the
# 27B lost only 11% in service (8.81 s -> ~9.9 s), because a model that far into
# the memory-bandwidth-bound regime barely notices losing SMs to the detector.
# So this over-derates a large candidate — conservatively, in the direction of
# not shipping something the appliance cannot carry, which is the direction to
# be wrong in.
IN_SERVICE_DERATE = 6.1 / 4.49

# The floor Phase 2 quotes for a working appliance: three camera-policy pairs at
# an 8 s interval. A candidate that cannot hold this is not a slower option, it
# is a smaller product.
MIN_POLICY_PAIRS = 3.0

# How much better a challenger has to be before "better" means anything. The
# counting slice is 68 cases per wording, where one flipped verdict is worth
# roughly a point and a half of F1 — so a two-point lead is a handful of frames
# going the other way and a swap of the shipping default should not turn on it.
# Five points is about four verdicts. This guards against the *sample*, not
# against decode jitter: two runs of one model on this box agreed on all 193
# scored verdicts (docs/BENCHMARKS.md, Phase 4). The way to justify a smaller
# threshold is a bigger labelled set, not a smaller threshold.
MATERIAL_F1_GAIN = 0.05

CAVEATS = [
    "**Every score is a small-sample score.** The n column is not decoration: "
    "with fewer than ~30 labelled cases in a cell, a single flipped verdict "
    "moves F1 by several points. Treat rank order as a hypothesis and the "
    "counting slice as the one result worth acting on, because it is the one "
    "designed around a known failure rather than around whatever footage "
    "happened to be on the box.",
    "**The triage slice is scored on frames that carry Argus's own claim.** "
    "The appliance hands its verifier the *annotated* snapshot — headline strip "
    "and all — so a model can score well there partly by reading a caption we "
    "drew. That is a measurement of the deployed system and not a mistake in "
    "the harness, but it means triage precision is an upper bound on what the "
    "model would do with a clean frame. Phase 3 measured the same effect on "
    "SigLIP and it was worth a quarter of the retrieval signal "
    "(docs/PHASE3_NOTES.md). `argus eval --crop-banner` runs the comparison.",
    "**The counting slice uses clean frames on purpose**, and asks each claim "
    "under two prompts: the production wording, and a frozen copy of the "
    "count-aware wording. A model that failed the first and passed the second "
    "meant the fix was a prompt, not a procurement — which is what happened, so "
    "the two now hold the same text and the pair guards it.",
    "**Half of the counting cases are zone crops and half are the wide shots "
    "they were cut from**, and that split is a measurement in itself: the "
    "production wording's failures on the `moderate` variant were 9 wide shots "
    "and 0 crops. A claim about a zone asked over a whole camera view is a "
    "question about impression; asked over the zone it is a question about "
    "counting. The verification pass crops (`VlmRunner._zone_view`) — but only "
    "where the zone is genuinely a sub-picture, and this box's own `Packing "
    "floor` zone spans most of its frame, so on that camera the crop is a "
    "no-op and the wide-shot column is the one that applies.",
    "**Latency is measured at concurrency 1 on an otherwise idle box.** The "
    "appliance runs this model alongside the detector, which cost Qwen3-VL-8B "
    "36% of its latency in Phase 2 and Qwen3.8-27B only 11% in Phase 4 — the "
    "penalty shrinks as a model becomes more memory-bandwidth-bound and less "
    "SM-bound (docs/BENCHMARKS.md). The camera-policy pair counts above apply "
    "the 8B's factor to every row, so a large candidate's budget there is "
    "pessimistic. Do not quote the raw latency column as in-service latency.",
    "**Cloud costs are estimates from published per-token prices** applied to "
    "this eval set's measured token counts. Image tokenisation differs by "
    "provider and a frame at a different resolution is a different bill.",
    "**The cloud candidates answer with a larger token budget than the local "
    "one, and they need it.** The local model decodes under a JSON-schema "
    "constraint and spends 50-90 tokens saying so; every cloud candidate is a "
    "reasoning model whose hidden chain of thought is billed and budgeted as "
    "completion tokens, and two of them (`google/gemini-3.7-flash`, "
    "`qwen/qwen3.8-max`) reject a request that tries to switch that off. Giving "
    "both sides the same 192 tokens is what produced the unusable 2026-08-25 "
    "run — 91% of cloud calls spent the budget thinking and returned no answer. "
    "So the comparison is each model given room to finish, not each model given "
    "the same number of tokens. It does mean the cloud latency and cost columns "
    "include thinking the appliance would never pay for.",
]


def render(run: EvalRun) -> str:
    """The whole document."""
    lines: list[str] = []
    out = lines.append

    out("# VLM evaluation — candidate models on Argus's own policies")
    out("")
    out(f"Generated by `argus eval` at {run.started_at} "
        f"({run.duration_sec:.1f}s). This file is regenerated, not edited — "
        "the raw per-case results are in `evals/results-<date>.json`.")
    out("")
    out(_purpose())
    out("")

    out("## The eval set")
    out("")
    out(_eval_set_table(run))
    out("")
    for note in run.notes:
        out(f"- {note}")
    out("")

    if not run.reports:
        out("**No model was evaluated in this run.** See the notes above.")
        out("")
    else:
        out("## Headline")
        out("")
        out(_headline_table(run.reports))
        out("")
        out(_headline_prose(run.reports))
        out("")

        out("## The counting slice")
        out("")
        out(_counting_section(run))
        out("")

        out("## Ship or hold")
        out("")
        out(_ship_section(run))
        out("")

        out("## Per-policy scores")
        out("")
        out(_policy_tables(run))
        out("")

        out("## Latency and cost")
        out("")
        out(_latency_table(run.reports))
        out("")

        out("## Disagreements")
        out("")
        out(_disagreement_section(run))
        out("")

        out("## Errors")
        out("")
        out(_error_section(run.reports))
        out("")

    out("## Candidates excluded before measurement")
    out("")
    for model, reason in EXCLUDED_CANDIDATES.items():
        out(f"- **`{model}`** — {reason}")
    out("")

    out("## How to read this")
    out("")
    for caveat in CAVEATS:
        out(f"- {caveat}")
    out("")

    out("## Reproducing")
    out("")
    out(_reproduce_section(run))
    out("")
    return "\n".join(lines)


def _purpose() -> str:
    return (
        "`vlm.verify.enabled` ships **false**. The reason was one measurement: "
        "Qwen3-VL-8B upheld a detector claim of *\"30 people (limit 3)\"* on a "
        "frame holding five, because \"over limit\" was directionally true "
        "(docs/PHASE2_NOTES.md). A verifier that cannot audit the one detector "
        "Argus ships — the one whose claim is a number — buys nothing.\n\n"
        "That measurement turned out to be about the *question*. On 2026-08-15 "
        "the verification pass was given the count clause this harness measured "
        "and the zone crop the claim is actually about, and the numbers below "
        "are what changed. The flag stays false — it is an opt-in, not a "
        "default, because it puts a second model between a detected violation "
        "and the person who needs to know — but the case for turning it on is "
        "now a table rather than a hope.\n\n"
        "This harness exists so that the next model is chosen against that "
        "failure and not against a leaderboard. Positive means *the model says "
        "the violation is there*: a false positive is an alert a safety officer "
        "gets for nothing, a false negative is a violation nobody hears about."
    )


def _eval_set_table(run: EvalRun) -> str:
    summary = run.eval_set.summary()
    rows = [
        "| slice | cases | labelled | what it measures |",
        "|---|---|---|---|",
    ]
    descriptions = {
        "triage": "reviewed events — labels the deployment produced, on annotated snapshots",
        "policy": "sampled frames judged by hand against the seeded prompt policies",
        "counting": "generated headcount claims against a hand-written ground-truth count",
    }
    for name in ("triage", "policy", "counting"):
        counts = summary.get(name)
        if counts is None:
            rows.append(f"| {name} | 0 | 0 | {descriptions[name]} |")
            continue
        rows.append(
            f"| {name} | {counts['cases']} | {counts['labelled']} | {descriptions[name]} |"
        )
    rows.append(
        f"| **total** | **{len(run.eval_set)}** | **{len(run.eval_set.scored)}** | |"
    )
    return "\n".join(rows)


def _headline_table(reports: list[ModelReport]) -> str:
    rows = [
        "| model | n | precision | recall | F1 | errors | mean latency | p95 | $/1k evals |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        table = report.overall
        rows.append(
            f"| `{report.label}` | {table.n} | {_pct(table.precision)} | "
            f"{_pct(table.recall)} | {_pct(table.f1)} | {table.errors} | "
            f"{_sec(report.latency.mean)} | {_sec(report.latency.quantile(0.95))} | "
            f"{_usd(report.cost_per_1k)} |"
        )
    # A row whose errors are all somebody's billing department gets said so
    # here, next to the number, and not only in a section further down that a
    # reader comparing F1 scores has no reason to scroll to.
    for report in reports:
        blameless = _blameless(report)
        if blameless:
            rows.append("")
            rows.append(
                f"> `{report.label}` answered only {report.overall.n} of the labelled "
                f"cases: {blameless} of its {report.overall.errors} errors are "
                f"`{'`, `'.join(k for k in report.error_kinds if k in NOT_THE_MODELS_FAULT)}`, "
                "which never reached the model. Read its row as an incomplete run "
                "rather than as a bad model — see **Errors** below."
            )
    return "\n".join(rows)


def _headline_prose(reports: list[ModelReport]) -> str:
    scored = [r for r in reports if r.overall.f1 is not None]
    if not scored:
        return (
            "No model produced a scored verdict — every case in this run was "
            "unlabelled or errored. The latency column is still valid."
        )
    best = max(scored, key=lambda r: r.overall.f1 or 0.0)
    # A run that was refused before it reached the model is a fast run, and it
    # would otherwise take the latency headline off whichever model actually did
    # the work. A refusal is not a measurement of inference speed.
    timed = [r for r in reports if not _mostly_blameless(r)] or reports
    fastest = min(timed, key=lambda r: r.latency.mean or float("inf"))
    lines = [
        f"Best F1: **`{best.label}`** at {_pct(best.overall.f1)} over "
        f"{best.overall.n} labelled case(s).",
        f"Lowest mean latency: **`{fastest.label}`** at {_sec(fastest.latency.mean)}.",
    ]
    if len(scored) == 1:
        lines.append(
            "Only one model ran, so there is no comparison here — this is a "
            "baseline for the next run, not a decision."
        )
    return "\n\n".join(lines)


def _counting_section(run: EvalRun) -> str:
    from argus.eval.dataset import COUNT_STRICT_POLICY, COUNT_VERIFY_POLICY

    counting = run.eval_set.by_slice("counting")
    if not counting:
        return (
            "**Empty.** No frame in `evals/labels.yaml` has a `people:` count, so "
            "the one slice designed around the known failure mode did not run. "
            "Fill in `people` for a few frames and re-run — this is the highest-"
            "value labelling in the file."
        )
    labelled = [case for case in counting if case.scored]
    same_wording = _wordings_agree(run)
    rows = [
        f"{len(counting)} case(s), {len(labelled)} labelled, generated from "
        f"{len({case.frame.path for case in counting})} frame(s) with a ground-truth "
        "headcount. Each claim is asked twice: once with the production "
        "verification prompt as it stands today, once with a frozen copy of the "
        "count-aware wording."
        + (
            " **They are currently the same text** — the clause was measured as a "
            "variant, won, and moved into production — so these two rows are one "
            "question asked twice, and any gap between them is this box's "
            "run-to-run noise. The day they differ, production has drifted from "
            "the wording that was proven."
            if same_wording else
            " **They currently differ**: the production prompt is no longer the "
            "wording the count clause was measured under. Read the gap between "
            "the two rows as the cost of that difference."
        ),
        "",
        "| model | prompt | n | precision | recall | F1 | errors |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for report in run.reports:
        for policy, pretty in (
            (COUNT_VERIFY_POLICY, "production wording"),
            (COUNT_STRICT_POLICY, "count-aware wording"),
        ):
            table = report.by_policy.get(policy)
            if table is None:
                continue
            rows.append(
                f"| `{report.label}` | {pretty} | {table.n} | {_pct(table.precision)} | "
                f"{_pct(table.recall)} | {_pct(table.f1)} | {table.errors} |"
            )
    rows.append("")
    drifted = _reworded_labels(run)
    if drifted:
        rows.append(
            "**The rows marked `baseline` above answered a different question.** "
            + ", ".join(f"`{label}` on {count} case(s)" for label, count in drifted)
            + " — same frames, same ids, a prompt that has changed since. That is "
            "what a fix to `verification_prompt()` looks like from inside this "
            "table: those rows are the *before*, and comparing a pre-change "
            "model's counting score against a post-change one as if they were "
            "rivals is the one mistake this section can produce."
        )
        rows.append("")
    rows.append(_counting_prose(run))
    return "\n".join(rows)


def _wordings_agree(run: EvalRun) -> bool:
    """Do the production and the frozen count-aware prompt ask the same thing?

    Read off this run's own cases rather than by importing both functions, so
    the answer describes the questions that were actually sent.
    """
    from argus.eval.dataset import COUNT_STRICT_POLICY, COUNT_VERIFY_POLICY

    strict = {
        case.id[: -len("-strict")]: case.prompt
        for case in run.eval_set.cases if case.policy == COUNT_STRICT_POLICY
    }
    pairs = [
        (case.prompt, strict[case.id])
        for case in run.eval_set.cases
        if case.policy == COUNT_VERIFY_POLICY and case.id in strict
    ]
    return bool(pairs) and all(production == frozen for production, frozen in pairs)


def _reworded_labels(run: EvalRun) -> list[tuple[str, int]]:
    """Merged rows whose stored answers were given to a different prompt."""
    return [(label, count) for label, count in run.prompt_drift.items() if count]


def _token_delta(before: ModelReport, after: ModelReport) -> str:
    """Where a latency difference between two wordings actually went.

    A longer question and a longer answer are not the same cost — one is
    prefill on an image-heavy prompt, the other is decode, and on this box decode
    is the expensive one. Printed because "the prompt got slower" is not
    actionable and "the model now writes its count into the reason" is.
    """
    parts = []
    for name, first, second in (
        ("prompt", before.tokens.mean_prompt, after.tokens.mean_prompt),
        ("completion", before.tokens.mean_completion, after.tokens.mean_completion),
    ):
        if first is None or second is None or round(second - first) == 0:
            continue
        parts.append(f"{second - first:+.0f} mean {name} token(s)")
    return " and ".join(parts)


VARIANTS = ("truthful", "moderate", "inflated", "understated", "phantom")


def _counting_prose(run: EvalRun) -> str:
    """The breakdown the whole slice exists to make writable.

    Split by *claim variant* rather than aggregated, because the aggregate is
    the one number that cannot answer the question: a model that vetoes
    everything and a model that counts correctly can post the same F1 here, and
    only the `exact`/`inflated` split tells them apart.
    """
    from argus.eval.dataset import COUNT_STRICT_POLICY

    case_by_id = {case.id: case for case in run.eval_set.cases}
    by_variant: dict[str, dict[str, list[bool]]] = {}
    for label, results in run.results.items():
        for result in results:
            case = case_by_id.get(result.case_id)
            if case is None or case.slice != "counting" or not case.scored:
                continue
            if result.triggered is None:
                continue
            wording = "count-aware" if case.policy == COUNT_STRICT_POLICY else "production"
            key = f"{label} · {wording}"
            by_variant.setdefault(key, {}).setdefault(
                _variant_of(case.id), []
            ).append(result.triggered == case.expected)

    if not by_variant:
        return ""
    lines = [
        "Per claim variant — how many each model got right:",
        "",
        "| model · prompt | " + " | ".join(VARIANTS) + " |",
        "|---|" + "---:|" * len(VARIANTS),
    ]
    for key, variants in by_variant.items():
        cells = []
        for variant in VARIANTS:
            hits = variants.get(variant) or []
            cells.append(f"{sum(hits)}/{len(hits)}" if hits else "—")
        lines.append(f"| `{key}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "**`moderate` is the column to read.** It claims twice the most people "
        "who could plausibly be in the frame — near the boundary rather than far "
        "past it, because a model's tolerance for an over-claim scales with how "
        "busy the scene is and a variant that only probes the far side reports a "
        "pass and learns nothing (docs/BENCHMARKS.md has the ladder). `inflated` "
        "is 6x: the control at the other end, an exaggeration nothing could "
        "defend. `truthful` is the control below — without it, a model that "
        "vetoes everything scores 1.0 on the rest and is useless in service. "
        "`understated` catches a model that has merely learned \"large number = "
        "suspicious\"; note that with the count too low the claim's *situation* "
        "is still true, so a miss there is only unambiguous under the "
        "count-aware wording. `phantom` is an empty frame, where even a presence "
        "checker should veto."
    )
    return "\n".join(lines)


def _variant_of(case_id: str) -> str:
    """``count-<frame>-inflated-strict`` -> ``inflated``."""
    body = case_id[: -len("-strict")] if case_id.endswith("-strict") else case_id
    return body.rsplit("-", 1)[-1]


def _policy_pairs(report: ModelReport, run: EvalRun) -> float | None:
    """How many camera-policy pairs this model sustains at the configured interval.

    ``max_concurrency`` evaluations in flight at ``mean`` seconds each is
    ``max_concurrency / mean`` per second; a pair asks for one every
    ``interval`` seconds. The derate is not optional — see
    :data:`IN_SERVICE_DERATE`.
    """
    mean = report.latency.mean
    if not mean:
        return None
    interval = float(run.context.get("vlm_interval_sec") or 8.0)
    concurrency = float(run.context.get("vlm_max_concurrency") or 2)
    return concurrency * interval / (mean * IN_SERVICE_DERATE)


def _ship_section(run: EvalRun) -> str:
    """The decision this document exists to support, re-derived every run.

    Deliberately mechanical. The rule is stated, the two inputs are printed
    beside it, and the verdict follows from them — so a future run with
    different numbers reaches a different conclusion by itself, instead of
    inheriting a sentence somebody wrote once and nobody re-checked.
    """
    from argus.eval.dataset import COUNT_VERIFY_POLICY

    incumbent = next((r for r in run.reports if r.label in run.baseline_labels), None)
    challengers = [r for r in run.reports if r is not incumbent]
    if incumbent is None or not challengers:
        return (
            "Only one model's numbers are in this report, so there is no "
            "ship/hold decision to take from it. `argus eval --baseline "
            "<a previous results JSON>` puts the incumbent's answers into these "
            "same tables: vLLM serves one model per process, so two local "
            "candidates are always two runs."
        )

    def counting(report: ModelReport) -> float | None:
        table = report.by_policy.get(COUNT_VERIFY_POLICY)
        return None if table is None else table.f1

    rule = (
        f"**The rule.** The shipping default stays the incumbent — "
        f"`{incumbent.label}` — unless a challenger is better by at least "
        f"{MATERIAL_F1_GAIN:.2f} F1 on the counting slice *under the production "
        "wording*, which is the failure that keeps `vlm.verify.enabled` false, "
        "**and** its latency still leaves at least "
        f"{MIN_POLICY_PAIRS:.0f} camera-policy pairs at the configured "
        f"{float(run.context.get('vlm_interval_sec') or 8.0):.0f} s interval. "
        "Both conditions, because a model that is better and unusable changes "
        "nothing at a site, and a model that is faster and no better is a "
        "migration for its own sake. The margin is there because one flipped "
        "verdict in 68 is worth a point and a half of F1, so a shipping default "
        "should not turn on a handful of frames going the other way."
    )
    if _reworded_labels(run):
        rule += (
            "\n\n**One column below is not comparable across rows this time.** "
            "The production wording changed since the merged rows were measured, "
            "so their counting F1 answers the old question and this run's answers "
            "the new one. A candidate held under the old wording has not been "
            "re-measured under the new one, and this table cannot do it for them "
            "— that needs the model loaded and `argus eval` run again."
        )

    rows = [
        "| model | counting F1 (production wording) | overall F1 | mean latency | "
        "camera-policy pairs at the configured interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for report in [incumbent, *challengers]:
        pairs = _policy_pairs(report, run)
        rows.append(
            f"| `{report.label}` | {_pct(counting(report))} | "
            f"{_pct(report.overall.f1)} | {_sec(report.latency.mean)} | "
            f"{'—' if pairs is None else f'{pairs:.1f}'} |"
        )

    verdicts = []
    for report in challengers:
        # The same model twice is not a candidate, it is the control: at
        # temperature 0 it should reproduce exactly, and a run that does not is
        # a fact about the serving stack rather than about any model here.
        if report.model == incumbent.model:
            delta = _delta(counting(report), counting(incumbent))
            # Two runs of one model differ for exactly two reasons: the serving
            # stack, or the question. Which one this is decides whether the line
            # below reads as a broken deployment or as a fix that worked, so it
            # is read off the prompts rather than guessed from the size of the
            # delta. Two rows merged from the *same* baseline asked each other's
            # questions, whatever they were, and are still a control.
            reworded = run.prompt_drift.get(incumbent.label, 0)
            if reworded != run.prompt_drift.get(report.label, 0):
                tokens = _token_delta(incumbent, report)
                verdicts.append(
                    f"- `{report.label}`: **BEFORE/AFTER** — the same model as "
                    f"`{incumbent.label}`, asked a different question. The "
                    f"incumbent's stored answers cover {reworded} case(s) whose "
                    "prompt has since changed, so the counting F1 move of "
                    f"{delta} is the prompt's doing and not the model's — and so "
                    f"is the latency: {_sec(incumbent.latency.mean)} -> "
                    f"{_sec(report.latency.mean)} per evaluation"
                    + (f", {tokens}" if tokens else "")
                    + ". Neither row is a candidate. This pair is the evidence "
                    "for a change that has already shipped, and the cost of it."
                )
                continue
            verdicts.append(
                f"- `{report.label}`: **CONTROL** — the same model as the "
                f"incumbent, run again. Counting F1 moved {delta}, mean latency "
                f"{_sec(incumbent.latency.mean)} -> {_sec(report.latency.mean)}. "
                "At `temperature: 0.0` this should be exactly zero, so it "
                "measures run-to-run determinism and nothing else: a difference "
                "here means the serving stack is not reproducible, and a "
                "difference of zero means every gap in the table above is a "
                "property of the models rather than of the harness. It does "
                "**not** bound the sampling error from a 68-case slice — that "
                "is what the "
                f"{MATERIAL_F1_GAIN:.2f} margin is for."
            )
            continue
        better = _better(counting(report), counting(incumbent))
        pairs = _policy_pairs(report, run)
        usable = pairs is not None and pairs >= MIN_POLICY_PAIRS
        if better and usable:
            verdict = "**SHIP**"
            because = (
                "materially better where it matters and still inside the "
                "latency budget"
            )
        else:
            verdict = "**HOLD**"
            reasons = []
            if not better:
                reasons.append(
                    "not materially better on the counting slice under the "
                    "production wording"
                )
            if not usable:
                reasons.append(
                    f"leaves {'no measurable' if pairs is None else f'{pairs:.1f}'} "
                    f"camera-policy pair(s) against a floor of {MIN_POLICY_PAIRS:.0f}"
                )
            because = " and ".join(reasons)
        verdicts.append(f"- `{report.label}`: {verdict} — {because}.")

    swap = (
        "**Either way it is one line.** `vlm.model` in `config/argus.yaml` is "
        "the whole switch — the weights are fetched by `scripts/fetch_vlm.sh "
        "<model id>` and `argus vlm start` passes the path to vLLM. What a site "
        "operator trades by changing it:"
    )
    trades = []
    for report in challengers:
        if report.model == incumbent.model:
            continue
        pairs = _policy_pairs(report, run)
        base_pairs = _policy_pairs(incumbent, run)
        if pairs is None or base_pairs is None:
            continue
        # The direction matters more than the magnitude, and a fixed sentence
        # would read as nonsense for half the candidates: buying accuracy with
        # capacity and buying capacity with accuracy are opposite decisions.
        gained = _better(counting(report), counting(incumbent))
        lost = _better(counting(incumbent), counting(report))
        if pairs < base_pairs:
            closing = (
                "Fewer questions asked more slowly is a smaller product, so "
                + ("the better answers are what pays for it."
                   if gained else
                   "there is nothing here paying for it.")
            )
        else:
            closing = (
                "More questions asked faster is a bigger product, "
                + ("and the answers hold up." if not lost else
                   "bought with answers that are measurably worse on the one "
                   "slice this harness exists for.")
            )
        trades.append(
            f"- `{incumbent.label}` -> `{report.label}`: prompt-policy capacity "
            f"goes from about {base_pairs:.1f} to {pairs:.1f} camera-policy "
            f"pairs at the configured interval "
            f"({_sec(incumbent.latency.mean)} -> {_sec(report.latency.mean)} per "
            f"evaluation), for counting F1 {_pct(counting(incumbent))} -> "
            f"{_pct(counting(report))} under the production wording. " + closing
        )
        for shift in _false_alarm_shifts(incumbent, report):
            trades.append(f"  - {shift}")

    return "\n\n".join([
        rule, "\n".join(rows), "\n".join(verdicts), swap, "\n".join(trades),
    ]).strip()


def _delta(challenger: float | None, incumbent: float | None) -> str:
    if challenger is None or incumbent is None:
        return "—"
    return f"{challenger - incumbent:+.3f}"


def _better(challenger: float | None, incumbent: float | None) -> bool:
    """Better by :data:`MATERIAL_F1_GAIN`, with an unmeasured score never counting."""
    if challenger is None:
        return False
    if incumbent is None:
        return True
    return challenger - incumbent >= MATERIAL_F1_GAIN


def _false_alarm_shifts(incumbent: ModelReport, challenger: ModelReport) -> list[str]:
    """Policies where the challenger raises a different number of false alarms.

    F1 cannot see this everywhere: on a policy whose labels are all negative —
    `phone_usage_on_line` on this footage — precision is 0 and recall is
    undefined, so a model that invents two violations and a model that invents
    none score the same "—". The false-positive count is the only column that
    separates them, and it is the column a safety officer feels.
    """
    shifts: list[str] = []
    for policy in sorted(set(incumbent.by_policy) | set(challenger.by_policy)):
        before = incumbent.by_policy.get(policy)
        after = challenger.by_policy.get(policy)
        if before is None or after is None or before.fp == after.fp:
            continue
        direction = "more" if after.fp > before.fp else "fewer"
        shifts.append(
            f"`{policy}`: {before.fp} -> {after.fp} false alarm(s) in "
            f"{after.n} case(s) — {abs(after.fp - before.fp)} {direction}"
        )
    return shifts


def _policy_tables(run: EvalRun) -> str:
    policies = sorted({case.policy for case in run.eval_set.cases})
    rows = [
        "| policy | model | n | should fire | precision | recall | F1 | errors | unlabelled |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    unlabelled_by_policy: dict[str, int] = {}
    positives_by_policy: dict[str, int] = {}
    labelled_by_policy: dict[str, int] = {}
    for case in run.eval_set.cases:
        if not case.scored:
            unlabelled_by_policy[case.policy] = unlabelled_by_policy.get(case.policy, 0) + 1
            continue
        labelled_by_policy[case.policy] = labelled_by_policy.get(case.policy, 0) + 1
        positives_by_policy[case.policy] = (
            positives_by_policy.get(case.policy, 0) + int(bool(case.expected))
        )
    for policy in policies:
        labelled = labelled_by_policy.get(policy, 0)
        positives = positives_by_policy.get(policy, 0)
        balance = f"{positives}/{labelled}" if labelled else "—"
        for report in run.reports:
            table = report.by_policy.get(policy) or Confusion()
            rows.append(
                f"| `{policy}` | `{report.label}` | {table.n} | {balance} | "
                f"{_pct(table.precision)} | "
                f"{_pct(table.recall)} | {_pct(table.f1)} | {table.errors} | "
                f"{unlabelled_by_policy.get(policy, 0)} |"
            )
    body = "\n".join(rows)
    lopsided = [
        (policy, positives_by_policy.get(policy, 0), labelled_by_policy[policy])
        for policy in policies
        if labelled_by_policy.get(policy, 0) >= 5
        and positives_by_policy.get(policy, 0) in (0, labelled_by_policy[policy])
    ]
    if lopsided:
        # A policy whose labelled frames are all one class has a precision that
        # is a fact about the footage, not about the model: score 1.0 on ten
        # frames that are all violations and you have measured that the clip
        # never showed a clean floor. Naming it here is what stops that number
        # being read as a pass — the fix is footage of the other state, from the
        # site, and it is a request to make before the policy leaves shadow.
        body += "\n\n" + "\n".join(
            f"- **`{policy}` has no counter-example**: {positives} of {labelled} labelled "
            f"frame(s) {'should' if positives else 'should not'} fire. Precision here "
            f"measures the footage, not the policy — this clip never showed the other "
            f"state, so ask the site for one that does before promoting it."
            for policy, positives, labelled in lopsided
        )
    return body


def _latency_table(reports: list[ModelReport]) -> str:
    rows = [
        "| model | evaluations | mean | p50 | p95 | max | mean prompt tok | "
        "mean completion tok | $/1k evals |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        latency, tokens = report.latency, report.tokens
        rows.append(
            f"| `{report.label}` | {latency.n} | {_sec(latency.mean)} | "
            f"{_sec(latency.quantile(0.5))} | {_sec(latency.quantile(0.95))} | "
            f"{_sec(max(latency.samples) if latency.samples else None)} | "
            f"{_num(tokens.mean_prompt)} | {_num(tokens.mean_completion)} | "
            f"{_usd(report.cost_per_1k)} |"
        )
    notes = [report.pricing_note for report in reports if report.pricing_note]
    body = "\n".join(rows)
    if notes:
        body += "\n\n" + "\n".join(f"- {note}" for note in notes)
    local = [r for r in reports if r.cost_per_1k is None and r.latency.mean]
    if local:
        body += "\n\n" + "\n".join(
            f"- `{r.label}` runs on this box: no per-token price, but "
            f"{r.latency.mean * 1000:.0f} GPU-seconds per 1 000 evaluations "
            f"({r.latency.mean * 1000 / 3600:.1f} GPU-hours), taken out of the "
            "detector's budget."
            for r in local
        )
    return body


def _disagreement_section(run: EvalRun) -> str:
    if len(run.results) < 2:
        return (
            "Fewer than two models ran, so there is nothing to disagree about. "
            "Run again with `--cloud` (and `OPENROUTER_API_KEY` set) to compare."
        )
    if not run.disagreements:
        return "Every model answered every case identically."
    lines = [
        f"{len(run.disagreements)} case(s) where the models split. This is the "
        "queue a human should review next: a disagreement on a labelled case "
        "says which model is wrong, and a disagreement on an unlabelled one is "
        "the most informative frame in the set to label.",
        "",
        "| case | policy | expected | verdicts | agreed with label |",
        "|---|---|---|---|---|",
    ]
    for item in run.disagreements[:60]:
        verdicts = ", ".join(
            f"`{model}`={_verdict(value)}" for model, value in sorted(item.verdicts.items())
        )
        agreed = ", ".join(f"`{m}`" for m in item.matches_expected) or "—"
        lines.append(
            f"| `{item.case.id}` | `{item.case.policy}` | "
            f"{_verdict(item.case.expected)} | {verdicts} | {agreed} |"
        )
    if len(run.disagreements) > 60:
        lines.append("")
        lines.append(f"({len(run.disagreements) - 60} more in the JSON results.)")
    lines.append("")
    lines.append("Full reasons, per model per case, are in the JSON results file.")
    return "\n".join(lines)


# Errors that say nothing about the model. An HTTP 402 is the account running
# dry: the request never reached the model, so counting it against the model's
# recall would be scoring our billing arrangement as if it were its eyesight.
# Kept as errors in the table — the run really is incomplete and the n column
# has to show that — but named here so the two are never read as one thing.
NOT_THE_MODELS_FAULT: dict[str, str] = {
    "HTTP 402": (
        "the OpenRouter account ran out of credit mid-run. These requests were "
        "refused before the model saw them, so they are missing measurements "
        "rather than failures — this row is an incomplete run, and its "
        "precision/recall describe only the cases that did get through"
    ),
}


def _blameless(report: ModelReport) -> int:
    """How many of this model's errors never reached the model."""
    return sum(
        count for kind, count in report.error_kinds.items() if kind in NOT_THE_MODELS_FAULT
    )


def _mostly_blameless(report: ModelReport) -> bool:
    """Whether this row describes an interrupted run rather than a model."""
    blameless = _blameless(report)
    return bool(blameless) and blameless >= max(1, report.attempted) / 2


def _classify_errors(report: ModelReport) -> list[str]:
    """The census of what went wrong, commonest first, with the excuses named.

    Counted from `error_kinds` rather than from the `errors` sample, which is
    capped at 50: a model that failed 334 times must not be described as having
    failed 50 times because that is all the report kept.
    """
    lines = []
    for kind, count in sorted(report.error_kinds.items(), key=lambda kv: -kv[1]):
        excuse = NOT_THE_MODELS_FAULT.get(kind)
        if excuse:
            lines.append(f"- **{count} × `{kind}`** — {excuse}.")
        else:
            lines.append(f"- {count} × `{kind}`")
    return lines


def _error_section(reports: list[ModelReport]) -> str:
    failing = [report for report in reports if report.errors]
    if not failing:
        return "No model failed to answer a case."
    lines = []
    for report in failing:
        lines.append(f"**`{report.label}`** — {report.overall.errors} error(s):")
        lines.append("")
        lines.extend(_classify_errors(report))
        lines.append("")
        lines.append("A sample, verbatim:")
        lines.append("")
        for error in report.errors[:10]:
            lines.append(f"- `{error}`")
        lines.append("")
    return "\n".join(lines)


def _reproduce_section(run: EvalRun) -> str:
    return (
        "```bash\n"
        "argus vlm start                 # the local candidate; ~2 min warm\n"
        "argus eval --build-labels       # sample frames, merge into evals/labels.yaml\n"
        "$EDITOR evals/labels.yaml       # fill in `people:` and the policy labels\n"
        "argus eval                      # local only; writes the report and the JSON\n"
        "OPENROUTER_API_KEY=... argus eval --cloud   # adds the cloud candidates\n"
        "argus eval --crop-banner        # the same set with Argus's headline cropped off\n"
        "\n"
        "# Two local candidates are two runs — vLLM serves one model per process:\n"
        "ARGUS_VLM_MODEL=Qwen/Qwen3.8-27B-FP8 argus vlm start\n"
        "ARGUS_VLM_MODEL=Qwen/Qwen3.8-27B-FP8 argus eval \\\n"
        "    --baseline evals/results-<date-of-the-other-run>.json \\\n"
        "    --json evals/results-<date>-<that-model>.json\n"
        "\n"
        "# Re-render this file from answers already measured — no model, no GPU:\n"
        "argus eval --no-local --baseline evals/results-<date>-<model>.json\n"
        "```\n\n"
        f"This run: {run.context.get('eval_set_size', 0)} case(s), "
        f"{run.context.get('labelled', 0)} labelled, concurrency "
        f"{run.context.get('concurrency', 1)}, "
        f"max_tokens {run.context.get('max_tokens')}."
    )


# --- formatting -----------------------------------------------------------


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _sec(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f} s"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def _usd(value: float | None) -> str:
    return "local" if value is None else f"${value:.3f}"


def _verdict(value: Any) -> str:
    if value is None:
        return "—"
    return "**fire**" if value else "no-fire"


__all__ = ["CAVEATS", "EXCLUDED_CANDIDATES", "render"]
