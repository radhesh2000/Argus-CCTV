"""Scoring: precision, recall, latency, cost — and what each of them hides.

**Positive means "the model said the violation is there".** So a false positive
is an alert a safety officer would have received for nothing, and a false
negative is a violation nobody was told about. Those two are not
interchangeable in this product and the report never averages them into a
single number without also printing both.

**Errors are their own column.** A model that fails to answer is neither right
nor wrong; it is unavailable, which for a verifier sitting in the path of an
alert is its own kind of bad. Folding errors into false negatives would let a
model that times out on the hard half look conservative rather than broken.

**Latency is measured over every attempted case, labelled or not.** That is the
whole reason unlabelled frames are still sent. It also means the latency sample
is usually several times larger than the accuracy sample, which is the honest
state of a young eval set and worth seeing in the table.

**p95 on a small sample is a large number, not a percentile.** With fewer than
20 measurements ``p95`` is effectively "the slowest one", and it is reported
anyway because the alternative — hiding the tail — is worse. The count sits
next to it so a reader can discount it appropriately.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from argus.eval.cases import EvalCase
from argus.eval.runners import RunnerResult


@dataclass(slots=True)
class Confusion:
    """A 2x2 table, plus the three ratios anybody actually quotes."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    errors: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float | None:
        denominator = self.tp + self.fp
        return None if denominator == 0 else self.tp / denominator

    @property
    def recall(self) -> float | None:
        denominator = self.tp + self.fn
        return None if denominator == 0 else self.tp / denominator

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    @property
    def accuracy(self) -> float | None:
        return None if self.n == 0 else (self.tp + self.tn) / self.n

    def add(self, expected: bool, predicted: bool) -> None:
        if predicted and expected:
            self.tp += 1
        elif predicted and not expected:
            self.fp += 1
        elif not predicted and expected:
            self.fn += 1
        else:
            self.tn += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "errors": self.errors,
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "f1": _round(self.f1),
            "accuracy": _round(self.accuracy),
        }


@dataclass(slots=True)
class Latency:
    """Wall-clock per evaluation, over every attempted case."""

    samples: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float | None:
        return None if not self.samples else sum(self.samples) / len(self.samples)

    def quantile(self, fraction: float) -> float | None:
        if not self.samples:
            return None
        ordered = sorted(self.samples)
        # Nearest-rank. On a handful of samples this is the only definition that
        # returns a number somebody actually measured rather than an interpolation
        # between two of them.
        index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
        return ordered[index]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_sec": _round(self.mean, 3),
            "p50_sec": _round(self.quantile(0.50), 3),
            "p95_sec": _round(self.quantile(0.95), 3),
            "max_sec": _round(max(self.samples) if self.samples else None, 3),
        }


@dataclass(slots=True)
class Tokens:
    """Mean tokens per evaluation — the input to the cost estimate."""

    prompt_total: int = 0
    completion_total: int = 0
    counted: int = 0

    @property
    def mean_prompt(self) -> float | None:
        return None if not self.counted else self.prompt_total / self.counted

    @property
    def mean_completion(self) -> float | None:
        return None if not self.counted else self.completion_total / self.counted

    def as_dict(self) -> dict[str, Any]:
        return {
            "reported_for": self.counted,
            "mean_prompt": _round(self.mean_prompt, 1),
            "mean_completion": _round(self.mean_completion, 1),
        }


def error_kind(error: str) -> str:
    """The class of an error string, with the case-specific detail stripped.

    Two failures are "the same thing" when a reader would act on them the same
    way, so an HTTP status is a kind and the JSON body that came with it is not,
    and `VlmResponseError: not JSON: ...` is one kind however the parser phrased
    its complaint about column 34. This is what lets the report say "334 of
    these are the account running out of credit" instead of showing ten
    identical lines and leaving the count to the reader.
    """
    text = error.strip()
    status = re.search(r"HTTP (\d{3})", text)
    if status:
        return f"HTTP {status.group(1)}"
    for prefix in ("VlmResponseError: empty response", "unexpected response shape"):
        if prefix in text:
            return prefix
    if "VlmResponseError:" in text:
        return "VlmResponseError: unreadable reply"
    head = text.split(":", 1)[0]
    return head[:60] if head else "error"


@dataclass
class ModelReport:
    """Everything measured about one model on one eval set."""

    model: str
    label: str
    overall: Confusion = field(default_factory=Confusion)
    by_policy: dict[str, Confusion] = field(default_factory=dict)
    by_slice: dict[str, Confusion] = field(default_factory=dict)
    latency: Latency = field(default_factory=Latency)
    tokens: Tokens = field(default_factory=Tokens)
    attempted: int = 0
    unlabelled: int = 0
    # A sample of error strings, capped — enough to read, not enough to drown a
    # report. `error_kinds` is the census, and it is not capped, because "how
    # many of these were the same thing" is the first question anybody asks of
    # an error column and a truncated sample cannot answer it.
    errors: list[str] = field(default_factory=list)
    error_kinds: dict[str, int] = field(default_factory=dict)
    # USD per 1 000 evaluations, when the provider publishes a price.
    cost_per_1k: float | None = None
    pricing_note: str = ""
    # The per-token rates that produced `cost_per_1k`, carried so a results file
    # can be replayed as a baseline without losing its cost column. `cost_per_1k`
    # alone would not do: a baseline is re-scored against the *current* case set,
    # so the token means move and the cost has to be recomputed from the rates.
    # Without these a cloud model chained through `--baseline` renders with an
    # empty cost cell, which `report._usd` prints as "local" — an OpenRouter
    # model labelled as running on the appliance.
    prompt_usd: float | None = None
    completion_usd: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "label": self.label,
            "attempted": self.attempted,
            "unlabelled": self.unlabelled,
            "overall": self.overall.as_dict(),
            "by_policy": {name: table.as_dict() for name, table in sorted(self.by_policy.items())},
            "by_slice": {name: table.as_dict() for name, table in sorted(self.by_slice.items())},
            "latency": self.latency.as_dict(),
            "tokens": self.tokens.as_dict(),
            "cost_per_1k_usd": self.cost_per_1k,
            "pricing_note": self.pricing_note,
            "prompt_usd": self.prompt_usd,
            "completion_usd": self.completion_usd,
            "errors": self.errors[:20],
            "error_kinds": dict(sorted(self.error_kinds.items(), key=lambda kv: -kv[1])),
        }


def score(
    model: str,
    label: str,
    cases: Iterable[EvalCase],
    results: Iterable[RunnerResult],
    *,
    pricing: tuple[float, float] | None = None,
) -> ModelReport:
    """Fold one model's results into a report."""
    by_id = {case.id: case for case in cases}
    report = ModelReport(model=model, label=label)

    for result in results:
        case = by_id.get(result.case_id)
        if case is None:  # pragma: no cover - defensive; ids come from the same set
            continue
        report.attempted += 1
        # Latency and tokens are recorded for every attempt, including the ones
        # that errored: a timeout is a latency measurement, and a slow failure
        # is worse than a fast one.
        report.latency.samples.append(result.latency_sec)
        if result.prompt_tokens is not None or result.completion_tokens is not None:
            report.tokens.prompt_total += int(result.prompt_tokens or 0)
            report.tokens.completion_total += int(result.completion_tokens or 0)
            report.tokens.counted += 1

        policy_table = report.by_policy.setdefault(case.policy, Confusion())
        slice_table = report.by_slice.setdefault(case.slice, Confusion())

        if result.error is not None or result.triggered is None:
            report.overall.errors += 1
            policy_table.errors += 1
            slice_table.errors += 1
            if result.error:
                kind = error_kind(result.error)
                report.error_kinds[kind] = report.error_kinds.get(kind, 0) + 1
                if len(report.errors) < 50:
                    report.errors.append(f"{result.case_id}: {result.error}")
            continue
        if case.expected is None:
            report.unlabelled += 1
            continue
        for table in (report.overall, policy_table, slice_table):
            table.add(case.expected, result.triggered)

    if pricing is not None:
        prompt_usd, completion_usd = pricing
        report.prompt_usd, report.completion_usd = prompt_usd, completion_usd
        mean_prompt = report.tokens.mean_prompt
        mean_completion = report.tokens.mean_completion
        if mean_prompt is not None and mean_completion is not None:
            report.cost_per_1k = round(
                1000 * (mean_prompt * prompt_usd + mean_completion * completion_usd), 4
            )
            report.pricing_note = (
                f"{report.tokens.counted} evaluation(s) with reported usage at "
                f"${prompt_usd * 1e6:.3f}/M prompt + ${completion_usd * 1e6:.3f}/M completion"
            )
        else:
            report.pricing_note = "provider reported no token usage — cost not estimated"
    return report


@dataclass(slots=True)
class Disagreement:
    """One case where the models under test did not answer the same way."""

    case: EvalCase
    verdicts: dict[str, bool | None]
    reasons: dict[str, str]
    confidences: dict[str, float]

    @property
    def matches_expected(self) -> list[str]:
        if self.case.expected is None:
            return []
        return sorted(
            model for model, verdict in self.verdicts.items() if verdict == self.case.expected
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.as_dict(),
            "verdicts": self.verdicts,
            "confidences": {model: round(value, 3) for model, value in self.confidences.items()},
            "reasons": self.reasons,
            "agreed_with_label": self.matches_expected,
        }


def disagreements(
    cases: Iterable[EvalCase], results_by_model: dict[str, list[RunnerResult]],
) -> list[Disagreement]:
    """Every case the models split on — the queue a human should review next.

    Includes cases where one model errored and another answered: "this model
    could not answer that frame" is exactly as interesting to a reviewer as
    "these two models read it differently", and both need the same eyes.
    """
    if len(results_by_model) < 2:
        return []
    indexed = {
        model: {result.case_id: result for result in results}
        for model, results in results_by_model.items()
    }
    out: list[Disagreement] = []
    for case in cases:
        verdicts: dict[str, bool | None] = {}
        reasons: dict[str, str] = {}
        confidences: dict[str, float] = {}
        for model, by_id in indexed.items():
            result = by_id.get(case.id)
            if result is None:
                continue
            verdicts[model] = result.triggered
            reasons[model] = result.reason or (result.error or "")
            confidences[model] = result.confidence
        if len(verdicts) < 2 or len(set(verdicts.values())) < 2:
            continue
        out.append(Disagreement(case=case, verdicts=verdicts, reasons=reasons,
                                confidences=confidences))
    return out


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


__all__ = [
    "Confusion",
    "Disagreement",
    "Latency",
    "ModelReport",
    "Tokens",
    "disagreements",
    "score",
]
