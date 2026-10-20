"""``argus eval`` — measuring candidate vision models on this product's own work.

Phase 2 shipped a VLM policy engine and, with it, a claim that is currently
unfalsifiable: *"Qwen3-VL-8B is a presence checker, not a counter."* That
sentence came from five hand-run prompts on one frame (docs/BENCHMARKS.md,
"Verification pass"). It was enough to ship `vlm.verify.enabled: false`, and it
is nowhere near enough to decide whether some other model should replace it.
This package turns that judgement into a command.

**What it measures.** Triples of (frame, policy prompt, expected verdict), drawn
from three places and scored the same way:

* **triage** — every event a human has reviewed. `confirmed` means the detector
  was right, so the verifier should uphold; `false_positive` means it should
  veto. These labels are free, they accumulate as the appliance is used, and
  they are the only ones that come from the deployment rather than from us.
* **policy** — frames sampled out of the test footage and labelled by hand in
  ``evals/labels.yaml`` against the seeded prompt policies.
* **counting** — generated from a frame's ground-truth headcount, and the reason
  this package exists. It reproduces the exact failure that keeps verification
  off: a claim of "30 people (limit 3)" about a frame holding five.

**What it deliberately does not do.** It does not touch the running appliance,
it does not write to `argus.sqlite3` (it reads events and reviews), and it is
not on any runtime path. `argus/eval/openrouter.py` — the only module here that
can reach the internet — is imported at call time behind two gates, and
`tests/test_eval.py` asserts that importing the engine, the API and the
scheduler leaves `argus.eval` out of ``sys.modules`` entirely.

**On unlabelled frames.** They are still sent, still timed, and never scored.
Latency and cost are properties of a model and a frame size; accuracy is a
property of a model and a *label*. Conflating them would mean a thin label set
also produced a thin latency measurement, which is exactly backwards — the
timing is the part that is trustworthy on day one.
"""

from __future__ import annotations

from argus.eval.cases import EvalCase, EvalSet, FrameRef
from argus.eval.metrics import Confusion, ModelReport, score
from argus.eval.runners import LocalRunner, RunnerResult

__all__ = [
    "Confusion",
    "EvalCase",
    "EvalSet",
    "FrameRef",
    "LocalRunner",
    "ModelReport",
    "RunnerResult",
    "score",
]
