"""The VLM policy engine — Phase 2 of the rebuild plan.

Plain-English safety policies, evaluated against live frames by a vision
language model running on this box. This is the half of the product that does
not need a trained detector: a new check is a new row in ``policies`` with a
prompt in it, not new code and not a labelling campaign.

    argus/vlm/client.py     async OpenAI-compatible client (stdlib HTTP)
    argus/vlm/schema.py     the JSON contract with the model, and its parser
    argus/vlm/scheduler.py  the runner: sampling, cooldowns, events, verification
    argus/vlm/evidence.py   the annotated snapshot for a VLM-raised event
    argus/vlm/server.py     supervising the local vLLM process (`argus vlm`)

Nothing in here imports ``argus.engine`` — the engine holds a
:class:`~argus.vlm.scheduler.VlmRunner` and hands it frames and callbacks, so
the dependency runs one way and the VLM can be tested with no engine at all.
"""

from argus.vlm.client import VlmClient, VlmUnavailable, encode_frame
from argus.vlm.schema import (
    RESPONSE_SCHEMA,
    PolicyVerdict,
    VlmResponseError,
    parse_verdict,
)
from argus.vlm.scheduler import Schedule, VlmRunner

__all__ = [
    "RESPONSE_SCHEMA",
    "PolicyVerdict",
    "Schedule",
    "VlmClient",
    "VlmResponseError",
    "VlmRunner",
    "VlmUnavailable",
    "encode_frame",
    "parse_verdict",
]
