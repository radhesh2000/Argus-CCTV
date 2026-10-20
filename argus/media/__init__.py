"""Evidence media: annotated snapshots and ring-buffer clips.

Every alert Argus raises carries the picture that justifies it and the ±8s of
video around it. That is the product's answer to "why should I trust this?", so
it lives in its own package rather than inside the engine.
"""

from argus.media.annotate import SEVERITY_COLORS, annotate_zone_event, save_snapshot
from argus.media.ring import RingRecorder

__all__ = ["SEVERITY_COLORS", "RingRecorder", "annotate_zone_event", "save_snapshot"]
