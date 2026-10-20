"""Vision core: detection, tracking, zone occupancy and event debouncing.

The pipeline is deliberately split so each stage is testable on its own:

``detector`` turns a batch of decoded frames into person boxes, ``tracker``
gives those boxes stable identities per camera, ``zones`` answers "who is
standing inside which polygon", and ``debounce`` decides whether a standing
violation is worth another alert.
"""

from argus.vision.debounce import Debouncer
from argus.vision.zones import Occupancy, Zone, ZoneSet

__all__ = ["Debouncer", "Occupancy", "Zone", "ZoneSet"]
