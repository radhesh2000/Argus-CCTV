"""Video ingest plane.

``gateway`` owns the single connection to each camera (go2rtc: RTSP/ONVIF in,
WebRTC out); ``decoder`` turns one of those streams into BGR numpy frames at the
configured detection fps and inference resolution.
"""

from argus.ingest.decoder import Frame, FrameSource, FrameSourceThread
from argus.ingest.gateway import Gateway

__all__ = ["Frame", "FrameSource", "FrameSourceThread", "Gateway"]
