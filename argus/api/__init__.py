"""The Argus HTTP API (FastAPI). See ``argus.api.app`` for the factory."""

from argus.api.app import create_app, run

__all__ = ["create_app", "run"]
