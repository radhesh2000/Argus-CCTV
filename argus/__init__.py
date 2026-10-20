"""Argus — fully-local industrial safety video analytics for the DGX Spark.

Package layout:

* ``argus.config``  — typed YAML/env configuration
* ``argus.db``      — SQLite schema v2, migrations, typed helpers
* ``argus.seed``    — declarative camera/zone/policy import
* ``argus.ingest``  — video plane: go2rtc gateway + ffmpeg decoders
* ``argus.vlm``     — prompt policies judged by a local vision language model
* ``argus.engine``  — the long-running engine process
* ``argus.cli``     — the ``argus`` console script
"""

__version__ = "2.0.0.dev0"

__all__ = ["__version__"]
