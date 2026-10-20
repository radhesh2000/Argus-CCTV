"""Shared request dependencies and error helpers.

The config and database live on ``app.state`` rather than in module globals, so
a test can stand up an app against a temporary data directory without patching
anything — and so two apps can coexist in one process.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request

from argus.config import ArgusConfig
from argus.db import Database


def get_config(request: Request) -> ArgusConfig:
    return request.app.state.config


def get_db(request: Request) -> Database:
    return request.app.state.db


ConfigDep = Annotated[ArgusConfig, Depends(get_config)]
DbDep = Annotated[Database, Depends(get_db)]


def found(row: Any, what: str, key: Any) -> Any:
    """Return ``row``, or raise the 404 the caller would have written anyway."""
    if row is None:
        raise HTTPException(status_code=404, detail=f"{what} not found: {key}")
    return row


def changed(rowcount: int, what: str, key: Any) -> None:
    """A guarded UPDATE that matched nothing means the row is gone."""
    if not rowcount:
        raise HTTPException(status_code=404, detail=f"{what} not found: {key}")


__all__ = ["ConfigDep", "DbDep", "changed", "found", "get_config", "get_db"]
