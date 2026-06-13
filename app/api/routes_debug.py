"""Debug route definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_settings
from app.infrastructure.storage.manifest_store import read_manifest
from app.utils.config import Settings

router = APIRouter(prefix="/debug", tags=["debug"])
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get("/trace")
def debug_trace_endpoint(
    settings: SettingsDep,
    path: str = Query(..., min_length=1),
) -> dict[str, Any]:
    """Read a trace file only when it is under configured trace directory."""
    candidate = Path(path).resolve()
    allowed_dir = settings.online_query.trace_dir.resolve()
    if allowed_dir not in candidate.parents and candidate != allowed_dir:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Trace path must be under configured trace directory.",
        )
    if not candidate.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trace file not found.")
    return read_manifest(candidate)
