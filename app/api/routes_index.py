"""Index status route definitions."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_settings
from app.infrastructure.storage.manifest_store import read_manifest
from app.infrastructure.vector_stores.qdrant_store import QdrantStore
from app.schemas.api import IndexStatusResponse
from app.utils.config import Settings

router = APIRouter(prefix="/index", tags=["index"])
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get("/status", response_model=IndexStatusResponse)
def index_status_endpoint(settings: SettingsDep) -> IndexStatusResponse:
    """Return cheap index metadata and latest manifest summary."""
    collection_count: int | None = None
    embedding_dimension: int | None = None

    store = QdrantStore(
        mode=settings.vector_store.mode,
        local_path=settings.vector_store.local_path,
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        url=settings.vector_store.url,
        api_key=settings.vector_store.api_key,
        timeout=settings.vector_store.timeout,
        prefer_grpc=settings.vector_store.prefer_grpc,
        collection_name=settings.vector_store.collection_name,
        distance=settings.vector_store.distance,
        upsert_batch_size=settings.vector_store.upsert_batch_size,
        wait=settings.vector_store.wait,
        payload_indexes=settings.vector_store.payload_indexes,
    )
    try:
        if store.collection_exists():
            collection_count = store.count()
    except Exception:
        collection_count = None
    finally:
        store.close()

    manifest_path = settings.indexing.manifest_path
    manifest_payload: dict[str, object] | None = None
    if manifest_path.exists():
        try:
            raw_manifest = read_manifest(manifest_path)
            embedding_dimension = raw_manifest.get("embedding_dim")
            if isinstance(embedding_dimension, bool) or not isinstance(embedding_dimension, int):
                embedding_dimension = None
            manifest_payload = {
                "run_id": raw_manifest.get("run_id"),
                "status": raw_manifest.get("status"),
                "points_total": raw_manifest.get("points_total"),
                "duration_ms": raw_manifest.get("duration_ms"),
            }
        except Exception:
            manifest_payload = None

    return IndexStatusResponse(
        provider=settings.vector_store.provider,
        collection_name=settings.vector_store.collection_name,
        collection_count=collection_count,
        embedding_model=settings.embeddings.model_name,
        embedding_dimension=embedding_dimension,
        index_manifest_path=str(manifest_path),
        index_manifest=manifest_payload,
    )
