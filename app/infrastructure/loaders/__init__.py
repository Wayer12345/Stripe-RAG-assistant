"""Loader infrastructure: file loading and parser registry."""

from app.infrastructure.loaders.file_loader import FileLoader
from app.infrastructure.loaders.source_registry import SourceRegistry

__all__ = ["FileLoader", "SourceRegistry"]
