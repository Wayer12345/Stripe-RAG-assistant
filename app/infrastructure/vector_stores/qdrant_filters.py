"""Mapping helpers from app-level filters to Qdrant filter models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qdrant_client import models

_KEYWORD_FILTER_FIELDS = {
    "document_id",
    "source_type",
    "source_path",
    "url",
    "category",
    "section",
    "content_hash",
    "chunk_id",
}


def _as_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _build_keyword_condition(key: str, raw_values: Iterable[Any]) -> models.FieldCondition | None:
    values = [item for item in raw_values if item is not None and str(item).strip()]
    if not values:
        return None
    if len(values) == 1:
        return models.FieldCondition(
            key=key,
            match=models.MatchValue(value=str(values[0])),
        )
    return models.FieldCondition(
        key=key,
        match=models.MatchAny(any=[str(item) for item in values]),
    )


def build_qdrant_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    """Build a Qdrant Filter from app-level filter dictionary."""
    if not filters:
        return None

    must_conditions: list[models.Condition] = []
    for field_name in _KEYWORD_FILTER_FIELDS:
        if field_name not in filters:
            continue
        condition = _build_keyword_condition(field_name, _as_values(filters[field_name]))
        if condition is not None:
            must_conditions.append(condition)

    token_count = filters.get("token_count")
    if isinstance(token_count, dict):
        range_kwargs: dict[str, int] = {}
        for key in ("gt", "gte", "lt", "lte"):
            value = token_count.get(key)
            if isinstance(value, int):
                range_kwargs[key] = value
        if range_kwargs:
            must_conditions.append(
                models.FieldCondition(
                    key="token_count",
                    range=models.Range(**range_kwargs),
                )
            )
    elif isinstance(token_count, int):
        must_conditions.append(
            models.FieldCondition(
                key="token_count",
                match=models.MatchValue(value=token_count),
            )
        )

    if not must_conditions:
        return None
    return models.Filter(must=must_conditions)

