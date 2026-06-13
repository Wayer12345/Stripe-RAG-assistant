"""Qdrant collection lifecycle and validation helpers."""

from __future__ import annotations

from qdrant_client import QdrantClient, models


def map_distance(distance: str) -> models.Distance:
    """Map distance string to Qdrant Distance enum."""
    mapping = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
        "manhattan": models.Distance.MANHATTAN,
    }
    normalized = distance.strip().lower()
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported distance metric {distance!r}. "
            f"Supported: {sorted(mapping)}."
        )
    return mapping[normalized]


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Return True when the collection exists in Qdrant."""
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(collection_name=collection_name))
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        return False
    return True


def create_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    vector_dim: int,
    distance: str,
) -> None:
    """Create a collection with expected dense vector settings."""
    if vector_dim <= 0:
        raise ValueError("vector_dim must be > 0.")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dim,
            distance=map_distance(distance),
        ),
    )


def recreate_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    vector_dim: int,
    distance: str,
) -> None:
    """Drop and recreate collection with expected vector config."""
    if vector_dim <= 0:
        raise ValueError("vector_dim must be > 0.")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dim,
            distance=map_distance(distance),
        ),
    )


def validate_collection_config(
    client: QdrantClient,
    *,
    collection_name: str,
    expected_vector_dim: int,
    expected_distance: str,
) -> None:
    """Validate existing collection vector size and distance settings."""
    if not collection_exists(client, collection_name):
        raise ValueError(f"Collection {collection_name!r} does not exist.")

    info = client.get_collection(collection_name=collection_name)
    vectors_config = info.config.params.vectors

    if isinstance(vectors_config, dict):
        raise ValueError(
            "Collection uses named vectors configuration; only single dense vector "
            "collections are supported in this stage."
        )

    actual_dim = vectors_config.size
    actual_distance = vectors_config.distance
    expected_distance_enum = map_distance(expected_distance)

    if actual_dim != expected_vector_dim:
        raise ValueError(
            "Collection vector dimension mismatch: "
            f"expected {expected_vector_dim}, got {actual_dim}."
        )
    if actual_distance != expected_distance_enum:
        raise ValueError(
            "Collection distance mismatch: "
            f"expected {expected_distance_enum.value!r}, got {actual_distance.value!r}."
        )


def create_payload_indexes(
    client: QdrantClient,
    *,
    collection_name: str,
    payload_indexes: dict[str, str],
) -> None:
    """Create payload indexes for configured payload fields."""
    schema_mapping = {
        "keyword": models.PayloadSchemaType.KEYWORD,
        "integer": models.PayloadSchemaType.INTEGER,
        "float": models.PayloadSchemaType.FLOAT,
        "bool": models.PayloadSchemaType.BOOL,
        "datetime": models.PayloadSchemaType.DATETIME,
        "text": models.PayloadSchemaType.TEXT,
    }

    for field_name, schema_name in payload_indexes.items():
        normalized = schema_name.strip().lower()
        if normalized not in schema_mapping:
            raise ValueError(
                f"Unsupported payload schema {schema_name!r} for {field_name!r}. "
                f"Supported: {sorted(schema_mapping)}."
            )
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_mapping[normalized],
        )

