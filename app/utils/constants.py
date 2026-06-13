"""Shared stable constants."""

STAGE_INGESTION = "ingestion"
STAGE_CLEANING = "cleaning"
STAGE_CHUNKING = "chunking"
STAGE_EMBEDDINGS = "embeddings"
STAGE_VECTOR_INDEXING = "vector_indexing"
STAGE_RETRIEVE = "retrieve"
STAGE_RERANK = "rerank"
STAGE_BUILD_CONTEXT = "build_context"
STAGE_GENERATE_ANSWER = "generate_answer"
STAGE_ONLINE_QUERY = "online_query"

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_PARTIAL = "partial"

DEFAULT_CONFIG_FILE_NAME = "config.yaml"
ARTIFACT_SCHEMA_VERSION = "1.0"
DEFAULT_RETRIEVE_TEXT_PREVIEW_CHARS = 300
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_NONE = "none"
