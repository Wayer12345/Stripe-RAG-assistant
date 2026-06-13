# Project Structure (actual)

```text
stripe-rag-assistant/
├── app/                                          # Main Python package
│   ├── __init__.py                               # Marks app as an importable Python package
│   ├── api/                                      # FastAPI routes and dependency wiring
│   │   ├── __init__.py                           # Exports API routers
│   │   ├── dependencies.py                       # FastAPI dependency providers for runtime state
│   │   ├── routes_debug.py                       # Debug endpoints for pipeline inspection
│   │   ├── routes_health.py                      # Health-check endpoint definitions
│   │   ├── routes_index.py                       # Index status and management endpoints
│   │   └── routes_query.py                       # Main query endpoint for RAG requests
│   ├── application/                              # High-level service orchestration
│   │   ├── __init__.py                           # Exports application services
│   │   ├── api_query_service.py                  # API-specific online query orchestration
│   │   ├── eval_service.py                       # Evaluation flow orchestration service
│   │   ├── indexing_service.py                   # Offline indexing flow orchestration service
│   │   └── query_service.py                      # Minimal online question-answering orchestration
│   ├── application_layers/                       # Runnable stage-level wrappers
│   │   ├── eval/
│   │   │   ├── __init__.py                       # Exports evaluation application-layer modules
│   │   │   ├── build_eval_dataset.py             # Builds evaluation datasets from artifacts
│   │   │   ├── run_citation_eval.py              # Runs citation evaluation mode
│   │   │   ├── run_context_eval.py               # Runs context evaluation mode
│   │   │   ├── run_generation_eval.py            # Runs generation evaluation mode
│   │   │   ├── run_retrieval_eval.py             # Runs retrieval evaluation mode
│   │   │   ├── run_rerank_eval.py                # Runs rerank evaluation mode
│   │   │   └── run_robustness_eval.py            # Runs robustness evaluation mode
│   │   ├── offline/
│   │   │   ├── build_chunks.py                   # Cleaned documents to chunk artifacts
│   │   │   ├── build_cleaned_docs.py             # Parsed documents to cleaned artifacts
│   │   │   ├── build_embeddings.py               # Chunks to embedded chunk artifacts
│   │   │   ├── build_parsed_docs.py              # Raw sources to parsed document artifacts
│   │   │   └── build_vector_index.py             # Embedded chunks to Qdrant vector index
│   │   └── online/
│   │       ├── build_context.py                  # Reranked results to final context bundle
│   │       ├── generate_answer.py                # Context bundle and question to grounded answer
│   │       ├── rerank.py                         # Retrieval candidates to reranked candidates
│   │       └── retrieve.py                       # Question to retrieval candidates
│   ├── domain/                                   # Domain models and contracts
│   │   ├── __init__.py                           # Exports domain layer
│   │   ├── interfaces/
│   │   │   ├── __init__.py                       # Exports domain interface contracts
│   │   │   ├── chunker.py                        # Chunking strategy contract
│   │   │   ├── cleaner.py                        # Document cleaning contract
│   │   │   ├── document_loader.py                # Raw document loading contract
│   │   │   ├── embedder.py                       # Dense embedding model contract
│   │   │   ├── evaluator.py                      # Evaluation component contract
│   │   │   ├── generator.py                      # Grounded answer generation contract
│   │   │   ├── lexical_index.py                  # Lexical index operations contract
│   │   │   ├── output_parser.py                  # Generation output parsing contract
│   │   │   ├── parser.py                         # Raw document parsing contract
│   │   │   ├── reranker.py                       # Reranking component contract
│   │   │   ├── retriever.py                      # Query-time retrieval contract
│   │   │   └── vector_store.py                   # Vector-store operations contract
│   │   └── models/
│   │       ├── __init__.py                       # Exports domain model package
│   │       ├── answer.py                         # Generated answer and confidence models
│   │       ├── chunk.py                          # Retrieval-ready chunk model
│   │       ├── context.py                        # Final context bundle model
│   │       ├── document.py                       # Normalized ingested document model
│   │       ├── embedded_chunk.py                 # Chunk model with embedding vector
│   │       ├── eval_case.py                      # Evaluation test-case model
│   │       ├── retrieval_result.py               # Retrieval/rerank candidate result model
│   │       └── source.py                         # Source model supporting an answer
│   ├── evaluation/                               trics, judges, and reporting
│   │   ├── __init__.py                           # Exports evaluation package
│   │   ├── citation_metrics.py                   # Deterministic citation and support metrics
│   │   ├── confidence_metrics.py                 # Deterministic confidence and abstention metrics
│   │   ├── context_metrics.py                    # Deterministic context quality metrics
│   │   ├── dataset_builder.py                    # Builds eval datasets from chunk artifacts
│   │   ├── datasets.py                           # Loads, validates, and filters eval datasets
│   │   ├── generation_metrics.py                 # Deterministic generation quality metrics
│   │   ├── judges.py                             # Judge adapters used by evaluation runner
│   │   ├── latency_metrics.py                    # Deterministic latency helpers for eval
│   │   ├── records.py                            # Core records for eval datasets and runs
│   │   ├── regression.py                         # Baseline comparison helpers for regressions
│   │   ├── reports.py                            # Eval report and artifact builders
│   │   ├── rerank_metrics.py                     # Deterministic reranking metrics
│   │   ├── retrieval_metrics.py                  # Deterministic retrieval metrics
│   │   ├── robustness_metrics.py                 # Deterministic robustness subset metrics
│   │   ├── runner.py                             # Core deterministic evaluation runner
│   │   └── utils.py                              # Shared evaluation utilities
│   ├── infrastructure/                           ncrete technical implementations
│   │   ├── __init__.py                           # Exports infrastructure layer
│   │   ├── chunking/
│   │   │   ├── __init__.py                       # Exports chunking implementations
│   │   │   ├── chunker_factory.py                # Factory for configured chunkers
│   │   │   └── semantic_chunker.py               # Structure-aware deterministic chunker
│   │   ├── cleaning/
│   │   │   ├── __init__.py                       # Exports cleaning implementations
│   │   │   ├── boilerplate.py                    # Conservative boilerplate line removal
│   │   │   ├── cleaner_factory.py                # Factory for configured TextCleaner instances
│   │   │   ├── html_cleaner.py                   # Residual HTML artifact cleanup
│   │   │   ├── normalizers.py                    # Pure text normalization functions
│   │   │   └── text_cleaner.py                   # Main TextCleaner implementing Cleaner protocol
│   │   ├── context/
│   │   │   ├── __init__.py                       # Exports context-building components
│   │   │   ├── context_builder.py                # Packs retrieval results into context bundle
│   │   │   ├── context_factory.py                # Factory for configured context builders
│   │   │   └── context_formatter.py              # Deterministic context rendering for LLM input
│   │   ├── embeddings/
│   │   │   ├── __init__.py                       # Exports embedding components
│   │   │   ├── embedder_factory.py               # Factory for configured embedders
│   │   │   ├── embedding_cache.py                # Local filesystem embedding cache
│   │   │   └── sentence_transformer_embedder.py  # Local sentence-transformers implementation
│   │   ├── generation/
│   │   │   ├── __init__.py                       # Exports generation components
│   │   │   ├── answer_generator.py               # Orchestrates grounded answer generation
│   │   │   ├── ollama_client.py                  # HTTP client for local Ollama service
│   │   │   ├── output_parser.py                  # Parses raw LLM output into GeneratedAnswer
│   │   │   └── prompt_renderer.py                # Renders Jinja prompt templates
│   │   ├── loaders/
│   │   │   ├── __init__.py                       # Exports loader components
│   │   │   ├── file_loader.py                    # Loads filesystem files into RawDocument payloads
│   │   │   └── source_registry.py                # Resolves parser by source type
│   │   ├── parsers/
│   │   │   ├── __init__.py                       # Exports parser implementations
│   │   │   ├── csv_parser.py                     # Parses CSV payloads into Document objects
│   │   │   ├── docx_parser.py                    # Parses DOCX payloads into Document objects
│   │   │   ├── html_parser.py                    # Parses HTML payloads into Document objects
│   │   │   ├── json_parser.py                    # Parses JSON/JSONL payloads into Document objects
│   │   │   ├── markdown_parser.py                # Parses Markdown payloads into Document objects
│   │   │   ├── pdf_parser.py                     # Parses PDF payloads into Document objects
│   │   │   └── txt_parser.py                     # Parses TXT payloads into Document objects
│   │   ├── reranking/
│   │   │   ├── __init__.py                       # Exports reranking components
│   │   │   ├── cross_encoder_reranker.py         # Cross-encoder reranker with optional cache
│   │   │   └── reranker_factory.py               # Factory for configured rerankers
│   │   ├── retrieval/
│   │   │   ├── __init__.py                       # Exports retrieval components
│   │   │   ├── dense_retriever.py                # Dense retriever using embedder and vector store
│   │   │   └── retriever_factory.py              # Factory for configured retrieval strategies
│   │   ├── storage/
│   │   │   ├── __init__.py                       # Exports storage components
│   │   │   ├── artifact_paths.py                 # Resolves paths for offline artifacts
│   │   │   ├── jsonl_store.py                    # Reads/writes JSONL pipeline artifacts
│   │   │   ├── manifest_store.py                 # Reads/writes pipeline run manifests
│   │   │   └── trace_loader.py                   # Loads stage trace JSON files into domain models
│   │   └── vector_stores/
│   │       ├── __init__.py                       # Exports vector store implementations
│   │       ├── qdrant_collections.py             # Qdrant collection lifecycle and validation helpers
│   │       ├── qdrant_filters.py                 # Maps app-level filters to Qdrant filters
│   │       ├── qdrant_store.py                   # Qdrant-backed vector store implementation
│   │       └── vector_store_factory.py           # Factory for configured local vector store instances
│   ├── schemas/
│   │   ├── __init__.py                           # Exports API schema models
│   │   ├── api.py                                # Shared API envelope schemas
│   │   ├── query.py                              # Request schemas for query endpoints
│   │   └── response.py                           # Response schemas for API endpoints
│   └── utils/
│       ├── __init__.py                           # Exports shared utility modules
│       ├── config.py                             # Typed settings loader from YAML and environment
│       ├── constants.py                          # Shared stable project constants
│       ├── hashing.py                            # Generic hashing helpers
│       ├── ids.py                                # Deterministic ID generation utilities
│       ├── logging.py                            # Project logging helpers
│       └── timing.py                             # Timing utilities for scripts and orchestration
├── configs/
│   └── config.yaml                               # Unified project configuration file
├── prompts/
│   ├── answer_prompt_v1.jinja                    # Main grounded-answer prompt template
│   ├── judge_prompt_v1.jinja                     # Judge prompt template for evaluation
│   ├── no_answer_prompt_v1.jinja                 # No-answer prompt when evidence is insufficient
│   └── query_rewrite_prompt_v1.jinja             # Query rewrite prompt template
├── scripts/
│   ├── eval/                                     # Smoke scripts for evaluation modes
│   │   ├── smoke_build_eval_dataset.py           # Smoke wrapper for eval dataset build
│   │   ├── smoke_run_answer_eval.py              # Deprecated smoke wrapper for generation eval
│   │   ├── smoke_run_citation_eval.py            # Smoke wrapper for citation eval mode
│   │   ├── smoke_run_context_eval.py             # Smoke wrapper for context eval mode
│   │   ├── smoke_run_eval.py                     # Smoke wrapper for full eval suite
│   │   ├── smoke_run_generation_eval.py          # Smoke wrapper for generation eval mode
│   │   ├── smoke_run_retrieval_eval.py           # Smoke wrapper for retrieval eval mode
│   │   ├── smoke_run_rerank_eval.py              # Smoke wrapper for rerank eval mode
│   │   └── smoke_run_robustness_eval.py          # Smoke wrapper for robustness eval mode
│   └── online/                                   # Smoke scripts for online layers
│       ├── smoke_build_context.py                # Smoke wrapper for build-context layer
│       ├── smoke_generate_answer.py              # Smoke wrapper for generate-answer layer
│       ├── smoke_rerank.py                       # Smoke wrapper for rerank layer
│       └── smoke_retrieve.py                     # Smoke wrapper for retrieve layer
├── tests/
│   ├── integration/
│   │   ├── test_api_lifecycle.py                 # Integration tests for API lifecycle behavior
│   │   ├── test_api_query.py                     # Integration tests for API query endpoint
│   │   ├── test_build_context_layer.py           # Integration tests for build-context layer
│   │   ├── test_chunking_pipeline.py             # Integration tests for chunking pipeline
│   │   ├── test_cleaning_pipeline.py             # Integration tests for cleaning pipeline
│   │   ├── test_embeddings_pipeline.py           # Integration tests for embeddings pipeline
│   │   ├── test_eval_application_layers.py       # Integration tests for eval application layers
│   │   ├── test_eval_pipeline.py                 # Integration tests for end-to-end eval pipeline
│   │   ├── test_eval_service.py                  # Integration tests for eval_service orchestration
│   │   ├── test_generate_answer_layer.py         # Integration tests for generate-answer layer
│   │   ├── test_indexing_service.py              # Integration tests for indexing_service
│   │   ├── test_ingestion_pipeline.py            # Integration tests for ingestion pipeline
│   │   ├── test_rerank_layer.py                  # Integration tests for rerank layer
│   │   ├── test_retrieve_layer.py                # Integration tests for retrieve layer
│   │   └── test_vector_indexing_pipeline.py      # Integration tests for vector indexing pipeline
│   └── unit/
│       ├── test_api_query_service.py             # Unit tests for API query service
│       ├── test_api_schemas.py                   # Unit tests for API schema models
│       ├── test_chunking.py                      # Unit tests for chunking behavior
│       ├── test_cleaning.py                      # Unit tests for cleaning behavior
│       ├── test_config.py                        # Unit tests for config loading and validation
│       ├── test_confidence.py                    # Unit tests for confidence logic
│       ├── test_context_builder.py               # Unit tests for context builder
│       ├── test_domain_interfaces.py             # Unit tests for domain interface contracts
│       ├── test_domain_models.py                 # Unit tests for domain models
│       ├── test_embeddings.py                    # Unit tests for embedding components
│       ├── test_evaluation_datasets.py           # Unit tests for eval dataset operations
│       ├── test_evaluation_judges.py             # Unit tests for judge adapters
│       ├── test_evaluation_metrics.py            # Unit tests for evaluation metrics
│       ├── test_evaluation_regression.py         # Unit tests for regression comparison logic
│       ├── test_evaluation_reports.py            # Unit tests for eval report generation
│       ├── test_evaluation_runner.py             # Unit tests for evaluation runner
│       ├── test_generation.py                    # Unit tests for generation logic
│       ├── test_ingestion_loaders.py             # Unit tests for ingestion loaders
│       ├── test_jsonl_store.py                   # Unit tests for JSONL artifact store
│       ├── test_manifest_store.py                # Unit tests for manifest store
│       ├── test_parsers.py                       # Unit tests for parser implementations
│       ├── test_reranking.py                     # Unit tests for reranking behavior
│       ├── test_retrieval.py                     # Unit tests for retrieval behavior
│       ├── test_storage.py                       # Unit tests for storage layer behavior
│       ├── test_utils.py                         # Unit tests for shared utilities
│       └── test_vector_store.py                  # Unit tests for vector store contracts
├── .dockerignore                                 # Exclusions for Docker build context
├── .gitignore                                    # Git exclusions
├── .python-version                               # Python version pin for local environment
├── Dockerfile                                    # Docker image definition
├── Makefile                                      # Commands for run, test, and pipeline workflows
├── Project structure.md                          # Project structure reference document
├── README.md                                     # Quick start and project overview
├── main.py                                       # FastAPI application entry point
├── pyproject.toml                                # Project metadata and tool configuration
└── requirements.txt                              # Pinned pip dependency list
```

