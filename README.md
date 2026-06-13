# stripe-rag-assistant

Local production-like RAG assistant over Stripe documentation, using local embeddings, local Qdrant, and local Ollama generation.

## Overview

`stripe-rag-assistant` answers questions against a local corpus of Stripe docs and guides.

The system builds a searchable index from raw source files, retrieves and reranks relevant chunks, and generates grounded answers with source attribution.

It also includes layer-wise evaluation so you can track retrieval, rerank, context, generation, citation, and robustness quality over time.

## What this project does

```text
raw Stripe docs
→ parsing/cleaning
→ chunking
→ embeddings
→ Qdrant indexing
→ retrieval/reranking
→ context building
→ Ollama generation
→ grounded answer with sources
→ eval/reporting
```

The project supports:

- offline indexing;
- online query pipeline;
- FastAPI service;
- local CLI/smoke scripts;
- eval datasets, runs, and reports.

## Architecture at a glance

Offline pipeline:

```text
data/raw
→ parsed docs
→ cleaned docs
→ chunks
→ embeddings
→ Qdrant vector index
→ manifests
```

Online pipeline:

```text
question
→ RetrieveLayer
→ RerankLayer
→ BuildContextLayer
→ GenerateAnswerLayer
→ answer + sources + confidence
```

Eval pipeline:

```text
eval dataset
→ eval runner
→ retrieval/rerank/context/generation/citation metrics
→ reports
→ regression comparison
```

Eval layer boundaries:

- `app/evaluation/` = core eval logic and metrics.
- `app/application_layers/eval/` = runnable eval layer wrappers.
- `app/application/eval_service.py` = full eval orchestration across suites.

## Repository structure

```text
app/
  api/
  application/
  application_layers/
  domain/
  infrastructure/
  evaluation/
  schemas/
  utils/

configs/
prompts/
data/
scripts/
tests/
docs/
```

- `app/api/`: FastAPI routes and HTTP adapter wiring.
- `app/application/`: end-to-end orchestration services.
- `app/application_layers/`: executable layer entrypoints (offline/online/eval).
- `app/domain/`: domain models, interfaces, and policies.
- `app/infrastructure/`: concrete implementations (Qdrant, Ollama, storage, etc.).
- `app/evaluation/`: core evaluation, reports, and regression helpers.
- `app/schemas/`: external request/response schemas.
- `app/utils/`: shared config/logging/errors/timing utilities.
- `configs/`: runtime settings (`config.yaml`).
- `prompts/`: generation prompt templates.
- `data/`: local artifacts (raw/interim/processed/index/eval/traces).
- `scripts/`: lightweight smoke wrappers.
- `tests/`: unit/integration/e2e coverage.
- `docs/`: deeper design and operations docs.

## Prerequisites

- Python 3.12.
- Ollama installed locally and running.
- Required Ollama model pulled locally.
- Qdrant available locally (default config uses embedded local mode; Docker mode is optional).

Default model from current config:

```bash
ollama pull llama3.1:8b
```

## Setup

From the repository root:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Optional Docker flow for the API:

```bash
make docker-build
make docker-up
make docker-health
```

Use Docker logs and shutdown commands when needed:

```bash
make docker-logs
make docker-down
```

Notes:

- `make docker-up` exposes API on `http://localhost:8000`.
- Ollama must still run on your host machine (`ollama serve`), reachable as `host.docker.internal:11434` from the container.
- The container mounts `data/indexes/qdrant` so local index data persists across restarts.
- Override Ollama endpoint at build time when needed:

```bash
make docker-build DOCKER_OLLAMA_BASE_URL="http://host.docker.internal:11434"
```

Configuration is loaded from `configs/config.yaml`; edit that file directly for local settings.

## Configuration

`configs/config.yaml` is the main runtime config file.

Key areas to adjust:

- `paths`: artifact directories;
- `embeddings`: model, batching, caching;
- `vector_store`: Qdrant mode/collection/connectivity;
- `retrieval`: strategy and top-k behavior;
- `reranking`: model and post-retrieval ranking behavior;
- `context`: token budget and context assembly;
- `generation`: Ollama URL/model and decoding settings;
- `eval`: dataset/run directories, defaults, preflight;
- `app`/`api`: environment, logging, API runtime behavior.

## Offline indexing pipeline

Run each offline layer explicitly:

```bash
python -m app.application_layers.offline.build_parsed_docs --config configs/config.yaml
python -m app.application_layers.offline.build_cleaned_docs --config configs/config.yaml
python -m app.application_layers.offline.build_chunks --config configs/config.yaml
python -m app.application_layers.offline.build_embeddings --config configs/config.yaml
python -m app.application_layers.offline.build_vector_index --config configs/config.yaml
```

Equivalent `make` targets:

```bash
make offline-parse
make offline-clean
make offline-chunk
make offline-embed
make offline-index
```

One-command orchestration:

```bash
python -m app.application.indexing_service --config configs/config.yaml
# or
make service-index
```

## Online query pipeline

Layer-by-layer online flow:

```bash
python -m app.application_layers.online.retrieve --question "How do PaymentIntents work?" --config configs/config.yaml
python -m app.application_layers.online.rerank --question "How do PaymentIntents work?" --input-path data/traces/queries/<retrieve-output>.json --config configs/config.yaml
python -m app.application_layers.online.build_context --question "How do PaymentIntents work?" --input-path data/traces/queries/<rerank-output>.json --config configs/config.yaml
python -m app.application_layers.online.generate_answer --question "How do PaymentIntents work?" --input-path data/traces/queries/<context-output>.json --config configs/config.yaml
```

Smoke wrappers:

```bash
python scripts/online/smoke_retrieve.py --question "How do PaymentIntents work?"
python scripts/online/smoke_rerank.py --question "How do PaymentIntents work?" --input-path <retrieve-output.json>
python scripts/online/smoke_build_context.py --question "How do PaymentIntents work?" --input-path <rerank-output.json>
python scripts/online/smoke_generate_answer.py --question "How do PaymentIntents work?" --input-path <context-output.json>
```

Full orchestration:

```bash
python -m app.application.query_service --question "How do PaymentIntents work?" --config configs/config.yaml
# or
make service-query Q="How do PaymentIntents work?"
```

## API usage

Start API locally:

```bash
uvicorn main:app --reload
# or
make api
```

Health check:

```bash
curl http://localhost:8000/health
```

Query endpoint:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How do PaymentIntents work?"}'
```

## Evaluation

Eval is layer-wise and local-first. Typical workflow:

- build eval dataset;
- run retrieval eval;
- run context eval;
- run generation/full eval;
- compare runs;
- inspect run artifacts.

Build dataset:

```bash
python -m app.application_layers.eval.build_eval_dataset \
  --chunks data/processed/chunks.jsonl \
  --output-dir data/eval/datasets \
  --dataset-id stripe_synthetic_v1 \
  --config configs/config.yaml
```

Run individual suites:

```bash
python -m app.application_layers.eval.run_retrieval_eval \
  --dataset data/eval/datasets/stripe_synthetic_v1/dataset.jsonl \
  --run-id retrieval_v1 \
  --config configs/config.yaml

python -m app.application_layers.eval.run_context_eval \
  --dataset data/eval/datasets/stripe_synthetic_v1/dataset.jsonl \
  --run-id context_v1 \
  --config configs/config.yaml

python -m app.application_layers.eval.run_generation_eval \
  --dataset data/eval/datasets/stripe_synthetic_v1/dataset.jsonl \
  --run-id generation_v1 \
  --config configs/config.yaml
```

Run full eval orchestration:

```bash
python -m app.application.eval_service \
  --dataset data/eval/datasets/stripe_synthetic_v1/dataset.jsonl \
  --run-id-prefix full_eval_v1 \
  --config configs/config.yaml
# or
python scripts/eval/smoke_run_eval.py --dataset data/eval/datasets/stripe_synthetic_v1/dataset.jsonl --run-id-prefix full_eval_v1
```

Metric groups tracked in reports include:

- retrieval: hit/recall/precision/MRR/nDCG families;
- rerank: rank delta, MRR delta, expected-source-kept behavior;
- context: expected source recall/drop and budget-related behavior;
- generation: output validity/no-answer/reference overlap proxies;
- citation: valid/invented source behavior and citation recall/precision;
- confidence/robustness/latency summaries.

Without human labels, eval cannot prove absolute answer correctness; it measures grounding, source behavior, robustness, regressions, and proxy quality signals.

## Artifacts

Main local artifact paths:

- `data/raw/`: raw source docs.
- `data/interim/`: parsed and cleaned intermediates.
- `data/processed/`: chunks and embedded chunks.
- `data/indexes/`: local indexes/caches (including Qdrant local path in embedded mode).
- `data/manifests/`: per-stage manifests.
- `data/eval/datasets/`: generated eval datasets.
- `data/eval/runs/`: eval run outputs (manifest/cases/metrics/report).
- `data/eval/baselines/`: baseline snapshots/comparisons.

Artifacts are local/generated and may be gitignored depending on type.

## Testing

Run full test suite:

```bash
pytest
# or
make test
```

Targeted test runs:

```bash
pytest tests/unit
pytest tests/integration
pytest tests/unit/test_evaluation_metrics.py
pytest tests/integration/test_eval_pipeline.py
```

## Common operations

### Rebuild the index

```bash
make offline-parse && make offline-clean && make offline-chunk && make offline-embed && make offline-index
# or
make service-index
```

### Run a local query

```bash
make service-query Q="How do PaymentIntents work?"
```

### Run full eval

```bash
python scripts/eval/smoke_run_eval.py \
  --dataset data/eval/datasets/stripe_synthetic_v1/dataset.jsonl \
  --run-id-prefix full_eval_v1
```

### Compare eval runs

```bash
python -c "from app.evaluation import compare_eval_runs; print(compare_eval_runs(baseline_run_dir='data/eval/runs/baseline_run', candidate_run_dir='data/eval/runs/candidate_run').summary)"
```

### Inspect failed eval cases

```bash
python -c "from app.evaluation import load_eval_run_artifacts; print(load_eval_run_artifacts('data/eval/runs/some_run').get('failures', [])[:3])"
```

## Troubleshooting

### Qdrant is not running

- Check `configs/config.yaml` under `vector_store`.
- If using Docker mode, run `make qdrant-start` and verify with `make qdrant-health`.
- If using embedded mode, ensure `data/indexes/qdrant` is writable.

### Ollama model is missing

- Ensure Ollama is running (`ollama serve`).
- Pull the configured model (`ollama pull llama3.1:8b` by default).
- Verify with `make ollama-check`.

### Retrieval returns no results

- Confirm offline index steps completed (`make offline-index`).
- Check collection/index status (`curl http://localhost:8000/index/status`).
- Verify `retrieval.dense_top_k` and filters in config/request.

### Eval dataset is empty

- Confirm `data/processed/chunks.jsonl` exists and is non-empty.
- Rebuild dataset with `build_eval_dataset` and inspect `dataset.jsonl`.
- Remove overly strict subset/type/difficulty filters.

### Generation returns malformed JSON

- Check `generation.model_name`, timeout, and prompt template settings.
- Try lower temperature (already low by default) and rerun.
- Inspect traces under `data/traces/queries`.

### API starts but `/query` fails

- Ensure index exists and is accessible.
- Ensure Ollama is reachable at configured `generation.base_url`.
- Check API logs for warmup/query errors and validate request body includes `question`.

## Documentation

- `[docs/Design Document.md](docs/Design%20Document.md)`
- `[docs/Architecture.md](docs/Architecture.md)`
- `[docs/Eval methodology.md](docs/Eval%20methodology.md)`
- `[docs/Operations.md](docs/Operations.md)`
- `[Project structure.md](Project%20structure.md)`

## Limitations

- Local-only system; no hosted LLM, hosted embeddings, or hosted vector DB.
- Answer quality depends on source corpus quality and index freshness.
- Synthetic and heuristic eval signals are proxies, not human-labeled truth.
- Ollama model choice and local resources strongly affect output quality/latency.
- This is not an official Stripe product.

