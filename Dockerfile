FROM python:3.12-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG OLLAMA_BASE_URL=http://host.docker.internal:11434

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY app ./app
COPY main.py ./
COPY configs ./configs
COPY prompts ./prompts
RUN sed -i "s|base_url: \"http://localhost:11434\"|base_url: \"${OLLAMA_BASE_URL}\"|" /app/configs/config.yaml \
    && mkdir -p /app/data/indexes/qdrant

EXPOSE 8000
VOLUME ["/app/data/indexes/qdrant"]

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
