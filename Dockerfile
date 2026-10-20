# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src/ ./src/
RUN uv sync --locked --no-dev --no-editable

FROM python:3.11-slim-bookworm

ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid rag --home-dir /app --no-create-home \
        --shell /usr/sbin/nologin rag \
    && mkdir -p /app/.rag_data /app/.cache/huggingface \
    && chown -R rag:rag /app/.rag_data /app/.cache
COPY --from=builder /app/.venv /app/.venv

USER rag
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3).close()"]
CMD ["uvicorn", "rag_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
