# Local Docker Workflow

Task 23 packages the existing CLI and read-only API in one Linux image. The
container does not ingest over HTTP, run a remote Qdrant server, or make the API
safe for public exposure. Build the image from the repository root with Docker
Engine running:

```powershell
docker build -t rag-pipeline:local .
docker run --rm rag-pipeline:local rag-pipeline --version
```

The build uses Python 3.11, the committed `uv.lock`, and runtime dependencies
only. It does not download embedding or generation models. The ML dependencies
make the image large; first-use model downloads require network access and may
take time. The final image contains the installed package, not the source tree,
development tools, local documents, `.env` files, or existing Qdrant data.

## Index A Collection

Create two named volumes. The first retains embedded Qdrant and optional
FastEmbed/reranker caches under `/app/.rag_data`; the second retains Hugging Face
model files under `/app/.cache`. Keep both across container restarts:

```powershell
docker volume create rag_data
docker volume create rag_models
$docs = (Resolve-Path 'evaluation/datasets/asteria-policies-v2/documents').Path
docker run --rm `
  --no-healthcheck `
  --mount 'type=volume,source=rag_data,target=/app/.rag_data' `
  --mount 'type=volume,source=rag_models,target=/app/.cache' `
  --mount "type=bind,source=$docs,target=/documents,readonly" `
  rag-pipeline:local rag-pipeline index /documents --collection-name asteria_demo
```

The bound directory is read-only. Qdrant stores absolute container source paths
such as `/documents/expense-reimbursement.md`; mount the same directory at `/documents`
for later indexing runs so source replacement remains stable. Do not run an
indexer and the API against the same embedded Qdrant volume concurrently.
They each need exclusive local access. Do not share a volume with multiple API
replicas or workers.

## Start The API

Stop the indexing container before starting the API. Generate a local token in
PowerShell and pass its environment variable to Docker without putting the
secret value in the command line:

```powershell
$env:RAG_API_BEARER_TOKEN = docker run --rm rag-pipeline:local python -c "import secrets; print(secrets.token_urlsafe(32))"
$docs = (Resolve-Path 'evaluation/datasets/asteria-policies-v2/documents').Path
docker run --rm `
  --publish 127.0.0.1:8000:8000 `
  --mount 'type=volume,source=rag_data,target=/app/.rag_data' `
  --mount 'type=volume,source=rag_models,target=/app/.cache' `
  --mount "type=bind,source=$docs,target=/documents,readonly" `
  --env RAG_API_BEARER_TOKEN `
  --env RAG_API_COLLECTION_NAME=asteria_demo `
  rag-pipeline:local
```

The API is available at `http://127.0.0.1:8000/docs`; `GET /health` is a
liveness-only check, also used by Docker's health check. The first authenticated
query may load or download the models. If port 8000 is occupied, change only the
host side of `--publish`, for example `127.0.0.1:8001:8000`, then use port 8001
in the browser. See [API reference](API_REFERENCE.md) for request examples and
all `RAG_API_*` settings. The API uses one process and CPU defaults.

The raw token remains in the host environment and Docker container metadata;
anyone with Docker daemon access can inspect it. Keep this a trusted local demo,
do not publish port 8000 on all interfaces, and do not commit tokens or private
documents. The image has no TLS, per-user identity, authorization, rate
limiting, secret manager, backups, or deployment automation. Persistent volumes
and read-only source mounts do not replace those controls.
