# Local HTTP API

Tasks 19-22 add a read-only FastAPI adapter with shared-token authentication,
structured request logging, and process-local monitoring over the existing
retrieval and answer workflows. It does not ingest documents, host remote
models, identify individual users, enforce document permissions, or make
embedded Qdrant a multi-user database. Index documents with the CLI first;
stop the API before modifying the same local Qdrant store with the CLI.

## Start Locally

From the repository root, after `uv sync --locked` and indexing a collection:

```powershell
$token = uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
$token
$env:RAG_API_BEARER_TOKEN = $token
$env:RAG_API_COLLECTION_NAME = 'asteria_demo'
uv run uvicorn rag_pipeline.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

The example collection name matches the README Quick Start. Without that
variable, the API uses the CLI's default `rag_documents` collection. Visit
`http://127.0.0.1:8000/docs` for the generated OpenAPI explorer. Bind only to
loopback: a shared bearer token is not user authorization or a substitute for
TLS and rate limiting on a network. Use one worker; the
local models and embedded Qdrant session are shared within one process, and
queries are serialized. The first query may download or initialize a model.

## Server Configuration

These variables are read when the ASGI application starts. They do not load a
`.env` file automatically. Match the indexed collection's embedding and search
settings; incompatible settings fail rather than silently mixing vectors.

| Variable | Default | Effect |
| --- | --- | --- |
| `RAG_API_BEARER_TOKEN` | required | Shared 32-256-character URL-safe secret for query routes. Generate at least 32 random bytes; missing or malformed values prevent startup. Rotation requires restart. |
| `RAG_API_STORE_PATH` | `.rag_data/qdrant` | Local embedded-Qdrant directory; use the same path as CLI `--store-path`. |
| `RAG_API_COLLECTION_NAME` | `rag_documents` | Existing collection; use the CLI `--collection-name` value. |
| `RAG_API_SEARCH_MODE` | `dense` | `dense` or `hybrid`; must match the collection schema. |
| `RAG_API_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Dense query model; must match the model used to index. |
| `RAG_API_EMBEDDING_REVISION` | unset | Optional matching Hugging Face model revision. |
| `RAG_API_SPARSE_MODEL` | `Qdrant/bm25` | Sparse query model, used only in `hybrid` mode. |
| `RAG_API_GENERATION_MODEL` | `google/flan-t5-small` | Local text-to-text answer model, loaded only for an answer with evidence. |
| `RAG_API_GENERATION_REVISION` | unset | Optional generation model revision. |

The API currently uses CPU defaults, no reranker, and the same 1,200-character
generation context policy as the CLI default. Other CLI tuning flags are not
HTTP configuration in this first adapter.

## Request Logging

The API writes one JSON object per HTTP request to stderr with a UTC timestamp,
server-generated request ID, method, available route template, HTTP status,
and elapsed milliseconds. Handled responses include the same ID in
`X-Request-ID`; a caller-supplied ID is ignored. Unknown URLs and framework
routes without an exposed template are recorded as `<unmatched>`. Bodies,
credentials, query strings, raw paths, response text, and source metadata are
not included in these application events.

Use `--no-access-log` as shown above: Uvicorn's separate access log can include
raw URLs and query strings. Uvicorn error logs and third-party libraries have
separate logging behavior; review their configuration before processing sensitive
data. These events support local diagnosis, not tamper-resistant audit trails,
per-user attribution, or centralized retention. Aggregated monitoring is
exposed separately at `/metrics`.

## Monitoring

`GET /metrics` requires the same bearer token as the query routes and returns
Prometheus text exposition. For example, from a second PowerShell terminal:

```powershell
$token = '<same token configured on the server>'
Invoke-WebRequest -Uri http://127.0.0.1:8000/metrics `
  -Headers @{ Authorization = "Bearer $token" } | Select-Object -ExpandProperty Content
```

The per-process registry exposes these series:

| Metric | Meaning |
| --- | --- |
| `rag_api_http_requests_total` | Completed HTTP requests by method, route template, and status code. |
| `rag_api_http_request_duration_seconds` | Histogram of full HTTP request duration by method and route template. |
| `rag_api_http_in_flight_requests` | Requests currently being processed. |
| `rag_api_retrieval_outcomes_total` | Successful retrievals with `empty` or `nonempty` selected evidence. |
| `rag_api_answer_outcomes_total` | Completed answers: `no_evidence` (no model call), `cited`, or model `abstained`. |

`/metrics` requests themselves are excluded from these counters and timings;
the normal request log still records them. Unknown paths use `<unmatched>` and
uncommon HTTP methods use `OTHER`, preventing caller-controlled labels. No
query text, credentials, document paths, model identifiers, or request IDs are
metric labels. Values reset on process restart. There is no multi-worker
aggregation, remote collector, alerting, dashboard, or guarantee that a cited
answer is correct. Keep the service on loopback; the bearer token is sent over
plain HTTP in this local setup.

## Endpoints

`POST /v1/retrieve`, `POST /v1/answer`, and `GET /metrics` require
`Authorization: Bearer <token>`. Copy the generated value securely into a
second terminal or use the OpenAPI explorer's **Authorize** control. The
OpenAPI document and `/health` remain public and reveal no indexed text.

### `GET /health`

Returns `{"status":"ok"}` without opening Qdrant or loading a model. It is a
process-liveness check, not proof that a collection exists or a model can run.

### `POST /v1/retrieve`

Example request:

```json
{"query":"When must expense claims be submitted?","top_k":4}
```

Returns `results` in ranked order, with score, score kind, source text, and
selected provenance fields (`source`, one-based `page_number`, `chunk_index`,
`chunk_id`). An empty list means no evidence passed retrieval. Raw Qdrant
metadata is not returned. `score_threshold` defaults to `null` (disabled).

### `POST /v1/answer`

Uses the same request fields. Its `score_threshold` defaults to `0.2`, matching
the CLI answer command; send JSON `null` explicitly to disable the gate. The
response includes `answer`, `generated`, model/prompt identifiers when a model
was initialized, and structured `citations`. No qualifying evidence returns
the deterministic abstention with `generated=false` and no citations, without
loading the generation model. A model may also abstain after invocation; that
has `generated=true` and no citations. Citations identify supplied evidence,
not claim-level correctness.

Both POST routes require a nonblank query of at most 2,000 characters, allow
`top_k` from 1 to 20, and accept a finite score threshold from -1 to 1 or
`null`. Unknown request fields are rejected, so clients cannot choose server
paths or arbitrary model names per request. For example, in a second PowerShell
terminal:

```powershell
$token = '<same token configured on the server>'
$headers = @{ Authorization = "Bearer $token" }
$body = @{ query = 'When must expense claims be submitted?' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/answer `
  -Headers $headers -ContentType 'application/json' -Body $body
```

## Failures And Limits

- `401`: missing or invalid bearer token; no retrieval or generation occurs.
- `422`: invalid request fields or query input.
- `404`: collection not indexed.
- `409`: collection and configured model/search settings are incompatible.
- `503`: local provider or storage unavailable, with provider details hidden.
- `500`: unexpected server failure.

Responses can include local source paths and document text. Keep this service
on a trusted machine: the shared token does not identify a caller or restrict
which documents they can read. It can be copied or stolen and is not a
replacement for TLS, per-user identity, permission-aware retrieval, per-client
rotation and revocation, rate limiting, or audit logging. Do not put it in a URL,
Git, or application logs. There is no upload, HTTP indexing, CORS policy,
distributed worker, streaming, or remote-Qdrant support.
