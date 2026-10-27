# RAG Pipeline

A local Python CLI and read-only HTTP API that turn business documents into
searchable evidence and source-cited answers using LangChain, Hugging Face
models, and Qdrant.

[![CI](https://github.com/Mojobra/rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Mojobra/rag-pipeline/actions/workflows/ci.yml)

## Overview

Retrieval-Augmented Generation (RAG) gives a language model relevant document
excerpts before it answers a question. This project implements that workflow
locally, from document extraction and indexing to retrieval, generation, and
evaluation.

It is intended for developers and AI engineers who want to inspect, test, and
compare RAG components on a reproducible baseline. The HTTP adapter requires a
local bearer token; it is not a hosted production service.

## Features

- **Document ingestion:** PDF, DOCX, Markdown, HTML, and plain text, with source
  metadata and page-level provenance for PDFs.
- **Retrieval:** dense semantic search, optional BM25 hybrid search with
  Reciprocal Rank Fusion, exact metadata filters, and cross-encoder reranking.
- **Grounded generation:** tokenizer-bounded prompts, deterministic source
  citations assembled outside the model, and abstention without usable evidence.
- **Persistent indexing:** embedded Qdrant, deterministic point IDs,
  source-scoped replacement, and collection compatibility checks.
- **Evaluation:** versioned synthetic corpora, positive/negative retrieval
  diagnostics, answer metrics, and isolated benchmarks with provenance,
  latency reports, regression gates, and paired prompt-evidence coverage.
- **Chunking experiments:** recursive chunk diagnostics plus opt-in
  structure-aware and semantic strategies in isolated benchmarks.
- **Local HTTP API:** bearer-protected retrieval and cited answers from an
  existing collection, optional source-scoped query tokens and separate local
  tenant stores, bounded JSON requests, request IDs, structured logs, and
  protected Prometheus metrics.
- **Local container image:** one runtime-only, non-root image for CLI indexing
  and the read-only API, with persistent data and model-cache volumes.
- **Manual SharePoint import:** bounded, read-only Microsoft Graph drive scan
  into a separate collection by default; it does not enforce source ACLs.
- **Manual Confluence import:** bounded, read-only Cloud space page scan into
  a separate collection by default; it does not enforce page ACLs.
- **Manual Google Drive import:** bounded, read-only folder scan of supported
  files and Google Docs into a separate collection; it does not enforce Drive ACLs.
- **Manual channel imports:** bounded, read-only Slack and Teams channel scans,
  including replies, into separate collections; message ACLs are not enforced.

## Quick Start

With [the prerequisites](#requirements) installed, run:

```powershell
git clone https://github.com/Mojobra/rag-pipeline.git
cd rag-pipeline
uv sync --locked
uv run rag index evaluation/datasets/asteria-policies-v2/documents --collection-name asteria_demo
uv run rag answer "When must expense claims be submitted?" --collection-name asteria_demo
```

This indexes five fictional business policies included in the repository, then
asks about their expense-submission deadline. No private documents or API keys
are needed. The first run downloads the embedding and generation models.

Answers include a `Sources` section identifying the evidence used. The small
default model is a functional baseline, not a guarantee of answer quality. Use
`retrieve` to inspect evidence independently of generation.

To query that collection over HTTP instead, start the local API in a second
terminal:

```powershell
$token = uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
$token
$env:RAG_API_BEARER_TOKEN = $token
$env:RAG_API_COLLECTION_NAME = 'asteria_demo'
uv run uvicorn rag_pipeline.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Open `http://127.0.0.1:8000/docs` or see the [API reference](docs/API_REFERENCE.md)
for authenticated query and `/metrics` examples. Keep the generated token private
and provide it to your local client; a new token requires a server restart. The
API writes one sanitized JSON event per request to stderr and returns
`X-Request-ID` for correlation; `--no-access-log` prevents Uvicorn's raw URL
access logs. Do not bind the service to a public network.

For the equivalent local container workflow, see [Docker setup](docs/DOCKER.md).

## Requirements

- Python **3.11 or newer**; CI verifies 3.11, 3.12, and 3.13.
- [uv](https://docs.astral.sh/uv/) for dependency management and execution.
- Git to clone the repository.
- Network access for initial dependency installation and uncached model
  downloads; disk space for dependencies, model caches, and the local index.

The defaults run on CPU. A GPU, separate Qdrant server, Docker, and hosted-model
API keys are not required for the Python workflow. Docker Engine is required
only for the optional container workflow. After dependencies and model artifacts
are cached, inference runs locally. HTTP queries require a locally generated
bearer token. The HTTP command uses bundled Uvicorn; no separate server install
is needed.

## Installation

The Quick Start installs from the repository using `uv sync --locked`. This
creates a `.venv` with the dependencies recorded in [uv.lock](uv.lock),
including the default development tools. Run subsequent commands from the
repository root.

Check the installation and available commands:

```powershell
uv run rag --version
uv run rag --help
```

`rag` is the short alias for `rag-pipeline`; both installed scripts and
`uv run python -m rag_pipeline` accept the same subcommands and flags.

## Usage

### Inspect Retrieved Evidence

Query the collection created in the Quick Start, without loading an answer
model:

```powershell
uv run rag-pipeline retrieve "When must expense claims be submitted?" --collection-name asteria_demo --top-k 3
```

Document inputs can be files or directories. Directories are scanned recursively
by default. Keep separate collection names for unrelated corpora, and use the
same collection and embedding settings for indexing and queries.

Re-running `index` replaces all chunks for each supplied file, including a file
that now produces no chunks. Other indexed sources remain untouched. To remove
sources that have disappeared from one directory, run `index` on that directory
with `--prune-missing`; this requires recursive scanning and deletes only points
whose source path is inside that directory. Index updates use multiple Qdrant
operations and are not atomic; after an interrupted run, repeat the command to
finish synchronization.

For a manual Microsoft Graph document-library import, see the
[SharePoint setup and security contract](docs/SHAREPOINT.md). This path needs an
Entra app and explicit acknowledgement that SharePoint permissions are **not**
enforced by retrieval.
For a Confluence Cloud space, see the
[Confluence setup and security contract](docs/CONFLUENCE.md). Its API-token
import has the same retrieval-permission limitation.
For a Google Drive folder, see the
[Drive setup and security contract](docs/GOOGLE_DRIVE.md). Its read-only ADC
import also does not carry source permissions into retrieval.
For a Slack or Teams channel, see the
[channel setup and security contract](docs/CHANNELS.md). These manual imports
include replies but do not preserve message permissions at retrieval time.

### Enable Hybrid Search And Reranking

Build a separate collection because hybrid search requires both dense and sparse
vectors. These multiline examples use PowerShell continuation syntax:

```powershell
uv run rag-pipeline index evaluation/datasets/asteria-policies-v2/documents `
  --collection-name asteria_hybrid --search-mode hybrid

uv run rag-pipeline retrieve "What does EXP-042 require?" `
  --collection-name asteria_hybrid --search-mode hybrid `
  --rerank --candidate-k 20 --top-k 4
```

Hybrid search adds a sparse model and index storage; reranking adds model
inference for each candidate. Neither is assumed to improve every corpus.

### Evaluate The Pipeline

Measure the existing demo collection against its committed retrieval labels:

```powershell
uv run rag-pipeline evaluate-retrieval evaluation/datasets/asteria-policies-v2/retrieval-v2.json --collection-name asteria_demo
```

For unsupported questions, archived-policy conflicts, and multi-document
evidence, use the separate [v3 retrieval stress set](evaluation/datasets/asteria-policies-v3/README.md).
Its negative cases report no-hit and false-positive rates separately from
positive-case Recall@k; it is not paired with answer labels.
The evaluator can also rerun v3 cases with explicit first-stage gates using
`--compare-score-threshold`. Run dense and hybrid modes separately; see the
[calibration procedure](docs/CLI_REFERENCE.md#score-gate-calibration).
To compare candidate-pool recall with final ranking at several widths, use
`--rerank --candidate-k 2 --compare-candidate-k 3 --compare-candidate-k 4`
with `--top-k 2` on v3 labels. See the
[candidate comparison procedure](docs/CLI_REFERENCE.md#candidate-and-reranking-comparison);
it requires a compatible collection and does not change search defaults.

For an isolated end-to-end benchmark, build a temporary index and save a report:

```powershell
uv run rag-pipeline benchmark `
  evaluation/datasets/asteria-policies-v2/documents `
  evaluation/datasets/asteria-policies-v2/retrieval-v2.json `
  evaluation/datasets/asteria-policies-v2/answers-v2.json `
  --output .rag_data/benchmarks/asteria-v2.json
```

Use a new output filename for each run; existing artifacts are preserved unless
`--overwrite` is explicit. Benchmark reports contain queries and generated
answers, so reports from private data need the same protection as their sources.

### Command Reference

| Command | Purpose |
| --- | --- |
| `ingest` | Discover supported files and inspect extracted documents. |
| `chunk` | Inspect recursive chunks and their source metadata. |
| `chunk-experiment` | Compare chunk sizes and overlap costs without model inference. |
| `embed` | Inspect dense-vector output without writing an index. |
| `index` | Build or update a compatible dense or hybrid collection. |
| `index-sharepoint` | Manually import supported files from one Graph drive; no ACL sync. |
| `index-confluence` | Manually import current pages from one Cloud space; no ACL sync. |
| `index-google-drive` | Manually import supported files and Google Docs from one Drive folder; no ACL sync. |
| `index-slack` | Manually import one bot-visible Slack channel and replies; no ACL sync. |
| `index-teams` | Manually import one Graph-visible Teams channel and replies; no ACL sync. |
| `retrieve` | Inspect ranked evidence, optionally filtered or reranked. |
| `answer` | Generate an answer and citations from retrieved evidence. |
| `evaluate-retrieval` | Score an existing collection against relevance labels. |
| `evaluate-answer` | Measure reference answers, abstention, and citation behavior. |
| `benchmark` | Rebuild and measure an isolated full pipeline. |
| `compare-benchmarks` | Compare compatible saved artifacts and regression gates. |

Every command exposes its arguments through `--help`, for example
`uv run rag-pipeline index --help`. See the
[CLI reference](docs/CLI_REFERENCE.md) for schemas, filter semantics, score
interpretation, and benchmark details, and the
[dataset guide](evaluation/datasets/asteria-policies-v2/README.md) for chunking
comparisons.

## Configuration

Configuration uses CLI flags and typed Python configuration objects. The HTTP
adapter reads `RAG_API_*` environment variables at startup, including the
required `RAG_API_BEARER_TOKEN`. Copy [.env.example](.env.example) to a private
`.env` as a configuration template, then export the needed values in your shell
or pass the file to a launcher that supports environment files: the pipeline
does not load `.env` automatically. Hosted OpenAI, Gemini, and Anthropic model
integration is not active; the template contains no hosted-provider settings.
The optional SharePoint importer reads three `RAG_SHAREPOINT_*` environment
variables, documented in [its setup guide](docs/SHAREPOINT.md).
The optional Confluence importer reads three `RAG_CONFLUENCE_*` variables;
see [its setup guide](docs/CONFLUENCE.md).
The Google Drive importer uses Application Default Credentials. An optional
`GOOGLE_APPLICATION_CREDENTIALS` path selects a private credential file;
see [its setup guide](docs/GOOGLE_DRIVE.md).
The Slack importer reads `RAG_SLACK_BOT_TOKEN`; the Teams importer reads three
`RAG_TEAMS_*` Entra app credentials. See [channel setup](docs/CHANNELS.md).

| Setting | Default | Practical effect |
| --- | --- | --- |
| `--model` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model, not the answer model. Index and query with the same model and revision. |
| `--generation-model` | `google/flan-t5-small` | Local text-to-text answer model; changing it does not change embeddings. |
| `--device` / `--generation-device` | `cpu` | Independent embedding and generation devices. CUDA requires compatible hardware and sufficient VRAM. |
| `--chunk-size` / `--chunk-overlap` | `1000` / `200` characters | Balance evidence size against chunk count, duplicated text, and storage. |
| `--store-path` | `.rag_data/qdrant` | Persistent embedded database directory; reuse it for subsequent queries. |
| `--collection-name` | `rag_documents` (remote import defaults: `sharepoint_documents`, `confluence_documents`, `google_drive_documents`, `slack_messages`, `teams_messages`) | Collection within the database; examples use explicit names to separate corpora. |
| `--search-mode` | `dense` | Select `hybrid` only for a collection indexed in that mode. |
| `--top-k` | `4` | Maximum final evidence chunks per query. |
| `--score-threshold` / `--no-score-threshold` | Disabled for `retrieve` and `evaluate-retrieval`; `0.2` for `answer`, `evaluate-answer`, and `benchmark` | Apply or explicitly disable the first-stage gate before reranking or generation. Scores are mode-specific, not confidence probabilities. |
| `--batch-size` / `--write-batch-size` | `32` / `64` chunks | Separate dense-inference and database-write batches; larger batches use more memory. |

Changing the embedding model, revision, or collection schema requires a new or
rebuilt compatible collection. Pin `--model-revision`,
`--generation-model-revision`, and, when used, `--reranker-model-revision` for
more reproducible comparisons.

Local Qdrant data, FastEmbed and reranker caches under `.rag_data/`, and
`.env` files are ignored by Git. Dense and generation models use Hugging Face
caching. See the [technical manual](TECHNICAL_MANUAL.md#local-state-and-io) for
storage locations and I/O behavior.

## Project Structure

```text
.
|-- src/rag_pipeline/
|   |-- cli/                 # Argument parsing and terminal output
|   |-- api/                 # HTTP auth, routes, logging, and monitoring
|   |-- application/         # Transport-neutral workflow orchestration
|   |-- ingestion/           # Discovery, extraction, and chunking
|   |-- infrastructure/      # Model, remote-source, and Qdrant adapters
|   |-- retrieval/           # Search, filtering, and reranking
|   |-- generation/          # Prompts, token budgets, and citations
|   |-- evaluation/          # Dataset schemas and quality metrics
|   `-- benchmarking/        # Isolated runs, provenance, and comparisons
|-- tests/                   # Offline unit and integration tests
|-- scripts/                 # Release-tag verification for CI
|-- evaluation/datasets/     # Versioned synthetic corpora and labels
|-- docs/CLI_REFERENCE.md    # Detailed CLI workflows and contracts
|-- docs/API_REFERENCE.md    # Local HTTP setup and endpoint contract
|-- docs/DOCKER.md           # Local container build and run workflow
|-- Dockerfile               # Runtime image for CLI and API
|-- .dockerignore            # Build-context allowlist
|-- .github/workflows/ci.yml # Quality, container checks, release image publishing
|-- pyproject.toml           # Package metadata and dependency constraints
`-- uv.lock                  # Reproducible dependency graph
```

The CLI delegates to application and feature services; model and storage I/O
remain behind explicit adapters. See [ARCHITECTURE.md](ARCHITECTURE.md) for
package boundaries and the contracts enforced between stages.

## Tech Stack

| Area | Technology |
| --- | --- |
| Language and CLI | Python 3.11+, standard-library `argparse` |
| HTTP API and monitoring | FastAPI, Uvicorn, `prometheus-client` |
| RAG interfaces and composition | LangChain and its Hugging Face/Qdrant integrations |
| Dense embeddings and reranking | Sentence Transformers / PyTorch |
| Local generation | Hugging Face Transformers |
| SharePoint import | MSAL and HTTPX against Microsoft Graph |
| Confluence import | HTTPX against Confluence Cloud REST API v2 |
| Google Drive import | Google Auth ADC and HTTPX against Drive REST API v3 |
| Channel imports | HTTPX against Slack Web API and MSAL/Graph for Teams |
| Sparse embeddings | FastEmbed BM25 |
| Vector storage | Embedded Qdrant via `qdrant-client` |
| PDF / DOCX extraction | pypdf / docx2txt |
| Packaging and quality | uv, setuptools, unittest, Coverage.py, Ruff, mypy, GitHub Actions |
| Local container packaging | Docker with a Python 3.11 runtime image |

The [technical manual](TECHNICAL_MANUAL.md#library-inventory) contains the full
application-library inventory, locked version snapshots, model defaults, and
operational trade-offs.

## Testing

Install the development environment and run the CI checks:

```powershell
uv sync --locked --dev
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run coverage run -m unittest discover -s tests
uv run coverage report
uv build
```

Tests use provider doubles and temporary stores: no model downloads, API keys,
network access, or GPU are needed for the default suite. Coverage includes
branches and enforces an 80 percent floor. The
[CI workflow](.github/workflows/ci.yml) runs the Python checks on pull
requests, pushes to `main`, and version tags. Pull requests also build and
smoke-test the container. A matching annotated `vMAJOR.MINOR.PATCH` tag on a
commit reachable from `main` publishes a versioned image to GitHub Container
Registry after those checks pass. See [Docker setup](docs/DOCKER.md#release-images)
for registry access and limitations. No service is deployed automatically.

## Limitations

- FLAN-T5 Small prioritizes local accessibility, not production answer quality.
  Citations identify supplied evidence; they do not prove claim-level support.
- Evaluation uses a small synthetic English corpus and deterministic lexical
  metrics. It does not establish semantic faithfulness or real-world accuracy.
- Benchmarks measure one local pass, not concurrent service throughput or
  statistically calibrated performance.
- HTML extraction keeps readable text and table rows while omitting scripts,
  styles, and navigation. CSS-dependent visibility and image text are not
  interpreted; scanned PDFs still require OCR that is not implemented.
- CLI metadata filters are not authorization. The HTTP service defaults to one
  shared bearer credential; its optional manual policy scopes query tokens to
  exact sources and, in policy v2, separate local tenant stores. This is not
  SSO, remote ACL sync, OS-level tenant isolation, or a multi-user database.
  There is no rate limiting, hosted-model integration, upload endpoint, or
  production deployment.
  Citations can expose local source paths or remote source IDs.
- SharePoint import has no per-user ACL enforcement, deletion sync, or scheduled
  refresh; its default collection is separate by name only, not by Qdrant file.
- Confluence import has the same ACL and deletion limits; complex Storage-HTML
  macros and non-page content are not fully represented.
- Google Drive import has the same ACL and deletion limits; only supported blob
  file types and native Google Docs are imported, not Sheets or Slides.
- Slack and Teams imports lack message ACL enforcement, deletion sync, and
  automatic refresh. They read channel text and replies, not DMs/chats or
  attachments; provider throttling aborts rather than retrying.
- Semantic chunking is benchmark-only and needs corpus-specific validation
  before use beyond experiments.

## Roadmap

The local pipeline, retrieval-quality follow-ups, evaluation, and benchmarking
are implemented. See the [completed quality plan](docs/RETRIEVAL_QUALITY_PLAN.md)
for the six follow-ups. Phase 4 now includes the local FastAPI service, shared
bearer guard, structured request logging, process-local monitoring metrics, and
a local Docker image, and release-tag image publishing CI. Hosting and
operating a production service are not implemented.

Bounded manual SharePoint drive, Confluence Cloud space, Google Drive folder,
Slack channel, and Teams channel imports are implemented. An opt-in local API
policy now restricts retrieval by exact source and caller token; policy v2
adds local tenant-separated stores. It does not sync remote ACLs. See the
[API access policy guide](docs/API_REFERENCE.md#optional-source-access-policy),
[tenant isolation guide](docs/API_REFERENCE.md#optional-local-tenant-isolation), and
[ROADMAP.md](ROADMAP.md).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup and quality requirements.
Start task branches from `develop` and open PRs into `develop`; promotions to
`main` require separate approval. Existing `release/rag-v<version>` snapshots
are immutable. Release history is documented in [CHANGELOG.md](CHANGELOG.md).

### Development Approach

This project was developed using AI-assisted software engineering with Codex. I
used AI as a development partner for implementation support while defining the
architecture, requirements, tests, and design decisions myself. All generated
code was reviewed, adapted, and validated through automated tests.

## License

This repository does not currently include a license file.
