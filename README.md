# RAG Pipeline

A local Python CLI that turns business documents into searchable evidence and
source-cited answers using LangChain, Hugging Face models, and Qdrant.

[![CI](https://github.com/Mojobra/rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Mojobra/rag-pipeline/actions/workflows/ci.yml)

## Overview

Retrieval-Augmented Generation (RAG) gives a language model relevant document
excerpts before it answers a question. This project implements that workflow
locally, from document extraction and indexing to retrieval, generation, and
evaluation.

It is intended for developers and AI engineers who want to inspect, test, and
compare RAG components on a reproducible baseline. The current interface is a
CLI, not a hosted application or production service.

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
  latency reports, and regression gates.
- **Chunking experiments:** recursive chunk diagnostics plus opt-in
  structure-aware and semantic strategies in isolated benchmarks.

## Quick Start

With [the prerequisites](#requirements) installed, run:

```powershell
git clone https://github.com/Mojobra/rag-pipeline.git
cd rag-pipeline
uv sync --locked
uv run rag-pipeline index evaluation/datasets/asteria-policies-v2/documents --collection-name asteria_demo
uv run rag-pipeline answer "When must expense claims be submitted?" --collection-name asteria_demo
```

This indexes five fictional business policies included in the repository, then
asks about their expense-submission deadline. No private documents or API keys
are needed. The first run downloads the embedding and generation models.

Answers include a `Sources` section identifying the evidence used. The small
default model is a functional baseline, not a guarantee of answer quality. Use
`retrieve` to inspect evidence independently of generation.

## Requirements

- Python **3.11 or newer**; CI currently runs Python 3.11.
- [uv](https://docs.astral.sh/uv/) for dependency management and execution.
- Git to clone the repository.
- Network access for initial dependency installation and uncached model
  downloads; disk space for dependencies, model caches, and the local index.

The defaults run on CPU. A GPU, separate Qdrant server, Docker, and API keys are
not required. After dependencies and model artifacts are cached, inference runs
locally.

## Installation

The Quick Start installs from the repository using `uv sync --locked`. This
creates a `.venv` with the dependencies recorded in [uv.lock](uv.lock),
including the default development tools. Run subsequent commands from the
repository root.

Check the installation and available commands:

```powershell
uv run rag-pipeline --version
uv run rag-pipeline --help
```

`uv run python -m rag_pipeline` is the equivalent module entry point; it accepts
the same subcommands and flags.

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

Configuration uses CLI flags and typed Python configuration objects. The current
pipeline does **not** load model settings or API keys from `.env`; no
`.env.example` or active OpenAI, Gemini, or Anthropic integration is included.

| Setting | Default | Practical effect |
| --- | --- | --- |
| `--model` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model, not the answer model. Index and query with the same model and revision. |
| `--generation-model` | `google/flan-t5-small` | Local text-to-text answer model; changing it does not change embeddings. |
| `--device` / `--generation-device` | `cpu` | Independent embedding and generation devices. CUDA requires compatible hardware and sufficient VRAM. |
| `--chunk-size` / `--chunk-overlap` | `1000` / `200` characters | Balance evidence size against chunk count, duplicated text, and storage. |
| `--store-path` | `.rag_data/qdrant` | Persistent embedded database directory; reuse it for subsequent queries. |
| `--collection-name` | `rag_documents` | Collection within the database; examples use explicit names to separate corpora. |
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
|   |-- application/         # Transport-neutral workflow orchestration
|   |-- ingestion/           # Discovery, extraction, and chunking
|   |-- infrastructure/      # Model adapters and Qdrant persistence
|   |-- retrieval/           # Search, filtering, and reranking
|   |-- generation/          # Prompts, token budgets, and citations
|   |-- evaluation/          # Dataset schemas and quality metrics
|   `-- benchmarking/        # Isolated runs, provenance, and comparisons
|-- tests/                   # Offline unit and integration tests
|-- evaluation/datasets/     # Versioned synthetic corpora and labels
|-- docs/CLI_REFERENCE.md    # Detailed CLI workflows and contracts
|-- .github/workflows/ci.yml # Automated quality checks
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
| RAG interfaces and composition | LangChain and its Hugging Face/Qdrant integrations |
| Dense embeddings and reranking | Sentence Transformers / PyTorch |
| Local generation | Hugging Face Transformers |
| Sparse embeddings | FastEmbed BM25 |
| Vector storage | Embedded Qdrant via `qdrant-client` |
| PDF / DOCX extraction | pypdf / docx2txt |
| Packaging and quality | uv, setuptools, unittest, Coverage.py, Ruff, mypy, GitHub Actions |

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
[CI workflow](.github/workflows/ci.yml) runs on pull requests and pushes to
`main`, currently on Ubuntu with Python 3.11.

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
- Metadata filters are not authorization. There is no authentication, RBAC,
  tenant isolation, hosted-model integration, HTTP service, or deployment setup.
  Citations currently expose local source paths.
- Semantic chunking is benchmark-only and needs corpus-specific validation
  before use beyond experiments.

## Roadmap

The local pipeline, retrieval-quality follow-ups, evaluation, and benchmarking
are in progress. See the [quality plan](docs/RETRIEVAL_QUALITY_PLAN.md) before
Phase 4. Later work includes a FastAPI service, followed by
authentication, logging, monitoring, containerization, and deployment CI/CD.
Enterprise connectors, permissions-aware retrieval, and multi-tenancy are later
stages. See [ROADMAP.md](ROADMAP.md); these are plans, not current features.

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
