# Production-Minded RAG Pipeline

An end-to-end Retrieval-Augmented Generation pipeline for question answering
over business documents. The project uses LangChain, configurable local or
hosted models, and a persistent local Qdrant vector store to turn PDF, DOCX,
Markdown, HTML, and text files into grounded answers with deterministic source
citations.

The implementation starts with a local, reproducible baseline while preserving
the contracts needed to evolve toward hosted models, production APIs,
evaluation, observability, and enterprise data sources.

## Development Approach

This project was developed using AI-assisted software engineering with Codex. I
used AI as a development partner for implementation support while defining the
architecture, requirements, tests, and design decisions myself. All generated
code was reviewed, adapted, and validated through automated tests.

## Highlights

- Multi-format ingestion and extraction for PDF, DOCX, Markdown, HTML, and text
- Configurable recursive chunking with page and character-level provenance
- Reproducible chunking experiments with distribution and overlap-cost metrics
- Local normalized MiniLM embeddings through LangChain
- Environment-configured Gemini, OpenAI, and Claude model profiles
- Hosted Gemini/OpenAI embeddings and Gemini/OpenAI/Claude generation through
  dedicated LangChain integrations
- Optional local BM25 sparse embeddings with native Qdrant RRF hybrid search
- Optional local cross-encoder reranking with first-stage score provenance
- Persistent Qdrant storage with deterministic IDs and idempotent upserts
- Collection compatibility checks for embedding model, vector dimension,
  distance metric, and schema version
- Ranked semantic retrieval with configurable top-k and score thresholds
- Typed exact-match metadata filters pushed into Qdrant before top-k selection
- Bounded local or hosted generation with a versioned grounding and abstention
  prompt
- Deterministic citations built from retrieval metadata, never model output
- Typed configuration, stage-specific exceptions, and automated tests

## Architecture

```mermaid
flowchart LR
    A["PDF / DOCX / Markdown / HTML / Text"] --> B["Extraction"]
    B --> C["LangChain chunking"]
    C --> D["Selected dense embedding model"]
    C --> S["BM25 sparse embeddings"]
    D --> E["Persistent Qdrant index"]
    S --> E
    Q["User question"] --> F["Dense and sparse prefetch"]
    M["Metadata filters"] --> F
    E --> F
    F --> R["Reciprocal-rank fusion"]
    R --> X["Optional cross-encoder reranking"]
    X --> G["Bounded numbered evidence blocks"]
    G --> P["LangChain grounded-v2 prompt"]
    P --> H["Selected local or hosted LLM"]
    H --> I["Answer and deterministic citations"]
```

## Engineering Decisions

| Decision | Production rationale |
| --- | --- |
| Preserve provenance during extraction and chunking | Citations cannot be reconstructed reliably after metadata is lost. |
| Compare chunking candidates on one document snapshot | Keeps input variance from being mistaken for a chunking effect. |
| Use deterministic chunk IDs | Re-indexing updates logical chunks instead of creating duplicates. |
| Record collection model and dimension | Incompatible query vectors fail before corrupting retrieval behavior. |
| Version dense and hybrid collection schemas separately | Prevents queries from silently using collections that do not contain the required sparse vectors. |
| Fuse dense and sparse ranks with RRF | Combines semantic and exact-term retrieval without mixing incomparable raw scores. |
| Push metadata filters into Qdrant | Selects top-k only from eligible chunks and avoids leaking excluded candidates into application code. |
| Overfetch before optional cross-encoder reranking | Lets the first stage optimize recall while the second stage improves precision without scoring the entire corpus. |
| Preserve first-stage rank and score after reranking | Keeps retrieval behavior auditable and avoids presenting incomparable scores as one metric. |
| Skip generation without evidence | Avoids unnecessary inference and unsupported answers. |
| Version the generation prompt and return its identifier | Makes answer behavior reproducible across evaluation runs, deployments, and incident analysis. |
| Delimit and number retrieved evidence independently of citations | Gives the model clear evidence boundaries while citation records remain deterministic application data. |
| Budget the complete prompt before generation | Local tokenizer counts and conservative hosted estimates prevent input overflow while keeping citations aligned with the exact evidence sent. |
| Build citations outside the LLM | Prevents fabricated filenames, pages, and source identifiers. |
| Keep provider boundaries behind LangChain interfaces | Makes later model and infrastructure changes less invasive. |
| Select provider profiles by stable CLI aliases | Keeps credentials and frequently changing model IDs out of commands and application code. |
| Keep `.env` untracked and redact credentials from profiles | Reduces accidental secret disclosure through source control, logs, and error messages. |
| Pair Claude with local embeddings | Anthropic does not expose an embeddings API, so this keeps the three-variable Claude profile functional without pretending its API supports embeddings. |

## Quick Start

Requirements:

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

Install the locked environment:

```powershell
uv sync
```

The local baseline works without API keys. To use hosted providers, create a
private `.env` from the committed skeleton and replace the relevant key:

```powershell
Copy-Item .env.example .env
```

The `.env` file is ignored by Git. Never commit it or paste real keys into
`.env.example`.

Index one file or an entire directory:

```powershell
uv run python -m rag_pipeline index path/to/documents
```

Ask a question against the persisted collection:

```powershell
uv run python -m rag_pipeline answer "Which vector database does this project use?"
```

Select one configured hosted profile with the same alias for indexing and
answering:

```powershell
uv run python -m rag_pipeline index path/to/documents --model gemini
uv run python -m rag_pipeline answer "What is the policy?" --model gemini
```

Use separate collection names for unrelated corpora. For example, index an
expense-policy corpus with `--collection-name expense_policies` and pass the
same option to `retrieve` and `answer`; local Qdrant collections persist across
commands.

The first embedding, reranking, and generation runs download their configured
Hugging Face model weights when those stages are enabled. Public local models
do not require an API key.

## Hosted Model Profiles

`.env.example` defines three values for each selectable profile:

```dotenv
GOOGLE_API_KEY=<API-KEY>
GEMINI=gemini-3.1-flash-lite
GEMINI_EMBED=gemini-embedding-2

OPENAI_API_KEY=<API-KEY>
OPENAI=gpt-5-mini
OPENAI_EMBED=text-embedding-3-small

ANTHROPIC_API_KEY=<API-KEY>
CLAUDE=claude-sonnet-4-6
CLAUDE_EMBED=sentence-transformers/all-MiniLM-L6-v2
```

`GEMINI`, `OPENAI`, and `CLAUDE` are generation model IDs. Their `_EMBED`
partners select the embedding model used for both indexing and queries. Model
IDs are passed directly to the provider integration, so they can be changed in
`.env` without editing Python. Process environment variables override `.env`,
which lets deployment systems inject secrets without writing a file.

Use `--model gemini`, `--model openai`, or `--model claude` on every command
that embeds or queries content. The `answer` command reuses that profile for
generation automatically:

```powershell
# Google Gemini generation and embeddings
uv run python -m rag_pipeline index path/to/documents --model gemini
uv run python -m rag_pipeline answer "What is required?" --model gemini

# OpenAI generation and embeddings
uv run python -m rag_pipeline index path/to/documents --model openai
uv run python -m rag_pipeline answer "What is required?" --model openai

# Anthropic generation with the configured local embedding model
uv run python -m rag_pipeline index path/to/documents --model claude
uv run python -m rag_pipeline answer "What is required?" --model claude
```

An OpenAI API key and API billing are separate from a ChatGPT subscription.
Anthropic currently [does not provide an embedding model](https://platform.claude.com/docs/en/build-with-claude/embeddings),
so `CLAUDE_EMBED` is intentionally interpreted as a local Hugging Face model.
A future hosted Voyage integration would require its own API key and billing
contract rather than reusing `ANTHROPIC_API_KEY` incorrectly.

Embedding model identity and vector dimension are recorded in Qdrant. After
changing any `_EMBED` value, reindex into a new collection or rebuild the old
one before retrieval; vectors from different embedding spaces cannot be mixed.
Hosted embedding sends chunk and query text to the selected provider, and
hosted generation sends the bounded retrieved evidence. That introduces usage
cost, network latency, provider rate limits, and external data-processing
considerations that do not apply to the local baseline.

## Example Output

```text
Answer:
The project uses a persistent local Qdrant vector store.

Sources:
[1] README.md (chunk 3, characters 1740-2050)
    The local prototype stores vectors in Qdrant under .rag_data/qdrant...
```

Citation records also retain the stable chunk ID, retrieval rank, and retrieval
score for programmatic use. Scores are intentionally not presented as answer
confidence values.

## CLI Workflow

```powershell
# Inspect supported commands
uv run python -m rag_pipeline -h

# Load supported documents
uv run python -m rag_pipeline ingest path/to/documents

# Inspect chunk counts
uv run python -m rag_pipeline chunk path/to/documents

# Compare several chunking configurations without model calls or indexing
uv run python -m rag_pipeline chunk-experiment path/to/documents --candidate 500:100 --candidate 1000:200 --candidate 1500:300

# Verify local embedding output
uv run python -m rag_pipeline embed path/to/documents

# Build or update the persistent Qdrant collection
uv run python -m rag_pipeline index path/to/documents

# Build a separate dual-vector collection for hybrid search
uv run python -m rag_pipeline index path/to/documents --collection-name policies_hybrid --search-mode hybrid

# Inspect ranked evidence without generation
uv run python -m rag_pipeline retrieve "What is the policy?" --top-k 3

# Retrieve 20 candidates, then return the best 3 after local reranking
uv run python -m rag_pipeline retrieve "What is the policy?" --rerank --candidate-k 20 --top-k 3

# Restrict retrieval before semantic top-k selection
uv run python -m rag_pipeline retrieve "What is the policy?" --filter file_extension=.pdf

# Retrieve evidence and generate a cited answer
uv run python -m rag_pipeline answer "What is the policy?" --top-k 3
```

Useful options include:

- `--model gemini|openai|claude` for an environment-backed provider profile, or
  `--model MODEL_ID` for a local Hugging Face embedding model
- `--model-revision` for local embeddings, including `CLAUDE_EMBED`
- `--generation-model` and `--generation-model-revision` for the local LLM path
- `--device` and `--generation-device` for CPU or CUDA placement
- `--temperature` for an explicit generation temperature; hosted profiles use
  their provider default when it is omitted
- `--top-k` and `--score-threshold` for retrieval behavior
- `--rerank` and `--candidate-k` for optional second-stage reranking
- `--reranker-model`, `--reranker-model-revision`, `--reranker-device`,
  `--reranker-cache-dir`, `--reranker-batch-size`, and
  `--reranker-max-length` for the local cross-encoder
- `--search-mode dense|hybrid` for the collection and retrieval strategy
- `--sparse-model`, `--sparse-cache-dir`, `--sparse-batch-size`, and
  `--sparse-threads` for local hybrid indexing and queries
- repeatable `--filter KEY=VALUE` for exact metadata filters with AND semantics
- repeatable `--candidate SIZE:OVERLAP` for chunking experiments
- `--output-format table|json` for human or machine-readable experiment reports
- `--max-input-tokens` for an optional limit below the tokenizer model maximum
- `--max-context-characters` as a secondary generation context guard
- `--collection-name` and `--store-path` for Qdrant persistence

## Chunking Experiments

The `chunk-experiment` command materializes one document snapshot and runs every
candidate against that same input. With no `--candidate` options, it compares
`500:100`, `1000:200`, and `1500:300`; each pair represents maximum chunk
characters and target overlap characters.

The report includes chunk count, min/mean/p95/max chunk length, total emitted
characters, and duplicated characters. Duplication is calculated from the
actual provenance intervals emitted by LangChain, because separator-aware
splitting does not always produce the configured overlap exactly. Chunk count
and total characters are useful cost proxies; the length distribution exposes
undersized tails and context pressure.

Use JSON when recording or comparing runs:

```powershell
uv run python -m rag_pipeline chunk-experiment path/to/documents --output-format json
```

These are structural diagnostics, not retrieval-quality scores. A candidate
with fewer chunks or less duplication is not automatically better. Production
selection should combine these measurements with representative queries,
relevance labels, latency, and cost; that benchmark is introduced in the later
retrieval-evaluation task.

## Metadata Filters

Both `retrieve` and `answer` accept repeatable exact-match filters. Conditions
are translated into structured Qdrant payload filters and applied before
semantic top-k selection:

```powershell
uv run python -m rag_pipeline answer "What is the policy?" `
  --filter file_extension=.pdf `
  --filter page=0
```

Repeated filters use AND semantics. Unquoted integers and lowercase JSON
booleans are typed automatically, while other unquoted values remain strings.
Use a quoted JSON string when a numeric-looking value must stay a string:

```powershell
uv run python -m rag_pipeline retrieve "Find policy 001" --filter 'policy_version="001"'
```

Automatically indexed fields include `source`, `file_name`, `file_stem`,
`file_extension`, `byte_size`, `extractor`, and, for PDFs, zero-based `page` and
`total_pages`. The same API supports custom scalar metadata added to LangChain
documents programmatically. A missing field or unmatched value produces no
eligible chunks; the `answer` command then abstains before loading the LLM.

Metadata filters improve relevance but are not an authorization system by
themselves. A production API must inject mandatory tenant and ACL constraints
server-side so callers cannot omit or replace them, and frequently filtered
fields should receive Qdrant payload indexes at scale.

## Hybrid Search

Dense search remains the default and existing schema-v1 collections continue to
work unchanged. Hybrid mode creates a schema-v2 collection containing named
MiniLM dense vectors and BM25 sparse vectors. Use a separate collection name
and pass the same mode to indexing and retrieval:

```powershell
uv run python -m rag_pipeline index path/to/documents `
  --collection-name policies_hybrid `
  --search-mode hybrid

uv run python -m rag_pipeline retrieve "What does code ZX-42 require?" `
  --collection-name policies_hybrid `
  --search-mode hybrid
```

The first hybrid command downloads the configured FastEmbed sparse model into
`.rag_data/fastembed`; it runs locally afterward and needs no API key. Qdrant's
IDF modifier is part of the hybrid collection schema. At query time LangChain
prefetches dense and sparse candidates, applies the same metadata filters to
both branches, and asks Qdrant to combine ranks with reciprocal-rank fusion.
Queries that produce an empty sparse vector safely fall back to the collection's
dense vector.

CLI results expose `score_kind=cosine` or `score_kind=rrf`. These scores are not
interchangeable: dense thresholds are based on cosine similarity, while hybrid
thresholds apply to the fused rank score. Calibrate each mode independently on
a representative evaluation dataset. The default `Qdrant/bm25` model favors
English tokenization and stemming; multilingual or domain-specific corpora may
need a different sparse model.

Attempting hybrid retrieval against an existing dense collection fails closed.
This is intentional: adding a sparse field does not populate old points, so the
documents must be reindexed into a hybrid collection before fusion is valid.

## Cross-Encoder Reranking

Reranking works after either dense or hybrid retrieval and does not require
reindexing. The first stage retrieves a wider candidate set, then the local
cross-encoder scores each query-chunk pair jointly and returns the final
`--top-k` results:

```powershell
uv run python -m rag_pipeline retrieve "What does code ZX-42 require?" `
  --collection-name policies_hybrid `
  --search-mode hybrid `
  --rerank `
  --candidate-k 20 `
  --top-k 4
```

The first reranked query downloads
`cross-encoder/ms-marco-MiniLM-L6-v2` into `.rag_data/rerankers`. It then runs
locally through Sentence Transformers and needs no API key. The model scores
all candidates in batches, with a maximum tokenized query-chunk length of 512
by default. The service uses LangChain's cross-encoder
`score([(query, document), ...])` contract, so a hosted or community scorer can
replace the local adapter later without changing retrieval or generation
contracts.

On Windows without Developer Mode, Hugging Face may warn that cache symlinks
are unavailable. Inference still works, but the fallback cache can consume more
disk space.

Metadata filters and `--score-threshold` apply during first-stage retrieval,
before any text reaches the reranker. This is important for permissions-aware
systems: reranking must never become a path around tenant or ACL filtering.
`--candidate-k` must be at least `--top-k`; increasing it may improve recall but
increases latency roughly in proportion to the number of scored pairs.

Reranked CLI results use `score_kind=cross_encoder` and retain
`retrieval_rank`, `retrieval_score`, and `retrieval_score_kind`. Cross-encoder
scores are model-specific logits, not calibrated probabilities, and must not be
compared directly with cosine or RRF scores. A later evaluation task should
measure whether the selected model and candidate width improve ranking on this
project's actual queries.

## Prompt Optimization

Answer generation uses a LangChain `PromptTemplate` identified as
`grounded-v2`. The prompt gives the model a compact contract: use supported
evidence only, ignore instructions found in retrieved text, abstain with one
exact response when evidence is missing, insufficient, or conflicting, and
return concise answer text without inventing sources or citations.

Ranked chunks are rendered as numbered `[Evidence n]` blocks. The same accepted
chunk sequence drives citation numbering, so evidence block 1 and citation
`[1]` always refer to the same retrieval result even when empty chunks are
skipped. Filenames and other provenance are not placed in the model prompt;
citations continue to be built and validated outside the LLM. Every
`GeneratedAnswer` records `prompt_identifier` alongside `model_identifier` for
reproducible evaluation and troubleshooting.

Character and tokenizer budgets include block labels, separators, the question,
all prompt instructions, and special tokens. When the final block must be
truncated, its citation excerpt still contains only the exact raw evidence
prefix sent to the model, excluding labels and the visual ellipsis.

Local generation counts the rendered prompt with the Hugging Face model's exact
tokenizer. Hosted profiles use a conservative UTF-8 byte estimate and an
8,192-token application cap. This avoids hidden token-count API calls during
binary-search packing and remains safely below the context windows of the
example models, at the cost of sometimes admitting less evidence than the
provider could accept. `--max-input-tokens` may lower, but not raise, that hosted
cap.

The evidence delimiters improve structure but do not sanitize hostile content
or form an authorization boundary. The local FLAN-T5 baseline may also ignore
instructions or abstain too often. Prompt quality therefore needs representative
faithfulness and abstention evaluation in the later answer-evaluation task. A
single text prompt is intentional for the current text-to-text model; a future
chat-model adapter can express the same contract with system and user messages.

## Local Baseline

The default embedding model is
`sentence-transformers/all-MiniLM-L6-v2`, producing 384-dimensional normalized
vectors. The default generation model is `google/flan-t5-small`, selected as a
small architectural baseline rather than a production-quality answer model.
Hybrid mode adds the local `Qdrant/bm25` FastEmbed sparse encoder and Qdrant RRF.
Optional reranking uses the small English
`cross-encoder/ms-marco-MiniLM-L6-v2` model as a CPU-friendly baseline.

Transformers is pinned below version 5 because the current LangChain T5 adapter
uses the `text2text-generation` pipeline API. Model revisions can be pinned
independently for reproducible indexing and generation.

Local generation counts the exact rendered prompt with the selected tokenizer,
including special tokens, and reserves an eight-token safety margin below the
model limit. The Hugging Face pipeline also enables truncation as a final guard,
although application-level budgeting remains authoritative so citation ranges
stay accurate.

The `answer` command defaults to a retrieval score threshold of `0.20` and
abstains when no chunk meets it. The `retrieve` command intentionally has no
default threshold so retrieval scores can be inspected during evaluation.
Dense and fused thresholds are model-, mode-, and corpus-specific and should be
calibrated rather than treated as confidence scores.

## Testing

Run the complete suite:

```powershell
uv run python -m unittest discover -s tests -v
```

The suite currently contains 101 tests covering ingestion, extraction, chunking,
chunking experiments, dense and sparse embedding contracts, persistent dense
and hybrid indexing, typed metadata filtering, cosine and RRF retrieval,
cross-encoder reranking, versioned guarded generation, evidence boundary and
token-budget behavior, deterministic citations, provider-profile loading and
secret redaction, hosted LangChain adapter configuration, and CLI integration.
Provider calls use test doubles; the local model path has also been verified end
to end with MiniLM, Qdrant, the MS MARCO cross-encoder, and FLAN-T5.

## Project Layout

```text
.
|-- src/
|   `-- rag_pipeline/
|       |-- __main__.py
|       |-- citations.py
|       |-- chunking.py
|       |-- chunking_experiments.py
|       |-- embeddings.py
|       |-- exceptions.py
|       |-- extraction.py
|       |-- generation.py
|       |-- ingestion.py
|       |-- model_profiles.py
|       |-- reranking.py
|       |-- retrieval.py
|       |-- sparse_embeddings.py
|       `-- vector_store.py
|-- tests/
|   |-- test_citations.py
|   |-- test_chunking.py
|   |-- test_chunking_experiments.py
|   |-- test_embeddings.py
|   |-- test_extraction.py
|   |-- test_generation.py
|   |-- test_ingestion.py
|   |-- test_model_profiles.py
|   |-- test_package.py
|   |-- test_reranking.py
|   |-- test_retrieval.py
|   |-- test_sparse_embeddings.py
|   `-- test_vector_store.py
|-- ARCHITECTURE.md
|-- .env.example
|-- PROJECT_BRIEF.md
|-- ROADMAP.md
|-- pyproject.toml
`-- uv.lock
```

## Current Limitations

- Hybrid retrieval currently uses fixed unweighted RRF; fusion tuning and
  learned sparse retrieval remain roadmap items.
- The default reranker is a small English MS MARCO baseline; candidate width,
  latency, domain fit, multilingual quality, and score calibration still need
  evaluation on representative business queries.
- Metadata filters currently support exact scalar AND conditions; range, OR,
  list-membership, and policy-composition support are future extensions.
- The default retrieval threshold is a conservative safety baseline pending
  calibration against an evaluation dataset.
- Citations identify the evidence supplied to the model but are not yet mapped
  to individual answer claims.
- The `grounded-v2` prompt is a hand-authored baseline; faithfulness,
  abstention, and prompt-injection resilience are not yet benchmarked on a
  representative answer dataset.
- Local source paths should become stable document IDs or authorized URLs before
  citations are exposed through a service.
- The default generation model prioritizes local accessibility over answer
  quality and throughput.
- Hosted calls are synchronous and currently have no application-level retry,
  timeout, rate-limit, or cost-budget policy beyond provider SDK defaults.
- Hosted prompt budgeting is deliberately conservative and does not yet query
  model metadata dynamically when a configured model ID changes.
- Anthropic has no native embedding API; the Claude profile uses local
  embeddings until a separately credentialed hosted embedding provider is added.

## Roadmap

The staged roadmap covers retrieval-quality experiments, evaluation datasets,
benchmarking, a FastAPI service, authentication, monitoring, containerization,
CI/CD, permissions-aware retrieval, and multi-tenant enterprise integrations.
See [ROADMAP.md](ROADMAP.md) for the full sequence.
