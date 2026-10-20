# RAG Pipeline Architecture

## High-Level Components

1. CLI Transport Adapter
2. Transport-Neutral Application Use Cases
3. Document Ingestion, Parsing, and Cleaning
4. Recursive Chunking plus Isolated Structure/Semantic Experiments
5. Embedding Generation
6. Vector Database
7. Hybrid Retrieval Layer (dense + BM25 sparse with RRF)
8. Local Cross-Encoder Reranking Layer
9. Versioned Prompt Construction and Token-Aware Evidence Packing
10. LLM Generation
11. Citation System
12. Versioned Retrieval and Answer Evaluation Framework with Test Datasets
13. Isolated Benchmark Runner, Artifacts, Comparisons, and Regression Gates
14. Local read-only FastAPI adapter with shared bearer authentication and request logging
15. Process-local API and RAG outcome monitoring
16. Frontend/UI (planned)
17. Local CLI/API container packaging (implemented); deployment infrastructure (planned)

---

## Target Evolution

Phase 1:
Local prototype

Phase 2:
Improved retrieval and prompt quality

Phase 3:
Evaluation framework

Phase 4:
Production hardening

Phase 5:
Enterprise integrations

---

## Package Ownership

The source tree uses shallow feature packages rather than one large flat module
directory or a deeply nested clean-architecture hierarchy:

- `rag_pipeline.cli` owns argparse configuration, command dispatch, and terminal
  rendering.
- `rag_pipeline.api` owns bearer authentication, bounded HTTP schemas,
  loopback-oriented service configuration, provider lifetime, HTTP errors,
  sanitized request events, and bounded operational metrics.
- `rag_pipeline.application` owns transport-neutral indexing, retrieval, and
  answer use cases that coordinate multiple pipeline stages.
- `rag_pipeline.ingestion` owns filesystem discovery, format extraction,
  recursive and experimental chunking policies, and structural diagnostics.
- `rag_pipeline.retrieval` owns first-stage search and optional cross-encoder
  reranking.
- `rag_pipeline.generation` owns evidence packing, prompt construction, model
  invocation, abstention, and citations.
- `rag_pipeline.evaluation` owns versioned retrieval and answer schemas, metrics,
  and report formatting.
- `rag_pipeline.benchmarking` owns isolated execution, provenance, timings,
  artifacts, comparisons, and regression thresholds.
- `rag_pipeline.infrastructure` owns local dense and sparse model adapters plus
  Qdrant persistence. These modules perform provider or storage I/O but do not
  own CLI or application workflow decisions.
- `rag_pipeline.exceptions` remains a shared root module so every layer can use
  one stable, actionable error hierarchy without depending on another feature.

Source, tests, and public examples use these canonical package paths. The package
root contains only metadata, the module entry point, the shared exception
hierarchy, and the typing marker; implementation and re-export ownership stays
inside feature packages. The pre-1.0 project deliberately removed its temporary
flat import aliases instead of carrying a migration layer into Phase 4. Tests
mirror the same feature boundaries while repository-wide CLI and package smoke
tests remain at the test root.

This structure adds a small navigation cost compared with a flat package, but it
keeps related changes together as the pipeline grows. The hierarchy deliberately
stops at one feature level: additional folders should be introduced only when a
feature develops multiple cohesive responsibilities, not for every class or
function.

---

## Current Adapter And Application Contract

- `rag_pipeline.__main__` remains the stable module entry point and re-exports
  `main` and `build_parser`, but it owns no command behavior.
- `rag_pipeline.cli.app` constructs the root parser and dispatches through an
  explicit handler attached by each subcommand. There is no central conditional
  covering every command.
- Shared option groups define one embedding, chunking, retrieval, reranking,
  generation, and storage vocabulary across commands.
- CLI configuration builders translate dynamic argparse values into validated
  application configuration objects before model or storage side effects.
- Command handlers are grouped by document, indexing, query, evaluation, and
  benchmark responsibilities. They delegate reusable indexing, retrieval,
  and no-evidence answer behavior to `rag_pipeline.application`.
- Application use cases own cross-stage invariants, provider lifecycle, and
  orchestration. They accept explicit dependencies where useful for deterministic
  tests and do not import argparse or terminal output helpers.
- Provider factories remain lazy, so parser creation and help output do not
  initialize models, download files, or open a database. Reranking models are
  not initialized when first-stage retrieval returns no candidates.
- Terminal-specific retrieval and answer rendering is isolated from service
  orchestration and can be tested without provider calls.

The application layer is intentionally small. It coordinates existing domain
services rather than replacing their validation or algorithms. The API reuses
these contracts without importing CLI handlers. HTTP authentication,
request/response schemas, and error mapping live in `rag_pipeline.api`.

## Current HTTP Boundary

- `GET /health` checks process liveness only. `POST /v1/retrieve` and
  `POST /v1/answer` read an existing local collection; no upload or indexing
  route is exposed.
- Startup requires a 32-256-character URL-safe `RAG_API_BEARER_TOKEN`. Only its
  SHA-256 digest stays in app state. FastAPI's bearer dependency rejects
  unauthenticated query requests before model or storage work; `/health` and
  OpenAPI documentation remain public. A token change requires restart.
- The API reads model and collection identity from server-owned `RAG_API_*`
  environment variables at startup, never from per-request file paths.
- One process lazily opens embedded Qdrant and the embedding model on first
  query, then loads generation only when answer evidence exists. A lifespan
  hook closes the store at shutdown; a lock serializes queries against these
  shared local resources.
- Pydantic bounds query length, result count, and score thresholds. Known
  collection errors receive stable HTTP statuses; provider details are hidden
  behind a generic unavailable response.
- A pure ASGI middleware emits one JSON request event to stderr and adds a
  server-generated `X-Request-ID` to handled responses. Events contain route
  templates, statuses, and duration, but never raw URLs, bodies, credentials,
  or source text. Documented Uvicorn startup disables its separate URL-bearing
  access log. This is operational logging, not audit logging.
- A second ASGI middleware records request count, duration, and in-flight load
  in an app-owned Prometheus registry. Route templates and fixed outcome labels
  bound cardinality; the protected `/metrics` scrape is excluded from counting.
  Successful query handlers also count empty/nonempty retrieval and no-evidence,
  cited, or model-abstained answers. These aggregates cannot establish factual
  answer quality, per-user usage, or document-level access. One process owns
  one registry; multi-worker aggregation and alerting are not implemented.
- The shared token authenticates possession, not a person or document scope.
  There is no user identity, authorization, rate limiting, or permission-aware
  retrieval. It must stay on loopback; source paths and document text in
  responses may be sensitive. See
  [the API reference](docs/API_REFERENCE.md).

## Current Container Boundary

The optional Docker image installs the locked runtime package in a builder stage
and copies only its non-editable virtual environment into a Python 3.11 runtime
stage. The final process runs as a non-root user. Both CLI indexing and the
single-process API use the same image; they must run sequentially against one
embedded Qdrant volume. The default working directory preserves the existing
`.rag_data/qdrant` path, while a second volume retains Hugging Face model caches.
The image does not include source documents or credentials. Its Docker health
check calls public `/health`, so it indicates process liveness rather than
collection readiness. The documented host port binds to loopback only. This is
local packaging, not an authenticated multi-user deployment; see
[Docker setup](docs/DOCKER.md).

The CI release boundary builds and smoke-tests this image on pull requests.
For a matching annotated version tag on `main`, the workflow repeats quality
checks and the container smoke test before publishing a versioned GHCR image.
Only the tag job receives package-write permission. Publication does not
configure storage, credentials, network exposure, or a running deployment.

---

## Current Index Synchronization Contract

- The CLI `index` use case discovers the complete file set before extraction and
  passes explicit source paths to Qdrant, including files yielding no chunks.
- `LocalVectorStore.replace_sources()` validates scope, snapshots old IDs by
  exact source, verifies collection compatibility before writing, upserts new
  points, then deletes obsolete IDs. `LocalVectorStore.index()` keeps its
  general upsert contract for other callers, including isolated benchmarks.
- Files no longer present in a directory are retained unless `--prune-missing`
  is explicitly requested for one recursively scanned directory. Sources
  outside that resolved directory are never included in directory pruning.
- A failure before upsert leaves the collection unchanged. Multiple upsert and
  delete calls are not atomic; a database failure may require retrying the same
  source scope to finish synchronization.

## Current Extraction Contract

HTML files are decoded into visible text with Python's `html.parser`. Script,
style, head, template, and navigation subtrees are omitted; headings, tables,
asides, and footers remain text. Plain text and Markdown retain their raw
content. HTML chunk `start_index` and `end_index` point into normalized
extracted text, not raw HTML byte positions. Source paths and PDF page metadata
are unchanged. Existing HTML sources must be reindexed for the new text and
`extractor=html.parser` metadata to take effect; a filter on the previous
`extractor=text` value will no longer match those HTML chunks.

---

## Current Generation Contract

- LangChain composes the `grounded-v2` prompt with the configured language model
  and string output parser.
- `rag_pipeline.generation.prompting` owns the versioned template, evidence
  boundaries, tokenizer protocol, and token-aware packing algorithm.
- `rag_pipeline.generation.service` owns model lifecycle, invocation, answer
  assembly, and deterministic citation integration. The
  `rag_pipeline.generation` package exposes the supported feature-level API.
- Ranked chunks are packed into numbered evidence blocks under exact character
  and tokenizer limits before a model is invoked.
- Retrieved text is explicitly treated as untrusted data, and unsupported or
  conflicting evidence maps to one deterministic abstention response.
- The answer result records both model and prompt identifiers; source citations
  are constructed outside the language model from validated chunk provenance.

Evidence delimiters are prompt structure, not a security or authorization
boundary. Deterministic reference and abstention evaluation is implemented, but
semantic faithfulness still requires calibrated human or model-based judgment
before production use.

---

## Current Chunking Strategy Contract

- Normal `chunk`, `embed`, and `index` workflows retain the established
  LangChain recursive-character policy and its defaults.
- Isolated benchmarks additionally accept a structure-aware recursive policy.
  Markdown and raw HTML documents use LangChain's language-specific separator
  priorities. Extracted HTML has no tags, so it and other formats use the
  baseline separator order.
- The semantic policy forms punctuation/paragraph units, embeds buffered
  sentence contexts through LangChain's `Embeddings` interface, and proposes
  boundaries above a configurable cosine-distance percentile.
- Semantic output has no overlap. A validated maximum character size remains a
  hard invariant, and oversized individual units fall back to zero-overlap
  LangChain recursive splitting.
- Every policy preserves source metadata, zero-based chunk order, character
  offsets, chunk counts, and character counts. Experimental chunks also record
  their strategy and applicable structure language.
- The benchmark initializes the dense model once. Semantic boundary inference
  and final chunk embedding share that validated provider and dimension state,
  although they remain separate inference passes and timing stages.

Semantic chunking can improve topical coherence but adds ingestion latency and
is sensitive to sentence segmentation, model choice, and threshold calibration.
The implementation is intentionally benchmark-only until representative quality
and operational results justify expanding the public indexing workflow.
The [RQ-05 local comparison](docs/RQ05_EXTRACTION_CHUNKING.md) found no
retrieval gain on the small v2 labels from switching the default policy, so
recursive indexing remains unchanged. PDF text-layer order and OCR remain
separate, unimplemented extraction concerns.

---

## Current Retrieval Evaluation Contract

- Strict schema-v1 JSON remains supported for exact document-metadata selectors.
  Schema v2 combines exact metadata with a case-sensitive source-content anchor,
  allowing one judgment to survive ordinary changes in chunk boundaries. Schema
  v3 keeps that positive selector and permits empty relevance for unsupported
  questions, while requiring at least one positive case per dataset.
- Evaluation runs the same LangChain/Qdrant dense or hybrid retriever, metadata
  filters, score gate, and optional cross-encoder reranker used by interactive
  commands; it never invokes generation or mutates the collection.
- V1/v2 reports retain per-case Hit@k, Precision@k, Recall@k, reciprocal rank,
  and their unchanged macro averages. V3 averages these only over positive
  cases; negative cases receive separate no-hit and false-positive rates.
- V3 also exposes distinct matched and missing evidence selectors, redundant
  overlap-hit ranks, and unwanted ranks alongside chunk-level precision.
- Optional first-stage score-gate comparison reruns the same v3 cases for an
  explicit baseline and candidate gates through one initialized pipeline. Its
  separate report records settings, observed score kind, and per-case evidence
  and unwanted-hit changes; reranking is excluded from this calibration.
- Optional candidate-width comparison reruns each v3 query at every requested
  width, then scores the same pool at full width, at the final cutoff without
  reranking, and after cross-encoder reranking. Per-case selector diagnostics
  separate missing first-stage evidence from final-selection losses. The
  application retrieval session exposes these stages without changing the
  ordinary `retrieve` contract; dense and hybrid collections are compared in
  separate compatible runs.
- Table output supports local diagnosis and JSON output supports saved
  comparisons.
- V1/v2 each supply 17 answerable synthetic policy cases. The separate v3
  retrieval-only stress set adds unsupported, ambiguous, multi-document, and
  current-versus-archived policy cases; it has no paired answer labels.

Exact v1 metadata selectors are transparent but depend on stable chunk
provenance. V2 content anchors support boundary experiments but require a new
dataset version when source wording changes. Portable business datasets should
eventually use immutable document/version identifiers and reviewed evidence
spans. The benchmark captures latency and immutable input fingerprints around
these metrics.

---

## Current Answer Evaluation Contract

- A separate strict schema-v1 JSON file supplies named questions, explicit
  answerability labels, and one or more accepted references for answerable cases.
- Evaluation runs the same LangChain retrieval, optional reranking,
  tokenizer-bounded `grounded-v2` prompt, local generation, and deterministic
  citation path used by the `answer` command.
- Reports separate normalized exact match and token F1 from abstention accuracy,
  precision, recall, and answerable response rate so class imbalance cannot hide
  a model that always answers or always refuses.
- Citation behavior checks that responses have citations and abstentions do not;
  per-case JSON also records model and prompt IDs, used-context count, and prompt
  truncation state.
- Models and database connections are reused for a complete dataset run, while
  the index remains read-only.

These deterministic metrics are regression signals, not semantic judges.
Lexical overlap does not establish factual entailment, and citation presence
does not prove claim-level support. The committed synthetic dataset supplies 17
answerable and four expected-abstention cases. Benchmark manifests are now
implemented; independent annotation, human review, and a calibrated NLI or LLM
judge remain future work.

---

## Current Test Dataset Contract

- `evaluation/datasets/asteria-policies-v1` contains five repository-owned,
  synthetic Markdown policies. No local user documents or customer data are
  part of the benchmark.
- `asteria-policies-v2` preserves the fictional policy domain in a new immutable
  asset and replaces chunk-index relevance with file metadata plus exact source
  anchors for recursive, structure-aware, and semantic comparisons.
- Retrieval and answer datasets share IDs and query text for all 17 answerable
  cases; the answer dataset adds four unsupported questions.
- Retrieval labels use `file_name` and zero-based `chunk_index` against the
  documented 1000-character size and 200-character overlap defaults.
- Integrity tests reproduce the ten expected chunks, require each selector to
  match exactly one chunk, align the paired schemas, and check reference
  vocabulary against labeled evidence.
- V2 integrity tests require every anchor to be unique in its source and resolve
  under the default configuration of all three strategies, then verify accepted
  references against each strategy's matched evidence.
- Dataset versions are repository artifacts and should be treated as immutable
  after benchmark results depend on them.

Synthetic data is safe, portable, and reviewable, but it cannot represent the
full language, layout, security, or ambiguity of production documents. The
labels were manually cross-checked but do not yet have independent annotator
agreement. Benchmark settings and thresholds remain separate from these
ground-truth assets so dataset versions do not encode a preferred pipeline.

---

## Current Benchmark Contract

- `rag_pipeline.benchmarking` is the stable orchestration package.
  `benchmarking.runner` performs execution while configuration, metrics, timing,
  threshold evaluation, provenance, artifact handling, and report construction
  live in focused sibling modules.
- `rag_pipeline.benchmarking.artifacts` owns artifact validation, regression
  gates, and comparison behavior under the canonical benchmark package.
- `benchmark` starts from one explicit corpus and paired retrieval/answer files,
  then builds a fresh temporary Qdrant collection. Retrieval schemas v1, v2,
  and v3 are accepted, but source-anchor labels are required for valid
  boundary-strategy and prompt-evidence comparisons. The
  index is closed and deleted after its storage size is recorded.
- Corpus relative paths and bytes plus both dataset files receive SHA-256
  fingerprints. Artifacts omit absolute corpus, cache, output, and work paths.
- One manifest records chunking, dense and optional sparse models, retrieval,
  filters, reranking, generation, prompt/token limits, devices, batching, Git
  state, package versions, CPU/platform data, and available CUDA devices.
- Models are initialized once. Reports separate model-load, extraction,
  chunking (including semantic boundary inference), final embedding, index,
  retrieval-evaluation, and answer-evaluation stage durations, with ordered
  per-case retrieval and end-to-end answer latency.
- When every answerable case has a paired source anchor, benchmark reports also
  classify each selector as not retrieved, retrieved but not packed, packed
  with an abstention, or packed with a non-abstaining answer. The check uses
  exact text prefixes passed to generation, not bounded citation excerpts.
  Older chunk-index or unpaired labels keep their existing artifact shape.
- Optional strict threshold profiles apply inclusive minimum or maximum bounds
  to allowlisted quality and operational metrics. Profiles bind to corpus and
  dataset hashes plus final top-k and fail before model loading on a mismatch.
  The report is still written when a metric gate fails, and the CLI returns
  status `1`.
- `compare-benchmarks` requires identical corpus and dataset hashes and final
  top-k. Pipeline configuration may differ intentionally. Operational deltas
  are marked diagnostic when the recorded environment or devices differ.

Fresh indexing is slower than reading an existing collection, but it prevents
stale points and unknown chunking state from producing misleading comparisons.
The current timing model is one local wall-clock pass, not a throughput or load
benchmark. Production evolution should add configurable warmups and repetitions,
confidence intervals, memory and accelerator telemetry, request concurrency,
cost metrics, durable artifact storage, retention policies, and CI calibration
on controlled runners.

Artifacts include evaluation queries and generated answers. Reports created from
non-synthetic data therefore require the same authorization, privacy, retention,
and deletion controls as their source documents. Quality thresholds should be
approved from reviewed baselines; example values are not production defaults.

---

## Current Quality Contract

- Ruff owns deterministic formatting, import ordering, and focused correctness
  linting for source and tests.
- Mypy checks all package source in strict mode. Dynamic JSON and provider values
  are validated before narrow casts; the untyped `docx2txt` dependency has the
  only module-level missing-stub exception.
- The offline unittest suite uses provider doubles and temporary in-memory or
  local Qdrant stores. It requires no credentials, model downloads, GPU, or
  pre-existing collection.
- Test modules are grouped by the same application, ingestion, retrieval,
  generation, evaluation, benchmarking, and infrastructure boundaries used by
  the source package.
- Coverage measures branches as well as statements and enforces an 80 percent
  repository floor.
- GitHub Actions runs the locked environment, format check, lint, strict typing,
  tests, coverage gate, and wheel/source-distribution build for pull requests,
  updates to `main`, and version tags. Pull requests additionally smoke-test
  the image; validated version tags publish it to GHCR.

These checks protect deterministic software contracts; they do not establish
model quality, production throughput, dependency vulnerability status, or
deployment readiness. Those require reviewed benchmark baselines and later
production tasks.

---

## Business Features To Add Later

- Per-user authentication and identity-provider integration
- Role-based access control
- Multi-tenancy
- Audit logging
- Cost monitoring
- Feedback collection
- Document versioning
- Data retention policies
- Compliance controls
