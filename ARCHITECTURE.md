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
14. Monitoring & Observability (planned)
15. API Layer (planned)
16. Frontend/UI (planned)
17. Deployment Infrastructure (planned)

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
- `rag_pipeline.application` owns transport-neutral indexing and retrieval use
  cases that coordinate multiple pipeline stages.
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
  benchmark responsibilities. They delegate reusable indexing and retrieval
  lifecycles to `rag_pipeline.application`.
- Application use cases own cross-stage invariants, provider lifecycle, and
  orchestration. They accept explicit dependencies where useful for deterministic
  tests and do not import argparse or terminal output helpers.
- Provider factories remain lazy, so parser creation and help output do not
  initialize models, download files, or open a database. Reranking models are
  not initialized when first-stage retrieval returns no candidates.
- Terminal-specific retrieval and answer rendering is isolated from service
  orchestration and can be tested without provider calls.

The application layer is intentionally small. It coordinates existing domain
services rather than replacing their validation or algorithms. A future API can
reuse these use cases without importing CLI handlers; API-specific request,
response, authentication, and error-mapping concerns remain future work.

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
  Markdown and HTML use LangChain's language-specific separator priorities;
  other formats use the baseline separator order.
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

---

## Current Retrieval Evaluation Contract

- Strict schema-v1 JSON remains supported for exact document-metadata selectors.
  Schema v2 combines exact metadata with a case-sensitive source-content anchor,
  allowing one judgment to survive ordinary changes in chunk boundaries.
- Evaluation runs the same LangChain/Qdrant dense or hybrid retriever, metadata
  filters, score gate, and optional cross-encoder reranker used by interactive
  commands; it never invokes generation or mutates the collection.
- Reports include per-case Hit@k, Precision@k, Recall@k, and reciprocal rank,
  plus macro averages that give every query equal weight.
- Table output supports local diagnosis and JSON output supports saved
  comparisons.
- Both committed synthetic policy dataset versions supply 17 manually
  cross-checked retrieval cases, including one multi-document judgment.

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
  then builds a fresh temporary Qdrant collection. Retrieval schemas v1 and v2
  are accepted, but v2 is required for valid boundary-strategy comparisons. The
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
  tests, coverage gate, and wheel/source-distribution build for pull requests
  and updates to `main`.

These checks protect deterministic software contracts; they do not establish
model quality, production throughput, dependency vulnerability status, or
deployment readiness. Those require reviewed benchmark baselines and later
production tasks.

---

## Business Features To Add Later

- User authentication
- Role-based access control
- Multi-tenancy
- Audit logging
- Cost monitoring
- Feedback collection
- Document versioning
- Data retention policies
- Compliance controls
