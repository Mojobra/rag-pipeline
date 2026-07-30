# RAG Pipeline Architecture

## High-Level Components

1. Document Ingestion
2. Document Parsing
3. Document Cleaning
4. Chunking
5. Embedding Generation
6. Vector Database
7. Hybrid Retrieval Layer (dense + BM25 sparse with RRF)
8. Local Cross-Encoder Reranking Layer
9. Versioned Prompt Construction and Token-Aware Evidence Packing
10. LLM Generation
11. Citation System
12. Versioned Retrieval and Answer Evaluation Framework with Test Datasets
13. Isolated Benchmark Runner, Artifacts, Comparisons, and Regression Gates
14. Modular CLI Adapter
15. Monitoring & Observability
16. API Layer
17. Frontend/UI
18. Deployment Infrastructure

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

## Current CLI Adapter Contract

- `rag_pipeline.__main__` remains the stable module entry point and re-exports
  `main` and `build_parser`, but it owns no command behavior.
- `rag_pipeline.cli.app` constructs the root parser and dispatches through an
  explicit handler attached by each subcommand. There is no central conditional
  covering every command.
- Shared option groups define one embedding, chunking, retrieval, reranking,
  generation, and storage vocabulary across commands.
- A dedicated configuration boundary translates dynamic argparse values into
  the existing validated stage dataclasses before model or storage side effects.
- Command handlers are grouped by document, indexing, query, evaluation, and
  benchmark responsibilities. Domain services remain outside the CLI package.
- Provider factories and Qdrant construction stay inside command handlers so
  parser creation and help output do not initialize models, download files, or
  open a database.
- Terminal-specific retrieval and answer rendering is isolated from service
  orchestration and can be tested without provider calls.

This is an adapter refactor, not a new application-service layer. Task 19 should
introduce transport-neutral use cases only where both CLI and FastAPI genuinely
need the same lifecycle orchestration; the API must not import CLI handlers.

---

## Current Generation Contract

- LangChain composes the `grounded-v2` prompt with the configured language model
  and string output parser.
- Ranked chunks are packed into numbered evidence blocks under exact character
  and tokenizer limits.
- Retrieved text is explicitly treated as untrusted data, and unsupported or
  conflicting evidence maps to one deterministic abstention response.
- The answer result records both model and prompt identifiers; source citations
  are constructed outside the language model from validated chunk provenance.

Evidence delimiters are prompt structure, not a security or authorization
boundary. Deterministic reference and abstention evaluation is implemented, but
semantic faithfulness still requires calibrated human or model-based judgment
before production use.

---

## Current Retrieval Evaluation Contract

- A strict schema-v1 JSON file supplies named query cases and one or more binary
  relevance judgments expressed as exact document-metadata selectors.
- Evaluation runs the same LangChain/Qdrant dense or hybrid retriever, metadata
  filters, score gate, and optional cross-encoder reranker used by interactive
  commands; it never invokes generation or mutates the collection.
- Reports include per-case Hit@k, Precision@k, Recall@k, and reciprocal rank,
  plus macro averages that give every query equal weight.
- Table output supports local diagnosis and JSON output supports saved
  comparisons.
- The committed synthetic policy corpus supplies 17 manually cross-checked
  retrieval cases, including one multi-document judgment.

Exact metadata selectors are transparent but depend on stable provenance.
Portable business datasets should eventually use immutable document and chunk
version identifiers instead of filenames and chunk positions. The benchmark
layer now captures latency and immutable input fingerprints around these
metrics.

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
- Retrieval and answer datasets share IDs and query text for all 17 answerable
  cases; the answer dataset adds four unsupported questions.
- Retrieval labels use `file_name` and zero-based `chunk_index` against the
  documented 1000-character size and 200-character overlap defaults.
- Integrity tests reproduce the ten expected chunks, require each selector to
  match exactly one chunk, align the paired schemas, and check reference
  vocabulary against labeled evidence.
- Dataset versions are repository artifacts and should be treated as immutable
  after benchmark results depend on them.

Synthetic data is safe, portable, and reviewable, but it cannot represent the
full language, layout, security, or ambiguity of production documents. The
labels were manually cross-checked but do not yet have independent annotator
agreement. Benchmark settings and thresholds remain separate from these
ground-truth assets so dataset versions do not encode a preferred pipeline.

---

## Current Benchmark Contract

- `benchmark` starts from one explicit corpus and the paired schema-v1
  evaluation files, then builds a fresh temporary Qdrant collection. The index
  is closed and deleted after its storage size is recorded.
- Corpus relative paths and bytes plus both dataset files receive SHA-256
  fingerprints. Artifacts omit absolute corpus, cache, output, and work paths.
- One manifest records chunking, dense and optional sparse models, retrieval,
  filters, reranking, generation, prompt/token limits, devices, batching, Git
  state, package versions, CPU/platform data, and available CUDA devices.
- Models are initialized once. Reports separate model-load, extraction,
  chunking, embedding, index, retrieval-evaluation, and answer-evaluation stage
  durations, with ordered per-case retrieval and end-to-end answer latency.
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
