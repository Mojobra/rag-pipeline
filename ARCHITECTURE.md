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
13. Monitoring & Observability
14. API Layer
15. Frontend/UI
16. Deployment Infrastructure

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
version identifiers instead of filenames and chunk positions. Latency capture,
benchmark manifests, and thresholds remain Task 18 work.

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
answerable and four expected-abstention cases. Benchmark manifests, independent
annotation, human review, and a calibrated NLI or LLM judge remain future work.

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
agreement. Task 18 will define run manifests and regression thresholds without
changing these ground-truth assets.

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
