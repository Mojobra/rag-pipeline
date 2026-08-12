# Changelog

Notable project milestones are recorded here. Versions follow Semantic
Versioning while the package remains in initial development.

## [0.2.0] - 2026-08-12

First tagged portfolio milestone for the production-minded local RAG pipeline.

### Added

- Multi-format ingestion with provenance-preserving recursive, structure-aware,
  and bounded semantic chunking workflows.
- Dense and optional BM25 hybrid retrieval in Qdrant, metadata filtering, score
  gates, and cross-encoder reranking.
- Token-bounded grounded generation with deterministic citations and abstention.
- Versioned retrieval and answer datasets, reproducible benchmarks, artifact
  comparison, and regression thresholds.
- Deterministic offline tests, strict typing, linting, branch coverage, package
  builds, and pull-request CI.

### Changed

- Organized implementations into canonical feature packages and removed the
  temporary root-level compatibility facades before the production phase.
- Made installed distribution metadata the runtime version source used by the
  Python package, CLI, and benchmark manifests.

### Known Limitations

- The supported interface is currently a local CLI; FastAPI, authentication,
  observability, containerization, and deployment remain roadmap work.
- Local default models prioritize accessibility over production answer quality
  and throughput.
