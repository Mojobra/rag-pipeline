# Changelog

Notable project milestones are recorded here. Versions follow Semantic
Versioning while the package remains in initial development.

## Unreleased

### Added

- Local read-only FastAPI routes for health, ranked retrieval, and grounded
  cited answers over an existing CLI-indexed collection.
- Bounded HTTP schemas and offline API lifecycle/error tests. Uploads and
  network deployment remain future tasks.
- Shared bearer-token authentication on the two query routes, with fail-closed
  startup and offline authentication tests. User identity, authorization, and
  rate limiting remain future work.

### Changed

- The local HTTP API now requires `RAG_API_BEARER_TOKEN` at startup and a matching
  `Authorization: Bearer` header for retrieval and answer requests. CLI behavior
  is unchanged.

## [0.3.0] - 2026-10-19

### Added

- Source-scoped index synchronization with pruning support and stable point
  identities for unchanged content.
- Expanded retrieval datasets and diagnostics, including negative cases,
  source anchors, score-gate calibration, and candidate/reranking comparisons.
- HTML text-cleaning and chunk-boundary audits with measured recursive,
  structure-aware, and semantic-chunking comparisons.
- Optional benchmark diagnostics that track labeled evidence from retrieval
  through exact prompt packing to answer or abstention.

### Changed

- Updated the retrieval-quality plan after RQ-01 through RQ-06 were integrated
  into `develop`. The production-phase roadmap remains planned.

### Known Limitations

- Evidence coverage measures whether source anchors reached the prompt, not
  whether generated claims are faithful or factually correct.
- The supported interface remains a local CLI; hosted providers, a service,
  authentication, and deployment are not included in this release.

## [0.2.0] - 2026-08-12

First versioned portfolio milestone for the production-minded local RAG pipeline.

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
