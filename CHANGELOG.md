# Changelog

Notable project milestones are recorded here. Versions follow Semantic
Versioning while the package remains in initial development.

## [0.4.0] - Unreleased

### Added

- A read-only FastAPI service for health checks, ranked retrieval, and cited
  answers over an existing local collection, with bounded request schemas.
- Shared bearer-token authentication, structured request events with generated
  IDs, and a protected endpoint for process-local Prometheus metrics.
- An opt-in private source-access policy with distinct query-principal tokens,
  exact-source grants, and Qdrant filtering before top-k selection.
- Policy version 2 binds each query principal and grant to one tenant and routes
  queries to a separate local Qdrant directory and collection.
- A non-root local Docker image for CLI indexing and API queries, plus pull
  request container checks and version-tagged image publication in CI.
- Bounded manual CLI imports for SharePoint drives, Confluence Cloud spaces,
  Google Drive folders, and Slack and Teams channels with replies.
- A sanitized `.env.example` and setup/security guides for the API, container,
  and manual source integrations.

### Changed

- The local HTTP API now requires `RAG_API_BEARER_TOKEN` at startup and a matching
  `Authorization: Bearer` header for retrieval and answer requests. Without an
  access policy the shared token still reads all sources; with a policy, query
  tokens are separate and the shared token protects `/metrics` only.
- CI quality checks now run on Python 3.11, 3.12, and 3.13. The local Docker
  image remains based on Python 3.11.

### Known Limitations

- Remote imports are manual snapshots. Source-system ACLs and remote deletions
  are not synchronized; identity verification and automatic refresh are absent.
- Tenant separation is limited to server-routed local stores in one trusted
  process; there is no OS-level isolation, rate limiting, or audit trail.
- The container CI smoke test checks CLI startup and API liveness, not a
  model-backed query. Hosting, TLS, and production operations are not supplied.

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
