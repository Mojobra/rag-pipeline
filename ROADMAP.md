# RAG Learning Roadmap

## Phase 1: Foundations

Task 01 - Create project skeleton
Task 02 - Document ingestion
Task 03 - Text extraction
Task 04 - Chunking
Task 05 - Embeddings
Task 06 - Vector database
Task 07 - Retrieval
Task 08 - Answer generation
Task 09 - Citations

## Phase 2: Quality

Task 10 - Chunking experiments
Task 11 - Metadata filters
Task 12 - Hybrid search
Task 13 - Reranking
Task 14 - Prompt optimization

## Phase 3: Evaluation

Task 15 - Retrieval evaluation
Task 16 - Answer evaluation
Task 17 - Test datasets: versioned synthetic corpus and paired labels
Task 18 - Benchmarking: isolated indexes, manifests, latency, comparisons, and gates

Completed quality follow-up - Semantic chunking experiment

The isolated benchmark can compare bounded semantic and structure-aware
chunking with the recursive baseline using chunk-index-independent schema-v2
relevance anchors. Recursive chunking remains the default outside experiments.

## Retrieval Quality Follow-Up (Complete)

The [retrieval quality implementation plan](docs/RETRIEVAL_QUALITY_PLAN.md)
records six separately reviewable tasks covering index freshness, evaluation
coverage, score calibration, candidate ranking, source text and chunking, and
evidence packing. RQ-01 through RQ-06 are integrated into `develop`; the
original roadmap task numbers below remain unchanged.

## Phase 4: Production

Task 19 - FastAPI service (local read-only API implemented)
Task 20 - Authentication (shared bearer token for local HTTP queries implemented)
Task 21 - Logging (privacy-conscious structured HTTP request events implemented)
Task 22 - Monitoring (protected process-local API and RAG outcome metrics implemented)
Task 23 - Dockerization (local CLI/API image and persistent-volume workflow implemented)
Task 24 - Deployment CI/CD (PR container checks and release-tag image publishing
implemented; hosting a service remains future work)

## Phase 5: Enterprise

Task 25 - SharePoint integration (bounded manual Graph drive import implemented;
ACL sync and remote deletion are not included)
Task 26 - Confluence integration (bounded manual Cloud space page import
implemented; ACL sync and remote deletion are not included)
Task 27 - Google Drive integration (bounded manual folder import implemented;
Google Drive ACL sync and remote deletion are not included)
Task 28 - Slack / Teams integration (bounded manual channel and thread imports
implemented; ACL sync, remote deletion, and incremental refresh are not included)
Task 29 - Permissions-aware retrieval
Task 30 - Multi-tenant architecture
