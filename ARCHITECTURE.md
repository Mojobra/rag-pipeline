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
12. Evaluation Framework
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
boundary. Answer faithfulness and abstention still require dataset-based
evaluation before production use.

---

## Business Features To Add Later

- User authentication
- Role-based access control
- Metadata filtering
- Multi-tenancy
- Audit logging
- Cost monitoring
- Feedback collection
- Document versioning
- Data retention policies
- Compliance controls
