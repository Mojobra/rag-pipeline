# RAG Pipeline Architecture

## High-Level Components

1. Document Ingestion
2. Document Parsing
3. Document Cleaning
4. Chunking
5. Environment-Backed Model Profile Resolution
6. Local or Hosted Embedding Generation
7. Vector Database
8. Hybrid Retrieval Layer (dense + BM25 sparse with RRF)
9. Local Cross-Encoder Reranking Layer
10. Versioned Prompt Construction and Token-Aware Evidence Packing
11. Local or Hosted LLM Generation
12. Citation System
13. Evaluation Framework
14. Monitoring & Observability
15. API Layer
16. Frontend/UI
17. Deployment Infrastructure

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
- Ranked chunks are packed into numbered evidence blocks under character and
  model-specific application limits. Local models use exact tokenizer counts;
  hosted profiles use a conservative local byte estimate.
- Retrieved text is explicitly treated as untrusted data, and unsupported or
  conflicting evidence maps to one deterministic abstention response.
- The answer result records both model and prompt identifiers; source citations
  are constructed outside the language model from validated chunk provenance.

Evidence delimiters are prompt structure, not a security or authorization
boundary. Answer faithfulness and abstention still require dataset-based
evaluation before production use.

---

## Current Model Provider Contract

- `--model gemini|openai|claude` selects only hosted answer generation;
  omitting it preserves the local generation path.
- `--embed-model gemini|openai|claude` selects configured embeddings, while a
  raw Hugging Face model ID selects local embeddings. Omitting it uses
  `DEFAULT_LOCAL_EMBEDDING_MODEL`.
- Each selector loads and validates only its role-specific model setting and
  credential, so generation experiments cannot implicitly change retrieval.
- Process environment values override `.env`; API keys are passed directly to
  LangChain integrations and excluded from profile representations and errors.
- Gemini and OpenAI offer hosted embedding adapters. `--embed-model claude`
  uses the local Hugging Face model named by `CLAUDE_EMBED`, because Anthropic
  exposes no embeddings API.
- Indexing and query commands must use the same embedding selection. Existing
  Qdrant compatibility checks reject changed model identities or dimensions.
- Generation providers may be changed without rebuilding a compatible vector
  collection; changing the embedding selection requires reindexing.

Hosted profiles introduce data egress, usage cost, network latency, rate
limits, and provider availability risk. Timeout, retry, cost-control, and
secret-manager policies remain production-hardening work.

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
