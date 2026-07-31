# Evaluation Assets

This directory contains versioned, repository-owned corpora and labels for
offline RAG evaluation. The assets are development benchmarks, not runtime
application data and not training data.

## Dataset Rules

- Keep source documents, retrieval labels, and answer labels in one versioned
  directory.
- Use synthetic or explicitly approved content. Never copy customer documents,
  credentials, personal data, or confidential material into a benchmark.
- Treat a published dataset version as immutable. Create a new version when
  corpus text, chunking assumptions, queries, relevance labels, or references
  change.
- Review labels against the complete corpus rather than only the current
  retriever's results. Otherwise existing misses cannot be discovered.
- Keep benchmark execution settings and result thresholds out of the dataset.
  The `benchmark` command records them in separate reproducible run artifacts.

## Available Datasets

| Dataset | Purpose |
| --- | --- |
| [`asteria-policies-v1`](datasets/asteria-policies-v1/README.md) | Synthetic English business policies with paired retrieval and answer labels. |
| [`asteria-policies-v2`](datasets/asteria-policies-v2/README.md) | The same policy domain with source-anchor relevance labels for chunking-strategy comparisons. |
