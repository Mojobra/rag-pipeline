# Asteria Policies v1

`asteria-policies-v1` is a compact synthetic corpus for repeatable local RAG
evaluation. Asteria Works is fictional; the policies, identifiers, people, and
rules in this directory were created for this project.

## Contents

- `documents/`: Five Markdown policies used as the indexed corpus.
- `retrieval-v1.json`: Answerable queries with exact relevant-chunk selectors.
- `answers-v1.json`: The same answerable queries with accepted references, plus
  unsupported questions that should trigger deterministic abstention.

The dataset covers direct facts, paraphrases, policy identifiers, numeric and
time constraints, similar vocabulary across documents, multi-document
questions, and unsupported requests.

## Labeling Contract

Retrieval selectors use `file_name` plus zero-based `chunk_index`. They are
portable across repository locations and intentionally avoid absolute `source`
paths. Labels target chunks produced by the current defaults:

```text
chunk_size = 1000 characters
chunk_overlap = 200 characters
```

Automated tests load and chunk the committed corpus, require every selector to
match exactly one chunk, align every answerable case with a retrieval case, and
check that accepted-answer vocabulary occurs in the labeled evidence.

The labels were manually cross-checked against the complete synthetic corpus.
They have not yet received independent second-annotator review, so this is a
development baseline rather than production ground truth.

## Run Locally

Create a fresh collection for the dataset:

```powershell
uv run python -m rag_pipeline index `
  evaluation/datasets/asteria-policies-v1/documents `
  --collection-name asteria_policies_v1 `
  --chunk-size 1000 `
  --chunk-overlap 200
```

Evaluate retrieval:

```powershell
uv run python -m rag_pipeline evaluate-retrieval `
  evaluation/datasets/asteria-policies-v1/retrieval-v1.json `
  --collection-name asteria_policies_v1 `
  --top-k 4
```

Evaluate full answers:

```powershell
uv run python -m rag_pipeline evaluate-answer `
  evaluation/datasets/asteria-policies-v1/answers-v1.json `
  --collection-name asteria_policies_v1 `
  --top-k 4
```

Use the same embedding model and search mode for indexing and evaluation.
Changing chunking settings invalidates the current selectors. Use a new
collection name for a clean comparison because indexing updates matching chunk
IDs but does not define a benchmark-run lifecycle.

Use [`asteria-policies-v2`](../asteria-policies-v2/README.md) for recursive,
structure-aware, or semantic chunking comparisons. Its schema-v2 relevance
anchors identify source evidence without depending on chunk indices.

Run an isolated benchmark without creating or reusing a persistent collection:

```powershell
uv run python -m rag_pipeline benchmark `
  evaluation/datasets/asteria-policies-v1/documents `
  evaluation/datasets/asteria-policies-v1/retrieval-v1.json `
  evaluation/datasets/asteria-policies-v1/answers-v1.json `
  --name asteria-dense-local-v1 `
  --output .rag_data/benchmarks/asteria-dense-local-v1.json
```

The benchmark command rebuilds the index from this exact corpus, records hashes,
configuration, quality, latency, and storage, then removes the temporary index.

## Known Limits

- The corpus is small, synthetic, English-only, and text-native.
- It does not cover OCR, tables, scans, multilingual queries, access control,
  tenant isolation, or document updates.
- Binary retrieval labels do not express degrees of relevance.
- Reference answers support deterministic lexical scoring, not semantic
  faithfulness judgment.
- Benchmark timings use one local pass; capacity and service-level analysis need
  controlled warmups, repetitions, concurrency, and production-like hardware.
