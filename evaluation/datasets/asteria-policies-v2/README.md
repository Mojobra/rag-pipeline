# Asteria Policies v2

`asteria-policies-v2` keeps the synthetic English business-policy corpus from
v1 and replaces chunk-index retrieval judgments with schema-v2 source anchors.
This makes the same ground truth usable across recursive, structure-aware, and
semantic chunking experiments.

## Contents

- `documents/`: Five fictional Markdown policies used as the indexed corpus.
- `retrieval-v2.json`: Answerable queries with file metadata and exact content
  anchors that identify relevant source evidence independently of chunk index.
- `answers-v2.json`: Matching answerable queries plus unsupported questions for
  deterministic abstention checks.

## Labeling Contract

Each retrieval judgment contains a `metadata` selector and a case-sensitive
`content_contains` anchor. A returned chunk is relevant only when both match.
Anchors are short verbatim source fragments selected to remain inside one
sentence, so supported chunking strategies can change boundaries without
changing the meaning of the labels.

Automated tests require every anchor to occur exactly once in the source corpus,
resolve under the default configuration of all three benchmark chunking
strategies, align answerable cases with answer labels, and keep accepted-answer
vocabulary grounded in the matched evidence. The labels were manually
cross-checked but have not received an independent second-annotator review.

## Compare Chunking Strategies

Run each benchmark with the same corpus, retrieval labels, answer labels, model
settings, and final top-k. Only the chunking strategy and its explicit controls
should differ.

```powershell
uv run python -m rag_pipeline benchmark `
  evaluation/datasets/asteria-policies-v2/documents `
  evaluation/datasets/asteria-policies-v2/retrieval-v2.json `
  evaluation/datasets/asteria-policies-v2/answers-v2.json `
  --chunking-strategy recursive `
  --name asteria-v2-recursive `
  --output .rag_data/benchmarks/asteria-v2-recursive.json

uv run python -m rag_pipeline benchmark `
  evaluation/datasets/asteria-policies-v2/documents `
  evaluation/datasets/asteria-policies-v2/retrieval-v2.json `
  evaluation/datasets/asteria-policies-v2/answers-v2.json `
  --chunking-strategy structure-aware `
  --name asteria-v2-structure `
  --output .rag_data/benchmarks/asteria-v2-structure.json

uv run python -m rag_pipeline benchmark `
  evaluation/datasets/asteria-policies-v2/documents `
  evaluation/datasets/asteria-policies-v2/retrieval-v2.json `
  evaluation/datasets/asteria-policies-v2/answers-v2.json `
  --chunking-strategy semantic `
  --name asteria-v2-semantic `
  --output .rag_data/benchmarks/asteria-v2-semantic.json
```

Then compare saved artifacts with `compare-benchmarks`. Semantic chunking uses
the selected dense model for both boundary detection and final chunk vectors,
so it has additional inference cost. One small synthetic corpus cannot establish
a generally superior policy; review per-case metrics and repeat the experiment
on approved representative data before changing an ingestion default.

## Known Limits

- The corpus is small, synthetic, English-only, and Markdown-native.
- Exact anchors are deterministic but require a new dataset version when source
  wording changes.
- A candidate chunk size shorter than an anchor can split that judgment; dataset
  integrity tests cover the committed default strategy configurations.
- Binary labels do not express graded relevance or partial evidence quality.
- The semantic splitter uses punctuation and blank lines as candidate unit
  boundaries; multilingual and poorly punctuated text need separate validation.
