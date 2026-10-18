# RQ-03: First-Stage Score-Gate Calibration

This is a reproducible **local synthetic-corpus observation**, not a production
threshold recommendation or a latency benchmark. The comparison used the
retrieval-only Asteria v3 labels: 13 positive and four negative queries over
four fictional Markdown policies. Each collection contained four chunks.

## Run Configuration

| Setting | Value |
| --- | --- |
| Date and host | 2026-10-18; Windows, Python 3.13.0, uv 0.11.26 |
| Code base | RQ-03 task working tree based on `develop` commit `322a6396a6df4f03be839640213a423b2020b2c3` |
| Dependencies | `uv.lock` SHA-256 `68914E4A0080F81214F3661C984D44D1EB2DF2D4EE24D8BE0F4C2D4226675749` |
| Dense model | `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU |
| Sparse model | `Qdrant/bm25` for hybrid only, CPU; all 17 hybrid baseline cases returned RRF scores |
| Index | Fresh, separate dense and hybrid Qdrant collections under ignored `.rag_data/rq03-calibration` |
| Chunking | Recursive, 1000 characters with 200-character overlap |
| Query settings | `top_k=4`, no metadata filter, no reranker, baseline gate disabled; each gate reran every query |
| Installed adapters | `langchain-qdrant` 1.1.0, `qdrant-client` 1.18.0, `sentence-transformers` 5.6.0, `fastembed` 0.8.0 |

Input SHA-256 fingerprints:

| File under `evaluation/datasets/asteria-policies-v3/` | SHA-256 |
| --- | --- |
| `retrieval-v3.json` | `B483BF21636685346461E66F1F581AB36A03E17C708B1B3FE9D4B78EF028FFDE` |
| `documents/business-travel.md` | `022244FCF855AA138578AB653E9DDF6A3C640A00CD4A04546EDE2AA6ED07D69E` |
| `documents/expense-archived.md` | `950CC98D5FF20FEE1B63B9090CCC64AF683F9D2D96F8D2476C2432E4EEC16C57` |
| `documents/expense-current.md` | `BDF15A29D82CD70CEB2AB14FB1F7E622B4F5D090D97A9E5AEF72967C29FDE8BF` |
| `documents/information-security.md` | `06CCEE044CAFF0FB0FA0EFC9759C9B244752C4B4E86B25E9763B2BD938F58F48` |

## Results At Top-K 4

All positive metrics are macro averages over 13 positive cases. Negative
no-hit is the fraction of four unsupported questions returning zero chunks.
Precision divides matching chunks by the fixed `k`, even when a gate reduces
the number of returned chunks.

| Mode | Gate | Positive Hit@4 | Precision@4 | Recall@4 | MRR@4 | Negative no-hit@4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense, cosine | Disabled | 1.000 | 0.288 | 1.000 | 0.962 | 0.000 |
| Dense, cosine | 0.2 | 1.000 | 0.288 | 1.000 | 0.962 | 0.000 |
| Dense, cosine | 0.3 | 1.000 | 0.288 | 1.000 | 0.962 | 0.500 |
| Dense, cosine | 0.35 | 1.000 | 0.269 | 0.962 | 0.962 | 0.500 |
| Dense, cosine | 0.4 | 0.923 | 0.250 | 0.923 | 0.885 | 0.500 |
| Hybrid, RRF | Disabled | 1.000 | 0.288 | 1.000 | 0.962 | 0.000 |
| Hybrid, RRF | 0.2 | 1.000 | 0.288 | 1.000 | 0.962 | 0.000 |
| Hybrid, RRF | 0.4 | 1.000 | 0.288 | 1.000 | 0.962 | 0.000 |
| Hybrid, RRF | 0.6 | 1.000 | 0.288 | 1.000 | 0.962 | 0.000 |
| Hybrid, RRF | 0.7 | 0.923 | 0.250 | 0.885 | 0.923 | 0.000 |

**Per-case findings:** Dense `0.3` removed all results for
`unsupported-parental-leave` and `unsupported-hotel-cap` without losing a
labeled positive selector. Dense `0.35` lost one of two selectors for
`travel-meal-cross-policy`; at `0.4` it lost both. Hybrid `0.7` lost evidence
for `current-expense-identifier` and `superseded-threshold` without improving
the strict negative no-hit rate. The report JSON retains all per-case counts,
observed score kinds, and selector-index deltas.

An exploratory `top_k=2` rerun showed the same Hit, Recall, and negative
no-hit outcomes for the disabled baseline, dense `0.2`/`0.3`, and hybrid
`0.2`/`0.6`/`0.7`; the table above remains the default-cutoff result.

## Decision And Limits

Keep existing CLI defaults. Dense `0.3` is a **candidate for wider validation**,
not a new default: this small set has one chunk per source, all positive cases
are covered at the baseline cutoff, and the strict negative label counts any
topically related chunk as unwanted. Hybrid values occupy a different RRF
scale; none of the tested gates improved no-hit without an evident risk to
positive evidence. The v3 set has no paired answer labels, so these retrieval
results do not establish answer accuracy or abstention behavior. Single local
runs provide no latency distribution, and no index-size comparison was made.
Before changing a default, repeat with reviewed representative documents,
more negative cases, matched answer evaluation, and explicit business costs
for false positives versus missed evidence.

## Reproduce

From the repository root, with the locked environment installed:

```powershell
$env:HF_HUB_OFFLINE = '1'
$revision = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
uv run rag-pipeline index evaluation/datasets/asteria-policies-v3/documents `
  --store-path .rag_data/rq03-calibration --collection-name asteria_v3_dense `
  --model-revision $revision
uv run rag-pipeline index evaluation/datasets/asteria-policies-v3/documents `
  --store-path .rag_data/rq03-calibration --collection-name asteria_v3_hybrid `
  --search-mode hybrid --model-revision $revision
uv run rag-pipeline evaluate-retrieval `
  evaluation/datasets/asteria-policies-v3/retrieval-v3.json `
  --store-path .rag_data/rq03-calibration --collection-name asteria_v3_dense `
  --search-mode dense --model-revision $revision --top-k 4 `
  --no-score-threshold --compare-score-threshold 0.2 `
  --compare-score-threshold 0.3 --compare-score-threshold 0.35 `
  --compare-score-threshold 0.4 --output-format json
uv run rag-pipeline evaluate-retrieval `
  evaluation/datasets/asteria-policies-v3/retrieval-v3.json `
  --store-path .rag_data/rq03-calibration --collection-name asteria_v3_hybrid `
  --search-mode hybrid --model-revision $revision --top-k 4 `
  --no-score-threshold --compare-score-threshold 0.2 `
  --compare-score-threshold 0.4 --compare-score-threshold 0.6 `
  --compare-score-threshold 0.7 --output-format json
```

`HF_HUB_OFFLINE=1` requires the pinned dense model and BM25 sparse model to be
cached already. Without them, remove the offline setting and allow the first
run to fetch public model files. Use fresh collection names for a new
experimental snapshot; never treat an existing mixed-source collection as an
isolated benchmark input.
