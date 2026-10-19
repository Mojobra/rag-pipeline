# RQ-04: Candidate Width And Reranking Comparison

This is a local observation on a **four-chunk synthetic corpus**, not evidence
for a production candidate width or a reranker-quality claim. The comparison
uses 13 positive and four unsupported Asteria v3 queries. No defaults, fusion
algorithm, or model were changed.

## Configuration

| Setting | Value |
| --- | --- |
| Date and host | 2026-10-19; Windows, Python 3.13.0, uv 0.11.26, CPU |
| Code base | RQ-04 task working tree based on `develop` commit `830339bd8897325464598109ebf6e64748a0df36` |
| Dependencies | `uv.lock` SHA-256 `68914E4A0080F81214F3661C984D44D1EB2DF2D4226675749` |
| Labels | `evaluation/datasets/asteria-policies-v3/retrieval-v3.json`, SHA-256 `B483BF21636685346461E66F1F581AB36A03E17C708B1B3FE9D4B78EF028FFDE` |
| Corpus | Four v3 Markdown policies, with source hashes matching the [RQ-03 record](RQ03_SCORE_GATE_CALIBRATION.md#run-configuration) |
| Dense model | `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Sparse model | `Qdrant/bm25` for hybrid; all observed hybrid first-stage scores were RRF |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` at `c5ee24cb16019beea0893ab7796b1df96625c6b8`, batch 16, max pair length 512 |
| Index | Fresh dense and hybrid collections under ignored `.rag_data/rq04-comparison`; four points each |
| Chunking | Recursive, 1000 characters with 200-character overlap |
| Retrieval | No score gate or metadata filter; `top-k=2` with widths 2/3/4, then diagnostic `top-k=1` with widths 1/2/4 |

The labels and document fingerprints are immutable within the v3 snapshot.
First-stage retrieval was rerun for every width; the unreranked and reranked
final rankings used exactly that width's candidates. Pool precision uses its
own cutoff, so it cannot be compared directly with final precision at a
different cutoff.

## Results

Positive metrics are macro averages over 13 cases. All four negative queries
returned at least one chunk in every tested setting (negative no-hit rate
`0.000`). Dense and hybrid returned the same aggregate quality values here.

| Final cutoff | Candidate width | Pool Hit / Recall / MRR | Unreranked final Hit / Recall / MRR | Reranked final Hit / Recall / MRR |
| ---: | ---: | --- | --- | --- |
| 2 | 2, 3, or 4 | 1.000 / 1.000 / 0.962 | 1.000 / 1.000 / 0.962 | 1.000 / 1.000 / 0.962 |
| 1 | 1 | 0.923 / 0.846 / 0.923 | 0.923 / 0.846 / 0.923 | 0.923 / 0.846 / 0.923 |
| 1 | 2 or 4 | 1.000 / 1.000 / 0.962 | 0.923 / 0.846 / 0.923 | 0.923 / 0.846 / 0.923 |

At `top-k=1`, pool precision is `0.923` at width 1, `0.577` at width 2,
and `0.288` at width 4; final precision remains `0.923` in both final rankings.
The v3 labels count a chunk matching a source anchor as relevant, and precision
uses the fixed cutoff as denominator.

**Per-query diagnosis:** At width 1, `current-expense-identifier`,
`superseded-threshold`, and `travel-meal-cross-policy` miss selector index 0
in the first-stage pool. At widths 2 and 4, every positive selector is in the
pool, but the final one-chunk selection still misses index 0 for the first two
cases and index 1 for the cross-policy case. The cross-policy case requires two
sources, so a one-chunk final cutoff cannot cover both selectors. The
cross-encoder changed the selected evidence for `travel-meal-cross-policy` at
widths 2 and 4: it gained selector 0 and lost selector 1 versus the unreranked
one-chunk result. Aggregate recall was unchanged, but the evidence was not
identical. At `top-k=2`, no labeled positive selector was missed.

Observed single-pass cumulative retrieval time over 17 queries was about
`0.09-0.13 s` per width for dense and `0.10-0.12 s` for hybrid. Reranking all
cases took roughly `0.27-0.77 s` per width; the wider pools generally required
more scoring work. These timings include possible lazy initialization,
filesystem cache effects, and process startup is excluded. They are not
repeated latency distributions or a sound dense-versus-hybrid performance
comparison. The two tiny local collection directories occupied 28,672 bytes
(dense) and 36,864 bytes (hybrid) at measurement time; filesystem allocation
and Qdrant overhead dominate at this size. Peak memory was not measured.

## Decision And Limits

Keep the current defaults (`top-k=4`, reranking optional, candidate width 20
when enabled). Widths above four cannot add candidates on this corpus, and
the diagnostic one-chunk cutoff is below the two-source requirement of one
label. The equal dense/hybrid results do not demonstrate equal performance on
business documents. No fusion or diversity logic is warranted from this
evidence. Before promoting a change, compare reviewed, representative corpora
with more chunks and difficult negatives, repeat timing measurements, inspect
answer quality, and agree on the cost of misses versus additional inference.

## Reproduce

From the repository root, with the locked environment installed:

```powershell
$env:HF_HUB_OFFLINE = '1'
$denseRevision = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
$rerankerRevision = 'c5ee24cb16019beea0893ab7796b1df96625c6b8'
$corpus = 'evaluation/datasets/asteria-policies-v3/documents'
$labels = 'evaluation/datasets/asteria-policies-v3/retrieval-v3.json'
$store = '.rag_data/rq04-comparison'

uv run rag-pipeline index $corpus --store-path $store `
  --collection-name asteria_v3_dense --model-revision $denseRevision
uv run rag-pipeline index $corpus --store-path $store `
  --collection-name asteria_v3_hybrid --search-mode hybrid `
  --model-revision $denseRevision

uv run rag-pipeline evaluate-retrieval $labels --store-path $store `
  --collection-name asteria_v3_dense --search-mode dense `
  --model-revision $denseRevision --reranker-model-revision $rerankerRevision `
  --no-score-threshold --rerank --top-k 2 --candidate-k 2 `
  --compare-candidate-k 3 --compare-candidate-k 4 --output-format json
uv run rag-pipeline evaluate-retrieval $labels --store-path $store `
  --collection-name asteria_v3_hybrid --search-mode hybrid `
  --model-revision $denseRevision --reranker-model-revision $rerankerRevision `
  --no-score-threshold --rerank --top-k 2 --candidate-k 2 `
  --compare-candidate-k 3 --compare-candidate-k 4 --output-format json
```

Repeat the last two commands with `--top-k 1 --candidate-k 1` and
`--compare-candidate-k 2 --compare-candidate-k 4` for the diagnostic cutoff.
`HF_HUB_OFFLINE=1` requires both pinned Hugging Face models and BM25 to be
cached; otherwise allow the initial public model download. Use fresh
collection names for an independently measured index snapshot. These commands
do not persist JSON reports automatically; redirect output to a private
location when evaluating confidential documents.
