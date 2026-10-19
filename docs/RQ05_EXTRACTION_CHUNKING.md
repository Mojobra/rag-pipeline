# RQ-05: Extracted Text And Chunk Boundaries

This is a local synthetic-corpus audit, not a production parser or chunking
benchmark. The change is deliberately narrow: HTML now contributes normalized
visible text rather than raw markup. The recursive indexing policy, PDF parser,
and semantic boundary selection remain unchanged. The structure-aware policy
now falls back to recursive splitting for cleaned HTML without source tags.

## Inputs And Controls

| Setting | Value |
| --- | --- |
| Date and environment | 2026-10-19; Windows, Python 3.13.0, CPU, offline cached models |
| Code base | RQ-05 working tree from `develop` commit `579d4d09bc30702a39f12ab42ed87ac3a3372ab0` |
| Dependencies | `uv.lock` SHA-256 `68914E4A0080F81214F3661C984D44D1EB2DF2D4EE24D8BE0F4C2D4226675749` |
| Retrieval corpus | Five Asteria v2 Markdown policies; benchmark corpus SHA-256 `e4c9da40c104e45792d0242f2b962e3d0267f980f04e5475d4a15fde2c798949` |
| Retrieval labels | v2 source-anchor JSON, SHA-256 `0931ed4818b3456dc53ceac707a34de352ed0c7c7d9752e689f45b7b0e9deacb` |
| Answer labels | Paired v2 JSON, SHA-256 `3b750bc2d9279a0b76aa0a94b4f3b20fc099630bc86a33fd42a20e43e02aa8d3` |
| Dense model | `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 256-token maximum |
| Generation model | Local cached `google/flan-t5-small`; cache ref resolved to `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` but CLI revision was omitted (see limitation below) |
| Retrieval settings | Dense, `top-k=4`, gate disabled, no filter or reranker; default character size 1000 and recursive/structure overlap 200 |

All three isolated benchmarks used the same corpus and labels. The benchmark
artifacts were written only under ignored `.rag_data/rq05-benchmarks/`; their
temporary Qdrant indexes were created under ignored `.rag_data/rq05-work/`.

## Extraction And Token Audit

The committed `noisy-policy.html` fixture includes a policy heading, approval
table, exception in an aside, effective date in a footer, HTML entities, and
non-evidence head/style/script/navigation content. Tests assert the business
facts remain, unwanted text is absent, and every emitted chunk equals the
recorded slice of the normalized extracted document. A separate long Markdown
fixture checks bounded chunks and source offsets. A generated text-layer PDF
places right-column text first in its draw stream: pypdf returns that order,
while page metadata and chunk offsets remain correct. This demonstrates a
layout limitation rather than a safe automatic reordering rule.

Using the pinned MiniLM tokenizer with special tokens and truncation disabled:

| Input under default recursive size | Characters | Tokens | Chunks above 256 tokens |
| --- | ---: | ---: | ---: |
| Raw HTML fixture (pre-change behavior) | 896 | 308 | 1 of 1 |
| Normalized visible HTML fixture | 294 | 64 | 0 of 1 |

This is one deliberately noisy file. Token counts are model-specific; the
raw-markup over-limit result is a reason to remove non-evidence HTML before
embedding, not proof that cleaning solves every truncation case. The parser
does not evaluate CSS, read text from images, or preserve raw HTML byte
offsets. It retains asides and footers because they may contain policy rules;
it omits navigation, which could still contain useful text on some sites.

For the Asteria v2 corpus, the maximum chunk token counts were 177
(recursive), 144 (structure-aware), and 165 (semantic); **zero of ten chunks**
exceeded the model's 256-token limit for any strategy. This provides no
evidence for changing the default character splitter to a token-aware one.

## Matched Chunking Comparison

The three benchmark artifacts report the following single-pass observations.
All 17 retrieval cases had identical returned-count, relevant-count, and
first-relevant-rank diagnostics; no per-case gain or loss was hidden by the
aggregate values.

| Strategy | Chunks | Hit@4 | Precision@4 | Recall@4 | MRR@4 | Chunking | Dense embedding | Vector index | Index bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recursive | 10 | 1.000 | 0.265 | 1.000 | 1.000 | 0.00021 s | 0.11919 s | 0.06484 s | 66,228 |
| Structure-aware | 10 | 1.000 | 0.265 | 1.000 | 1.000 | 0.00023 s | 0.09934 s | 0.09884 s | 66,228 |
| Semantic | 10 | 1.000 | 0.265 | 1.000 | 1.000 | 0.30266 s | 0.09231 s | 0.04662 s | 66,228 |

The semantic timing includes an extra embedding pass for boundary decisions.
Its larger observed value is expected in kind, but one run is not a stable
latency estimate. Dense-embedding and Qdrant-index stage times vary across
runs and do not establish a strategy advantage. Answer token F1 was 0.253,
0.207, and 0.236 respectively,
despite identical retrieval metrics. That difference is not causal evidence
for a chunker preference; evidence-to-prompt effects need separate RQ-06
analysis. None of the v2 Markdown results establishes HTML or PDF retrieval
quality on representative business documents.

## Decision And Compatibility

- Use the standard-library HTML parser for `.html` and `.htm` with no new
  dependency. Keep headings, paragraphs, lists, table row/cell boundaries,
  asides, and footers; omit head, script, style, template, noscript, and nav.
- HTML `extractor` metadata changes from `text` to `html.parser`. Reindex
  existing HTML sources to replace obsolete markup chunks. Filters on
  `extractor=text` no longer include HTML. HTML citation offsets refer to
  normalized extracted text, not positions in the original HTML bytes.
- Keep recursive chunking as the normal indexing default. Structure-aware and
  semantic policies remain benchmark-only, with no model or collection-schema
  change. Cleaned HTML uses the recursive fallback under structure-aware mode;
  raw HTML documents retain the previous HTML-specific split behavior. New
  benchmark manifests name this distinction explicitly.
  Reevaluate on larger, independently reviewed labels before promotion.
- Do not silently reorder PDFs or add OCR in this task. Text-layer order can
  disagree with visual columns; an OCR/layout choice needs representative PDFs,
  page-level tests, and a separate provenance decision.

The first pinned generation benchmark attempt failed before indexing because
the current LangChain/Transformers adapter passed `revision` twice when
`--generation-model-revision` was supplied. The successful matched runs omitted
that flag and used the same offline cached model. This limits exact generator
reproducibility, but did not alter retrieval labels, index settings, or the
chunking comparison. It is an existing generation-adapter issue, not fixed by
RQ-05.

## Reproduce

From the repository root with locked dependencies installed and the model
artifacts cached:

```powershell
$env:HF_HUB_OFFLINE = '1'
$corpus = 'evaluation/datasets/asteria-policies-v2/documents'
$retrieval = 'evaluation/datasets/asteria-policies-v2/retrieval-v2.json'
$answers = 'evaluation/datasets/asteria-policies-v2/answers-v2.json'
$revision = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'

uv run rag-pipeline benchmark $corpus $retrieval $answers `
  --name rq05-recursive --chunking-strategy recursive `
  --model-revision $revision --search-mode dense --no-score-threshold `
  --top-k 4 --work-dir .rag_data/rq05-work `
  --output .rag_data/rq05-benchmarks/recursive.json
uv run rag-pipeline benchmark $corpus $retrieval $answers `
  --name rq05-structure-aware --chunking-strategy structure-aware `
  --model-revision $revision --search-mode dense --no-score-threshold `
  --top-k 4 --work-dir .rag_data/rq05-work `
  --output .rag_data/rq05-benchmarks/structure-aware.json
uv run rag-pipeline benchmark $corpus $retrieval $answers `
  --name rq05-semantic --chunking-strategy semantic `
  --model-revision $revision --search-mode dense --no-score-threshold `
  --top-k 4 --work-dir .rag_data/rq05-work `
  --output .rag_data/rq05-benchmarks/semantic.json
```

Use distinct output names for another run; the CLI preserves existing
artifacts unless `--overwrite` is explicit. The benchmark artifacts contain
queries and generated answers, so keep private-data runs outside Git. Remove
`HF_HUB_OFFLINE` only when an initial public model download is acceptable.
