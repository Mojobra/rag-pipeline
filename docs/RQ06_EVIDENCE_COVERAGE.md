# RQ-06: Evidence That Reaches The Prompt

This is a diagnostic on a small synthetic English corpus, not a faithfulness
score or a production context-budget recommendation. Retrieval, ranking,
packing, token limits, citations, and abstention behavior are unchanged.

## Diagnostic Contract

The isolated benchmark emits `results.evidence_coverage` only when every
answerable answer case has a matching retrieval case (same ID and query) with
exact source-text anchors from retrieval schema v2 or v3. V1 and unpaired
datasets retain their previous artifact shape. Each labeled anchor has one
state, even if overlapping chunks or duplicate hits contain it:

| State | Meaning |
| --- | --- |
| `not_retrieved` | No final retrieval result contains the metadata and complete text anchor. |
| `retrieved_not_packed` | A result contains the anchor, but no source prefix actually sent to the model contains it in full. |
| `packed_but_not_answered` | The anchor reached the prompt, but the model returned the configured abstention answer. |
| `packed_and_answered` | The anchor reached the prompt and the model returned a non-abstaining answer. This does **not** establish correctness. |

The evaluator uses the exact packed source prefixes retained by
`GeneratedAnswer`, not its citations' shortened display excerpts. It validates
prefix, retrieval, and citation alignment before reporting. Answer exact match
and token F1 remain separate; this report does not judge whether a particular
claim is supported. It adds no model inference or retrieval call, but keeps
the benchmark's per-case retrieval results in memory until evaluation ends.
Coverage is diagnostic only, not an allowlisted benchmark regression gate.

## Paired Local Measurement

Both runs used five Asteria v2 Markdown policies (10 chunks), 17 answerable
and four unsupported answer cases, with 19 labeled evidence anchors. Corpus
SHA-256 was `e4c9da40c104e45792d0242f2b962e3d0267f980f04e5475d4a15fde2c798949`;
retrieval labels SHA-256 was
`0931ed4818b3456dc53ceac707a34de352ed0c7c7d9752e689f45b7b0e9deacb`;
answer labels SHA-256 was
`3b750bc2d9279a0b76aa0a94b4f3b20fc099630bc86a33fd42a20e43e02aa8d3`.
The dense model was `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Both runs used dense search,
`top-k=4`, no score gate/filter/reranker, recursive chunks of 1000 characters
with 200 overlap, the same cached local `google/flan-t5-small` generator,
temperature 0, CPU, and fresh temporary Qdrant indexes (66,228 bytes each).
Only `max_context_characters` changed. Runs were single-pass on Windows/Python
3.13 with offline model caches. The generation revision was not pinned because
the current adapter fails to initialize when the revision flag is supplied;
artifacts correctly warn about this limitation. They also mark tracked source
as dirty because the task changes were uncommitted during measurement.

| Metric | Default cap 1200 | Tight cap 200 |
| --- | ---: | ---: |
| Anchors retrieved | 19/19 | 19/19 |
| Anchors fully packed | 19/19 | 6/19 |
| Retrieved but not packed | 0 | 13 |
| Packed but model abstained | 0 | 0 |
| Cases with every anchor packed | 17/17 | 6/17 |
| Answer exact match | 0.000 | 0.000 |
| Answer mean token F1 | 0.253 | 0.197 |

Every default-cap answerable case had `context_was_truncated=true`, yet all
its labeled anchors survived. The tighter cap deliberately creates a real
packing loss without changing retrieval. Per-case packed anchors:

| Case ID | 1200 | 200 |
| --- | ---: | ---: |
| `expense-receipt-threshold` | 1/1 | 0/1 |
| `expense-card-slip` | 1/1 | 0/1 |
| `expense-director-approval` | 1/1 | 1/1 |
| `expense-submission-deadline` | 1/1 | 0/1 |
| `travel-booking-channel` | 1/1 | 1/1 |
| `travel-flight-cabin` | 2/2 | 0/2 |
| `travel-hotel-limit` | 1/1 | 1/1 |
| `security-phishing-response` | 1/1 | 0/1 |
| `security-lost-device` | 1/1 | 1/1 |
| `security-policy-identifier` | 1/1 | 1/1 |
| `remote-work-days` | 1/1 | 1/1 |
| `remote-public-wifi` | 1/1 | 0/1 |
| `retention-support-tickets` | 1/1 | 0/1 |
| `retention-expense-records` | 1/1 | 0/1 |
| `retention-security-logs` | 1/1 | 0/1 |
| `retention-legal-hold` | 1/1 | 0/1 |
| `cross-policy-travel-meal-receipt` | 2/2 | 0/2 |

The default policy does not show an anchor-packing failure on this corpus, so
RQ-06 does not introduce overlap-aware deduplication, focused evidence windows,
or a new context default. Low answer EM/F1 despite full evidence availability
is a separate generation-quality signal, not proof of any particular cause.
The 200-character experiment demonstrates that the diagnostic identifies
losses under pressure; its lower F1 is one paired observation, not a calibrated
performance claim. Larger independently reviewed, multi-document datasets,
answer support review, and repeated measurements remain necessary.

## Reproduce

From the repository root with locked dependencies installed and model weights
already cached:

```powershell
$env:HF_HUB_OFFLINE = '1'
$corpus = 'evaluation/datasets/asteria-policies-v2/documents'
$retrieval = 'evaluation/datasets/asteria-policies-v2/retrieval-v2.json'
$answers = 'evaluation/datasets/asteria-policies-v2/answers-v2.json'
$revision = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'

uv run rag-pipeline benchmark $corpus $retrieval $answers `
  --name rq06-default --model-revision $revision `
  --search-mode dense --no-score-threshold --top-k 4 `
  --max-context-characters 1200 --work-dir .rag_data/rq06-work `
  --output .rag_data/rq06-benchmarks/default.json
uv run rag-pipeline benchmark $corpus $retrieval $answers `
  --name rq06-tight --model-revision $revision `
  --search-mode dense --no-score-threshold --top-k 4 `
  --max-context-characters 200 --work-dir .rag_data/rq06-work `
  --output .rag_data/rq06-benchmarks/tight.json
```

Choose new output names for repeated runs or pass `--overwrite` explicitly.
Artifacts contain queries and generated answers; private-data runs require
appropriate access and retention controls and should not be committed.
