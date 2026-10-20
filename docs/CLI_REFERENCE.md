# CLI Reference

Detailed workflows and contracts for the local CLI. Start with the
[Quick Start](../README.md#quick-start) for installation and a minimal example.
Run all commands from the repository root; multiline commands use PowerShell
continuation syntax. The schemas shown below are JSON examples, not
additional files installed by the application.

For the optional `index-sharepoint` command, see the
[SharePoint setup and security contract](SHAREPOINT.md). It shares `index`'s
local chunking, embedding, and store settings but reads one Graph drive via
environment-supplied app credentials, defaults to `sharepoint_documents`, and
does not import source permissions.
For `index-confluence`, see the
[Confluence setup and security contract](CONFLUENCE.md). It reads current pages
from one Cloud space with environment-supplied API-token credentials, defaults
to `confluence_documents`, and likewise does not enforce source permissions on
retrieval.

## HTML Extraction

`.html` and `.htm` files are decoded into normalized visible text before
chunking. Head, script, style, template, and navigation content is omitted;
headings, tables, asides, and footers remain. The `source` metadata still names
the original file, but `start_index`/`end_index` and citation excerpts refer
to the extracted text, not offsets in raw HTML. The extractor metadata changes
from `text` to `html.parser` for HTML only. Reindex affected HTML sources so
old markup chunks are replaced, and adjust any `extractor=text` metadata
filter intended to include HTML. The parser does not implement CSS visibility,
OCR, image text, or visual page layout.

The [RQ-05 audit](RQ05_EXTRACTION_CHUNKING.md) records an over-limit token count
for raw HTML and the extraction/strategy measurements behind this change.

## Index Synchronization

`index` replaces the complete indexed chunk set for each discovered file. This
removes stale chunks after content shrinks, a chunk-size change, or an empty
document. Other files in the same collection are not removed. The underlying
`LocalVectorStore.index()` API remains an ordinary deterministic upsert for
callers such as isolated benchmarks; it does not prune old points.

To reconcile files deleted from a directory, opt in explicitly:

```powershell
uv run rag-pipeline index evaluation/datasets/asteria-policies-v2/documents `
  --collection-name asteria_demo --prune-missing
```

`--prune-missing` accepts exactly one existing directory with recursive scanning
(the default); it cannot be combined with `--no-recursive`. It removes only
indexed sources whose resolved absolute paths are inside that directory and
are no longer discovered there. Check the directory and collection before
using it; missing files are removed from the index. A failed extraction or
embedding run performs no database write. Qdrant upserts and deletions are
synchronous but not one atomic transaction: if a batch or deletion fails, the
collection may be partly updated; rerun the same command to reconcile it.

## Evaluation Setup

The standalone evaluation examples use the committed v1 corpus. Index it first:

```powershell
uv run rag-pipeline index evaluation/datasets/asteria-policies-v1/documents --collection-name asteria_v1
```

Use the same model, revision, search mode, storage path, and collection for
indexing and evaluation. For chunking comparisons, use the
[v2 source-anchor dataset](../evaluation/datasets/asteria-policies-v2/README.md)
instead; its relevance labels do not depend on chunk indices.

## Chunking Experiments

The `chunk-experiment` command materializes one document snapshot and runs every
candidate against that same input. With no `--candidate` options, it compares
`500:100`, `1000:200`, and `1500:300`; each pair represents maximum chunk
characters and target overlap characters.

The report includes chunk count, min/mean/p95/max chunk length, total emitted
characters, and duplicated characters. Duplication is calculated from the
actual provenance intervals emitted by LangChain, because separator-aware
splitting does not always produce the configured overlap exactly. Chunk count
and total characters are useful cost proxies; the length distribution exposes
undersized tails and context pressure.

Use JSON when recording or comparing runs:

```powershell
uv run python -m rag_pipeline chunk-experiment evaluation/datasets/asteria-policies-v2/documents --output-format json
```

These are structural diagnostics, not retrieval-quality scores. A candidate
with fewer chunks or less duplication is not automatically better. Use the
retrieval evaluator below to compare candidates with representative queries and
relevance labels, then use the benchmark command to record end-to-end quality,
latency, and index size for promising policies.

## Retrieval Evaluation

The `evaluate-retrieval` command measures an existing collection without
calling the generation model or modifying the index. It reuses the same dense
or hybrid retrieval, exact metadata filters, score threshold, and optional
cross-encoder reranking as `retrieve` and `answer`.

Evaluation datasets are UTF-8 JSON with a strict, versioned schema:

```json
{
  "schema_version": 1,
  "name": "expense-policies-v1",
  "cases": [
    {
      "id": "itemized-receipts",
      "query": "What evidence is required for an expense claim?",
      "relevant": [
        {
          "file_name": "expense-reimbursement.md",
          "chunk_index": 0
        }
      ]
    }
  ]
}
```

Each object under `relevant` is an exact metadata selector: every listed field
and value must match a returned LangChain document. `page` and `chunk_index`
use the zero-based values stored in Qdrant. Use an application-level stable
`document_id` when datasets must survive file renames or path changes; the
current `chunk_id` remains stable only while source provenance is unchanged.
Each selector should identify one distinct relevant chunk as narrowly as
possible. Broad selectors can match several chunks and make precision look more
optimistic than the underlying evidence quality.

```powershell
uv run python -m rag_pipeline evaluate-retrieval evaluation/datasets/asteria-policies-v1/retrieval-v1.json `
  --collection-name asteria_v1 `
  --top-k 5 `
  --output-format json
```

The report includes these binary metrics at the selected cutoff:

- **Hit@k:** `1` when at least one relevant chunk is returned, otherwise `0`.
- **Precision@k:** relevant returned chunks divided by `k`; missing result slots
  caused by filtering or thresholds count as misses.
- **Recall@k:** relevance selectors matched by at least one returned chunk,
  divided by all selectors for that query.
- **RR@k / MRR@k:** reciprocal rank of the first relevant chunk for each query,
  or `0` for a miss; MRR is the macro average across queries.

Aggregate values are macro averages, so every query has equal weight. The
committed datasets provide manually cross-checked project labels for
this command. The benchmark runner reuses these exact metrics in versioned,
comparable run artifacts.

Schema v2 replaces chunk-index labels with a `metadata` object and an exact,
case-sensitive `content_contains` anchor. Schema v3 retains that positive
selector format and additionally allows `"relevant": []` for unsupported
questions. V1/v2 still require at least one selector; their JSON report fields
and metric denominators are unchanged. A v3 dataset must include at least one
positive case so positive macro metrics remain defined.

For v3, Hit@k, Precision@k, Recall@k, and MRR@k average **positive cases only**.
Negative cases show `null` for those per-case metrics and instead report
`no_hit_at_k` (`1` when no chunk is returned) and `false_positive_at_k` (`1`
when any chunk is returned). The negative macro rates average negative cases
only and are `null` if the dataset has none. This is a strict retrieval
diagnostic, not answer abstention accuracy. Per-case JSON also lists zero-based
matched and missing selector indices, one-based redundant relevant ranks
(overlap hits adding no new evidence), and one-based unwanted ranks. Chunk-level
Precision@k still counts every matching returned chunk, so inspect it alongside
unique selector coverage.

The [v3 stress-set guide](../evaluation/datasets/asteria-policies-v3/README.md)
documents its current-versus-archived policy labels and a runnable example.
It is retrieval-only and should not be paired with unrelated answer labels in
an end-to-end benchmark.

### Score-Gate Calibration

Use v3 positive and negative labels to compare first-stage gates on one
compatible collection. The ordinary `evaluate-retrieval` output is unchanged
when `--compare-score-threshold` is absent. In comparison mode, the existing
`--score-threshold` is the baseline (disabled by default); repeat
`--compare-score-threshold` for distinct candidate values in `[-1, 1]`.

```powershell
uv run rag-pipeline evaluate-retrieval `
  evaluation/datasets/asteria-policies-v3/retrieval-v3.json `
  --collection-name asteria_v3 --search-mode dense --top-k 4 `
  --no-score-threshold --compare-score-threshold 0.2 `
  --compare-score-threshold 0.4 --output-format json
```

The same embedding model, collection, metadata filters, and `top-k` are reused
for every gate; each query is actually rerun because Qdrant applies the gate
before final selection. The JSON report includes the exact gate, search mode,
nominal score kind, observed per-case score kind (`null` when no result),
ordinary v3 metrics, lost/gained zero-based evidence-selector indices, and the
change in unwanted result count. A lower candidate gate can add unwanted
context; a higher gate can lose needed evidence. The comparison disallows
`--rerank` to isolate first-stage behavior. Repeating queries costs additional
embedding and database work but does not modify the index.

Calibrate dense and hybrid collections in separate runs with the matching
`--search-mode`, `--model`, and, for hybrid, `--sparse-model`. Hybrid queries
with no sparse terms fall back to cosine; inspect the **observed** score kind
per case before interpreting a gate. Cosine and RRF values are ranking signals,
not confidence probabilities or interchangeable thresholds.

For an answer-quality comparison, use a **paired answer dataset and its own
matching corpus**, not the retrieval-only v3 labels. Set the same collection,
search mode, embedding/sparse models, filters, `top-k`, and gate explicitly on
both evaluations. For example, on an existing v2 collection:

```powershell
uv run rag-pipeline evaluate-retrieval `
  evaluation/datasets/asteria-policies-v2/retrieval-v2.json `
  --collection-name asteria_demo --search-mode dense --top-k 4 `
  --score-threshold 0.2 --output-format json
uv run rag-pipeline evaluate-answer `
  evaluation/datasets/asteria-policies-v2/answers-v2.json `
  --collection-name asteria_demo --search-mode dense --top-k 4 `
  --score-threshold 0.2 --output-format json
```

Use `--no-score-threshold` on **both** commands for a disabled-gate comparison;
this matters because `evaluate-answer` otherwise defaults to `0.2`. Record
the invocations alongside saved outputs: standalone v1/v2 report JSON keeps
its established shape and does not embed all CLI settings. No default changes
are implied by a result on the synthetic set; review representative business
queries and failure cases before selecting a production policy.
The [RQ-03 synthetic calibration record](RQ03_SCORE_GATE_CALIBRATION.md)
captures one reproducible local comparison and its limitations.

### Candidate And Reranking Comparison

Use schema-v3 retrieval labels to inspect whether a miss occurs in the
first-stage pool or after the final `--top-k` selection. Enable `--rerank`, set
the baseline width with `--candidate-k`, and repeat `--compare-candidate-k`
for at least one distinct width. Every width must be at least `--top-k`.

```powershell
uv run rag-pipeline evaluate-retrieval `
  evaluation/datasets/asteria-policies-v3/retrieval-v3.json `
  --collection-name asteria_v3 --search-mode dense `
  --no-score-threshold --rerank --top-k 2 --candidate-k 2 `
  --compare-candidate-k 3 --compare-candidate-k 4 --output-format json
```

Build `asteria_v3` from the matching v3 documents first. For hybrid, build a
separate hybrid collection from the same documents and repeat with its name,
`--search-mode hybrid`, and matching sparse-model settings. The command reads
existing collections and does not change their contents. It cannot combine
candidate comparison with `--compare-score-threshold`.

The JSON report contains three evaluations per width: `pool` at
`candidate_k`, `unreranked` at `top_k` from that same pool, and `reranked` at
`top_k` after cross-encoder scoring. Zero-based selector indices distinguish
`first_stage_missing_selector_indices` from
`lost_after_reranking_selector_indices`; gained/lost selector indices versus
the unreranked baseline expose ranking effects. Negative no-hit metrics remain
separate. Each width reruns retrieval rather than truncating a larger result,
because hybrid fusion can change ranks. Observed stage times include model
initialization and are not stable latency estimates. The v3 demo has only four
chunks, so widths above four add no candidates; validate promising settings
on a larger reviewed corpus before changing defaults. See the
[RQ-04 local comparison](RQ04_CANDIDATE_RANKING.md).

## Answer Evaluation

The `evaluate-answer` command runs labeled questions through the same dense or
hybrid retrieval, filters, score threshold, optional reranker, `grounded-v2`
LangChain prompt, and generation model used by `answer`. It reuses provider
instances across the dataset and reads the existing collection without changing
it. The default score threshold is the same conservative `0.2` used by
`answer`, and the default generation temperature is deterministic `0`.

Answer datasets use a separate strict schema because answerability and accepted
wording are different labels from retrieval relevance:

```json
{
  "schema_version": 1,
  "name": "expense-policy-answers-v1",
  "cases": [
    {
      "id": "itemized-receipts",
      "query": "What evidence is required for an expense claim?",
      "should_abstain": false,
      "reference_answers": [
        "Expense claims require itemized receipts."
      ]
    },
    {
      "id": "unsupported-founder",
      "query": "Who founded the company?",
      "should_abstain": true,
      "reference_answers": []
    }
  ]
}
```

Answerable cases require one or more accepted references. Cases expected to
abstain require an empty reference list. This explicit combination catches
contradictory labels before models or Qdrant are initialized.

```powershell
uv run python -m rag_pipeline evaluate-answer evaluation/datasets/asteria-policies-v1/answers-v1.json `
  --collection-name asteria_v1 `
  --top-k 3 `
  --output-format json
```

The report includes:

- **Exact-match rate:** normalized case-, punctuation-, and whitespace-insensitive
  matches, taking the best result across accepted references.
- **Mean token F1:** multiset word overlap with the best accepted reference,
  using Unicode-aware normalization.
- **Abstention accuracy, precision, and recall:** classification metrics for the
  pipeline's one exact insufficient-context response.
- **Answerable response rate:** the share of answerable cases the pipeline did
  not incorrectly refuse.
- **Citation-behavior rate:** whether answered cases contain citations and
  abstentions contain none.

Lexical overlap is reproducible and useful for regression testing, but it can
penalize valid paraphrases and reward unsupported wording that resembles a
reference. Citation behavior checks presence, not whether each claim is entailed
by its cited evidence. These metrics therefore do not prove semantic
faithfulness. The committed cases provide a reproducible development
baseline. The benchmark runner adds manifests and thresholds, while future
production evaluation can add a separately calibrated human, NLI, or LLM judge.

## Reproducible Benchmarking

The `benchmark` command measures a complete configuration from raw documents to
generated answers. It always creates a fresh temporary Qdrant index, runs both
evaluation datasets with shared model instances, measures one pass, records the
temporary index size, and deletes the index afterward. This costs more time than
evaluating an existing collection, but prevents stale points or undocumented
chunking settings from invalidating a comparison.

```powershell
uv run python -m rag_pipeline benchmark `
  evaluation/datasets/asteria-policies-v1/documents `
  evaluation/datasets/asteria-policies-v1/retrieval-v1.json `
  evaluation/datasets/asteria-policies-v1/answers-v1.json `
  --name asteria-dense-local-v1 `
  --top-k 4 `
  --output .rag_data/benchmarks/asteria-dense-local-v1.json
```

The schema-v1 artifact contains:

- SHA-256 fingerprints for the corpus and both label files, using portable
  relative corpus paths rather than absolute local paths
- the Git commit and tracked-worktree state when available
- chunking, model revisions, devices, search, filters, reranking, prompt, token
  limits, decoding, batching, and index configuration
- Python, package, CPU/platform, and available CUDA device metadata
- document, chunk, point, vector-dimension, and temporary-index storage totals
- stage timings plus ordered per-case retrieval and end-to-end answer latency
- the complete retrieval and answer evaluation reports
- per-anchor prompt-evidence states when answerable cases have matching v2/v3
  source-anchor retrieval labels; v1 and unpaired datasets omit this section
- reproducibility warnings for unpinned models, sampled generation, or dirty
  tracked source files

Benchmark artifacts contain dataset queries and generated answers. Treat reports
created from private or customer data according to the same access, retention,
and deletion rules as the source data; do not commit them by default.

The optional `results.evidence_coverage` section distinguishes anchors not
retrieved, retrieved but not packed, packed with model abstention, and packed
with a non-abstaining answer. It matches exact text in the packed source
prefixes, not a citation's shortened display excerpt. Counts are per labeled
anchor, so overlapping or repeated hits do not inflate coverage. An answered
state does not prove correctness; inspect answer EM/F1 separately. This
diagnostic is not yet an allowlisted regression-gate metric. See the
[RQ-06 measured comparison](RQ06_EVIDENCE_COVERAGE.md).

### Regression Thresholds

An optional strict JSON profile applies inclusive `minimum` or `maximum` checks:

```json
{
  "schema_version": 1,
  "name": "asteria-local-ci-v1",
  "applies_to": {
    "corpus_sha256": "e4c9da40c104e45792d0242f2b962e3d0267f980f04e5475d4a15fde2c798949",
    "retrieval_dataset_sha256": "4344c821c85f665cf8494c1233743a06611b63d7a1baa2d720a6007b044b8e52",
    "answer_dataset_sha256": "babd60d5819651a9cf19e01ec8ff5df80f3fbe50bad0354cd1b8cea89e92dec3",
    "top_k": 4
  },
  "checks": [
    {
      "metric": "retrieval.mean_recall_at_k",
      "operator": "minimum",
      "value": 0.9
    },
    {
      "metric": "answer.abstention_accuracy",
      "operator": "minimum",
      "value": 0.8
    },
    {
      "metric": "latency.answer.p95_seconds",
      "operator": "maximum",
      "value": 5.0
    }
  ]
}
```

This is an illustrative profile, not a checked-in configuration or a measured
quality claim. Create a profile file before supplying its path through
`--thresholds`. The numbers above demonstrate the schema; calibrate real gates
from reviewed runs rather than adopting example values blindly. The `applies_to` hashes shown
are the committed Asteria v1 inputs; take these values from an ungated baseline
artifact for another dataset. A profile is rejected before model loading when
its corpus, labels, or final top-k differ, so an easier dataset cannot
accidentally satisfy an approved gate. Supported checks cover all retrieval
quality rates, answer quality/abstention/citation rates, retrieval and answer
mean or p95 seconds, total runtime seconds, and index storage bytes. A configured
optional metric that is undefined for the dataset fails explicitly.

A failed gate still writes the diagnostic artifact and returns exit status `1`;
invalid configuration or schemas use the normal CLI error status. Existing
artifacts are preserved unless `--overwrite` is explicit.

### Comparing Runs

Pass two previously saved artifact paths as the baseline and candidate
arguments to `compare-benchmarks`; inspect the argument contract with:

```powershell
uv run rag-pipeline compare-benchmarks --help
```

Comparison requires identical corpus and dataset fingerprints plus the same
final `top-k`. Chunking, embedding, search mode, filters, reranking, generation,
and code revision may differ because those are valid experiment variables.
The comparison lists which top-level configuration sections changed alongside
the metric deltas. Quality remains comparable across runtime environments.
Latency, total runtime, and storage deltas are marked diagnostic when recorded
software, platform, accelerator, or inference-device fields differ.

Timings are wall-clock measurements from one local pass, including normal
first-call effects but separating model-load and pipeline stages. Use the same
machine, power settings, cache state, and background-load conditions for useful
regression signals. Production capacity planning still needs warmups, repeated
runs, concurrency, confidence intervals, memory telemetry, and representative
service traffic. `runtime.total_seconds` covers benchmark preparation and
execution through provenance capture; JSON serialization and artifact writing
occur afterward and are not included.

### Comparing Chunking Strategies

The `benchmark` command keeps recursive chunking as its default and exposes two
opt-in alternatives:

- `structure-aware` uses LangChain's Markdown separators. Normal extracted
  HTML has no tags, so it uses recursive fallback like PDF, DOCX, and text;
  directly supplied raw HTML documents retain HTML separator priorities.
- `semantic` embeds sentence contexts, creates boundaries at unusually large
  cosine-distance changes, and enforces `--chunk-size` as a hard character cap.
  It emits no overlap and reuses the benchmark's dense model for final vectors.

Run every candidate against the v2 source-anchor labels and keep all unrelated
settings fixed:

```powershell
uv run python -m rag_pipeline benchmark `
  evaluation/datasets/asteria-policies-v2/documents `
  evaluation/datasets/asteria-policies-v2/retrieval-v2.json `
  evaluation/datasets/asteria-policies-v2/answers-v2.json `
  --chunking-strategy semantic `
  --name asteria-v2-semantic `
  --output .rag_data/benchmarks/asteria-v2-semantic.json
```

Repeat with `recursive` and `structure-aware`, then use `compare-benchmarks`.
Review retrieval and answer quality together with chunk count, index bytes,
chunking time, dense-embedding time, and per-case failures. Semantic chunking
performs extra embedding inference during ingestion and is not assumed to be
better; representative reviewed results must justify changing a default.

## Metadata Filters

`retrieve`, `answer`, `evaluate-retrieval`, and `evaluate-answer` accept
repeatable exact-match filters. Conditions are translated into structured
Qdrant payload filters and applied before semantic top-k selection:

```powershell
uv run python -m rag_pipeline answer "What evidence is required for expenses?" `
  --collection-name asteria_v1 `
  --filter file_extension=.md `
  --filter file_name=expense-reimbursement.md
```

Repeated filters use AND semantics. Unquoted integers and lowercase JSON
booleans are typed automatically, while other unquoted values remain strings.
Use a quoted JSON string when a numeric-looking value must stay a string:

For custom metadata, pass `policy_version="001"` as one shell argument to keep
a numeric-looking value as a string. This is a syntax example; the committed
corpus does not populate a `policy_version` field.

Automatically indexed fields include `source`, `file_name`, `file_stem`,
`file_extension`, `byte_size`, `extractor`, and, for PDFs, zero-based `page` and
`total_pages`. The same API supports custom scalar metadata added to LangChain
documents programmatically. A missing field or unmatched value produces no
eligible chunks; the `answer` command then abstains before loading the LLM.

Metadata filters improve relevance but are not an authorization system by
themselves. A production API must inject mandatory tenant and ACL constraints
server-side so callers cannot omit or replace them, and frequently filtered
fields should receive Qdrant payload indexes at scale.

## Hybrid Search

Dense search remains the default and existing schema-v1 collections continue to
work unchanged. Hybrid mode creates a schema-v2 collection containing named
dense and sparse vectors. Use a separate collection name
and pass the same mode to indexing and retrieval:

```powershell
uv run python -m rag_pipeline index evaluation/datasets/asteria-policies-v2/documents `
  --collection-name policies_hybrid `
  --search-mode hybrid

uv run python -m rag_pipeline retrieve "What does EXP-042 require?" `
  --collection-name policies_hybrid `
  --search-mode hybrid
```

The first hybrid command downloads the configured FastEmbed sparse model into
`.rag_data/fastembed`; it runs locally afterward and needs no API key. Qdrant's
IDF modifier is part of the hybrid collection schema. At query time LangChain
prefetches dense and sparse candidates, applies the same metadata filters to
both branches, and asks Qdrant to combine ranks with reciprocal-rank fusion.
Queries that produce an empty sparse vector safely fall back to the collection's
dense vector.

CLI results expose `score_kind=cosine` or `score_kind=rrf`. These scores are not
interchangeable: dense thresholds are based on cosine similarity, while hybrid
thresholds apply to the fused rank score. Calibrate each mode independently on
a representative evaluation dataset. The default `Qdrant/bm25` model favors
English tokenization and stemming; multilingual or domain-specific corpora may
need a different sparse model.

Attempting hybrid retrieval against an existing dense collection fails closed.
This is intentional: adding a sparse field does not populate old points, so the
documents must be reindexed into a hybrid collection before fusion is valid.

## Cross-Encoder Reranking

Reranking works after either dense or hybrid retrieval and does not require
reindexing. The first stage retrieves a wider candidate set, then the local
cross-encoder scores each query-chunk pair jointly and returns the final
`--top-k` results:

```powershell
uv run python -m rag_pipeline retrieve "What does EXP-042 require?" `
  --collection-name policies_hybrid `
  --search-mode hybrid `
  --rerank `
  --candidate-k 20 `
  --top-k 4
```

The first reranked query downloads
`cross-encoder/ms-marco-MiniLM-L6-v2` into `.rag_data/rerankers`. It then runs
locally through Sentence Transformers and needs no API key. The model scores
all candidates in batches, with a maximum tokenized query-chunk length of 512
by default. The service uses LangChain's cross-encoder
`score([(query, document), ...])` contract, so a hosted or community scorer can
replace the local adapter later without changing retrieval or generation
contracts.

On Windows without Developer Mode, Hugging Face may warn that cache symlinks
are unavailable. Inference still works, but the fallback cache can consume more
disk space.

Metadata filters and `--score-threshold` apply during first-stage retrieval,
before any text reaches the reranker. This is important for permissions-aware
systems: reranking must never become a path around tenant or ACL filtering.
`--candidate-k` must be at least `--top-k`; increasing it may improve recall but
increases latency roughly in proportion to the number of scored pairs.

Reranked CLI results use `score_kind=cross_encoder` and retain
`retrieval_rank`, `retrieval_score`, and `retrieval_score_kind`. Cross-encoder
scores are model-specific ranking values, not calibrated probabilities, and
must not be compared directly with cosine or RRF scores. Use the implemented
retrieval evaluator and isolated benchmarks to measure whether the selected
model and candidate width improve ranking on this project's actual queries.

## Prompt Optimization

Answer generation uses a LangChain `PromptTemplate` identified as
`grounded-v2`. The prompt gives the model a compact contract: use supported
evidence only, ignore instructions found in retrieved text, abstain with one
exact response when evidence is missing, insufficient, or conflicting, and
return concise answer text without inventing sources or citations.

Ranked chunks are rendered as numbered `[Evidence n]` blocks. The same accepted
chunk sequence drives citation numbering, so evidence block 1 and citation
`[1]` always refer to the same retrieval result even when empty chunks are
skipped. Filenames and other provenance are not placed in the model prompt;
citations continue to be built and validated outside the LLM. Every
`GeneratedAnswer` records `prompt_identifier` alongside `model_identifier` for
reproducible evaluation and troubleshooting.

Character and tokenizer budgets include block labels, separators, the question,
all prompt instructions, and special tokens. When the final block must be
truncated, its citation excerpt still contains only the exact raw evidence
prefix sent to the model, excluding labels and the visual ellipsis.

The evidence delimiters improve structure but do not sanitize hostile content
or form an authorization boundary. The local FLAN-T5 baseline may also ignore
instructions or abstain too often. The implemented answer evaluator covers
lexical, abstention, and citation behavior; semantic faithfulness and
prompt-injection resilience still need separate validation. A single text prompt
is intentional for the current text-to-text model; a future chat-model adapter
can express the same contract with system and user messages.
