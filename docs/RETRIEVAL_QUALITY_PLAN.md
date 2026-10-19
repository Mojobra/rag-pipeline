# Retrieval Quality Implementation Plan

Status: in progress. This document records follow-up work identified during a
codebase review; RQ-01 through RQ-04 are integrated into `develop`. Later tasks remain
separately reviewable and are marked complete only after integration.

## Goal And Scope

Improve the relevance, freshness, and coverage of evidence returned by the
local RAG pipeline. Keep search quality distinct from answer quality: a good
retrieval result can still be lost while evidence is packed into the generation
prompt. The work follows the existing LangChain, Qdrant, evaluation, and
benchmarking architecture. Phase 4's API and authentication tasks remain
separate in the [roadmap](../ROADMAP.md).

Current baseline: recursive chunks of 1000 characters with 200 characters of
overlap; dense search by default; optional BM25 hybrid search and cross-encoder
reranking. `retrieve` and `evaluate-retrieval` have no default score gate, while
`answer`, `evaluate-answer`, and `benchmark` default to `0.2`. The committed
Asteria v3 retrieval set has 13 answerable and four unsupported questions over
four synthetic English Markdown policies. These are configuration and dataset
facts, not measured quality results.

## Task Tracker

| Task | Deliverable | Status |
| --- | --- | --- |
| RQ-01 | Source-scoped index synchronization | Complete |
| RQ-02 | Broader retrieval labels and metrics | Complete |
| RQ-03 | Score-gate calibration | Complete |
| RQ-04 | Candidate and reranking comparison | Complete |
| RQ-05 | Extraction and chunking quality | Planned |
| RQ-06 | Evidence-to-prompt coverage | Planned |

## How To Execute The Plan

- Take one task at a time on a focused branch from the latest `develop`, with a
  separate pull request into `develop`. Do not mark a task complete merely
  because an experiment ran or code was written.
- Preserve existing CLI behavior and v1/v2 evaluation schemas unless a task
  explicitly justifies and tests a change. Add a new dataset version rather
  than modifying benchmark inputs on which previous results depend.
- Start each comparison from a fresh, isolated index. Record corpus and label
  fingerprints, Git revision, model names and revisions, chunking settings,
  search mode, `top-k`, candidate width, score gate, and environment. Change one
  retrieval variable at a time when attributing an effect.
- Compare per-question results as well as Hit@k, Recall@k, Precision@k, and
  reciprocal rank. Record latency and index size when a candidate adds compute
  or storage. Existing benchmarks are single passes, so do not treat a small
  latency difference as statistically established.
- Keep default automated tests deterministic and offline. Run model-dependent
  quality experiments separately. Preserve source provenance, citations,
  collection compatibility, and prompt token-budget safeguards.
- Treat a gain on the small synthetic set as a reason for broader validation,
  not as proof of improved performance on business documents. Agree on
  acceptable quality, latency, and storage trade-offs after a baseline is
  measured; this plan sets no invented numeric targets.

## RQ-01: Keep The Index In Sync With Source Documents

**Problem.** Indexing upserts deterministic chunk IDs. If a source becomes
shorter, is removed, or produces fewer chunks after a chunk-size change, its
older points remain searchable. A two-chunk to one-chunk reindex was reproduced
in an in-memory Qdrant collection: the obsolete second point remained.

**Implementation.** Define a source-scoped replacement operation distinct from
the current general upsert. Determine the complete new point set for one source
before removing obsolete points. Make deletion of files from a directory an
explicit, scoped synchronization decision; never delete unrelated sources in a
shared collection. Specify behavior for empty documents and for failures at
extraction, embedding, and database-write stages.

**Done when.** Reindexing a changed or empty source cannot return obsolete
chunks after a successful run; indexing another source leaves existing sources
intact. Offline tests cover shrinking, removed/empty sources, repeated runs,
multi-page PDFs, and failure paths. The CLI and documentation make replacement
versus ordinary upsert semantics clear.

**Trade-off.** Source replacement needs extra reads and writes. An in-place
multi-batch update is not automatically atomic; the implementation must make
partial-write behavior explicit or use a staged replacement where necessary.

## RQ-02: Make Retrieval Evaluation Representative

**Problem.** The current retrieval dataset contains only positive questions,
and its schema requires at least one relevant selector per case. Overlapping
chunks can also match the same anchor more than once, inflating the apparent
number of useful results.

**Implementation.** Add a new, versioned dataset and a backward-compatible
schema extension for questions with no relevant evidence. Define separate
negative-case behavior, such as false-positive or no-hit accuracy, without
making Recall@k undefined or silently changing v1/v2 metrics. Add hard
paraphrases, exact identifiers, ambiguous wording, multi-document questions,
and conflicting or superseded policies. Report unique matched evidence
selectors alongside chunk-level precision. Use approved, private
representative documents for optional validation; do not commit them.

**Done when.** New labels have documented review criteria and integrity tests.
Existing v1/v2 reports remain compatible. Tests cover empty relevance, duplicate
overlap hits, partial multi-document matches, and macro aggregation. Per-case
reports reveal misses and unwanted matches, not only aggregate scores.

**Trade-off.** More nuanced labels and negative metrics take annotation effort.
Hard cases should be reviewed independently where possible; synthetic labels
alone cannot establish production accuracy.

## RQ-03: Calibrate The First-Stage Score Gate

**Problem.** A fixed `0.2` threshold applies to answer-oriented commands before
reranking, but diagnostic retrieval leaves it disabled. Cosine and RRF scores
have different scales. A relevant candidate rejected here cannot be recovered
by the reranker or answer model.

**Implementation.** With RQ-02 labels, compare disabled and candidate gates
separately for dense and hybrid search. Record which positive questions lose
evidence and which negative questions gain unwanted context. Ensure retrieval
diagnostics and answer evaluations use explicitly matched settings in every
comparison. Decide from results whether defaults, per-mode configuration, or
documentation should change; do not assume one universal threshold.

**Done when.** Reports show per-case outcomes, score kind, and the exact gate
used. A default change requires regression tests, updated CLI help and manuals,
and evidence that the chosen false-positive/false-negative trade-off is useful.

**Trade-off.** Raising the gate can reduce irrelevant context but increases
missed answers. Provider scores are ranking signals, not calibrated confidence
probabilities.

The [synthetic calibration record](RQ03_SCORE_GATE_CALIBRATION.md)
documents the first dense and hybrid comparison, including exact settings and
per-case losses. RQ-03 was integrated into `develop` in PR #16.

## RQ-04: Tune Candidate Retrieval And Reranking

**Problem.** Hybrid search and reranking already exist, but the reranker can
only reorder its first-stage candidate pool (`--candidate-k`, default 20).
That pool, fusion mode, and final `--top-k` need evaluation together.

**Implementation.** Compare dense and hybrid collections on the same corpus and
labels. Measure first-stage Recall@candidate-k before comparing final rank and
Recall@top-k with and without the cross-encoder. Sweep a small, documented set
of candidate widths. Inspect per-query failures before considering fusion
changes, diversity limits, or a different local embedding/reranking model.
Any model change gets a fresh compatible collection and pinned revision when
available.

**Done when.** The comparison identifies whether misses arise before or after
reranking and records latency, memory or storage implications where measured.
Only a repeatable, reviewed benefit changes a default or adds ranking logic;
otherwise the existing options remain available without extra complexity.

**Trade-off.** Wider candidate pools and cross-encoders increase inference
cost. A source-diversity rule could suppress multiple genuinely useful passages
from the same document, so it requires labeled multi-evidence cases.

The [synthetic candidate comparison](RQ04_CANDIDATE_RANKING.md) records the
first width sweep and its per-query misses. RQ-04 was integrated into
`develop` in PR #17.

## RQ-05: Improve Extracted Text And Chunk Boundaries

**Problem.** HTML is indexed as raw markup, PDF extraction depends on an
existing text layer and reading order, and the normal splitter uses character
counts rather than embedding-token counts. Structure-aware and semantic
chunking are currently benchmark-only.

**Implementation.** Add reviewed fixtures for noisy HTML, policy headings,
tables, long passages, and problematic PDF text order. Measure extraction
coverage and embedding-token truncation before changing parsers or splitting
defaults. Compare recursive, structure-aware, and semantic policies with the
v2 anchor labels where valid. Consider cleaned text, heading context, or
token-aware limits only when a specific failure is demonstrated. Treat OCR as
a separate decision if scanned documents are in scope.

**Done when.** Tests show extracted facts and source/page/character provenance
remain correct. Quality experiments report per-case retrieval effects, chunk
counts, indexing time, and index size. A promoted chunking strategy retains
stable citation and collection-compatibility behavior.

**Trade-off.** Cleaning can remove meaningful text; smaller or overlapping
chunks increase index size; semantic boundaries add embedding work. Parser
changes can invalidate existing offsets and evaluation anchors.

The task-branch [extraction and chunking record](RQ05_EXTRACTION_CHUNKING.md)
documents the fixture audit and matched v2 benchmarks. RQ-05 remains planned
in the tracker until reviewed integration into `develop`.

## RQ-06: Measure Evidence That Reaches The Prompt

**Problem.** Search may find the correct chunk while rank-ordered packing
truncates it or stops before it reaches the generation model. The current
1200-character context cap and tokenizer limit protect execution but do not
measure whether labeled evidence survives packing.

**Implementation.** Add a diagnostic that checks which labeled evidence spans
are present in the packed prompt, separate from retrieval metrics and answer
word overlap. Use it to assess overlap-aware deduplication or focused evidence
windows before changing packing. Keep exact source spans, deterministic
citations, abstention rules, and token-limit validation intact.

**Done when.** A report distinguishes "not retrieved", "retrieved but not
packed", and "packed but not answered" for answerable cases. Tests cover
truncation, repeated evidence, multi-source questions, and citation alignment.
Any packing change improves measured evidence coverage without exceeding the
configured prompt budget.

**Trade-off.** More context can dilute useful evidence and raise latency or
memory use. Extracting a smaller window can omit a needed qualification.

## Suggested Order And Completion Record

Work in the order RQ-01 through RQ-06. RQ-02 supplies stronger labels for later
calibration; RQ-03 and RQ-04 use those labels before RQ-05 or RQ-06 changes
defaults. Each task ends with a short record of the baseline configuration,
per-case and aggregate results, tests, changed contracts, limitations, and the
next proposed task. Update this plan's status only after the corresponding
reviewed task is integrated into `develop`.
