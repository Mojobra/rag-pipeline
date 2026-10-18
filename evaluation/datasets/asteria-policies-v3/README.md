# Asteria Policies v3: Retrieval Stress Set

This synthetic English corpus tests retrieval decisions that the earlier
answerable-only datasets cannot measure. It is an independent snapshot, not a
replacement for the v1 or v2 benchmark inputs.

## Contents

- `documents/`: Four fictional Markdown policies, including distinct current
  and archived expense rules.
- `retrieval-v3.json`: Thirteen answerable and four unsupported queries. Positive
  judgments use exact source metadata plus a case-sensitive content anchor;
  negative judgments use `"relevant": []`.

The positive cases cover paraphrases, policy identifiers, exact monetary
boundaries, historical versus current rules, and questions requiring evidence
from two documents. The negative cases ask for facts absent from the corpus,
including a hotel cap not covered by the travel policy.

## Label Review Criteria

1. A positive selector must identify exactly one source document and at least
   one chunk under the documented default recursive splitter. Each selector
   represents a distinct fact to retrieve, not a chunk index.
2. A multi-document question lists evidence from each required source. A
   current-policy question does not treat an archived threshold alone as
   sufficient evidence.
3. A negative question must have no supporting answer in this corpus. A
   topically related chunk is still an unwanted retrieval hit for the strict
   no-hit diagnostic; reviewers should revisit a label if a source actually
   answers the question.
4. Case IDs, source files, and anchors are immutable within this version.
   Correcting a judgment or changing the corpus requires a new version so
   saved results remain comparable.

Automated tests check counts, schema, positive-anchor uniqueness, default
chunk resolution, and cross-document selectors. The synthetic judgments have
not undergone independent second-annotator review and cannot establish quality
on private business data.

## Run Locally

```powershell
uv run rag-pipeline index evaluation/datasets/asteria-policies-v3/documents `
  --collection-name asteria_v3
uv run rag-pipeline evaluate-retrieval `
  evaluation/datasets/asteria-policies-v3/retrieval-v3.json `
  --collection-name asteria_v3 --output-format json

uv run rag-pipeline evaluate-retrieval `
  evaluation/datasets/asteria-policies-v3/retrieval-v3.json `
  --collection-name asteria_v3 --search-mode dense --top-k 4 `
  --no-score-threshold --compare-score-threshold 0.2 `
  --output-format json
```

Use the same embedding model and collection for both commands. The first run
may download the local embedding model. The retrieval command does not load a
generation model or modify the collection. It has no default score threshold;
an irrelevant returned chunk on an unsupported query counts as a false
positive. This version has no paired answer-evaluation file, so it is intended
for `evaluate-retrieval`, not an end-to-end benchmark with unrelated answer
labels.

The comparison reruns every question for the disabled baseline and the
candidate gate. It records per-case evidence losses, unwanted-result changes,
the actual score kind when results exist, and the exact threshold. Run hybrid
calibration separately against a hybrid index with `--search-mode hybrid` and
the matching sparse model; cosine and RRF scores are not interchangeable.
These commands are a procedure, not measured quality results or a recommendation
to use `0.2` for either mode.
