"""Compare first-stage candidate widths and final cross-encoder rankings.

Each width runs a fresh provider query. The same returned pool supplies both
the unreranked baseline and the reranked result, isolating selection effects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, cast

from rag_pipeline.evaluation.retrieval import (
    RetrievalEvaluationDataset,
    RetrievalEvaluationReport,
    retrieval_evaluation_to_dict,
    score_retrieval_results,
)
from rag_pipeline.exceptions import RetrievalEvaluationInputError
from rag_pipeline.retrieval import RetrievalConfig, RetrievalResult

CandidateRetriever = Callable[[str, RetrievalConfig], Sequence[RetrievalResult]]
CandidateReranker = Callable[[str, list[RetrievalResult]], Sequence[RetrievalResult]]
SearchModeName = Literal["dense", "hybrid"]


@dataclass(frozen=True, slots=True)
class CandidateWidthRun:
    """Metrics and observed work for one first-stage candidate width."""

    candidate_k: int
    pool: RetrievalEvaluationReport
    baseline: RetrievalEvaluationReport
    reranked: RetrievalEvaluationReport
    retrieval_seconds: float
    reranking_seconds: float
    observed_score_kinds: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class CandidateRankingComparison:
    """Paired stage results over one corpus, mode, gate, and label snapshot."""

    search_mode: SearchModeName
    top_k: int
    score_threshold: float | None
    runs: tuple[CandidateWidthRun, ...]


def validate_candidate_widths(
    dataset: RetrievalEvaluationDataset,
    *,
    top_k: int,
    search_mode: SearchModeName,
    candidate_widths: Sequence[int],
) -> tuple[int, ...]:
    """Reject invalid sweeps before opening models or a vector collection."""
    if not isinstance(dataset, RetrievalEvaluationDataset):
        raise TypeError("dataset must be a RetrievalEvaluationDataset.")
    if dataset.schema_version != 3:
        raise RetrievalEvaluationInputError(
            "candidate comparison requires a schema-v3 retrieval dataset."
        )
    if search_mode not in ("dense", "hybrid"):
        raise RetrievalEvaluationInputError("search_mode must be dense or hybrid.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise RetrievalEvaluationInputError("top_k must be a positive integer.")
    if isinstance(candidate_widths, (str, bytes)) or not isinstance(
        candidate_widths, Sequence
    ):
        raise RetrievalEvaluationInputError("candidate_widths must be a sequence.")
    widths = tuple(candidate_widths)
    if len(widths) < 2:
        raise RetrievalEvaluationInputError(
            "candidate comparison requires at least two widths."
        )
    for width in widths:
        if isinstance(width, bool) or not isinstance(width, int) or width < top_k:
            raise RetrievalEvaluationInputError(
                "each candidate width must be an integer at least top_k."
            )
    if len(set(widths)) != len(widths):
        raise RetrievalEvaluationInputError("candidate widths must be distinct.")
    return widths


def evaluate_candidate_ranking(
    dataset: RetrievalEvaluationDataset,
    retrieve: CandidateRetriever,
    rerank: CandidateReranker,
    *,
    top_k: int,
    search_mode: SearchModeName,
    candidate_widths: Sequence[int],
    retrieval_config: RetrievalConfig,
) -> CandidateRankingComparison:
    """Rerun each query per width and locate pre- versus post-ranking misses.

    Callbacks may perform model inference and Qdrant I/O. Timings are observed
    wall time including any lazy model load, not stable latency benchmarks.
    The score gate and metadata filters are shared across all widths.
    """
    widths = validate_candidate_widths(
        dataset,
        top_k=top_k,
        search_mode=search_mode,
        candidate_widths=candidate_widths,
    )
    if not callable(retrieve) or not callable(rerank):
        raise TypeError("retrieve and rerank must be callable.")
    if not isinstance(retrieval_config, RetrievalConfig):
        raise TypeError("retrieval_config must be a RetrievalConfig.")

    runs: list[CandidateWidthRun] = []
    for width in widths:
        config = RetrievalConfig(
            top_k=width,
            score_threshold=retrieval_config.score_threshold,
            metadata_filters=retrieval_config.metadata_filters,
        )
        pools: list[list[RetrievalResult]] = []
        observed_kinds: list[str | None] = []
        retrieval_seconds = 0.0
        for case in dataset.cases:
            start = perf_counter()
            candidates = list(retrieve(case.query, config))
            retrieval_seconds += perf_counter() - start
            if len(candidates) > width:
                raise RetrievalEvaluationInputError(
                    f"retriever returned more than {width} candidates for "
                    f"case {case.case_id!r}."
                )
            kinds = {
                result.score_kind
                for result in candidates
                if isinstance(result, RetrievalResult)
            }
            if len(kinds) > 1 or kinds - {"cosine", "rrf"}:
                raise RetrievalEvaluationInputError(
                    f"case {case.case_id!r} has inconsistent first-stage score kinds."
                )
            observed_kinds.append(next(iter(kinds), None))
            pools.append(candidates)

        pool_report = score_retrieval_results(dataset, pools, top_k=width)
        baseline_report = score_retrieval_results(dataset, pools, top_k=top_k)

        reranked: list[Sequence[RetrievalResult]] = []
        reranking_seconds = 0.0
        for case, candidates in zip(dataset.cases, pools, strict=True):
            start = perf_counter()
            results = rerank(case.query, candidates)
            reranking_seconds += perf_counter() - start
            if len(results) > top_k:
                raise RetrievalEvaluationInputError(
                    f"reranker returned more than {top_k} results for "
                    f"case {case.case_id!r}."
                )
            candidate_ids = {id(item.document) for item in candidates}
            if any(
                not isinstance(item, RetrievalResult)
                or id(item.document) not in candidate_ids
                for item in results
            ):
                raise RetrievalEvaluationInputError(
                    f"reranker returned a result outside the candidate pool for "
                    f"case {case.case_id!r}."
                )
            reranked.append(results)

        reranked_report = score_retrieval_results(dataset, reranked, top_k=top_k)
        runs.append(
            CandidateWidthRun(
                candidate_k=width,
                pool=pool_report,
                baseline=baseline_report,
                reranked=reranked_report,
                retrieval_seconds=retrieval_seconds,
                reranking_seconds=reranking_seconds,
                observed_score_kinds=tuple(observed_kinds),
            )
        )
    return CandidateRankingComparison(
        search_mode=search_mode,
        top_k=top_k,
        score_threshold=retrieval_config.score_threshold,
        runs=tuple(runs),
    )


def candidate_ranking_to_dict(
    comparison: CandidateRankingComparison,
) -> dict[str, object]:
    """Serialize stage metrics and per-selector miss attribution for review."""
    if not isinstance(comparison, CandidateRankingComparison):
        raise TypeError("comparison must be a CandidateRankingComparison.")
    runs: list[dict[str, object]] = []
    for run in comparison.runs:
        pool = retrieval_evaluation_to_dict(run.pool)
        baseline = retrieval_evaluation_to_dict(run.baseline)
        reranked = retrieval_evaluation_to_dict(run.reranked)
        cases = cast(list[dict[str, object]], reranked["cases"])
        for index, case_data in enumerate(cases):
            pool_case = run.pool.cases[index]
            baseline_case = run.baseline.cases[index]
            reranked_case = run.reranked.cases[index]
            pool_matched = set(pool_case.matched_selector_indices)
            baseline_matched = set(baseline_case.matched_selector_indices)
            final_matched = set(reranked_case.matched_selector_indices)
            case_data.update(
                {
                    "first_stage_missing_selector_indices": list(
                        pool_case.missing_selector_indices
                    ),
                    "lost_after_reranking_selector_indices": sorted(
                        pool_matched - final_matched
                    ),
                    "gained_vs_unreranked_selector_indices": sorted(
                        final_matched - baseline_matched
                    ),
                    "lost_vs_unreranked_selector_indices": sorted(
                        baseline_matched - final_matched
                    ),
                    "observed_first_stage_score_kind": run.observed_score_kinds[index],
                }
            )
        runs.append(
            {
                "candidate_k": run.candidate_k,
                "retrieval_seconds": run.retrieval_seconds,
                "reranking_seconds": run.reranking_seconds,
                "pool": pool,
                "unreranked": baseline,
                "reranked": reranked,
            }
        )
    return {
        "dataset_name": comparison.runs[0].pool.dataset_name,
        "schema_version": comparison.runs[0].pool.schema_version,
        "search_mode": comparison.search_mode,
        "top_k": comparison.top_k,
        "score_threshold": comparison.score_threshold,
        "runs": runs,
    }


def format_candidate_ranking_table(
    comparison: CandidateRankingComparison,
) -> str:
    """Show stage recall and observed work for an interactive sweep."""
    lines = [
        f"Candidate comparison: {comparison.runs[0].pool.dataset_name}",
        f"Mode: {comparison.search_mode} | Final top-k: {comparison.top_k} "
        f"| Gate: {comparison.score_threshold}",
        "Width  Stage             Recall  MRR    Neg no-hit  Observed seconds",
    ]
    for run in comparison.runs:
        for label, report, seconds in (
            ("pool", run.pool, run.retrieval_seconds),
            ("unreranked", run.baseline, run.retrieval_seconds),
            (
                "reranked",
                run.reranked,
                run.retrieval_seconds + run.reranking_seconds,
            ),
        ):
            aggregate = report.aggregate
            no_hit = aggregate.negative_no_hit_rate_at_k
            lines.append(
                f"{run.candidate_k:>5}  {label:<16}  "
                f"{aggregate.mean_recall_at_k:.3f}   "
                f"{aggregate.mean_reciprocal_rank_at_k:.3f}  "
                f"{'-' if no_hit is None else f'{no_hit:.3f}':>10}  {seconds:.3f}"
            )
        pool_misses = [
            case.case_id
            for case in run.pool.cases
            if case.relevant_count and case.missing_selector_indices
        ]
        final_losses = [
            pool_case.case_id
            for pool_case, final_case in zip(
                run.pool.cases, run.reranked.cases, strict=True
            )
            if set(pool_case.matched_selector_indices)
            - set(final_case.matched_selector_indices)
        ]
        if pool_misses:
            lines.append("       Pool misses:")
            lines.extend(f"         {case_id}" for case_id in pool_misses)
        if final_losses:
            lines.append("       Final selection losses:")
            lines.extend(f"         {case_id}" for case_id in final_losses)
    lines.append("Timings include possible lazy model initialization; not a benchmark.")
    return "\n".join(lines)
