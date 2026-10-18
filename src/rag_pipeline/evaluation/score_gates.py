"""Compare first-stage score gates against one labeled retrieval snapshot.

Each gate executes the real retrieval callback independently. This preserves
Qdrant's selection behavior instead of pretending that filtering a saved top-k
list reproduces a new provider query.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from rag_pipeline.evaluation.retrieval import (
    RetrievalEvaluationDataset,
    RetrievalEvaluationReport,
    evaluate_retrieval,
    retrieval_evaluation_to_dict,
)
from rag_pipeline.exceptions import (
    InvalidRetrievalConfigurationError,
    RetrievalEvaluationInputError,
)
from rag_pipeline.retrieval import RetrievalConfig, RetrievalResult

SearchModeName = Literal["dense", "hybrid"]
GateRetrievalFunction = Callable[[str, RetrievalConfig], Sequence[RetrievalResult]]


@dataclass(frozen=True, slots=True)
class ScoreGateRun:
    """One gate's labeled metrics and observed first-stage score provenance."""

    score_threshold: float | None
    evaluation: RetrievalEvaluationReport
    observed_score_kinds: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class ScoreGateComparison:
    """Paired gate runs on one corpus, mode, query set, and result cutoff."""

    search_mode: SearchModeName
    baseline: ScoreGateRun
    candidates: tuple[ScoreGateRun, ...]


def evaluate_score_gates(
    dataset: RetrievalEvaluationDataset,
    retrieve: GateRetrievalFunction,
    *,
    top_k: int,
    search_mode: SearchModeName,
    baseline_gate: float | None,
    candidate_gates: Sequence[float],
) -> ScoreGateComparison:
    """Run each gate through the same retriever and compare v3 judgments.

    The callback may perform embedding inference and Qdrant I/O. Invalid gates,
    missing negative labels, and duplicate settings fail before any query.
    Dense and hybrid modes must be run separately against compatible indexes;
    no model-quality result is inferred by this function.
    """
    if not callable(retrieve):
        raise TypeError("retrieve must be callable.")
    configs = validate_score_gate_settings(
        dataset,
        top_k=top_k,
        search_mode=search_mode,
        baseline_gate=baseline_gate,
        candidate_gates=candidate_gates,
    )

    def run(config: RetrievalConfig) -> ScoreGateRun:
        observed: list[str | None] = []

        def retrieve_case(query: str) -> Sequence[RetrievalResult]:
            results = retrieve(query, config)
            # evaluate_retrieval validates ranks and result objects after this call.
            if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
                kinds = {
                    result.retrieval_score_kind or result.score_kind
                    for result in results[:top_k]
                    if isinstance(result, RetrievalResult)
                }
                if len(kinds) > 1 or kinds - {"cosine", "rrf"}:
                    raise RetrievalEvaluationInputError(
                        "one query returned inconsistent first-stage score kinds."
                    )
                observed.append(next(iter(kinds), None))
            else:
                observed.append(None)
            return results

        report = evaluate_retrieval(dataset, retrieve_case, top_k=top_k)
        return ScoreGateRun(
            score_threshold=config.score_threshold,
            evaluation=report,
            observed_score_kinds=tuple(observed),
        )

    runs = tuple(run(config) for config in configs)
    return ScoreGateComparison(
        search_mode=search_mode,
        baseline=runs[0],
        candidates=runs[1:],
    )


def validate_score_gate_settings(
    dataset: RetrievalEvaluationDataset,
    *,
    top_k: int,
    search_mode: SearchModeName,
    baseline_gate: float | None,
    candidate_gates: Sequence[float],
) -> tuple[RetrievalConfig, ...]:
    """Reject invalid comparison inputs before provider initialization or I/O."""
    if not isinstance(dataset, RetrievalEvaluationDataset):
        raise TypeError("dataset must be a RetrievalEvaluationDataset.")
    if dataset.schema_version != 3 or not any(
        not case.relevant_documents for case in dataset.cases
    ):
        raise RetrievalEvaluationInputError(
            "score-gate comparison requires a schema-v3 dataset with negative cases."
        )
    if search_mode not in ("dense", "hybrid"):
        raise RetrievalEvaluationInputError("search_mode must be dense or hybrid.")
    if isinstance(candidate_gates, (str, bytes)):
        raise RetrievalEvaluationInputError("candidate_gates must be a sequence.")
    try:
        gates = tuple(candidate_gates)
    except TypeError as exc:
        raise RetrievalEvaluationInputError(
            "candidate_gates must be a sequence."
        ) from exc
    if not gates:
        raise RetrievalEvaluationInputError("at least one candidate gate is required.")

    configs: list[RetrievalConfig] = []
    for gate in (baseline_gate, *gates):
        try:
            configs.append(RetrievalConfig(top_k=top_k, score_threshold=gate))
        except InvalidRetrievalConfigurationError as exc:
            raise RetrievalEvaluationInputError(str(exc)) from exc
    if len({config.score_threshold for config in configs}) != len(configs):
        raise RetrievalEvaluationInputError("score gates must be distinct.")
    return tuple(configs)


def score_gate_comparison_to_dict(
    comparison: ScoreGateComparison,
) -> dict[str, object]:
    """Serialize gate settings, per-case deltas, and existing v3 metrics."""
    if not isinstance(comparison, ScoreGateComparison):
        raise TypeError("comparison must be a ScoreGateComparison.")

    def run_data(run: ScoreGateRun) -> dict[str, object]:
        data = retrieval_evaluation_to_dict(run.evaluation)
        data["score_threshold"] = run.score_threshold
        cases = cast(list[dict[str, object]], data["cases"])
        for case, kind in zip(cases, run.observed_score_kinds, strict=True):
            case["observed_score_kind"] = kind
        return data

    baseline = run_data(comparison.baseline)
    candidates: list[dict[str, object]] = []
    for run in comparison.candidates:
        candidate = run_data(run)
        cases = cast(list[dict[str, object]], candidate["cases"])
        for index, case in enumerate(cases):
            baseline_case = comparison.baseline.evaluation.cases[index]
            current_case = run.evaluation.cases[index]
            baseline_selectors = set(baseline_case.matched_selector_indices)
            current_selectors = set(current_case.matched_selector_indices)
            case["lost_selector_indices"] = sorted(
                baseline_selectors - current_selectors
            )
            case["gained_selector_indices"] = sorted(
                current_selectors - baseline_selectors
            )
            case["unwanted_hit_change"] = len(current_case.unwanted_ranks) - len(
                baseline_case.unwanted_ranks
            )
        candidates.append(candidate)

    return {
        "dataset_name": comparison.baseline.evaluation.dataset_name,
        "schema_version": comparison.baseline.evaluation.schema_version,
        "top_k": comparison.baseline.evaluation.top_k,
        "search_mode": comparison.search_mode,
        "nominal_score_kind": (
            "cosine" if comparison.search_mode == "dense" else "rrf"
        ),
        "baseline": baseline,
        "candidates": candidates,
    }


def format_score_gate_comparison(comparison: ScoreGateComparison) -> str:
    """Render paired gate outcomes without treating scores as confidence."""
    data = score_gate_comparison_to_dict(comparison)
    lines = [
        f"Score-gate comparison: {data['dataset_name']}",
        f"Mode: {data['search_mode']} | Nominal score: {data['nominal_score_kind']} "
        f"| Cutoff: {data['top_k']}",
    ]
    for label, run in (
        ("Baseline", comparison.baseline),
        *(("Candidate", candidate) for candidate in comparison.candidates),
    ):
        aggregate = run.evaluation.aggregate
        gate = "disabled" if run.score_threshold is None else str(run.score_threshold)
        lines.append(
            f"{label} gate={gate}: Hit={aggregate.hit_rate_at_k:.3f} "
            f"Recall={aggregate.mean_recall_at_k:.3f} "
            f"negative-no-hit={aggregate.negative_no_hit_rate_at_k:.3f}"
        )
    for run in comparison.candidates:
        gate = "disabled" if run.score_threshold is None else str(run.score_threshold)
        lines.extend(("", f"Per-case changes at gate={gate}:"))
        for baseline_case, current_case, baseline_kind, kind in zip(
            comparison.baseline.evaluation.cases,
            run.evaluation.cases,
            comparison.baseline.observed_score_kinds,
            run.observed_score_kinds,
            strict=True,
        ):
            lost = sorted(
                set(baseline_case.matched_selector_indices)
                - set(current_case.matched_selector_indices)
            )
            gained = sorted(
                set(current_case.matched_selector_indices)
                - set(baseline_case.matched_selector_indices)
            )
            lines.append(
                f"  {current_case.case_id}: "
                f"score={baseline_kind or 'no hit'}->{kind or 'no hit'} "
                f"hits={baseline_case.retrieved_count}->{current_case.retrieved_count} "
                f"evidence={baseline_case.matched_relevant_count}"
                f"->{current_case.matched_relevant_count} lost={lost} gained={gained} "
                f"unwanted={len(baseline_case.unwanted_ranks)}"
                f"->{len(current_case.unwanted_ranks)}"
            )
    return "\n".join(lines)
