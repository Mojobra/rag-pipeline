"""Register and execute reproducible benchmark CLI commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_pipeline.cli.options import (
    DEFAULT_ANSWER_SCORE_THRESHOLD,
    add_chunking_arguments,
    add_embedding_arguments,
    add_generation_arguments,
    add_hybrid_search_arguments,
    add_reranking_arguments,
    add_retrieval_arguments,
)


def register_benchmark_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register isolated benchmark execution and artifact comparison handlers."""
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run and save an isolated full-pipeline quality benchmark.",
        description=(
            "Build a fresh temporary Qdrant index from one corpus, run paired "
            "retrieval and answer evaluations through the configured LangChain "
            "pipeline, and save a versioned JSON artifact with provenance, "
            "quality, latency, and optional regression gates."
        ),
    )
    benchmark_parser.add_argument(
        "corpus",
        help=(
            "File or directory used to build the isolated benchmark index. "
            "Directories are scanned recursively; supported files are hashed so "
            "later comparisons can prove they used identical source content."
        ),
    )
    benchmark_parser.add_argument(
        "retrieval_dataset",
        help=(
            "Schema-v1 retrieval evaluation JSON whose queries and exact "
            "metadata selectors score the freshly indexed corpus."
        ),
    )
    benchmark_parser.add_argument(
        "answer_dataset",
        help=(
            "Schema-v1 answer evaluation JSON containing answerability labels "
            "and accepted references for end-to-end generation scoring."
        ),
    )
    benchmark_parser.add_argument(
        "--name",
        default="rag-benchmark",
        help=(
            "Human-readable experiment identity stored in the artifact and "
            "comparison output. Use a stable descriptive name for review and CI "
            "history (default: rag-benchmark)."
        ),
    )
    benchmark_parser.add_argument(
        "--output",
        required=True,
        help=(
            "Destination .json artifact. Parent directories are created; an "
            "existing file is preserved unless --overwrite is supplied."
        ),
    )
    benchmark_parser.add_argument(
        "--thresholds",
        help=(
            "Optional schema-v1 JSON profile of inclusive minimum quality or "
            "maximum latency/runtime/storage checks, bound to the input hashes "
            "and final top-k. A failed gate still writes the report and makes "
            "the command exit with status 1."
        ),
    )
    benchmark_parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow --output to replace an existing benchmark artifact atomically. "
            "Leave disabled when historical run files must remain immutable."
        ),
    )
    benchmark_parser.add_argument(
        "--work-dir",
        help=(
            "Parent directory for the temporary Qdrant database, which is deleted "
            "after the run. Change it for disk capacity or performance needs; "
            "storage location can affect latency."
        ),
    )
    add_chunking_arguments(benchmark_parser)
    add_embedding_arguments(
        benchmark_parser,
        isolated_collection=True,
    )
    add_hybrid_search_arguments(
        benchmark_parser,
        isolated_collection=True,
    )
    add_retrieval_arguments(
        benchmark_parser,
        default_score_threshold=DEFAULT_ANSWER_SCORE_THRESHOLD,
    )
    add_reranking_arguments(benchmark_parser)
    add_generation_arguments(benchmark_parser)
    benchmark_parser.add_argument(
        "--write-batch-size",
        type=int,
        default=64,
        help=(
            "Chunk vectors written per synchronous Qdrant upsert while building "
            "the temporary index. Larger batches reduce calls but use more memory "
            "and can change indexing time (default: 64)."
        ),
    )
    benchmark_parser.set_defaults(_handler=run_benchmark)

    comparison_parser = subparsers.add_parser(
        "compare-benchmarks",
        help="Compare quality and operational metrics from two saved runs.",
        description=(
            "Compare schema-v1 benchmark artifacts after verifying identical "
            "corpus and dataset fingerprints plus the same top-k cutoff. Model "
            "and pipeline settings may differ intentionally; operational metrics "
            "are marked diagnostic when runtime environments differ."
        ),
    )
    comparison_parser.add_argument(
        "baseline",
        help=(
            "Existing schema-v1 benchmark JSON used as the reference. Its corpus, "
            "labels, and top-k must match the candidate for a valid comparison."
        ),
    )
    comparison_parser.add_argument(
        "candidate",
        help=(
            "Existing schema-v1 benchmark JSON containing the proposed pipeline "
            "configuration and measurements to compare with the baseline."
        ),
    )
    comparison_parser.add_argument(
        "--output-format",
        choices=("table", "json"),
        default="table",
        help=(
            "Render the same metric deltas as a readable table or structured JSON. "
            "Use JSON for automation and saved review data (default: table)."
        ),
    )
    comparison_parser.set_defaults(_handler=run_compare_benchmarks)


def run_benchmark(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Run an isolated benchmark and atomically persist its versioned artifact.

    Validation protects input files before corpus processing or model loading.
    A failed configured metric gate still writes the report and returns status 1.
    """
    from rag_pipeline.benchmark_reporting import (
        format_benchmark_summary,
        validate_benchmark_output_path,
        write_benchmark_report,
    )
    from rag_pipeline.benchmark_thresholds import (
        load_benchmark_threshold_profile,
    )
    from rag_pipeline.benchmarking import run_benchmark as execute_benchmark
    from rag_pipeline.cli.config import build_benchmark_config
    from rag_pipeline.exceptions import (
        AnswerEvaluationError,
        BenchmarkError,
        ChunkingError,
        EmbeddingInputError,
        GenerationInputError,
        IngestionError,
        InvalidBenchmarkConfigurationError,
        InvalidEmbeddingConfigurationError,
        InvalidGenerationConfigurationError,
        InvalidRerankingConfigurationError,
        InvalidRetrievalConfigurationError,
        InvalidVectorStoreConfigurationError,
        RerankingInputError,
        RetrievalEvaluationError,
        RetrievalInputError,
        VectorStoreInputError,
    )

    try:
        output_path = validate_benchmark_output_path(
            args.output,
            overwrite=args.overwrite,
        )
        protected_inputs = {
            Path(args.retrieval_dataset).expanduser().resolve(),
            Path(args.answer_dataset).expanduser().resolve(),
        }
        if args.thresholds is not None:
            protected_inputs.add(Path(args.thresholds).expanduser().resolve())
        if output_path in protected_inputs:
            raise InvalidBenchmarkConfigurationError(
                "benchmark output must not replace an input dataset or "
                "threshold profile."
            )

        benchmark_config = build_benchmark_config(args)
        threshold_profile = (
            None
            if args.thresholds is None
            else load_benchmark_threshold_profile(args.thresholds)
        )
        report = execute_benchmark(
            args.corpus,
            args.retrieval_dataset,
            args.answer_dataset,
            config=benchmark_config,
            thresholds=threshold_profile,
        )
        written_path = write_benchmark_report(
            report,
            output_path,
            overwrite=args.overwrite,
        )
    except (
        AnswerEvaluationError,
        BenchmarkError,
        ChunkingError,
        EmbeddingInputError,
        GenerationInputError,
        IngestionError,
        InvalidEmbeddingConfigurationError,
        InvalidGenerationConfigurationError,
        InvalidRerankingConfigurationError,
        InvalidRetrievalConfigurationError,
        InvalidVectorStoreConfigurationError,
        RerankingInputError,
        RetrievalEvaluationError,
        RetrievalInputError,
        VectorStoreInputError,
    ) as exc:
        parser.error(str(exc))

    print(format_benchmark_summary(report))
    print(f"Artifact: {written_path}")
    if report.threshold_gate is not None and not report.threshold_gate.passed:
        return 1
    return 0


def run_compare_benchmarks(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Validate and compare two benchmark artifacts without running models."""
    from rag_pipeline.benchmark_artifacts import (
        benchmark_comparison_to_dict,
        compare_benchmark_artifacts,
        format_benchmark_comparison_table,
        load_benchmark_artifact,
    )
    from rag_pipeline.exceptions import BenchmarkError

    try:
        baseline_artifact = load_benchmark_artifact(args.baseline)
        candidate_artifact = load_benchmark_artifact(args.candidate)
        comparison = compare_benchmark_artifacts(
            baseline_artifact,
            candidate_artifact,
        )
    except BenchmarkError as exc:
        parser.error(str(exc))

    if args.output_format == "json":
        print(json.dumps(benchmark_comparison_to_dict(comparison), indent=2))
    else:
        print(format_benchmark_comparison_table(comparison))
    return 0
