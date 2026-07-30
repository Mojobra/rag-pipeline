"""Register and execute document inspection and chunking CLI commands."""

from __future__ import annotations

import argparse
import json

from rag_pipeline.cli.options import (
    add_chunking_arguments,
    add_document_input_arguments,
)


def register_document_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register ingestion and chunking commands with explicit handlers."""
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Discover and extract supported local documents.",
        description=(
            "Discover supported local files and extract LangChain documents. "
            "Prints loaded sources without chunking, model inference, or storage."
        ),
    )
    add_document_input_arguments(ingest_parser)
    ingest_parser.set_defaults(_handler=run_ingest)

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Preview retrieval chunks without running models or indexing.",
        description=(
            "Extract and split local documents, then report the resulting chunk "
            "count. This diagnostic command performs no model inference or writes."
        ),
    )
    add_document_input_arguments(chunk_parser)
    add_chunking_arguments(chunk_parser)
    chunk_parser.set_defaults(_handler=run_chunk)

    experiment_parser = subparsers.add_parser(
        "chunk-experiment",
        help="Compare structural costs of several chunking policies.",
        description=(
            "Apply multiple character-based chunking policies to one document "
            "snapshot. Reports size and duplication metrics without evaluating "
            "retrieval quality, calling models, or writing an index."
        ),
    )
    add_document_input_arguments(experiment_parser)
    experiment_parser.add_argument(
        "--candidate",
        action="append",
        dest="chunking_candidates",
        metavar="SIZE:OVERLAP",
        help=(
            "Chunk size and overlap in characters, formatted SIZE:OVERLAP. "
            "Repeat to compare policies; overlap must be smaller than size. "
            "If omitted, compares 500:100, 1000:200, and 1500:300."
        ),
    )
    experiment_parser.add_argument(
        "--output-format",
        choices=("table", "json"),
        default="table",
        help=(
            "Render identical metrics as a readable table or structured JSON. "
            "Use JSON for scripts and saved comparisons (default: table)."
        ),
    )
    experiment_parser.set_defaults(_handler=run_chunk_experiment)


def run_ingest(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Load supported documents and print their normalized source paths.

    The command performs document discovery and extraction but does not chunk,
    embed, or persist content. Ingestion errors retain their existing exception
    behavior rather than being rewritten as parser errors.
    """
    del parser
    from rag_pipeline.ingestion import load_documents

    documents = load_documents(args.paths, recursive=args.recursive)
    print(f"Ingested {len(documents)} document(s).")
    for document in documents:
        print(f"- {document.metadata['source']}")
    return 0


def run_chunk(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Load and split documents, reporting counts without model inference."""
    from rag_pipeline.chunking import chunk_documents
    from rag_pipeline.cli.config import build_chunking_config
    from rag_pipeline.exceptions import InvalidChunkingConfigurationError
    from rag_pipeline.ingestion import load_documents

    try:
        config = build_chunking_config(args)
    except InvalidChunkingConfigurationError as exc:
        parser.error(str(exc))

    documents = load_documents(args.paths, recursive=args.recursive)
    chunks = chunk_documents(documents, config=config)
    print(f"Chunked {len(documents)} document(s) into {len(chunks)} chunk(s).")
    return 0


def run_chunk_experiment(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Compare structural chunking metrics for one extracted document snapshot."""
    from rag_pipeline.chunking_experiments import (
        DEFAULT_CHUNKING_CANDIDATES,
        chunking_experiment_to_dict,
        format_chunking_experiment_table,
        parse_chunking_candidate,
        run_chunking_experiment,
    )
    from rag_pipeline.exceptions import InvalidChunkingExperimentError
    from rag_pipeline.ingestion import load_documents

    try:
        candidates = (
            DEFAULT_CHUNKING_CANDIDATES
            if args.chunking_candidates is None
            else tuple(
                parse_chunking_candidate(value) for value in args.chunking_candidates
            )
        )
        documents = load_documents(args.paths, recursive=args.recursive)
        report = run_chunking_experiment(documents, candidates=candidates)
    except InvalidChunkingExperimentError as exc:
        parser.error(str(exc))

    if args.output_format == "json":
        print(json.dumps(chunking_experiment_to_dict(report), indent=2))
    else:
        print(format_chunking_experiment_table(report))
    return 0
