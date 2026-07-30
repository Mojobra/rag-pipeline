"""Define reusable argparse option groups for the local RAG CLI.

The helpers in this module keep shared embedding, retrieval, storage, and
generation flags consistent across commands. They only describe parser
contracts and never initialize models, read files, or access Qdrant.
"""

from __future__ import annotations

import argparse

from rag_pipeline.embeddings import DEFAULT_LOCAL_EMBEDDING_MODEL
from rag_pipeline.reranking import (
    DEFAULT_LOCAL_RERANKER_MODEL,
    DEFAULT_RERANKER_CACHE_DIR,
)
from rag_pipeline.sparse_embeddings import (
    DEFAULT_FASTEMBED_CACHE_DIR,
    DEFAULT_LOCAL_SPARSE_MODEL,
)


DEFAULT_ANSWER_SCORE_THRESHOLD = 0.2


def add_retrieval_arguments(
    command_parser: argparse.ArgumentParser,
    *,
    default_score_threshold: float | None = None,
) -> None:
    """Attach shared result, score-gate, and metadata-filter options.

    Answer-oriented commands supply a conservative default score gate, while
    diagnostic retrieval leaves the threshold unset for calibration.
    """
    command_parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help=(
            "Maximum final chunks retained for each query. It is the cutoff for "
            "retrieval evaluation and the evidence limit for answer generation; "
            "higher values can improve recall but add work. With --rerank, it "
            "must not exceed --candidate-k (default: 4)."
        ),
    )
    command_parser.add_argument(
        "--score-threshold",
        type=float,
        default=default_score_threshold,
        help=(
            "Minimum first-stage Qdrant score in [-1, 1]; scales differ between "
            "dense and hybrid modes. Raise it to reject weak matches before "
            "reranking, at the risk of returning no context"
            + (
                " (default: disabled)."
                if default_score_threshold is None
                else f" (default: {default_score_threshold})."
            )
        ),
    )
    command_parser.add_argument(
        "--filter",
        action="append",
        dest="metadata_filters",
        metavar="KEY=VALUE",
        help=(
            "Exact Qdrant metadata condition applied before top-k selection. "
            "Repeat for AND semantics; unquoted integers and JSON booleans are "
            "typed automatically. Example: --filter file_extension=.pdf."
        ),
    )


def add_hybrid_search_arguments(
    command_parser: argparse.ArgumentParser,
    *,
    isolated_collection: bool = False,
) -> None:
    """Attach dense/hybrid collection and sparse-model options to a command.

    The options configure later service construction only; parser assembly does
    not initialize FastEmbed or access its cache.
    """
    command_parser.add_argument(
        "--search-mode",
        choices=("dense", "hybrid"),
        default="dense",
        help=(
            "Qdrant schema and retrieval strategy. Hybrid adds local sparse "
            "vectors and RRF fusion for keyword recall, with extra CPU, storage, "
            "and latency; "
            + (
                "the benchmark builds the selected schema in isolation "
                "(default: dense)."
                if isolated_collection
                else "must match the existing collection (default: dense)."
            )
        ),
    )
    command_parser.add_argument(
        "--sparse-model",
        default=DEFAULT_LOCAL_SPARSE_MODEL,
        help=(
            (
                "FastEmbed sparse model used in the temporary hybrid index and "
                "queries. Changing it changes keyword retrieval and benchmark "
                "storage; the index is rebuilt automatically "
                if isolated_collection
                else (
                    "FastEmbed sparse model used only in hybrid indexing and "
                    "queries. Changing it alters retrieval and requires a new "
                    "or rebuilt hybrid collection "
                )
            )
            + f"(default: {DEFAULT_LOCAL_SPARSE_MODEL})."
        ),
    )
    command_parser.add_argument(
        "--sparse-cache-dir",
        default=str(DEFAULT_FASTEMBED_CACHE_DIR),
        help=(
            "Directory for downloaded FastEmbed sparse-model files, used only in "
            "hybrid mode. Changing it relocates disk use and may trigger another "
            f"download (default: {DEFAULT_FASTEMBED_CACHE_DIR})."
        ),
    )
    command_parser.add_argument(
        "--sparse-batch-size",
        type=int,
        default=256,
        help=(
            "Texts processed per FastEmbed sparse inference batch in hybrid mode. "
            "This mainly affects indexing; larger values can improve throughput "
            "but use more RAM (default: 256)."
        ),
    )
    command_parser.add_argument(
        "--sparse-threads",
        type=int,
        help=(
            "Positive CPU thread count passed to FastEmbed in hybrid mode. More "
            "threads can improve throughput but increase CPU contention; omit to "
            "use the provider default."
        ),
    )


def add_reranking_arguments(command_parser: argparse.ArgumentParser) -> None:
    """Attach optional second-stage candidate and model settings.

    Candidate width remains distinct from final top-k so the first stage can
    overfetch before cross-encoder scoring.
    """
    command_parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Score the first --candidate-k results with a local cross-encoder, "
            "then keep --top-k. This can improve ordering but adds a model "
            "download, inference latency, and memory use."
        ),
    )
    command_parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help=(
            "First-stage chunks retrieved when --rerank is enabled; ignored "
            "otherwise. Larger pools give the reranker more recall but cost more "
            "inference and must be at least --top-k (default: 20)."
        ),
    )
    command_parser.add_argument(
        "--reranker-model",
        default=DEFAULT_LOCAL_RERANKER_MODEL,
        help=(
            "Sentence Transformers cross-encoder loaded only with --rerank. "
            "Different models trade ranking quality against download size, memory, "
            f"and latency (default: {DEFAULT_LOCAL_RERANKER_MODEL})."
        ),
    )
    command_parser.add_argument(
        "--reranker-model-revision",
        help=(
            "Optional Hugging Face commit or tag for --reranker-model. Pin a "
            "revision for reproducible scores; omitting it follows the model "
            "repository default."
        ),
    )
    command_parser.add_argument(
        "--reranker-device",
        default="cpu",
        help=(
            "Device passed to the cross-encoder, such as cpu, cuda, or cuda:0. "
            "A GPU can reduce reranking latency but consumes VRAM (default: cpu)."
        ),
    )
    command_parser.add_argument(
        "--reranker-cache-dir",
        default=str(DEFAULT_RERANKER_CACHE_DIR),
        help=(
            "Directory for downloaded cross-encoder files. Change it to control "
            "disk placement; an empty cache causes a model download when reranking "
            f"first runs (default: {DEFAULT_RERANKER_CACHE_DIR})."
        ),
    )
    command_parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=16,
        help=(
            "Query-chunk pairs scored per cross-encoder inference batch. Larger "
            "values can improve throughput but use more RAM or VRAM (default: 16)."
        ),
    )
    command_parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=512,
        help=(
            "Maximum tokens retained for each query-chunk pair by the cross-encoder. "
            "Larger values preserve more text but increase compute and memory "
            "use (default: 512)."
        ),
    )


def add_generation_arguments(command_parser: argparse.ArgumentParser) -> None:
    """Attach local language-model and prompt-budget settings for answers.

    These options are isolated to generation so retrieval diagnostics never
    initialize or configure a language model.
    """
    command_parser.add_argument(
        "--generation-model",
        default="google/flan-t5-small",
        help=(
            "Hugging Face model used by the text2text-generation pipeline. Model "
            "choice affects answer quality, download size, memory, and latency "
            "and is reused across an evaluation run "
            "(default: google/flan-t5-small)."
        ),
    )
    command_parser.add_argument(
        "--generation-model-revision",
        help=(
            "Optional Hugging Face commit or tag for --generation-model. Pin it "
            "for reproducible generation; omitting it follows the repository "
            "default."
        ),
    )
    command_parser.add_argument(
        "--generation-device",
        default="cpu",
        help=(
            "Generation device: cpu, cuda, or cuda:<index>. CUDA can reduce "
            "latency but requires a compatible GPU and sufficient VRAM "
            "(default: cpu)."
        ),
    )
    command_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help=(
            "Maximum tokens the answer model may generate. Higher limits allow "
            "longer answers but increase inference time and memory use "
            "(default: 128)."
        ),
    )
    command_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature in [0, 2]. Zero disables sampling for more "
            "repeatable answers; higher values increase variation and may reduce "
            "groundedness (default: 0)."
        ),
    )
    command_parser.add_argument(
        "--max-context-characters",
        type=int,
        default=1200,
        help=(
            "Maximum characters of formatted retrieved evidence considered for "
            "the prompt, before the token limit is enforced. Lower values reduce "
            "work but can omit useful evidence (default: 1200)."
        ),
    )
    command_parser.add_argument(
        "--max-input-tokens",
        type=int,
        help=(
            "Optional cap on tokenized instructions, question, and evidence before "
            "an internal safety margin. It cannot exceed the tokenizer limit; lower "
            "values truncate evidence sooner. Required when that limit is unknown."
        ),
    )


def add_embedding_arguments(
    command_parser: argparse.ArgumentParser,
    *,
    isolated_collection: bool = False,
) -> None:
    """Attach the dense model identity, device, and batching options.

    Indexing and retrieval share this group because they must use a compatible
    embedding contract.
    """
    command_parser.add_argument(
        "--model",
        default=DEFAULT_LOCAL_EMBEDDING_MODEL,
        help=(
            (
                "Sentence Transformers-compatible Hugging Face model for the "
                "temporary index and query vectors. Changing it changes quality, "
                "memory, and latency; the index is rebuilt automatically. "
                if isolated_collection
                else (
                    "Sentence Transformers-compatible Hugging Face model for "
                    "document and query dense vectors. Index and query with the "
                    "same model; changing it requires a new or rebuilt collection. "
                )
            )
            + f"Default: {DEFAULT_LOCAL_EMBEDDING_MODEL}."
        ),
    )
    command_parser.add_argument(
        "--model-revision",
        help=(
            "Optional Hugging Face commit or tag for --model. Use the same pinned "
            "revision for indexing and queries to reproduce vectors; omitting it "
            "follows the repository default."
        ),
    )
    command_parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Dense-embedding device understood by Sentence Transformers, such as "
            "cpu, cuda, or cuda:0. A GPU can improve throughput but consumes VRAM "
            "(default: cpu)."
        ),
    )
    command_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Document chunks processed per dense-embedding inference batch. Larger "
            "values can improve embed/index throughput but use more RAM or VRAM; "
            "single-query embedding is unaffected (default: 32)."
        ),
    )


def add_vector_store_location_arguments(
    command_parser: argparse.ArgumentParser,
) -> None:
    """Attach persistent Qdrant location and collection identity options."""
    command_parser.add_argument(
        "--store-path",
        default=".rag_data/qdrant",
        help=(
            "Directory containing the persistent local Qdrant database. Use the "
            "same path for indexing and later query or evaluation commands; "
            "different paths isolate stored collections "
            "(default: .rag_data/qdrant)."
        ),
    )
    command_parser.add_argument(
        "--collection-name",
        default="rag_documents",
        help=(
            "Qdrant collection within --store-path. Use separate names for corpora "
            "or incompatible embedding/search settings, and reuse the name when "
            "querying (default: rag_documents)."
        ),
    )


def add_chunking_arguments(command_parser: argparse.ArgumentParser) -> None:
    """Attach character-based chunk size and overlap options."""
    command_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help=(
            "Maximum characters per retrieval chunk. Smaller chunks can sharpen "
            "matches but create more vectors; larger chunks retain context but may "
            "dilute relevance. Must exceed --chunk-overlap (default: 1000)."
        ),
    )
    command_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help=(
            "Target characters repeated between adjacent chunks. More overlap "
            "preserves boundary context but increases embedding work and storage; "
            "must be non-negative and below --chunk-size (default: 200)."
        ),
    )


def add_document_input_arguments(
    command_parser: argparse.ArgumentParser,
) -> None:
    """Attach supported document paths and recursive discovery behavior."""
    command_parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "One or more files or directories containing .txt, .md, .markdown, "
            ".html, .htm, .pdf, or .docx documents. Directories are scanned "
            "recursively by default; discovered files are deduplicated and sorted."
        ),
    )
    command_parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help=(
            "Scan only direct files inside each directory instead of its full tree. "
            "Use for large directory trees or deliberate scope control; explicitly "
            "listed files are still processed."
        ),
    )
