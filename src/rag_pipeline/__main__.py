"""Expose the local RAG pipeline stages through one command-line interface.

The module assembles validated stage configurations, performs lazy provider
initialization, coordinates filesystem/model/Qdrant work, and formats terminal
output without moving domain logic out of the underlying services.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_pipeline import __version__
from rag_pipeline.embeddings import DEFAULT_LOCAL_EMBEDDING_MODEL
from rag_pipeline.reranking import (
    DEFAULT_LOCAL_RERANKER_MODEL,
    DEFAULT_RERANKER_CACHE_DIR,
)
from rag_pipeline.sparse_embeddings import (
    DEFAULT_FASTEMBED_CACHE_DIR,
    DEFAULT_LOCAL_SPARSE_MODEL,
)

if TYPE_CHECKING:
    from rag_pipeline.embeddings import LocalEmbeddingConfig
    from rag_pipeline.reranking import LocalRerankerConfig, RerankingConfig
    from rag_pipeline.retrieval import RetrievalConfig, RetrievalResult
    from rag_pipeline.sparse_embeddings import LocalSparseEmbeddingConfig
    from rag_pipeline.vector_store import VectorStoreConfig


DEFAULT_ANSWER_SCORE_THRESHOLD = 0.2


@dataclass(frozen=True, slots=True)
class _RetrievalRuntimeConfig:
    """Validated service settings shared by all retrieval-based commands.

    Keeping one assembled contract prevents interactive retrieval, evaluation,
    and answer generation from interpreting common CLI options differently.
    Model and provider resources are still initialized lazily by each command.
    """

    embedding: LocalEmbeddingConfig
    vector_store: VectorStoreConfig
    sparse_embedding: LocalSparseEmbeddingConfig | None
    retrieval: RetrievalConfig
    local_reranker: LocalRerankerConfig | None
    reranking: RerankingConfig | None


def build_parser() -> argparse.ArgumentParser:
    """Build the complete parser for the package module entry point.

    Related option groups are reused across commands so indexing, retrieval, and
    answering construct the same model and storage contracts. Parser creation
    has no provider, filesystem, or database side effects.
    """
    parser = argparse.ArgumentParser(
        prog="rag_pipeline",
        description=(
            "Run individual stages of the local RAG pipeline, from document "
            "inspection through grounded answer generation."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rag-pipeline {__version__}",
        help="Print the installed rag-pipeline version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Discover and extract supported local documents.",
        description=(
            "Discover supported local files and extract LangChain documents. "
            "Prints loaded sources without chunking, model inference, or storage."
        ),
    )
    _add_document_input_arguments(ingest_parser)

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Preview retrieval chunks without running models or indexing.",
        description=(
            "Extract and split local documents, then report the resulting chunk "
            "count. This diagnostic command performs no model inference or writes."
        ),
    )
    _add_document_input_arguments(chunk_parser)
    _add_chunking_arguments(chunk_parser)

    chunk_experiment_parser = subparsers.add_parser(
        "chunk-experiment",
        help="Compare structural costs of several chunking policies.",
        description=(
            "Apply multiple character-based chunking policies to one document "
            "snapshot. Reports size and duplication metrics without evaluating "
            "retrieval quality, calling models, or writing an index."
        ),
    )
    _add_document_input_arguments(chunk_experiment_parser)
    chunk_experiment_parser.add_argument(
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
    chunk_experiment_parser.add_argument(
        "--output-format",
        choices=("table", "json"),
        default="table",
        help=(
            "Render identical metrics as a readable table or structured JSON. "
            "Use JSON for scripts and saved comparisons (default: table)."
        ),
    )

    embed_parser = subparsers.add_parser(
        "embed",
        help="Inspect local dense embedding output without indexing.",
        description=(
            "Extract, chunk, and embed local documents, then report vector count "
            "and dimension. Models may be downloaded, but no Qdrant data is written."
        ),
    )
    _add_document_input_arguments(embed_parser)
    _add_chunking_arguments(embed_parser)
    _add_embedding_arguments(embed_parser)

    index_parser = subparsers.add_parser(
        "index",
        help="Build or update a persistent local Qdrant collection.",
        description=(
            "Extract, chunk, and embed documents, then upsert deterministic points "
            "into local Qdrant. Model and search-mode settings become part of the "
            "collection compatibility contract."
        ),
    )
    _add_document_input_arguments(index_parser)
    _add_chunking_arguments(index_parser)
    _add_embedding_arguments(index_parser)
    _add_vector_store_location_arguments(index_parser)
    _add_hybrid_search_arguments(index_parser)
    index_parser.add_argument(
        "--write-batch-size",
        type=int,
        default=64,
        help=(
            "Number of chunk vectors sent in each synchronous Qdrant upsert. "
            "Larger batches reduce write calls but use more memory; reduce after "
            "memory-related write failures (default: 64)."
        ),
    )

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Inspect ranked chunks from an existing Qdrant collection.",
        description=(
            "Embed a query, search a compatible Qdrant collection, and print ranked "
            "evidence. Optional filtering, hybrid search, and reranking apply before "
            "output; no answer model is loaded."
        ),
    )
    retrieve_parser.add_argument(
        "query",
        help=(
            "Text to embed and match against indexed chunks. Specific wording and "
            "keywords can materially change dense and hybrid retrieval results."
        ),
    )
    _add_embedding_arguments(retrieve_parser)
    _add_vector_store_location_arguments(retrieve_parser)
    _add_hybrid_search_arguments(retrieve_parser)
    _add_retrieval_arguments(retrieve_parser)
    _add_reranking_arguments(retrieve_parser)

    evaluation_parser = subparsers.add_parser(
        "evaluate-retrieval",
        help="Measure retrieval quality on a labeled query dataset.",
        description=(
            "Run every query in a versioned JSON dataset through the configured "
            "Qdrant retrieval path and report binary top-k metrics. Optional "
            "filters, hybrid search, and reranking are included; generation is "
            "never invoked and the index is not modified."
        ),
    )
    evaluation_parser.add_argument(
        "dataset",
        help=(
            "UTF-8 JSON file containing schema_version, name, and labeled query "
            "cases with exact metadata selectors under relevant. Change it to "
            "evaluate a different corpus or relevance-judgment snapshot."
        ),
    )
    _add_embedding_arguments(evaluation_parser)
    _add_vector_store_location_arguments(evaluation_parser)
    _add_hybrid_search_arguments(evaluation_parser)
    _add_retrieval_arguments(evaluation_parser)
    _add_reranking_arguments(evaluation_parser)
    evaluation_parser.add_argument(
        "--output-format",
        choices=("table", "json"),
        default="table",
        help=(
            "Render the same per-case and macro metrics as an aligned terminal "
            "table or structured JSON. Use JSON when saving or comparing runs "
            "programmatically (default: table)."
        ),
    )

    answer_parser = subparsers.add_parser(
        "answer",
        help="Generate a cited local answer from retrieved evidence.",
        description=(
            "Retrieve eligible chunks from Qdrant, optionally rerank them, and pass "
            "bounded evidence to a local generation model. Returns a deterministic "
            "abstention when no chunk passes retrieval criteria."
        ),
    )
    answer_parser.add_argument(
        "query",
        help=(
            "Question used for both retrieval and grounded generation. Clear, "
            "specific wording generally yields more focused evidence and answers."
        ),
    )
    _add_embedding_arguments(answer_parser)
    _add_vector_store_location_arguments(answer_parser)
    _add_hybrid_search_arguments(answer_parser)
    _add_retrieval_arguments(
        answer_parser,
        default_score_threshold=DEFAULT_ANSWER_SCORE_THRESHOLD,
    )
    _add_reranking_arguments(answer_parser)
    _add_generation_arguments(answer_parser)
    return parser


def _add_retrieval_arguments(
    command_parser: argparse.ArgumentParser,
    *,
    default_score_threshold: float | None = None,
) -> None:
    """Attach shared result, score-gate, and metadata-filter options.

    ``answer`` supplies a conservative default score gate, while diagnostic
    retrieval intentionally leaves the threshold unset for calibration.
    """
    command_parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help=(
            "Maximum final chunks retained for each query and the evaluation "
            "metric cutoff. Higher values can improve recall but increase output, "
            "evaluation, or prompt work; with --rerank, must not exceed "
            "--candidate-k (default: 4)."
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


def _add_hybrid_search_arguments(command_parser: argparse.ArgumentParser) -> None:
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
            "and latency; must match the existing collection (default: dense)."
        ),
    )
    command_parser.add_argument(
        "--sparse-model",
        default=DEFAULT_LOCAL_SPARSE_MODEL,
        help=(
            "FastEmbed sparse model used only in hybrid indexing and queries. "
            "Changing it alters retrieval and requires a new or rebuilt hybrid "
            f"collection (default: {DEFAULT_LOCAL_SPARSE_MODEL})."
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


def _add_reranking_arguments(command_parser: argparse.ArgumentParser) -> None:
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


def _add_generation_arguments(command_parser: argparse.ArgumentParser) -> None:
    """Attach local language-model and prompt-budget settings for answers.

    These options are isolated to generation so retrieval diagnostics never
    initialize or configure a language model.
    """
    command_parser.add_argument(
        "--generation-model",
        default="google/flan-t5-small",
        help=(
            "Hugging Face model loaded with the text2text-generation pipeline "
            "after answer retrieval succeeds. Model choice affects answer quality, "
            "download size, memory, and latency (default: google/flan-t5-small)."
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


def _add_embedding_arguments(command_parser: argparse.ArgumentParser) -> None:
    """Attach the dense model identity, device, and batching options.

    Indexing and retrieval share this group because they must use a compatible
    embedding contract.
    """
    command_parser.add_argument(
        "--model",
        default=DEFAULT_LOCAL_EMBEDDING_MODEL,
        help=(
            "Sentence Transformers-compatible Hugging Face model for document and "
            "query dense vectors. Index and query with the same model; changing it "
            "requires a new or rebuilt collection. "
            f"Default: {DEFAULT_LOCAL_EMBEDDING_MODEL}."
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


def _add_vector_store_location_arguments(
    command_parser: argparse.ArgumentParser,
) -> None:
    command_parser.add_argument(
        "--store-path",
        default=".rag_data/qdrant",
        help=(
            "Directory containing the persistent local Qdrant database. Use the "
            "same path for index, retrieve, and answer; different paths isolate "
            "stored collections (default: .rag_data/qdrant)."
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


def _add_chunking_arguments(command_parser: argparse.ArgumentParser) -> None:
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


def _add_document_input_arguments(command_parser: argparse.ArgumentParser) -> None:
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


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and execute one local pipeline command.

    Depending on the command, execution may read documents, download/cache and
    run models, mutate or query local Qdrant, and print results to stdout.
    Invalid command configuration is reported through ``argparse`` and may raise
    ``SystemExit``; successful commands return zero.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        from rag_pipeline.ingestion import load_documents

        documents = load_documents(args.paths, recursive=args.recursive)
        print(f"Ingested {len(documents)} document(s).")
        for document in documents:
            print(f"- {document.metadata['source']}")
        return 0

    if args.command == "chunk":
        from rag_pipeline.chunking import (
            ChunkingConfig,
            InvalidChunkingConfigurationError,
            chunk_documents,
        )
        from rag_pipeline.ingestion import load_documents

        try:
            config = ChunkingConfig(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        except InvalidChunkingConfigurationError as exc:
            parser.error(str(exc))

        documents = load_documents(args.paths, recursive=args.recursive)
        chunks = chunk_documents(documents, config=config)
        print(
            f"Chunked {len(documents)} document(s) into {len(chunks)} chunk(s)."
        )
        return 0

    if args.command == "chunk-experiment":
        import json

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
                    parse_chunking_candidate(value)
                    for value in args.chunking_candidates
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

    if args.command == "embed":
        from rag_pipeline.chunking import (
            ChunkingConfig,
            InvalidChunkingConfigurationError,
            chunk_documents,
        )
        from rag_pipeline.embeddings import (
            InvalidEmbeddingConfigurationError,
            LocalEmbeddingConfig,
            create_local_embedding_service,
        )
        from rag_pipeline.ingestion import load_documents

        try:
            chunking_config = ChunkingConfig(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            embedding_config = LocalEmbeddingConfig(
                model_name=args.model,
                model_revision=args.model_revision,
                device=args.device,
                batch_size=args.batch_size,
            )
        except (
            InvalidChunkingConfigurationError,
            InvalidEmbeddingConfigurationError,
        ) as exc:
            parser.error(str(exc))

        documents = load_documents(args.paths, recursive=args.recursive)
        chunks = chunk_documents(documents, config=chunking_config)
        service = create_local_embedding_service(embedding_config)
        embedded_documents = service.embed_documents(chunks)

        if not embedded_documents:
            print("Embedded 0 chunk(s); no vectors were created.")
            return 0

        print(
            f"Embedded {len(embedded_documents)} chunk(s) into "
            f"{embedded_documents[0].dimension}-dimensional vectors using "
            f"{service.model_identifier}."
        )
        return 0

    if args.command == "index":
        from rag_pipeline.chunking import (
            ChunkingConfig,
            InvalidChunkingConfigurationError,
            chunk_documents,
        )
        from rag_pipeline.embeddings import (
            InvalidEmbeddingConfigurationError,
            LocalEmbeddingConfig,
            create_local_embedding_service,
        )
        from rag_pipeline.exceptions import InvalidVectorStoreConfigurationError
        from rag_pipeline.ingestion import load_documents
        from rag_pipeline.sparse_embeddings import (
            LocalSparseEmbeddingConfig,
            create_local_sparse_embedding_service,
        )
        from rag_pipeline.vector_store import (
            LocalVectorStore,
            SearchMode,
            VectorStoreConfig,
        )

        try:
            chunking_config = ChunkingConfig(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            embedding_config = LocalEmbeddingConfig(
                model_name=args.model,
                model_revision=args.model_revision,
                device=args.device,
                batch_size=args.batch_size,
            )
            search_mode = SearchMode(args.search_mode)
            vector_store_config = VectorStoreConfig(
                path=args.store_path,
                collection_name=args.collection_name,
                write_batch_size=args.write_batch_size,
                search_mode=search_mode,
            )
            sparse_embedding_config = (
                LocalSparseEmbeddingConfig(
                    model_name=args.sparse_model,
                    cache_dir=args.sparse_cache_dir,
                    batch_size=args.sparse_batch_size,
                    threads=args.sparse_threads,
                )
                if search_mode == SearchMode.HYBRID
                else None
            )
        except (
            InvalidChunkingConfigurationError,
            InvalidEmbeddingConfigurationError,
            InvalidVectorStoreConfigurationError,
        ) as exc:
            parser.error(str(exc))

        documents = load_documents(args.paths, recursive=args.recursive)
        chunks = chunk_documents(documents, config=chunking_config)
        embedding_service = create_local_embedding_service(embedding_config)
        embedded_documents = embedding_service.embed_documents(chunks)
        sparse_embedding_service = (
            create_local_sparse_embedding_service(sparse_embedding_config)
            if sparse_embedding_config is not None
            else None
        )
        sparse_vectors = (
            sparse_embedding_service.embed_documents(chunks)
            if sparse_embedding_service is not None
            else None
        )

        with LocalVectorStore(vector_store_config) as vector_store:
            result = vector_store.index(
                embedded_documents,
                model_identifier=embedding_service.model_identifier,
                sparse_vectors=sparse_vectors,
                sparse_model_identifier=(
                    None
                    if sparse_embedding_service is None
                    else sparse_embedding_service.model_identifier
                ),
            )

        print(
            f"Indexed {result.indexed_count} chunk(s) into "
            f"{result.collection_name!r}; collection now contains "
            f"{result.total_count} point(s)."
        )
        return 0

    if args.command == "retrieve":
        from rag_pipeline.embeddings import (
            InvalidEmbeddingConfigurationError,
            create_local_embedding_service,
        )
        from rag_pipeline.exceptions import (
            InvalidRerankingConfigurationError,
            InvalidRetrievalConfigurationError,
            InvalidVectorStoreConfigurationError,
        )
        from rag_pipeline.retrieval import RetrieverService
        from rag_pipeline.sparse_embeddings import (
            create_local_sparse_embedding_service,
        )
        from rag_pipeline.vector_store import LocalVectorStore

        try:
            runtime_config = _build_retrieval_runtime_config(args)
        except (
            InvalidEmbeddingConfigurationError,
            InvalidVectorStoreConfigurationError,
            InvalidRetrievalConfigurationError,
            InvalidRerankingConfigurationError,
        ) as exc:
            parser.error(str(exc))

        embedding_service = create_local_embedding_service(
            runtime_config.embedding
        )
        sparse_embedding_service = (
            create_local_sparse_embedding_service(
                runtime_config.sparse_embedding
            )
            if runtime_config.sparse_embedding is not None
            else None
        )
        with LocalVectorStore(runtime_config.vector_store) as vector_store:
            results = RetrieverService(
                embedding_service,
                vector_store,
                sparse_embedding_service,
            ).retrieve(args.query, config=runtime_config.retrieval)

        results = _rerank_results(
            args.query,
            results,
            local_config=runtime_config.local_reranker,
            config=runtime_config.reranking,
        )

        if not results:
            print("No chunks met the retrieval criteria.")
            return 0

        for result in results:
            metadata = result.document.metadata
            location = f"source={metadata.get('source', '<unknown>')}"
            page = metadata.get("page")
            if isinstance(page, int) and not isinstance(page, bool):
                location += f" page={page + 1}"
            chunk_index = metadata.get("chunk_index")
            if isinstance(chunk_index, int) and not isinstance(chunk_index, bool):
                location += f" chunk={chunk_index}"

            ranking_details = f"score_kind={result.score_kind}"
            if result.retrieval_rank is not None:
                ranking_details += (
                    f" retrieval_rank={result.retrieval_rank}"
                    f" retrieval_score={result.retrieval_score:.4f}"
                    f" retrieval_score_kind={result.retrieval_score_kind}"
                    f" reranker_model={result.reranker_model}"
                )
            print(
                f"{result.rank}. score={result.score:.4f} {location} "
                f"{ranking_details}"
            )
            print(f"   {_content_preview(result.document.page_content)}")
        return 0

    if args.command == "evaluate-retrieval":
        import json

        from rag_pipeline.embeddings import (
            InvalidEmbeddingConfigurationError,
            create_local_embedding_service,
        )
        from rag_pipeline.exceptions import (
            InvalidRerankingConfigurationError,
            InvalidRetrievalConfigurationError,
            InvalidRetrievalEvaluationDatasetError,
            InvalidVectorStoreConfigurationError,
        )
        from rag_pipeline.reranking import create_local_reranker_service
        from rag_pipeline.retrieval import RetrieverService
        from rag_pipeline.retrieval_evaluation import (
            evaluate_retrieval,
            format_retrieval_evaluation_table,
            load_retrieval_evaluation_dataset,
            retrieval_evaluation_to_dict,
        )
        from rag_pipeline.sparse_embeddings import (
            create_local_sparse_embedding_service,
        )
        from rag_pipeline.vector_store import LocalVectorStore

        try:
            dataset = load_retrieval_evaluation_dataset(args.dataset)
            runtime_config = _build_retrieval_runtime_config(args)
        except (
            InvalidEmbeddingConfigurationError,
            InvalidVectorStoreConfigurationError,
            InvalidRetrievalConfigurationError,
            InvalidRerankingConfigurationError,
            InvalidRetrievalEvaluationDatasetError,
        ) as exc:
            parser.error(str(exc))

        embedding_service = create_local_embedding_service(
            runtime_config.embedding
        )
        sparse_embedding_service = (
            create_local_sparse_embedding_service(
                runtime_config.sparse_embedding
            )
            if runtime_config.sparse_embedding is not None
            else None
        )
        reranker = (
            create_local_reranker_service(runtime_config.local_reranker)
            if runtime_config.local_reranker is not None
            else None
        )

        with LocalVectorStore(runtime_config.vector_store) as vector_store:
            retriever = RetrieverService(
                embedding_service,
                vector_store,
                sparse_embedding_service,
            )

            def retrieve_for_evaluation(query: str) -> list[RetrievalResult]:
                """Run one query through the shared retriever and reranker."""
                results = retriever.retrieve(
                    query,
                    config=runtime_config.retrieval,
                )
                if reranker is None:
                    return results
                if runtime_config.reranking is None:
                    raise RuntimeError(
                        "Reranker service has no result-limit configuration."
                    )
                return reranker.rerank(
                    query,
                    results,
                    config=runtime_config.reranking,
                )

            report = evaluate_retrieval(
                dataset,
                retrieve_for_evaluation,
                top_k=args.top_k,
            )

        if args.output_format == "json":
            print(json.dumps(retrieval_evaluation_to_dict(report), indent=2))
        else:
            print(format_retrieval_evaluation_table(report))
        return 0

    if args.command == "answer":
        from rag_pipeline.citations import format_citation
        from rag_pipeline.embeddings import (
            InvalidEmbeddingConfigurationError,
            create_local_embedding_service,
        )
        from rag_pipeline.exceptions import (
            InvalidGenerationConfigurationError,
            InvalidRerankingConfigurationError,
            InvalidRetrievalConfigurationError,
            InvalidVectorStoreConfigurationError,
        )
        from rag_pipeline.generation import (
            INSUFFICIENT_CONTEXT_ANSWER,
            GenerationConfig,
            LocalGenerationConfig,
            create_local_answer_generator,
        )
        from rag_pipeline.retrieval import RetrieverService
        from rag_pipeline.sparse_embeddings import (
            create_local_sparse_embedding_service,
        )
        from rag_pipeline.vector_store import LocalVectorStore

        try:
            runtime_config = _build_retrieval_runtime_config(args)
            local_generation_config = LocalGenerationConfig(
                model_name=args.generation_model,
                model_revision=args.generation_model_revision,
                device=args.generation_device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            generation_config = GenerationConfig(
                max_context_characters=args.max_context_characters,
                max_input_tokens=args.max_input_tokens,
            )
        except (
            InvalidEmbeddingConfigurationError,
            InvalidVectorStoreConfigurationError,
            InvalidRetrievalConfigurationError,
            InvalidRerankingConfigurationError,
            InvalidGenerationConfigurationError,
        ) as exc:
            parser.error(str(exc))

        embedding_service = create_local_embedding_service(
            runtime_config.embedding
        )
        sparse_embedding_service = (
            create_local_sparse_embedding_service(
                runtime_config.sparse_embedding
            )
            if runtime_config.sparse_embedding is not None
            else None
        )
        with LocalVectorStore(runtime_config.vector_store) as vector_store:
            retrieval_results = RetrieverService(
                embedding_service,
                vector_store,
                sparse_embedding_service,
            ).retrieve(args.query, config=runtime_config.retrieval)

        retrieval_results = _rerank_results(
            args.query,
            retrieval_results,
            local_config=runtime_config.local_reranker,
            config=runtime_config.reranking,
        )

        if not retrieval_results:
            print("Answer:")
            print(INSUFFICIENT_CONTEXT_ANSWER)
            return 0

        answer_generator = create_local_answer_generator(local_generation_config)
        generated_answer = answer_generator.generate(
            args.query,
            retrieval_results,
            config=generation_config,
        )
        print("Answer:")
        print(generated_answer.answer)
        if generated_answer.citations:
            print()
            print("Sources:")
            for citation in generated_answer.citations:
                print(format_citation(citation))
        return 0

    print("RAG Pipeline skeleton is ready.")
    return 0


def _build_retrieval_runtime_config(
    args: argparse.Namespace,
) -> _RetrievalRuntimeConfig:
    """Translate shared CLI fields into one validated retrieval contract.

    The function validates dense and optional sparse models, Qdrant location,
    result limits, metadata filters, and optional reranking settings. It performs
    no model initialization, downloads, vector-store I/O, or inference.
    """
    from rag_pipeline.embeddings import LocalEmbeddingConfig
    from rag_pipeline.retrieval import RetrievalConfig, parse_metadata_filter
    from rag_pipeline.sparse_embeddings import LocalSparseEmbeddingConfig
    from rag_pipeline.vector_store import SearchMode, VectorStoreConfig

    embedding_config = LocalEmbeddingConfig(
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
    )
    search_mode = SearchMode(args.search_mode)
    vector_store_config = VectorStoreConfig(
        path=args.store_path,
        collection_name=args.collection_name,
        search_mode=search_mode,
    )
    sparse_embedding_config = (
        LocalSparseEmbeddingConfig(
            model_name=args.sparse_model,
            cache_dir=args.sparse_cache_dir,
            batch_size=args.sparse_batch_size,
            threads=args.sparse_threads,
        )
        if search_mode == SearchMode.HYBRID
        else None
    )
    (
        local_reranker_config,
        reranking_config,
        retrieval_top_k,
    ) = _build_reranking_configs(args)
    retrieval_config = RetrievalConfig(
        top_k=retrieval_top_k,
        score_threshold=args.score_threshold,
        metadata_filters=tuple(
            parse_metadata_filter(value)
            for value in (args.metadata_filters or ())
        ),
    )
    return _RetrievalRuntimeConfig(
        embedding=embedding_config,
        vector_store=vector_store_config,
        sparse_embedding=sparse_embedding_config,
        retrieval=retrieval_config,
        local_reranker=local_reranker_config,
        reranking=reranking_config,
    )


def _build_reranking_configs(
    args: argparse.Namespace,
) -> tuple[
    LocalRerankerConfig | None,
    RerankingConfig | None,
    int,
]:
    """Translate CLI reranking arguments into service and result-limit settings.

    Disabled reranking preserves the requested top-k as first-stage width.
    Enabled reranking validates that candidate width can satisfy final top-k and
    returns the wider retrieval count. No model is initialized here.
    """
    from rag_pipeline.exceptions import InvalidRerankingConfigurationError
    from rag_pipeline.reranking import LocalRerankerConfig, RerankingConfig

    if not args.rerank:
        return None, None, args.top_k

    reranking_config = RerankingConfig(top_n=args.top_k)
    if args.candidate_k < reranking_config.top_n:
        raise InvalidRerankingConfigurationError(
            "candidate_k must be greater than or equal to top_k."
        )
    local_config = LocalRerankerConfig(
        model_name=args.reranker_model,
        model_revision=args.reranker_model_revision,
        device=args.reranker_device,
        cache_dir=args.reranker_cache_dir,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
    )
    return local_config, reranking_config, args.candidate_k


def _rerank_results(
    query: str,
    results: list[RetrievalResult],
    *,
    local_config: LocalRerankerConfig | None,
    config: RerankingConfig | None,
) -> list[RetrievalResult]:
    """Optionally initialize the local reranker and reorder retrieved results.

    Empty input or disabled reranking returns the original list unchanged.
    Otherwise the function may download/cache a model and performs cross-encoder
    inference through ``RerankerService``.
    """
    if not results or config is None:
        return results
    if local_config is None:
        raise RuntimeError("Reranking config has no local model configuration.")

    from rag_pipeline.reranking import create_local_reranker_service

    reranker = create_local_reranker_service(local_config)
    return reranker.rerank(query, results, config=config)


def _content_preview(content: str, *, max_length: int = 240) -> str:
    preview = " ".join(content.split())
    if len(preview) <= max_length:
        return preview
    return f"{preview[: max_length - 3]}..."


if __name__ == "__main__":
    raise SystemExit(main())
