"""Expose the local RAG pipeline stages through one command-line interface.

The module assembles validated stage configurations, performs lazy provider
initialization, coordinates filesystem/model/Qdrant work, and formats terminal
output without moving domain logic out of the underlying services.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import TYPE_CHECKING

from rag_pipeline import __version__
from rag_pipeline.embeddings import DEFAULT_LOCAL_EMBEDDING_MODEL
from rag_pipeline.exceptions import InvalidModelProviderConfigurationError
from rag_pipeline.reranking import (
    DEFAULT_LOCAL_RERANKER_MODEL,
    DEFAULT_RERANKER_CACHE_DIR,
)
from rag_pipeline.sparse_embeddings import (
    DEFAULT_FASTEMBED_CACHE_DIR,
    DEFAULT_LOCAL_SPARSE_MODEL,
)

if TYPE_CHECKING:
    from rag_pipeline.embeddings import EmbeddingService, LocalEmbeddingConfig
    from rag_pipeline.generation import (
        AnswerGenerator,
        HostedGenerationConfig,
        LocalGenerationConfig,
    )
    from rag_pipeline.model_profiles import ProviderModelProfile
    from rag_pipeline.reranking import LocalRerankerConfig, RerankingConfig
    from rag_pipeline.retrieval import RetrievalResult


DEFAULT_ANSWER_SCORE_THRESHOLD = 0.2


def build_parser() -> argparse.ArgumentParser:
    """Build the complete parser for the package module entry point.

    Related option groups are reused across commands so indexing, retrieval, and
    answering construct the same model and storage contracts. Parser creation
    has no provider, filesystem, or database side effects.
    """
    parser = argparse.ArgumentParser(
        prog="rag_pipeline",
        description="Run the production-minded RAG pipeline.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rag-pipeline {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Load supported local files into LangChain Document objects.",
    )
    _add_document_input_arguments(ingest_parser)

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Load and split local documents into retrieval-sized chunks.",
    )
    _add_document_input_arguments(chunk_parser)
    _add_chunking_arguments(chunk_parser)

    chunk_experiment_parser = subparsers.add_parser(
        "chunk-experiment",
        help="Compare chunking candidates against the same documents.",
    )
    _add_document_input_arguments(chunk_experiment_parser)
    chunk_experiment_parser.add_argument(
        "--candidate",
        action="append",
        dest="chunking_candidates",
        metavar="SIZE:OVERLAP",
        help=(
            "Candidate to evaluate; repeat for multiple settings "
            "(defaults: 500:100, 1000:200, 1500:300)."
        ),
    )
    chunk_experiment_parser.add_argument(
        "--output-format",
        choices=("table", "json"),
        default="table",
        help="Report format (default: table).",
    )

    embed_parser = subparsers.add_parser(
        "embed",
        help="Load, chunk, and locally embed supported documents.",
    )
    _add_document_input_arguments(embed_parser)
    _add_chunking_arguments(embed_parser)
    _add_embedding_arguments(embed_parser)

    index_parser = subparsers.add_parser(
        "index",
        help="Load, chunk, embed, and persist documents in local Qdrant.",
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
        help="Vectors per Qdrant upsert batch (default: 64).",
    )

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Find semantically similar chunks in an indexed collection.",
    )
    retrieve_parser.add_argument(
        "query",
        help="Natural-language question or search query.",
    )
    _add_embedding_arguments(retrieve_parser)
    _add_vector_store_location_arguments(retrieve_parser)
    _add_hybrid_search_arguments(retrieve_parser)
    _add_retrieval_arguments(retrieve_parser)
    _add_reranking_arguments(retrieve_parser)

    answer_parser = subparsers.add_parser(
        "answer",
        help="Retrieve context and generate a grounded answer.",
    )
    answer_parser.add_argument(
        "query",
        help="Natural-language question to answer.",
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
        help="Maximum number of chunks to return (default: 4).",
    )
    command_parser.add_argument(
        "--score-threshold",
        type=float,
        default=default_score_threshold,
        help=(
            "Minimum retrieval score from -1 to 1"
            + (
                "."
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
            "Exact metadata condition; repeat for AND semantics "
            "(integers and booleans are typed automatically)."
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
        help="Collection and retrieval mode (default: dense).",
    )
    command_parser.add_argument(
        "--sparse-model",
        default=DEFAULT_LOCAL_SPARSE_MODEL,
        help=(
            "FastEmbed sparse model used in hybrid mode "
            f"(default: {DEFAULT_LOCAL_SPARSE_MODEL})."
        ),
    )
    command_parser.add_argument(
        "--sparse-cache-dir",
        default=str(DEFAULT_FASTEMBED_CACHE_DIR),
        help=(
            "Sparse model cache directory "
            f"(default: {DEFAULT_FASTEMBED_CACHE_DIR})."
        ),
    )
    command_parser.add_argument(
        "--sparse-batch-size",
        type=int,
        default=256,
        help="Texts encoded per sparse batch (default: 256).",
    )
    command_parser.add_argument(
        "--sparse-threads",
        type=int,
        help="Optional FastEmbed CPU thread count.",
    )


def _add_reranking_arguments(command_parser: argparse.ArgumentParser) -> None:
    """Attach optional second-stage candidate and model settings.

    Candidate width remains distinct from final top-k so the first stage can
    overfetch before cross-encoder scoring.
    """
    command_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank a wider candidate set with a local cross-encoder.",
    )
    command_parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Chunks retrieved before optional reranking (default: 20).",
    )
    command_parser.add_argument(
        "--reranker-model",
        default=DEFAULT_LOCAL_RERANKER_MODEL,
        help=(
            "Local cross-encoder model used for reranking "
            f"(default: {DEFAULT_LOCAL_RERANKER_MODEL})."
        ),
    )
    command_parser.add_argument(
        "--reranker-model-revision",
        help="Optional reranker model commit or tag for reproducibility.",
    )
    command_parser.add_argument(
        "--reranker-device",
        default="cpu",
        help="Sentence Transformers reranker device (default: cpu).",
    )
    command_parser.add_argument(
        "--reranker-cache-dir",
        default=str(DEFAULT_RERANKER_CACHE_DIR),
        help=(
            "Reranker model cache directory "
            f"(default: {DEFAULT_RERANKER_CACHE_DIR})."
        ),
    )
    command_parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=16,
        help="Query-chunk pairs scored per batch (default: 16).",
    )
    command_parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=512,
        help="Maximum tokenized query-chunk length (default: 512).",
    )


def _add_generation_arguments(command_parser: argparse.ArgumentParser) -> None:
    """Attach language-model decoding and prompt-budget settings for answers.

    Local selections use all model/device flags. Hosted profiles take model IDs
    from environment configuration while sharing output and prompt limits.
    """
    command_parser.add_argument(
        "--generation-model",
        default="google/flan-t5-small",
        help=(
            "Local Hugging Face generation model; ignored by hosted profiles "
            "(default: google/flan-t5-small)."
        ),
    )
    command_parser.add_argument(
        "--generation-model-revision",
        help="Optional local generation-model commit or tag for reproducibility.",
    )
    command_parser.add_argument(
        "--generation-device",
        default="cpu",
        help="Local generation device: cpu, cuda, or cuda:<index> (default: cpu).",
    )
    command_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens (default: 128).",
    )
    command_parser.add_argument(
        "--temperature",
        type=float,
        help=(
            "Optional sampling temperature from 0 to 2; hosted providers use "
            "their model default when omitted, while local generation uses 0."
        ),
    )
    command_parser.add_argument(
        "--max-context-characters",
        type=int,
        default=1200,
        help="Secondary context character cap (default: 1200).",
    )
    command_parser.add_argument(
        "--max-input-tokens",
        type=int,
        help=(
            "Optional prompt cap; defaults to the local tokenizer limit or the "
            "hosted profile's conservative application limit."
        ),
    )


def _add_embedding_arguments(command_parser: argparse.ArgumentParser) -> None:
    """Attach provider-profile or local dense embedding options.

    Indexing and retrieval share this group because they must use a compatible
    embedding contract. Device, revision, and batching settings affect local
    embeddings, including the Claude profile's local embedding model.
    """
    command_parser.add_argument(
        "--model",
        default=DEFAULT_LOCAL_EMBEDDING_MODEL,
        help=(
            "Model profile alias (gemini, openai, or claude) loaded from .env, "
            "or a local Hugging Face embedding model "
            f"(default: {DEFAULT_LOCAL_EMBEDDING_MODEL})."
        ),
    )
    command_parser.add_argument(
        "--model-revision",
        help="Optional local embedding-model commit or tag for reproducibility.",
    )
    command_parser.add_argument(
        "--device",
        default="cpu",
        help="Local sentence-transformers device (default: cpu).",
    )
    command_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Texts embedded per local inference batch (default: 32).",
    )


def _add_vector_store_location_arguments(
    command_parser: argparse.ArgumentParser,
) -> None:
    command_parser.add_argument(
        "--store-path",
        default=".rag_data/qdrant",
        help="Directory for the local Qdrant database (default: .rag_data/qdrant).",
    )
    command_parser.add_argument(
        "--collection-name",
        default="rag_documents",
        help="Qdrant collection to open (default: rag_documents).",
    )


def _add_chunking_arguments(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000).",
    )
    command_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Target overlap in characters (default: 200).",
    )


def _add_document_input_arguments(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to ingest.",
    )
    command_parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Only scan the top level of provided directories.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and execute one pipeline command.

    Depending on the command, execution may read documents, download/cache and
    run local models, call hosted providers, mutate or query local Qdrant, and
    print results to stdout.
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
        )
        from rag_pipeline.ingestion import load_documents

        try:
            chunking_config = ChunkingConfig(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            embedding_model_config = _build_embedding_model_config(args)
        except (
            InvalidChunkingConfigurationError,
            InvalidEmbeddingConfigurationError,
            InvalidModelProviderConfigurationError,
        ) as exc:
            parser.error(str(exc))

        documents = load_documents(args.paths, recursive=args.recursive)
        chunks = chunk_documents(documents, config=chunking_config)
        service = _create_embedding_service(embedding_model_config, args)
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
            embedding_model_config = _build_embedding_model_config(args)
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
            InvalidModelProviderConfigurationError,
            InvalidVectorStoreConfigurationError,
        ) as exc:
            parser.error(str(exc))

        documents = load_documents(args.paths, recursive=args.recursive)
        chunks = chunk_documents(documents, config=chunking_config)
        embedding_service = _create_embedding_service(
            embedding_model_config,
            args,
        )
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
        )
        from rag_pipeline.exceptions import (
            InvalidRerankingConfigurationError,
            InvalidRetrievalConfigurationError,
            InvalidVectorStoreConfigurationError,
        )
        from rag_pipeline.retrieval import (
            RetrievalConfig,
            RetrieverService,
            parse_metadata_filter,
        )
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
            embedding_model_config = _build_embedding_model_config(args)
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
        except (
            InvalidEmbeddingConfigurationError,
            InvalidModelProviderConfigurationError,
            InvalidVectorStoreConfigurationError,
            InvalidRetrievalConfigurationError,
            InvalidRerankingConfigurationError,
        ) as exc:
            parser.error(str(exc))

        embedding_service = _create_embedding_service(
            embedding_model_config,
            args,
        )
        sparse_embedding_service = (
            create_local_sparse_embedding_service(sparse_embedding_config)
            if sparse_embedding_config is not None
            else None
        )
        with LocalVectorStore(vector_store_config) as vector_store:
            results = RetrieverService(
                embedding_service,
                vector_store,
                sparse_embedding_service,
            ).retrieve(args.query, config=retrieval_config)

        results = _rerank_results(
            args.query,
            results,
            local_config=local_reranker_config,
            config=reranking_config,
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

    if args.command == "answer":
        from rag_pipeline.citations import format_citation
        from rag_pipeline.embeddings import (
            InvalidEmbeddingConfigurationError,
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
        )
        from rag_pipeline.retrieval import (
            RetrievalConfig,
            RetrieverService,
            parse_metadata_filter,
        )
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
            embedding_model_config = _build_embedding_model_config(args)
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
            generation_model_config = _build_generation_model_config(
                args,
                embedding_model_config,
            )
            generation_config = GenerationConfig(
                max_context_characters=args.max_context_characters,
                max_input_tokens=args.max_input_tokens,
            )
        except (
            InvalidEmbeddingConfigurationError,
            InvalidModelProviderConfigurationError,
            InvalidVectorStoreConfigurationError,
            InvalidRetrievalConfigurationError,
            InvalidRerankingConfigurationError,
            InvalidGenerationConfigurationError,
        ) as exc:
            parser.error(str(exc))

        embedding_service = _create_embedding_service(
            embedding_model_config,
            args,
        )
        sparse_embedding_service = (
            create_local_sparse_embedding_service(sparse_embedding_config)
            if sparse_embedding_config is not None
            else None
        )
        with LocalVectorStore(vector_store_config) as vector_store:
            retrieval_results = RetrieverService(
                embedding_service,
                vector_store,
                sparse_embedding_service,
            ).retrieve(args.query, config=retrieval_config)

        retrieval_results = _rerank_results(
            args.query,
            retrieval_results,
            local_config=local_reranker_config,
            config=reranking_config,
        )

        if not retrieval_results:
            print("Answer:")
            print(INSUFFICIENT_CONTEXT_ANSWER)
            return 0

        answer_generator = _create_answer_generator(generation_model_config)
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


def _build_embedding_model_config(
    args: argparse.Namespace,
) -> LocalEmbeddingConfig | ProviderModelProfile:
    """Resolve ``--model`` into either local settings or one dotenv profile.

    Provider aliases load all three profile variables before document or model
    I/O begins. Any other value remains a Hugging Face model name and preserves
    the original local CLI behavior.
    """
    from rag_pipeline.embeddings import LocalEmbeddingConfig
    from rag_pipeline.model_profiles import (
        is_provider_alias,
        load_provider_model_profile,
    )

    if is_provider_alias(args.model):
        profile = load_provider_model_profile(args.model)
        if profile.uses_local_embeddings:
            LocalEmbeddingConfig(
                model_name=profile.embedding_model,
                model_revision=args.model_revision,
                device=args.device,
                batch_size=args.batch_size,
            )
        return profile
    return LocalEmbeddingConfig(
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
    )


def _create_embedding_service(
    model_config: LocalEmbeddingConfig | ProviderModelProfile,
    args: argparse.Namespace,
) -> EmbeddingService:
    """Construct the local or hosted embedding boundary selected by the CLI.

    Provider-client or local-model initialization can allocate resources,
    access caches, or prepare network clients. Claude profiles pass their
    embedding model through the local factory because Anthropic has no native
    embeddings endpoint.
    """
    from rag_pipeline.embeddings import (
        LocalEmbeddingConfig,
        create_local_embedding_service,
        create_profile_embedding_service,
    )
    from rag_pipeline.model_profiles import ProviderModelProfile

    if isinstance(model_config, ProviderModelProfile):
        return create_profile_embedding_service(
            model_config,
            local_device=args.device,
            local_batch_size=args.batch_size,
            local_model_revision=args.model_revision,
        )
    if isinstance(model_config, LocalEmbeddingConfig):
        return create_local_embedding_service(model_config)
    raise TypeError("model_config must contain local settings or a provider profile.")


def _build_generation_model_config(
    args: argparse.Namespace,
    embedding_model_config: LocalEmbeddingConfig | ProviderModelProfile,
) -> LocalGenerationConfig | HostedGenerationConfig:
    """Pair answer generation with the embedding selection used for retrieval.

    A provider profile is reused unchanged so ``--model gemini|openai|claude``
    cannot accidentally mix credentials or generation models. Local embedding
    selections continue to use the independent local generation flags.
    """
    from rag_pipeline.exceptions import InvalidGenerationConfigurationError
    from rag_pipeline.generation import (
        DEFAULT_HOSTED_MODEL_INPUT_TOKENS,
        HostedGenerationConfig,
        LocalGenerationConfig,
    )
    from rag_pipeline.model_profiles import ProviderModelProfile

    if isinstance(embedding_model_config, ProviderModelProfile):
        if (
            args.max_input_tokens is not None
            and args.max_input_tokens > DEFAULT_HOSTED_MODEL_INPUT_TOKENS
        ):
            raise InvalidGenerationConfigurationError(
                "Hosted model profiles use a conservative maximum input limit "
                f"of {DEFAULT_HOSTED_MODEL_INPUT_TOKENS} tokens; "
                "--max-input-tokens may only lower it."
            )
        return HostedGenerationConfig(
            profile=embedding_model_config,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    return LocalGenerationConfig(
        model_name=args.generation_model,
        model_revision=args.generation_model_revision,
        device=args.generation_device,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0 if args.temperature is None else args.temperature,
    )


def _create_answer_generator(
    model_config: LocalGenerationConfig | HostedGenerationConfig,
) -> AnswerGenerator:
    """Construct the local or hosted guarded generation service.

    Hosted client initialization receives the secret from the in-memory profile
    and never writes it back to process environment state. Actual provider I/O
    occurs later when the answer generator invokes generation from evidence.
    """
    from rag_pipeline.generation import (
        HostedGenerationConfig,
        LocalGenerationConfig,
        create_local_answer_generator,
        create_profile_answer_generator,
    )

    if isinstance(model_config, HostedGenerationConfig):
        return create_profile_answer_generator(model_config)
    if isinstance(model_config, LocalGenerationConfig):
        return create_local_answer_generator(model_config)
    raise TypeError("model_config must contain local settings or a provider profile.")


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
