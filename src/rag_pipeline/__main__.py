"""Command entry point for the local RAG pipeline prototype."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from rag_pipeline import __version__
from rag_pipeline.embeddings import DEFAULT_LOCAL_EMBEDDING_MODEL


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the package entry point."""
    parser = argparse.ArgumentParser(
        prog="rag_pipeline",
        description="Run the local RAG pipeline prototype.",
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
    _add_retrieval_arguments(retrieve_parser)

    answer_parser = subparsers.add_parser(
        "answer",
        help="Retrieve context and generate a grounded local answer.",
    )
    answer_parser.add_argument(
        "query",
        help="Natural-language question to answer.",
    )
    _add_embedding_arguments(answer_parser)
    _add_vector_store_location_arguments(answer_parser)
    _add_retrieval_arguments(answer_parser)
    _add_generation_arguments(answer_parser)
    return parser


def _add_retrieval_arguments(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Maximum number of chunks to return (default: 4).",
    )
    command_parser.add_argument(
        "--score-threshold",
        type=float,
        help="Optional minimum cosine similarity from -1 to 1.",
    )


def _add_generation_arguments(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--generation-model",
        default="google/flan-t5-small",
        help="Local Hugging Face generation model (default: google/flan-t5-small).",
    )
    command_parser.add_argument(
        "--generation-model-revision",
        help="Optional generation-model commit or tag for reproducibility.",
    )
    command_parser.add_argument(
        "--generation-device",
        default="cpu",
        help="Generation device: cpu, cuda, or cuda:<index> (default: cpu).",
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
        default=0.0,
        help="Sampling temperature from 0 to 2 (default: 0).",
    )
    command_parser.add_argument(
        "--max-context-characters",
        type=int,
        default=1200,
        help="Maximum retrieved context characters in the prompt (default: 1200).",
    )


def _add_embedding_arguments(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--model",
        default=DEFAULT_LOCAL_EMBEDDING_MODEL,
        help=f"Hugging Face embedding model (default: {DEFAULT_LOCAL_EMBEDDING_MODEL}).",
    )
    command_parser.add_argument(
        "--model-revision",
        help="Optional Hugging Face model commit or tag for reproducibility.",
    )
    command_parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device understood by sentence-transformers (default: cpu).",
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
    """Run the command-line entry point."""
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
        from rag_pipeline.vector_store import LocalVectorStore, VectorStoreConfig

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
            vector_store_config = VectorStoreConfig(
                path=args.store_path,
                collection_name=args.collection_name,
                write_batch_size=args.write_batch_size,
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

        with LocalVectorStore(vector_store_config) as vector_store:
            result = vector_store.index(
                embedded_documents,
                model_identifier=embedding_service.model_identifier,
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
            LocalEmbeddingConfig,
            create_local_embedding_service,
        )
        from rag_pipeline.exceptions import (
            InvalidRetrievalConfigurationError,
            InvalidVectorStoreConfigurationError,
        )
        from rag_pipeline.retrieval import RetrievalConfig, RetrieverService
        from rag_pipeline.vector_store import LocalVectorStore, VectorStoreConfig

        try:
            embedding_config = LocalEmbeddingConfig(
                model_name=args.model,
                model_revision=args.model_revision,
                device=args.device,
                batch_size=args.batch_size,
            )
            vector_store_config = VectorStoreConfig(
                path=args.store_path,
                collection_name=args.collection_name,
            )
            retrieval_config = RetrievalConfig(
                top_k=args.top_k,
                score_threshold=args.score_threshold,
            )
        except (
            InvalidEmbeddingConfigurationError,
            InvalidVectorStoreConfigurationError,
            InvalidRetrievalConfigurationError,
        ) as exc:
            parser.error(str(exc))

        embedding_service = create_local_embedding_service(embedding_config)
        with LocalVectorStore(vector_store_config) as vector_store:
            results = RetrieverService(embedding_service, vector_store).retrieve(
                args.query,
                config=retrieval_config,
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

            print(f"{result.rank}. score={result.score:.4f} {location}")
            print(f"   {_content_preview(result.document.page_content)}")
        return 0

    if args.command == "answer":
        from rag_pipeline.citations import format_citation
        from rag_pipeline.embeddings import (
            InvalidEmbeddingConfigurationError,
            LocalEmbeddingConfig,
            create_local_embedding_service,
        )
        from rag_pipeline.exceptions import (
            InvalidGenerationConfigurationError,
            InvalidRetrievalConfigurationError,
            InvalidVectorStoreConfigurationError,
        )
        from rag_pipeline.generation import (
            INSUFFICIENT_CONTEXT_ANSWER,
            GenerationConfig,
            LocalGenerationConfig,
            create_local_answer_generator,
        )
        from rag_pipeline.retrieval import RetrievalConfig, RetrieverService
        from rag_pipeline.vector_store import LocalVectorStore, VectorStoreConfig

        try:
            embedding_config = LocalEmbeddingConfig(
                model_name=args.model,
                model_revision=args.model_revision,
                device=args.device,
                batch_size=args.batch_size,
            )
            vector_store_config = VectorStoreConfig(
                path=args.store_path,
                collection_name=args.collection_name,
            )
            retrieval_config = RetrievalConfig(
                top_k=args.top_k,
                score_threshold=args.score_threshold,
            )
            local_generation_config = LocalGenerationConfig(
                model_name=args.generation_model,
                model_revision=args.generation_model_revision,
                device=args.generation_device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            generation_config = GenerationConfig(
                max_context_characters=args.max_context_characters,
            )
        except (
            InvalidEmbeddingConfigurationError,
            InvalidVectorStoreConfigurationError,
            InvalidRetrievalConfigurationError,
            InvalidGenerationConfigurationError,
        ) as exc:
            parser.error(str(exc))

        embedding_service = create_local_embedding_service(embedding_config)
        with LocalVectorStore(vector_store_config) as vector_store:
            retrieval_results = RetrieverService(
                embedding_service,
                vector_store,
            ).retrieve(args.query, config=retrieval_config)

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


def _content_preview(content: str, *, max_length: int = 240) -> str:
    preview = " ".join(content.split())
    if len(preview) <= max_length:
        return preview
    return f"{preview[: max_length - 3]}..."


if __name__ == "__main__":
    raise SystemExit(main())
