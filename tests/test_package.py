from __future__ import annotations

import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


class PromptTokenizerStub:
    model_max_length = 2000

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        verbose: bool = False,
    ) -> list[int]:
        special_tokens = 1 if add_special_tokens else 0
        return [0] * (len(text) + special_tokens)


class PackageSmokeTests(unittest.TestCase):
    def test_package_exposes_semantic_version(self) -> None:
        import rag_pipeline

        self.assertRegex(rag_pipeline.__version__, re.compile(r"^\d+\.\d+\.\d+$"))

    def test_module_entry_point_runs(self) -> None:
        from rag_pipeline.__main__ import main

        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("RAG Pipeline skeleton is ready.", output.getvalue())

    def test_answer_has_quality_gate_while_retrieve_remains_diagnostic(self) -> None:
        from rag_pipeline.__main__ import build_parser

        parser = build_parser()

        answer_args = parser.parse_args(["answer", "Question"])
        retrieve_args = parser.parse_args(["retrieve", "Question"])
        hybrid_index_args = parser.parse_args(
            ["index", "documents", "--search-mode", "hybrid"]
        )

        self.assertEqual(answer_args.score_threshold, 0.2)
        self.assertIsNone(retrieve_args.score_threshold)
        self.assertIsNone(answer_args.metadata_filters)
        self.assertIsNone(retrieve_args.metadata_filters)
        self.assertEqual(answer_args.search_mode, "dense")
        self.assertEqual(retrieve_args.search_mode, "dense")
        self.assertEqual(hybrid_index_args.search_mode, "hybrid")

    def test_ingest_command_reports_loaded_documents(self) -> None:
        from rag_pipeline.__main__ import main

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "notes.txt"
            file_path.write_text("Local RAG note.", encoding="utf-8")

            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(["ingest", str(file_path)])

        self.assertEqual(exit_code, 0)
        self.assertIn("Ingested 1 document(s).", output.getvalue())

    def test_chunk_command_reports_created_chunks(self) -> None:
        from rag_pipeline.__main__ import main

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "alphabet.txt"
            file_path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "chunk",
                        "--chunk-size",
                        "10",
                        "--chunk-overlap",
                        "2",
                        str(file_path),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "Chunked 1 document(s) into 3 chunk(s).",
            output.getvalue(),
        )

    def test_chunk_experiment_command_compares_candidates_as_json(self) -> None:
        from rag_pipeline.__main__ import main

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "alphabet.txt"
            file_path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "chunk-experiment",
                        "--candidate",
                        "10:3",
                        "--candidate",
                        "13:0",
                        "--output-format",
                        "json",
                        str(file_path),
                    ]
                )

        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["input_document_count"], 1)
        self.assertEqual(report["candidates"][0]["chunk_count"], 4)
        self.assertEqual(report["candidates"][0]["duplicated_characters"], 9)
        self.assertEqual(report["candidates"][1]["chunk_count"], 2)

    def test_chunk_experiment_command_uses_readable_table_by_default(self) -> None:
        from rag_pipeline.__main__ import main

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "alphabet.txt"
            file_path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "chunk-experiment",
                        "--candidate",
                        "10:3",
                        str(file_path),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Chunking experiment", output.getvalue())
        self.assertIn("9 (25.7%)", output.getvalue())

    def test_embed_command_reports_vector_dimension_without_downloading_model(
        self,
    ) -> None:
        from langchain_core.embeddings import DeterministicFakeEmbedding

        from rag_pipeline.__main__ import main
        from rag_pipeline.embeddings import EmbeddingService

        service = EmbeddingService(
            DeterministicFakeEmbedding(size=4),
            model_name="test-embedding-model",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "alphabet.txt"
            file_path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
            output = io.StringIO()

            with patch(
                "rag_pipeline.embeddings.create_local_embedding_service",
                return_value=service,
            ):
                with redirect_stdout(output):
                    exit_code = main(
                        [
                            "embed",
                            "--chunk-size",
                            "10",
                            "--chunk-overlap",
                            "2",
                            str(file_path),
                        ]
                    )

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "Embedded 3 chunk(s) into 4-dimensional vectors using "
            "test-embedding-model.",
            output.getvalue(),
        )

    def test_index_command_persists_vectors_without_downloading_model(self) -> None:
        from langchain_core.embeddings import DeterministicFakeEmbedding

        from rag_pipeline.__main__ import main
        from rag_pipeline.embeddings import EmbeddingService
        from rag_pipeline.vector_store import LocalVectorStore, VectorStoreConfig

        service = EmbeddingService(
            DeterministicFakeEmbedding(size=4),
            model_name="test-embedding-model",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "alphabet.txt"
            file_path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
            store_path = Path(temp_dir) / "qdrant"
            output = io.StringIO()

            with patch(
                "rag_pipeline.embeddings.create_local_embedding_service",
                return_value=service,
            ):
                with redirect_stdout(output):
                    exit_code = main(
                        [
                            "index",
                            "--chunk-size",
                            "10",
                            "--chunk-overlap",
                            "2",
                            "--store-path",
                            str(store_path),
                            str(file_path),
                        ]
                    )

            with LocalVectorStore(
                VectorStoreConfig(path=store_path)
            ) as reopened_store:
                stored_count = reopened_store.count()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stored_count, 3)
        self.assertIn(
            "Indexed 3 chunk(s) into 'rag_documents'; collection now "
            "contains 3 point(s).",
            output.getvalue(),
        )

    def test_hybrid_index_and_retrieve_commands_use_rrf_without_downloads(
        self,
    ) -> None:
        from langchain_core.embeddings import Embeddings
        from langchain_qdrant import SparseEmbeddings, SparseVector

        from rag_pipeline.__main__ import main
        from rag_pipeline.embeddings import EmbeddingService
        from rag_pipeline.sparse_embeddings import SparseEmbeddingService

        class HybridDenseEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [
                    [0.0, 1.0] if "ZX-42" in text else [1.0, 0.0]
                    for text in texts
                ]

            def embed_query(self, text: str) -> list[float]:
                return [1.0, 0.0]

        class HybridSparseEmbeddings(SparseEmbeddings):
            def embed_documents(self, texts: list[str]) -> list[SparseVector]:
                return [self._embed(text) for text in texts]

            def embed_query(self, text: str) -> SparseVector:
                return self._embed(text)

            @staticmethod
            def _embed(text: str) -> SparseVector:
                if "zx-42" in text.lower():
                    return SparseVector(indices=[42], values=[1.0])
                return SparseVector(indices=[], values=[])

        dense_service = EmbeddingService(
            HybridDenseEmbeddings(),
            model_name="test-dense-model",
        )
        sparse_service = SparseEmbeddingService(
            HybridSparseEmbeddings(),
            model_name="test-sparse-model",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            document_dir = Path(temp_dir) / "documents"
            document_dir.mkdir()
            (document_dir / "exact.txt").write_text(
                "Repair code ZX-42 requires approval.",
                encoding="utf-8",
            )
            (document_dir / "semantic.txt").write_text(
                "Conceptually related equipment policy.",
                encoding="utf-8",
            )
            store_path = Path(temp_dir) / "qdrant"

            with patch(
                "rag_pipeline.embeddings.create_local_embedding_service",
                return_value=dense_service,
            ):
                with patch(
                    "rag_pipeline.sparse_embeddings."
                    "create_local_sparse_embedding_service",
                    return_value=sparse_service,
                ):
                    index_output = io.StringIO()
                    with redirect_stdout(index_output):
                        index_exit_code = main(
                            [
                                "index",
                                str(document_dir),
                                "--store-path",
                                str(store_path),
                                "--collection-name",
                                "hybrid-policies",
                                "--search-mode",
                                "hybrid",
                            ]
                        )

                    retrieve_output = io.StringIO()
                    with redirect_stdout(retrieve_output):
                        retrieve_exit_code = main(
                            [
                                "retrieve",
                                "What does ZX-42 require?",
                                "--store-path",
                                str(store_path),
                                "--collection-name",
                                "hybrid-policies",
                                "--search-mode",
                                "hybrid",
                                "--top-k",
                                "2",
                            ]
                        )

        retrieval_text = retrieve_output.getvalue()
        self.assertEqual(index_exit_code, 0)
        self.assertEqual(retrieve_exit_code, 0)
        self.assertIn("Indexed 2 chunk(s)", index_output.getvalue())
        self.assertIn("score_kind=rrf", retrieval_text)
        self.assertLess(
            retrieval_text.index("exact.txt"),
            retrieval_text.index("semantic.txt"),
        )

    def test_retrieve_command_reports_ranked_evidence_without_model_download(
        self,
    ) -> None:
        from langchain_core.documents import Document
        from langchain_core.embeddings import DeterministicFakeEmbedding

        from rag_pipeline.__main__ import main
        from rag_pipeline.embeddings import EmbeddingService
        from rag_pipeline.vector_store import LocalVectorStore, VectorStoreConfig

        service = EmbeddingService(
            DeterministicFakeEmbedding(size=4),
            model_name="test-embedding-model",
        )
        documents = [
            Document(
                page_content="Expense claims require receipts.",
                metadata={
                    "source": "expenses.txt",
                    "chunk_index": 0,
                    "department": "finance",
                },
            ),
            Document(
                page_content="Annual leave requests use the HR portal.",
                metadata={
                    "source": "leave.txt",
                    "chunk_index": 0,
                    "department": "hr",
                },
            ),
        ]
        embedded_documents = service.embed_documents(documents)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = VectorStoreConfig(
                path=Path(temp_dir) / "qdrant",
                collection_name="policies",
            )
            with LocalVectorStore(config) as store:
                store.index(
                    embedded_documents,
                    model_identifier=service.model_identifier,
                )

            output = io.StringIO()
            with patch(
                "rag_pipeline.embeddings.create_local_embedding_service",
                return_value=service,
            ):
                with redirect_stdout(output):
                    exit_code = main(
                        [
                            "retrieve",
                            "Expense claims require receipts.",
                            "--store-path",
                            str(config.resolved_path),
                            "--collection-name",
                            "policies",
                            "--top-k",
                            "2",
                            "--filter",
                            "department=finance",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        self.assertIn("1. score=1.0000 source=expenses.txt chunk=0", output.getvalue())
        self.assertIn("Expense claims require receipts.", output.getvalue())
        self.assertNotIn("Annual leave", output.getvalue())

    def test_answer_command_generates_from_retrieved_context_without_downloads(
        self,
    ) -> None:
        from langchain_core.documents import Document
        from langchain_core.embeddings import DeterministicFakeEmbedding
        from langchain_core.language_models.fake import FakeListLLM

        from rag_pipeline.__main__ import main
        from rag_pipeline.embeddings import EmbeddingService
        from rag_pipeline.generation import AnswerGenerator
        from rag_pipeline.vector_store import LocalVectorStore, VectorStoreConfig

        embedding_service = EmbeddingService(
            DeterministicFakeEmbedding(size=4),
            model_name="test-embedding-model",
        )
        documents = [
            Document(
                page_content="Expense claims require receipts.",
                metadata={
                    "source": "expenses.txt",
                    "chunk_index": 0,
                    "department": "finance",
                },
            ),
            Document(
                page_content="Annual leave requests use the HR portal.",
                metadata={
                    "source": "leave.txt",
                    "chunk_index": 0,
                    "department": "hr",
                },
            ),
        ]
        embedded_documents = embedding_service.embed_documents(documents)
        answer_generator = AnswerGenerator(
            FakeListLLM(responses=["Receipts are required."]),
            model_identifier="test-generation-model",
            tokenizer=PromptTokenizerStub(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = VectorStoreConfig(
                path=Path(temp_dir) / "qdrant",
                collection_name="policies",
            )
            with LocalVectorStore(config) as store:
                store.index(
                    embedded_documents,
                    model_identifier=embedding_service.model_identifier,
                )

            output = io.StringIO()
            with patch(
                "rag_pipeline.embeddings.create_local_embedding_service",
                return_value=embedding_service,
            ):
                with patch(
                    "rag_pipeline.generation.create_local_answer_generator",
                    return_value=answer_generator,
                ):
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "answer",
                                "Expense claims require receipts.",
                                "--store-path",
                                str(config.resolved_path),
                                "--collection-name",
                                "policies",
                                "--top-k",
                                "1",
                                "--filter",
                                "department=finance",
                            ]
                        )

        self.assertEqual(exit_code, 0)
        self.assertIn("Answer:\nReceipts are required.", output.getvalue())
        self.assertIn("Sources:\n[1] expenses.txt (chunk 1)", output.getvalue())
        self.assertIn("Expense claims require receipts.", output.getvalue())

    def test_answer_command_default_gate_skips_irrelevant_retrieval(self) -> None:
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings

        from rag_pipeline.__main__ import main
        from rag_pipeline.embeddings import EmbeddingService
        from rag_pipeline.generation import INSUFFICIENT_CONTEXT_ANSWER
        from rag_pipeline.vector_store import LocalVectorStore, VectorStoreConfig

        class OrthogonalEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return [0.0, 1.0]

        embedding_service = EmbeddingService(
            OrthogonalEmbeddings(), model_name="test-embedding-model"
        )
        embedded_documents = embedding_service.embed_documents(
            [Document(page_content="Expense claims require receipts.")]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = VectorStoreConfig(path=Path(temp_dir) / "qdrant")
            with LocalVectorStore(config) as store:
                store.index(
                    embedded_documents,
                    model_identifier=embedding_service.model_identifier,
                )

            output = io.StringIO()
            with patch(
                "rag_pipeline.embeddings.create_local_embedding_service",
                return_value=embedding_service,
            ):
                with patch(
                    "rag_pipeline.generation.create_local_answer_generator"
                ) as generation_factory:
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "answer",
                                "An unrelated question",
                                "--store-path",
                                str(config.resolved_path),
                            ]
                        )

        self.assertEqual(exit_code, 0)
        self.assertIn(INSUFFICIENT_CONTEXT_ANSWER, output.getvalue())
        self.assertNotIn("Sources:", output.getvalue())
        generation_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
