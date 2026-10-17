"""Exercise source-scoped indexing and explicit directory synchronization."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import DeterministicFakeEmbedding
from pypdf import PdfWriter
from tests.ingestion.test_extraction import write_minimal_pdf

from rag_pipeline.__main__ import main
from rag_pipeline.exceptions import EmbeddingProviderError
from rag_pipeline.infrastructure.embeddings import EmbeddingService
from rag_pipeline.infrastructure.vector_store import LocalVectorStore, VectorStoreConfig


class IndexSynchronizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_service = EmbeddingService(
            DeterministicFakeEmbedding(size=4), model_name="test-model"
        )

    def _index(self, source: Path, store_path: Path, *options: str) -> str:
        output = io.StringIO()
        with (
            patch(
                "rag_pipeline.infrastructure.embeddings.create_local_embedding_service",
                return_value=self.embedding_service,
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(
                main(
                    [
                        "index",
                        str(source),
                        "--store-path",
                        str(store_path),
                        *options,
                    ]
                ),
                0,
            )
        return output.getvalue()

    def _count(self, store_path: Path) -> int:
        with LocalVectorStore(VectorStoreConfig(path=store_path)) as store:
            return store.count()

    def test_shrinking_and_empty_source_remove_old_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "policy.txt"
            store_path = Path(temp_dir) / "qdrant"
            source.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
            self._index(
                source, store_path, "--chunk-size", "10", "--chunk-overlap", "2"
            )
            self.assertEqual(self._count(store_path), 3)

            source.write_text("brief", encoding="utf-8")
            output = self._index(
                source, store_path, "--chunk-size", "10", "--chunk-overlap", "2"
            )
            self.assertEqual(self._count(store_path), 1)
            self.assertIn("Removed 2 obsolete point(s).", output)

            source.write_text("", encoding="utf-8")
            self._index(source, store_path)
            self.assertEqual(self._count(store_path), 0)

    def test_chunk_size_change_replaces_stale_points(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "policy.txt"
            store_path = Path(temp_dir) / "qdrant"
            source.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
            self._index(
                source, store_path, "--chunk-size", "10", "--chunk-overlap", "2"
            )
            self._index(
                source, store_path, "--chunk-size", "100", "--chunk-overlap", "0"
            )
            self.assertEqual(self._count(store_path), 1)

    def test_reindexing_shorter_pdf_removes_page_points(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_page = root / "first.pdf"
            second_page = root / "second.pdf"
            source = root / "policy.pdf"
            store_path = root / "qdrant"
            write_minimal_pdf(first_page, "First page policy")
            write_minimal_pdf(second_page, "Second page policy")
            writer = PdfWriter()
            writer.append(str(first_page))
            writer.append(str(second_page))
            writer.write(str(source))
            writer.close()

            self._index(source, store_path)
            self.assertEqual(self._count(store_path), 2)
            writer = PdfWriter()
            writer.append(str(first_page))
            writer.write(str(source))
            writer.close()
            self._index(source, store_path)
            self.assertEqual(self._count(store_path), 1)

    def test_directory_pruning_is_opt_in_and_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected = root / "docs"
            sibling = root / "docs-extra"
            selected.mkdir()
            sibling.mkdir()
            first = selected / "first.txt"
            removed = selected / "removed.txt"
            outside = sibling / "outside.txt"
            for path in (first, removed, outside):
                path.write_text(path.stem, encoding="utf-8")
            store_path = root / "qdrant"
            self._index(selected, store_path)
            self._index(outside, store_path)
            self.assertEqual(self._count(store_path), 3)

            removed.unlink()
            self._index(selected, store_path)
            self.assertEqual(self._count(store_path), 3)
            self._index(selected, store_path, "--prune-missing")
            self.assertEqual(self._count(store_path), 2)
            with LocalVectorStore(VectorStoreConfig(path=store_path)) as store:
                self.assertEqual(store.list_sources(), {str(first), str(outside)})

            first.unlink()
            self._index(selected, store_path, "--prune-missing")
            self.assertEqual(self._count(store_path), 1)

    def test_pruning_requires_one_recursive_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "policy.txt"
            source.write_text("Policy", encoding="utf-8")
            store_path = root / "qdrant"
            for arguments in (
                [str(source), "--prune-missing"],
                [str(root), "--prune-missing", "--no-recursive"],
                [str(root), str(source), "--prune-missing"],
            ):
                with self.subTest(arguments=arguments):
                    error_output = io.StringIO()
                    with redirect_stderr(error_output), self.assertRaises(SystemExit):
                        main(["index", *arguments, "--store-path", str(store_path)])
                    self.assertIn("prune_missing requires", error_output.getvalue())
            self.assertFalse(store_path.exists())

    def test_extraction_and_embedding_failures_leave_index_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "policy.txt"
            store_path = Path(temp_dir) / "qdrant"
            source.write_text("Original policy", encoding="utf-8")
            self._index(source, store_path)
            source.write_text("Replacement policy", encoding="utf-8")
            with (
                patch(
                    "rag_pipeline.application.indexing.load_documents",
                    side_effect=OSError,
                ),
                self.assertRaises(OSError),
            ):
                self._index(source, store_path)
            self.assertEqual(self._count(store_path), 1)
            with (
                patch.object(
                    self.embedding_service,
                    "embed_documents",
                    side_effect=EmbeddingProviderError("inference failed"),
                ),
                self.assertRaises(EmbeddingProviderError),
            ):
                self._index(source, store_path)
            self.assertEqual(self._count(store_path), 1)


if __name__ == "__main__":
    unittest.main()
