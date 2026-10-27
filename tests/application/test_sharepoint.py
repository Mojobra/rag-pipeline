"""Verify remote staging, provenance, and source-scoped replacement offline."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_pipeline.__main__ import build_parser, main
from rag_pipeline.application.indexing import IndexingPipelineConfig
from rag_pipeline.application.sharepoint import index_sharepoint_drive
from rag_pipeline.exceptions import SharePointInputError
from rag_pipeline.infrastructure.embeddings import (
    EmbeddingService,
    LocalEmbeddingConfig,
)
from rag_pipeline.infrastructure.sharepoint import (
    SharePointFile,
    SharePointImportLimits,
)
from rag_pipeline.infrastructure.vector_store import (
    IndexingResult,
    LocalVectorStore,
    SearchMode,
    VectorStoreConfig,
)
from rag_pipeline.ingestion.chunking import ChunkingConfig


class FakeDriveReader:
    """Provide one mutable Graph snapshot without credentials or HTTP."""

    def __init__(self, entries: list[tuple[SharePointFile, bytes]]) -> None:
        self.entries = entries

    def list_files(self, _limits: SharePointImportLimits) -> list[SharePointFile]:
        return [file for file, _contents in self.entries]

    def download_file(self, file: SharePointFile, *, max_bytes: int) -> bytes:
        del max_bytes
        return next(contents for listed, contents in self.entries if listed == file)


class SharePointImportTests(unittest.TestCase):
    """Assert extraction completes before writes and keeps stable remote sources."""

    def setUp(self) -> None:
        self.embedding_service = EmbeddingService(
            DeterministicFakeEmbedding(size=4), model_name="test-model"
        )

    def _config(self, store_path: Path) -> IndexingPipelineConfig:
        return IndexingPipelineConfig(
            chunking=ChunkingConfig(chunk_size=10, chunk_overlap=0),
            embedding=LocalEmbeddingConfig(),
            vector_store=VectorStoreConfig(
                path=store_path, collection_name="sharepoint_documents"
            ),
        )

    def test_reimport_replaces_renamed_item_but_not_missing_item(self) -> None:
        first = SharePointFile("drive", "first", "old.txt", 26)
        second = SharePointFile("drive", "second", "second.txt", 6)
        reader = FakeDriveReader(
            [(first, b"abcdefghijklmnopqrstuvwxyz"), (second, b"second")]
        )
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory) / "store")
            with patch(
                "rag_pipeline.infrastructure.embeddings.create_local_embedding_service",
                return_value=self.embedding_service,
            ):
                first_result = index_sharepoint_drive(
                    reader, config=config, limits=SharePointImportLimits()
                )
                self.assertEqual(first_result.total_count, 4)
                reader.entries = [
                    (SharePointFile("drive", "first", "renamed.txt", 5), b"short")
                ]
                second_result = index_sharepoint_drive(
                    reader, config=config, limits=SharePointImportLimits()
                )
            self.assertEqual(second_result.total_count, 2)
            self.assertEqual(second_result.removed_count, 2)
            with LocalVectorStore(config.vector_store) as store:
                self.assertEqual(
                    store.list_sources(),
                    {"sharepoint://drive/first", "sharepoint://drive/second"},
                )

    def test_provenance_replaces_temporary_path(self) -> None:
        file = SharePointFile(
            "drive", "item", "Policy.md", 6, "https://tenant.sharepoint.com/policy"
        )
        reader = FakeDriveReader([(file, b"Policy")])
        captured: list[Document] = []

        def capture(documents: list[Document], **_kwargs: object) -> object:
            captured.extend(documents)
            return object()

        with patch(
            "rag_pipeline.application.sharepoint.index_extracted_documents",
            side_effect=capture,
        ):
            index_sharepoint_drive(
                reader,
                config=self._config(Path("unused")),
                limits=SharePointImportLimits(),
            )
        self.assertEqual(len(captured), 1)
        metadata = captured[0].metadata
        self.assertEqual(metadata["source"], "sharepoint://drive/item")
        self.assertEqual(metadata["file_name"], "Policy.md")
        self.assertEqual(metadata["sharepoint_item_id"], "item")
        self.assertEqual(
            metadata["sharepoint_web_url"], "https://tenant.sharepoint.com/policy"
        )
        self.assertNotIn("rag-sharepoint-", str(metadata))

    def test_failed_download_does_not_create_collection(self) -> None:
        reader = FakeDriveReader(
            [
                (SharePointFile("drive", "first", "a.txt", 1), b"a"),
                (SharePointFile("drive", "second", "b.txt", 1), b"too large"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "store"
            with self.assertRaises(SharePointInputError):
                index_sharepoint_drive(
                    reader,
                    config=self._config(store_path),
                    limits=SharePointImportLimits(max_file_bytes=2),
                )
            self.assertFalse(store_path.exists())


class SharePointCliTests(unittest.TestCase):
    """Keep risk acknowledgement and collection separation visible at the CLI."""

    def test_acknowledgement_is_required_before_credentials(self) -> None:
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit):
            main(["index-sharepoint", "drive"])
        self.assertIn("--acknowledge-no-acl", error.getvalue())

    def test_default_collection_and_positive_limits(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["index-sharepoint", "drive", "--acknowledge-no-acl"])
        self.assertEqual(args.collection_name, "sharepoint_documents")
        self.assertEqual(args.max_files, 100)
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit):
            main(
                [
                    "index-sharepoint",
                    "drive",
                    "--acknowledge-no-acl",
                    "--max-files",
                    "0",
                ]
            )
        self.assertIn("max_files", error.getvalue())

    def test_success_path_passes_default_collection_to_import(self) -> None:
        output = io.StringIO()
        result = IndexingResult(
            collection_name="sharepoint_documents",
            indexed_count=2,
            total_count=2,
            point_ids=("one", "two"),
            embedding_model="test-model",
            embedding_dimension=4,
            search_mode=SearchMode.DENSE,
            sparse_embedding_model=None,
        )
        with (
            patch(
                "rag_pipeline.infrastructure.sharepoint.create_sharepoint_client",
                return_value=nullcontext(object()),
            ),
            patch(
                "rag_pipeline.application.sharepoint.index_sharepoint_drive",
                return_value=result,
            ) as import_drive,
            redirect_stdout(output),
        ):
            self.assertEqual(
                main(["index-sharepoint", "drive", "--acknowledge-no-acl"]), 0
            )
        self.assertEqual(
            import_drive.call_args.kwargs["config"].vector_store.collection_name,
            "sharepoint_documents",
        )
        self.assertIn("Indexed 2 SharePoint chunk(s)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
