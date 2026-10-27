"""Verify Drive staging, stable provenance, CLI gates, and no-write failures."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_pipeline.__main__ import build_parser, main
from rag_pipeline.application.google_drive import index_google_drive_folder
from rag_pipeline.application.indexing import IndexingPipelineConfig
from rag_pipeline.exceptions import GoogleDriveInputError
from rag_pipeline.infrastructure.embeddings import (
    EmbeddingService,
    LocalEmbeddingConfig,
)
from rag_pipeline.infrastructure.google_drive import (
    GoogleDriveFile,
    GoogleDriveImportLimits,
)
from rag_pipeline.infrastructure.vector_store import LocalVectorStore, VectorStoreConfig
from rag_pipeline.ingestion.chunking import ChunkingConfig

_ROOT = "folder_123"


def _file(file_id: str, name: str = "policy.txt") -> GoogleDriveFile:
    return GoogleDriveFile(_ROOT, file_id, name, "text/plain", 5)


class FakeDriveReader:
    """Supply a mutable Drive snapshot without credentials or HTTP."""

    def __init__(
        self, files: list[GoogleDriveFile], contents: bytes = b"hello"
    ) -> None:
        self.files = files
        self.contents = contents

    def list_files(self, _limits: GoogleDriveImportLimits) -> list[GoogleDriveFile]:
        return self.files

    def download_file(self, _file: GoogleDriveFile, *, max_bytes: int) -> bytes:
        return self.contents


class GoogleDriveImportTests(unittest.TestCase):
    """Ensure full staging precedes embedding and stable IDs drive replacement."""

    def _config(self, store_path: Path) -> IndexingPipelineConfig:
        return IndexingPipelineConfig(
            chunking=ChunkingConfig(chunk_size=10, chunk_overlap=0),
            embedding=LocalEmbeddingConfig(),
            vector_store=VectorStoreConfig(
                path=store_path, collection_name="google_drive_documents"
            ),
        )

    def test_reimport_replaces_renamed_source_but_not_missing_file(self) -> None:
        reader = FakeDriveReader(
            [_file("a"), _file("b")], b"abcdefghijklmnopqrstuvwxyz"
        )
        embeddings = EmbeddingService(
            DeterministicFakeEmbedding(size=4), model_name="test-model"
        )
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory) / "store")
            with patch(
                "rag_pipeline.infrastructure.embeddings.create_local_embedding_service",
                return_value=embeddings,
            ):
                first = index_google_drive_folder(
                    reader, config=config, limits=GoogleDriveImportLimits()
                )
                reader.files = [_file("a", "new-name.txt")]
                reader.contents = b"short"
                second = index_google_drive_folder(
                    reader, config=config, limits=GoogleDriveImportLimits()
                )
            self.assertGreater(first.total_count, second.total_count)
            self.assertGreater(second.removed_count, 0)
            with LocalVectorStore(config.vector_store) as store:
                self.assertEqual(store.list_sources(), {"gdrive://a", "gdrive://b"})

    def test_google_doc_export_uses_text_parser_and_provenance(self) -> None:
        reader = FakeDriveReader(
            [
                GoogleDriveFile(
                    _ROOT,
                    "doc_1",
                    "Expense policy",
                    "application/vnd.google-apps.document",
                    None,
                )
            ],
            b"Expense claims within 30 days.",
        )
        captured: list[Document] = []

        def capture(documents: list[Document], **_kwargs: object) -> object:
            captured.extend(documents)
            return object()

        with patch(
            "rag_pipeline.application.google_drive.index_extracted_documents",
            side_effect=capture,
        ):
            index_google_drive_folder(
                reader,
                config=self._config(Path("unused")),
                limits=GoogleDriveImportLimits(),
            )
        self.assertEqual(len(captured), 1)
        self.assertIn("Expense claims", captured[0].page_content)
        self.assertEqual(captured[0].metadata["source"], "gdrive://doc_1")
        self.assertEqual(captured[0].metadata["file_name"], "Expense policy")
        self.assertEqual(captured[0].metadata["google_drive_folder_id"], _ROOT)

    def test_oversized_content_never_creates_collection(self) -> None:
        reader = FakeDriveReader([_file("a")], b"far too much content")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store"
            with self.assertRaises(GoogleDriveInputError):
                index_google_drive_folder(
                    reader,
                    config=self._config(path),
                    limits=GoogleDriveImportLimits(max_file_bytes=5),
                )
            self.assertFalse(path.exists())


class GoogleDriveCliTests(unittest.TestCase):
    """Require operator acknowledgement before loading ADC or models."""

    def test_help_and_default_collection(self) -> None:
        args = build_parser().parse_args(
            ["index-google-drive", _ROOT, "--acknowledge-no-acl"]
        )
        self.assertEqual(args.collection_name, "google_drive_documents")
        self.assertEqual(args.max_files, 100)
        self.assertEqual(args.max_file_mib, 20)
        self.assertEqual(args.max_total_mib, 100)

    def test_acknowledgement_precedes_credentials(self) -> None:
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit):
            main(["index-google-drive", _ROOT])
        self.assertIn("--acknowledge-no-acl", error.getvalue())

    def test_invalid_limit_and_missing_adc_are_actionable(self) -> None:
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit):
            main(
                [
                    "index-google-drive",
                    _ROOT,
                    "--acknowledge-no-acl",
                    "--max-files",
                    "0",
                ]
            )
        self.assertIn("max_files", error.getvalue())
        error = io.StringIO()
        with (
            patch(
                "rag_pipeline.infrastructure.google_drive.create_google_drive_client",
                side_effect=GoogleDriveInputError("Drive credentials unavailable."),
            ),
            redirect_stderr(error),
            self.assertRaises(SystemExit),
        ):
            main(["index-google-drive", _ROOT, "--acknowledge-no-acl"])
        self.assertIn("credentials unavailable", error.getvalue())

    def test_success_path_uses_requested_folder(self) -> None:
        output = io.StringIO()
        reader = FakeDriveReader([_file("a")])

        class ReaderContext:
            def __enter__(self) -> FakeDriveReader:
                return reader

            def __exit__(self, *_exc: object) -> None:
                pass

        with (
            patch(
                "rag_pipeline.infrastructure.google_drive.create_google_drive_client",
                return_value=ReaderContext(),
            ) as create,
            patch(
                "rag_pipeline.application.google_drive.index_google_drive_folder"
            ) as index,
            redirect_stdout(output),
        ):
            index.return_value.indexed_count = 1
            index.return_value.collection_name = "google_drive_documents"
            index.return_value.total_count = 1
            index.return_value.removed_count = 0
            self.assertEqual(
                main(["index-google-drive", _ROOT, "--acknowledge-no-acl"]), 0
            )
        create.assert_called_once_with(_ROOT)
        self.assertIn("Indexed 1 Google Drive chunk", output.getvalue())
