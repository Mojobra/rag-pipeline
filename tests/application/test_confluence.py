"""Verify bounded Confluence staging, provenance, replacement, and CLI paths."""

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
from rag_pipeline.application.confluence import index_confluence_space
from rag_pipeline.application.indexing import IndexingPipelineConfig
from rag_pipeline.exceptions import ConfluenceInputError
from rag_pipeline.infrastructure.confluence import (
    ConfluenceImportLimits,
    ConfluencePage,
)
from rag_pipeline.infrastructure.embeddings import (
    EmbeddingService,
    LocalEmbeddingConfig,
)
from rag_pipeline.infrastructure.vector_store import (
    IndexingResult,
    LocalVectorStore,
    SearchMode,
    VectorStoreConfig,
)
from rag_pipeline.ingestion.chunking import ChunkingConfig


def _page(page_id: str, title: str, body: str) -> ConfluencePage:
    return ConfluencePage(
        "team.atlassian.net",
        "42",
        page_id,
        title,
        body,
        3,
        f"https://team.atlassian.net/wiki/spaces/OPS/pages/{page_id}",
    )


class FakeSpaceReader:
    """Supply a mutable page snapshot without HTTP or credentials."""

    def __init__(self, pages: list[ConfluencePage]) -> None:
        self.pages = pages

    def list_pages(
        self, _space_id: str, _limits: ConfluenceImportLimits
    ) -> list[ConfluencePage]:
        return self.pages


class ConfluenceImportTests(unittest.TestCase):
    """Preserve page IDs, normalized evidence, and no-write failure behavior."""

    def setUp(self) -> None:
        self.embedding_service = EmbeddingService(
            DeterministicFakeEmbedding(size=4), model_name="test-model"
        )

    def _config(self, store_path: Path) -> IndexingPipelineConfig:
        return IndexingPipelineConfig(
            chunking=ChunkingConfig(chunk_size=10, chunk_overlap=0),
            embedding=LocalEmbeddingConfig(),
            vector_store=VectorStoreConfig(
                path=store_path, collection_name="confluence_documents"
            ),
        )

    def test_reimport_replaces_renamed_page_but_not_missing_page(self) -> None:
        reader = FakeSpaceReader(
            [
                _page("100", "Old title", "<p>abcdefghijklmnopqrstuvwxyz</p>"),
                _page("200", "Other", "<p>Second page</p>"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory) / "store")
            with patch(
                "rag_pipeline.infrastructure.embeddings.create_local_embedding_service",
                return_value=self.embedding_service,
            ):
                first = index_confluence_space(
                    reader,
                    space_id="42",
                    config=config,
                    limits=ConfluenceImportLimits(),
                )
                reader.pages = [_page("100", "New title", "<p>Brief</p>")]
                second = index_confluence_space(
                    reader,
                    space_id="42",
                    config=config,
                    limits=ConfluenceImportLimits(),
                )
            self.assertGreater(first.total_count, second.total_count)
            self.assertGreater(second.removed_count, 0)
            with LocalVectorStore(config.vector_store) as store:
                self.assertEqual(
                    store.list_sources(),
                    {
                        "confluence://team.atlassian.net/100",
                        "confluence://team.atlassian.net/200",
                    },
                )

    def test_html_normalization_title_and_provenance(self) -> None:
        reader = FakeSpaceReader(
            [
                _page(
                    "100",
                    "Expense policy",
                    "<script>secret()</script><p>Claim in 30 days.</p>",
                )
            ]
        )
        captured: list[Document] = []

        def capture(documents: list[Document], **_kwargs: object) -> object:
            captured.extend(documents)
            return object()

        with patch(
            "rag_pipeline.application.confluence.index_extracted_documents",
            side_effect=capture,
        ):
            index_confluence_space(
                reader,
                space_id="42",
                config=self._config(Path("unused")),
                limits=ConfluenceImportLimits(),
            )
        self.assertEqual(len(captured), 1)
        self.assertIn("Expense policy", captured[0].page_content)
        self.assertIn("Claim in 30 days.", captured[0].page_content)
        self.assertNotIn("secret()", captured[0].page_content)
        self.assertEqual(
            captured[0].metadata["source"], "confluence://team.atlassian.net/100"
        )
        self.assertEqual(captured[0].metadata["confluence_version"], 3)
        self.assertEqual(
            captured[0].metadata["confluence_web_url"],
            "https://team.atlassian.net/wiki/spaces/OPS/pages/100",
        )

    def test_invalid_snapshot_does_not_create_collection(self) -> None:
        reader = FakeSpaceReader(
            [_page("100", "First", "<p>Short</p>"), _page("200", "Second", "x" * 20)]
        )
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "store"
            with self.assertRaises(ConfluenceInputError):
                index_confluence_space(
                    reader,
                    space_id="42",
                    config=self._config(store_path),
                    limits=ConfluenceImportLimits(max_page_bytes=15),
                )
            self.assertFalse(store_path.exists())


class ConfluenceCliTests(unittest.TestCase):
    """Gate operator risk before credentials and preserve separate defaults."""

    def test_acknowledgement_precedes_credentials(self) -> None:
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit):
            main(["index-confluence", "42"])
        self.assertIn("--acknowledge-no-acl", error.getvalue())

    def test_default_collection_and_limit_validation(self) -> None:
        args = build_parser().parse_args(
            ["index-confluence", "42", "--acknowledge-no-acl"]
        )
        self.assertEqual(args.collection_name, "confluence_documents")
        self.assertEqual(args.max_pages, 100)
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit):
            main(
                [
                    "index-confluence",
                    "42",
                    "--acknowledge-no-acl",
                    "--max-pages",
                    "0",
                ]
            )
        self.assertIn("max_pages", error.getvalue())

    def test_success_path_passes_space_and_default_collection(self) -> None:
        output = io.StringIO()
        result = IndexingResult(
            collection_name="confluence_documents",
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
                "rag_pipeline.infrastructure.confluence.ConfluenceCloudClient.from_environment",
                return_value=nullcontext(object()),
            ),
            patch(
                "rag_pipeline.application.confluence.index_confluence_space",
                return_value=result,
            ) as import_space,
            redirect_stdout(output),
        ):
            self.assertEqual(
                main(["index-confluence", "42", "--acknowledge-no-acl"]), 0
            )
        self.assertEqual(import_space.call_args.kwargs["space_id"], "42")
        self.assertEqual(
            import_space.call_args.kwargs["config"].vector_store.collection_name,
            "confluence_documents",
        )
        self.assertIn("Indexed 2 Confluence chunk(s)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
