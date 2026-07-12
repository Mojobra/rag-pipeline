from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from rag_pipeline.ingestion import (  # noqa: E402
    IngestionPathNotFoundError,
    UnsupportedDocumentTypeError,
    discover_files,
    load_documents,
)


class DocumentIngestionTests(unittest.TestCase):
    def test_discovers_supported_files_recursively_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "b.md").write_text("# B", encoding="utf-8")
            (root / "a.txt").write_text("A", encoding="utf-8")
            (root / "report.pdf").write_bytes(b"%PDF-1.4")
            (root / "ignored.png").write_bytes(b"not supported yet")
            nested = root / "nested"
            nested.mkdir()
            (nested / "c.html").write_text("<p>C</p>", encoding="utf-8")

            files = discover_files([root])

        self.assertEqual(
            [path.name for path in files],
            ["a.txt", "b.md", "c.html", "report.pdf"],
        )

    def test_can_scan_directory_without_recursing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("A", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.txt").write_text("B", encoding="utf-8")

            files = discover_files([root], recursive=False)

        self.assertEqual([path.name for path in files], ["a.txt"])

    def test_loads_text_files_as_langchain_documents_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "policy.md"
            file_path.write_text("# Policy\nKeep answers grounded.", encoding="utf-8")

            documents = load_documents([file_path])

        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0], Document)
        self.assertEqual(documents[0].page_content, "# Policy\nKeep answers grounded.")
        self.assertEqual(documents[0].metadata["file_name"], "policy.md")
        self.assertEqual(documents[0].metadata["file_extension"], ".md")
        self.assertGreater(documents[0].metadata["byte_size"], 0)

    def test_rejects_explicit_unsupported_file_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "diagram.png"
            file_path.write_bytes(b"not supported yet")

            with self.assertRaises(UnsupportedDocumentTypeError):
                discover_files([file_path])

    def test_rejects_missing_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.md"

            with self.assertRaises(IngestionPathNotFoundError):
                discover_files([missing_path])


if __name__ == "__main__":
    unittest.main()
