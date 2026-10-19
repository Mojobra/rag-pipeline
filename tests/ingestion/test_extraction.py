"""Test format-specific text extraction into provenance-bearing documents."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from langchain_core.documents import Document

from rag_pipeline.ingestion.chunking import ChunkingConfig, chunk_documents
from rag_pipeline.ingestion.extraction import extract_documents

FIXTURES = Path(__file__).parent / "fixtures"


def write_minimal_pdf(
    path: Path,
    text: str,
    *,
    text_operations: tuple[tuple[int, int, str], ...] | None = None,
) -> None:
    """Write a tiny text-layer PDF with an optional deliberate draw order."""
    operations = text_operations or ((72, 720, text),)
    stream_parts = []
    for x, y, value in operations:
        safe_text = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_parts.append(f"BT /F1 24 Tf {x} {y} Td ({safe_text}) Tj ET\n")
    stream = "".join(stream_parts).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"endstream",
    ]

    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for index, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n".encode("ascii"))
        content.extend(obj)
        content.extend(b"\nendobj\n")

    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    content.extend(
        (
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(content))


def write_minimal_docx(path: Path, text: str) -> None:
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:r>
        <w:t>{escape(text)}</w:t>
      </w:r>
    </w:p>
  </w:body>
</w:document>
"""
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("word/document.xml", document_xml)


class TextExtractionTests(unittest.TestCase):
    def test_extracts_text_file_to_single_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "notes.txt"
            file_path.write_text("Grounded answers need sources.", encoding="utf-8")

            documents = extract_documents(file_path)

        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0], Document)
        self.assertEqual(documents[0].page_content, "Grounded answers need sources.")
        self.assertEqual(documents[0].metadata["extractor"], "text")

    def test_extracts_pdf_with_page_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "policy.pdf"
            write_minimal_pdf(file_path, "Quarterly revenue policy")

            documents = extract_documents(file_path)

        self.assertEqual(len(documents), 1)
        self.assertIn("Quarterly revenue policy", documents[0].page_content)
        self.assertEqual(documents[0].metadata["file_extension"], ".pdf")
        self.assertEqual(documents[0].metadata["extractor"], "pypdf")
        self.assertEqual(documents[0].metadata["page"], 0)

    def test_extracts_docx_to_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "handbook.docx"
            write_minimal_docx(file_path, "Expense approvals require receipts.")

            documents = extract_documents(file_path)

        self.assertEqual(len(documents), 1)
        self.assertIn("Expense approvals require receipts.", documents[0].page_content)
        self.assertEqual(documents[0].metadata["file_extension"], ".docx")
        self.assertEqual(documents[0].metadata["extractor"], "docx2txt")

    def test_html_keeps_policy_facts_and_table_without_page_chrome(self) -> None:
        document = extract_documents(FIXTURES / "noisy-policy.html")[0]
        content = document.page_content

        self.assertEqual(document.metadata["extractor"], "html.parser")
        self.assertEqual(document.metadata["file_extension"], ".html")
        self.assertEqual(
            document.metadata["source"], str((FIXTURES / "noisy-policy.html").resolve())
        )
        self.assertIn("Expense Policy EXP-042\n", content)
        self.assertIn("Receipts are required above \u20ac25.", content)
        self.assertIn("Amount | Approver", content)
        self.assertIn("Up to \u20ac500 | Manager", content)
        self.assertIn("Above \u20ac500 | Director", content)
        self.assertIn("Submit every claim within 30 calendar days.", content)
        self.assertIn("Exceptions require written approval from Finance.", content)
        self.assertIn("Effective date: 1 January 2026.", content)
        for noise in (
            "WARNING_STYLE_TOKEN",
            "WARNING_SCRIPT_TOKEN",
            "WARNING_NAV_TOKEN",
            "Portal page title",
        ):
            self.assertNotIn(noise, content)
        self.assertNotIn("<", content)

        chunks = chunk_documents(
            [document], config=ChunkingConfig(chunk_size=96, chunk_overlap=12)
        )
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            start = chunk.metadata["start_index"]
            end = chunk.metadata["end_index"]
            self.assertEqual(chunk.page_content, content[start:end])
            self.assertEqual(chunk.metadata["source"], document.metadata["source"])

    def test_html_fragment_handles_void_elements_inside_ignored_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fragment.htm"
            path.write_text(
                "<head><meta charset='utf-8'><title>Hidden</title></head>"
                "<p>Visible&nbsp;fact<br/>Next line</p>"
                "<script>hidden <img src='x'> text</script>",
                encoding="utf-8",
            )
            document = extract_documents(path)[0]

        self.assertEqual(document.page_content, "Visible fact\nNext line")

    def test_long_markdown_keeps_exact_source_offsets(self) -> None:
        document = extract_documents(FIXTURES / "long-policy.md")[0]
        self.assertEqual(document.metadata["extractor"], "text")
        chunks = chunk_documents([document], config=ChunkingConfig())

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.page_content) <= 1000 for chunk in chunks))
        for chunk in chunks:
            start = chunk.metadata["start_index"]
            end = chunk.metadata["end_index"]
            self.assertEqual(chunk.page_content, document.page_content[start:end])
            self.assertEqual(chunk.metadata["chunk_count"], len(chunks))

    def test_pdf_draw_order_limitation_preserves_page_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "two-columns.pdf"
            write_minimal_pdf(
                path,
                "unused",
                text_operations=(
                    (300, 720, "Right column first in stream"),
                    (72, 720, "Left column second in stream"),
                ),
            )
            document = extract_documents(path)[0]

        self.assertLess(
            document.page_content.index("Right column"),
            document.page_content.index("Left column"),
        )
        self.assertEqual(document.metadata["page"], 0)
        self.assertEqual(document.metadata["total_pages"], 1)
        self.assertEqual(document.metadata["extractor"], "pypdf")
        chunks = chunk_documents(
            [document], config=ChunkingConfig(chunk_size=32, chunk_overlap=4)
        )
        for chunk in chunks:
            start = chunk.metadata["start_index"]
            end = chunk.metadata["end_index"]
            self.assertEqual(chunk.page_content, document.page_content[start:end])


if __name__ == "__main__":
    unittest.main()
