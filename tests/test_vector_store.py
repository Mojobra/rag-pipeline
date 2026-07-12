from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from rag_pipeline.embeddings import EmbeddedDocument  # noqa: E402
from rag_pipeline.exceptions import (  # noqa: E402
    InvalidVectorStoreConfigurationError,
    VectorStoreCompatibilityError,
    VectorStoreInputError,
)
from rag_pipeline.vector_store import (  # noqa: E402
    LocalVectorStore,
    VectorStoreConfig,
    build_chunk_point_id,
)


def make_embedded_document(
    content: str,
    *,
    source: str = "policy.txt",
    chunk_index: int = 0,
    vector: tuple[float, ...] = (1.0, 0.0),
    extra_metadata: dict[str, object] | None = None,
) -> EmbeddedDocument:
    metadata: dict[str, object] = {
        "source": source,
        "chunk_index": chunk_index,
        "start_index": chunk_index * 10,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return EmbeddedDocument(
        document=Document(page_content=content, metadata=metadata),
        embedding=vector,
    )


class LocalVectorStoreTests(unittest.TestCase):
    def test_indexes_vectors_and_exposes_langchain_documents(self) -> None:
        records = [
            make_embedded_document("Travel expenses need receipts."),
            make_embedded_document(
                "Managers approve exceptions.",
                chunk_index=1,
                vector=(0.0, 1.0),
            ),
        ]
        original_metadata = records[0].document.metadata.copy()

        with LocalVectorStore(
            VectorStoreConfig(path=None, collection_name="policies")
        ) as store:
            result = store.index(records, model_identifier="test-model@revision")
            stored_documents = store.as_langchain_vector_store().get_by_ids(
                list(result.point_ids)
            )

        self.assertEqual(result.indexed_count, 2)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(result.embedding_dimension, 2)
        self.assertEqual(result.embedding_model, "test-model@revision")
        self.assertEqual(len(stored_documents), 2)
        self.assertTrue(all(UUID(point_id).version == 5 for point_id in result.point_ids))

        stored_by_id = {
            document.metadata["_id"]: document for document in stored_documents
        }
        first = stored_by_id[result.point_ids[0]]
        self.assertEqual(first.page_content, "Travel expenses need receipts.")
        self.assertEqual(first.metadata["source"], "policy.txt")
        self.assertEqual(first.metadata["chunk_id"], result.point_ids[0])
        self.assertEqual(first.metadata["embedding_model"], "test-model@revision")
        self.assertEqual(first.metadata["embedding_dimension"], 2)
        self.assertEqual(records[0].document.metadata, original_metadata)

    def test_repeated_and_changed_chunks_upsert_the_same_logical_id(self) -> None:
        original = make_embedded_document("Old policy wording.")
        changed = make_embedded_document(
            "New policy wording.",
            vector=(0.5, 0.5),
        )

        self.assertEqual(
            build_chunk_point_id(original.document),
            build_chunk_point_id(changed.document),
        )

        with LocalVectorStore(
            VectorStoreConfig(path=None, collection_name="idempotent")
        ) as store:
            first_result = store.index([original], model_identifier="test-model")
            second_result = store.index([original], model_identifier="test-model")
            changed_result = store.index([changed], model_identifier="test-model")
            stored = store.as_langchain_vector_store().get_by_ids(
                list(changed_result.point_ids)
            )

        self.assertEqual(first_result.point_ids, second_result.point_ids)
        self.assertEqual(second_result.point_ids, changed_result.point_ids)
        self.assertEqual(changed_result.total_count, 1)
        self.assertEqual(stored[0].page_content, "New policy wording.")

    def test_persists_collection_across_reopen(self) -> None:
        record = make_embedded_document("Persistent policy chunk.")

        with tempfile.TemporaryDirectory() as temp_dir:
            config = VectorStoreConfig(
                path=Path(temp_dir) / "qdrant",
                collection_name="persistent",
            )
            with LocalVectorStore(config) as first_store:
                result = first_store.index([record], model_identifier="test-model")

            with LocalVectorStore(config) as reopened_store:
                count = reopened_store.count()
                stored = reopened_store.as_langchain_vector_store().get_by_ids(
                    list(result.point_ids)
                )

        self.assertEqual(count, 1)
        self.assertEqual(stored[0].page_content, "Persistent policy chunk.")

    def test_rejects_incompatible_model_and_dimension(self) -> None:
        with LocalVectorStore(
            VectorStoreConfig(path=None, collection_name="compatibility")
        ) as store:
            store.index(
                [make_embedded_document("Original")],
                model_identifier="model-a",
            )

            with self.assertRaisesRegex(
                VectorStoreCompatibilityError, "embedding_model"
            ):
                store.index(
                    [
                        make_embedded_document(
                            "Different model",
                            source="other.txt",
                        )
                    ],
                    model_identifier="model-b",
                )

            with self.assertRaisesRegex(
                VectorStoreCompatibilityError, "dimension is 2.*use 3"
            ):
                store.index(
                    [
                        make_embedded_document(
                            "Different dimension",
                            source="dimension.txt",
                            vector=(1.0, 0.0, 0.0),
                        )
                    ],
                    model_identifier="model-a",
                )

    def test_rejects_duplicate_ids_and_non_json_metadata(self) -> None:
        duplicate_records = [
            make_embedded_document("First"),
            make_embedded_document("Second", vector=(0.0, 1.0)),
        ]
        invalid_metadata = make_embedded_document(
            "Invalid metadata",
            source="invalid.txt",
            extra_metadata={"path_object": Path("not-json")},
        )

        with LocalVectorStore(
            VectorStoreConfig(path=None, collection_name="invalid-input")
        ) as store:
            with self.assertRaisesRegex(VectorStoreInputError, "Duplicate"):
                store.index(duplicate_records, model_identifier="test-model")

            with self.assertRaisesRegex(
                VectorStoreInputError, "metadata must be JSON-serializable"
            ):
                store.index([invalid_metadata], model_identifier="test-model")

    def test_rejects_invalid_configuration(self) -> None:
        invalid_settings = [
            ({"path": " "}, "path must be non-empty"),
            ({"collection_name": ""}, "collection_name must be a non-empty"),
            ({"write_batch_size": 0}, "write_batch_size must be greater"),
            ({"write_batch_size": True}, "write_batch_size must be an integer"),
        ]

        for settings, message in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(
                    InvalidVectorStoreConfigurationError, message
                ):
                    VectorStoreConfig(**settings)


if __name__ == "__main__":
    unittest.main()
