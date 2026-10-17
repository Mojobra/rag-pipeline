"""Test Qdrant persistence, schema compatibility, and deterministic upserts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from langchain_core.documents import Document

from rag_pipeline.exceptions import (
    InvalidVectorStoreConfigurationError,
    VectorStoreCompatibilityError,
    VectorStoreInputError,
    VectorStoreProviderError,
)
from rag_pipeline.infrastructure.embeddings import EmbeddedDocument
from rag_pipeline.infrastructure.sparse_embeddings import SparseEmbeddingVector
from rag_pipeline.infrastructure.vector_store import (
    LocalVectorStore,
    SearchMode,
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
        self.assertTrue(
            all(UUID(point_id).version == 5 for point_id in result.point_ids)
        )

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

    def test_source_replacement_removes_stale_chunks_and_keeps_other_sources(
        self,
    ) -> None:
        initial = [
            make_embedded_document("Old first chunk"),
            make_embedded_document("Old second chunk", chunk_index=1),
            make_embedded_document("Other source", source="other.txt"),
        ]
        updated = make_embedded_document("New first chunk")

        with LocalVectorStore(VectorStoreConfig(path=None)) as store:
            store.index(initial, model_identifier="test-model")
            result = store.replace_sources(
                [updated], sources={"policy.txt"}, model_identifier="test-model"
            )
            repeated = store.replace_sources(
                [updated], sources={"policy.txt"}, model_identifier="test-model"
            )
            stored = store.as_langchain_vector_store().get_by_ids(
                [build_chunk_point_id(record.document) for record in initial]
            )

        self.assertEqual(result.removed_count, 1)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(repeated.removed_count, 0)
        self.assertEqual(repeated.total_count, 2)
        self.assertEqual(
            {document.page_content for document in stored},
            {"New first chunk", "Other source"},
        )

    def test_empty_replacement_removes_only_named_source(self) -> None:
        with LocalVectorStore(VectorStoreConfig(path=None)) as store:
            store.index(
                [
                    make_embedded_document("Old source"),
                    make_embedded_document("Keep", source="other.txt"),
                ],
                model_identifier="test-model",
            )
            result = store.replace_sources(
                [], sources={"policy.txt"}, model_identifier="test-model"
            )
            remaining = store.list_sources()

        self.assertEqual(result.removed_count, 1)
        self.assertEqual(result.total_count, 1)
        self.assertEqual(remaining, {"other.txt"})

    def test_replacement_handles_multiple_pdf_pages(self) -> None:
        first = make_embedded_document(
            "First page", source="policy.pdf", extra_metadata={"page": 0}
        )
        second = make_embedded_document(
            "Second page", source="policy.pdf", extra_metadata={"page": 1}
        )
        with LocalVectorStore(VectorStoreConfig(path=None)) as store:
            store.index([first, second], model_identifier="test-model")
            result = store.replace_sources(
                [first], sources={"policy.pdf"}, model_identifier="test-model"
            )
            stored = store.as_langchain_vector_store().get_by_ids(
                [
                    build_chunk_point_id(first.document),
                    build_chunk_point_id(second.document),
                ]
            )

        self.assertEqual(result.removed_count, 1)
        self.assertEqual([document.metadata["page"] for document in stored], [0])

    def test_replacement_rejects_invalid_scope_and_incompatible_empty_model(
        self,
    ) -> None:
        with LocalVectorStore(VectorStoreConfig(path=None)) as store:
            store.index(
                [make_embedded_document("Original")], model_identifier="test-model"
            )
            with self.assertRaisesRegex(VectorStoreInputError, "outside"):
                store.replace_sources(
                    [make_embedded_document("Other", source="other.txt")],
                    sources={"policy.txt"},
                    model_identifier="test-model",
                )
            with self.assertRaisesRegex(
                VectorStoreCompatibilityError, "embedding_model"
            ):
                store.replace_sources(
                    [], sources={"policy.txt"}, model_identifier="wrong-model"
                )
            self.assertEqual(store.count(), 1)

    def test_failed_upsert_does_not_delete_old_source_points(self) -> None:
        with LocalVectorStore(VectorStoreConfig(path=None)) as store:
            original = [
                make_embedded_document("First"),
                make_embedded_document("Second", chunk_index=1),
            ]
            store.index(original, model_identifier="test-model")
            with patch.object(
                store._client, "upsert", side_effect=RuntimeError("write")
            ):
                with self.assertRaises(VectorStoreProviderError):
                    store.replace_sources(
                        [original[0]],
                        sources={"policy.txt"},
                        model_identifier="test-model",
                    )
            self.assertEqual(store.count(), 2)

    def test_failed_snapshot_does_not_write_and_failed_delete_is_retryable(
        self,
    ) -> None:
        with LocalVectorStore(
            VectorStoreConfig(path=None, write_batch_size=1)
        ) as store:
            original = [
                make_embedded_document("First"),
                make_embedded_document("Second", chunk_index=1),
            ]
            store.index(original, model_identifier="test-model")
            with patch.object(
                store._client, "scroll", side_effect=RuntimeError("read")
            ):
                with self.assertRaises(VectorStoreProviderError):
                    store.replace_sources(
                        [original[0]],
                        sources={"policy.txt"},
                        model_identifier="test-model",
                    )
            self.assertEqual(store.count(), 2)

            with patch.object(
                store._client, "delete", side_effect=RuntimeError("delete")
            ):
                with self.assertRaises(VectorStoreProviderError):
                    store.replace_sources(
                        [original[0]],
                        sources={"policy.txt"},
                        model_identifier="test-model",
                    )
            self.assertEqual(store.count(), 2)
            retry = store.replace_sources(
                [original[0]], sources={"policy.txt"}, model_identifier="test-model"
            )
            self.assertEqual(retry.removed_count, 1)
            self.assertEqual(retry.total_count, 1)

    def test_hybrid_replacement_prunes_stale_sparse_points(self) -> None:
        with LocalVectorStore(
            VectorStoreConfig(path=None, search_mode=SearchMode.HYBRID)
        ) as store:
            original = [
                make_embedded_document("First"),
                make_embedded_document("Second", chunk_index=1),
            ]
            store.index(
                original,
                model_identifier="dense-model",
                sparse_vectors=[
                    SparseEmbeddingVector(indices=(1,), values=(1.0,)),
                    SparseEmbeddingVector(indices=(2,), values=(1.0,)),
                ],
                sparse_model_identifier="sparse-model",
            )
            result = store.replace_sources(
                [original[0]],
                sources={"policy.txt"},
                model_identifier="dense-model",
                sparse_vectors=[SparseEmbeddingVector(indices=(1,), values=(1.0,))],
                sparse_model_identifier="sparse-model",
            )
            self.assertEqual(result.removed_count, 1)
            self.assertEqual(result.total_count, 1)

    def test_indexes_hybrid_vectors_with_versioned_compatible_schema(self) -> None:
        records = [
            make_embedded_document("Policy ZX-42"),
            make_embedded_document(
                "Punctuation only",
                chunk_index=1,
                vector=(0.0, 1.0),
            ),
        ]
        sparse_vectors = [
            SparseEmbeddingVector(indices=(42,), values=(1.5,)),
            SparseEmbeddingVector(indices=(), values=()),
        ]

        with LocalVectorStore(
            VectorStoreConfig(
                path=None,
                collection_name="hybrid-index",
                search_mode=SearchMode.HYBRID,
            )
        ) as store:
            result = store.index(
                records,
                model_identifier="dense-model",
                sparse_vectors=sparse_vectors,
                sparse_model_identifier="sparse-model",
            )
            store.validate_compatibility(
                model_identifier="dense-model",
                dimension=2,
                sparse_model_identifier="sparse-model",
            )
            stored_documents = store.as_langchain_vector_store().get_by_ids(
                list(result.point_ids)
            )

        self.assertEqual(result.search_mode, SearchMode.HYBRID)
        self.assertEqual(result.sparse_embedding_model, "sparse-model")
        self.assertEqual(result.total_count, 2)
        self.assertEqual(len(stored_documents), 2)
        self.assertEqual(
            stored_documents[0].metadata["sparse_embedding_model"],
            "sparse-model",
        )

    def test_rejects_missing_or_incompatible_hybrid_inputs(self) -> None:
        record = make_embedded_document("Policy ZX-42")
        vector = SparseEmbeddingVector(indices=(42,), values=(1.0,))

        with LocalVectorStore(
            VectorStoreConfig(
                path=None,
                collection_name="hybrid-inputs",
                search_mode="hybrid",
            )
        ) as store:
            with self.assertRaisesRegex(
                VectorStoreInputError,
                "requires one sparse vector",
            ):
                store.index(
                    [record],
                    model_identifier="dense-model",
                    sparse_model_identifier="sparse-model",
                )

            with self.assertRaisesRegex(
                VectorStoreInputError,
                "0 sparse vector.*1 document",
            ):
                store.index(
                    [record],
                    model_identifier="dense-model",
                    sparse_vectors=[],
                    sparse_model_identifier="sparse-model",
                )

            malformed_vectors = [
                (
                    SparseEmbeddingVector(indices=(42,), values=()),
                    "different index and value counts",
                ),
                (
                    SparseEmbeddingVector(
                        indices=(42, 42),
                        values=(1.0, 2.0),
                    ),
                    "duplicate sparse index",
                ),
                (
                    SparseEmbeddingVector(
                        indices=(42,),
                        values=(float("nan"),),
                    ),
                    "non-finite value",
                ),
            ]
            for malformed_vector, message in malformed_vectors:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(VectorStoreInputError, message):
                        store.index(
                            [record],
                            model_identifier="dense-model",
                            sparse_vectors=[malformed_vector],
                            sparse_model_identifier="sparse-model",
                        )

            store.index(
                [record],
                model_identifier="dense-model",
                sparse_vectors=[vector],
                sparse_model_identifier="sparse-model",
            )
            with self.assertRaisesRegex(
                VectorStoreCompatibilityError,
                "sparse_embedding_model",
            ):
                store.validate_compatibility(
                    model_identifier="dense-model",
                    dimension=2,
                    sparse_model_identifier="other-sparse-model",
                )

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

            hybrid_config = VectorStoreConfig(
                path=config.path,
                collection_name=config.collection_name,
                search_mode=SearchMode.HYBRID,
            )
            with LocalVectorStore(hybrid_config) as hybrid_store:
                with self.assertRaisesRegex(
                    VectorStoreCompatibilityError,
                    "not a hybrid schema v2 collection",
                ):
                    hybrid_store.validate_compatibility(
                        model_identifier="test-model",
                        dimension=2,
                        sparse_model_identifier="test-sparse-model",
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
            ({"search_mode": "unsupported"}, "search_mode must be one of"),
        ]

        for settings, message in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(
                    InvalidVectorStoreConfigurationError, message
                ):
                    VectorStoreConfig(**settings)


if __name__ == "__main__":
    unittest.main()
