"""Capture portable corpus, dataset, source, and runtime benchmark provenance.

The module hashes benchmark inputs without serializing absolute paths and
records reproducibility metadata that does not include usernames, hostnames,
cache locations, credentials, or document contents.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from importlib import metadata as importlib_metadata
from pathlib import Path

from rag_pipeline import __version__
from rag_pipeline.exceptions import BenchmarkInputError
from rag_pipeline.ingestion import discover_files

_PACKAGE_DISTRIBUTIONS = (
    "langchain",
    "langchain-core",
    "langchain-huggingface",
    "langchain-qdrant",
    "langchain-text-splitters",
    "qdrant-client",
    "sentence-transformers",
    "sentencepiece",
    "transformers",
    "fastembed",
    "torch",
    "pypdf",
    "docx2txt",
)


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    """Content identity and portable location for one benchmark input file."""

    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CorpusFingerprint:
    """Deterministic identity for every supported file in one corpus root.

    Relative paths are included in the aggregate digest so renames and content
    changes both produce a new identity without exposing local filesystem paths.
    """

    sha256: str
    file_count: int
    total_bytes: int
    files: tuple[FileFingerprint, ...]


@dataclass(frozen=True, slots=True)
class DatasetFingerprint:
    """Identity and declared schema metadata for one evaluation dataset."""

    file_name: str
    sha256: str
    name: str
    schema_version: int
    case_count: int


def fingerprint_corpus(path: str | Path) -> CorpusFingerprint:
    """Hash supported corpus files without exposing their absolute paths.

    The function recursively discovers and reads every supported file. An empty
    corpus is rejected because it cannot produce a meaningful benchmark index.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("corpus path must be a string or pathlib.Path.")
    root = Path(path).expanduser().resolve()
    files = discover_files([root])
    if not files:
        raise BenchmarkInputError(
            f"benchmark corpus contains no supported documents: {root}"
        )

    fingerprints = tuple(
        _fingerprint_corpus_file(file_path, root) for file_path in files
    )
    canonical_files = [
        {
            "relative_path": item.relative_path,
            "size_bytes": item.size_bytes,
            "sha256": item.sha256,
        }
        for item in fingerprints
    ]
    aggregate = sha256(
        json.dumps(
            canonical_files,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return CorpusFingerprint(
        sha256=aggregate,
        file_count=len(fingerprints),
        total_bytes=sum(item.size_bytes for item in fingerprints),
        files=fingerprints,
    )


def fingerprint_dataset(
    path: str | Path,
    *,
    name: str,
    schema_version: int,
    case_count: int,
) -> DatasetFingerprint:
    """Hash one loaded dataset and retain only its portable file name.

    Dataset schema validation remains the responsibility of the retrieval or
    answer evaluator. This function performs an additional byte read so the
    artifact identifies the exact serialized label snapshot.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("dataset path must be a string or pathlib.Path.")
    resolved_path = Path(path).expanduser().resolve()
    try:
        content = resolved_path.read_bytes()
    except OSError as exc:
        raise BenchmarkInputError(
            f"failed to fingerprint evaluation dataset {resolved_path}: {exc}"
        ) from exc
    return DatasetFingerprint(
        file_name=resolved_path.name,
        sha256=sha256(content).hexdigest(),
        name=name,
        schema_version=schema_version,
        case_count=case_count,
    )


def runtime_environment() -> dict[str, object]:
    """Return non-secret software and platform fields relevant to comparisons."""
    package_versions: dict[str, str | None] = {"rag-pipeline": __version__}
    for distribution in _PACKAGE_DISTRIBUTIONS:
        try:
            package_versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            package_versions[distribution] = None
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
        },
        "accelerator": _accelerator_environment(),
        "packages": package_versions,
    }


def read_source_revision() -> dict[str, object]:
    """Read the current Git commit and tracked-dirty state when available.

    The subprocess calls are bounded and ignore untracked filenames so local
    private assets cannot leak into artifacts. Installed packages outside a Git
    checkout return null fields instead of failing a benchmark.
    """
    repository_root = Path(__file__).resolve().parents[2]
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tracked_worktree_dirty": None}
    commit = commit_result.stdout.strip()
    if len(commit) != 40:
        return {"commit": None, "tracked_worktree_dirty": None}
    return {
        "commit": commit,
        "tracked_worktree_dirty": bool(status_result.stdout.strip()),
    }


def corpus_fingerprint_to_dict(
    fingerprint: CorpusFingerprint,
) -> dict[str, object]:
    """Serialize a corpus fingerprint to JSON-compatible primitives."""
    return {
        "sha256": fingerprint.sha256,
        "file_count": fingerprint.file_count,
        "total_bytes": fingerprint.total_bytes,
        "files": [
            {
                "relative_path": item.relative_path,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in fingerprint.files
        ],
    }


def dataset_fingerprint_to_dict(
    fingerprint: DatasetFingerprint,
) -> dict[str, object]:
    """Serialize a dataset fingerprint to JSON-compatible primitives."""
    return {
        "file_name": fingerprint.file_name,
        "sha256": fingerprint.sha256,
        "name": fingerprint.name,
        "schema_version": fingerprint.schema_version,
        "case_count": fingerprint.case_count,
    }


def _fingerprint_corpus_file(
    file_path: Path,
    root: Path,
) -> FileFingerprint:
    relative_path = _relative_corpus_path(file_path, root)
    try:
        content = file_path.read_bytes()
    except OSError as exc:
        raise BenchmarkInputError(
            f"failed to fingerprint corpus file {file_path}: {exc}"
        ) from exc
    return FileFingerprint(
        relative_path=relative_path,
        size_bytes=len(content),
        sha256=sha256(content).hexdigest(),
    )


def _relative_corpus_path(file_path: Path, root: Path) -> str:
    if root.is_file():
        return root.name
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise BenchmarkInputError(
            f"corpus file resolves outside the corpus root: {file_path}"
        ) from exc


def _accelerator_environment() -> dict[str, object]:
    """Inspect CUDA identity when PyTorch is available without requiring it."""
    try:
        torch = import_module("torch")
        cuda = torch.cuda
        available = bool(cuda.is_available())
        device_names = (
            [cuda.get_device_name(index) for index in range(cuda.device_count())]
            if available
            else []
        )
        return {
            "cuda_available": available,
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_names": device_names,
        }
    # Accelerator telemetry is diagnostic; broken optional CUDA/PyTorch state
    # must not prevent an otherwise valid benchmark from running.
    except Exception:  # noqa: BLE001
        return {
            "cuda_available": None,
            "cuda_version": None,
            "device_names": [],
        }
