# Contributing

Thank you for improving the RAG pipeline. Changes should preserve deterministic
local behavior, provenance, collection compatibility, citation integrity, and
evaluation semantics unless a pull request explicitly documents why a contract
must change.

## Development Setup

Requirements:

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

Install the exact runtime and development environment:

```powershell
uv sync --locked --dev
```

The project uses the `src` package layout. Run commands through `uv` so the
editable package and locked dependencies are available:

```powershell
uv run rag-pipeline --help
uv run python -m rag_pipeline --help
```

Both entry points execute the same CLI adapter.

## Quality Checks

Run the same checks used by CI before opening a pull request:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run coverage run -m unittest discover -s tests
uv run coverage report
uv build
```

Formatting changes can be applied with:

```powershell
uv run ruff format .
uv run ruff check . --fix
```

The default tests must remain deterministic and offline. Use test doubles for
model providers; do not add tests that download models, call hosted APIs, or
depend on an existing local Qdrant collection.

## Change Design

- Keep CLI parsing and terminal rendering in `rag_pipeline.cli`.
- Put transport-neutral workflow orchestration in `rag_pipeline.application`.
- Keep extraction, chunking, embedding, retrieval, reranking, generation, and
  evaluation behavior in their focused domain modules.
- Validate dynamic JSON and provider responses at their boundaries.
- Preserve stable public imports or provide a compatibility facade.
- Add concise docstrings for public APIs, I/O, state mutation, and non-obvious
  algorithms.
- Add focused regression tests for changed behavior and failure modes.

Use a short-lived feature branch and a focused Conventional Commit-style
message. Pull requests should explain the behavior, design trade-offs, tests
run, compatibility risks, and any deliberate breaking change.
