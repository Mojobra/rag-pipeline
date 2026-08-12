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

## Releases And Versioning

`project.version` in `pyproject.toml` is the only editable runtime version
source. Package imports, `--version`, and benchmark provenance read the installed
distribution metadata; do not add a second version literal to Python modules.

A release pull request must update `project.version`, `uv.lock`, and
`CHANGELOG.md`, then pass the complete quality suite and package inspection.
After that pull request is merged and `main` is current, create an annotated
`vMAJOR.MINOR.PATCH` tag on the validated `main` commit and push that tag. Never
tag an unmerged feature-branch commit.

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
- Put document discovery, extraction, and chunking in `rag_pipeline.ingestion`;
  search and reranking in `rag_pipeline.retrieval`; prompts, citations, and model
  invocation in `rag_pipeline.generation`; and metrics in
  `rag_pipeline.evaluation` or `rag_pipeline.benchmarking` as appropriate.
- Keep provider and persistence adapters in `rag_pipeline.infrastructure`.
- Use canonical feature-package imports throughout source, tests, and
  documentation. Do not add root-level re-export modules that hide ownership.
- Keep the package hierarchy shallow. Add a subpackage when several modules have
  one cohesive owner, not merely to wrap a single class or function.
- Validate dynamic JSON and provider responses at their boundaries.
- Treat feature-package imports as public API. Document any deliberate breaking
  change explicitly, especially while the project remains pre-1.0.
- Add concise docstrings for public APIs, I/O, state mutation, and non-obvious
  algorithms.
- Add focused regression tests for changed behavior and failure modes.

Use a short-lived feature branch and a focused Conventional Commit-style
message. Pull requests should explain the behavior, design trade-offs, tests
run, compatibility risks, and any deliberate breaking change.
