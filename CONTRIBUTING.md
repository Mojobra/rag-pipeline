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

## Branch Workflow

`develop` is the development and integration branch. Start every coding,
documentation, configuration, and hotfix task on a short-lived branch from the
latest `develop`:

```powershell
git switch develop
git pull --ff-only origin develop
git switch -c codex/<task-name>
```

Open the task pull request against `develop`. After review and passing checks,
merge the finished feature into `develop`, then propose a `develop` to `main`
pull request. Each merge requires explicit approval. Squash task branches into
`develop` when appropriate; preserve shared history with a merge commit when
promoting `develop` to `main`. Synchronize `develop` with approved `main` merges
without rebasing the shared branch.

`main` accepts reviewed, completed work from `develop` and is the only source
for release snapshots. Never develop, edit files, or create task commits on
`main` or `release/*`.

`release/rag-v<version>` branches are immutable snapshots of an approved `main`
commit. After their initial creation and push, never add commits, merge, rebase,
reset, force-push, or delete them. Corrections go through the development
workflow and produce a new release version and snapshot.

## Releases And Versioning

`project.version` in `pyproject.toml` is the only editable runtime version
source. Package imports, `--version`, and benchmark provenance read the installed
distribution metadata; do not add a second version literal to Python modules.

Prepare version changes on a task branch from `develop`: update
`project.version`, `uv.lock`, `CHANGELOG.md`, and affected manual sections, then
pass the complete quality suite and package inspection. Review the task into
`develop` and promote the approved development state to `main`.

From the validated `main` commit, create and push a new
`release/rag-vMAJOR.MINOR.PATCH` snapshot and, when publishing a tag, an annotated
`vMAJOR.MINOR.PATCH` tag. Build release artifacts from that commit. Never prepare
release changes directly on `main` or edit an existing release branch.

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

## Documentation Maintenance

Review `TECHNICAL_MANUAL.md` with every coding change. Update it in the same
pull request when a change affects dependencies, supported runtimes, models,
providers, persistent storage, data flow, package ownership, CLI or API
surfaces, evaluation behavior, build tooling, tests, CI, or deployment. Also
update its `Last verified` date when documented facts change.

An internal implementation change that leaves every documented fact unchanged
does not require wording churn, but the pull-request self-review must still
confirm that the manual remains accurate. Keep delivered behavior separate from
roadmap plans and use `pyproject.toml`, `uv.lock`, source code, and CI
configuration as the authoritative inputs.

Use a short-lived task branch from `develop` and a focused Conventional
Commit-style message. Task pull requests target `develop` and explain the
behavior, design trade-offs, tests run, compatibility risks, and any deliberate
breaking change.
