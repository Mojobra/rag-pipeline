# Repository Agent Instructions

These instructions apply to the entire repository. Read them before making
changes, together with the relevant source, tests, `README.md`, `ROADMAP.md`,
`pyproject.toml`, and CI configuration. Also read `CONTRIBUTING.md`,
`ARCHITECTURE.md`, and `TECHNICAL_MANUAL.md` when present on the current branch.
An ignored local `CODEX_RULES.md` may provide additional working preferences.
If instructions conflict, explain the conflict and follow the user's explicit
direction without silently bypassing safety checks.

## Scope And Collaboration

- Act as a senior AI engineer and tutor. Before implementation, explain what
  will be built, why it matters, the design, important trade-offs, affected
  components, and planned validation. Keep explanations practical and concise.
- Implement only the requested task. Do not start the next roadmap task without
  approval. For opinion, explanation, or planning requests, do not change code.
- Inspect the existing implementation before proposing abstractions or edits.
  Preserve behavior and public interfaces unless an explicitly requested change
  or a verified defect requires otherwise. Document deliberate breaking changes.
- Provide short progress updates and finish with actual changes, validation
  results, remaining limitations, and review status. Never claim an unperformed
  test, push, merge, deployment, or benchmark result.

## Branch And Release Workflow

- Check `git status --short --branch`, local branches, worktrees, and remote
  state before making changes. Fetch `origin` when available; report access
  limitations rather than assuming a local reference is current.
- `develop` is the permanent development and integration branch. Normally use
  a focused, short-lived `codex/<description>` task branch from the latest
  `develop`. Do not start new work from historical feature branches.
- Development takes place on or from `develop`, never directly on `main` or a
  release branch. Documentation, configuration, hotfixes, and release preparation
  follow the same task-branch workflow as code.
- Validate and self-review the complete diff, stage only intended files, create
  focused Conventional Commit-style commits, and push the task branch when
  authorized. Open a pull request into `develop`; if tooling is unavailable,
  prepare an exact PR title, description, and comparison link.
- Merge only with explicit user authorization and satisfied checks and review
  requirements. Prefer squash merges for task PRs into `develop`.
- Promote reviewed work through a separate `develop` -> `main` pull request.
  Use a merge commit for this permanent-branch promotion, not a squash merge,
  so shared ancestry is preserved. Synchronize approved promotion commits back
  into `develop` through a reviewed merge without rewriting shared history.
- `main` supplies validated commits for releases; it is not a development
  branch. Never commit or push directly to it or bypass branch protection.
- `release/rag-v<version>` is an immutable snapshot of a validated `main` commit.
  Initial creation and push require a user request. After creation, never edit,
  add commits, merge, rebase, reset, force-push, delete, or otherwise move it.
  Corrections and new instructions belong in development and a future release,
  not in existing release snapshots.
- A requested documentation backport must contain only the intended document
  changes. Do not merge unrelated features into an older branch or recreate
  deleted remote branches merely to distribute documentation.

## Data And Git Safety

- Preserve unrelated modifications and untracked files. Do not discard, stash,
  reset, clean, overwrite, or broadly reformat user work without permission.
- Do not rewrite shared history, use `git reset --hard`, use plain
  `git push --force`, bypass hooks, or change branch protection to finish a task.
  Rebasing an unshared task branch may be appropriate; any authorized push after
  a rebase must use `--force-with-lease`.
- Never stage or publish secrets, `.env` files, credentials, private documents,
  model caches, local vector data, or private working notes. A sanitized
  `.env.example` may document configuration without real credentials.
- `CODEX_RULES.md` and `ARCHITECTURE-DECISIONS.md` are private local files. Keep
  them ignored and untracked; do not include their contents in commits or PRs.
- Inspect historical branches before publishing them. A documentation update
  is not permission to publish unrelated private history or local-only work.

## Python And RAG Design

- Use LangChain for the project's RAG building blocks. Prefer existing
  interfaces and focused, explicit Python over opaque abstractions or a rewrite.
  Explain any justified exception and avoid unnecessary dependencies.
- Keep CLI parsing and terminal rendering separate from transport-neutral
  application workflows, domain logic, model adapters, and persistence.
  Follow the current feature-package ownership and canonical imports; do not
  reintroduce root-level compatibility facades or needless package nesting.
- Use descriptive names and type hints, validate dynamic input at boundaries,
  and return actionable errors without leaking sensitive information.
- Preserve provenance, stable identifiers, citation integrity, token-budget
  checks, collection compatibility, evaluation semantics, and ingestion safety.
  Do not weaken these contracts to simplify an implementation or pass tests.
- Respect the declared Python support in `pyproject.toml`. Keep dependency and
  lockfile changes intentional and explain their operational trade-offs.
- Distinguish implemented behavior from future architecture. Do not describe
  hosted providers, RBAC, authentication, deployment, or other planned features
  as delivered without checking the actual code.

## Documentation And CLI Help

- Add short module docstrings for important modules, class docstrings describing
  responsibility and context, and concise docstrings for non-trivial APIs,
  service methods, I/O, state mutations, and algorithms.
- Document relevant inputs, outputs, side effects, validation boundaries,
  constraints, and edge cases. Usually keep docstrings to 3-8 lines; omit
  redundant documentation for obvious small private helpers.
- Comments should explain decisions and constraints, not narrate statements.
  Verify documentation against implementation and remove misleading claims.
- Every new CLI argument must include concise help describing its purpose,
  practical effect, important trade-offs, units, accepted values, and
  relationships to other arguments where relevant. Inspect actual behavior
  before changing help; do not invent defaults or performance guarantees.
- Review `TECHNICAL_MANUAL.md` after each coding task and update it on the same
  task branch when documented technology, architecture, interfaces, runtime,
  storage, evaluation, tooling, CI, or deployment facts change. Update its
  `Last verified` date when its content changes; avoid unnecessary wording churn.
- Keep relevant README, architecture, contribution, roadmap, and changelog
  documentation accurate. Mark roadmap work complete only when implemented.
- After a completed roadmap task, update the local, ignored
  `ARCHITECTURE-DECISIONS.md` with business relevance, design rationale,
  trade-offs, validation, and remaining production work. If creating it is
  appropriate, ensure it remains ignored before writing private notes.
- `project.version` in `pyproject.toml` is the single editable version source.
  Prepare intentional releases with matching lockfile and changelog updates;
  do not invent automatic version increments or duplicate version literals.

## Validation And Review

- Add focused tests for behavior changes, edge cases, and failure modes. Keep
  default tests deterministic and offline: use doubles instead of downloading
  models, calling hosted APIs, or relying on an existing local collection.
- Use the current branch's declared tools and CI commands. The current quality
  workflow uses `uv sync --locked --dev` followed by:

  ```powershell
  uv run ruff format --check .
  uv run ruff check .
  uv run mypy
  uv run coverage run -m unittest discover -s tests
  uv run coverage report
  uv build
  ```

- Scale validation to risk. Run focused tests during development and broader
  checks for shared contracts and user-facing workflows. For CLI changes, also
  inspect affected command help and representative success and error paths.
- Never delete tests, weaken assertions, relax validation, lower quality gates,
  or add blanket suppressions merely to obtain passing checks.
- Review the complete diff for correctness, security, compatibility, accidental
  files, dependencies, and unnecessary complexity. Report checks that could not
  run, the reason, and remaining risk; distinguish local results from remote CI.
- Leave the work reviewable. Merging and release creation are separately
  authorized actions, not automatic consequences of completing implementation.
