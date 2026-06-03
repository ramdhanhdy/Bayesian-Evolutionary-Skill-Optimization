# Development Workflow

## Local Setup

Install the project with development dependencies, then run tests from the
repository root.

```powershell
uv sync --extra dev --extra llm
uv run pytest
```

If `uv` is unavailable, use an equivalent Python environment with the dependencies
declared in `pyproject.toml`.

## Working Rules

- Keep `main` releasable.
- Keep edits scoped to the requested behavior.
- Preserve unrelated user changes in the worktree.
- Do not silently change default experiment parameters.
- Do not commit `.env`, raw traces, secrets, private datasets, or bulky generated
  artifacts.
- Record user-visible behavior changes and research-protocol changes in
  `CHANGELOG.md`.

## Documentation Rules

- Put detailed methodology and mathematical material in `docs/design/`.
- Put accepted decisions in `docs/decisions/`.
- Put exploratory ideas and unresolved questions in `docs/notes/`.
- Promote a note to an ADR only after the decision is reviewed and accepted.
- Use version metadata in ADRs instead of copying documents into every release
  directory.

## Experiment Hygiene

Every report-quality run should record:

```yaml
run_id: YYYY-MM-DD_<dataset-or-task>_seed<seed>_beso-v<version>_<short-commit>
package_version: 0.1.0
git_commit: <sha>
git_dirty: false
config: configs/...
seed: 42
dataset: ...
model: provider/model-name
```

Exploratory runs may be useful, but they should not be treated as report-quality
evidence unless the configuration, commit, seed, and evaluation protocol are
recoverable.

