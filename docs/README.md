# BESO Documentation

This directory separates long-lived design material, experiment records,
architecture notes, and accepted decisions.

## Index

- [Roadmap](roadmap.md): milestones, release targets, and current priorities.
- [Development](development.md): local workflow, testing, documentation, and
  release hygiene.
- [Architecture overview](architecture/overview.md): concise map of the
  implemented package.
- [Design documents](design/): methodology, technical specification,
  mathematical breakdown, and lineage.
- [Decision records](decisions/): accepted architecture and research decisions.
- [Experiments](experiments/): durable protocols and compact result summaries.
- [Research notes](notes/): observations and unresolved questions.
- [Release notes](releases/): release-specific notes when the changelog is not
  enough.

## Documentation Boundaries

- Use `docs/design/` for detailed, durable design references.
- Use `docs/architecture/` for shorter implementation maps.
- Use `docs/experiments/` for reviewed protocols and compact result summaries.
- Use `docs/notes/` for unresolved ideas, observations, and task-scoped
  coordination notes.
- Use `docs/decisions/` for formal proposed or accepted ADRs. Keep exploratory
  coordination in `docs/notes/` until it is ready for ADR review.

Do not duplicate the full documentation tree per version. Use Git history,
release notes, and ADR metadata to track when a decision or document changed.
