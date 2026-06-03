# Decision Records

This directory stores formal Architecture Decision Records (ADRs) and
research-design decisions.

Use ADRs for consequential decisions that are proposed for review or accepted as
future guidance. Do not use ADRs for temporary notes, task boards, or speculative
ideas.

## Directory Layout

Group ADRs by the package minor version where the decision was introduced:

```text
docs/decisions/
  README.md
  v0.0/
    0001-example-decision.md
  v0.1/
    0002-example-decision.md
```

The directory answers when the decision was introduced. The metadata inside the
ADR answers whether it still applies later.

## ADR Metadata

Every ADR should start with:

```yaml
---
adr: 0001
title: Example decision
status: accepted
introduced_in: 0.1.0
applies_to: ">=0.1.0"
milestones: [M1]
supersedes: null
superseded_by: null
---
```

Allowed statuses:

- `proposed`
- `accepted`
- `deprecated`
- `superseded`

## ADR Template

```markdown
---
adr: 0001
title: Example decision
status: proposed
introduced_in: 0.1.0
applies_to: ">=0.1.0"
milestones: [M1]
supersedes: null
superseded_by: null
---

# ADR 0001: Example Decision

## Context

## Decision

## Alternatives Considered

## Consequences

## Version Lifecycle
```

## Index

| ADR | Status | Introduced | Title |
| --- | --- | --- | --- |
| [0001](v0.0/0001-separate-deployable-promotion-from-exploration-archive-admission.md) | proposed | `0.0.1` | Separate deployable promotion from exploration-archive admission under high-performance plateaus |
