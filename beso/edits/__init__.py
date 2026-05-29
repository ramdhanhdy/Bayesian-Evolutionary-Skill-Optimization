"""Edit operations, application, and deterministic linting.

Implements the ``EditApplicator`` protocol (delegating to SkillOpt's substring
apply_edit/apply_patch via an adapter) and deterministic feasibility nu(z, e)
used as a hard pre-acquisition filter (schema / budget / invariant checks).

Planned modules: ``operations.py``, ``applicator.py``, ``lint.py``.
"""
