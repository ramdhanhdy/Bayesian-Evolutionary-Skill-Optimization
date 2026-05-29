"""Persistence layer for runs, history, and archive snapshots.

SQLite-backed storage of observations, candidates, archive entries, and run
metadata for resumability and post-hoc analysis (aligns with the MCP SQLite
server). Keeps experiment state inspectable, satisfying the v0 acceptance
criterion that the optimization trace be auditable.

Planned modules: ``sqlite_store.py``, ``schema.sql``.
"""
