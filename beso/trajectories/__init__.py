"""Trajectory and observation persistence (history H_t).

Stores rollout traces and noisy minibatch observations so the surrogate can
recover bar_y and SE from repeated evaluations. The store is SQLite-backed to
align with the available MCP SQLite server.

Planned modules: ``logger.py``, ``store.py``, ``filters.py``.
"""
