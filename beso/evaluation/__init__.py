"""Evaluation layer: rollout scoring, metrics, splits, and the statistical gate.

Implements the ``Evaluator`` protocol (metric mu, aggregate hat_J_S with
per-example scores) and the ``AcceptanceGate`` (paired test + Benjamini-Hochberg
multiplicity control + noise-scaled delta + confidence-bounded constraints).
Split management enforces the disjoint roles D_fb / D_opt / D_val / D_test.

Planned modules: ``metrics.py``, ``splits.py``, ``gate.py``, ``judge.py``.
"""
