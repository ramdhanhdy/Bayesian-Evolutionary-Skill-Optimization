"""Evolutionary archive (the ``Archive`` protocol).

Replaces SkillOpt's single best_skill.md incumbent with a multi-tier archive
(best / Pareto / diverse / failed), Pareto win-count parent selection, and
constrained subset pruning under a size cap (Breakdown S4.2, S10; Spec S15).

Planned modules: ``manager.py``, ``pareto.py``, ``lineage.py``.
"""
