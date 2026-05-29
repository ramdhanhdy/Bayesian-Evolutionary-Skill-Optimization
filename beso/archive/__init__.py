"""Evolutionary archive (the ``Archive`` protocol).

Replaces SkillOpt's single best_skill.md incumbent with a multi-tier archive
(best / Pareto / diverse / failed), Pareto win-count parent selection, and
constrained subset pruning under a size cap (Breakdown S4.2, S10; Spec S15).
"""

from beso.archive.manager import ArchiveConfig, EvolutionaryArchive
from beso.archive.pareto import (
    compute_pareto_win_counts,
    objective_vector,
    pareto_dominates,
    pareto_front,
)

__all__ = [
    "ArchiveConfig",
    "EvolutionaryArchive",
    "compute_pareto_win_counts",
    "objective_vector",
    "pareto_dominates",
    "pareto_front",
]
