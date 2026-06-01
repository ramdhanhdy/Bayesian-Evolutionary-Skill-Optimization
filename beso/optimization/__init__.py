"""The BESO optimization loop and budget control.

Orchestrates parents -> reflection pool -> hard filter -> featurize -> surrogate
-> acquisition -> submodular batch -> evaluate -> gate -> archive update, under a
rollout budget. Hosts the ``RegimeDetector`` that auto-disables the surrogate
when it is not yet predictive (cold start / negligible candidate variance).
"""

from beso.optimization.accept_reject import (
    PARETO_CLEANUP_REASON,
    AcceptanceGateConfig,
    PairedBootstrapAcceptanceGate,
    PairedTestResult,
    apply_benjamini_hochberg,
    exact_mcnemar_one_sided,
    paired_differences,
    paired_test,
    secondary_metric_gains,
)
from beso.optimization.logger import JSONLLogger
from beso.optimization.loop import (
    BESOOptimizer,
    BESOOptimizerConfig,
    IterationRecord,
    OptimizationResult,
)
from beso.optimization.regime import (
    RegimeDetectorConfig,
    VarianceRankRegimeDetector,
    spearman_rank_correlation,
)

__all__ = [
    "PARETO_CLEANUP_REASON",
    "AcceptanceGateConfig",
    "BESOOptimizer",
    "BESOOptimizerConfig",
    "IterationRecord",
    "JSONLLogger",
    "OptimizationResult",
    "PairedBootstrapAcceptanceGate",
    "PairedTestResult",
    "RegimeDetectorConfig",
    "VarianceRankRegimeDetector",
    "apply_benjamini_hochberg",
    "exact_mcnemar_one_sided",
    "paired_differences",
    "paired_test",
    "secondary_metric_gains",
    "spearman_rank_correlation",
]
