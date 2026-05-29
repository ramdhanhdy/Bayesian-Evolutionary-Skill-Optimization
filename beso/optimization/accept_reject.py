"""Paired, noise-aware validation gate."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from beso.core.protocols import GateDecision
from beso.core.types import EvaluationResult


@dataclass(frozen=True)
class AcceptanceGateConfig:
    """Statistical and constraint settings for validation gating."""

    alpha: float = 0.05
    confidence: float = 0.95
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 0
    noise_scaled_delta_c: float = 1.0
    exact_mcnemar_max_n: int = 50
    max_invalid_rate: float = 1.0
    max_cost_per_task: float | None = None
    max_latency_seconds: float | None = None
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")


@dataclass(frozen=True)
class PairedTestResult:
    """Paired validation statistics for candidate minus parent scores."""

    mean_diff: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    method: str


class PairedBootstrapAcceptanceGate:
    """AcceptanceGate implementation with paired bootstrap / McNemar tests."""

    def __init__(self, config: AcceptanceGateConfig | None = None) -> None:
        self.config = config or AcceptanceGateConfig()

    def decide(
        self,
        candidate_eval: EvaluationResult,
        parent_eval: EvaluationResult,
    ) -> GateDecision:
        diffs = paired_differences(candidate_eval, parent_eval)
        test = paired_test(
            diffs,
            candidate_scores=_ordered_scores(candidate_eval),
            parent_scores=_ordered_scores(parent_eval),
            config=self.config,
        )
        threshold = self.config.noise_scaled_delta_c * test.se
        constraints_satisfied, constraint_reason = self._constraints(candidate_eval)

        accepted = (
            constraints_satisfied
            and test.ci_low > 0.0
            and test.mean_diff >= threshold
            and test.p_value <= self.config.alpha
        )
        reason = _reason(
            accepted=accepted,
            method=test.method,
            constraints_satisfied=constraints_satisfied,
            constraint_reason=constraint_reason,
            ci_low=test.ci_low,
            mean_diff=test.mean_diff,
            threshold=threshold,
            p_value=test.p_value,
            alpha=self.config.alpha,
        )
        return GateDecision(
            candidate_id=candidate_eval.candidate_id,
            accepted=accepted,
            reason=reason,
            mean_diff=test.mean_diff,
            ci_low=test.ci_low,
            ci_high=test.ci_high,
            p_value=test.p_value,
            noise_scaled_threshold=threshold,
            constraints_satisfied=constraints_satisfied,
        )

    def _constraints(self, ev: EvaluationResult) -> tuple[bool, str]:
        if ev.invalid_rate > self.config.max_invalid_rate + self.config.eps:
            return False, "invalid_rate"
        if (
            self.config.max_cost_per_task is not None
            and ev.mean_cost_tokens > self.config.max_cost_per_task + self.config.eps
        ):
            return False, "cost"
        if (
            self.config.max_latency_seconds is not None
            and ev.mean_latency_seconds > self.config.max_latency_seconds + self.config.eps
        ):
            return False, "latency"
        return True, ""


def paired_differences(
    candidate_eval: EvaluationResult,
    parent_eval: EvaluationResult,
) -> np.ndarray:
    """Return d_i = r_i(candidate) - r_i(parent) on the exact same draw."""

    cand_ids = set(candidate_eval.per_example_scores)
    parent_ids = set(parent_eval.per_example_scores)
    if cand_ids != parent_ids:
        missing_parent = sorted(cand_ids - parent_ids)
        missing_candidate = sorted(parent_ids - cand_ids)
        raise ValueError(
            "candidate and parent validation draws must contain identical "
            f"example ids; missing_parent={missing_parent}, "
            f"missing_candidate={missing_candidate}"
        )
    if not cand_ids:
        raise ValueError("paired validation gate requires at least one example")
    ids = sorted(cand_ids)
    return np.asarray(
        [
            float(candidate_eval.per_example_scores[i])
            - float(parent_eval.per_example_scores[i])
            for i in ids
        ],
        dtype=np.float64,
    )


def paired_test(
    diffs: Sequence[float],
    *,
    candidate_scores: Sequence[float] | None = None,
    parent_scores: Sequence[float] | None = None,
    config: AcceptanceGateConfig | None = None,
) -> PairedTestResult:
    """Run the configured paired test over already aligned differences."""

    cfg = config or AcceptanceGateConfig()
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError("paired_test requires at least one finite difference")

    mean_diff = float(np.mean(d))
    se = _standard_error(d)
    ci_low, ci_high = _bootstrap_ci(d, cfg)
    method = "paired_bootstrap"
    p_value = _bootstrap_one_sided_p_value(d, cfg)

    if (
        candidate_scores is not None
        and parent_scores is not None
        and d.size <= cfg.exact_mcnemar_max_n
        and _is_binary(candidate_scores, cfg.eps)
        and _is_binary(parent_scores, cfg.eps)
    ):
        p_value = exact_mcnemar_one_sided(candidate_scores, parent_scores)
        method = "exact_mcnemar"

    return PairedTestResult(
        mean_diff=mean_diff,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        method=method,
    )


def apply_benjamini_hochberg(
    decisions: Sequence[GateDecision],
    *,
    alpha: float = 0.05,
) -> list[GateDecision]:
    """Apply Benjamini-Hochberg correction to a round of gate decisions."""

    if not decisions:
        return []
    m = len(decisions)
    ordered = sorted(enumerate(decisions), key=lambda row: row[1].p_value)
    cutoff_rank = -1
    for rank, (_, decision) in enumerate(ordered, start=1):
        if decision.p_value <= alpha * rank / m:
            cutoff_rank = rank

    accepted_ids = set()
    if cutoff_rank >= 1:
        accepted_ids = {
            decision.candidate_id
            for _, decision in ordered[:cutoff_rank]
            if decision.accepted
        }

    corrected: list[GateDecision] = []
    for decision in decisions:
        accepted = decision.accepted and decision.candidate_id in accepted_ids
        reason = decision.reason
        if decision.accepted and not accepted:
            reason = f"{reason}; bh_reject"
        elif accepted:
            reason = f"{reason}; bh_accept"
        corrected.append(replace(decision, accepted=accepted, reason=reason))
    return corrected


def exact_mcnemar_one_sided(
    candidate_scores: Sequence[float],
    parent_scores: Sequence[float],
) -> float:
    """Exact one-sided McNemar/binomial p-value for binary paired scores."""

    cand = np.asarray(candidate_scores, dtype=np.float64)
    parent = np.asarray(parent_scores, dtype=np.float64)
    if cand.shape != parent.shape:
        raise ValueError("candidate_scores and parent_scores must have the same shape")
    improvements = int(np.sum((cand > parent) & np.isclose(parent, 0.0)))
    regressions = int(np.sum((parent > cand) & np.isclose(cand, 0.0)))
    discordant = improvements + regressions
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(improvements, discordant + 1)
    )
    return float(tail / (2**discordant))


def _bootstrap_ci(
    diffs: np.ndarray,
    config: AcceptanceGateConfig,
) -> tuple[float, float]:
    if diffs.size == 1:
        val = float(diffs[0])
        return val, val
    means = _bootstrap_means(diffs, config)
    tail = (1.0 - config.confidence) / 2.0
    return (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


def _bootstrap_one_sided_p_value(
    diffs: np.ndarray,
    config: AcceptanceGateConfig,
) -> float:
    if diffs.size == 1:
        return 0.0 if float(diffs[0]) > 0.0 else 1.0
    means = _bootstrap_means(diffs, config)
    return float((np.sum(means <= 0.0) + 1.0) / (means.size + 1.0))


def _bootstrap_means(
    diffs: np.ndarray,
    config: AcceptanceGateConfig,
) -> np.ndarray:
    rng = np.random.default_rng(config.bootstrap_seed)
    idx = rng.integers(0, diffs.size, size=(config.bootstrap_samples, diffs.size))
    return diffs[idx].mean(axis=1)


def _standard_error(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _ordered_scores(ev: EvaluationResult) -> list[float]:
    return [float(ev.per_example_scores[i]) for i in sorted(ev.per_example_scores)]


def _is_binary(values: Sequence[float], eps: float) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isclose(arr, 0.0, atol=eps) | np.isclose(arr, 1.0, atol=eps)))


def _reason(
    *,
    accepted: bool,
    method: str,
    constraints_satisfied: bool,
    constraint_reason: str,
    ci_low: float,
    mean_diff: float,
    threshold: float,
    p_value: float,
    alpha: float,
) -> str:
    if accepted:
        return f"accepted:{method}"
    if not constraints_satisfied:
        return f"reject_constraint:{constraint_reason}"
    if ci_low <= 0.0:
        return f"reject_ci:{method}"
    if mean_diff < threshold:
        return "reject_noise_scaled_delta"
    if p_value > alpha:
        return f"reject_p_value:{method}"
    return "reject"


__all__ = [
    "AcceptanceGateConfig",
    "PairedBootstrapAcceptanceGate",
    "PairedTestResult",
    "apply_benjamini_hochberg",
    "exact_mcnemar_one_sided",
    "paired_differences",
    "paired_test",
]
