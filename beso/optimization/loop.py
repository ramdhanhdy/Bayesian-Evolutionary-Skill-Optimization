"""Main BESO optimizer orchestration loop."""

from __future__ import annotations

import difflib
import inspect
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from beso.core.protocols import (
    AcceptanceGate,
    AcquisitionFunction,
    Archive,
    BatchSelector,
    DatasetProvider,
    EditApplicator,
    Evaluator,
    Featurizer,
    GateDecision,
    PoolStatistics,
    ReflectionProposer,
    RegimeDetector,
    Surrogate,
)
from beso.core.types import (
    ArchiveEntry,
    Candidate,
    CandidateFeatures,
    EditCategory,
    EditProposal,
    EditOperation,
    EvaluationResult,
    Observation,
    RolloutBudget,
    SkillArtifact,
    SkillSection,
    SplitRole,
    SurrogatePrediction,
    Trajectory,
)
from beso.optimization.accept_reject import (
    PARETO_CLEANUP_REASON,
    apply_benjamini_hochberg,
)
from beso.optimization.logger import JSONLLogger

CandidateFilter = Callable[[Candidate], bool]
MultiplicityCorrection = Callable[[Sequence[GateDecision]], list[GateDecision]]


@dataclass(frozen=True)
class BESOOptimizerConfig:
    """Loop-level settings; mathematical work stays in injected modules."""

    max_iterations: int = 30
    candidate_pool_size: int = 24
    batch_size: int = 2
    parent_count: int = 1
    optimization_batch_size: int = 8
    feedback_batch_size: int = 0
    validation_batch_size: int = 8
    seed: int = 0
    fallback_strategy: str = "random"
    rotate_validation: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.parent_count <= 0:
            raise ValueError("parent_count must be positive")
        if self.optimization_batch_size <= 0:
            raise ValueError("optimization_batch_size must be positive")
        if self.feedback_batch_size < 0:
            raise ValueError("feedback_batch_size must be non-negative")
        if self.validation_batch_size <= 0:
            raise ValueError("validation_batch_size must be positive")
        if self.fallback_strategy not in {"random", "greedy"}:
            raise ValueError("fallback_strategy must be 'random' or 'greedy'")


@dataclass
class IterationRecord:
    """Auditable state transition for one optimizer iteration."""

    iteration: int
    used_surrogate: bool
    fallback_reason: str = ""
    parent_ids: list[str] = field(default_factory=list)
    pool_ids: list[str] = field(default_factory=list)
    selected_ids: list[str] = field(default_factory=list)
    evaluated_ids: list[str] = field(default_factory=list)
    accepted_ids: list[str] = field(default_factory=list)
    rejected_ids: list[str] = field(default_factory=list)
    budget_spent: int = 0
    budget_remaining: int = 0


@dataclass
class OptimizationResult:
    """Final optimizer output and full loop trace."""

    best: Optional[ArchiveEntry]
    budget: RolloutBudget
    iterations: list[IterationRecord]
    observations: list[Observation]
    rejected_edits: list[EditProposal]


class BESOOptimizer:
    """Thin conductor that wires protocol implementations together."""

    def __init__(
        self,
        *,
        dataset: DatasetProvider,
        evaluator: Evaluator,
        proposer: ReflectionProposer,
        applicator: EditApplicator,
        featurizer: Featurizer,
        surrogate: Surrogate,
        acquisition: AcquisitionFunction,
        batch_selector: BatchSelector,
        gate: AcceptanceGate,
        archive: Archive,
        regime_detector: RegimeDetector,
        config: Optional[BESOOptimizerConfig] = None,
        candidate_filter: CandidateFilter | None = None,
        multiplicity_correction: MultiplicityCorrection | None = None,
        logger: Optional[JSONLLogger] = None,
    ) -> None:
        self.dataset = dataset
        self.evaluator = evaluator
        self.proposer = proposer
        self.applicator = applicator
        self.featurizer = featurizer
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.batch_selector = batch_selector
        self.gate = gate
        self.archive = archive
        self.regime_detector = regime_detector
        self.config = config or BESOOptimizerConfig()
        self.candidate_filter = candidate_filter or (lambda _: True)
        self.multiplicity_correction = (
            multiplicity_correction or apply_benjamini_hochberg
        )
        self.logger = logger

        self.observations: list[Observation] = []
        self.features: dict[str, CandidateFeatures] = {}
        self.trajectories: dict[str, list[Trajectory]] = {}
        self.rejected_edits: list[EditProposal] = []
        self.validation_cache: dict[tuple[str, tuple[str, ...]], EvaluationResult] = {}

    def optimize(
        self,
        initial_skill: SkillArtifact,
        budget: RolloutBudget,
    ) -> OptimizationResult:
        """Run BESO until rollout budget or iteration cap stops the loop."""

        iterations: list[IterationRecord] = []
        if budget.exhausted:
            return OptimizationResult(
                best=self.archive.best(),
                budget=budget,
                iterations=iterations,
                observations=self.observations,
                rejected_edits=self.rejected_edits,
            )

        seed_candidate = Candidate(
            candidate_id=initial_skill.skill_id,
            skill=initial_skill,
        )
        seed_eval = self._evaluate_with_budget(
            initial_skill,
            SplitRole.VALIDATION_GATE,
            self.config.validation_batch_size,
            self.config.seed,
            budget,
        )
        if seed_eval is not None:
            self._remember_evaluation(seed_candidate, seed_eval)
            self.validation_cache[
                (seed_candidate.candidate_id, tuple(seed_eval.per_example_scores))
            ] = seed_eval
            seed_candidate.features = self.featurizer.featurize(
                seed_candidate,
                None,
                self.observations,
            )
            self.features[seed_candidate.candidate_id] = seed_candidate.features
            self.archive.update([seed_candidate], [seed_eval])

        if self.config.feedback_batch_size > 0 and not budget.exhausted:
            feedback_eval = self._evaluate_with_budget(
                initial_skill,
                SplitRole.FEEDBACK_TRAIN,
                self.config.feedback_batch_size,
                self.config.seed,
                budget,
            )
            if feedback_eval is not None:
                self._remember_evaluation(seed_candidate, feedback_eval)

        for iteration in range(self.config.max_iterations):
            if budget.exhausted:
                break
            record = self._run_iteration(iteration, budget)
            if record is None:
                break
            iterations.append(record)

        return OptimizationResult(
            best=self.archive.best(),
            budget=budget,
            iterations=iterations,
            observations=list(self.observations),
            rejected_edits=list(self.rejected_edits),
        )

    def _run_iteration(
        self,
        iteration: int,
        budget: RolloutBudget,
    ) -> IterationRecord | None:
        parents = self.archive.select_parents(self.config.parent_count, self._seed(iteration))
        if not parents:
            return None

        pool = self._build_candidate_pool(parents, iteration)
        if not pool:
            return None

        for candidate in pool:
            parent = self._parent_artifact(candidate.parent_id, parents)
            candidate.features = self.featurizer.featurize(
                candidate,
                parent,
                self.observations,
            )
            self.features[candidate.candidate_id] = candidate.features

        use_surrogate = self._precheck_surrogate()
        fallback_reason = "" if use_surrogate else "regime_precheck"
        if use_surrogate:
            use_surrogate, fallback_reason = self._score_with_surrogate(pool)

        selected = (
            self._select_with_acquisition(pool)
            if use_surrogate
            else self._fallback_select(pool, iteration)
        )
        if not selected:
            return None

        opt_evals: dict[str, EvaluationResult] = {}
        val_evals: dict[str, EvaluationResult] = {}
        decisions: list[GateDecision] = []

        for candidate in selected:
            if budget.exhausted:
                break
            opt_eval = self._evaluate_with_budget(
                candidate.skill,
                SplitRole.OPTIMIZATION_MINIBATCH,
                self.config.optimization_batch_size,
                self._seed(iteration, offset=101),
                budget,
            )
            if opt_eval is None:
                break
            self._remember_evaluation(candidate, opt_eval)
            opt_evals[candidate.candidate_id] = opt_eval

            decision, val_eval = self._gate_candidate(candidate, iteration, budget)
            if decision is not None and val_eval is not None:
                decisions.append(decision)
                val_evals[candidate.candidate_id] = val_eval

        decisions = self.multiplicity_correction(decisions) if decisions else []
        accepted_ids = {d.candidate_id for d in decisions if d.accepted}
        rejected_ids = {d.candidate_id for d in decisions if not d.accepted}
        cleanup_ids = {
            d.candidate_id
            for d in decisions
            if d.accepted and d.reason.startswith(PARETO_CLEANUP_REASON)
        }
        accepted = [c for c in selected if c.candidate_id in accepted_ids]
        accepted_evals = [val_evals[c.candidate_id] for c in accepted]
        if accepted:
            self._archive_update(accepted, accepted_evals, cleanup_ids)
        for candidate in selected:
            if candidate.candidate_id in rejected_ids and candidate.edit is not None:
                self.rejected_edits.append(candidate.edit)

        record = IterationRecord(
            iteration=iteration,
            used_surrogate=use_surrogate,
            fallback_reason=fallback_reason,
            parent_ids=[p.candidate_id for p in parents],
            pool_ids=[c.candidate_id for c in pool],
            selected_ids=[c.candidate_id for c in selected],
            evaluated_ids=list(opt_evals),
            accepted_ids=sorted(accepted_ids),
            rejected_ids=sorted(rejected_ids),
            budget_spent=budget.spent_rollouts,
            budget_remaining=budget.remaining,
        )
        self._log_iteration(
            iteration=iteration,
            budget=budget,
            pool=pool,
            selected=selected,
            parents=parents,
            opt_evals=opt_evals,
            val_evals=val_evals,
            decisions=decisions,
        )
        return record

    def _log_iteration(
        self,
        *,
        iteration: int,
        budget: RolloutBudget,
        pool: Sequence[Candidate],
        selected: Sequence[Candidate],
        parents: Sequence[ArchiveEntry],
        opt_evals: dict[str, EvaluationResult],
        val_evals: dict[str, EvaluationResult],
        decisions: Sequence[GateDecision],
    ) -> None:
        if self.logger is None:
            return

        evals: dict[str, dict[str, dict[str, float]]] = {}
        for cid, ev in opt_evals.items():
            evals.setdefault(cid, {})["optimization"] = dict(ev.per_example_scores)
        for cid, ev in val_evals.items():
            evals.setdefault(cid, {})["validation"] = dict(ev.per_example_scores)

        payload = {
            "iteration": iteration,
            "spent_rollouts": budget.spent_rollouts,
            "budget_remaining": budget.remaining,
            "pool_edits": [c.edit for c in pool if c.edit is not None],
            "diffs": {
                c.candidate_id: self._candidate_diff(c, parents) for c in selected
            },
            "predictions": {c.candidate_id: c.prediction for c in pool},
            "acquisition_scores": {
                c.candidate_id: c.acquisition_score for c in pool
            },
            "selected_ids": [c.candidate_id for c in selected],
            "evals": evals,
            "gate_decisions": list(decisions),
            "archive_snapshot": list(self.archive.entries()),
        }
        self.logger.log(payload)

    def _candidate_diff(
        self,
        candidate: Candidate,
        parents: Sequence[ArchiveEntry],
    ) -> str:
        parent = self._parent_artifact(candidate.parent_id, parents)
        parent_doc = parent.document if parent is not None else ""
        diff = difflib.unified_diff(
            parent_doc.splitlines(),
            candidate.skill.document.splitlines(),
            fromfile="parent",
            tofile="child",
            lineterm="",
        )
        return "\n".join(diff)

    def _archive_update(
        self,
        accepted: Sequence[Candidate],
        accepted_evals: Sequence[EvaluationResult],
        cleanup_ids: set[str],
    ) -> None:
        """Update the archive, forwarding cleanup ids when supported."""

        if cleanup_ids and self._archive_supports_cleanup():
            self.archive.update(accepted, accepted_evals, cleanup_ids=cleanup_ids)
        else:
            self.archive.update(accepted, accepted_evals)

    def _archive_supports_cleanup(self) -> bool:
        try:
            params = inspect.signature(self.archive.update).parameters
        except (TypeError, ValueError):
            return False
        return "cleanup_ids" in params

    def _build_candidate_pool(
        self,
        parents: Sequence[ArchiveEntry],
        iteration: int,
    ) -> list[Candidate]:
        per_parent = max(1, int(np.ceil(self.config.candidate_pool_size / len(parents))))
        candidates: list[Candidate] = []
        seen_documents: set[str] = set()
        for parent in parents:
            edits = self.proposer.propose_pool(
                parent.artifact,
                self.trajectories.get(parent.candidate_id, []),
                self.rejected_edits,
                per_parent,
            )
            for edit in edits:
                child = self.applicator.apply(parent.artifact, edit)
                if child.document == parent.artifact.document:
                    continue
                candidate_id = child.skill_id or f"{parent.candidate_id}:{edit.edit_id}"
                candidate = Candidate(
                    candidate_id=candidate_id,
                    skill=child,
                    parent_id=parent.candidate_id,
                    edit=edit,
                )
                if child.document in seen_documents:
                    continue
                if not self.candidate_filter(candidate):
                    continue
                seen_documents.add(child.document)
                candidates.append(candidate)
                if len(candidates) >= self.config.candidate_pool_size:
                    return candidates
        return candidates

    def _precheck_surrogate(self) -> bool:
        recent_scores = [obs.observed_score for obs in self.observations]
        return self.regime_detector.use_surrogate(self.surrogate, recent_scores)

    def _score_with_surrogate(self, pool: Sequence[Candidate]) -> tuple[bool, str]:
        try:
            self.surrogate.fit(list(self.features.values()), self.observations)
            predictions = [self.surrogate.predict(c.features) for c in pool if c.features]
        except (RuntimeError, ValueError):
            return False, "surrogate_fit_failed"
        if len(predictions) != len(pool):
            return False, "missing_features"
        if not self.regime_detector.use_surrogate(
            self.surrogate,
            [p.mu for p in predictions],
        ):
            return False, "regime_pool_check"
        self._score_pool(pool, predictions)
        return True, ""

    def _score_pool(
        self,
        pool: Sequence[Candidate],
        predictions: Sequence[SurrogatePrediction],
    ) -> None:
        if hasattr(self.acquisition, "score_pool"):
            self.acquisition.score_pool(pool, predictions, self.archive.entries())
            return
        stats = _prediction_pool_statistics(predictions)
        pred_by_id = {p.candidate_id: p for p in predictions}
        for candidate in pool:
            pred = pred_by_id[candidate.candidate_id]
            candidate.prediction = pred
            candidate.acquisition_score = self.acquisition.score(
                candidate,
                pred,
                self.archive.entries(),
                stats,
            )

    def _select_with_acquisition(self, pool: Sequence[Candidate]) -> list[Candidate]:
        if hasattr(self.batch_selector, "archive"):
            setattr(self.batch_selector, "archive", self.archive.entries())
        if hasattr(self.batch_selector, "feature_lookup") and hasattr(
            self.archive,
            "feature_lookup",
        ):
            setattr(self.batch_selector, "feature_lookup", self.archive.feature_lookup)
        return self.batch_selector.select(pool, self.config.batch_size)

    def _fallback_select(
        self,
        pool: Sequence[Candidate],
        iteration: int,
    ) -> list[Candidate]:
        candidates = list(pool)
        if self.config.fallback_strategy == "greedy":
            ranked = sorted(
                enumerate(candidates),
                key=lambda row: (-_fallback_repair_priority(row[1]), row[0]),
            )
            return [candidate for _, candidate in ranked[: self.config.batch_size]]
        rng = np.random.default_rng(self._seed(iteration, offset=303))
        rng.shuffle(candidates)
        return candidates[: self.config.batch_size]

    def _gate_candidate(
        self,
        candidate: Candidate,
        iteration: int,
        budget: RolloutBudget,
    ) -> tuple[GateDecision | None, EvaluationResult | None]:
        parent_entry = self._archive_entry(candidate.parent_id)
        if parent_entry is None:
            return None, None
        seed = self._validation_seed(iteration)
        val_ids = self._batch_ids(
            SplitRole.VALIDATION_GATE,
            self.config.validation_batch_size,
            seed,
            budget.remaining,
        )
        if not val_ids:
            return None, None

        parent_key = (parent_entry.candidate_id, tuple(val_ids))
        parent_eval = self.validation_cache.get(parent_key)
        parent_cost = 0 if parent_eval is not None else len(val_ids)
        needed = len(val_ids) + parent_cost
        if budget.remaining < needed:
            return None, None

        if parent_eval is None:
            parent_eval = self.evaluator.evaluate(
                parent_entry.artifact,
                SplitRole.VALIDATION_GATE,
                val_ids,
                seed,
            )
            budget.spend(parent_eval.n)
            self.validation_cache[parent_key] = parent_eval

        candidate_eval = self.evaluator.evaluate(
            candidate.skill,
            SplitRole.VALIDATION_GATE,
            val_ids,
            seed,
        )
        budget.spend(candidate_eval.n)
        self.validation_cache[(candidate.candidate_id, tuple(val_ids))] = candidate_eval
        self._remember_evaluation(candidate, candidate_eval)
        return self.gate.decide(candidate_eval, parent_eval), candidate_eval

    def _evaluate_with_budget(
        self,
        skill: SkillArtifact,
        role: SplitRole,
        requested_size: int,
        seed: int,
        budget: RolloutBudget,
    ) -> EvaluationResult | None:
        ids = self._batch_ids(role, requested_size, seed, budget.remaining)
        if not ids:
            return None
        result = self.evaluator.evaluate(skill, role, ids, seed)
        budget.spend(result.n)
        return result

    def _batch_ids(
        self,
        role: SplitRole,
        requested_size: int,
        seed: int,
        remaining: int,
    ) -> tuple[str, ...]:
        size = min(requested_size, remaining)
        if size <= 0:
            return ()
        return tuple(self.dataset.batch(role, size, seed))

    def _remember_evaluation(
        self,
        candidate: Candidate,
        ev: EvaluationResult,
    ) -> None:
        self.observations.append(
            Observation(
                candidate_id=candidate.candidate_id,
                batch_ids=tuple(ev.per_example_scores),
                observed_score=ev.mean_score,
                cost=float(ev.n),
                split=ev.split,
            )
        )
        if ev.split in {
            SplitRole.FEEDBACK_TRAIN,
            SplitRole.OPTIMIZATION_MINIBATCH,
        }:
            self.trajectories[candidate.candidate_id] = list(ev.trajectories)

    def _parent_artifact(
        self,
        parent_id: str | None,
        parents: Sequence[ArchiveEntry],
    ) -> SkillArtifact | None:
        for parent in parents:
            if parent.candidate_id == parent_id:
                return parent.artifact
        entry = self._archive_entry(parent_id)
        return entry.artifact if entry is not None else None

    def _archive_entry(self, candidate_id: str | None) -> ArchiveEntry | None:
        if candidate_id is None:
            return None
        for entry in self.archive.entries():
            if entry.candidate_id == candidate_id:
                return entry
        return None

    def _seed(self, iteration: int, *, offset: int = 0) -> int:
        return int(self.config.seed + iteration * 997 + offset)

    def _validation_seed(self, iteration: int) -> int:
        if self.config.rotate_validation:
            return self._seed(iteration, offset=211)
        return self.config.seed


def _prediction_pool_statistics(predictions: Sequence[SurrogatePrediction]) -> PoolStatistics:
    mus = np.asarray([p.mu for p in predictions], dtype=np.float64)
    sigmas = np.asarray([p.sigma for p in predictions], dtype=np.float64)
    return PoolStatistics(
        means={"mu": float(np.mean(mus)), "sigma": float(np.mean(sigmas))},
        stds={"mu": float(np.std(mus, ddof=0)), "sigma": float(np.std(sigmas, ddof=0))},
        mins={"mu": float(np.min(mus)), "sigma": float(np.min(sigmas))},
        maxs={"mu": float(np.max(mus)), "sigma": float(np.max(sigmas))},
    )


def _fallback_repair_priority(candidate: Candidate) -> float:
    """Prioritize likely repair edits during cold start while preserving order."""

    edit = candidate.edit
    if edit is None:
        return 0.0

    score = 0.0
    if edit.operation in {EditOperation.REPLACE, EditOperation.DELETE}:
        score += 4.0
    elif edit.operation is EditOperation.INSERT_AFTER:
        score += 1.0

    if edit.target_section is SkillSection.CORE_PROCEDURE:
        score += 8.0
    elif edit.target_section in {
        SkillSection.RECOVERY_RULES,
        SkillSection.VERIFICATION_CHECKLIST,
        SkillSection.OUTPUT_RULES,
    }:
        score += 1.0

    if edit.category in {
        EditCategory.REPLACE_RULE,
        EditCategory.DELETE_RULE,
        EditCategory.ADD_RECOVERY_RULE,
    }:
        score += 2.0

    evidence = " ".join(
        [
            edit.target,
            edit.content,
            edit.rationale,
            edit.expected_effect,
        ]
    ).lower()
    if "return 0" in evidence:
        score += 10.0
    if "do not recalculate" in evidence:
        score += 4.0

    document = candidate.skill.document.lower()
    if "return 0" not in document:
        score += 16.0
    elif edit.operation is EditOperation.APPEND:
        score -= 1.0

    return score


__all__ = [
    "BESOOptimizer",
    "BESOOptimizerConfig",
    "IterationRecord",
    "OptimizationResult",
]
