"""Protocol boundary for the SkillOpt fork.

BESO forks Microsoft SkillOpt (github.com/microsoft/SkillOpt, MIT). This module
defines the *seam* between the two systems as structural interfaces
(:class:`typing.Protocol`), so that:

1. The reusable SkillOpt "plumbing" (execution harness/adapter, deterministic
   SKILL.md edit applicator, evaluator, dataloaders) is consumed by BESO through
   thin adapters in ``beso/adapters/skillopt.py`` rather than imported directly.
2. The new BESO "brains" (parallel reflection pool, featurizer, Bayesian
   surrogate, pool-normalized acquisition, submodular batch selection,
   statistical gate, evolutionary archive) are owned by BESO and depend only on
   these interfaces.

Scientific-rigor invariant
---------------------------
The ``ExecutionHarness``, ``Evaluator``, and ``DatasetProvider`` protocols are
shared by BOTH the BESO optimizer and the unmodified-SkillOpt baseline runner.
Only the selection layer differs between them. This guarantees the baseline runs
on identical tasks, seeds, dataset splits, and rollout budgets, which is
required to attribute any measured lift to Bayesian planning rather than to
incidental harness differences.

These are interfaces only; no concrete binding to SkillOpt symbols happens here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

from beso.core.types import (
    ArchiveEntry,
    Candidate,
    CandidateFeatures,
    EditProposal,
    EvaluationResult,
    Observation,
    SkillArtifact,
    SplitRole,
    SurrogatePrediction,
    Trajectory,
)


# --------------------------------------------------------------------------- #
# Small contract-bound data types                                              #
# --------------------------------------------------------------------------- #
@dataclass
class PoolStatistics:
    """Per-term summary statistics over a candidate pool C_t.

    Used to make the acquisition function dimensionless: each raw term
    (mu, sigma, diversity, cost, invalid risk) is normalized against these
    pool-relative statistics before the weighted sum, so the acquisition
    weights kappa/lambda/alpha/gamma are comparable across iterations.
    """

    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    mins: dict[str, float] = field(default_factory=dict)
    maxs: dict[str, float] = field(default_factory=dict)


@dataclass
class GateDecision:
    """Result of the validation-gate decision for one candidate (Spec S16.5).

    Records the paired test statistics so accept/reject decisions are auditable
    and so multiplicity correction (Benjamini-Hochberg) can be applied across a
    round's candidates downstream.
    """

    candidate_id: str
    accepted: bool
    reason: str = ""
    mean_diff: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    p_value: float = 1.0
    noise_scaled_threshold: float = 0.0
    constraints_satisfied: bool = True


# --------------------------------------------------------------------------- #
# Reused-from-SkillOpt boundary (consumed by BESO via adapters)                #
# --------------------------------------------------------------------------- #
@runtime_checkable
class SkillSerializer(Protocol):
    """Bridge between BESO's typed skill and SkillOpt's markdown SKILL.md.

    Wraps SkillOpt's skill rendering/parsing so BESO can hold a typed
    :class:`SkillArtifact` internally while the harness consumes a markdown
    document (best_skill.md / skill_vXXXX.md).
    """

    def render(self, skill: SkillArtifact) -> str:
        """Serialize a skill artifact to its markdown SKILL.md representation."""
        ...

    def parse(self, markdown: str, skill_id: str) -> SkillArtifact:
        """Parse a markdown SKILL.md document back into a typed artifact."""
        ...


@runtime_checkable
class DatasetProvider(Protocol):
    """Provides reproducible task batches from train/val/test split dirs.

    Wraps SkillOpt's dataloaders (skillopt/envs/<benchmark>/dataloader.py) and
    the split-directory layout (train/ val/ test/). Batches must be deterministic
    given a seed so BESO and the SkillOpt baseline draw identical examples.
    """

    def batch(self, role: SplitRole, size: int, seed: int) -> Sequence[str]:
        """Return a deterministic list of example ids for the given split role."""
        ...

    def split_size(self, role: SplitRole) -> int:
        """Total number of examples available in a split role."""
        ...


@runtime_checkable
class ExecutionHarness(Protocol):
    """Runs the frozen agent with a skill and returns scored trajectories.

    This is SkillOpt's harness-agnostic adapter interface: it injects the
    current skill into the agent context, runs the native harness (direct chat,
    Codex, Claude Code, ALFWorld, ...), and returns observable traces. It does
    not change model weights (Theta_frozen).
    """

    def rollout(
        self,
        skill: SkillArtifact,
        example_ids: Sequence[str],
        seed: int,
    ) -> list[Trajectory]:
        """Execute (y, tau) = Phi(x; C(z)) over a batch and return trajectories."""
        ...


@runtime_checkable
class EditApplicator(Protocol):
    """Deterministic SKILL.md patcher (reused from SkillOpt).

    Applies a structured edit instruction (op/section/text) to a parent skill,
    producing a child artifact. Pure and deterministic: z' = e(z).
    """

    def apply(self, parent: SkillArtifact, edit: EditProposal) -> SkillArtifact:
        """Apply a single bounded edit, returning the child skill artifact."""
        ...

    def apply_sequence(
        self, parent: SkillArtifact, edits: Sequence[EditProposal]
    ) -> SkillArtifact:
        """Apply an ordered sequence of edits e_K o ... o e_1 (z)."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores trajectories into the metric mu (reused from SkillOpt)."""

    def score_trajectory(self, trajectory: Trajectory) -> float:
        """Single-example score r_i(z) = mu(y_i, tau_i, m_i)."""
        ...

    def evaluate(
        self, skill: SkillArtifact, role: SplitRole, example_ids: Sequence[str], seed: int
    ) -> EvaluationResult:
        """Aggregate evaluation hat_J_S(z) with per-example scores retained."""
        ...


# --------------------------------------------------------------------------- #
# New BESO brains (owned by BESO; the contribution layer)                      #
# --------------------------------------------------------------------------- #
@runtime_checkable
class ReflectionProposer(Protocol):
    """Parallel candidate-pool generation (replaces SkillOpt's 1-edit proposal).

    Modified optimizer-model prompt returns a dense pool of bounded edits in a
    single call: e_{t,j} ~ Q_psi(e | z_p, traces, feedback, archive, rejected).
    """

    def propose_pool(
        self,
        parent: SkillArtifact,
        trajectories: Sequence[Trajectory],
        rejected: Sequence[EditProposal],
        pool_size: int,
    ) -> list[EditProposal]:
        """Return ``pool_size`` candidate edits for a parent skill."""
        ...


@runtime_checkable
class Featurizer(Protocol):
    """Builds the block-separated feature map phi(z) (Breakdown S6).

    Features are parent-centered (delta vs parent) to improve stationarity.
    Blocks remain separate; normalization/assembly is done by the surrogate's
    own normalizer so the high-dimensional text block cannot swamp the cheap
    structured signals.
    """

    def featurize(
        self,
        candidate: Candidate,
        parent: Optional[SkillArtifact],
        history: Sequence[Observation],
    ) -> CandidateFeatures:
        """Compute phi(z) blocks for one candidate."""
        ...


@runtime_checkable
class Surrogate(Protocol):
    """Bayesian surrogate over candidate utility (Breakdown S7).

    v0 default: bootstrap-bagged ensemble modeling the parent-relative delta
    Delta(z, z_p), exposing calibrated mu_t and sigma_t (epistemic + aleatoric).
    """

    def fit(
        self,
        features: Sequence[CandidateFeatures],
        observations: Sequence[Observation],
    ) -> None:
        """Refit the posterior p(f | H_t) from evaluated history."""
        ...

    def predict(self, features: CandidateFeatures) -> SurrogatePrediction:
        """Return mu_t(z), sigma_t(z) with variance decomposition."""
        ...

    @property
    def is_calibrated(self) -> bool:
        """Whether the surrogate currently meets its calibration threshold."""
        ...


@runtime_checkable
class AcquisitionFunction(Protocol):
    """Pool-normalized acquisition a_BESO (Breakdown S8.7).

        a(z) = mu~ + kappa*sigma~ + lambda*d~(z,A) - alpha*c~(z) - gamma*q~_invalid(z)

    Each term is normalized against ``pool_stats`` so weights are dimensionless.
    """

    def score(
        self,
        candidate: Candidate,
        prediction: SurrogatePrediction,
        archive: Sequence[ArchiveEntry],
        pool_stats: PoolStatistics,
    ) -> float:
        """Acquisition value for a single candidate."""
        ...


@runtime_checkable
class BatchSelector(Protocol):
    """Submodular down-selection of the pool to top-K (Lineage S7.2).

    Greedy max-min / facility-location (or DPP) selection that updates the
    reference set with already-selected members during the pass to avoid
    intra-batch near-duplicates.
    """

    def select(self, candidates: Sequence[Candidate], k: int) -> list[Candidate]:
        """Choose K diverse, high-acquisition candidates to evaluate."""
        ...


@runtime_checkable
class AcceptanceGate(Protocol):
    """Validation gate: paired test + multiplicity control + noise-scaled delta.

    Decides Accept(z) using paired per-example differences d_i = r_i(z) - r_i(p)
    on D_val, a noise-scaled threshold, and confidence-bounded constraints.
    Round-level Benjamini-Hochberg correction is applied by the caller across
    the returned decisions.
    """

    def decide(
        self,
        candidate_eval: EvaluationResult,
        parent_eval: EvaluationResult,
    ) -> GateDecision:
        """Return an auditable accept/reject decision for one candidate."""
        ...


@runtime_checkable
class Archive(Protocol):
    """Evolutionary archive replacing SkillOpt's single best_skill.md incumbent.

    Maintains best / Pareto / diverse / failed tiers, selects parents, and prunes
    under a size cap (Breakdown S4.2, S10; Spec S15).
    """

    def update(self, candidates: Sequence[Candidate], evals: Sequence[EvaluationResult]) -> None:
        """Incorporate newly evaluated candidates into the archive."""
        ...

    def select_parents(self, n: int, seed: int) -> list[ArchiveEntry]:
        """Sample parents by Pareto win-count + UCB-style score + diversity."""
        ...

    def best(self) -> Optional[ArchiveEntry]:
        """Return the best-validated entry (the deployable artifact)."""
        ...

    def entries(self) -> list[ArchiveEntry]:
        """All current archive entries."""
        ...


@runtime_checkable
class RegimeDetector(Protocol):
    """Decides whether the Bayesian layer is currently worth using.

    Monitors candidate-score variance Var_{z in C_t}[J(z)] and surrogate
    calibration/rank-correlation; when uninformative (cold start or negligible
    variance), BESO falls back to greedy/random selection so the surrogate never
    adds overhead without gain. Doubles as the built-in "no surrogate" ablation.
    """

    def use_surrogate(
        self,
        surrogate: Surrogate,
        recent_scores: Sequence[float],
    ) -> bool:
        """Whether to drive acquisition with the surrogate this iteration."""
        ...


__all__ = [
    "PoolStatistics",
    "GateDecision",
    "SkillSerializer",
    "DatasetProvider",
    "ExecutionHarness",
    "EditApplicator",
    "Evaluator",
    "ReflectionProposer",
    "Featurizer",
    "Surrogate",
    "AcquisitionFunction",
    "BatchSelector",
    "AcceptanceGate",
    "Archive",
    "RegimeDetector",
]
