This file is a merged representation of a subset of the codebase, containing files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where comments have been removed.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching these patterns are excluded: migrations/**, .next/**, .repomix/**, *.env, *.env.local, .eslintrc.json, .gitignore, node_modules/**, *.pdf, /.venv, /.repomix, /__pycache__, /.vscode, *.csv, *.log., unused/, *.md, *.txt, /.venv/, .venv/**, package-lock.json, *.db, prd.md, tsconfig.tsbuildinfo, *.nprmc, .gitignore, .trae, .repomix
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Code comments have been removed from supported file types
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
beso/__init__.py
beso/acquisition/__init__.py
beso/adapters/__init__.py
beso/adapters/skillopt.py
beso/archive/__init__.py
beso/artifacts/__init__.py
beso/compiler/__init__.py
beso/core/__init__.py
beso/core/protocols.py
beso/core/types.py
beso/edits/__init__.py
beso/evaluation/__init__.py
beso/experiments/__init__.py
beso/features/__init__.py
beso/features/featurizer.py
beso/features/normalization.py
beso/llm/__init__.py
beso/optimization/__init__.py
beso/reflection/__init__.py
beso/store/__init__.py
beso/surrogate/__init__.py
beso/surrogate/base.py
beso/surrogate/calibration.py
beso/surrogate/ensemble.py
beso/trajectories/__init__.py
configs/default.yaml
docs/Bayesian Evolutionary Skill Optimization (BESO) - GEPA SkillOpt BESO Mathematical Lineage.md
docs/Bayesian Evolutionary Skill Optimization (BESO) - Mathematical Breakdown.md
docs/Bayesian Evolutionary Skill Optimization (BESO) - Methodology.md
docs/Bayesian Evolutionary Skill Optimization (BESO) - Technical Specification.md
pyproject.toml
tests/__init__.py
tests/test_core_contracts.py
tests/test_features.py
tests/test_surrogate.py
```

# Files

## File: beso/__init__.py
````python
__version__ = "0.0.1"
````

## File: beso/acquisition/__init__.py
````python

````

## File: beso/adapters/__init__.py
````python

````

## File: beso/adapters/skillopt.py
````python
from __future__ import annotations

from typing import Sequence

from beso.core.types import (
    EditProposal,
    EvaluationResult,
    SkillArtifact,
    SplitRole,
    Trajectory,
)

_UNBOUND = (
    "SkillOpt is not yet wired in. Vendor/install `skillopt` and bind this "
    "adapter to the corresponding upstream component before use."
)


class SkillOptSerializer:


    def render(self, skill: SkillArtifact) -> str:
        raise NotImplementedError(_UNBOUND)

    def parse(self, markdown: str, skill_id: str) -> SkillArtifact:
        raise NotImplementedError(_UNBOUND)


class SkillOptDatasetProvider:


    def batch(self, role: SplitRole, size: int, seed: int) -> Sequence[str]:
        raise NotImplementedError(_UNBOUND)

    def split_size(self, role: SplitRole) -> int:
        raise NotImplementedError(_UNBOUND)


class SkillOptHarness:


    def rollout(
        self, skill: SkillArtifact, example_ids: Sequence[str], seed: int
    ) -> list[Trajectory]:
        raise NotImplementedError(_UNBOUND)


class SkillOptEditApplicator:


    def apply(self, parent: SkillArtifact, edit: EditProposal) -> SkillArtifact:
        raise NotImplementedError(_UNBOUND)

    def apply_sequence(
        self, parent: SkillArtifact, edits: Sequence[EditProposal]
    ) -> SkillArtifact:
        raise NotImplementedError(_UNBOUND)


class SkillOptEvaluator:


    def score_trajectory(self, trajectory: Trajectory) -> float:
        raise NotImplementedError(_UNBOUND)

    def evaluate(
        self,
        skill: SkillArtifact,
        role: SplitRole,
        example_ids: Sequence[str],
        seed: int,
    ) -> EvaluationResult:
        raise NotImplementedError(_UNBOUND)
````

## File: beso/archive/__init__.py
````python

````

## File: beso/artifacts/__init__.py
````python

````

## File: beso/compiler/__init__.py
````python

````

## File: beso/core/__init__.py
````python
from beso.core.types import (
    ArchiveEntry,
    Candidate,
    CandidateFeatures,
    EditCategory,
    EditOperation,
    EditProposal,
    EvaluationResult,
    Observation,
    RolloutBudget,
    SkillArtifact,
    SkillMetadata,
    SkillSection,
    SplitRole,
    SurrogatePrediction,
    Trajectory,
)

__all__ = [
    "ArchiveEntry",
    "Candidate",
    "CandidateFeatures",
    "EditCategory",
    "EditOperation",
    "EditProposal",
    "EvaluationResult",
    "Observation",
    "RolloutBudget",
    "SkillArtifact",
    "SkillMetadata",
    "SkillSection",
    "SplitRole",
    "SurrogatePrediction",
    "Trajectory",
]
````

## File: beso/core/protocols.py
````python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

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





@dataclass
class PoolStatistics:








    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    mins: dict[str, float] = field(default_factory=dict)
    maxs: dict[str, float] = field(default_factory=dict)


@dataclass
class GateDecision:







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







Whether to drive acquisition with the surrogate this iteration."""
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
````

## File: beso/core/types.py
````python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np





class SkillSection(str, Enum):


    GOAL = "goal"
    SCOPE = "scope"
    CORE_PROCEDURE = "core_procedure"
    REASONING_POLICY = "reasoning_policy"
    TOOL_USE_POLICY = "tool_use_policy"
    VERIFICATION_CHECKLIST = "verification_checklist"
    COMMON_FAILURE_MODES = "common_failure_modes"
    RECOVERY_RULES = "recovery_rules"
    OUTPUT_RULES = "output_rules"
    EXAMPLES = "examples"
    CHANGE_LOG = "change_log"


class EditOperation(str, Enum):








    APPEND = "append"
    INSERT_AFTER = "insert_after"
    REPLACE = "replace"
    DELETE = "delete"
    MERGE = "merge"


class EditCategory(str, Enum):







    ADD_RULE = "add_rule"
    DELETE_RULE = "delete_rule"
    REPLACE_RULE = "replace_rule"
    SPECIALIZE_RULE = "specialize_rule"
    GENERALIZE_RULE = "generalize_rule"
    REORDER_STEPS = "reorder_steps"
    ADD_EXAMPLE = "add_example"
    DELETE_EXAMPLE = "delete_example"
    COMPRESS_SECTION = "compress_section"
    SPLIT_SECTION = "split_section"
    MERGE_SECTIONS = "merge_sections"
    ADD_FAILURE_MODE = "add_failure_mode"
    ADD_RECOVERY_RULE = "add_recovery_rule"
    CROSSOVER = "crossover"


class SplitRole(str, Enum):


    FEEDBACK_TRAIN = "feedback_train"
    OPTIMIZATION_MINIBATCH = "optimization_minibatch"
    VALIDATION_GATE = "validation_gate"
    FINAL_TEST = "final_test"


class CompilerMode(str, Enum):


    FULL = "full"
    SECTION = "section_selection"
    DISTILL = "distill"


class ArchiveTier(str, Enum):


    BEST = "best"
    PARETO = "pareto"
    DIVERSE = "diverse"
    FAILED = "failed"





@dataclass
class SkillMetadata:


    created_by: str = "optimizer"
    parent_id: Optional[str] = None
    edit_summary: str = ""
    token_count: int = 0
    lineage_depth: int = 0
    created_at_iteration: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillArtifact:
    """A skill artifact z = (s_1, ..., s_L) (Breakdown S1.4).

    Fork note: SkillOpt represents a skill as a single markdown *string* and its
    deterministic applicator edits that string by substring ops. Therefore
    ``document`` (the raw markdown) is the **source of truth** that round-trips
    losslessly through SkillOpt's ``apply_edit``/harness. ``sections`` is an
    optional parsed convenience view (populated by a :class:`SkillSerializer`)
    used for structural features and section-targeted edits; it must not be
    treated as authoritative over ``document``.
    """

    skill_id: str
    name: str
    document: str = ""
    version: int = 0
    sections: dict[SkillSection, Any] = field(default_factory=dict)
    metadata: SkillMetadata = field(default_factory=SkillMetadata)

    def section(self, key: SkillSection) -> Any:
        return self.sections.get(key)


# --------------------------------------------------------------------------- #
# Edits and candidates                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class EditProposal:
    """A single reflection-proposed edit e (Spec S11.2 / S17.3).

    The physical fields (``operation``/``content``/``target``) map directly onto
    SkillOpt's ``Edit`` (op/content/target) so a :class:`EditApplicator` can
    delegate to ``apply_edit`` with no information loss. The remaining fields are
    BESO metadata for featurization, provenance, and reflection grounding;
    validity nu(z, e) in {0,1} is checked separately by the lint/schema layer.
    """

    edit_id: str
    parent_skill_id: str
    operation: EditOperation
    content: str = ""
    target: str = ""
    # BESO semantic metadata (provenance + phi_edit features).
    category: Optional[EditCategory] = None
    target_section: Optional[SkillSection] = None
    source_type: Optional[str] = None  # "failure" | "success" (SkillOpt)
    rationale: str = ""
    expected_effect: str = ""
    risk: str = ""
    estimated_scope: str = ""
    edit_size_tokens: int = 0
    # For MERGE crossover the second parent is recorded here.
    secondary_parent_id: Optional[str] = None


@dataclass
class CandidateFeatures:
    """Structured candidate featurization phi(z) (Breakdown S6.2, Spec S12).

    Blocks are kept separate so the featurizer can standardize and weight each
    block independently (avoids the high-dimensional text block swamping the
    cheap structured signals). The flat vector is assembled downstream.
    """

    candidate_id: str
    # phi_text(z): embedding of changed text / delta (raw, pre-reduction).
    text_embedding: Optional[np.ndarray] = None
    # phi_struct(z): token counts, #rules, #examples, deltas vs parent, ...
    structural: dict[str, float] = field(default_factory=dict)
    # phi_edit(z): one-hot/categorical edit op x target section, edit size.
    edit: dict[str, float] = field(default_factory=dict)
    # phi_hist(z): parent score/variance, lineage depth, edit-type success rate.
    history: dict[str, float] = field(default_factory=dict)
    # phi_sem(z): LLM-labeled emphases (verification, tool-use, caution, ...).
    semantic: dict[str, float] = field(default_factory=dict)


@dataclass
class Candidate:
    """A proposed skill artifact under consideration (Breakdown S3.2).

    Wraps the child skill, the edit that produced it, its parent, and (lazily)
    its features and surrogate prediction.
    """

    candidate_id: str
    skill: SkillArtifact
    parent_id: Optional[str] = None
    edit: Optional[EditProposal] = None
    features: Optional[CandidateFeatures] = None
    prediction: Optional["SurrogatePrediction"] = None
    acquisition_score: Optional[float] = None





@dataclass
class Trajectory:





    example_id: str
    task_input: str
    compiled_prompt: str = ""
    output: str = ""
    score: float = 0.0
    feedback: str = ""
    error: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieval_context: list[str] = field(default_factory=list)
    cost_tokens: float = 0.0
    latency_seconds: float = 0.0
    valid_output: bool = True


@dataclass
class EvaluationResult:
    """Aggregate of a candidate evaluated on a minibatch B_t (Breakdown S2.3).

    ``per_example_scores`` r_i(z) supports paired bootstrap gating (S9.2) and
    the instance-level Pareto win matrix S[k, i] (S10.1).
    """

    candidate_id: str
    split: SplitRole
    per_example_scores: dict[str, float] = field(default_factory=dict)
    trajectories: list[Trajectory] = field(default_factory=list)
    invalid_rate: float = 0.0
    mean_cost_tokens: float = 0.0
    mean_latency_seconds: float = 0.0

    @property
    def mean_score(self) -> float:
        """Empirical utility hat_J_B(z) = (1/|B|) sum_i r_i(z)."""
        if not self.per_example_scores:
            return 0.0
        return float(np.mean(list(self.per_example_scores.values())))

    @property
    def n(self) -> int:
        return len(self.per_example_scores)


@dataclass
class Observation:
    """A single noisy minibatch observation, one row of history H_t (S4.1).

        tilde_y = hat_J_B(z) + eps,   eps ~ N(0, sigma_eps^2(z, B))

    Repeated observations of the same candidate_id are stored as separate rows;
    bar_y and SE are recovered by aggregation, not stored redundantly.
    """

    candidate_id: str
    batch_ids: tuple[str, ...]
    observed_score: float
    cost: float = 0.0
    iteration: int = 0
    split: SplitRole = SplitRole.OPTIMIZATION_MINIBATCH


# --------------------------------------------------------------------------- #
# Surrogate output and archive entries                                          #
# --------------------------------------------------------------------------- #
@dataclass
class SurrogatePrediction:
    """Posterior-predictive summary for a candidate (Breakdown S7.1).

    Decomposes total predictive variance into epistemic (model disagreement)
    and aleatoric (observation noise sigma_eps^2) components so acquisition can
    use a calibrated sigma_t(z) rather than raw ensemble disagreement.
    """

    candidate_id: str
    mu: float
    sigma: float
    epistemic_var: float = 0.0
    aleatoric_var: float = 0.0
    # Optional improvement-over-parent estimate Delta(z, z_p) (Breakdown S7.5).
    mu_delta: Optional[float] = None


@dataclass
class ArchiveEntry:
    """A stored, useful, or diagnostically important candidate (Spec S15.2)."""

    candidate_id: str
    parent_id: Optional[str]
    artifact: SkillArtifact
    tier: ArchiveTier = ArchiveTier.BEST
    optimization_mean: float = 0.0
    validation_mean: float = 0.0
    validation_se: float = 0.0
    format_score: float = 0.0
    cost_per_task: float = 0.0
    latency_seconds: float = 0.0
    invalid_rate: float = 0.0
    lineage_depth: int = 0
    winning_examples: list[str] = field(default_factory=list)
    pareto_win_count: int = 0
    known_strengths: list[str] = field(default_factory=list)
    known_weaknesses: list[str] = field(default_factory=list)
    accepted_edit_summary: str = ""
    created_at_iteration: int = 0


# --------------------------------------------------------------------------- #
# Budget bookkeeping                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class RolloutBudget:
    """Tracks the budget constraint sum_t c(z_t, B_t) <= B (Breakdown S0)."""

    max_rollouts: int
    spent_rollouts: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_rollouts - self.spent_rollouts)

    @property
    def exhausted(self) -> bool:
        return self.spent_rollouts >= self.max_rollouts

    def spend(self, n: int) -> None:
        self.spent_rollouts += n
````

## File: beso/edits/__init__.py
````python

````

## File: beso/evaluation/__init__.py
````python

````

## File: beso/experiments/__init__.py
````python

````

## File: beso/features/__init__.py
````python
from beso.features.featurizer import (
    FeatureExtractor,
    HashingEmbedder,
    approx_tokens,
    compute_structural_metrics,
)
from beso.features.normalization import FeatureNormalizer, NormalizerConfig

__all__ = [
    "FeatureExtractor",
    "HashingEmbedder",
    "approx_tokens",
    "compute_structural_metrics",
    "FeatureNormalizer",
    "NormalizerConfig",
]
````

## File: beso/features/featurizer.py
````python
from __future__ import annotations

import hashlib
import re
from typing import Callable, Optional, Sequence

import numpy as np

from beso.core.types import (
    Candidate,
    CandidateFeatures,
    EditCategory,
    EditOperation,
    Observation,
    SkillArtifact,
    SkillSection,
)

EmbedFn = Callable[[str], np.ndarray]

_HEADING_RE = re.compile(r"^\s{0,3}
_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```")
_SENTENCE_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"\S+")

# Structural metric keys, in fixed order, used for parent-relative deltas.
STRUCTURAL_METRICS: tuple[str, ...] = (
    "tokens",
    "chars",
    "lines",
    "headings",
    "bullets",
    "numbered",
    "rules",
    "code_blocks",
    "examples",
    "avg_sentence_tokens",
)


def approx_tokens(text: str) -> int:

    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def compute_structural_metrics(document: str) -> dict[str, float]:

    document = document or ""
    tokens = approx_tokens(document)
    headings = len(_HEADING_RE.findall(document))
    bullets = len(_BULLET_RE.findall(document))
    numbered = len(_NUMBERED_RE.findall(document))
    code_blocks = len(_CODE_FENCE_RE.findall(document)) // 2
    sentences = [s for s in _SENTENCE_RE.split(document) if s.strip()]
    avg_sentence_tokens = (
        float(np.mean([approx_tokens(s) for s in sentences])) if sentences else 0.0
    )
    return {
        "tokens": float(tokens),
        "chars": float(len(document)),
        "lines": float(document.count("\n") + 1 if document else 0),
        "headings": float(headings),
        "bullets": float(bullets),
        "numbered": float(numbered),
        "rules": float(bullets + numbered),
        "code_blocks": float(code_blocks),
        "examples": float(len(re.findall(r"(?i)\bexample\b", document))),
        "avg_sentence_tokens": avg_sentence_tokens,
    }


class HashingEmbedder:







    def __init__(self, dim: int = 128, ngram: int = 3) -> None:
        self.dim = int(dim)
        self.ngram = int(ngram)

    def __call__(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        text = (text or "").lower().strip()
        if not text:
            return vec
        padded = f"  {text}  "
        for i in range(len(padded) - self.ngram + 1):
            gram = padded[i : i + self.ngram]
            h = int.from_bytes(
                hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(),
                "little",
                signed=False,
            )
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class FeatureExtractor:


















    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = HashingEmbedder(),
        *,
        text_mode: str = "edit_content",
        semantic_labeler: Optional[Callable[[SkillArtifact], dict[str, float]]] = None,
        edit_success_rates: Optional[dict[str, float]] = None,
        cold_start_rate: float = 0.5,
    ) -> None:
        if text_mode not in ("edit_content", "document_delta"):
            raise ValueError(f"unknown text_mode: {text_mode!r}")
        self.embed_fn = embed_fn
        self.text_mode = text_mode
        self.semantic_labeler = semantic_labeler
        self.edit_success_rates = dict(edit_success_rates or {})
        self.cold_start_rate = float(cold_start_rate)

    # -- protocol entry point ------------------------------------------------ #
    def featurize(
        self,
        candidate: Candidate,
        parent: Optional[SkillArtifact],
        history: Sequence[Observation],
    ) -> CandidateFeatures:
        return CandidateFeatures(
            candidate_id=candidate.candidate_id,
            text_embedding=self._text_block(candidate, parent),
            structural=self._structural_block(candidate.skill, parent),
            edit=self._edit_block(candidate),
            history=self._history_block(candidate, parent, history),
            semantic=self._semantic_block(candidate.skill),
        )

    # -- Tier 1: structural deltas ------------------------------------------- #
    def _structural_block(
        self, child: SkillArtifact, parent: Optional[SkillArtifact]
    ) -> dict[str, float]:
        child_m = compute_structural_metrics(child.document)
        parent_m = (
            compute_structural_metrics(parent.document)
            if parent is not None
            else {k: 0.0 for k in STRUCTURAL_METRICS}
        )
        block: dict[str, float] = {}
        for k in STRUCTURAL_METRICS:
            c = child_m[k]
            p = parent_m[k]
            block[f"d_{k}"] = c - p  # parent-relative delta
            block[f"rel_{k}"] = (c - p) / (abs(p) + 1.0)  # scale-stable relative delta
        block["child_tokens"] = child_m["tokens"]
        return block


    def _edit_block(self, candidate: Candidate) -> dict[str, float]:
        block: dict[str, float] = {}
        for op in EditOperation:
            block[f"op_{op.value}"] = 0.0
        for cat in EditCategory:
            block[f"cat_{cat.value}"] = 0.0
        for sec in SkillSection:
            block[f"sec_{sec.value}"] = 0.0
        block["edit_size_tokens"] = 0.0
        block["content_tokens"] = 0.0
        block["target_tokens"] = 0.0
        block["has_target"] = 0.0
        block["src_failure"] = 0.0
        block["src_success"] = 0.0

        edit = candidate.edit
        if edit is None:
            return block

        block[f"op_{edit.operation.value}"] = 1.0
        if edit.category is not None:
            block[f"cat_{edit.category.value}"] = 1.0
        if edit.target_section is not None:
            block[f"sec_{edit.target_section.value}"] = 1.0
        content_tokens = approx_tokens(edit.content)
        block["content_tokens"] = float(content_tokens)
        block["target_tokens"] = float(approx_tokens(edit.target))
        block["edit_size_tokens"] = float(edit.edit_size_tokens or content_tokens)
        block["has_target"] = 1.0 if edit.target else 0.0
        if edit.source_type == "failure":
            block["src_failure"] = 1.0
        elif edit.source_type == "success":
            block["src_success"] = 1.0
        return block


    def _history_block(
        self,
        candidate: Candidate,
        parent: Optional[SkillArtifact],
        history: Sequence[Observation],
    ) -> dict[str, float]:
        parent_id = candidate.parent_id or (parent.skill_id if parent else None)
        parent_scores = [
            o.observed_score for o in history if o.candidate_id == parent_id
        ]
        n_obs = len(parent_scores)
        parent_mean = float(np.mean(parent_scores)) if n_obs else 0.0
        parent_std = float(np.std(parent_scores, ddof=0)) if n_obs > 1 else 0.0

        lineage_depth = float(candidate.skill.metadata.lineage_depth)
        if lineage_depth == 0.0 and parent is not None:
            lineage_depth = float(parent.metadata.lineage_depth + 1)

        op_value = candidate.edit.operation.value if candidate.edit else None
        edit_success = self.edit_success_rates.get(op_value, self.cold_start_rate)

        return {
            "parent_mean": parent_mean,
            "parent_std": parent_std,
            "parent_n_obs": float(n_obs),
            "parent_observed": 1.0 if n_obs else 0.0,
            "lineage_depth": lineage_depth,
            "parent_tokens": float(
                compute_structural_metrics(parent.document)["tokens"]
                if parent is not None
                else 0.0
            ),
            "edit_type_success_rate": float(edit_success),
        }


    def _semantic_block(self, child: SkillArtifact) -> dict[str, float]:
        if self.semantic_labeler is None:
            return {}
        labels = self.semantic_labeler(child) or {}
        return {str(k): float(v) for k, v in labels.items()}


    def _text_block(
        self, candidate: Candidate, parent: Optional[SkillArtifact]
    ) -> Optional[np.ndarray]:
        if self.embed_fn is None:
            return None
        if self.text_mode == "document_delta":
            child_emb = self.embed_fn(candidate.skill.document)
            parent_emb = (
                self.embed_fn(parent.document)
                if parent is not None
                else np.zeros_like(child_emb)
            )
            return np.asarray(child_emb, dtype=np.float64) - np.asarray(
                parent_emb, dtype=np.float64
            )

        changed = candidate.edit.content if candidate.edit else candidate.skill.document
        return np.asarray(self.embed_fn(changed), dtype=np.float64)


__all__ = [
    "EmbedFn",
    "STRUCTURAL_METRICS",
    "approx_tokens",
    "compute_structural_metrics",
    "HashingEmbedder",
    "FeatureExtractor",
]
````

## File: beso/features/normalization.py
````python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from beso.core.types import CandidateFeatures


_DICT_BLOCKS: tuple[str, ...] = ("structural", "edit", "history", "semantic")
_TEXT_BLOCK = "text"
_EPS = 1e-8


class _PCA:







    def __init__(self, n_components: int, whiten: bool = False) -> None:
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self.n_components_: int = 0

    def fit_transform(self, mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat, dtype=np.float64)
        n_samples, n_features = mat.shape
        k = max(1, min(self.n_components, n_features, n_samples))
        self.mean_ = mat.mean(axis=0)
        centered = mat - self.mean_
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[:k]
        self.n_components_ = k
        denom = max(n_samples - 1, 1)
        self._scale = (s[:k] / np.sqrt(denom))
        self._scale = np.where(self._scale < _EPS, 1.0, self._scale)
        return self.transform(mat)

    def transform(self, mat: np.ndarray) -> np.ndarray:
        centered = np.asarray(mat, dtype=np.float64) - self.mean_
        reduced = centered @ self.components_.T
        if self.whiten:
            reduced = reduced / self._scale
        return reduced


@dataclass
class NormalizerConfig:


    block_weights: dict[str, float] = field(
        default_factory=lambda: {
            "structural": 1.0,
            "edit": 1.0,
            "history": 1.0,
            "semantic": 1.0,
            "text": 1.0,
        }
    )
    text_pca_dims: int = 32
    standardize: bool = True
    balance_block_dims: bool = True
    whiten_pca: bool = False


class FeatureNormalizer:








    def __init__(self, config: Optional[NormalizerConfig] = None) -> None:
        self.config = config or NormalizerConfig()
        self._fitted = False

        self._keys: dict[str, list[str]] = {}
        self._mean: dict[str, np.ndarray] = {}
        self._std: dict[str, np.ndarray] = {}

        self._has_text = False
        self._pca = None
        self._text_mean: Optional[np.ndarray] = None
        self._text_std: Optional[np.ndarray] = None

        self._block_scale: dict[str, float] = {}
        self.feature_names_: list[str] = []


    def fit(self, corpus: Sequence[CandidateFeatures]) -> "FeatureNormalizer":
        if not corpus:
            raise ValueError("cannot fit FeatureNormalizer on an empty corpus")

        for block in _DICT_BLOCKS:
            keys = self._collect_keys(corpus, block)
            self._keys[block] = keys
            if not keys:
                self._mean[block] = np.zeros(0)
                self._std[block] = np.ones(0)
                continue
            mat = np.stack([self._dict_to_vec(getattr(f, block), keys) for f in corpus])
            self._mean[block] = mat.mean(axis=0)
            std = mat.std(axis=0, ddof=0)
            self._std[block] = np.where(std < _EPS, 1.0, std)
            standardized = (mat - self._mean[block]) / self._std[block]
            self._block_scale[block] = self._variance_scale(standardized)

        self._fit_text(corpus)
        self._build_feature_names()
        self._fitted = True
        return self

    def fit_transform(self, corpus: Sequence[CandidateFeatures]) -> np.ndarray:
        self.fit(corpus)
        return self.transform(corpus)

    def _fit_text(self, corpus: Sequence[CandidateFeatures]) -> None:
        embs = [
            np.asarray(f.text_embedding, dtype=np.float64)
            for f in corpus
            if f.text_embedding is not None
        ]
        self._has_text = len(embs) == len(corpus) and len(embs) > 0
        if not self._has_text:
            self._pca = None
            self._text_mean = None
            self._text_std = None
            return

        mat = np.stack(embs)
        self._pca = _PCA(
            n_components=self.config.text_pca_dims, whiten=self.config.whiten_pca
        )
        reduced = self._pca.fit_transform(mat)
        self._text_mean = reduced.mean(axis=0)
        std = reduced.std(axis=0, ddof=0)
        self._text_std = np.where(std < _EPS, 1.0, std)
        standardized = (reduced - self._text_mean) / self._text_std
        self._block_scale[_TEXT_BLOCK] = self._variance_scale(standardized)


    def transform(self, features) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer.transform called before fit")
        single = isinstance(features, CandidateFeatures)
        items: Sequence[CandidateFeatures] = [features] if single else features
        rows = [self._transform_one(f) for f in items]
        out = np.stack(rows) if rows else np.zeros((0, len(self.feature_names_)))
        return out[0] if single else out

    def _transform_one(self, f: CandidateFeatures) -> np.ndarray:
        parts: list[np.ndarray] = []
        for block in _DICT_BLOCKS:
            keys = self._keys.get(block, [])
            if not keys:
                continue
            vec = self._dict_to_vec(getattr(f, block), keys)
            vec = self._standardize(vec, self._mean[block], self._std[block])
            parts.append(self._weight(vec, block))

        if self._has_text:
            parts.append(self._transform_text(f))

        if not parts:
            return np.zeros(len(self.feature_names_), dtype=np.float64)
        return np.concatenate(parts)

    def _transform_text(self, f: CandidateFeatures) -> np.ndarray:
        dim = self._pca.n_components_
        if f.text_embedding is None:
            return np.zeros(dim, dtype=np.float64)
        emb = np.asarray(f.text_embedding, dtype=np.float64).reshape(1, -1)
        reduced = self._pca.transform(emb)[0]
        reduced = self._standardize(reduced, self._text_mean, self._text_std)
        return self._weight(reduced, _TEXT_BLOCK)


    def _standardize(
        self, vec: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        if not self.config.standardize:
            return vec
        return (vec - mean) / std

    def _weight(self, vec: np.ndarray, block: str) -> np.ndarray:
        w = float(self.config.block_weights.get(block, 1.0))
        if self.config.balance_block_dims:
            w = w * self._block_scale.get(block, 1.0)
        return vec * w

    @staticmethod
    def _variance_scale(standardized: np.ndarray) -> float:

        total_var = float(np.sum(np.var(standardized, axis=0, ddof=0)))
        return 1.0 / np.sqrt(total_var) if total_var > _EPS else 1.0

    @staticmethod
    def _collect_keys(corpus: Sequence[CandidateFeatures], block: str) -> list[str]:
        keys: set[str] = set()
        for f in corpus:
            keys.update(getattr(f, block).keys())
        return sorted(keys)

    @staticmethod
    def _dict_to_vec(d: dict[str, float], keys: list[str]) -> np.ndarray:
        return np.array([float(d.get(k, 0.0)) for k in keys], dtype=np.float64)

    def _build_feature_names(self) -> None:
        names: list[str] = []
        for block in _DICT_BLOCKS:
            names.extend(f"{block}::{k}" for k in self._keys.get(block, []))
        if self._has_text:
            names.extend(f"text::pca_{i}" for i in range(self._pca.n_components_))
        self.feature_names_ = names

    # -- introspection ------------------------------------------------------- #
    def block_slices(self) -> dict[str, slice]:
        """Return the column slice occupied by each block in the output vector."""
        slices: dict[str, slice] = {}
        start = 0
        for block in _DICT_BLOCKS:
            n = len(self._keys.get(block, []))
            if n:
                slices[block] = slice(start, start + n)
                start += n
        if self._has_text:
            n = self._pca.n_components_
            slices[_TEXT_BLOCK] = slice(start, start + n)
        return slices

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)


__all__ = ["NormalizerConfig", "FeatureNormalizer"]
````

## File: beso/llm/__init__.py
````python

````

## File: beso/optimization/__init__.py
````python

````

## File: beso/reflection/__init__.py
````python

````

## File: beso/store/__init__.py
````python

````

## File: beso/surrogate/__init__.py
````python
from beso.surrogate.base import BaseSurrogate
from beso.surrogate.calibration import (
    Calibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    TemperatureScaler,
)
from beso.surrogate.ensemble import BaggingEnsembleSurrogate, RidgeRegressor

__all__ = [
    "BaseSurrogate",
    "BaggingEnsembleSurrogate",
    "RidgeRegressor",
    "Calibrator",
    "IdentityCalibrator",
    "TemperatureScaler",
    "IsotonicCalibrator",
]
````

## File: beso/surrogate/base.py
````python
from __future__ import annotations

import abc
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np

from beso.core.types import CandidateFeatures, Observation, SurrogatePrediction
from beso.features.normalization import FeatureNormalizer
from beso.surrogate.calibration import Calibrator, IdentityCalibrator, TemperatureScaler

PARENT_MEAN_KEY = "parent_mean"
_EPS = 1e-12


class BaseSurrogate(abc.ABC):


    def __init__(
        self,
        *,
        normalizer: Optional[FeatureNormalizer] = None,
        calibrator: Optional[Calibrator] = None,
        min_obs_for_calibration: int = 8,
        aleatoric_floor: float = 1e-6,
    ) -> None:
        self.normalizer = normalizer if normalizer is not None else FeatureNormalizer()
        self.calibrator = calibrator if calibrator is not None else TemperatureScaler()
        self.min_obs_for_calibration = int(min_obs_for_calibration)
        self.aleatoric_floor = float(aleatoric_floor)
        self._fitted = False
        self._calibrated = False
        self._aleatoric_var = 0.0
        self._n_train = 0


    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        features: Sequence[CandidateFeatures],
        observations: Sequence[Observation],
    ) -> None:
        if not features:
            raise ValueError("cannot fit surrogate with no candidate features")
        feat_by_id = {f.candidate_id: f for f in features}

        rows: list[CandidateFeatures] = []
        targets: list[float] = []
        for obs in observations:
            f = feat_by_id.get(obs.candidate_id)
            if f is None:
                continue
            parent_mean = float(f.history.get(PARENT_MEAN_KEY, 0.0))
            rows.append(f)
            targets.append(float(obs.observed_score) - parent_mean)
        if not rows:
            raise ValueError(
                "no observations matched the provided candidate features; "
                "cannot build a training set"
            )


        self.normalizer.fit(features)
        X = np.atleast_2d(self.normalizer.transform(rows))
        y = np.asarray(targets, dtype=np.float64)
        self._n_train = X.shape[0]

        self._fit_core(X, y)
        oob_mu, oob_epi = self._oob_predict(X)

        self._aleatoric_var = self._estimate_aleatoric(observations, y, oob_mu, oob_epi)
        self._fit_calibration(y, oob_mu, oob_epi)
        self._fitted = True

    def predict(self, features: CandidateFeatures) -> SurrogatePrediction:
        return self._predict_batch([features])[0]

    def predict_many(
        self, features: Sequence[CandidateFeatures]
    ) -> list[SurrogatePrediction]:
        feats = list(features)
        return self._predict_batch(feats) if feats else []


    def _predict_batch(
        self, feats: Sequence[CandidateFeatures]
    ) -> list[SurrogatePrediction]:
        if not self._fitted:
            raise RuntimeError("surrogate.predict called before fit")
        X = np.atleast_2d(self.normalizer.transform(list(feats)))
        mu_delta, epi = self._predict_core(X)
        aleatoric = float(self._aleatoric_var)
        preds: list[SurrogatePrediction] = []
        for i, f in enumerate(feats):
            parent_mean = float(f.history.get(PARENT_MEAN_KEY, 0.0))
            epistemic = max(float(epi[i]), 0.0)
            total_var = epistemic + aleatoric
            total_sigma = float(np.sqrt(total_var)) if total_var > 0 else 0.0
            cal_sigma = (
                float(self.calibrator.transform(total_sigma))
                if total_sigma > 0
                else 0.0
            )
            ratio = (cal_sigma / total_sigma) ** 2 if total_sigma > _EPS else 1.0
            d = float(mu_delta[i])
            preds.append(
                SurrogatePrediction(
                    candidate_id=f.candidate_id,
                    mu=parent_mean + d,
                    sigma=cal_sigma,
                    epistemic_var=epistemic * ratio,
                    aleatoric_var=aleatoric * ratio,
                    mu_delta=d,
                )
            )
        return preds

    def _estimate_aleatoric(
        self,
        observations: Sequence[Observation],
        y: np.ndarray,
        oob_mu: np.ndarray,
        oob_epi: np.ndarray,
    ) -> float:






        by_cand: dict[str, list[float]] = defaultdict(list)
        for obs in observations:
            by_cand[obs.candidate_id].append(float(obs.observed_score))
        within = [
            float(np.var(v, ddof=1)) for v in by_cand.values() if len(v) >= 2
        ]
        if within:
            return max(float(np.mean(within)), self.aleatoric_floor)

        resid_var = float(np.var(y - oob_mu)) if y.size else 0.0
        mean_epi = float(np.mean(oob_epi)) if oob_epi.size else 0.0
        return max(resid_var - mean_epi, self.aleatoric_floor)

    def _fit_calibration(
        self, y: np.ndarray, oob_mu: np.ndarray, oob_epi: np.ndarray
    ) -> None:
        residuals = y - oob_mu
        total_var = np.maximum(oob_epi + self._aleatoric_var, 0.0)
        sigmas = np.sqrt(total_var)
        enough = residuals.size >= self.min_obs_for_calibration
        if enough and not isinstance(self.calibrator, IdentityCalibrator):
            self.calibrator.fit(residuals, sigmas)
            self._calibrated = True
        else:
            self.calibrator = IdentityCalibrator()
            self.calibrator.fit(residuals, sigmas)
            self._calibrated = False


    @abc.abstractmethod
    def _fit_core(self, X: np.ndarray, y: np.ndarray) -> None:


    @abc.abstractmethod
    def _predict_core(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:


    @abc.abstractmethod
    def _oob_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:



__all__ = ["BaseSurrogate", "PARENT_MEAN_KEY"]
````

## File: beso/surrogate/calibration.py
````python
from __future__ import annotations

import abc

import numpy as np

_EPS = 1e-9

_HALF_NORMAL_MEAN = np.sqrt(2.0 / np.pi)


class Calibrator(abc.ABC):


    is_fitted: bool = False

    @abc.abstractmethod
    def fit(self, residuals: np.ndarray, sigmas: np.ndarray) -> "Calibrator":


    @abc.abstractmethod
    def transform(self, sigma):



class IdentityCalibrator(Calibrator):


    def __init__(self) -> None:
        self.is_fitted = True

    def fit(self, residuals: np.ndarray, sigmas: np.ndarray) -> "IdentityCalibrator":
        self.is_fitted = True
        return self

    def transform(self, sigma):
        return sigma


class TemperatureScaler(Calibrator):







    def __init__(self, floor: float = _EPS) -> None:
        self.floor = float(floor)
        self.scale = 1.0
        self.is_fitted = False

    def fit(self, residuals: np.ndarray, sigmas: np.ndarray) -> "TemperatureScaler":
        r = np.asarray(residuals, dtype=np.float64).ravel()
        s = np.asarray(sigmas, dtype=np.float64).ravel()
        s = np.where(s < self.floor, self.floor, s)
        if r.size == 0:
            self.scale = 1.0
        else:
            z2 = (r / s) ** 2
            val = float(np.sqrt(np.mean(z2)))
            self.scale = val if np.isfinite(val) and val > 0 else 1.0
        self.is_fitted = True
        return self

    def transform(self, sigma):
        scaled = np.asarray(sigma, dtype=np.float64) * self.scale
        return float(scaled) if np.ndim(sigma) == 0 else scaled


class IsotonicCalibrator(Calibrator):








    def __init__(self, min_points: int = 8, floor: float = _EPS) -> None:
        self.min_points = int(min_points)
        self.floor = float(floor)
        self._x: np.ndarray | None = None
        self._g: np.ndarray | None = None
        self._fallback = TemperatureScaler(floor=floor)
        self.is_fitted = False

    def fit(self, residuals: np.ndarray, sigmas: np.ndarray) -> "IsotonicCalibrator":
        r = np.abs(np.asarray(residuals, dtype=np.float64).ravel())
        s = np.asarray(sigmas, dtype=np.float64).ravel()
        s = np.where(s < self.floor, self.floor, s)
        self._fallback.fit(residuals, sigmas)
        if r.size < self.min_points:
            self._x = None
            self._g = None
            self.is_fitted = True
            return self
        order = np.argsort(s)
        xs = s[order]
        ys = r[order]
        self._x = xs
        self._g = _pav_nondecreasing(ys)
        self.is_fitted = True
        return self

    def transform(self, sigma):
        scalar = np.ndim(sigma) == 0
        s = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
        if self._x is None or self._g is None:
            out = self._fallback.transform(s)
        else:
            g = np.interp(s, self._x, self._g, left=self._g[0], right=self._g[-1])
            out = np.maximum(g / _HALF_NORMAL_MEAN, self.floor)
        out = np.asarray(out, dtype=np.float64)
        return float(out[0]) if scalar else out


def _pav_nondecreasing(y: np.ndarray) -> np.ndarray:

    y = np.asarray(y, dtype=np.float64).copy()
    n = y.size
    if n == 0:
        return y
    values = y.copy()
    weights = np.ones(n)
    idx = list(range(n + 1))

    level_values: list[float] = []
    level_weights: list[float] = []
    level_counts: list[int] = []
    for i in range(n):
        v = values[i]
        w = weights[i]
        c = 1
        while level_values and level_values[-1] >= v:
            pv = level_values.pop()
            pw = level_weights.pop()
            pc = level_counts.pop()
            v = (v * w + pv * pw) / (w + pw)
            w = w + pw
            c = c + pc
        level_values.append(v)
        level_weights.append(w)
        level_counts.append(c)
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for v, c in zip(level_values, level_counts):
        out[pos : pos + c] = v
        pos += c
    return out


__all__ = [
    "Calibrator",
    "IdentityCalibrator",
    "TemperatureScaler",
    "IsotonicCalibrator",
]
````

## File: beso/surrogate/ensemble.py
````python
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from beso.surrogate.base import BaseSurrogate

_EPS = 1e-12


class RidgeRegressor:


    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        A = np.hstack([X, np.ones((n, 1))])
        reg = self.alpha * np.eye(d + 1)
        reg[-1, -1] = 0.0
        ata = A.T @ A + reg
        aty = A.T @ y
        try:
            coef = np.linalg.solve(ata, aty)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.w = coef[:-1]
        self.b = float(coef[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.w + self.b


BaseFactory = Callable[[], object]


class BaggingEnsembleSurrogate(BaseSurrogate):


    def __init__(
        self,
        *,
        n_members: int = 8,
        alpha: float = 1.0,
        base_factory: Optional[BaseFactory] = None,
        feature_subsample: float = 0.8,
        bootstrap: bool = True,
        random_state: int = 0,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        self.n_members = int(n_members)
        self.alpha = float(alpha)
        self.base_factory: BaseFactory = base_factory or (
            lambda: RidgeRegressor(alpha=self.alpha)
        )
        self.feature_subsample = float(feature_subsample)
        self.bootstrap = bool(bootstrap)
        self.random_state = int(random_state)

        self._members: list[tuple[object, np.ndarray, set[int]]] = []
        self._n_features = 0

    def _fit_core(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        self._n_features = d
        self._members = []
        k = max(1, int(round(self.feature_subsample * d)))
        for _ in range(self.n_members):
            idx = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            feat = np.sort(rng.choice(d, size=k, replace=False))
            learner = self.base_factory()
            learner.fit(X[idx][:, feat], y[idx])
            self._members.append((learner, feat, set(int(i) for i in idx)))

    def _predict_core(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = self._member_predictions(X)
        mu = preds.mean(axis=0)
        if preds.shape[0] > 1:
            epi = preds.var(axis=0, ddof=1)
        else:
            epi = np.zeros(X.shape[0])
        return mu, epi

    def _oob_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        m = len(self._members)
        preds = np.full((m, n), np.nan)
        for mi, (learner, feat, in_bag) in enumerate(self._members):
            oob_mask = np.array([i not in in_bag for i in range(n)])
            if oob_mask.any():
                preds[mi, oob_mask] = learner.predict(X[oob_mask][:, feat])

        mu = np.empty(n)
        epi = np.empty(n)
        full = None
        for i in range(n):
            col = preds[:, i]
            valid = col[~np.isnan(col)]
            if valid.size >= 2:
                mu[i] = valid.mean()
                epi[i] = valid.var(ddof=1)
            elif valid.size == 1:
                mu[i] = valid[0]
                epi[i] = 0.0
            else:
                if full is None:
                    full = self._member_predictions(X)
                col_full = full[:, i]
                mu[i] = col_full.mean()
                epi[i] = col_full.var(ddof=1) if col_full.size > 1 else 0.0
        return mu, epi

    def _member_predictions(self, X: np.ndarray) -> np.ndarray:
        return np.stack(
            [learner.predict(X[:, feat]) for learner, feat, _ in self._members],
            axis=0,
        )


__all__ = ["BaggingEnsembleSurrogate", "RidgeRegressor"]
````

## File: beso/trajectories/__init__.py
````python

````

## File: configs/default.yaml
````yaml
experiment:
  name: beso_v0
  seed: 42

artifact:
  type: skill
  max_tokens: 900
  compiler_mode: full

optimization:
  max_rollouts: 300
  max_iterations: 30
  batch_size: 2
  candidate_pool_size: 24
  archive_size: 32
  min_improvement_delta: 0.01

edits:
  max_added_tokens_per_iteration: 120
  max_deleted_tokens_per_iteration: 80
  max_replaced_tokens_per_iteration: 160
  max_sections_modified_per_iteration: 2
  allowed_operations:
    - append
    - insert_after
    - replace
    - delete

surrogate:
  type: ensemble
  target: delta
  ensemble_members: 8
  uncertainty: epistemic_plus_aleatoric
  calibration: isotonic
  cold_start_iterations: 5

features:
  parent_centered: true
  per_block_standardize: true
  text_embedding_dims: 32

acquisition:
  type: pool_normalized_ucb
  normalize_terms: true
  kappa: 1.5
  diversity_lambda: 0.2
  cost_alpha: 0.1
  invalid_gamma: 0.1
  batch_selection: max_min

gate:
  test: paired_bootstrap
  multiplicity_correction: benjamini_hochberg
  noise_scaled_delta_c: 1.0
  constraint_confidence_bound: true
  revalidate_incumbent_every: 5

archive:
  strategy: pareto_and_diversity
  top_by_validation: 8
  top_by_pareto: 8
  top_by_diversity: 8
  top_failed_informative: 8

regime_detector:
  enabled: true
  min_candidate_variance: 1.0e-4
  min_rank_correlation: 0.1

reflection:
  model: optimizer_model
  require_trace_grounding: true
  include_rejected_edits: true

evaluation:
  target_model: target_model
  decoding_temperature: 0.0
  score_field: hard
  repeated_eval_for_top_candidates: true

splits:
  feedback_train: 0.40
  optimization_minibatch: 0.20
  validation_gate: 0.20
  final_test: 0.20
````

## File: docs/Bayesian Evolutionary Skill Optimization (BESO) - GEPA SkillOpt BESO Mathematical Lineage.md
````markdown
---
type: research-note
tags: [beso, gepa, skillopt, bayesian-optimization, evolutionary-search, skill-optimization, llm-agents]
date: 2026-05-28
source:
  - [[Bayesian Evolutionary Skill Optimization (BESO) - Technical Specification]]
  - [[Bayesian Evolutionary Skill Optimization (BESO) - Mathematical Breakdown]]
external_sources:
  - https://gepa-ai.github.io/gepa/
  - https://arxiv.org/abs/2507.19457
  - https://microsoft.github.io/SkillOpt/
  - https://arxiv.org/html/2605.23904v2
status: draft
---

# GEPA → SkillOpt → BESO: mathematical lineage and contribution map

## 0. Executive summary

This note formalizes the line from GEPA to SkillOpt to BESO.

The short version:

```text
GEPA:
  Optimize textual parameters, usually prompts, using reflective mutation and Pareto-aware evolutionary search.

SkillOpt:
  Move the optimization target from raw prompts to compact reusable skill documents. Add controlled bounded edits, validation gates, rejected-edit feedback, and slow/meta updates.

BESO:
  Keep SkillOpt's skill-level abstraction, but add Bayesian experiment planning: learn a probabilistic surrogate over candidate skill edits and use acquisition functions to decide which candidates deserve expensive rollout evaluation.
```

Mathematically, all three methods optimize external text for a frozen LLM system:

$$
\max_{z \in \mathcal{Z}} J(z)
= \max_{z \in \mathcal{Z} }
\mathbb{E}_{(x,m)\sim\mathcal{T}}
\left[\mu\left(\Phi(x; z, \Theta_{\mathrm{frozen}}),m\right)\right]
$$

The difference is what $z$ means and how the next $z$ is chosen.

| Method   | Optimized object $z$                          | Candidate generation                                      | Candidate selection                            | Main contribution                                                                    |
| -------- | --------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------ |
| GEPA     | Prompt / textual parameter / system component | LLM reflection over trajectories + mutation / merge       | Pareto-aware evolutionary selection            | Language reflection as a rich learning signal; Pareto search over textual candidates |
| SkillOpt | Skill document as external trainable state    | Optimizer model proposes bounded add/delete/replace edits | Held-out validation gate; strict accept/reject | Train reusable skills like model parameters, but in text space                       |
| BESO     | Structured skill artifact                     | Reflection-generated bounded skill edits                  | Bayesian surrogate + acquisition + archive     | Sample-efficient experiment planning for skill evolution under low rollout budgets   |

BESO is not mainly a new reflection mechanism. It is a new selection and budget-allocation layer over SkillOpt-style candidate skill edits.

The clean contribution claim is:

> BESO extends GEPA-style reflective text evolution and SkillOpt-style skill-document training by adding Bayesian candidate selection: a probabilistic surrogate predicts candidate utility and uncertainty, then an acquisition function chooses which skill variants to evaluate under a limited rollout budget.

---

## 1. Shared mathematical foundation

### 1.1 Frozen LLM system

Let the frozen LLM system be:

$$
\Phi_{\Theta}: \mathcal{X}\times\mathcal{A}\to\mathcal{Y}\times\mathcal{R}
$$

where:

- $\mathcal{X}$ is the task-input space.
- $\mathcal{A}$ is the external artifact space: prompt, textual parameter, or skill.
- $\mathcal{Y}$ is the final-output space.
- $\mathcal{R}$ is the trajectory / trace space.
- $\Theta=\Theta_{\mathrm{frozen}}$ are fixed model weights.

For task input $x$ and artifact $z$:

$$
(y,\tau)=\Phi_{\Theta_{\mathrm{frozen}}}(x;z)
$$

where:

- $y$ is the final answer or action result.
- $\tau$ is the observable execution trace: prompts, messages, tool calls, tool results, verifier output, error messages, intermediate outputs, cost, and latency.

Important: $\tau$ should be treated as observable trace, not hidden chain-of-thought.

### 1.2 Task distribution and evaluator

Let task instances be:

$$
\xi=(x,m)\sim\mathcal{T}
$$

where:

- $x$ is the task input.
- $m$ is metadata, expected answer, rubric, verifier, test suite, or scoring reference.
- $\mathcal{T}$ is the task distribution.

The evaluator is:

$$
\mu: \mathcal{Y}\times\mathcal{R}\times\mathcal{M}\to\mathbb{R}
$$

A single-task score is:

$$
r(z; x,m)=\mu(y,\tau,m)
$$

with:

$$
(y,\tau)=\Phi_{\Theta_{\mathrm{frozen}}}(x;z)
$$

Most papers use scores in $[0,1]$, but the framework can support arbitrary bounded reward.

### 1.3 Common optimization objective

The true objective is:

$$
J(z)=\mathbb{E}_{(x,m)\sim\mathcal{T}}[r(z;x,m)]
$$

The optimizer wants:

$$
z^*\in\arg\max_{z\in\mathcal{Z}}J(z)
$$

Because $\mathcal{T}$ is unknown, use a dataset:

$$
D=\{(x_i,m_i)\}_{i=1}^{n}
$$

Empirical score:

$$
\hat{J}_D(z)=\frac{1}{n}\sum_{i=1}^{n}r(z;x_i,m_i)
$$

In practice, all three methods face the same black-box setting:

$$
z \mapsto \hat{J}_D(z)
$$

is expensive, noisy, nondifferentiable, and defined over text.

---

## 2. GEPA: reflective prompt/text evolution

### 2.1 Optimization target

In GEPA, the optimized artifact is a textual candidate:

$$
p\in\mathcal{P}
$$

The candidate may be a prompt, a module instruction, a DSPy prompt, a textual system component, or in broader GEPA usage, another textual artifact.

For the canonical prompt-optimization case:

$$
y=\Phi(x;p,\Theta_{\mathrm{frozen}})
$$

Objective:

$$
p^*\in\arg\max_{p\in\mathcal{P}}
\mathbb{E}_{(x,m)\sim\mathcal{T}}
\left[\mu\left(\Phi(x;p,\Theta_{\mathrm{frozen}}),m\right)\right]
$$

### 2.2 Trajectory-conditioned reflection

GEPA's key move is that it does not reduce the rollout to only a scalar reward. It uses trajectory and side information.

A rollout produces:

$$
(y_i,\tau_i,s_i,f_i)
$$

where:

- $y_i$ is output,
- $\tau_i$ is trace,
- $s_i=\mu(y_i,\tau_i,m_i)$ is score,
- $f_i$ is actionable side information: errors, judge feedback, profiling data, test failures, reasoning summaries, etc.

A reflection operator proposes an improved candidate:

$$
p' = R_{\psi}(p, \{(x_i,m_i,y_i,\tau_i,s_i,f_i)\}_{i\in B})
$$

where $B$ is a minibatch and $R_{\psi}$ is usually an optimizer/reflection LLM.

The conceptual analogy is:

```text
scalar reward: tells the optimizer that the candidate failed
trajectory + side information: helps the optimizer infer why it failed
reflection: turns the why into a text update
```

### 2.3 Evolutionary population

GEPA maintains a population of candidates:

$$
P_t=\{p_{t,1},p_{t,2},\dots,p_{t,K}\}
$$

Each candidate is evaluated on examples, producing a score matrix:

$$
S_t[k,i]=r(p_{t,k};x_i,m_i)
$$

The population is updated through reflective mutation and sometimes merge / recombination:

$$
p_{\mathrm{mut}}=R_{\psi}(p_{\mathrm{parent}},\mathcal{D}_{\mathrm{trace}})
$$

$$
p_{\mathrm{merge}}=M_{\psi}(p_a,p_b,\mathcal{D}_{\mathrm{trace}})
$$

### 2.4 Pareto-aware candidate preservation

A central GEPA idea is that the best average candidate is not always the best search parent.

For each example $i$, define the best score achieved by any candidate:

$$
s_i^*=\max_k S_t[k,i]
$$

Winner set:

$$
W_i=\{p_{t,k}:S_t[k,i]=s_i^*\}
$$

Candidate win count:

$$
w(p_{t,k})=\sum_i\mathbb{1}[p_{t,k}\in W_i]
$$

Candidates with nonzero win counts may represent useful specialists. Selection can favor them even if they are not the global average winner.

A simplified parent-selection probability is:

$$
\Pr(p_{t,k})=\frac{w(p_{t,k})+\epsilon}{\sum_j(w(p_{t,j})+\epsilon)}
$$

This preserves diversity across task instances.

### 2.5 GEPA's core mathematical identity

GEPA can be summarized as:

$$
\boxed{
\text{GEPA} = \text{ReflectiveMutation}(\tau,f) + \text{ParetoEvolution}(P_t)
}
$$

More explicitly:

$$
\mathcal{C}_t=\{R_{\psi}(p,\mathcal{D}_p):p\sim\pi_{\mathrm{pareto}}(P_t)\}
$$

$$
P_{t+1}=\operatorname{ParetoUpdate}(P_t\cup\mathcal{C}_t)
$$

where candidates are evaluated by rollout calls.

### 2.6 GEPA limitation that motivates the next abstraction

GEPA is strong because it learns from rich traces. But prompt-level candidates can be brittle:

- A prompt mixes role, procedure, examples, output format, safety rules, tool policy, and failure handling into one text block.
- Mutations may improve one part while accidentally damaging another.
- Learned behavior may be less reusable if it is embedded in one prompt string.
- Candidate selection is mainly evolutionary / Pareto, not explicitly probabilistic experiment planning.

SkillOpt addresses the first three issues by changing the artifact. BESO addresses the fourth by changing the candidate-selection layer.

---

## 3. SkillOpt: train skills as external text-state

### 3.1 Shift in optimization target

SkillOpt moves from prompt optimization to skill-document optimization.

Instead of:

$$
p\in\mathcal{P}
$$

SkillOpt optimizes:

$$
s\in\mathcal{S}
$$

where $s$ is a compact natural-language skill document, often deployed as `best_skill.md`.

A skill can encode:

- procedure,
- tool-use policy,
- evidence-gathering rules,
- verification routines,
- output-format constraints,
- known failure modes,
- recovery rules.

The frozen agent executes with skill $s$:

$$
(\tau(s),r(s))=h(M,x,s),\qquad r(s)\in[0,1]
$$

using SkillOpt paper notation:

- $M$ is the frozen target model,
- $h$ is the execution harness,
- $x$ is the task,
- $s$ is the skill,
- $\tau(s)$ is the trajectory,
- $r(s)$ is the score.

The expected objective is:

$$
s^*\in\arg\max_{s\in\mathcal{S}}J(s)
=\arg\max_{s\in\mathcal{S}}\mathbb{E}_{x\sim\mathcal{T}}[r(s;x)]
$$

### 3.2 Skill as external trainable state

SkillOpt's key abstraction is:

$$
\text{trainable state} = s
$$

while:

$$
\Theta_{\mathrm{target}} \text{ is fixed}
$$

The target model, backend, and harness do not change. Only the skill document changes.

This means the deployment object is not a new model and not an online optimizer. It is a static skill artifact:

$$
s_{\mathrm{deploy}}=\texttt{best\_skill.md}
$$

with no extra inference-time model calls.

### 3.3 Bounded edit operators

SkillOpt uses structured edits:

$$
e\in\mathcal{E}=\{\mathrm{add},\mathrm{delete},\mathrm{replace}\}
$$

An edit transforms a skill:

$$
s'=e(s)
$$

A bounded edit budget acts like a textual learning rate.

Let edit distance or edit size be:

$$
\Delta(s,s')
$$

SkillOpt constrains updates:

$$
\Delta(s,s')\le \eta_{\mathrm{text}}
$$

where $\eta_{\mathrm{text}}$ is the textual learning-rate budget.

This prevents the optimizer from performing destructive broad rewrites.

### 3.4 Reflection as text-space backward pass

SkillOpt's loop can be written:

1. Rollout with current skill:

$$
\{(\tau_i,r_i)\}_{i\in B}=\operatorname{Rollout}(M,h,s_t,B)
$$

2. Reflect over successes and failures:

$$
\mathcal{E}_t=R_{\psi}(s_t,\{(\tau_i,r_i)\}_{i\in B},R_t)
$$

where $R_t$ may include rejected edits or optimizer memory.

3. Merge/rank bounded edits into a candidate:

$$
s_t'=\operatorname{ApplyBounded}(s_t,\mathcal{E}_t,\eta_{\mathrm{text}})
$$

4. Gate on held-out validation:

$$
s_{t+1}=\begin{cases}
s_t' & \text{if } \hat{J}_{\mathrm{sel}}(s_t')>\hat{J}_{\mathrm{sel}}(s_t) \\
s_t & \text{otherwise}
\end{cases}
$$

where $D_{\mathrm{sel}}$ is the held-out selection / validation split.

This is the mathematical heart of SkillOpt.

### 3.5 Validation gate

SkillOpt is not just self-revision. It is propose-and-test.

Acceptance rule:

$$
\operatorname{Accept}(s_t')=1
\iff
\hat{J}_{D_{\mathrm{sel}}}(s_t')>\hat{J}_{D_{\mathrm{sel}}}(s_t)
$$

A stricter rule can include threshold $\delta$:

$$
\hat{J}_{D_{\mathrm{sel}}}(s_t')>\hat{J}_{D_{\mathrm{sel}}}(s_t)+\delta
$$

Rejected candidates are stored:

$$
\mathcal{R}_{t+1}=\mathcal{R}_t\cup\{(s_t',\mathrm{reason}) : \operatorname{Accept}(s_t')=0\}
$$

The rejected-edit buffer gives negative feedback to future reflection.

### 3.6 Slow/meta update

SkillOpt also uses longer-horizon optimizer-side memory, described as slow update or meta skill.

We can represent optimizer memory as:

$$
m_t\in\mathcal{M}_{\mathrm{opt}}
$$

Reflection becomes:

$$
\mathcal{E}_t=R_{\psi}(s_t,\mathcal{B}_t,\mathcal{R}_t,m_t)
$$

and memory updates epoch-wise:

$$
m_{e+1}=U_{\psi}(m_e,\mathcal{H}_e)
$$

where $\mathcal{H}_e$ is the history from epoch $e$.

The deployed skill remains compact; optimizer memory does not need to be included at inference time.

### 3.7 SkillOpt's core mathematical identity

SkillOpt can be summarized as:

$$
\boxed{
\text{SkillOpt} = \text{SkillArtifact}(s) + \text{BoundedTextEdits}(\eta_{\mathrm{text}}) + \text{ValidationGate} + \text{RejectedEditMemory}
}
$$

Algorithmically:

$$
s_t' = \operatorname{ApplyBounded}\left(s_t, R_{\psi}(s_t,\mathcal{B}_t,\mathcal{R}_t,m_t), \eta_{\mathrm{text}}\right)
$$

$$
s_{t+1}=\begin{cases}
s_t' & \hat{J}_{\mathrm{sel}}(s_t')>\hat{J}_{\mathrm{sel}}(s_t) \\
s_t & \text{otherwise}
\end{cases}
$$

### 3.8 What SkillOpt contributes over GEPA

SkillOpt inherits the GEPA insight:

$$
\text{trajectory feedback} \rightarrow \text{natural-language reflection} \rightarrow \text{text update}
$$

But it changes the abstraction level:

$$
\text{prompt candidate }p
\quad\Longrightarrow\quad
\text{skill document }s
$$

This matters because skill documents are:

- reusable across tasks,
- deployable as standalone artifacts,
- more modular than prompts,
- easier to edit with add/delete/replace operations,
- easier to gate by validation,
- often transferable across models and harnesses.

So the GEPA → SkillOpt link is:

```text
GEPA learns textual candidates by reflection.
SkillOpt applies that idea to skills as the trainable text-state and adds training discipline.
```

---

## 4. BESO: Bayesian Evolutionary Skill Optimization

### 4.1 BESO starts from SkillOpt's abstraction

BESO accepts SkillOpt's main thesis:

$$
\text{the optimized object should be a skill artifact, not just a prompt string.}
$$

Let BESO's skill artifact be:

$$
z\in\mathcal{Z}_{\mathrm{skill}}
$$

with structured sections:

$$
z=(z_{\mathrm{goal}},z_{\mathrm{scope}},z_{\mathrm{procedure}},z_{\mathrm{tool}},z_{\mathrm{verify}},z_{\mathrm{failures}},z_{\mathrm{output}},z_{\mathrm{examples}})
$$

A compiler converts the skill into runtime prompt material:

$$
p=C(z,x,q)
$$

The frozen system runs:

$$
(y,\tau)=\Phi_{\Theta_{\mathrm{frozen}}}(x;C(z,x,q))
$$

The expected objective is:

$$
z^*\in\arg\max_{z\in\mathcal{Z}_{\mathrm{skill}}}
\mathbb{E}_{(x,m)\sim\mathcal{T}}
\left[\mu\left(\Phi_{\Theta_{\mathrm{frozen}}}(x;C(z,x,q)),m\right)\right]
$$

### 4.2 The problem BESO targets

SkillOpt proposes bounded edits and uses validation to accept/reject them. But candidate evaluation is expensive.

If an optimizer proposes $M$ candidate skill edits per round and each candidate needs $b$ rollouts, the naive cost is:

$$
\mathrm{Cost}_{\mathrm{naive}}=M\cdot b
$$

Under small rollout budget $B$, many candidates cannot be tested.

BESO asks:

> Given a pool of reflection-generated skill edits, which candidates should we evaluate first?

This is a Bayesian experiment-planning problem.

### 4.3 Candidate pool generation

Like SkillOpt, BESO uses reflection to generate candidate edits.

For parent skill $z_p$:

$$
\mathcal{C}_t=\{z_{t,1},\dots,z_{t,M}\}
$$

where:

$$
z_{t,j}=e_{t,j}(z_p)
$$

and:

$$
e_{t,j}\sim Q_{\psi}(e\mid z_p,H_t,R_t,A_t)
$$

$Q_{\psi}$ is the reflection proposal distribution.

So far, this is close to SkillOpt.

### 4.4 BESO's extra layer: probabilistic surrogate

BESO adds a surrogate model over candidate performance:

$$
f(z)=J(z)
$$

The true $f(z)$ is unknown. BESO observes noisy evaluations:

$$
\tilde{y}_i=f(z_i)+\epsilon_i
$$

History:

$$
H_t=\{(z_i,\tilde{y}_i,\tau_i,c_i)\}_{i=1}^{t}
$$

Bayesian posterior:

$$
p(f\mid H_t)
$$

For any candidate $z$:

$$
\mu_t(z)=\mathbb{E}[f(z)\mid H_t]
$$

$$
\sigma_t^2(z)=\operatorname{Var}[f(z)\mid H_t]
$$

Interpretation:

- $\mu_t(z)$ estimates how good the candidate is likely to be.
- $\sigma_t(z)$ estimates how uncertain the optimizer is.

### 4.5 Candidate featurization

Because $z$ is text, BESO needs a feature map:

$$
\varphi:\mathcal{Z}_{\mathrm{skill}}\to\mathbb{R}^{d}
$$

A useful decomposition:

$$
\varphi(z)=
[\varphi_{\mathrm{text}}(z),
\varphi_{\mathrm{struct}}(z),
\varphi_{\mathrm{edit}}(z),
\varphi_{\mathrm{history}}(z),
\varphi_{\mathrm{semantic}}(z)]
$$

Features can include:

- embedding of changed text,
- target section,
- edit operation,
- edit size,
- parent score,
- token count,
- number of rules,
- similarity to accepted candidates,
- similarity to rejected candidates,
- LLM-labeled emphasis on verification, decomposition, tool use, caution, etc.

This is a key BESO design point: Bayesian optimization does not operate directly on raw text; it operates on representations of candidate skill artifacts.

### 4.6 Acquisition function

BESO chooses candidates by acquisition:

$$
z_{t+1}\in\arg\max_{z\in\mathcal{C}_t}a_t(z)
$$

A default Upper Confidence Bound acquisition:

$$
a_{\mathrm{UCB}}(z)=\mu_t(z)+\kappa\sigma_t(z)
$$

A BESO-specific practical acquisition:

$$
\boxed{
a_{\mathrm{BESO}}(z)
=\mu_t(z)+\kappa\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)-\gamma\widehat{q}_{\mathrm{invalid}}(z)
}
$$

where:

- $d(z,A_t)$ rewards diversity from the archive.
- $\widehat{c}(z)$ penalizes cost or token bloat.
- $\widehat{q}_{\mathrm{invalid}}(z)$ penalizes predicted invalid output or schema breakage.
- $\kappa$ controls exploration.

This is the mathematical heart of BESO.

### 4.7 Archive management

BESO should maintain an archive:

$$
A_t=A_t^{\mathrm{best}}\cup A_t^{\mathrm{pareto}}\cup A_t^{\mathrm{diverse}}\cup A_t^{\mathrm{failed}}
$$

This combines GEPA's Pareto preservation with SkillOpt's rejected-edit memory.

- $A_t^{\mathrm{best}}$: high validation score.
- $A_t^{\mathrm{pareto}}$: specialists on different examples or objectives.
- $A_t^{\mathrm{diverse}}$: semantically distinct skills.
- $A_t^{\mathrm{failed}}$: informative failures that teach the surrogate and reflection module what not to repeat.

### 4.8 BESO's update recurrence

At iteration $t$:

1. Select parent skills:

$$
P_t\sim\pi_{\mathrm{parent}}(A_t,H_t)
$$

2. Generate candidates through reflection:

$$
\mathcal{C}_t=G_{\psi}(P_t,H_t,R_t,A_t)
$$

3. Fit / update Bayesian surrogate:

$$
p_t(f)=p(f\mid H_t)
$$

4. Select candidates to evaluate:

$$
S_t=\operatorname{TopK}_{z\in\mathcal{C}_t}a_t(z)
$$

5. Evaluate selected candidates:

$$
\tilde{y}_{z}=\hat{J}_{B_t}(z)+\epsilon_z,\qquad z\in S_t
$$

6. Gate candidates:

$$
\operatorname{Accept}(z)=1
\iff
\hat{J}_{D_{\mathrm{val}}}(z)>\hat{J}_{D_{\mathrm{val}}}(p(z))+\delta
$$

7. Update history and archive:

$$
H_{t+1}=H_t\cup\{(z,\tilde{y}_z,\tau_z,c_z):z\in S_t\}
$$

$$
A_{t+1}=\operatorname{ArchiveUpdate}(A_t,S_t,H_{t+1})
$$

### 4.9 BESO's core mathematical identity

BESO can be summarized as:

$$
\boxed{
\text{BESO} = \text{SkillOpt-style bounded skill edits} + \text{Bayesian acquisition over candidate skill variants} + \text{evolutionary archive}
}
$$

Or:

$$
\boxed{
\text{BESO} = \text{GEPA reflection} + \text{SkillOpt skill abstraction} + \text{Bayesian experiment planning}
}
$$

---

## 5. The lineage: what is inherited and what is new

### 5.1 From GEPA to SkillOpt

GEPA establishes:

$$
\tau + f \rightarrow R_{\psi}(\cdot) \rightarrow \text{textual candidate improvement}
$$

SkillOpt inherits that idea but changes the state variable:

$$
p \in \mathcal{P}
\quad\longrightarrow\quad
s\in\mathcal{S}
$$

So the shift is:

```text
GEPA: text candidate is often a prompt or textual system parameter.
SkillOpt: text candidate is a deployable skill document.
```

SkillOpt adds:

$$
\Delta(s,s')\le\eta_{\mathrm{text}}
$$

and:

$$
s_{t+1}=s_t' \;\text{only if}\; \hat{J}_{\mathrm{sel}}(s_t')>\hat{J}_{\mathrm{sel}}(s_t)
$$

That gives skill learning the discipline of:

- minibatches,
- learning-rate-like bounded edits,
- validation gates,
- rejected update memory,
- train/deploy separation.

### 5.2 From SkillOpt to BESO

SkillOpt generates and tests candidate edits. BESO asks which candidate edits deserve testing.

SkillOpt selection is primarily validation-gated:

$$
s_t' \rightarrow \operatorname{Eval}(s_t') \rightarrow \operatorname{Accept/Reject}
$$

BESO inserts a surrogate before expensive evaluation:

$$
\mathcal{C}_t \rightarrow p(f\mid H_t) \rightarrow a_t(z) \rightarrow \operatorname{Eval}(\operatorname{TopK}(a_t))
$$

So the shift is:

```text
SkillOpt: propose bounded edit, evaluate, accept if validation improves.
BESO: propose many bounded edits, predict value/uncertainty, evaluate the most promising or informative ones, then validate/archive.
```

Mathematically:

SkillOpt:

$$
z_t'=G_{\psi}(z_t,H_t,R_t),\qquad z_{t+1}=\operatorname{Gate}(z_t',z_t)
$$

BESO:

$$
\mathcal{C}_t=G_{\psi}(z_t,H_t,R_t)
$$

$$
z_t'\in\arg\max_{z\in\mathcal{C}_t}\left[\mu_t(z)+\kappa\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)\right]
$$

$$
z_{t+1}=\operatorname{Gate}(z_t',z_t)
$$

BESO's novelty is between proposal and evaluation.

---

## 6. Contribution map by component

| Component | GEPA | SkillOpt | BESO contribution |
|---|---|---|---|
| Frozen target model | Yes | Yes | Same assumption |
| Optimized text artifact | Prompt / arbitrary textual parameter | Skill document | Structured skill artifact, possibly compiled into runtime prompts |
| Reflection over traces | Core mechanism | Core mechanism | Inherited; used for candidate generation |
| Actionable side information | Core GEPA idea | Rollout/verifier feedback | Inherited; becomes surrogate and edit features too |
| Pareto diversity | Core selection mechanism | Less central; focus on best skill via validation | Reintroduced as archive layer with diversity and specialization |
| Bounded edits | Not the main framing | Core textual learning rate | Inherited from SkillOpt |
| Validation gate | Present in many optimizer loops | Core accept/reject mechanism | Inherited; can add statistical / risk-aware gates |
| Rejected-edit buffer | Not central | Core stability mechanism | Inherited; also used as negative features for surrogate/acquisition |
| Slow/meta update | Not central | SkillOpt stability mechanism | Optional; can condition reflection/candidate generation |
| Bayesian surrogate | No | No / not central | Main BESO addition |
| Acquisition function | No | No / not central | Main BESO addition |
| Low-budget experiment planning | Indirect via sample-efficient reflection | Indirect via bounded/gated edits | Directly optimized objective |
| Candidate featurization | Not central | Not central | Required for surrogate modeling |
| Uncertainty-aware exploration | Pareto diversity gives implicit exploration | Validation gate gives stability | Explicit through $\sigma_t(z)$ and acquisition |

---

## 7. BESO's contribution precisely stated

### 7.1 Not the contribution

BESO should not claim:

> We invented reflective prompt/skill mutation.

GEPA and SkillOpt already use trajectory-grounded reflection.

BESO should not claim:

> We invented skill documents as trainable state.

SkillOpt already makes this central.

BESO should not claim:

> We invented validation-gated skill updates.

SkillOpt already does this.

### 7.2 Actual contribution

BESO's contribution is:

> A Bayesian experiment-planning layer for skill-document evolution.

More formally:

Given a reflection-generated candidate set:

$$
\mathcal{C}_t=\{z_{t,1},\dots,z_{t,M}\}
$$

and a limited budget allowing only $K\ll M$ evaluations, BESO chooses:

$$
S_t\in\arg\max_{S\subset\mathcal{C}_t, |S|=K}
\sum_{z\in S}a_t(z)-\eta\sum_{z\ne z'\in S}\operatorname{sim}(z,z')
$$

where:

$$
a_t(z)=\mu_t(z)+\kappa\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)-\gamma\widehat{q}_{\mathrm{invalid}}(z)
$$

This turns skill evolution into an active learning / Bayesian optimization problem.

### 7.3 The scientific hypothesis

BESO's core testable hypothesis:

$$
\mathrm{AUC}_B(V_{\mathrm{BESO}})>
\mathrm{AUC}_B(V_{\mathrm{SkillOpt}})
$$

under the same rollout budget $B$, where:

$$
V(b)=\max_{z\in A(b)}\hat{J}_{\mathrm{val}}(z)
$$

That is: BESO should reach good skills faster, not merely produce a good final skill after unlimited trials.

Another hypothesis:

$$
\tau_{\gamma}^{\mathrm{BESO}} < \tau_{\gamma}^{\mathrm{SkillOpt}}
$$

where:

$$
\tau_{\gamma}=\min\{b:V(b)\ge\gamma\}
$$

BESO wins if it needs fewer rollouts to reach the same quality threshold.

---

## 8. Mathematical comparison of update rules

### 8.1 GEPA update

Prompt / text candidate population:

$$
P_t=\{p_{t,k}\}_{k=1}^{K}
$$

Candidate generation:

$$
\mathcal{C}_t=\{R_{\psi}(p,\mathcal{D}_{p}) : p\sim\pi_{\mathrm{pareto}}(P_t)\}
$$

Population update:

$$
P_{t+1}=\operatorname{ParetoUpdate}(P_t\cup\mathcal{C}_t)
$$

### 8.2 SkillOpt update

Single incumbent skill:

$$
s_t
$$

Reflection-generated bounded edit:

$$
s_t'=\operatorname{ApplyBounded}(s_t,R_{\psi}(s_t,\mathcal{B}_t,R_t,m_t),\eta_{\mathrm{text}})
$$

Validation-gated update:

$$
s_{t+1}=\begin{cases}
s_t' & \hat{J}_{\mathrm{sel}}(s_t')>\hat{J}_{\mathrm{sel}}(s_t) \\
s_t & \text{otherwise}
\end{cases}
$$

Rejected buffer:

$$
R_{t+1}=R_t\cup\{s_t' : \hat{J}_{\mathrm{sel}}(s_t')\le\hat{J}_{\mathrm{sel}}(s_t)\}
$$

### 8.3 BESO update

Archive and history:

$$
A_t, H_t, R_t
$$

Candidate pool:

$$
\mathcal{C}_t=G_{\psi}(A_t,H_t,R_t)
$$

Surrogate posterior:

$$
p_t(f)=p(f\mid H_t)
$$

Acquisition ranking:

$$
z_{t,j}\mapsto a_t(z_{t,j})
$$

Selected evaluation batch:

$$
S_t=\operatorname{TopK}_{z\in\mathcal{C}_t}a_t(z)
$$

Evaluation:

$$
\tilde{y}_z=\hat{J}_{B_t}(z)+\epsilon_z
$$

Archive update:

$$
A_{t+1}=\operatorname{ArchiveUpdate}(A_t,S_t,H_{t+1})
$$

So BESO generalizes SkillOpt from:

$$
\text{one proposed candidate} \rightarrow \text{evaluate/gate}
$$

to:

$$
\text{many proposed candidates} \rightarrow \text{Bayesian rank/select} \rightarrow \text{evaluate/gate/archive}
$$

---

## 9. Where BESO can empirically win

BESO should be strongest when:

1. Rollouts are expensive.

$$
B \text{ is small relative to } |\mathcal{C}_t|
$$

2. Reflection can generate many plausible edits.

$$
|\mathcal{C}_t|\gg K
$$

3. Candidate quality varies significantly.

$$
\operatorname{Var}_{z\in\mathcal{C}_t}[J(z)] \text{ is large}
$$

4. Features predict some performance signal.

$$
I(\varphi(z);J(z))>0
$$

where $I$ denotes mutual information.

5. Uncertainty-aware exploration matters.

$$
\exists z: \mu_t(z) \text{ moderate but } \sigma_t(z) \text{ high and } J(z) \text{ high}
$$

If all reflection-generated edits are similar, cheap to evaluate, or impossible to predict, BESO's Bayesian layer may add overhead without gain.

---

## 10. Experimental design to prove BESO's contribution

### 10.1 Baselines

BESO should compare against:

1. No skill.
2. Human skill.
3. One-shot LLM skill.
4. GEPA-style prompt evolution.
5. GEPA-style skill evolution if available.
6. SkillOpt-style bounded skill editing.
7. Random candidate selection from reflection-generated edits.
8. Greedy reflection-ranked candidate selection.
9. Bandit over edit types.
10. BESO without Pareto archive.
11. BESO without uncertainty term.
12. BESO without structured features.

### 10.2 Main metric: optimization curve

The most important curve is:

$$
V(b)=\max_{z\in A(b)}\hat{J}_{\mathrm{val}}(z)
$$

where $b$ is rollout budget spent.

Report:

$$
\mathrm{AUC}_B=\frac{1}{B}\int_0^B V(b)\,db
$$

and:

$$
\hat{J}_{\mathrm{test}}(z_{\mathrm{final}})
$$

BESO's first claim should be sample efficiency:

```text
same final budget: BESO reaches good skills earlier
same low budget: BESO finds better skills
same target score: BESO uses fewer rollouts
```

### 10.3 Ablations specific to BESO

| Ablation | Mathematical question |
|---|---|
| No surrogate | Does $p(f\mid H_t)$ add value? |
| Mean-only acquisition | Is $\sigma_t(z)$ useful? |
| No diversity term | Does $d(z,A_t)$ prevent local collapse? |
| No cost penalty | Does $\widehat{c}(z)$ control bloat? |
| Embeddings only | Is semantic text similarity enough? |
| Structured features only | Are cheap edit/section features enough? |
| Absolute-score model only | Is modeling $J(z)$ enough? |
| Delta model only | Is modeling $J(z)-J(p(z))$ better? |
| No rejected buffer features | Do failed edits improve future selection? |

### 10.4 Hypotheses as equations

BESO vs SkillOpt under budget $B$:

$$
\mathbb{E}[\hat{J}_{\mathrm{test}}(z_{\mathrm{BESO}}(B))]
>
\mathbb{E}[\hat{J}_{\mathrm{test}}(z_{\mathrm{SkillOpt}}(B))]
$$

especially for small $B$.

Sample efficiency:

$$
\mathrm{AUC}_B(V_{\mathrm{BESO}})>
\mathrm{AUC}_B(V_{\mathrm{SkillOpt}})
$$

Surrogate usefulness:

$$
\operatorname{corr}(\mu_t(z),\hat{J}(z))>0
$$

Uncertainty calibration:

$$
\mathbb{E}[(\hat{J}(z)-\mu_t(z))^2\mid \sigma_t(z)=s] \approx s^2
$$

Acquisition usefulness:

$$
\mathbb{E}_{z\sim\pi_{\mathrm{BESO}}}[J(z)]>
\mathbb{E}_{z\sim\pi_{\mathrm{random}}}[J(z)]
$$

where:

$$
\pi_{\mathrm{BESO}}(z)\propto\exp(a_t(z))
$$

---

## 11. Final compact formulation

All three methods solve:

$$
\max_{z\in\mathcal{Z}}\mathbb{E}_{(x,m)\sim\mathcal{T}}
\left[\mu\left(\Phi_{\Theta_{\mathrm{frozen}}}(x;z),m\right)\right]
$$

but choose different $\mathcal{Z}$ and different update rules.

### GEPA

$$
\mathcal{Z}=\mathcal{P}\quad\text{or broad textual parameter space}
$$

$$
z_{t+1}=R_{\psi}(z_t,\tau_t,f_t)
$$

with Pareto population update:

$$
P_{t+1}=\operatorname{ParetoUpdate}(P_t\cup\mathcal{C}_t)
$$

### SkillOpt

$$
\mathcal{Z}=\mathcal{S}\quad\text{skill-document space}
$$

$$
z_t'=\operatorname{ApplyBounded}(z_t,R_{\psi}(z_t,\mathcal{B}_t,R_t,m_t),\eta_{\mathrm{text}})
$$

$$
z_{t+1}=z_t'\quad\text{iff}\quad \hat{J}_{\mathrm{sel}}(z_t')>\hat{J}_{\mathrm{sel}}(z_t)
$$

### BESO

$$
\mathcal{Z}=\mathcal{S}_{\mathrm{structured}}\quad\text{structured skill artifacts}
$$

Generate many reflection candidates:

$$
\mathcal{C}_t=G_{\psi}(A_t,H_t,R_t)
$$

Model unknown utility:

$$
p(f\mid H_t),\qquad \mu_t(z),\sigma_t(z)
$$

Select by acquisition:

$$
z_{t+1}^{\mathrm{eval}}=\arg\max_{z\in\mathcal{C}_t}
\left[\mu_t(z)+\kappa\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)-\gamma\widehat{q}_{\mathrm{invalid}}(z)\right]
$$

Then validate and archive:

$$
A_{t+1}=\operatorname{ArchiveUpdate}(A_t,z_{t+1}^{\mathrm{eval}},H_{t+1})
$$

---

## 12. Clean contribution statement for the project

A strong paper-style contribution statement for BESO would be:

> We study Bayesian Evolutionary Skill Optimization (BESO), a sample-efficient optimizer for frozen LLM agents that treats structured natural-language skills as trainable external state. BESO inherits trajectory-grounded reflective mutation from GEPA and skill-document training from SkillOpt, but adds a probabilistic experiment-planning layer: reflection generates bounded candidate edits, a surrogate model predicts their utility and uncertainty from semantic, structural, edit, and history features, and an acquisition function selects which variants to evaluate under a limited rollout budget. This directly targets the evaluation bottleneck in skill evolution and should be most beneficial in low-budget regimes where many plausible edits cannot all be tested.

The key novelty in one sentence:

> GEPA teaches text candidates from trajectories; SkillOpt trains skills as external state; BESO decides which skill edits are worth spending rollouts on.
````

## File: docs/Bayesian Evolutionary Skill Optimization (BESO) - Mathematical Breakdown.md
````markdown
---
type: research-note
tags: [beso, bayesian-optimization, evolutionary-search, skill-optimization, llm-agents]
date: 2026-05-28
source: [[Bayesian Evolutionary Skill Optimization (BESO) - Technical Specification]]
status: draft
---

# Bayesian Evolutionary Skill Optimization (BESO): mathematical breakdown

## 0. One-line mathematical summary

BESO optimizes a natural-language skill artifact $z$ for a frozen LLM system $\Phi$ by using trajectory-grounded reflection to generate candidate edits, a Bayesian surrogate to model uncertain candidate utility, an acquisition function to allocate a limited rollout budget, and an evolutionary archive to preserve high-performing or specialized skill variants.

The central objective is:

$$
z^* \in \arg\max_{z \in \mathcal{Z}} \; J(z)
= \arg\max_{z \in \mathcal{Z}} \; \mathbb{E}_{(x,m) \sim \mathcal{T}}\left[\mu\left(\Phi(x; C(z), \Theta_{\mathrm{frozen}}), m\right)\right]
$$

where:

- $z$ is a structured skill artifact.
- $\mathcal{Z}$ is the space of valid skill artifacts.
- $C$ is the skill compiler that converts a skill artifact into runtime prompt material.
- $\Phi$ is the frozen LLM system or agent runtime.
- $\Theta_{\mathrm{frozen}}$ are fixed model weights.
- $x$ is a task input.
- $m$ is the task metadata, label, rubric, expected answer, or test case.
- $\mu$ is the evaluator.
- $\mathcal{T}$ is the unknown task distribution.
- $J(z)$ is the true expected utility of skill $z$.

In practical form, BESO solves a budgeted, noisy, black-box, discrete optimization problem:

$$
\max_{z \in \mathcal{Z}} J(z)
\quad \text{subject to} \quad
\sum_{t=1}^{T} c(z_t, B_t) \le B
$$

where $B$ is the total rollout budget and $c(z_t, B_t)$ is the cost of evaluating candidate $z_t$ on minibatch $B_t$.

---

## 1. Objects and spaces

### 1.1 Task distribution

Let the task distribution be:

$$
\mathcal{T} \in \Delta(\mathcal{X} \times \mathcal{M})
$$

where:

- $\mathcal{X}$ is the input space.
- $\mathcal{M}$ is the space of metadata, labels, rubrics, test cases, or evaluator references.
- $\Delta(\cdot)$ denotes a probability distribution over a space.

A task instance is:

$$
\xi = (x,m) \sim \mathcal{T}
$$

For a multi-hop QA task, $x$ might be the question and $m$ might be the expected answer plus rubric. For code generation, $x$ might be a problem statement and $m$ might be a unit-test suite. For structured extraction, $x$ might be a document and $m$ might be the target JSON fields.

### 1.2 Dataset approximation

The true distribution $\mathcal{T}$ is unknown. We observe a finite dataset:

$$
D = \{\xi_i\}_{i=1}^{n} = \{(x_i,m_i)\}_{i=1}^{n}
$$

BESO should split this dataset into disjoint roles:

$$
D = D_{\mathrm{fb}} \cup D_{\mathrm{opt}} \cup D_{\mathrm{val}} \cup D_{\mathrm{test}}
$$

where:

- $D_{\mathrm{fb}}$ generates trajectories for reflection.
- $D_{\mathrm{opt}}$ gives fast candidate estimates during optimization.
- $D_{\mathrm{val}}$ gates candidate acceptance.
- $D_{\mathrm{test}}$ is untouched until final reporting.

This split matters because BESO edits text artifacts. Text artifacts can overfit by encoding benchmark quirks, overly specific rules, or accidental validation shortcuts.

### 1.3 Frozen system

Let the frozen LLM system be:

$$
\Phi_{\Theta}: \mathcal{X} \times \mathcal{P} \to \mathcal{Y} \times \mathcal{R}
$$

where:

- $\Theta = \Theta_{\mathrm{frozen}}$ are fixed model parameters.
- $\mathcal{P}$ is the runtime prompt / instruction space.
- $\mathcal{Y}$ is the final output space.
- $\mathcal{R}$ is the trajectory / trace space.

Given input $x$ and runtime prompt $p$, the system returns:

$$
(y, \tau) = \Phi_{\Theta_{\mathrm{frozen}}}(x; p)
$$

where:

- $y$ is the final output.
- $\tau$ is the recorded trajectory.

The trajectory may contain prompts, intermediate outputs, tool calls, tool results, retrieved documents, errors, evaluator feedback, costs, and latency. It should not be assumed to contain hidden chain-of-thought; it contains only the observable trace made available by the system.

### 1.4 Skill artifact space

BESO does not directly optimize model weights. It optimizes skill artifacts:

$$
z \in \mathcal{Z}
$$

A skill artifact can be modeled as a structured object:

$$
z = (s_1, s_2, \dots, s_L)
$$

where each $s_{\ell}$ is a section, such as:

- goal,
- scope,
- core procedure,
- reasoning policy,
- tool-use policy,
- verification checklist,
- common failure modes,
- recovery rules,
- output rules,
- examples,
- change log.

The space $\mathcal{Z}$ is not the space of all strings. It is the constrained space of schema-valid skill artifacts:

$$
\mathcal{Z} = \{z \in \Sigma^* : \mathrm{SchemaValid}(z)=1, \; \mathrm{BudgetValid}(z)=1, \; \mathrm{InvariantValid}(z)=1\}
$$

where $\Sigma^*$ is the space of finite text strings.

This distinction is important. Raw prompt optimization searches in an unstructured text space. BESO searches in a typed, sectioned, linted text space.

### 1.5 Skill compiler

The skill artifact is compiled into runtime prompt material:

$$
p = C(z, x, q)
$$

where:

- $C$ is the compiler.
- $z$ is the skill artifact.
- $x$ is the task input.
- $q$ is runtime context: available tools, output schema, module role, cost constraints, or selected skill sections.

The compiler may use one of several modes:

$$
C \in \{C_{\mathrm{full}}, C_{\mathrm{section}}, C_{\mathrm{distill}}\}
$$

where:

- $C_{\mathrm{full}}$ injects the whole skill.
- $C_{\mathrm{section}}$ selects relevant sections.
- $C_{\mathrm{distill}}$ compresses the skill into a compact prompt.

The compiled prompt is not the optimization target. It is the runtime consequence of the skill artifact.

---

## 2. Evaluation and objective functions

### 2.1 Single-example score

Given a skill artifact $z$ and task instance $\xi_i=(x_i,m_i)$:

$$
p_i = C(z, x_i, q_i)
$$

$$
(y_i, \tau_i) = \Phi_{\Theta_{\mathrm{frozen}}}(x_i; p_i)
$$

The evaluator assigns a score:

$$
r_i(z) = \mu(y_i, \tau_i, m_i) \in \mathbb{R}
$$

Often:

$$
r_i(z) \in [0,1]
$$

but the framework can support arbitrary bounded metrics.

The evaluator may depend on the final answer only:

$$
r_i(z) = \mu(y_i,m_i)
$$

or on the trajectory as well:

$$
r_i(z) = \mu(y_i,\tau_i,m_i)
$$

Trajectory-dependent scoring is important for tool-use tasks, invalid-output penalties, cost penalties, and safety constraints.

### 2.2 True expected skill utility

The ideal objective is:

$$
J(z) = \mathbb{E}_{(x,m)\sim\mathcal{T}}\left[\mu\left(y,\tau,m\right)\right]
$$

with:

$$
(y,\tau)=\Phi_{\Theta_{\mathrm{frozen}}}(x; C(z,x,q))
$$

Thus:

$$
J(z) = \mathbb{E}_{(x,m)\sim\mathcal{T}}\left[\mu\left(\Phi_{\Theta_{\mathrm{frozen}}}(x; C(z,x,q)),m\right)\right]
$$

The target skill is:

$$
z^* \in \arg\max_{z\in\mathcal{Z}} J(z)
$$

### 2.3 Empirical risk / empirical utility

Since $\mathcal{T}$ is unknown, define empirical utility on dataset $D$:

$$
\hat{J}_D(z)=\frac{1}{|D|}\sum_{(x_i,m_i)\in D} r_i(z)
$$

For a split $S \subset D$:

$$
\hat{J}_S(z)=\frac{1}{|S|}\sum_{i\in S} r_i(z)
$$

The optimization-time estimate may use minibatches:

$$
\hat{J}_{B_t}(z)=\frac{1}{|B_t|}\sum_{i\in B_t} r_i(z)
$$

where:

$$
B_t \subset D_{\mathrm{opt}}
$$

### 2.4 Noisy observation model

Evaluation is noisy because LLM outputs, judge scores, sampled minibatches, tool results, and retrieval contexts can vary.

Model an observed candidate score as:

$$
\tilde{y}_t = \hat{J}_{B_t}(z_t) + \epsilon_t
$$

where:

$$
\epsilon_t \sim \mathcal{N}(0, \sigma_{\epsilon}^2(z_t, B_t))
$$

This noise can be decomposed:

$$
\sigma_{\epsilon}^2
= \sigma_{\mathrm{model}}^2
+ \sigma_{\mathrm{judge}}^2
+ \sigma_{\mathrm{batch}}^2
+ \sigma_{\mathrm{tool}}^2
$$

where:

- $\sigma_{\mathrm{model}}^2$ comes from stochastic decoding or model variance.
- $\sigma_{\mathrm{judge}}^2$ comes from LLM-as-judge or rubric instability.
- $\sigma_{\mathrm{batch}}^2$ comes from using a minibatch rather than the whole dataset.
- $\sigma_{\mathrm{tool}}^2$ comes from nondeterministic tools, retrieval, network state, or environment state.

For deterministic decoding and deterministic evaluators, some of these terms shrink, but minibatch noise usually remains.

### 2.5 Cost-aware utility

BESO should not optimize quality alone if a skill becomes too expensive at runtime. Define cost:

$$
\mathrm{Cost}(z; x) = c_{\mathrm{tok}}(z,x) + c_{\mathrm{tool}}(z,x) + c_{\mathrm{lat}}(z,x)
$$

A scalarized cost-aware utility can be:

$$
U(z) = J_{\mathrm{quality}}(z) - \lambda_{\mathrm{tok}}J_{\mathrm{tokens}}(z) - \lambda_{\mathrm{tool}}J_{\mathrm{tools}}(z) - \lambda_{\mathrm{lat}}J_{\mathrm{latency}}(z)
$$

where:

$$
J_{\mathrm{tokens}}(z)=\mathbb{E}_{x\sim\mathcal{T}}[\mathrm{Tokens}(C(z,x,q))]
$$

This prevents prompt bloat from masquerading as improvement.

---

## 3. BESO as black-box optimization over structured text

### 3.1 Why this is black-box

The function $J(z)$ is not analytically available. We cannot compute gradients through:

- natural-language skill documents,
- discrete edit operations,
- LLM sampling,
- external tools,
- evaluator logic,
- unit tests,
- human or LLM judges.

So BESO treats $J$ as a black-box objective:

$$
z \mapsto J(z)
$$

We can query the function by evaluating candidates, but each query costs rollouts.

### 3.2 Why this is not ordinary Bayesian optimization

Classical Bayesian optimization usually assumes a continuous input space:

$$
z \in \mathbb{R}^d
$$

BESO has a structured discrete text space:

$$
z \in \mathcal{Z} \subset \Sigma^*
$$

Therefore BESO uses a two-stage strategy:

1. A reflection model proposes a finite candidate set:

$$
\mathcal{C}_t = G(A_t, H_t, \mathcal{T}_t)
$$

2. A Bayesian surrogate ranks candidates inside that finite set:

$$
z_{t+1} \in \arg\max_{z\in\mathcal{C}_t} a_t(z)
$$

This makes BESO closer to Bayesian-guided evolutionary search than pure continuous Bayesian optimization.

The Bayesian model does not need to search all possible text. It only needs to decide which generated candidates are worth evaluating.

---

## 4. History, archive, and state

### 4.1 Optimization history

After $t$ evaluated candidates, the history is:

$$
H_t = \{(z_j, B_j, \tilde{y}_j, \mathcal{R}_j, c_j)\}_{j=1}^{t}
$$

where:

- $z_j$ is an evaluated skill candidate.
- $B_j$ is the evaluation minibatch.
- $\tilde{y}_j$ is the observed score.
- $\mathcal{R}_j = \{\tau_{j,i}\}_{i\in B_j}$ is the set of trajectories.
- $c_j$ is the evaluation cost.

If candidate $z_j$ is evaluated multiple times, store repeated observations:

$$
H_t = \{(z_j, B_{j,k}, \tilde{y}_{j,k}, \mathcal{R}_{j,k}, c_{j,k})\}_{j,k}
$$

Then estimate its mean and uncertainty:

$$
\bar{y}_j = \frac{1}{K_j}\sum_{k=1}^{K_j}\tilde{y}_{j,k}
$$

$$
\widehat{\mathrm{SE}}(z_j)=\frac{\widehat{\sigma}(z_j)}{\sqrt{K_j}}
$$

### 4.2 Archive

The archive stores candidates that are useful for exploitation, diversity, specialization, or negative learning:

$$
A_t \subseteq \{z_1,\dots,z_t\}
$$

The archive may be decomposed:

$$
A_t = A_t^{\mathrm{best}} \cup A_t^{\mathrm{pareto}} \cup A_t^{\mathrm{diverse}} \cup A_t^{\mathrm{failed}}
$$

where:

- $A_t^{\mathrm{best}}$ contains high-average performers.
- $A_t^{\mathrm{pareto}}$ contains candidates that win on specific examples or objectives.
- $A_t^{\mathrm{diverse}}$ contains semantically distinct candidates.
- $A_t^{\mathrm{failed}}$ contains informative failures to avoid repeated mistakes.

### 4.3 Parent selection

Candidate generation starts from parents:

$$
P_t \subset A_t
$$

A parent selection distribution can combine validation score, Pareto win count, and diversity:

$$
\Pr(z \in P_t) \propto
\exp\left(
\beta_1 \hat{J}_{\mathrm{val}}(z)
+ \beta_2 w_t(z)
+ \beta_3 d(z,A_t)
- \beta_4 \mathrm{Cost}(z)
\right)
$$

where:

- $w_t(z)$ is a Pareto win-count or specialization score.
- $d(z,A_t)$ is a diversity measure.
- $\beta_1,\dots,\beta_4$ are selection weights.

---

## 5. Reflection-generated candidate edits

### 5.1 Edit operators

Let $\mathcal{E}$ be the space of valid edit operations:

$$
e \in \mathcal{E}
$$

Examples:

- add rule,
- delete rule,
- replace rule,
- specialize rule,
- generalize rule,
- reorder steps,
- add example,
- delete example,
- compress section,
- add failure mode,
- add recovery rule.

Each edit is a transformation:

$$
e: \mathcal{Z} \to \mathcal{Z}
$$

Applying edit $e$ to skill $z$ gives:

$$
z' = e(z)
$$

A sequence of edits $\mathbf{e}=(e_1,\dots,e_K)$ gives:

$$
z' = e_K \circ e_{K-1} \circ \cdots \circ e_1(z)
$$

### 5.2 Valid edit constraint

Not all edits are allowed. Define an edit validity indicator:

$$
\nu(z,e) \in \{0,1\}
$$

where:

$$
\nu(z,e)=1
$$

if and only if the edit preserves schema, token budget, section constraints, output format rules, tool availability, and task invariants.

Thus the feasible edit set for skill $z$ is:

$$
\mathcal{E}(z)=\{e\in\mathcal{E}: \nu(z,e)=1\}
$$

The feasible one-step neighborhood is:

$$
\mathcal{N}(z)=\{e(z): e\in\mathcal{E}(z)\}
$$

### 5.3 Reflection as proposal distribution

The reflection module is a proposal distribution over edits:

$$
Q_{\psi}(e \mid z, \mathcal{R}, F, A_t, H_t)
$$

where:

- $\psi$ are the reflection model parameters.
- $z$ is the parent skill.
- $\mathcal{R}$ is trajectory evidence.
- $F$ is evaluator feedback.
- $A_t$ is the archive.
- $H_t$ is the history.

The candidate pool is sampled from this distribution:

$$
e_{t,1},\dots,e_{t,M} \sim Q_{\psi}(\cdot \mid z, \mathcal{R}, F, A_t,H_t)
$$

$$
\mathcal{C}_t = \{e_{t,j}(z): j=1,\dots,M, \; \nu(z,e_{t,j})=1\}
$$

BESO can also use multiple parents:

$$
\mathcal{C}_t = \bigcup_{z\in P_t}\{e(z): e\sim Q_{\psi}(\cdot\mid z,\mathcal{R}_z,F_z,A_t,H_t),\; \nu(z,e)=1\}
$$

### 5.4 Crossover / semantic merge

For two parent skills $z_a,z_b\in A_t$, define a semantic merge operator:

$$
M_{\psi}: \mathcal{Z}\times\mathcal{Z}\times\mathcal{R} \to \mathcal{Z}
$$

The merged candidate is:

$$
z' = M_{\psi}(z_a,z_b,\mathcal{R}_{a,b})
$$

The merge should preserve complementary strengths:

$$
z' \approx \operatorname{Combine}\left(\mathrm{Strengths}(z_a), \mathrm{Strengths}(z_b)\right)
$$

but it must still satisfy:

$$
z' \in \mathcal{Z}
$$

This is not literal string crossover. It is semantic recombination under skill-schema constraints.

---

## 6. Candidate featurization

### 6.1 Feature map

Bayesian surrogates require numeric inputs. Define a feature map:

$$
\varphi: \mathcal{Z} \to \mathbb{R}^{d}
$$

A candidate skill becomes:

$$
\mathbf{x}_z = \varphi(z)
$$

To avoid confusion with task input $x$, use $\mathbf{x}_z$ for candidate features.

### 6.2 Feature decomposition

The feature map can be decomposed:

$$
\varphi(z) = \left[
\varphi_{\mathrm{text}}(z),
\varphi_{\mathrm{struct}}(z),
\varphi_{\mathrm{edit}}(z),
\varphi_{\mathrm{hist}}(z),
\varphi_{\mathrm{sem}}(z)
\right]
$$

where:

- $\varphi_{\mathrm{text}}(z)$ is an embedding of the skill text or changed section.
- $\varphi_{\mathrm{struct}}(z)$ includes token count, number of rules, number of examples, etc.
- $\varphi_{\mathrm{edit}}(z)$ encodes edit operation, target section, edit size, and parent ID.
- $\varphi_{\mathrm{hist}}(z)$ includes parent score, lineage depth, and edit-type success rates.
- $\varphi_{\mathrm{sem}}(z)$ includes LLM-labeled semantic properties such as verification emphasis or tool-use aggressiveness.

A useful first representation is:

$$
\varphi(z) = [E(z_{\Delta}); \; h(z); \; g(e); \; \ell(z)]
$$

where:

- $E(z_{\Delta})$ is the embedding of the changed text.
- $h(z)$ is structural metadata.
- $g(e)$ is edit-operation metadata.
- $\ell(z)$ is lineage and historical metadata.

### 6.3 Similarity kernel

If using a Gaussian-process-like surrogate, define a kernel over candidates:

$$
k(z,z') = k_{\mathrm{text}}(z,z') + k_{\mathrm{struct}}(z,z') + k_{\mathrm{edit}}(z,z')
$$

Text kernel:

$$
k_{\mathrm{text}}(z,z') = \exp\left(-\frac{\|E(z)-E(z')\|_2^2}{2\ell_{\mathrm{text}}^2}\right)
$$

Structural kernel:

$$
k_{\mathrm{struct}}(z,z') = \exp\left(-\frac{\|h(z)-h(z')\|_2^2}{2\ell_{\mathrm{struct}}^2}\right)
$$

Edit kernel:

$$
k_{\mathrm{edit}}(z,z') = \mathbb{1}[\mathrm{op}(z)=\mathrm{op}(z')] \cdot \mathbb{1}[\mathrm{section}(z)=\mathrm{section}(z')]
$$

A weighted composite kernel:

$$
k(z,z') = \alpha k_{\mathrm{text}}(z,z') + \beta k_{\mathrm{struct}}(z,z') + \gamma k_{\mathrm{edit}}(z,z')
$$

with:

$$
\alpha,\beta,\gamma \ge 0
$$

---

## 7. Bayesian surrogate

### 7.1 Unknown performance function

Let:

$$
f(z)=J(z)
$$

BESO cannot observe $f(z)$ directly. It observes noisy minibatch scores:

$$
\tilde{y}_t = f(z_t) + \epsilon_t
$$

or, more precisely:

$$
\tilde{y}_t = \hat{J}_{B_t}(z_t) + \epsilon_t
$$

The surrogate maintains a posterior:

$$
p(f \mid H_t)
$$

From this posterior, for any candidate $z$, it estimates:

$$
\mu_t(z)=\mathbb{E}[f(z)\mid H_t]
$$

$$
\sigma_t^2(z)=\mathrm{Var}[f(z)\mid H_t]
$$

### 7.2 Gaussian-process surrogate

If using a Gaussian process:

$$
f \sim \mathcal{GP}(m_0,k)
$$

Given evaluated feature vectors:

$$
X_t = [\varphi(z_1),\dots,\varphi(z_t)]^\top
$$

and observations:

$$
\mathbf{y}_t = [\tilde{y}_1,\dots,\tilde{y}_t]^\top
$$

The posterior predictive mean for candidate $z$ is:

$$
\mu_t(z)=m_0(z)+\mathbf{k}_t(z)^\top (K_t+\sigma_\epsilon^2 I)^{-1}(\mathbf{y}_t-m_0(X_t))
$$

The posterior predictive variance is:

$$
\sigma_t^2(z)=k(z,z)-\mathbf{k}_t(z)^\top (K_t+\sigma_\epsilon^2 I)^{-1}\mathbf{k}_t(z)
$$

where:

$$
\mathbf{k}_t(z)=[k(z_1,z),\dots,k(z_t,z)]^\top
$$

and:

$$
K_t[i,j]=k(z_i,z_j)
$$

A GP is elegant but may struggle if embeddings are high-dimensional and the number of candidates grows.

### 7.3 Bayesian linear / ridge surrogate

A simpler model assumes:

$$
f(z)=\mathbf{w}^\top \varphi(z)+\epsilon
$$

with prior:

$$
\mathbf{w}\sim\mathcal{N}(0,\lambda^{-1}I)
$$

and noise:

$$
\epsilon\sim\mathcal{N}(0,\sigma^2)
$$

Posterior:

$$
p(\mathbf{w}\mid H_t)=\mathcal{N}(\mathbf{m}_t, S_t)
$$

Then:

$$
\mu_t(z)=\mathbf{m}_t^\top\varphi(z)
$$

$$
\sigma_t^2(z)=\varphi(z)^\top S_t \varphi(z) + \sigma^2
$$

This is less expressive but robust and easy to inspect.

### 7.4 Ensemble surrogate

For v0, an ensemble surrogate may be more practical.

Let there be $M$ predictive models:

$$
\{g_m\}_{m=1}^{M}
$$

Each model predicts:

$$
\hat{f}_m(z)=g_m(\varphi(z))
$$

Mean prediction:

$$
\mu_t(z)=\frac{1}{M}\sum_{m=1}^{M}\hat{f}_m(z)
$$

Uncertainty from disagreement:

$$
\sigma_t^2(z)=\frac{1}{M-1}\sum_{m=1}^{M}(\hat{f}_m(z)-\mu_t(z))^2
$$

This uncertainty is not fully Bayesian, but it is often usable for acquisition when true posterior modeling is difficult.

### 7.5 Improvement model instead of absolute score model

Instead of modeling absolute utility $J(z)$, BESO can model improvement over parent:

$$
\Delta(z,z_p)=J(z)-J(z_p)
$$

Observation:

$$
\tilde{\Delta}_t = \tilde{y}(z_t)-\tilde{y}(z_{p(t)})
$$

The surrogate becomes:

$$
p(\Delta \mid H_t)
$$

This may be easier because candidate quality is often relative to the parent skill. An edit that helps one parent may not help another.

A useful hybrid score is:

$$
\mu_t^{\mathrm{hybrid}}(z)=\mu_t^{\mathrm{abs}}(z)+\rho\mu_t^{\Delta}(z,z_p)
$$

where $\rho$ controls how much the optimizer trusts improvement modeling.

---

## 8. Acquisition functions

### 8.1 Acquisition as budget allocation

The acquisition function decides which candidates deserve expensive evaluation.

At iteration $t$, candidate pool:

$$
\mathcal{C}_t=\{z_{t,1},\dots,z_{t,M}\}
$$

BESO selects:

$$
z_{t+1}\in\arg\max_{z\in\mathcal{C}_t}a_t(z)
$$

or, for batch size $K$:

$$
S_t\in\arg\max_{S\subset\mathcal{C}_t, |S|=K}\sum_{z\in S}a_t(z) - \eta\sum_{z,z'\in S}\mathrm{sim}(z,z')
$$

The second form discourages selecting near-duplicate candidates in the same batch.

### 8.2 Upper Confidence Bound

Default acquisition:

$$
a_{\mathrm{UCB}}(z)=\mu_t(z)+\kappa_t\sigma_t(z)
$$

where:

- $\mu_t(z)$ rewards predicted performance.
- $\sigma_t(z)$ rewards uncertainty.
- $\kappa_t$ controls exploration.

High $\kappa_t$ means "try uncertain candidates." Low $\kappa_t$ means "exploit predicted winners."

A decaying exploration schedule:

$$
\kappa_t = \kappa_0 \sqrt{\frac{\log(t+1)}{t+1}}
$$

A non-decaying schedule may be better if candidate generation distribution shifts heavily over time.

### 8.3 Expected Improvement

Let the current best validated score be:

$$
f^+ = \max_{z\in A_t}\hat{J}_{\mathrm{val}}(z)
$$

Improvement random variable:

$$
I(z)=\max(0,f(z)-f^+-\xi)
$$

Expected improvement:

$$
\mathrm{EI}(z)=\mathbb{E}[I(z)]
$$

If $f(z)\sim\mathcal{N}(\mu_t(z),\sigma_t^2(z))$, then:

$$
\mathrm{EI}(z)=(\mu_t(z)-f^+-\xi)\Phi(\gamma)+\sigma_t(z)\phi(\gamma)
$$

where:

$$
\gamma=\frac{\mu_t(z)-f^+-\xi}{\sigma_t(z)}
$$

and $\Phi$ and $\phi$ are the standard normal CDF and PDF.

EI is useful when BESO mainly wants to beat the current best skill, but it may over-focus on short-term improvement.

### 8.4 Probability of Improvement

$$
\mathrm{PI}(z)=\Pr(f(z)>f^+ + \xi)
$$

Assuming Gaussian posterior:

$$
\mathrm{PI}(z)=\Phi\left(\frac{\mu_t(z)-f^+-\xi}{\sigma_t(z)}\right)
$$

PI is simple but often too greedy.

### 8.5 Thompson sampling

Sample a plausible function from the posterior:

$$
\tilde{f}_t \sim p(f\mid H_t)
$$

Select:

$$
z_{t+1}\in\arg\max_{z\in\mathcal{C}_t}\tilde{f}_t(z)
$$

For ensemble surrogates, a practical approximation is:

1. sample one model $g_m$ from the ensemble,
2. choose the best candidate according to $g_m$.

Thompson sampling is useful in batched settings because it naturally diversifies choices.

### 8.6 Diversity-aware acquisition

Let distance from archive be:

$$
d(z,A_t)=\min_{z'\in A_t}D(\varphi(z),\varphi(z'))
$$

where $D$ can be cosine distance, edit distance, section-level distance, or embedding distance.

Then:

$$
a_{\mathrm{div}}(z)=\mu_t(z)+\kappa_t\sigma_t(z)+\lambda d(z,A_t)
$$

This rewards candidates that are both promising and not redundant.

### 8.7 Cost-aware acquisition

Let predicted runtime cost be:

$$
\widehat{c}_t(z)
$$

Then:

$$
a_{\mathrm{cost}}(z)=\mu_t(z)+\kappa_t\sigma_t(z)-\alpha\widehat{c}_t(z)
$$

A richer form:

$$
a(z)=\mu_t(z)+\kappa_t\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{\mathrm{Cost}}(z)-\gamma\widehat{\mathrm{InvalidRisk}}(z)
$$

This is a strong default for BESO:

$$
\boxed{
a_{\mathrm{BESO}}(z)=\mu_t(z)+\kappa_t\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)-\gamma\widehat{q}_{\mathrm{invalid}}(z)
}
$$

---

## 9. Acceptance, rejection, and validation gates

### 9.1 Parent-relative acceptance

Let $p(z)$ be the parent of candidate $z$.

A simple acceptance rule:

$$
\mathrm{Accept}(z)=1
\iff
\hat{J}_{D_{\mathrm{val}}}(z) > \hat{J}_{D_{\mathrm{val}}}(p(z)) + \delta
$$

where $\delta$ is a minimum improvement threshold.

### 9.2 Bootstrap acceptance

Because validation estimates are noisy, use bootstrap resampling.

For candidate $z$ and parent $p$, compute per-example differences:

$$
d_i = r_i(z)-r_i(p)
$$

Mean difference:

$$
\bar{d}=\frac{1}{|D_{\mathrm{val}}|}\sum_{i\in D_{\mathrm{val}}}d_i
$$

Bootstrap confidence interval:

$$
\mathrm{CI}_{1-\alpha}(\bar{d})=[L,U]
$$

Accept if:

$$
L > 0
$$

or, less strictly:

$$
\bar{d}>\delta \quad \text{and} \quad L > -\epsilon
$$

This avoids accepting edits that look good only due to minibatch luck.

### 9.3 Risk-constrained acceptance

A candidate must also satisfy constraints:

$$
\mathrm{Accept}(z)=1
$$

only if:

$$
\hat{J}_{\mathrm{val}}(z) \ge \hat{J}_{\mathrm{val}}(p(z))+\delta
$$

$$
\mathrm{InvalidRate}(z) \le \rho_{\max}
$$

$$
\mathrm{Cost}(z) \le C_{\max}
$$

$$
\mathrm{Tokens}(z) \le T_{\max}
$$

$$
\mathrm{SchemaValid}(z)=1
$$

$$
\mathrm{InvariantValid}(z)=1
$$

This prevents the optimizer from improving one score while breaking output format, cost, or safety invariants.

### 9.4 Rejected edits as negative evidence

Rejected edits are not useless. They provide information about harmful regions of the edit space.

Let $R_t$ be the rejected-edit buffer:

$$
R_t=\{(e_j,z_j,\mathrm{reason}_j)\}_{j=1}^{r_t}
$$

Reflection should condition on $R_t$:

$$
Q_{\psi}(e\mid z,\mathcal{R},F,A_t,H_t,R_t)
$$

The surrogate can also use similarity to rejected candidates:

$$
\varphi_{\mathrm{reject}}(z)=\min_{z'\in R_t}D(\varphi(z),\varphi(z'))
$$

or a penalty:

$$
\mathrm{RejectPenalty}(z)=\exp\left(-\min_{z'\in R_t}\frac{D(\varphi(z),\varphi(z'))^2}{2\ell_R^2}\right)
$$

Then acquisition becomes:

$$
a(z)=\mu_t(z)+\kappa_t\sigma_t(z)+\lambda d(z,A_t)-\eta\mathrm{RejectPenalty}(z)
$$

---

## 10. Pareto archive mathematics

### 10.1 Instance-level specialization

Let candidate $z_k$ have score on example $i$:

$$
S[k,i]=r_i(z_k)
$$

For each example:

$$
s_i^*=\max_k S[k,i]
$$

Winner set:

$$
W_i=\{z_k:S[k,i]=s_i^*\}
$$

Candidate win count:

$$
w(z_k)=\sum_{i=1}^{n}\mathbb{1}[z_k\in W_i]
$$

Selection probability:

$$
P_{\mathrm{win}}(z_k)=\frac{w(z_k)+\epsilon}{\sum_j(w(z_j)+\epsilon)}
$$

The small $\epsilon$ prevents zero probability for candidates that are good but not exact winners.

### 10.2 Pareto dominance over objective vectors

For multi-objective evaluation, define:

$$
F(z)=\left(F_1(z),F_2(z),\dots,F_K(z)\right)
$$

Example:

$$
F(z)=\left(J_{\mathrm{acc}}(z),J_{\mathrm{format}}(z),-\mathrm{Cost}(z),-\mathrm{Latency}(z),-\mathrm{InvalidRate}(z)\right)
$$

Candidate $z_a$ Pareto-dominates $z_b$ if:

$$
z_a \succ z_b
$$

when:

$$
F_k(z_a)\ge F_k(z_b) \quad \forall k
$$

and:

$$
\exists k: F_k(z_a)>F_k(z_b)
$$

The Pareto front is:

$$
\mathcal{P}_t=\{z\in A_t:\nexists z'\in A_t \text{ such that } z'\succ z\}
$$

This preserves candidates that represent different trade-offs: high accuracy, low cost, strong format validity, fast latency, or robustness on hard examples.

### 10.3 Archive pruning as constrained subset selection

If the archive is too large, choose a subset:

$$
A_t' \subset A_t
$$

with:

$$
|A_t'|\le M_A
$$

One objective:

$$
A_t' \in \arg\max_{A\subseteq A_t, |A|\le M_A}
\left[
\sum_{z\in A}\hat{J}_{\mathrm{val}}(z)
+ \lambda_1 \mathrm{Coverage}(A)
+ \lambda_2 \mathrm{Diversity}(A)
+ \lambda_3 \mathrm{ParetoValue}(A)
\right]
$$

Diversity can be:

$$
\mathrm{Diversity}(A)=\frac{1}{|A|(|A|-1)}\sum_{z\ne z'\in A}D(\varphi(z),\varphi(z'))
$$

Coverage can be instance-level:

$$
\mathrm{Coverage}(A)=\sum_{i=1}^{n}\max_{z\in A}S[z,i]
$$

This says: keep an archive that collectively covers many examples well, not just one average winner.

---

## 11. Multi-objective and constrained formulations

### 11.1 Scalarized objective

For a single scalar training objective:

$$
J_{\mathrm{scalar}}(z)=\sum_{k=1}^{K}w_kJ_k(z)
$$

with:

$$
\sum_{k=1}^{K}w_k=1, \quad w_k\ge0
$$

Example:

$$
J_{\mathrm{scalar}}(z)=
w_{\mathrm{acc}}J_{\mathrm{acc}}(z)
+w_{\mathrm{fmt}}J_{\mathrm{format}}(z)
-w_{\mathrm{cost}}J_{\mathrm{cost}}(z)
-w_{\mathrm{lat}}J_{\mathrm{latency}}(z)
$$

### 11.2 Constrained objective

A cleaner formulation is often constrained optimization:

$$
\max_{z\in\mathcal{Z}} J_{\mathrm{acc}}(z)
$$

subject to:

$$
J_{\mathrm{format}}(z)\ge \rho_{\mathrm{format}}
$$

$$
\mathrm{Cost}(z)\le C_{\max}
$$

$$
\mathrm{Latency}(z)\le L_{\max}
$$

$$
\mathrm{InvalidRate}(z)\le \rho_{\mathrm{invalid}}
$$

This is useful when format validity or safety is non-negotiable.

### 11.3 Lagrangian relaxation

The constrained problem can be relaxed:

$$
\mathcal{L}(z,\lambda)=J_{\mathrm{acc}}(z)
-\lambda_1\max(0,\rho_{\mathrm{format}}-J_{\mathrm{format}}(z))
-\lambda_2\max(0,\mathrm{Cost}(z)-C_{\max})
-\lambda_3\max(0,\mathrm{InvalidRate}(z)-\rho_{\mathrm{invalid}})
$$

Then the acquisition function can use predicted Lagrangian utility:

$$
a(z)=\mathbb{E}[\mathcal{L}(z,\lambda)\mid H_t]+\kappa\sqrt{\mathrm{Var}[\mathcal{L}(z,\lambda)\mid H_t]}
$$

---

## 12. Budget accounting and sample efficiency

### 12.1 Rollout cost

One rollout is one execution of the system on one task example.

If candidate $z_t$ is evaluated on minibatch $B_t$:

$$
\mathrm{Rollouts}(z_t,B_t)=|B_t|
$$

Total rollout budget:

$$
\sum_{t=1}^{T}|B_t|\le B
$$

If repeated evaluation is used for top candidates:

$$
\sum_{t=1}^{T}\sum_{k=1}^{K_t}|B_{t,k}|\le B
$$

### 12.2 Token and money budget

Rollouts may have variable cost. Define:

$$
c_t = c_{\mathrm{in}}\cdot \mathrm{InputTokens}_t + c_{\mathrm{out}}\cdot \mathrm{OutputTokens}_t + c_{\mathrm{tool}}\cdot \mathrm{ToolCalls}_t
$$

Total cost constraint:

$$
\sum_{t=1}^{T}c_t\le C_{\mathrm{total}}
$$

BESO can therefore optimize under either rollout budget, token budget, monetary budget, or wall-clock budget.

### 12.3 Optimization curve

Let best validation score after $b$ rollouts be:

$$
V(b)=\max_{z\in A(b)}\hat{J}_{\mathrm{val}}(z)
$$

where $A(b)$ is the archive after spending $b$ rollouts.

Sample efficiency can be measured by area under the optimization curve:

$$
\mathrm{AUC}_{B}=\frac{1}{B}\int_{0}^{B}V(b)\,db
$$

In discrete form:

$$
\mathrm{AUC}_{B}\approx\frac{1}{B}\sum_{t=1}^{T}V(b_t)(b_t-b_{t-1})
$$

BESO's central empirical claim should be about this curve, not only the final score.

### 12.4 Rollouts-to-threshold

Given a target score $\gamma$:

$$
\tau_{\gamma}=\min\{b:V(b)\ge \gamma\}
$$

If BESO is sample-efficient, it should have lower $\tau_{\gamma}$ than baselines.

---

## 13. Generalization and transfer

### 13.1 Train-test gap

Optimization score:

$$
\hat{J}_{\mathrm{opt}}(z)
$$

Final test score:

$$
\hat{J}_{\mathrm{test}}(z)
$$

Generalization gap:

$$
G(z)=\hat{J}_{\mathrm{opt}}(z)-\hat{J}_{\mathrm{test}}(z)
$$

Large positive $G(z)$ indicates overfitting.

### 13.2 Skill specificity penalty

Because skills are text, overfitting often appears as overly specific rules. Define a specificity score:

$$
\mathrm{Spec}(z) \in [0,1]
$$

estimated by structural or LLM-labeled features.

A regularized objective:

$$
J_{\mathrm{reg}}(z)=J(z)-\lambda_{\mathrm{spec}}\mathrm{Spec}(z)-\lambda_{\mathrm{len}}\mathrm{Length}(z)
$$

This encourages general rules rather than benchmark memorization.

### 13.3 Transfer across models or tasks

Let $\mathcal{T}_a$ be the source task distribution and $\mathcal{T}_b$ be the target distribution.

Source-optimized skill:

$$
z_a^*\in\arg\max_z J_{\mathcal{T}_a}(z)
$$

Transfer utility:

$$
\mathrm{Transfer}(z_a^*;\mathcal{T}_b)=J_{\mathcal{T}_b}(z_a^*)-J_{\mathcal{T}_b}(z_0)
$$

For cross-model transfer, compare:

$$
\mathrm{TransferModel}(z^*;\Theta_a,\Theta_b)=J_{\Theta_b}(z^*)-J_{\Theta_b}(z_0)
$$

The research question is whether skill artifacts encode reusable behavior or only model-specific prompt hacks.

---

## 14. Relation to GEPA-style prompt evolution

GEPA-style reflective prompt evolution can be abstracted as:

$$
p' = R(p,\tau,s,f)
$$

where $p$ is a prompt and $R$ is a reflection-based prompt updater.

BESO shifts the object from prompt $p$ to skill artifact $z$:

$$
z' = R_z(z,\tau,s,f)
$$

and adds Bayesian selection:

$$
z_{t+1}\in\arg\max_{z\in\mathcal{C}_t}a_t(z)
$$

So the key distinction is:

$$
\text{GEPA-like: reflect } \rightarrow \text{ mutate prompt } \rightarrow \text{ evaluate}
$$

$$
\text{BESO: reflect } \rightarrow \text{ generate skill edits } \rightarrow \text{ predict utility/uncertainty } \rightarrow \text{ selectively evaluate}
$$

BESO's proposed advantage is not that it can generate better edits automatically. The reflection model may generate the same candidate edits as a GEPA-like system. The advantage is that BESO spends the rollout budget more selectively.

---

## 15. Relation to bandits

If each edit type is treated as an arm:

$$
a \in \mathcal{A}=\{\mathrm{add\_rule},\mathrm{replace\_rule},\mathrm{compress},\dots\}
$$

then BESO could be reduced to a contextual bandit:

$$
\Pr(a_t\mid \mathrm{context}_t)
$$

with reward:

$$
r_t=\hat{J}(z_{t+1})-\hat{J}(z_t)
$$

But BESO is richer than a simple bandit because:

- actions are not only edit types; they are concrete semantic edits,
- the candidate space changes every iteration,
- candidates have text embeddings and lineage,
- evaluation can be multi-objective,
- archive diversity matters.

A contextual bandit baseline is still useful:

$$
a_t = \arg\max_a \mathrm{UCB}_t(a,\mathrm{context}_t)
$$

But BESO should outperform it if candidate-level semantic features matter.

---

## 16. Full BESO objective as nested optimization

BESO can be written as a nested optimization process.

The outer goal:

$$
z^* = \arg\max_{z\in\mathcal{Z}}J(z)
$$

The optimizer cannot search $\mathcal{Z}$ directly. At each iteration it constructs a local candidate set:

$$
\mathcal{C}_t = \mathcal{G}_{\psi}(A_t,H_t,R_t)
$$

where $\mathcal{G}_{\psi}$ is reflection-guided candidate generation.

Then it selects candidates by acquisition:

$$
S_t = \operatorname{TopK}_{z\in\mathcal{C}_t} a_t(z)
$$

Then it evaluates:

$$
\tilde{y}_{t,z}=\hat{J}_{B_t}(z)+\epsilon_{t,z}, \quad z\in S_t
$$

Then it updates:

$$
H_{t+1}=H_t\cup\{(z,B_t,\tilde{y}_{t,z},\mathcal{R}_{t,z},c_{t,z}):z\in S_t\}
$$

$$
A_{t+1}=\operatorname{ArchiveUpdate}(A_t,S_t,H_{t+1})
$$

$$
p(f\mid H_{t+1})=\operatorname{SurrogateUpdate}(p(f\mid H_t),H_{t+1})
$$

The final output is:

$$
z_{\mathrm{final}}=\arg\max_{z\in A_T}\hat{J}_{D_{\mathrm{val}}}(z)
$$

and the final reported score is:

$$
\hat{J}_{D_{\mathrm{test}}}(z_{\mathrm{final}})
$$

---

## 17. Algorithm in mathematical pseudocode

Input:

$$
D, z_0, B, M, K, \Theta_{\mathrm{frozen}}, C, \mu
$$

Initialize:

$$
A_0=\{z_0\}, \quad H_0=\emptyset, \quad R_0=\emptyset
$$

Evaluate seed skill:

$$
\tilde{y}_0,\mathcal{R}_0=\operatorname{Eval}(z_0,D_{\mathrm{val}})
$$

$$
H_0\leftarrow H_0\cup\{(z_0,D_{\mathrm{val}},\tilde{y}_0,\mathcal{R}_0)
\}
$$

For $t=0,1,\dots,T-1$ while budget remains:

1. Select parents:

$$
P_t\sim \pi_{\mathrm{parent}}(\cdot\mid A_t,H_t)
$$

2. Generate candidate edits:

$$
\mathcal{C}_t=\bigcup_{z_p\in P_t}\{e(z_p):e\sim Q_{\psi}(\cdot\mid z_p,H_t,R_t),\nu(z_p,e)=1\}
$$

3. Featurize candidates:

$$
X_t=\{\varphi(z):z\in\mathcal{C}_t\}
$$

4. Fit/update surrogate:

$$
p_t(f)=p(f\mid H_t)
$$

5. Score candidates by acquisition:

$$
a_t(z)=\mu_t(z)+\kappa_t\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)
$$

6. Select batch:

$$
S_t=\operatorname{TopK}_{z\in\mathcal{C}_t}(a_t(z),K)
$$

7. Evaluate selected candidates:

$$
(\tilde{y}_z,\mathcal{R}_z,c_z)=\operatorname{Eval}(z,B_t), \quad z\in S_t
$$

8. Validate candidates:

$$
\mathrm{accept}_z=\operatorname{Gate}(z,p(z),D_{\mathrm{val}})
$$

9. Update archive:

$$
A_{t+1}=\operatorname{Prune}\left(A_t\cup\{z\in S_t:\mathrm{accept}_z=1\}\right)
$$

10. Update rejection buffer:

$$
R_{t+1}=R_t\cup\{z\in S_t:\mathrm{accept}_z=0\}
$$

Return:

$$
z_{\mathrm{final}}=\arg\max_{z\in A_T}\hat{J}_{D_{\mathrm{val}}}(z)
$$

Report:

$$
\hat{J}_{D_{\mathrm{test}}}(z_{\mathrm{final}})
$$

---

## 18. What exactly is Bayesian in BESO?

BESO is Bayesian in the candidate-selection layer, not necessarily in the text-generation layer.

The reflection model gives proposals:

$$
\mathcal{C}_t \sim Q_{\psi}(\cdot\mid H_t,A_t,R_t)
$$

The Bayesian layer asks:

$$
\text{Given previous evaluations, what do we believe about the utility of each candidate?}
$$

That belief is:

$$
p(f\mid H_t)
$$

The acquisition function converts belief into action:

$$
a_t(z)=\text{value of evaluating }z\text{ next}
$$

So BESO's Bayesian claim should be phrased carefully:

> BESO uses a probabilistic surrogate to guide which reflection-generated skill variants should be evaluated under a limited rollout budget.

It should not claim that the LLM reflection process itself is Bayesian unless that is explicitly modeled.

---

## 19. Key research hypotheses in mathematical form

### H1: Skill optimization beats prompt optimization

Let $\mathcal{Z}_{\mathrm{skill}}$ be skill artifacts and $\mathcal{Z}_{\mathrm{prompt}}$ be raw prompts.

Hypothesis:

$$
\mathbb{E}_{s}\left[\max_{z\in\mathcal{Z}_{\mathrm{skill}},\; c\le B} \hat{J}_{\mathrm{test}}(z)\right]
>
\mathbb{E}_{s}\left[\max_{p\in\mathcal{Z}_{\mathrm{prompt}},\; c\le B} \hat{J}_{\mathrm{test}}(p)\right]
$$

where expectation is over random seeds, datasets, and optimizer stochasticity.

### H2: Bayesian acquisition improves sample efficiency

Let $V_{\mathrm{BESO}}(b)$ be the best validation score after $b$ rollouts.

Let $V_{\mathrm{baseline}}(b)$ be the corresponding curve for random, greedy, or Pareto-only selection.

Hypothesis:

$$
\mathrm{AUC}_B(V_{\mathrm{BESO}})>
\mathrm{AUC}_B(V_{\mathrm{baseline}})
$$

and:

$$
\tau_{\gamma}^{\mathrm{BESO}}<\tau_{\gamma}^{\mathrm{baseline}}
$$

for meaningful thresholds $\gamma$.

### H3: Reflection improves candidate proposal quality

Let $\mathcal{C}_t^{\mathrm{reflect}}$ be reflection-generated candidates and $\mathcal{C}_t^{\mathrm{random}}$ be randomly mutated candidates.

Hypothesis:

$$
\mathbb{E}\left[\max_{z\in\mathcal{C}_t^{\mathrm{reflect}}}J(z)\right]
>
\mathbb{E}\left[\max_{z\in\mathcal{C}_t^{\mathrm{random}}}J(z)\right]
$$

### H4: Pareto archive improves robustness

Let $A_t^{\mathrm{pareto}}$ be archive with diversity / Pareto preservation and $A_t^{\mathrm{best}}$ be best-only archive.

Hypothesis:

$$
\hat{J}_{\mathrm{hard}}(z_{\mathrm{final}}^{\mathrm{pareto}})
>
\hat{J}_{\mathrm{hard}}(z_{\mathrm{final}}^{\mathrm{best}})
$$

where $D_{\mathrm{hard}}$ is a hard or rare-example subset.

### H5: Optimized skills transfer

For source task distribution $\mathcal{T}_a$ and target distribution $\mathcal{T}_b$:

$$
J_{\mathcal{T}_b}(z_a^*)-J_{\mathcal{T}_b}(z_0)>0
$$

This says the source-optimized skill remains useful on nearby target tasks.

---

## 20. Minimal v0 mathematical design

A clean first prototype should use the simplest version of the math.

### 20.1 Skill space

One skill document:

$$
z\in\mathcal{Z}_{\mathrm{skill}}
$$

Bounded single-section edits:

$$
|\Delta z|\le L_{\max}
$$

### 20.2 Objective

Single scalar metric:

$$
J(z)=\mathbb{E}_{(x,m)\sim\mathcal{T}}[\mu(\Phi(x;C(z),\Theta),m)]
$$

Empirical minibatch estimate:

$$
\hat{J}_{B_t}(z)=\frac{1}{|B_t|}\sum_{i\in B_t}r_i(z)
$$

### 20.3 Surrogate

Use ensemble mean and disagreement:

$$
\mu_t(z)=\frac{1}{M}\sum_{m=1}^{M}g_m(\varphi(z))
$$

$$
\sigma_t(z)=\sqrt{\frac{1}{M-1}\sum_{m=1}^{M}(g_m(\varphi(z))-\mu_t(z))^2}
$$

### 20.4 Acquisition

Use UCB with cost penalty:

$$
a_t(z)=\mu_t(z)+\kappa\sigma_t(z)-\alpha\mathrm{Length}(z)
$$

### 20.5 Acceptance

Use validation gate:

$$
\hat{J}_{\mathrm{val}}(z)>\hat{J}_{\mathrm{val}}(p(z))+\delta
$$

with schema validity:

$$
z\in\mathcal{Z}
$$

### 20.6 Final report

Report:

$$
\hat{J}_{\mathrm{test}}(z_{\mathrm{final}})
$$

plus optimization curve:

$$
V(b)=\max_{z\in A(b)}\hat{J}_{\mathrm{val}}(z)
$$

This is enough to test the core claim without overbuilding the system.

---

## 21. Where the mathematical novelty lives

The research should not be framed as "Bayesian optimization for prompts". That is too broad.

The sharper mathematical object is:

$$
\text{Bayesian optimization over reflection-generated neighborhoods of structured skill artifacts.}
$$

More explicitly:

1. BESO does not optimize all text directly.

$$
\mathcal{Z} \ne \Sigma^*
$$

It optimizes constrained skill artifacts:

$$
\mathcal{Z}=\{z:\mathrm{SchemaValid}(z),\mathrm{InvariantValid}(z),\mathrm{BudgetValid}(z)\}
$$

2. BESO does not require the Bayesian surrogate to generate language.

$$
\mathcal{C}_t = G_{\psi}(A_t,H_t,R_t)
$$

Reflection proposes. Bayesian acquisition selects.

3. BESO performs local Bayesian optimization over candidate neighborhoods:

$$
z_{t+1}\in\arg\max_{z\in\mathcal{C}_t}a_t(z)
$$

4. BESO maintains an evolutionary archive rather than a single incumbent:

$$
A_t = A_t^{\mathrm{best}}\cup A_t^{\mathrm{pareto}}\cup A_t^{\mathrm{diverse}}\cup A_t^{\mathrm{failed}}
$$

5. The system is evaluated by sample-efficiency curves, not just final performance:

$$
\mathrm{AUC}_B(V_{\mathrm{BESO}}) > \mathrm{AUC}_B(V_{\mathrm{baseline}})
$$

That combination is the real research contribution.

---

## 22. Final compact formulation

BESO can be summarized as the following constrained stochastic optimization problem:

$$
\begin{aligned}
&\underset{z\in\mathcal{Z}}{\text{maximize}}
&& J(z)=\mathbb{E}_{(x,m)\sim\mathcal{T}}\left[\mu\left(\Phi_{\Theta_{\mathrm{frozen}}}(x;C(z,x,q)),m\right)\right] \\
&\text{subject to}
&& \sum_{t=1}^{T}|B_t|\le B, \\
&&& \mathrm{SchemaValid}(z)=1, \\
&&& \mathrm{InvariantValid}(z)=1, \\
&&& \mathrm{Tokens}(z)\le T_{\max}, \\
&&& \mathrm{Cost}(z)\le C_{\max}. \\
\end{aligned}
$$

Because $J$ is unknown, noisy, expensive, and defined over structured text, BESO uses the iterative approximation:

$$
\mathcal{C}_t=G_{\psi}(A_t,H_t,R_t)
$$

$$
p_t(f)=p(f\mid H_t)
$$

$$
z_{t+1}=\arg\max_{z\in\mathcal{C}_t}\left[\mu_t(z)+\kappa_t\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)\right]
$$

$$
H_{t+1}=H_t\cup\operatorname{Eval}(z_{t+1})
$$

$$
A_{t+1}=\operatorname{ArchiveUpdate}(A_t,z_{t+1},H_{t+1})
$$

Final selection:

$$
z_{\mathrm{final}}=\arg\max_{z\in A_T}\hat{J}_{D_{\mathrm{val}}}(z)
$$

Final report:

$$
\hat{J}_{D_{\mathrm{test}}}(z_{\mathrm{final}})
$$

In plain language:

> BESO treats a skill document as trainable external state. Reflection proposes possible semantic edits. A Bayesian surrogate estimates which edits are promising or informative. An acquisition function spends the rollout budget. A Pareto-diverse archive keeps useful variants alive. The final skill is selected by validation and judged on untouched test data.
````

## File: docs/Bayesian Evolutionary Skill Optimization (BESO) - Methodology.md
````markdown
---
type: research-note
tags: [beso, methodology, bayesian-optimization, evolutionary-search, skill-optimization, llm-agents]
date: 2026-05-29
source:
  - [[Bayesian Evolutionary Skill Optimization (BESO) - Technical Specification]]
  - [[Bayesian Evolutionary Skill Optimization (BESO) - Mathematical Breakdown]]
  - [[Bayesian Evolutionary Skill Optimization (BESO) - GEPA SkillOpt BESO Mathematical Lineage]]
status: draft
---

# Bayesian Evolutionary Skill Optimization (BESO): methodology

## 0. Purpose of this document

This note specifies the **operational methodology** for the BESO v0 implementation.
It translates the mathematical breakdown, the technical specification, and the
GEPA → SkillOpt → BESO lineage into a concrete, reproducible procedure, and it
records the **design decisions** that resolve the loose assumptions and edge
cases identified during the mathematical review.

Where the breakdown answers *what* the objects are and the specification answers
*what* the system should contain, this note answers *how* we run the optimizer,
*which* defaults we commit to for v0, and *why* those choices are statistically
defensible.

---

## 1. Problem restatement

BESO optimizes a structured natural-language skill artifact $z$ for a frozen LLM
system $\Phi$:

$$
z^* \in \arg\max_{z \in \mathcal{Z}} \; J(z)
= \arg\max_{z \in \mathcal{Z}} \;
\mathbb{E}_{(x,m)\sim\mathcal{T}}\left[\mu\left(\Phi(x; C(z), \Theta_{\mathrm{frozen}}), m\right)\right]
$$

subject to a rollout budget:

$$
\sum_{t=1}^{T} c(z_t, B_t) \le B
$$

The methodology treats this as a **budgeted, noisy, black-box, discrete**
optimization problem solved by Bayesian-guided evolutionary search over a
finite, reflection-generated candidate set at each iteration.

The model weights $\Theta_{\mathrm{frozen}}$ never change. The only trainable
state is the skill artifact $z$, deployed as a standalone document.

---

## 2. Methodological pillars

| Pillar | Role | Inherited from |
| --- | --- | --- |
| Skill-level abstraction | Optimize a typed, sectioned, linted artifact, not a raw prompt string | SkillOpt |
| Trajectory-grounded reflection | Convert *why* a rollout failed into bounded natural-language edits | GEPA |
| Bayesian experiment planning | Predict candidate utility and uncertainty, then spend rollouts where they are most informative | BESO (new) |
| Evolutionary archive | Preserve high performers, specialists, diverse variants, and informative failures | GEPA + SkillOpt |

BESO's distinct contribution is the layer **between proposal and evaluation**: a
surrogate $p(f\mid H_t)$ plus an acquisition function $a_t(z)$ that allocates a
limited rollout budget across a candidate pool $\mathcal{C}_t$.

---

## 3. Data regime

The dataset is split into four disjoint roles (Breakdown S1.2; Spec S16.1):

$$
D = D_{\mathrm{fb}} \cup D_{\mathrm{opt}} \cup D_{\mathrm{val}} \cup D_{\mathrm{test}}
$$

- $D_{\mathrm{fb}}$ — feedback/train: generates trajectories for reflection.
- $D_{\mathrm{opt}}$ — optimization minibatches $B_t$: fast, noisy candidate estimates.
- $D_{\mathrm{val}}$ — validation gate: accept/reject decisions.
- $D_{\mathrm{test}}$ — final test: untouched until reporting.

**Recommended default split** (small datasets): 40 / 20 / 20 / 20.
For benchmarks with a fixed test set, draw $D_{\mathrm{fb}}, D_{\mathrm{opt}}, D_{\mathrm{val}}$
from train and reserve the official test set for $D_{\mathrm{test}}$.

**Methodological guard (anti-overfitting).** Because text artifacts can encode
benchmark quirks, the validation gate is the only place acceptance is decided,
and the test split is never consulted during optimization. To limit progressive
overfitting to a fixed $D_{\mathrm{val}}$ (see S7), we rotate / periodically
refresh the validation subset and re-validate the incumbent.

---

## 4. The optimization loop

### 4.1 High-level procedure

```text
1.  Initialize and lint skill z_0; assert z_0 in Z.
2.  Evaluate z_0 on a validation seed; log trajectories, scores, feedback.
3.  Seed history H, archive A, rejected-edit buffer R.
repeat until budget B exhausted:
4.    Select parents P_t from A (Pareto win-count + UCB-style score).
5.    Reflect over traces to propose bounded edits; apply -> candidate pool C_t.
6.    Hard-filter infeasible candidates (lint / schema / budget / invariants).
7.    Deduplicate C_t.
8.    Featurize C_t (block-separated phi(z)); center features on the parent.
9.    Fit / update the Bayesian surrogate on H; predict mu_t, sigma_t.
10.   Score candidates with the pool-normalized acquisition a_BESO.
11.   Select an evaluation batch S_t via submodular (max-min / DPP) selection.
12.   Evaluate S_t on optimization minibatches; append observations to H.
13.   Gate survivors on D_val (paired test + multiplicity control + noise-scaled delta).
14.   Update archive A (best / Pareto / diverse / failed); push rejects to R.
15.   Prune archive to the size cap.
return best validated skill on the final validation set.
```

### 4.2 Update recurrence (formal)

$$
P_t \sim \pi_{\mathrm{parent}}(A_t, H_t)
\qquad
\mathcal{C}_t = G_{\psi}(P_t, H_t, R_t, A_t)
$$

$$
p_t(f) = p(f \mid H_t)
\qquad
S_t = \operatorname{Select}_{z\in\mathcal{C}_t}\; a_t(z)
$$

$$
\tilde{y}_z = \hat{J}_{B_t}(z) + \epsilon_z,\quad z\in S_t
\qquad
A_{t+1} = \operatorname{ArchiveUpdate}(A_t, S_t, H_{t+1})
$$

---

## 5. Candidate generation (reflection)

The reflection module is a proposal distribution over bounded edits:

$$
e_{t,j} \sim Q_{\psi}(e \mid z_p, \mathcal{R}, F, A_t, H_t, R_t),
\qquad
\mathcal{C}_t = \{e_{t,j}(z_p) : \nu(z_p, e_{t,j}) = 1\}
$$

- Reflection consumes parent skill, successful and failed trajectories,
  evaluator feedback, accepted-edit history, and the rejected-edit buffer.
- Output is **structured JSON** (diagnosis, failure modes, proposed edits with
  rationale and risk); malformed or budget-violating proposals are discarded.
- **Bounded edits act as a textual learning rate** $\Delta(z, z') \le \eta_{\mathrm{text}}$,
  enforced through per-iteration token/section caps. This prevents destructive
  rewrites and keeps the candidate distribution closer to stationary.

The Bayesian layer never generates text; it only ranks the finite pool that
reflection produces. BESO's ceiling is therefore bounded by reflection quality
(see S9 regime detector).

---

## 6. Surrogate and featurization (committed v0 design)

### 6.1 Featurization $\varphi(z)$

Block-decomposed and **kept separate until normalization**:

$$
\varphi(z) = \left[
\varphi_{\mathrm{text}}(z),\;
\varphi_{\mathrm{struct}}(z),\;
\varphi_{\mathrm{edit}}(z),\;
\varphi_{\mathrm{hist}}(z),\;
\varphi_{\mathrm{sem}}(z)
\right]
$$

Committed treatment:

- **Per-block standardization** to unit variance before assembly, so the
  high-dimensional text block cannot swamp the cheap structured signals.
- **Text block reduced** (PCA / random projection, ~16–32 dims) fit on the
  archive; reweighted by a block weight mirroring the composite-kernel weights
  $\alpha,\beta,\gamma$.
- **Parent-centered (delta) features**: structural features encode
  $\Delta\text{tokens}, \Delta\text{rules}, \Delta\text{checklist}$ relative to
  the parent, improving stationarity under a drifting proposal distribution.
- **Cold-start defaults** for undefined history features (e.g. edit-type success
  rate) at small $t$.

### 6.2 Surrogate

Target the **parent-relative improvement** (Breakdown S7.5):

$$
\Delta(z, z_p) = J(z) - J(z_p),
\qquad
\mu_t^{\mathrm{hybrid}}(z) = \mu_t^{\mathrm{abs}}(z) + \rho\,\mu_t^{\Delta}(z, z_p)
$$

Model: a **homogeneous bootstrap-bagged ensemble** over the reduced vector
(clean resampling-induced epistemic variance), with the composite-kernel GP path
available for very small $t$.

**Total predictive variance** (not raw disagreement):

$$
\sigma_t^2(z)
= \underbrace{\tfrac{1}{M-1}\sum_{m}\big(\hat{f}_m(z)-\mu_t(z)\big)^2}_{\text{epistemic}}
+ \underbrace{\hat{\sigma}_{\epsilon}^2(z, B)}_{\text{aleatoric}},
\qquad
\hat{\sigma}_{\epsilon}^2(z, B) \approx \frac{\hat{\sigma}^2(z)}{|B|}
$$

Predicted intervals are **recalibrated** on a held-out calibration slice
(isotonic on the PIT histogram or a single temperature scalar). The aleatoric
term ties to the noise decomposition
$\sigma_{\epsilon}^2 = \sigma_{\mathrm{model}}^2 + \sigma_{\mathrm{judge}}^2 + \sigma_{\mathrm{batch}}^2 + \sigma_{\mathrm{tool}}^2$.

---

## 7. Acquisition and batch selection (committed v0 design)

### 7.1 Pool-normalized acquisition

$$
a_{\mathrm{BESO}}(z)
= \tilde{\mu}_t(z)
+ \kappa\,\tilde{\sigma}_t(z)
+ \lambda\,\tilde{d}(z, A_t)
- \alpha\,\tilde{c}(z)
- \gamma\,\tilde{q}_{\mathrm{invalid}}(z)
$$

where each term $\tilde{(\cdot)}$ is **normalized over the current pool
$\mathcal{C}_t$** (z-score or min–max) so the weights $\kappa,\lambda,\alpha,\gamma$
are dimensionless and comparable across iterations. This removes the unit
mismatch between $\mu\in[0,1]$, distances, token costs, and probabilities.

### 7.2 Decisions

- **Exploration term decoupling.** $\sigma_t$ carries posterior uncertainty;
  the diversity term $d(z, A_t)$ is restricted to accepted/archive members and
  acts as anti-redundancy only, to avoid double-counting novelty.
- **Submodular batch selection.** For batch size $K$, select greedily by
  max-min / facility-location (or a DPP) and **update the reference set with
  already-selected members during the greedy pass**, preventing intra-batch
  near-duplicates.
- **Exploration schedule.** Non-decaying (or calibration-error-adaptive)
  $\kappa$, because the candidate domain $\mathcal{C}_t$ is regenerated each
  round; a decaying schedule can prematurely kill exploration when reflection
  opens a new region.
- **Deterministic vs probabilistic infeasibility.** Schema / budget / invariant
  violations are hard-filtered **before** acquisition (loop step 6); the learned
  penalty $\gamma\,\tilde{q}_{\mathrm{invalid}}$ targets only residual,
  non-deterministic risk (e.g. format drift).

---

## 8. Acceptance, gating, and rejection (committed v0 design)

### 8.1 Paired, noise-aware gate

Acceptance compares candidate and parent on the **same** validation draw, using
paired per-example differences $d_i = r_i(z) - r_i(p)$:

$$
\bar{d} = \frac{1}{|D_{\mathrm{val}}|}\sum_{i} d_i,
\qquad
\mathrm{CI}_{1-\alpha}(\bar{d}) = [L, U]
$$

Accept if the lower bound clears a **noise-scaled threshold**:

$$
\operatorname{Accept}(z) = 1
\iff
L > 0
\quad\text{and}\quad
\bar{d} \ge c \cdot \widehat{\mathrm{SE}}(\bar{d})
$$

For binary metrics with small $|D_{\mathrm{val}}|$, use an exact paired test
(McNemar) instead of a bootstrap CI.

### 8.2 Multiplicity control

Because each iteration tests $K$ candidates against the same split across many
iterations, raw thresholds inflate false acceptances (winner's curse). We apply
**Benjamini–Hochberg** correction across each round's candidate tests and
**periodically re-validate the incumbent** to avoid a lucky-but-false acceptance
permanently raising the bar.

### 8.3 Constraint gates

A candidate must also satisfy, on a **one-sided confidence bound** (not the noisy
point estimate):

$$
\mathrm{InvalidRate}(z) \le \rho_{\max},
\quad
\mathrm{Cost}(z) \le C_{\max},
\quad
\mathrm{Tokens}(z) \le T_{\max},
\quad
\mathrm{SchemaValid}(z) = \mathrm{InvariantValid}(z) = 1
$$

### 8.4 Rejections as negative evidence

Rejected edits enter the buffer $R_t$ with a reason and condition future
reflection. They may also contribute a similarity penalty to acquisition.

---

## 9. Archive management

The archive preserves four tiers (Breakdown S4.2):

$$
A_t = A_t^{\mathrm{best}} \cup A_t^{\mathrm{pareto}} \cup A_t^{\mathrm{diverse}} \cup A_t^{\mathrm{failed}}
$$

- **best** — high validation mean.
- **pareto** — instance- or objective-level specialists; win count
  $w(z_k) = \sum_i \mathbb{1}[z_k \in W_i]$.
- **diverse** — semantically distinct skills.
- **failed** — informative failures.

Parent selection mixes validation score, Pareto win count, diversity, and a cost
penalty (Breakdown S4.3). Pruning is constrained subset selection that trades
off validation value, coverage, diversity, and Pareto value, capped at
`max_archive_size`.

### 9.1 Regime detector (built-in ablation)

An online detector monitors candidate-score variance $\operatorname{Var}_{z\in\mathcal{C}_t}[J(z)]$
and surrogate calibration / rank-correlation. When the surrogate is not yet
predictive (cold start) or candidate variance is negligible, BESO **auto-falls
back** to greedy or random selection, so the Bayesian layer never adds overhead
without gain. This doubles as the "no surrogate" ablation.

---

## 10. Evaluation protocol and metrics

- **Decoding** defaults to temperature 0 for the target model; repeated
  evaluation is enabled for top candidates to shrink $\widehat{\mathrm{SE}}$.
- **Primary metric** is task-dependent (exact accuracy, pass@1, macro-F1,
  field-level F1, schema validity + content accuracy).
- **Secondary metrics**: token cost, latency, invalid-output rate, tool-call
  count, verbosity, calibration behavior, robustness on hard examples.
- **Primary curve**: optimization curve

$$
V(b) = \max_{z\in A(b)} \hat{J}_{\mathrm{val}}(z),
\qquad
\mathrm{AUC}_B = \frac{1}{B}\int_0^B V(b)\,db
$$

plus the final untouched-test score $\hat{J}_{\mathrm{test}}(z_{\mathrm{final}})$.

---

## 11. Scientific hypotheses

BESO's primary claim is **sample efficiency**, not merely a good final skill:

$$
\mathrm{AUC}_B(V_{\mathrm{BESO}}) > \mathrm{AUC}_B(V_{\mathrm{SkillOpt}}),
\qquad
\tau_{\gamma}^{\mathrm{BESO}} < \tau_{\gamma}^{\mathrm{SkillOpt}},
\quad
\tau_{\gamma} = \min\{b : V(b) \ge \gamma\}
$$

BESO is expected to win when rollouts are expensive, reflection produces many
plausible edits ($|\mathcal{C}_t| \gg K$), candidate quality varies, features
carry signal ($I(\varphi(z); J(z)) > 0$), and uncertainty-aware exploration
matters.

---

## 12. Baselines and ablations

**Baselines:** no skill; human skill; one-shot LLM skill; GEPA-style prompt
evolution; SkillOpt-style bounded editing; random candidate selection; greedy
reflection-ranked selection; bandit over edit types.

**BESO ablations:** no surrogate; mean-only acquisition (drop $\sigma$); no
diversity term; no cost penalty; embeddings-only vs structured-only features;
absolute-score vs delta model; no rejected-buffer features.

---

## 13. v0 commitments (summary)

| Decision | v0 default | Rationale |
| --- | --- | --- |
| Optimization target | Level-3 skill artifact | Reusable, modular, gateable |
| Surrogate target | Parent-relative $\Delta(z, z_p)$ hybrid | Stationarity under proposal drift |
| Surrogate model | Homogeneous bootstrap-bagged ensemble | Clean epistemic variance |
| Uncertainty | Epistemic + aleatoric, recalibrated | Calibrated $\sigma_t$ for acquisition |
| Featurization | Block-separated, standardized, parent-centered, reduced text | No scale domination |
| Acquisition | Pool-normalized $a_{\mathrm{BESO}}$ (UCB + diversity − cost − invalid) | Dimensionless, interpretable weights |
| Batch selection | Submodular max-min / DPP with in-batch updates | No intra-batch duplicates |
| Exploration schedule | Non-decaying / adaptive $\kappa$ | Non-stationary candidate domain |
| Gate | Paired test + BH multiplicity + noise-scaled $\delta$ | Controls winner's curse |
| Constraints | One-sided confidence bounds | Noisy constraint estimates |
| Fallback | Regime detector auto-disables surrogate | Avoids overhead without gain |

---

## 14. Acceptance criteria (v0)

1. End-to-end loop runs without manual intervention.
2. Skill artifacts remain schema-valid after edits.
3. Candidate evaluations are logged with trajectories and scores.
4. Surrogate selects candidates via acquisition.
5. Final skill improves over the initial skill on validation.
6. Final test result is reported on the untouched test split.
7. At least one baseline comparison is included.
8. Optimization trace is inspectable.

Minimum success: `BESO > initial skill` and `BESO > random edit search`.
Stronger success: `BESO >= SkillOpt-style editing` and
`BESO >= GEPA-style prompt evolution` under the same rollout budget.
````

## File: docs/Bayesian Evolutionary Skill Optimization (BESO) - Technical Specification.md
````markdown
## Technical Specification

## 1. Executive Summary

**Bayesian Evolutionary Skill Optimization (BESO)** is a proposed optimization framework for improving frozen LLM-based systems by treating natural-language **skills** as trainable external artifacts.

Instead of directly optimizing model weights, BESO optimizes a structured skill document that encodes reusable behavioral rules, procedures, tool-use policies, verification habits, and failure-avoidance strategies. The skill is then compiled into prompts or injected into an agent runtime.

BESO combines four ideas:

1. **Skill-level abstraction**: optimize reusable skill artifacts rather than brittle raw prompts.
    
2. **Reflection-based mutation**: use trajectory feedback to propose meaningful natural-language edits.
    
3. **Bayesian experiment planning**: use a surrogate model to decide which candidate edits are worth evaluating under a limited rollout budget.
    
4. **Evolutionary archive management**: preserve high-performing and specialized candidates instead of greedily keeping only the single best artifact.
    

In short:

> BESO is a sample-efficient optimizer for frozen LLM systems that evolves natural-language skills using reflection, evaluates candidates under a rollout budget, and uses Bayesian acquisition to choose the most promising or informative candidates to test next.

---

## 2. Motivation

LLM behavior is highly sensitive to instructions. Many downstream failures are not caused by a lack of model capability, but by weak task framing, missing procedural rules, poor tool-use policies, vague output constraints, or absent verification steps.

Traditional improvement paths include:

- manual prompt engineering,
    
- automatic prompt optimization,
    
- reinforcement learning,
    
- fine-tuning,
    
- textual-gradient-style optimization,
    
- evolutionary prompt search,
    
- skill-document optimization.
    

Each has trade-offs.

Manual prompt engineering is interpretable but slow and unscalable. Reinforcement learning and fine-tuning can be powerful but expensive, opaque, and often unavailable for closed-source models. Evolutionary prompt optimization can be flexible but may waste rollout budget by evaluating too many weak candidates. Skill-document optimization is more stable than raw prompt editing, but candidate selection can still be local or greedy.

BESO addresses this by asking:

> Can we use Bayesian optimization to make natural-language skill evolution more sample-efficient?

The key hypothesis is that a Bayesian surrogate can learn from previous skill evaluations and guide future candidate selection toward edits that are either likely to improve performance or likely to reveal useful information.

---

## 3. Core Research Question

**Primary question:**

> Can Bayesian-guided evolutionary optimization of structured skill documents outperform prompt-level evolution and gradient-like skill editing under low rollout budgets?

**Secondary questions:**

1. Are skill documents a better optimization target than raw prompts?
    
2. Does Bayesian candidate selection improve sample efficiency compared with random, greedy, bandit, or Pareto-only selection?
    
3. Which representation of text artifacts works best for the Bayesian surrogate?
    
4. Does the optimized skill transfer across models, tasks, and execution harnesses?
    
5. Does preserving diverse skill variants improve generalization compared with keeping only the highest-average candidate?
    

---

## 4. Design Thesis

BESO is built on the following thesis:

> The best search object for prompt-like optimization is not the final prompt string. It is a structured, reusable skill artifact that can be compiled into runtime prompts.

Raw prompts are brittle because they mix many concerns into one text block: role, task definition, reasoning strategy, tool policy, output format, safety behavior, examples, and failure handling.

A structured skill artifact separates these concerns. This makes editing easier, evaluation more interpretable, and Bayesian modeling more meaningful.

The preferred hierarchy is:

```text
Skill artifact
    ↓ compile / inject
Module prompt(s)
    ↓ execute
LLM system behavior
    ↓ evaluate
Trajectory + score + feedback
    ↓ optimize
Updated skill artifact
```

---

## 5. Scope

### 5.1 In Scope

BESO optimizes:

- natural-language skill documents,
    
- prompt sections,
    
- module-level instruction policies,
    
- failure-mode checklists,
    
- tool-use policies,
    
- reasoning procedures,
    
- output-format rules,
    
- validation or verification habits,
    
- few-shot examples inside the skill artifact.
    

BESO can be applied to:

- single-call LLM tasks,
    
- multi-step reasoning tasks,
    
- tool-use agents,
    
- code-generation agents,
    
- retrieval-augmented generation systems,
    
- structured extraction systems,
    
- classification and judgment systems,
    
- benchmark-solving agents.
    

### 5.2 Out of Scope

The initial version does not optimize:

- model weights,
    
- retrieval indexes,
    
- tool implementations,
    
- benchmark labels,
    
- evaluator prompts unless explicitly configured,
    
- full agent architecture search,
    
- arbitrary code generation for new tools.
    

These may become future extensions.

---

## 6. Key Definitions

### 6.1 Frozen Agent

A frozen agent is an LLM-based system whose underlying model weights are not updated during optimization.

```text
Model weights: fixed
External skill state: trainable
```

### 6.2 Skill Artifact

A skill artifact is a structured natural-language document that encodes reusable behavioral knowledge.

It may contain:

- goal definition,
    
- task scope,
    
- step-by-step procedure,
    
- reasoning strategy,
    
- tool-use policy,
    
- verification checklist,
    
- common failure modes,
    
- recovery rules,
    
- output format,
    
- examples,
    
- constraints.
    

### 6.3 Runtime Prompt

A runtime prompt is the actual prompt passed to the LLM during execution. It may be generated by compiling the skill artifact into a task-specific prompt.

### 6.4 Trajectory

A trajectory is the recorded execution path of the system on one task instance.

It may include:

- input,
    
- selected skill,
    
- compiled prompt,
    
- intermediate model outputs,
    
- reasoning summaries,
    
- tool calls,
    
- tool results,
    
- retrieval context,
    
- final answer,
    
- evaluator score,
    
- textual feedback,
    
- error messages.
    

### 6.5 Rollout

A rollout is one execution of the system on one task example.

### 6.6 Candidate

A candidate is a proposed skill artifact or prompt artifact being evaluated.

### 6.7 Archive

The archive is the stored population of accepted, useful, or diagnostically important candidates.

### 6.8 Surrogate Model

A surrogate model is a learned predictor that estimates candidate performance and uncertainty without fully evaluating every candidate.

### 6.9 Acquisition Function

An acquisition function scores unevaluated candidates based on predicted performance and uncertainty. It determines which candidate should be evaluated next.

---

## 7. Optimization Target

BESO supports three levels of optimization.

### 7.1 Level 1: Raw Prompt Optimization

Artifact:

```text
z = prompt string
```

This is the simplest version. It is useful for single-call tasks but fragile for complex systems.

### 7.2 Level 2: Structured Prompt Section Optimization

Artifact:

```text
z = {
  role,
  task_instruction,
  reasoning_strategy,
  tool_policy,
  output_format,
  constraints,
  examples
}
```

This improves control by allowing targeted edits to specific prompt sections.

### 7.3 Level 3: Skill Artifact Optimization

Artifact:

```text
z = skill document
```

This is the recommended target for the main BESO method.

The skill artifact is reusable and may be compiled into multiple prompts.

### 7.4 Recommended Default

The default BESO configuration should optimize Level 3 artifacts:

> Optimize skill documents first. Compile them into prompts at runtime.

Raw prompt optimization can be included as a baseline or ablation.

---

## 8. Skill Artifact Schema

A default skill artifact should use a structured markdown format.

```markdown
# Skill: <skill_name>

## Goal
Describe what this skill helps the agent accomplish.

## Scope
Define when this skill should and should not be used.

## Core Procedure
1. Step one.
2. Step two.
3. Step three.

## Reasoning Policy
Explain how the agent should reason through the task.

## Tool-Use Policy
Explain when to use tools, when not to use tools, and how to validate tool outputs.

## Verification Checklist
- Check 1.
- Check 2.
- Check 3.

## Common Failure Modes
- Failure mode 1.
- Failure mode 2.
- Failure mode 3.

## Recovery Rules
Explain how to recover from uncertainty, missing evidence, conflicting evidence, or invalid intermediate results.

## Output Rules
Define the final answer format, concision level, citation rules, JSON schema, or other output constraints.

## Examples
Optional few-shot examples.

## Change Log
Track accepted edits and why they were made.
```

### 8.1 Machine-Readable Representation

Internally, the skill can be represented as JSON:

```json
{
  "skill_id": "multi_step_qa_v3",
  "name": "Multi-Step Question Answering",
  "version": 3,
  "sections": {
    "goal": "Answer multi-step questions accurately using evidence.",
    "scope": "Use for questions requiring intermediate facts.",
    "core_procedure": [
      "Identify the main question.",
      "Identify missing intermediate facts.",
      "Resolve intermediate facts before answering.",
      "Verify that the final answer follows from evidence."
    ],
    "reasoning_policy": "Decompose before answering.",
    "tool_use_policy": "Use search or calculator tools when the required fact or calculation is uncertain.",
    "verification_checklist": [
      "Does the answer directly answer the question?",
      "Is each intermediate fact supported?",
      "Are there contradictions?"
    ],
    "common_failure_modes": [
      "Answering from the first retrieved fact only.",
      "Confusing related entities.",
      "Skipping verification."
    ],
    "recovery_rules": [
      "If evidence conflicts, report uncertainty.",
      "If a tool call fails, retry with a narrower query."
    ],
    "output_rules": [
      "Give the final answer first.",
      "Explain only the necessary reasoning."
    ],
    "examples": []
  },
  "metadata": {
    "created_by": "optimizer",
    "parent_id": "multi_step_qa_v2",
    "edit_summary": "Added verification checklist for entity confusion.",
    "token_count": 412
  }
}
```

---

## 9. Mathematical Formulation

### 9.1 System Definition

Let:

```text
x = task input
m = metadata, label, expected answer, rubric, or test case
z = skill artifact
Θ = frozen model weights
Φ = full AI system
μ = scoring function
```

The system output is:

```text
y = Φ(x; z, Θ_frozen)
```

The score is:

```text
μ(Φ(x; z, Θ_frozen), m) ∈ [0, 1]
```

### 9.2 Expected Performance Objective

The true objective is to find the skill artifact that maximizes expected performance across the task distribution:

```text
z* = argmax_z E_(x,m)~T [ μ(Φ(x; z, Θ_frozen), m) ]
```

This means:

> Find the skill artifact that makes the frozen system perform best across many possible examples.

### 9.3 Empirical Objective

Because the true task distribution is unknown, use a dataset:

```text
D = {(x_1, m_1), ..., (x_n, m_n)}
```

Approximate expected performance with empirical average:

```text
J(z) = (1/n) Σ_i μ(Φ(x_i; z, Θ_frozen), m_i)
```

Optimization target:

```text
z* ≈ argmax_z J(z)
```

### 9.4 Rollout Budget Constraint

Each evaluation consumes rollouts.

```text
z* ≈ argmax_z J(z)
subject to #rollouts ≤ B
```

where:

```text
B = maximum rollout budget
```

### 9.5 Noisy Evaluation

In practice, evaluating a candidate on a minibatch gives a noisy estimate:

```text
ŷ_t = J_M(z_t) + ε_t
```

where:

```text
M = minibatch
ε_t = evaluation noise
```

The optimizer therefore maintains an evaluation dataset:

```text
H_t = {(z_1, ŷ_1), ..., (z_t, ŷ_t)}
```

### 9.6 Bayesian Surrogate

BESO learns a surrogate model over artifact performance:

```text
p(f | H_t)
```

where:

```text
f(z) = true but unknown performance function
```

For each candidate z, the surrogate estimates:

```text
μ_t(z) = predicted performance
σ_t(z) = uncertainty
```

### 9.7 Acquisition Function

BESO uses an acquisition function to decide which candidate to evaluate next.

Default acquisition: Upper Confidence Bound.

```text
a(z) = μ_t(z) + κ σ_t(z)
```

where:

```text
κ = exploration weight
```

Candidate selection:

```text
z_(t+1) = argmax_{z ∈ C_t} a(z)
```

where:

```text
C_t = candidate pool generated at iteration t
```

### 9.8 Multi-Objective Extension

For some tasks, performance has multiple objectives:

```text
score_accuracy
score_format
score_latency
score_cost
score_safety
```

Then define a vector objective:

```text
F(z) = [J_accuracy(z), J_format(z), -Cost(z), -Latency(z)]
```

BESO may use either:

1. scalarization,
    
2. Pareto archive selection,
    
3. constrained optimization.
    

Example scalar objective:

```text
J_total(z) = w_acc J_acc(z) + w_fmt J_fmt(z) - w_cost Cost(z) - w_lat Latency(z)
```

---

## 10. Algorithm Overview

### 10.1 High-Level Loop

```text
1. Initialize skill artifact z_0.
2. Evaluate z_0 on a seed validation set.
3. Store trajectories, scores, and feedback.
4. Generate candidate skill edits using reflection.
5. Featurize candidates.
6. Fit or update Bayesian surrogate.
7. Use acquisition function to select candidates for evaluation.
8. Evaluate selected candidates with rollout budget.
9. Accept, reject, or archive candidates.
10. Repeat until budget is exhausted.
11. Return best validated skill artifact.
```

### 10.2 Pseudocode

```python
initialize skill z0
H = []
archive = []
rejected_edits = []

score0, traces0 = evaluate(z0, eval_split="validation_seed")
H.append((z0, score0, traces0))
archive.append(z0)

for t in range(max_iterations):
    parents = select_parents(archive, strategy="pareto_plus_ucb")

    candidate_pool = []
    for parent in parents:
        relevant_traces = retrieve_traces(parent, H)
        edit_proposals = reflection_model.propose_edits(
            skill=parent,
            traces=relevant_traces,
            rejected_edits=rejected_edits
        )
        candidates = apply_bounded_edits(parent, edit_proposals)
        candidate_pool.extend(candidates)

    candidate_pool = filter_invalid_candidates(candidate_pool)
    candidate_pool = deduplicate(candidate_pool)

    X_candidates = featurize(candidate_pool)
    surrogate = fit_surrogate(H)
    acquisition_scores = compute_acquisition(surrogate, X_candidates)

    selected = select_top_k(candidate_pool, acquisition_scores, k=batch_size)

    for z in selected:
        score, traces = evaluate(z, eval_split="optimization_minibatch")
        H.append((z, score, traces))

        validation_score = validate_candidate(z)

        if accept_candidate(z, validation_score, archive):
            archive.append(z)
        else:
            rejected_edits.append(extract_edit_record(z, traces, validation_score))

    archive = prune_archive(archive, strategy="pareto_and_diversity")

return select_best_final_skill(archive, final_validation_set)
```

---

## 11. Candidate Generation

BESO does not ask the Bayesian model to generate text directly. Instead:

```text
Reflection model generates candidate edits.
Bayesian surrogate chooses which edits to evaluate.
```

This separation is important.

The reflection model is good at semantic text editing. The Bayesian surrogate is good at experiment planning under uncertainty.

### 11.1 Candidate Generation Inputs

The reflection model receives:

- parent skill artifact,
    
- successful trajectories,
    
- failed trajectories,
    
- evaluator feedback,
    
- previous accepted edits,
    
- rejected edit buffer,
    
- current failure taxonomy,
    
- edit budget.
    

### 11.2 Edit Proposal Schema

```json
{
  "edit_id": "edit_00017",
  "parent_skill_id": "skill_v4",
  "target_section": "verification_checklist",
  "operation": "add",
  "proposed_text": "Before finalizing, verify that the answer directly resolves the original question rather than an intermediate sub-question.",
  "rationale": "Several failed trajectories answered intermediate facts instead of the final user query.",
  "expected_effect": "Improve final-answer alignment on multi-hop questions.",
  "risk": "May increase verbosity or slow down direct questions.",
  "estimated_scope": "multi_step_reasoning",
  "edit_size_tokens": 27
}
```

### 11.3 Edit Operations

Supported operations:

|Operation|Description|
|---|---|
|add_rule|Add a new rule to a section|
|delete_rule|Remove harmful or redundant instruction|
|replace_rule|Rewrite an existing instruction|
|specialize_rule|Make a general instruction more task-specific|
|generalize_rule|Convert narrow fix into reusable rule|
|reorder_steps|Change procedural order|
|add_example|Add a few-shot example|
|delete_example|Remove misleading example|
|compress_section|Reduce verbosity|
|split_section|Separate overloaded section|
|merge_sections|Combine redundant sections|
|add_failure_mode|Add common error pattern|
|add_recovery_rule|Add fallback behavior|

### 11.4 Bounded Edit Constraints

Each edit must satisfy constraints:

```text
max_added_tokens_per_iteration
max_deleted_tokens_per_iteration
max_replaced_tokens_per_iteration
max_sections_modified_per_iteration
max_examples_added_per_iteration
```

Default constraints:

```yaml
max_added_tokens_per_iteration: 120
max_deleted_tokens_per_iteration: 80
max_replaced_tokens_per_iteration: 160
max_sections_modified_per_iteration: 2
max_examples_added_per_iteration: 1
```

Bounded edits prevent unstable prompt drift.

---

## 12. Candidate Featurization

Bayesian optimization requires numeric candidate representations.

### 12.1 Feature Vector

Each candidate z is converted into:

```text
φ(z) = numeric feature vector
```

Potential features:

#### Text Embedding Features

- embedding of full skill artifact,
    
- embedding of changed section,
    
- embedding of edit rationale,
    
- embedding of proposed delta.
    

#### Structural Features

- total token count,
    
- edit size,
    
- section modified,
    
- number of rules,
    
- number of examples,
    
- number of checklist items,
    
- number of failure modes,
    
- number of recovery rules.
    

#### Semantic LLM-Labeled Features

- decomposition emphasis,
    
- verification emphasis,
    
- tool-use aggressiveness,
    
- caution level,
    
- verbosity level,
    
- specificity level,
    
- format strictness,
    
- uncertainty-handling strength.
    

#### Historical Features

- parent score,
    
- parent variance,
    
- edit type success rate,
    
- section success rate,
    
- similarity to previously failed candidates,
    
- similarity to accepted candidates,
    
- number of previous mutations from same lineage.
    

### 12.2 Example Feature Record

```json
{
  "candidate_id": "skill_v7_candidate_3",
  "parent_score": 0.64,
  "edit_operation": "add_rule",
  "target_section": "tool_use_policy",
  "edit_size_tokens": 44,
  "skill_token_count": 612,
  "num_rules": 18,
  "num_examples": 2,
  "semantic_features": {
    "decomposition_emphasis": 0.72,
    "verification_emphasis": 0.81,
    "tool_use_aggressiveness": 0.43,
    "caution_level": 0.66,
    "verbosity_level": 0.58
  },
  "embedding": "<vector>"
}
```

---

## 13. Bayesian Surrogate Options

BESO should support multiple surrogate models.

### 13.1 Gaussian Process

Useful when the number of evaluated candidates is small and feature dimension is manageable.

Pros:

- principled uncertainty,
    
- strong Bayesian foundation.
    

Cons:

- scales poorly,
    
- struggles with high-dimensional text embeddings.
    

### 13.2 Bayesian Ridge Regression

Useful as a simple baseline.

Pros:

- simple,
    
- fast,
    
- interpretable.
    

Cons:

- limited nonlinearity.
    

### 13.3 Random Forest or Extra Trees Surrogate

Useful for structured features.

Pros:

- robust,
    
- handles mixed features,
    
- uncertainty can be approximated from tree variance.
    

Cons:

- uncertainty is heuristic.
    

### 13.4 TPE-Style Surrogate

Tree-structured Parzen Estimator can model good vs bad regions.

Pros:

- practical,
    
- works well for hyperparameter-like spaces,
    
- robust to mixed feature types.
    

Cons:

- less direct posterior interpretation.
    

### 13.5 Ensemble Surrogate

Recommended default.

Use an ensemble of lightweight models:

```text
surrogate = ensemble(
  BayesianRidge,
  RandomForest,
  KNN_on_embeddings,
  lightweight_neural_regressor_optional
)
```

Estimate uncertainty through prediction disagreement.

Recommended for v0:

> Use ensemble surrogate with structured features + embedding similarity.

---

## 14. Acquisition Functions

BESO should support several acquisition strategies.

### 14.1 Upper Confidence Bound

```text
a(z) = μ_t(z) + κ σ_t(z)
```

Best default because it is simple and explicitly balances exploitation and exploration.

### 14.2 Expected Improvement

```text
EI(z) = E[max(0, f(z) - f_best)]
```

Useful when the goal is improvement over current best.

### 14.3 Probability of Improvement

```text
PI(z) = P(f(z) > f_best + ξ)
```

Simple but can over-exploit.

### 14.4 Thompson Sampling

Sample a possible performance function from the posterior and choose the best candidate under that sample.

Good for batched and noisy settings.

### 14.5 Diversity-Aware Acquisition

To avoid testing near-duplicate candidates:

```text
a_diverse(z) = a(z) + λ diversity(z, archive)
```

where:

```text
diversity(z, archive) = minimum distance from z to archived candidates
```

### 14.6 Risk-Aware Acquisition

Penalize candidates predicted to increase latency, cost, verbosity, or invalid output risk:

```text
a_risk(z) = μ_t(z) + κσ_t(z) - α cost(z) - β latency(z) - γ invalidity_risk(z)
```

Recommended default:

```text
a(z) = μ_t(z) + κσ_t(z) + λ diversity(z) - α cost(z)
```

---

## 15. Archive and Selection Strategy

BESO should not maintain only one best skill.

It should maintain an archive with:

1. best-average candidates,
    
2. Pareto-specialized candidates,
    
3. diverse candidates,
    
4. candidates that reveal useful negative information.
    

### 15.1 Score Matrix

Let:

```text
S[k, i] = score of candidate k on example i
```

For each example:

```text
s_i* = max_k S[k, i]
```

Winner set:

```text
W_i = {z_k : S[k, i] = s_i*}
```

Candidate win count:

```text
f(z_k) = |{i : z_k ∈ W_i}|
```

Selection probability:

```text
P(z_k) = f(z_k) / Σ_j f(z_j)
```

### 15.2 Archive Entry Schema

```json
{
  "candidate_id": "skill_v8",
  "parent_id": "skill_v6",
  "artifact": "<skill document>",
  "scores": {
    "optimization_mean": 0.71,
    "validation_mean": 0.68,
    "format_score": 0.94,
    "cost_per_task": 0.012,
    "latency_seconds": 4.2
  },
  "lineage_depth": 5,
  "winning_examples": ["ex_003", "ex_018", "ex_041"],
  "known_strengths": ["multi-hop decomposition", "format compliance"],
  "known_weaknesses": ["slower on direct questions"],
  "accepted_edit_summary": "Added intermediate-answer verification step.",
  "created_at_iteration": 12
}
```

### 15.3 Archive Pruning

Archive pruning should preserve:

- top-k by validation score,
    
- top-k by Pareto win count,
    
- top-k by diversity,
    
- most informative failed candidates.
    

Default:

```yaml
max_archive_size: 32
top_by_validation: 8
top_by_pareto: 8
top_by_diversity: 8
top_failed_informative: 8
```

---

## 16. Evaluation Protocol

### 16.1 Dataset Splits

Use at least four splits:

|Split|Purpose|
|---|---|
|feedback_train|Generate trajectories and reflections|
|optimization_minibatch|Fast candidate scoring|
|validation_gate|Accept/reject candidates|
|final_test|Report final unbiased results|

### 16.2 Why Separate Splits Matter

If candidates are generated and accepted on the same examples, the optimizer may overfit. A separate validation gate reduces this risk.

### 16.3 Recommended Split

For small datasets:

```text
40% feedback_train
20% optimization_minibatch rotation
20% validation_gate
20% final_test
```

For benchmark datasets with fixed train/test:

```text
train split → feedback + optimization + validation
held-out test split → final test only
```

### 16.4 Evaluation Metrics

Primary metric depends on task:

|Task Type|Metric|
|---|---|
|Math QA|exact answer accuracy|
|Code generation|pass@1 or unit test pass rate|
|Classification|accuracy, macro-F1|
|Extraction|field-level F1|
|JSON output|schema validity + content accuracy|
|Tool-use task|final accuracy + tool efficiency|

Secondary metrics:

- token cost,
    
- latency,
    
- invalid output rate,
    
- number of tool calls,
    
- verbosity,
    
- calibration / uncertainty behavior,
    
- robustness on hard examples.
    

### 16.5 Acceptance Rule

A candidate is accepted if:

```text
validation_score(candidate) > validation_score(parent) + δ
```

where:

```text
δ = minimum improvement threshold
```

Optional statistical guard:

```text
accept if improvement is positive on bootstrap confidence interval
```

### 16.6 Rejection Rule

Reject candidate if:

- validation score decreases,
    
- output validity drops below threshold,
    
- cost increases beyond budget,
    
- skill becomes too long,
    
- candidate violates invariants,
    
- candidate is too similar to archived candidates without improvement.
    

---

## 17. Reflection Module

### 17.1 Role

The reflection module transforms trajectory evidence into candidate edits.

It should not directly decide acceptance. It proposes edits. Evaluation decides whether the edits survive.

### 17.2 Reflection Prompt Inputs

The reflection model receives:

```text
- current skill artifact
- task examples
- successful trajectories
- failed trajectories
- evaluator feedback
- previous accepted edits
- rejected edits
- edit budget
- target section constraints
```

### 17.3 Reflection Output

The reflection model must output structured JSON:

```json
{
  "diagnosis": "The model often answers intermediate facts instead of resolving the original question.",
  "failure_modes": [
    "Premature final answer",
    "Weak final-question alignment"
  ],
  "proposed_edits": [
    {
      "operation": "add_rule",
      "target_section": "verification_checklist",
      "text": "Before finalizing, confirm that the answer resolves the original user question, not merely an intermediate sub-question.",
      "rationale": "Prevents premature answers in multi-hop questions.",
      "risk": "May add unnecessary checking on simple questions."
    }
  ]
}
```

### 17.4 Reflection Quality Checks

Reject reflection outputs if:

- invalid JSON,
    
- proposed edit exceeds budget,
    
- target section does not exist,
    
- rationale is missing,
    
- edit contradicts existing constraints,
    
- edit adds vague advice without operational behavior,
    
- edit is a duplicate of rejected edit.
    

---

## 18. Skill Compiler

### 18.1 Purpose

The skill compiler turns a skill artifact into runtime prompts.

### 18.2 Compiler Inputs

```text
skill artifact
task input
module role
available tools
output schema
runtime constraints
```

### 18.3 Compiler Output

```text
runtime prompt
```

### 18.4 Compiler Modes

#### Full Injection

Inject the full skill into the prompt.

Pros:

- maximum instruction availability.
    

Cons:

- high token cost,
    
- possible distraction.
    

#### Section Selection

Inject only relevant sections.

Pros:

- efficient,
    
- modular.
    

Cons:

- requires routing.
    

#### Distilled Prompt

Summarize the skill into a compact runtime instruction.

Pros:

- low token cost.
    

Cons:

- may lose detail.
    

Recommended v0:

> Use section selection with deterministic templates.

---

## 19. System Architecture

### 19.1 Components

```text
Dataset Manager
Evaluation Runner
Trajectory Logger
Metric Evaluator
Reflection Proposer
Edit Applicator
Candidate Featurizer
Bayesian Surrogate
Acquisition Selector
Validation Gate
Archive Manager
Skill Compiler
Experiment Tracker
```

### 19.2 Architecture Flow

```text
Dataset Manager
    ↓
Evaluation Runner ← Skill Compiler ← Candidate Skill
    ↓
Trajectory Logger
    ↓
Metric Evaluator
    ↓
Reflection Proposer
    ↓
Edit Applicator
    ↓
Candidate Pool
    ↓
Candidate Featurizer
    ↓
Bayesian Surrogate
    ↓
Acquisition Selector
    ↓
Validation Gate
    ↓
Archive Manager
    ↓
Best Skill Artifact
```

---

## 20. Data Models

### 20.1 Task Example

```json
{
  "example_id": "ex_001",
  "input": "Question text or task input",
  "metadata": {
    "expected_answer": "...",
    "rubric": "...",
    "difficulty": "medium",
    "category": "multi_step_reasoning"
  }
}
```

### 20.2 Trajectory Record

```json
{
  "trajectory_id": "traj_001",
  "candidate_id": "skill_v3",
  "example_id": "ex_001",
  "compiled_prompt": "...",
  "model_outputs": [
    {
      "step": 1,
      "type": "reasoning_summary",
      "content": "Identified intermediate entity."
    },
    {
      "step": 2,
      "type": "tool_call",
      "tool_name": "search",
      "arguments": {"query": "..."}
    }
  ],
  "final_output": "...",
  "score": 0.0,
  "feedback": "The answer resolves the intermediate entity but not the final question.",
  "cost": {
    "input_tokens": 1200,
    "output_tokens": 340,
    "tool_calls": 1
  },
  "latency_seconds": 5.3
}
```

### 20.3 Candidate Record

```json
{
  "candidate_id": "skill_v5_candidate_2",
  "parent_id": "skill_v5",
  "artifact": "<skill document>",
  "edit_record": {
    "operation": "add_rule",
    "target_section": "verification_checklist",
    "text": "...",
    "rationale": "..."
  },
  "features": "<feature vector>",
  "surrogate_prediction": {
    "mean": 0.69,
    "uncertainty": 0.08,
    "acquisition_score": 0.77
  },
  "evaluation_result": null
}
```

### 20.4 Evaluation Result

```json
{
  "candidate_id": "skill_v5_candidate_2",
  "split": "validation_gate",
  "mean_score": 0.71,
  "standard_error": 0.04,
  "num_examples": 25,
  "secondary_metrics": {
    "format_validity": 0.96,
    "avg_latency_seconds": 4.9,
    "avg_tool_calls": 1.2,
    "avg_tokens": 1320
  },
  "accepted": true,
  "acceptance_reason": "Validation score improved by 0.04 over parent."
}
```

### 20.5 Rejected Edit Record

```json
{
  "edit_id": "edit_00031",
  "candidate_id": "skill_v9_candidate_1",
  "reason": "Increased verbosity and reduced direct-answer accuracy.",
  "target_section": "core_procedure",
  "operation": "add_rule",
  "text": "...",
  "observed_failure": "The model over-explained simple examples.",
  "do_not_repeat": true
}
```

---

## 21. Baselines

BESO should be evaluated against both prompt-level and skill-level baselines.

### 21.1 No Optimization

Use the initial prompt or initial skill.

### 21.2 Manual Skill

Human-written skill document.

### 21.3 One-Shot LLM Skill

Ask an LLM to generate a skill once, with no iterative optimization.

### 21.4 Random Search

Randomly generate candidate edits and evaluate them.

### 21.5 Greedy Reflection

Always apply the edit that looks best according to reflection, without Bayesian selection.

### 21.6 GEPA-Style Prompt Evolution

Reflectively evolve prompts directly.

### 21.7 SkillOpt-Style Bounded Skill Edits

Use bounded add/delete/replace edits with validation gates, but no Bayesian surrogate.

### 21.8 Bandit Selection

Use multi-armed bandit selection over edit types or parent candidates.

### 21.9 Bayesian Prompt Optimization

Apply Bayesian selection directly to raw prompt candidates.

This baseline tests whether skill-level abstraction matters.

---

## 22. Ablation Studies

Required ablations:

|Ablation|Purpose|
|---|---|
|No Bayesian surrogate|Test value of Bayesian selection|
|No reflection|Test value of semantic edit generation|
|No Pareto archive|Test value of diversity preservation|
|Prompt-only search|Test value of skill abstraction|
|No rejected-edit buffer|Test stability benefit|
|No structured features|Test value of feature engineering|
|Embeddings only|Test whether semantic embeddings are enough|
|Structured features only|Test whether cheap features are enough|
|No validation gate|Test overfitting risk|
|Full skill injection vs section selection|Test compiler strategy|

---

## 23. Experimental Plan

### 23.1 Phase 1: Toy Validation

Goal:

> Verify that the optimizer loop works.

Tasks:

- small arithmetic word problems,
    
- simple classification,
    
- JSON extraction.
    

Budget:

```yaml
rollouts: 50-200
candidate_pool_size: 8-16
archive_size: 8
```

Success criterion:

```text
BESO improves over initial skill and random search.
```

### 23.2 Phase 2: Benchmark Evaluation

Tasks:

- multi-step QA,
    
- math reasoning,
    
- code generation,
    
- structured extraction,
    
- tool-use tasks.
    

Budget:

```yaml
rollouts: 200-1000
candidate_pool_size: 16-64
archive_size: 16-32
```

Success criterion:

```text
BESO beats or matches GEPA-style prompt evolution and SkillOpt-style bounded editing under the same rollout budget.
```

### 23.3 Phase 3: Low-Budget Stress Test

Evaluate at budgets:

```text
25, 50, 100, 200, 500 rollouts
```

Primary hypothesis:

> BESO should show its advantage most clearly under low rollout budgets.

### 23.4 Phase 4: Transfer Test

Train skill on one setting and test on:

- nearby task distribution,
    
- harder benchmark split,
    
- different model,
    
- different execution harness.
    

Success criterion:

```text
Optimized skill retains value outside the exact optimization setting.
```

---

## 24. Metrics

### 24.1 Primary Metrics

- final test accuracy,
    
- validation score under fixed budget,
    
- area under optimization curve,
    
- best score achieved at budget B.
    

### 24.2 Sample Efficiency Metrics

```text
score after 25 rollouts
score after 50 rollouts
score after 100 rollouts
rollouts required to beat baseline
```

### 24.3 Robustness Metrics

- performance on hard examples,
    
- variance across random seeds,
    
- sensitivity to initial skill,
    
- sensitivity to evaluator noise.
    

### 24.4 Cost Metrics

- total tokens used during optimization,
    
- inference-time token overhead,
    
- total model calls,
    
- total tool calls,
    
- latency.
    

### 24.5 Interpretability Metrics

- number of accepted edits,
    
- percentage of accepted edits with clear rationale,
    
- human rating of skill readability,
    
- edit locality,
    
- presence of contradictory rules.
    

---

## 25. Main Hypotheses

### H1: Skill-Level Optimization Beats Prompt-Level Optimization

Structured skill artifacts will produce better generalization and more stable optimization than raw prompt strings.

### H2: Bayesian Acquisition Improves Sample Efficiency

Bayesian candidate selection will outperform random, greedy, and Pareto-only selection under low rollout budgets.

### H3: Reflection Improves Candidate Quality

Reflection-generated edits will outperform mutation operators that do not use trajectory feedback.

### H4: Pareto Archives Improve Robustness

Maintaining diverse specialized candidates will improve final performance compared with keeping only the best-average candidate.

### H5: Optimized Skills Transfer

Optimized skill artifacts will retain some performance gain when transferred to nearby tasks or models.

---

## 26. Failure Modes

### 26.1 Surrogate Miscalibration

The Bayesian model may predict high value for candidates that fail in real evaluation.

Mitigation:

- use uncertainty calibration,
    
- use ensemble disagreement,
    
- periodically evaluate random candidates,
    
- track prediction error.
    

### 26.2 Overfitting to Validation Examples

The skill may become too tailored to validation tasks.

Mitigation:

- separate feedback, optimization, validation, and test splits,
    
- rotate minibatches,
    
- use held-out final test,
    
- penalize overly specific rules.
    

### 26.3 Prompt Bloat

The skill may grow too long.

Mitigation:

- token budget,
    
- compression edits,
    
- cost-aware acquisition,
    
- inference-time section selection.
    

### 26.4 Contradictory Rules

Multiple edits may introduce conflicting instructions.

Mitigation:

- contradiction checker,
    
- skill linting,
    
- reflection prompt must identify conflicts,
    
- periodic consolidation.
    

### 26.5 Reflection Hallucination

The reflection model may invent failure causes not supported by traces.

Mitigation:

- require trace-grounded rationales,
    
- cite trajectory IDs in edit proposals,
    
- reject unsupported edits.
    

### 26.6 Edit Myopia

Small bounded edits may fail to discover larger useful shifts.

Mitigation:

- occasional large mutation,
    
- crossover between archived skills,
    
- meta-edit phase every N iterations.
    

### 26.7 Evaluation Noise

LLM judges and stochastic model outputs may produce noisy scores.

Mitigation:

- repeated evaluation for top candidates,
    
- bootstrap confidence intervals,
    
- deterministic decoding when possible,
    
- judge consistency checks.
    

---

## 27. Invariants and Safety Checks

The optimizer must preserve these invariants:

1. Skill artifact must remain valid under schema.
    
2. Skill must not contradict task rules.
    
3. Output format requirements must remain intact.
    
4. Tool-use policy must not authorize unavailable tools.
    
5. Token count must stay under configured budget.
    
6. No accepted edit may reduce validation score beyond tolerance.
    
7. Final reported score must use untouched final test split.
    

Skill linting checks:

```text
schema_validity
section_presence
token_budget
contradiction_check
unsafe_instruction_check
format_rule_preservation
unavailable_tool_check
```

---

## 28. Implementation Plan

### 28.1 Minimal v0

Goal:

> Build the simplest working optimizer.

Components:

- skill schema,
    
- evaluation runner,
    
- trajectory logger,
    
- reflection edit proposer,
    
- edit applicator,
    
- simple ensemble surrogate,
    
- UCB acquisition,
    
- validation gate,
    
- archive manager.
    

Recommended stack:

```yaml
language: Python
storage: SQLite or local JSONL
experiment_tracking: Weights & Biases, MLflow, or local logs
surrogate_models: scikit-learn
LLM_calls: provider-agnostic adapter
```

### 28.2 v0 Algorithm

```text
Skill artifact only.
Single target model.
Single benchmark.
Single scoring metric.
UCB acquisition.
Ensemble surrogate.
Validation-gated acceptance.
```

### 28.3 v1

Add:

- multi-objective scoring,
    
- Pareto archive,
    
- rejected-edit buffer,
    
- section-selection compiler,
    
- more baselines,
    
- ablation suite.
    

### 28.4 v2

Add:

- transfer evaluation,
    
- hierarchical skill-to-module prompt compilation,
    
- task-adaptive skill routing,
    
- multi-model skill robustness,
    
- inference-time cost optimization.
    

---

## 29. Suggested Repository Structure

```text
beso/
  README.md
  pyproject.toml
  configs/
    default.yaml
    experiments/
  beso/
    __init__.py
    artifacts/
      skill.py
      prompt.py
      schema.py
    compiler/
      skill_compiler.py
      section_selector.py
    evaluation/
      runner.py
      metrics.py
      judge.py
      splits.py
    trajectories/
      logger.py
      store.py
      filters.py
    reflection/
      proposer.py
      prompts.py
      validators.py
    edits/
      operations.py
      applicator.py
      lint.py
    features/
      featurizer.py
      embeddings.py
      semantic_labels.py
    surrogate/
      base.py
      ensemble.py
      gaussian_process.py
      tpe.py
    acquisition/
      ucb.py
      expected_improvement.py
      thompson.py
      diversity.py
    archive/
      manager.py
      pareto.py
      lineage.py
    optimization/
      loop.py
      accept_reject.py
      budget.py
    experiments/
      baselines.py
      ablations.py
      reporting.py
  tests/
    test_skill_schema.py
    test_edit_applicator.py
    test_archive.py
    test_acquisition.py
  examples/
    arithmetic_word_problems/
    json_extraction/
    multi_step_qa/
```

---

## 30. Configuration Example

```yaml
experiment:
  name: beso_multi_step_qa_v0
  seed: 42

artifact:
  type: skill
  max_tokens: 900
  compiler_mode: section_selection

optimization:
  max_rollouts: 300
  max_iterations: 30
  batch_size: 2
  candidate_pool_size: 24
  archive_size: 32
  min_improvement_delta: 0.01

edits:
  max_added_tokens_per_iteration: 120
  max_deleted_tokens_per_iteration: 80
  max_replaced_tokens_per_iteration: 160
  max_sections_modified_per_iteration: 2
  allowed_operations:
    - add_rule
    - delete_rule
    - replace_rule
    - add_failure_mode
    - add_recovery_rule
    - compress_section

surrogate:
  type: ensemble
  models:
    - bayesian_ridge
    - random_forest
    - knn_embedding
  uncertainty: ensemble_disagreement

acquisition:
  type: ucb_diversity_cost
  kappa: 1.5
  diversity_lambda: 0.2
  cost_alpha: 0.1

archive:
  strategy: pareto_and_diversity
  top_by_validation: 8
  top_by_pareto: 8
  top_by_diversity: 8
  top_failed_informative: 8

reflection:
  model: optimizer_model
  require_trace_grounding: true
  include_rejected_edits: true

evaluation:
  target_model: target_model
  decoding_temperature: 0.0
  metric: exact_or_judge_score
  repeated_eval_for_top_candidates: true
```

---

## 31. Acceptance Criteria for v0

BESO v0 is successful if it demonstrates:

1. End-to-end optimization loop runs without manual intervention.
    
2. Skill artifacts remain schema-valid after edits.
    
3. Candidate evaluations are logged with trajectories and scores.
    
4. Bayesian surrogate selects candidates based on acquisition score.
    
5. Final optimized skill improves over initial skill on validation.
    
6. Final test result is reported on untouched test split.
    
7. At least one baseline comparison is included.
    
8. Optimization trace is inspectable.
    

Minimum experimental success:

```text
BESO > initial skill
BESO > random edit search
```

Stronger success:

```text
BESO ≥ SkillOpt-style bounded edits under same rollout budget
BESO ≥ GEPA-style prompt evolution under same rollout budget
```

---

## 32. Reporting Format

Each experiment should report:

```text
- dataset
- target model
- initial skill
- optimizer configuration
- rollout budget
- final optimized skill
- validation score curve
- final test score
- baselines
- ablations
- cost
- accepted edits
- rejected edits
- qualitative failure analysis
```

### 32.1 Optimization Curve

Report score as a function of rollout budget:

```text
rollouts → best validation score
```

This is crucial because BESO’s central claim is sample efficiency.

### 32.2 Edit Trace

Report accepted edits:

|Iteration|Section|Operation|Rationale|Validation Change|
|---|---|---|---|---|
|1|verification_checklist|add_rule|Prevent intermediate-answer errors|+0.03|
|4|output_rules|replace_rule|Reduce verbosity|+0.02|
|9|tool_use_policy|add_rule|Improve calculator use|+0.04|

---

## 33. Research Contribution Claim

The strongest possible contribution is not merely:

> We use Bayesian optimization for prompts.

That is too broad and likely not novel enough.

The stronger claim is:

> We introduce a Bayesian-guided evolutionary optimizer for natural-language skill artifacts, where trajectory-grounded reflection proposes bounded semantic edits and a probabilistic surrogate selects which skill variants to evaluate under a limited rollout budget.

The specific contributions:

1. A skill-level optimization target that is more structured than raw prompts.
    
2. A Bayesian surrogate for predicting candidate skill utility and uncertainty.
    
3. Acquisition-guided selection of natural-language edits.
    
4. A Pareto-diverse archive for preserving specialized skill variants.
    
5. A controlled evaluation against prompt evolution, gradient-like text editing, and random/greedy baselines.
    

---

## 34. Expected Strengths

BESO should be strongest when:

- rollout budget is small,
    
- evaluations are expensive,
    
- trajectories contain useful diagnostic feedback,
    
- the task benefits from reusable procedures,
    
- prompt wording is not enough and higher-level skill policy matters,
    
- there are multiple competing behavioral strategies.
    

---

## 35. Expected Weaknesses

BESO may struggle when:

- the evaluation metric is weak,
    
- feedback is vague,
    
- the task has little reusable structure,
    
- skill edits do not affect the bottleneck,
    
- the surrogate cannot model text-performance relationships,
    
- the candidate generator produces low-quality edits,
    
- the optimized skill overfits to benchmark quirks.
    

---

## 36. Token-Cost Motivation and Cost-Aware BESO

A practical concern with SkillOpt-style skill optimization is token consumption. The cost issue has two separate forms:

1. **Optimization-time token cost**
    
    - target-agent rollouts during training,
        
    - optimizer-model reflection calls,
        
    - candidate validation calls,
        
    - repeated evaluation of candidate edits,
        
    - rejected-edit context,
        
    - slow/meta update context.
        
2. **Inference-time token cost**
    
    - the final skill document injected into the runtime context,
        
    - extra instructions added by the skill,
        
    - longer model outputs caused by more verbose procedures,
        
    - additional tool calls caused by the learned policy.
        

BESO should explicitly address both, but its primary advantage is expected at **optimization time**.

SkillOpt-style optimization can be described as:

```text
rollout → reflect → propose bounded edit → evaluate candidate → accept/reject
```

If the optimizer proposes weak or redundant edits, the system may still spend expensive rollout and validation budget evaluating them.

BESO inserts a budget-aware selection layer:

```text
rollout → reflect → propose many candidate edits
        → surrogate predicts utility + uncertainty + cost
        → acquisition selects only worthwhile candidates
        → evaluate selected candidates
        → accept/reject/archive
```

If reflection proposes `M` candidate edits and each candidate requires `b` rollout examples to evaluate, naive evaluation costs:

```text
Cost_naive ≈ M × b
```

BESO evaluates only `K` selected candidates, where `K << M`:

```text
Cost_BESO ≈ K × b
```

The Bayesian layer is therefore not just about quality. It can be framed as **cost-aware experiment planning**.

### 36.1 Cost-aware acquisition

The acquisition function should include token and rollout cost directly:

```text
a_BESO(z) = μ_t(z) + κσ_t(z) + λ diversity(z)
            - α C_train(z)
            - β C_infer(z)
            - γ invalidity_risk(z)
```

where:

```text
μ_t(z) = predicted candidate utility
σ_t(z) = surrogate uncertainty
diversity(z) = distance from archived candidates
C_train(z) = predicted optimization-time cost
C_infer(z) = predicted inference-time cost
invalidity_risk(z) = predicted probability of schema/output failure
```

Training cost can be decomposed as:

```text
C_train(z)
= rollout_tokens(z)
+ reflection_tokens(z)
+ validation_tokens(z)
+ tool_call_cost(z)
```

Inference cost can be decomposed as:

```text
C_infer(z)
= token_count(compiled_skill)
+ expected_extra_output_tokens(z)
+ expected_extra_tool_tokens(z)
```

This turns candidate selection into the question:

> Is this edit worth the tokens?

not merely:

> Does this edit improve score?

### 36.2 Cost-aware objective

A cost-aware utility can be defined as:

```text
U(z) = J_quality(z) - λ_train C_train(z) - λ_infer C_infer(z)
```

where:

```text
J_quality(z) = expected task score
C_train(z) = optimization-time token/call cost
C_infer(z) = deployment-time token/call cost
```

Then BESO selects candidates by uncertainty-aware expected utility:

```text
z_(t+1) = argmax_{z ∈ C_t} E[U(z) | H_t] + κ Var[U(z) | H_t]^(1/2)
```

This makes token efficiency a first-class optimization target rather than a post-hoc concern.

### 36.3 Hard token constraints

BESO should also support hard constraints:

```text
Tokens(z) ≤ T_skill_max
Tokens(C(z, x, q)) ≤ T_runtime_max
C_train(z) ≤ C_train_max
C_infer(z) ≤ C_infer_max
```

Hard constraints are stricter than cost penalties. A candidate that exceeds the deployment token budget should be invalid even if it improves validation accuracy.

### 36.4 Compression and section-selection compiler

A major way for BESO to address inference-time cost is to optimize not only the skill artifact, but also the compiler that projects the artifact into runtime context.

Instead of always injecting the full skill:

```text
p = C_full(z)
```

BESO can use section selection:

```text
p = C_select(z, x, q)
```

For example, a simple math task may need only:

```text
Core Procedure
Verification Checklist
Output Rules
```

and may not need:

```text
All failure modes
All examples
Full recovery policy
Change log
```

This allows the skill artifact to remain rich while the runtime prompt stays compact.

A stronger BESO formulation optimizes both the skill artifact and its compiler:

```text
z*, C* = argmax_{z,C} E[ μ(Φ(x; C(z,x,q), Θ_frozen), m) ]
```

subject to:

```text
E[ Tokens(C(z,x,q)) ] ≤ T_budget
```

This extends the research question from:

> What skill should we learn?

into:

> What skill should we learn, and which parts of that skill should be injected for this task?

### 36.5 Positioning against SkillOpt

SkillOpt already tries to keep deployed skills compact, but its optimization process can still be token-heavy because it uses rollout batches, optimizer reflection, validation gates, rejected-edit feedback, and slow/meta updates.

BESO addresses this at the experiment-planning layer:

> Instead of evaluating every reflected edit, BESO predicts which edits are worth their rollout cost.

If compiler optimization is included, BESO also addresses deployment cost:

> Instead of injecting the whole learned skill, BESO learns or selects a compact task-relevant skill projection.

A concise contribution claim:

> SkillOpt makes skill learning stable through bounded edits and validation gates, but still spends rollout and reflection budget on candidate updates before knowing whether they are promising. BESO treats candidate evaluation as a budgeted Bayesian experiment: it models expected improvement, uncertainty, and token cost, then evaluates only candidates whose expected value justifies their cost.

An even shorter positioning:

> BESO is cost-aware skill evolution under a rollout and token budget.

### 36.6 Caveat

BESO does not reduce token use automatically. If the acquisition function optimizes only quality, BESO may still learn bloated skills. It may even prefer longer skills if longer skills correlate with higher validation performance.

Therefore token consumption must be included through:

- cost-aware acquisition,
    
- hard token constraints,
    
- compression edits,
    
- section-selection compilation,
    
- inference-time cost reporting.
    

---

## 37. Open Design Questions

1. What is the best feature representation for skill artifacts?
    
2. Should the surrogate model predict absolute performance or improvement over parent?
    
3. Should acquisition happen at candidate level, edit-operation level, or section level?
    
4. Should the archive preserve failed candidates for negative learning?
    
5. How large should bounded edits be?
    
6. How often should the optimizer perform compression or consolidation?
    
7. How should the skill compiler decide which sections to inject?
    
8. Can optimized skills transfer across model families?
    
9. Can Bayesian uncertainty remain useful in high-dimensional text spaces?
    
10. Does Pareto diversity improve final generalization or only optimization-time exploration?
    

---

## 38. Recommended First Prototype

The first prototype should be deliberately small.

### Task

Use one task type with clear scoring, such as:

```text
arithmetic word problems
structured JSON extraction
small code-generation unit tests
```

### Artifact

Use one skill document.

### Optimizer

Use:

```text
reflection-generated bounded edits
ensemble surrogate
UCB acquisition
validation-gated acceptance
simple archive
```

### Baselines

Compare against:

```text
initial skill
random edits
greedy reflection edits
prompt-only Bayesian search
```

### Success Criterion

The prototype is promising if:

```text
BESO achieves higher validation score than random and greedy baselines under the same rollout budget.
```

Do not start with a complex multi-agent task. First prove the optimizer works on a small controlled benchmark.

---

## 39. Final Summary

BESO is a proposed optimizer for frozen LLM systems that treats natural-language skills as trainable external state.

The core loop is:

```text
Run skill → collect trajectories → reflect → generate edits → featurize candidates → predict utility and uncertainty → choose candidates with Bayesian acquisition → evaluate → validate → archive → repeat
```

The central mathematical objective is:

```text
z* = argmax_z E_(x,m)~T [ μ(Φ(x; z, Θ_frozen), m) ]
```

where:

```text
z = skill artifact
Φ = frozen LLM system
μ = task metric
```

The central engineering decision is:

> Optimize structured skills, not raw prompts.

The central research hypothesis is:

> Bayesian-guided skill evolution should be more sample-efficient than purely evolutionary or gradient-like prompt/skill optimization, especially under low rollout budgets.

If validated, BESO would occupy a meaningful middle ground between GEPA-style reflective prompt evolution and SkillOpt-style trainable skill documents: it keeps the interpretability of natural-language skills, the flexibility of evolutionary search, and the sample-efficiency discipline of Bayesian experiment planning.
````

## File: pyproject.toml
````toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "beso"
version = "0.0.1"
description = "Bayesian Evolutionary Skill Optimization: trajectory-grounded reflective evolution of natural-language skill artifacts for frozen LLM agents, with a Bayesian surrogate + acquisition experiment-planning layer."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "BESO Authors" }]
keywords = [
    "bayesian-optimization",
    "evolutionary-search",
    "skill-optimization",
    "llm-agents",
    "prompt-optimization",
]

dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "pydantic>=2.5",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
embeddings = ["sentence-transformers>=2.2"]
tracking = ["mlflow>=2.9"]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.1",
    "mypy>=1.7",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["beso*"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
````

## File: tests/__init__.py
````python

````

## File: tests/test_core_contracts.py
````python
from __future__ import annotations

from beso.core import (
    Candidate,
    EditCategory,
    EditOperation,
    EditProposal,
    EvaluationResult,
    Observation,
    RolloutBudget,
    SkillArtifact,
    SplitRole,
    SurrogatePrediction,
)
from beso.core import protocols as P


def test_skill_document_is_source_of_truth() -> None:
    skill = SkillArtifact(skill_id="z0", name="seed", document="# Goal\nSolve tasks.\n")
    assert skill.document.startswith("# Goal")
    assert skill.sections == {}


def test_edit_proposal_maps_to_skillopt_fields() -> None:
    edit = EditProposal(
        edit_id="e1",
        parent_skill_id="z0",
        operation=EditOperation.APPEND,
        content="Always verify the final answer.",
        target="",
        category=EditCategory.ADD_RULE,
    )
    assert edit.operation.value == "append"
    assert edit.category is EditCategory.ADD_RULE


def test_evaluation_result_mean_and_n() -> None:
    ev = EvaluationResult(
        candidate_id="z1",
        split=SplitRole.OPTIMIZATION_MINIBATCH,
        per_example_scores={"a": 1.0, "b": 0.0, "c": 1.0},
    )
    assert ev.n == 3
    assert abs(ev.mean_score - (2 / 3)) < 1e-9


def test_rollout_budget_accounting() -> None:
    b = RolloutBudget(max_rollouts=10)
    b.spend(4)
    assert b.remaining == 6
    assert not b.exhausted
    b.spend(6)
    assert b.exhausted


def test_protocol_surface_exists() -> None:
    for name in (
        "ExecutionHarness",
        "EditApplicator",
        "Evaluator",
        "DatasetProvider",
        "SkillSerializer",
        "ReflectionProposer",
        "Featurizer",
        "Surrogate",
        "AcquisitionFunction",
        "BatchSelector",
        "AcceptanceGate",
        "Archive",
        "RegimeDetector",
    ):
        assert hasattr(P, name), f"missing protocol {name}"


def test_prediction_and_candidate_wiring() -> None:
    pred = SurrogatePrediction(candidate_id="z1", mu=0.7, sigma=0.1, epistemic_var=0.008, aleatoric_var=0.002)
    cand = Candidate(candidate_id="z1", skill=SkillArtifact(skill_id="z1", name="c1"), prediction=pred)
    assert cand.prediction is not None
    assert cand.prediction.mu == 0.7
    _ = Observation(candidate_id="z1", batch_ids=("a", "b"), observed_score=0.66)
````

## File: tests/test_features.py
````python
from __future__ import annotations

import numpy as np
import pytest

from beso.core.types import (
    Candidate,
    EditCategory,
    EditOperation,
    EditProposal,
    Observation,
    SkillArtifact,
    SkillMetadata,
    SkillSection,
)
from beso.features import (
    FeatureExtractor,
    FeatureNormalizer,
    NormalizerConfig,
    compute_structural_metrics,
)
from beso.features.featurizer import HashingEmbedder

PARENT_DOC = (
    "# Goal\nSolve the task accurately.\n\n"
    "## Core Procedure\n- Read the question.\n- Search the context.\n"
)


def _parent() -> SkillArtifact:
    return SkillArtifact(
        skill_id="z0",
        name="seed",
        document=PARENT_DOC,
        metadata=SkillMetadata(lineage_depth=0),
    )


def _child(doc: str, op: EditOperation, content: str, cid: str) -> Candidate:
    skill = SkillArtifact(
        skill_id=cid,
        name=cid,
        document=doc,
        metadata=SkillMetadata(parent_id="z0", lineage_depth=1),
    )
    edit = EditProposal(
        edit_id=f"e_{cid}",
        parent_skill_id="z0",
        operation=op,
        content=content,
        target="",
        category=EditCategory.ADD_RULE,
        target_section=SkillSection.VERIFICATION_CHECKLIST,
        source_type="failure",
    )
    return Candidate(candidate_id=cid, skill=skill, parent_id="z0", edit=edit)


def _pool() -> list[Candidate]:
    parent_doc = PARENT_DOC
    cands = []
    for i in range(12):
        added = "\n".join(f"- Verify step {j}." for j in range(i + 1))
        doc = parent_doc + "\n
        cands.append(
            _child(doc, EditOperation.APPEND, content=added, cid=f"z{i+1}")
        )
    return cands


def test_structural_features_are_parent_relative_deltas() -> None:
    fx = FeatureExtractor()
    parent = _parent()
    # Child identical to parent -> all deltas zero.
    same = Candidate(
        candidate_id="zsame",
        skill=SkillArtifact(skill_id="zsame", name="s", document=PARENT_DOC),
        parent_id="z0",
        edit=EditProposal(
            edit_id="e", parent_skill_id="z0", operation=EditOperation.APPEND
        ),
    )
    feats = fx.featurize(same, parent, history=[])
    for k in compute_structural_metrics(PARENT_DOC):
        assert feats.structural[f"d_{k}"] == pytest.approx(0.0)

    # Child with more bullets -> positive delta on bullets/rules.
    bigger_doc = PARENT_DOC + "\n- Extra rule one.\n- Extra rule two.\n"
    bigger = _child(bigger_doc, EditOperation.APPEND, "- Extra rule one.", "zbig")
    bf = fx.featurize(bigger, parent, history=[])
    assert bf.structural["d_bullets"] == pytest.approx(2.0)
    assert bf.structural["d_rules"] == pytest.approx(2.0)
    assert bf.structural["d_tokens"] > 0.0


def test_edit_block_one_hot_and_history_cold_start() -> None:
    fx = FeatureExtractor()
    parent = _parent()
    cand = _child(PARENT_DOC + "\n- x.", EditOperation.APPEND, "- x.", "z1")
    history = [
        Observation(candidate_id="z0", batch_ids=("a",), observed_score=0.5),
        Observation(candidate_id="z0", batch_ids=("b",), observed_score=0.7),
    ]
    feats = fx.featurize(cand, parent, history=history)
    assert feats.edit["op_append"] == 1.0
    assert feats.edit["op_replace"] == 0.0
    assert feats.edit["cat_add_rule"] == 1.0
    assert feats.edit["sec_verification_checklist"] == 1.0
    assert feats.edit["src_failure"] == 1.0

    assert feats.history["parent_mean"] == pytest.approx(0.6)
    assert feats.history["parent_n_obs"] == 2.0

    assert feats.history["edit_type_success_rate"] == pytest.approx(0.5)


def test_normalizer_standardizes_and_pca_reduces_text() -> None:
    fx = FeatureExtractor(embed_fn=HashingEmbedder(dim=128))
    parent = _parent()
    pool = _pool()
    feats = [fx.featurize(c, parent, history=[]) for c in pool]

    norm = FeatureNormalizer(NormalizerConfig(text_pca_dims=8))
    X = norm.fit_transform(feats)

    assert X.shape[0] == len(pool)
    assert np.all(np.isfinite(X))

    slices = norm.block_slices()
    assert "text" in slices

    assert (slices["text"].stop - slices["text"].start) <= 8


    s = slices["structural"]
    col_means = X[:, s].mean(axis=0)
    assert np.allclose(col_means, 0.0, atol=1e-6)


def test_dimension_balancing_prevents_text_swamping() -> None:
    fx = FeatureExtractor(embed_fn=HashingEmbedder(dim=256))
    parent = _parent()
    pool = _pool()
    feats = [fx.featurize(c, parent, history=[]) for c in pool]

    balanced = FeatureNormalizer(
        NormalizerConfig(text_pca_dims=10, balance_block_dims=True)
    ).fit_transform(feats)
    unbalanced = FeatureNormalizer(
        NormalizerConfig(text_pca_dims=10, balance_block_dims=False)
    )
    Xun = unbalanced.fit_transform(feats)

    sl = unbalanced.block_slices()
    text_share_un = _block_var_share(Xun, sl, "text")

    bnorm = FeatureNormalizer(
        NormalizerConfig(text_pca_dims=10, balance_block_dims=True)
    )
    Xb = bnorm.fit_transform(feats)
    text_share_b = _block_var_share(Xb, bnorm.block_slices(), "text")


    assert text_share_b < text_share_un
    assert np.all(np.isfinite(balanced))


def _block_var_share(X: np.ndarray, slices: dict, block: str) -> float:
    total = float(np.sum(np.var(X, axis=0)))
    if total <= 0:
        return 0.0
    block_var = float(np.sum(np.var(X[:, slices[block]], axis=0)))
    return block_var / total
````

## File: tests/test_surrogate.py
````python
from __future__ import annotations

import numpy as np
import pytest

from beso.core.types import (
    Candidate,
    EditCategory,
    EditOperation,
    EditProposal,
    Observation,
    SkillArtifact,
    SkillMetadata,
    SkillSection,
)
from beso.features import FeatureExtractor
from beso.features.featurizer import HashingEmbedder
from beso.surrogate import (
    BaggingEnsembleSurrogate,
    IsotonicCalibrator,
    TemperatureScaler,
)
from beso.surrogate.calibration import _pav_nondecreasing

PARENT_MEAN = 0.5
NOISE_SD = 0.05
RNG = np.random.default_rng(7)

PARENT_DOC = "# Goal\nSolve.\n\n## Core Procedure\n- Read.\n- Search.\n"


def _parent() -> SkillArtifact:
    return SkillArtifact(skill_id="z0", name="seed", document=PARENT_DOC)


def _candidate(i: int) -> Candidate:
    bullets = "\n".join(f"- Verify rule {j}." for j in range(i + 1))
    doc = PARENT_DOC + "\n
    skill = SkillArtifact(
        skill_id=f"z{i+1}",
        name=f"z{i+1}",
        document=doc,
        metadata=SkillMetadata(parent_id="z0", lineage_depth=1),
    )
    edit = EditProposal(
        edit_id=f"e{i+1}",
        parent_skill_id="z0",
        operation=EditOperation.APPEND,
        content=bullets,
        category=EditCategory.ADD_RULE,
        target_section=SkillSection.VERIFICATION_CHECKLIST,
        source_type="failure",
    )
    return Candidate(candidate_id=f"z{i+1}", skill=skill, parent_id="z0", edit=edit)


def _true_delta(features) -> float:

    return 0.02 * float(features.structural["d_bullets"])


def _build_dataset(n_candidates: int = 24, reps: int = 4):
    fx = FeatureExtractor(embed_fn=HashingEmbedder(dim=96))
    parent = _parent()
    parent_history = [
        Observation(candidate_id="z0", batch_ids=("p1",), observed_score=PARENT_MEAN),
        Observation(candidate_id="z0", batch_ids=("p2",), observed_score=PARENT_MEAN),
    ]
    features = []
    observations = []
    truth = {}
    for i in range(n_candidates):
        cand = _candidate(i)
        feats = fx.featurize(cand, parent, history=parent_history)
        features.append(feats)
        d = _true_delta(feats)
        truth[cand.candidate_id] = d
        true_abs = PARENT_MEAN + d
        for r in range(reps):
            noisy = float(true_abs + RNG.normal(0.0, NOISE_SD))
            observations.append(
                Observation(
                    candidate_id=cand.candidate_id,
                    batch_ids=(f"b{r}",),
                    observed_score=noisy,
                )
            )
    return features, observations, truth


def test_delta_target_and_absolute_reconstruction() -> None:
    features, observations, truth = _build_dataset()
    surr = BaggingEnsembleSurrogate(n_members=12, alpha=1.0, random_state=1)
    surr.fit(features, observations)
    assert surr.is_fitted
    assert surr.normalizer.n_features > 0

    preds = surr.predict_many(features)

    for p in preds:
        assert p.mu == pytest.approx(PARENT_MEAN + p.mu_delta, abs=1e-9)


    pred_delta = np.array([p.mu_delta for p in preds])
    true_delta = np.array([truth[p.candidate_id] for p in preds])
    corr = float(np.corrcoef(pred_delta, true_delta)[0, 1])
    assert corr > 0.7


def test_total_variance_decomposition() -> None:
    features, observations, _ = _build_dataset()
    surr = BaggingEnsembleSurrogate(n_members=12, random_state=2)
    surr.fit(features, observations)

    preds = surr.predict_many(features)
    for p in preds:
        assert p.sigma > 0.0
        assert p.epistemic_var >= 0.0
        assert p.aleatoric_var > 0.0

        assert p.sigma ** 2 == pytest.approx(
            p.epistemic_var + p.aleatoric_var, rel=1e-6, abs=1e-9
        )


    assert surr._aleatoric_var == pytest.approx(NOISE_SD ** 2, rel=1.5)


def test_calibration_improves_coverage() -> None:
    features, observations, _ = _build_dataset()
    surr = BaggingEnsembleSurrogate(
        n_members=16, random_state=3, calibrator=TemperatureScaler()
    )
    surr.fit(features, observations)
    assert surr.is_calibrated


    feat_by_id = {f.candidate_id: f for f in features}
    pred_by_id = {p.candidate_id: p for p in surr.predict_many(features)}
    z2 = []
    for obs in observations:
        p = pred_by_id[obs.candidate_id]
        if p.sigma > 0:
            z2.append(((obs.observed_score - p.mu) / p.sigma) ** 2)
    mean_z2 = float(np.mean(z2))
    assert 0.4 < mean_z2 < 2.5


def test_isotonic_calibrator_runs() -> None:
    features, observations, _ = _build_dataset()
    surr = BaggingEnsembleSurrogate(
        n_members=12, random_state=4, calibrator=IsotonicCalibrator(min_points=8)
    )
    surr.fit(features, observations)
    assert surr.is_calibrated
    preds = surr.predict_many(features)
    assert all(np.isfinite(p.sigma) and p.sigma > 0 for p in preds)


def test_pav_is_nondecreasing() -> None:
    y = np.array([3.0, 1.0, 2.0, 0.5, 4.0])
    g = _pav_nondecreasing(y)
    assert np.all(np.diff(g) >= -1e-9)

    assert float(np.sum(g)) == pytest.approx(float(np.sum(y)))
````
