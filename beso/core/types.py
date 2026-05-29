"""Foundational dataclasses for BESO.

These types are the shared vocabulary for every downstream module (compiler,
evaluation, reflection, edits, features, surrogate, acquisition, archive,
optimization). Field names follow the Technical Specification schemas
(skill JSON S8.1, edit schema S11.2, feature record S12.2, archive entry S15.2)
and the Mathematical Breakdown notation (history H_t S4.1, observation model
S2.4, candidate featurization S6).

Design notes encoded here:
- Skill membership in the constrained space Z is asserted by ``schema.py``
  (SchemaValid & BudgetValid & InvariantValid); these dataclasses are plain
  containers and do not self-validate.
- ``Observation`` is a single noisy minibatch evaluation tilde_y = J_B(z) + eps;
  repeated observations of the same candidate are stored as multiple rows so the
  archive/surrogate can recover bar_y and the standard error SE (Breakdown S4.1).
- Candidate features are stored as a structured ``CandidateFeatures`` record so
  the featurizer can apply per-block normalization (no flat vector leaks scale).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Enumerations                                                                  #
# --------------------------------------------------------------------------- #
class SkillSection(str, Enum):
    """Canonical sections of a skill artifact (Spec S8 schema)."""

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
    """Physical edit operator applied to the skill document.

    Aligned with SkillOpt's ``Edit.op`` vocabulary (skillopt/optimizer/skill.py
    ``apply_edit``), which performs substring operations on the markdown skill
    string. ``MERGE`` is BESO's semantic crossover M_psi(z_a, z_b) (Breakdown
    S5.4) and is lowered to concrete edits before application.
    """

    APPEND = "append"
    INSERT_AFTER = "insert_after"
    REPLACE = "replace"
    DELETE = "delete"
    MERGE = "merge"


class EditCategory(str, Enum):
    """Semantic label for an edit (BESO provenance + featurization).

    Orthogonal to the physical :class:`EditOperation`: it records *intent*
    (Spec S11.3, Breakdown S5.1) for phi_edit(z) and edit-type success-rate
    history, while the physical op is what SkillOpt's applicator executes.
    """

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
    """Disjoint dataset roles D = D_fb U D_opt U D_val U D_test (Breakdown S1.2)."""

    FEEDBACK_TRAIN = "feedback_train"
    OPTIMIZATION_MINIBATCH = "optimization_minibatch"
    VALIDATION_GATE = "validation_gate"
    FINAL_TEST = "final_test"


class CompilerMode(str, Enum):
    """Skill compiler modes C in {C_full, C_section, C_distill} (Breakdown S1.5)."""

    FULL = "full"
    SECTION = "section_selection"
    DISTILL = "distill"


class ArchiveTier(str, Enum):
    """Archive decomposition A_t (Breakdown S4.2)."""

    BEST = "best"
    PARETO = "pareto"
    DIVERSE = "diverse"
    FAILED = "failed"


# --------------------------------------------------------------------------- #
# Skill artifact (the optimization target z)                                    #
# --------------------------------------------------------------------------- #
@dataclass
class SkillMetadata:
    """Provenance/lineage metadata for a skill artifact (Spec S8.1 metadata)."""

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


# --------------------------------------------------------------------------- #
# Trajectories, evaluation, observations                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    """Observable execution trace tau on one task instance (Breakdown S1.3).

    Treated as observable trace only (no hidden chain-of-thought).
    """

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
