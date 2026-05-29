"""SkillOpt adapter stubs (binding points for the upstream fork).

These classes declare how BESO's protocol boundary maps onto the real SkillOpt
package (``skillopt``). They are intentionally unbound: each method documents
which upstream component it should delegate to, and raises
:class:`NotImplementedError` until the SkillOpt dependency is vendored/installed
and the concrete symbols are wired in.

Upstream reference (github.com/microsoft/SkillOpt, MIT):
- ``skillopt/envs/<benchmark>/dataloader.py`` -> task items + train/val/test splits.
- Harness adapter interface -> inject skill, run native harness, return scored traces.
- Deterministic SKILL.md edit applicator/patcher -> apply add/delete/replace edits.
- Evaluator/parsers -> turn raw agent traces into metric scores.
- Skill rendering/parsing -> best_skill.md / skill_vXXXX.md <-> structured sections.

Binding policy: keep these adapters thin. Any optimization/selection logic must
live in BESO modules (beso/surrogate, beso/acquisition, beso/optimization), never
here, so the SkillOpt baseline can reuse the exact same adapters unchanged.
"""

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
    """Maps SkillArtifact <-> SkillOpt markdown (best_skill.md / skill_vXXXX.md)."""

    def render(self, skill: SkillArtifact) -> str:
        raise NotImplementedError(_UNBOUND)

    def parse(self, markdown: str, skill_id: str) -> SkillArtifact:
        raise NotImplementedError(_UNBOUND)


class SkillOptDatasetProvider:
    """Wraps skillopt/envs/<benchmark>/dataloader.py + split directories."""

    def batch(self, role: SplitRole, size: int, seed: int) -> Sequence[str]:
        raise NotImplementedError(_UNBOUND)

    def split_size(self, role: SplitRole) -> int:
        raise NotImplementedError(_UNBOUND)


class SkillOptHarness:
    """Wraps SkillOpt's harness-agnostic adapter (inject skill -> run -> trace)."""

    def rollout(
        self, skill: SkillArtifact, example_ids: Sequence[str], seed: int
    ) -> list[Trajectory]:
        raise NotImplementedError(_UNBOUND)


class SkillOptEditApplicator:
    """Wraps SkillOpt's deterministic SKILL.md patcher."""

    def apply(self, parent: SkillArtifact, edit: EditProposal) -> SkillArtifact:
        raise NotImplementedError(_UNBOUND)

    def apply_sequence(
        self, parent: SkillArtifact, edits: Sequence[EditProposal]
    ) -> SkillArtifact:
        raise NotImplementedError(_UNBOUND)


class SkillOptEvaluator:
    """Wraps SkillOpt's evaluator/parsers into the metric mu."""

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
