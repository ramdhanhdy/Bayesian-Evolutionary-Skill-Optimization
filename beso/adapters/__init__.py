"""Adapters binding external systems to BESO's protocol boundary.

The SkillOpt adapters wrap the reusable upstream "plumbing" (execution harness,
edit applicator, evaluator, dataloaders, skill serialization) so the rest of
BESO depends only on ``beso.core.protocols`` interfaces.
"""

from beso.adapters.skillopt import (
    GSM8KMiniDatasetProvider,
    ProposedEdit,
    ReflectionOutput,
    SkillOptDatasetProvider,
    SkillOptEditApplicator,
    SkillOptEvaluator,
    SkillOptHarness,
    SkillOptReflectionProposer,
    SkillOptSerializer,
    apply_markdown_edit,
    exact_match_score,
    llm_generate,
)

__all__ = [
    "GSM8KMiniDatasetProvider",
    "ProposedEdit",
    "ReflectionOutput",
    "SkillOptDatasetProvider",
    "SkillOptEditApplicator",
    "SkillOptEvaluator",
    "SkillOptHarness",
    "SkillOptReflectionProposer",
    "SkillOptSerializer",
    "apply_markdown_edit",
    "exact_match_score",
    "llm_generate",
]
