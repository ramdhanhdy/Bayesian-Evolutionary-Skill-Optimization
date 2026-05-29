"""Adapters binding external systems to BESO's protocol boundary.

The SkillOpt adapters wrap the reusable upstream "plumbing" (execution harness,
edit applicator, evaluator, dataloaders, skill serialization) so the rest of
BESO depends only on ``beso.core.protocols`` interfaces.
"""
