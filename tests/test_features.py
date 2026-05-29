"""Verification tests for the features layer (featurizer + normalization).

Checks the three contracted properties:
1. Two-tier separation with PCA-reduced, separately-weighted text block.
2. Parent-relative delta centering of structural features.
3. Standardization + dimension balancing so no block swamps the surrogate.
"""

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
        doc = parent_doc + "\n## Verification Checklist\n" + added + "\n"
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
    # parent observed mean recovered from history.
    assert feats.history["parent_mean"] == pytest.approx(0.6)
    assert feats.history["parent_n_obs"] == 2.0
    # cold-start default for unseen edit-type success rate.
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
    # PCA reduced the 128-dim embedding to <= 8 columns.
    assert (slices["text"].stop - slices["text"].start) <= 8

    # Structural columns are standardized: ~zero mean across the corpus.
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

    # Balancing must reduce the text block's share of total variance.
    assert text_share_b < text_share_un
    assert np.all(np.isfinite(balanced))


def _block_var_share(X: np.ndarray, slices: dict, block: str) -> float:
    total = float(np.sum(np.var(X, axis=0)))
    if total <= 0:
        return 0.0
    block_var = float(np.sum(np.var(X[:, slices[block]], axis=0)))
    return block_var / total
