from __future__ import annotations

from beso.archive import ArchiveConfig, EvolutionaryArchive, pareto_front
from beso.core.types import (
    ArchiveTier,
    Candidate,
    CandidateFeatures,
    EvaluationResult,
    SkillArtifact,
    SplitRole,
)


def _candidate(cid: str, doc: str, x: float) -> Candidate:
    skill = SkillArtifact(skill_id=cid, name=cid, document=doc)
    features = CandidateFeatures(
        candidate_id=cid,
        structural={"child_tokens": 10.0 + x},
        semantic={f"axis_{int(x)}": 1.0},
    )
    return Candidate(candidate_id=cid, skill=skill, features=features)


def _eval(cid: str, scores: dict[str, float]) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=cid,
        split=SplitRole.VALIDATION_GATE,
        per_example_scores=scores,
    )


def test_archive_stores_feature_lookup_and_pareto_win_counts() -> None:
    archive = EvolutionaryArchive(
        ArchiveConfig(
            max_size=4,
            top_by_validation=3,
            top_by_pareto=2,
            top_by_diversity=1,
            top_failed_informative=0,
        )
    )
    cands = [
        _candidate("z1", "alpha", 1.0),
        _candidate("z2", "beta", 2.0),
        _candidate("z3", "gamma", 3.0),
    ]
    evals = [
        _eval("z1", {"a": 1.0, "b": 0.0}),
        _eval("z2", {"a": 0.0, "b": 1.0}),
        _eval("z3", {"a": 0.5, "b": 0.5}),
    ]

    archive.update(cands, evals)
    entries = {entry.candidate_id: entry for entry in archive.entries()}

    assert archive.feature_lookup("z1") is not None
    assert entries["z1"].pareto_win_count == 1
    assert entries["z2"].pareto_win_count == 1
    assert entries["z3"].pareto_win_count == 0
    assert archive.best() is not None


def test_archive_prunes_to_size_cap_and_keeps_diverse_entries() -> None:
    archive = EvolutionaryArchive(
        ArchiveConfig(
            max_size=3,
            top_by_validation=1,
            top_by_pareto=1,
            top_by_diversity=3,
            top_failed_informative=0,
        )
    )
    cands = [
        _candidate("z1", "cluster same one", 1.0),
        _candidate("z2", "cluster same one", 1.1),
        _candidate("z3", "distant verification policy", 5.0),
        _candidate("z4", "another distant recovery rule", 9.0),
    ]
    evals = [
        _eval("z1", {"a": 0.9, "b": 0.8}),
        _eval("z2", {"a": 0.89, "b": 0.8}),
        _eval("z3", {"a": 0.7, "b": 0.7}),
        _eval("z4", {"a": 0.65, "b": 0.65}),
    ]

    archive.update(cands, evals)
    ids = {entry.candidate_id for entry in archive.entries()}
    tiers = {entry.tier for entry in archive.entries()}

    assert len(ids) == 3
    assert "z1" in ids
    assert {"z3", "z4"} & ids
    assert ArchiveTier.DIVERSE in tiers or ArchiveTier.PARETO in tiers


def test_pareto_front_keeps_tradeoff_candidate() -> None:
    archive = EvolutionaryArchive()
    cands = [
        _candidate("high_quality", "high", 1.0),
        _candidate("low_cost", "low", 2.0),
    ]
    evals = [
        _eval("high_quality", {"a": 1.0, "b": 1.0}),
        _eval("low_cost", {"a": 0.8, "b": 0.8}),
    ]
    archive.update(cands, evals)
    entries = archive.entries()
    by_id = {entry.candidate_id: entry for entry in entries}
    by_id["high_quality"].cost_per_task = 100.0
    by_id["low_cost"].cost_per_task = 1.0

    front = pareto_front(entries)

    assert {entry.candidate_id for entry in front} == {"high_quality", "low_cost"}
