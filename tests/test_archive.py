from __future__ import annotations

from beso.archive import ArchiveConfig, EvolutionaryArchive, pareto_front
from beso.core.types import (
    ArchiveAdmissionMode,
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


def test_cleanup_entry_cannot_evict_deployable_incumbent_under_tight_cap() -> None:
    archive = EvolutionaryArchive(
        ArchiveConfig(
            max_size=1,
            top_by_validation=1,
            top_by_pareto=1,
            top_by_diversity=1,
            top_failed_informative=0,
        )
    )
    incumbent = _candidate("a_incumbent", "stable", 1.0)
    cleanup = _candidate("z_cleanup", "stable but cheaper", 2.0)

    archive.update([incumbent], [_eval("a_incumbent", {"a": 1.0})])
    archive.update(
        [cleanup],
        [_eval("z_cleanup", {"a": 1.0})],
        cleanup_ids=["z_cleanup"],
    )

    assert archive.best() is not None
    assert archive.best().candidate_id == "a_incumbent"
    assert [entry.candidate_id for entry in archive.entries()] == ["a_incumbent"]


def test_cleanup_entry_with_raw_score_increase_stays_archive_only() -> None:
    archive = EvolutionaryArchive()
    incumbent = _candidate("incumbent", "stable", 1.0)
    cleanup = _candidate("cleanup", "cheaper experiment", 2.0)

    archive.update([incumbent], [_eval("incumbent", {"a": 0.0, "b": 1.0})])
    archive.update(
        [cleanup],
        [_eval("cleanup", {"a": 1.0, "b": 1.0})],
        cleanup_ids=["cleanup"],
    )

    assert archive.best() is not None
    assert archive.best().candidate_id == "incumbent"
    assert {entry.candidate_id for entry in archive.entries()} == {
        "cleanup",
        "incumbent",
    }


def test_exploration_entry_is_parent_eligible_but_not_deployable_best() -> None:
    archive = EvolutionaryArchive(
        ArchiveConfig(
            max_size=4,
            top_by_validation=2,
            top_by_pareto=2,
            top_by_diversity=2,
            top_failed_informative=0,
        )
    )
    incumbent = _candidate("incumbent", "stable", 1.0)
    specialist = _candidate("specialist", "specialist rule", 2.0)

    archive.update([incumbent], [_eval("incumbent", {"a": 0.0, "b": 1.0})])
    archive.update(
        [specialist],
        [_eval("specialist", {"a": 1.0, "b": 1.0})],
        exploration_ids=["specialist"],
        admission_reasons={"specialist": ["per_example_specialist"]},
    )

    by_id = {entry.candidate_id: entry for entry in archive.entries()}
    assert archive.best() is not None
    assert archive.best().candidate_id == "incumbent"
    assert by_id["specialist"].admission_mode is ArchiveAdmissionMode.ARCHIVE_EXPLORATION
    assert not by_id["specialist"].deployable_eligible
    assert by_id["specialist"].archive_parent_eligible
    assert by_id["specialist"].admission_reasons == ["per_example_specialist"]
    assert by_id["specialist"].best_exclusion_reason == "archive_only_exploration"
    assert "specialist" in {entry.candidate_id for entry in archive.select_parents(2, seed=7)}


def test_failed_archive_only_entry_is_not_parent_eligible() -> None:
    archive = EvolutionaryArchive(
        ArchiveConfig(
            max_invalid_rate=0.0,
            top_by_validation=0,
            top_by_pareto=0,
            top_by_diversity=0,
            top_failed_informative=1,
        )
    )
    invalid = _candidate("invalid", "bad format", 1.0)
    invalid_eval = EvaluationResult(
        candidate_id="invalid",
        split=SplitRole.VALIDATION_GATE,
        per_example_scores={"a": 1.0},
        invalid_rate=1.0,
    )

    archive.update(
        [invalid],
        [invalid_eval],
        exploration_ids=["invalid"],
        admission_reasons={"invalid": ["meaningful_novelty"]},
    )

    entry = archive.entries()[0]
    assert entry.tier is ArchiveTier.FAILED
    assert entry.admission_mode is ArchiveAdmissionMode.ARCHIVE_EXPLORATION
    assert not entry.archive_parent_eligible
    assert archive.select_parents(1, seed=7) == []


def test_parent_selection_table_exposes_weight_terms_and_selected_flag() -> None:
    archive = EvolutionaryArchive(
        ArchiveConfig(
            max_size=4,
            top_by_validation=3,
            top_by_pareto=3,
            top_by_diversity=3,
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
    selected = archive.select_parents(1, seed=5)

    table = archive.parent_selection_table(
        1,
        seed=5,
        selected_ids=[entry.candidate_id for entry in selected],
    )

    assert table["seed"] == 5
    assert table["requested_parent_count"] == 1
    assert table["eligible_count"] == 3
    assert abs(sum(row["probability"] for row in table["eligible"]) - 1.0) < 1e-9
    assert any(row["selected"] for row in table["eligible"])
    assert all("weighted_logit" in row["terms"] for row in table["eligible"])
    assert all("admission_mode" in row for row in table["eligible"])
