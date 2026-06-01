from __future__ import annotations

import json

from beso.adapters import (
    GSM8KMiniDatasetProvider,
    SkillOptDatasetProvider,
    SkillOptEditApplicator,
    SkillOptEvaluator,
    SkillOptHarness,
    SkillOptReflectionProposer,
    SkillOptSerializer,
)
from beso.core import protocols as P
from beso.core.types import (
    EditOperation,
    EditProposal,
    SkillArtifact,
    SkillSection,
    SplitRole,
)


SKILL_MD = """# Skill: Search QA

## Goal
Answer questions with evidence.

## Core Procedure
- Read the question.
- Verify the answer.

## Output Rules
Return the concise final answer.
"""


def _strict_edit(**overrides):
    row = {
        "edit_id": "e1",
        "op": "append",
        "content": "- Verify evidence.",
        "target": "",
        "target_section": "verification_checklist",
        "category": "add_rule",
        "rationale": "trace:none; conservative verification improvement.",
        "expected_effect": "Improve output verification.",
        "risk": "low",
        "estimated_scope": "local",
        "edit_size_tokens": 3,
    }
    row.update(overrides)
    return row


def test_serializer_round_trips_markdown_losslessly() -> None:
    serializer = SkillOptSerializer()
    skill = SkillArtifact(skill_id="z0", name="Search QA", document=SKILL_MD)

    markdown = serializer.render(skill)
    parsed = serializer.parse(markdown, skill_id="z0")

    assert isinstance(serializer, P.SkillSerializer)
    assert markdown == SKILL_MD
    assert parsed.document == SKILL_MD
    assert serializer.render(parsed) == SKILL_MD
    assert parsed.name == "Search QA"
    assert parsed.sections[SkillSection.GOAL] == "Answer questions with evidence."
    assert "- Verify the answer." in parsed.sections[SkillSection.CORE_PROCEDURE]


def test_serializer_renders_structured_sections_when_document_is_empty() -> None:
    serializer = SkillOptSerializer()
    skill = SkillArtifact(
        skill_id="z1",
        name="Structured",
        sections={
            SkillSection.GOAL: "Solve accurately.",
            SkillSection.VERIFICATION_CHECKLIST: ["Check answer", "Check format"],
        },
    )

    markdown = serializer.render(skill)
    parsed = serializer.parse(markdown, skill_id="z1")

    assert "# Skill: Structured" in markdown
    assert "## Verification Checklist" in markdown
    assert "- Check format" in parsed.sections[SkillSection.VERIFICATION_CHECKLIST]


def test_edit_applicator_applies_bounded_markdown_edits() -> None:
    applicator = SkillOptEditApplicator()
    parent = SkillArtifact(skill_id="z0", name="Search QA", document=SKILL_MD)

    appended = applicator.apply(
        parent,
        EditProposal(
            edit_id="z1",
            parent_skill_id="z0",
            operation=EditOperation.APPEND,
            content="## Recovery Rules\n- Retry narrower search.",
        ),
    )
    assert isinstance(applicator, P.EditApplicator)
    assert appended.skill_id == "z1"
    assert appended.metadata.parent_id == "z0"
    assert "Retry narrower search." in appended.document

    replaced = applicator.apply(
        appended,
        EditProposal(
            edit_id="z2",
            parent_skill_id="z1",
            operation=EditOperation.REPLACE,
            target="Return the concise final answer.",
            content="Return only the final answer.",
        ),
    )
    assert "Return only the final answer." in replaced.document
    assert "Return the concise final answer." not in replaced.document

    deleted = applicator.apply(
        replaced,
        EditProposal(
            edit_id="z3",
            parent_skill_id="z2",
            operation=EditOperation.DELETE,
            target="- Read the question.\n",
        ),
    )
    assert "- Read the question." not in deleted.document


def test_insert_after_falls_back_to_append_when_target_missing() -> None:
    applicator = SkillOptEditApplicator()
    parent = SkillArtifact(skill_id="z0", name="Search QA", document=SKILL_MD)

    child = applicator.apply(
        parent,
        EditProposal(
            edit_id="z1",
            parent_skill_id="z0",
            operation=EditOperation.INSERT_AFTER,
            target="not present",
            content="- Use citations.",
        ),
    )

    assert child.document.rstrip().endswith("- Use citations.")


def test_replace_can_target_section_when_substring_is_missing() -> None:
    applicator = SkillOptEditApplicator()
    parent = SkillArtifact(
        skill_id="z0",
        name="Toy",
        document=(
            "# Skill: Toy\n\n"
            "## Core Procedure\n"
            "- Return 0 for every question.\n\n"
            "## Output Rules\n"
            "- Return only one integer.\n"
        ),
    )

    child = applicator.apply(
        parent,
        EditProposal(
            edit_id="z1",
            parent_skill_id="z0",
            operation=EditOperation.REPLACE,
            target="the degenerate default rule",
            target_section=SkillSection.CORE_PROCEDURE,
            content="- Parse the question and compute the requested arithmetic.",
        ),
    )

    assert "- Return 0 for every question." not in child.document
    assert "- Parse the question" in child.document
    assert "## Core Procedure" in child.document
    assert "## Output Rules" in child.document


def test_delete_uses_normalized_line_match_when_target_is_not_exact() -> None:
    applicator = SkillOptEditApplicator()
    parent = SkillArtifact(
        skill_id="z0",
        name="Toy",
        document=(
            "# Skill: Toy\n\n"
            "## Core Procedure\n"
            "- Return 0 for every question.\n"
            "- Keep answers concise.\n"
        ),
    )

    child = applicator.apply(
        parent,
        EditProposal(
            edit_id="z1",
            parent_skill_id="z0",
            operation=EditOperation.DELETE,
            target="Return 0 for every question",
        ),
    )

    assert "- Return 0 for every question." not in child.document
    assert "- Keep answers concise." in child.document


def test_dataset_harness_and_evaluator_return_exact_accuracy() -> None:
    provider = SkillOptDatasetProvider(
        items_by_role={
            SplitRole.VALIDATION_GATE: [
                {"id": "q1", "question": "Capital of France?", "answers": ["Paris"]},
                {"id": "q2", "question": "Capital of Spain?", "answers": ["Madrid"]},
            ]
        }
    )
    harness = SkillOptHarness(provider, llm=lambda prompt: "Paris")
    evaluator = SkillOptEvaluator(harness)
    skill = SkillArtifact(skill_id="z0", name="Geo", document="# Skill: Geo\n")

    ids = provider.batch(SplitRole.VALIDATION_GATE, size=2, seed=1)
    result = evaluator.evaluate(skill, SplitRole.VALIDATION_GATE, ids, seed=1)

    assert isinstance(provider, P.DatasetProvider)
    assert isinstance(harness, P.ExecutionHarness)
    assert isinstance(evaluator, P.Evaluator)
    assert result.n == 2
    assert result.mean_score == 0.5
    assert result.invalid_rate == 0.0
    assert all(t.compiled_prompt.startswith("# Skill: Geo") for t in result.trajectories)


def test_reflection_proposer_parses_json_edit_pool() -> None:
    payload = {
        "edits": [
            _strict_edit(),
            _strict_edit(
                edit_id="e2",
                op="delete",
                content="",
                target="bad rule",
                target_section="core_procedure",
                category="delete_rule",
                rationale="trace:none; remove a known bad rule.",
                expected_effect="Remove the bad rule.",
                edit_size_tokens=2,
            ),
        ]
    }
    proposer = SkillOptReflectionProposer(llm=lambda prompt: json.dumps(payload))
    parent = SkillArtifact(skill_id="z0", name="Search QA", document=SKILL_MD)

    edits = proposer.propose_pool(parent, trajectories=[], rejected=[], pool_size=2)

    assert isinstance(proposer, P.ReflectionProposer)
    assert len(edits) == 2
    assert edits[0].operation is EditOperation.APPEND
    assert edits[0].target_section is SkillSection.VERIFICATION_CHECKLIST
    assert edits[1].operation is EditOperation.DELETE


def test_reflection_proposer_repairs_fractured_json() -> None:
    fixed = json.dumps(
        {
            "edits": [
                _strict_edit()
            ]
        }
    )
    responses = iter(
        [
            # Schema-invalid: the edit is missing the required "op" field.
            '{"edits": [{"content": "missing op"}]}',
            fixed,
        ]
    )
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    proposer = SkillOptReflectionProposer(llm=llm)
    parent = SkillArtifact(skill_id="z0", name="Search QA", document=SKILL_MD)

    edits = proposer.propose_pool(parent, trajectories=[], rejected=[], pool_size=1)

    assert len(edits) == 1
    assert edits[0].operation is EditOperation.APPEND
    assert edits[0].target_section is SkillSection.VERIFICATION_CHECKLIST
    # The model was re-prompted with a repair instruction containing the error.
    assert any("Fix the JSON schema" in p for p in prompts)
    assert any("Validation Error" in p for p in prompts)


def test_reflection_proposer_drops_invalid_edits_after_failed_repair() -> None:
    broken = json.dumps(
        {
            "edits": [
                _strict_edit(edit_id="ok", content="- Keep this."),
                {"edit_id": "bad", "content": "no op here"},
            ]
        }
    )
    calls = {"n": 0}

    def llm(prompt: str) -> str:
        calls["n"] += 1
        return broken

    proposer = SkillOptReflectionProposer(llm=llm, max_repair_attempts=2)
    parent = SkillArtifact(skill_id="z0", name="Search QA", document=SKILL_MD)

    edits = proposer.propose_pool(parent, trajectories=[], rejected=[], pool_size=2)

    # Repair never succeeds (always the same broken payload), so the proposer
    # degrades by keeping the valid edit and dropping the invalid one.
    assert [e.edit_id for e in edits] == ["ok"]
    assert calls["n"] == 3  # initial attempt + 2 repair retries


def test_reflection_proposer_rejects_oversized_edit_content() -> None:
    oversized = json.dumps(
        {"edits": [_strict_edit(content="word " * 501, edit_size_tokens=501)]}
    )
    proposer = SkillOptReflectionProposer(
        llm=lambda prompt: oversized,
        max_repair_attempts=0,
    )
    parent = SkillArtifact(skill_id="z0", name="Search QA", document=SKILL_MD)

    edits = proposer.propose_pool(parent, trajectories=[], rejected=[], pool_size=1)

    assert edits == []


def test_reflection_proposer_repairs_undersized_pool() -> None:
    undersized = json.dumps({"edits": [_strict_edit()]})
    fixed = json.dumps(
        {
            "edits": [
                _strict_edit(),
                _strict_edit(
                    edit_id="e2",
                    content="- Add a recovery rule.",
                    target_section="recovery_rules",
                    category="add_recovery_rule",
                    rationale="trace:none; add a bounded recovery route.",
                    expected_effect="Improve recovery.",
                    edit_size_tokens=5,
                ),
            ]
        }
    )
    responses = iter([undersized, fixed])
    proposer = SkillOptReflectionProposer(llm=lambda prompt: next(responses))
    parent = SkillArtifact(skill_id="z0", name="Search QA", document=SKILL_MD)

    edits = proposer.propose_pool(parent, trajectories=[], rejected=[], pool_size=2)

    assert [edit.edit_id for edit in edits] == ["e1", "e2"]


def test_reflection_proposer_salvage_drops_duplicate_edit_ids() -> None:
    duplicate_ids = json.dumps(
        {
            "edits": [
                _strict_edit(edit_id="same"),
                _strict_edit(
                    edit_id="same",
                    content="- Add a recovery rule.",
                    target_section="recovery_rules",
                    category="add_recovery_rule",
                    rationale="trace:none; add a bounded recovery route.",
                    expected_effect="Improve recovery.",
                    edit_size_tokens=5,
                ),
            ]
        }
    )
    proposer = SkillOptReflectionProposer(
        llm=lambda prompt: duplicate_ids,
        max_repair_attempts=0,
    )
    parent = SkillArtifact(skill_id="z0", name="Search QA", document=SKILL_MD)

    edits = proposer.propose_pool(parent, trajectories=[], rejected=[], pool_size=2)

    assert [edit.edit_id for edit in edits] == ["same"]


def test_gsm8k_mini_provider_loads_standard_jsonl(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text(
        json.dumps({"question": "What is 2 + 3?", "answer": "Compute. #### 5"}) + "\n",
        encoding="utf-8-sig",
    )
    val_path.write_text(
        json.dumps({"question": "What is 10 - 4?", "answer": "Compute. #### 6"}) + "\n",
        encoding="utf-8",
    )

    provider = GSM8KMiniDatasetProvider.from_jsonl(train_path, val_path)

    train_ids = provider.batch(SplitRole.OPTIMIZATION_MINIBATCH, size=1, seed=1)
    val_ids = provider.batch(SplitRole.VALIDATION_GATE, size=1, seed=1)
    assert provider.item(train_ids[0])["answers"] == ["5"]
    assert provider.item(val_ids[0])["answers"] == ["6"]
