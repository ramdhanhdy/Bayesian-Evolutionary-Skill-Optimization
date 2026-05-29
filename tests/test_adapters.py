from __future__ import annotations

import json

from beso.adapters import (
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
            {
                "edit_id": "e1",
                "op": "append",
                "content": "- Verify evidence.",
                "target_section": "verification_checklist",
                "category": "add_rule",
            },
            {"edit_id": "e2", "op": "delete", "target": "bad rule"},
        ]
    }
    proposer = SkillOptReflectionProposer(llm=lambda prompt: json.dumps(payload))
    parent = SkillArtifact(skill_id="z0", name="Search QA", document=SKILL_MD)

    edits = proposer.propose_pool(parent, trajectories=[], rejected=[], pool_size=4)

    assert isinstance(proposer, P.ReflectionProposer)
    assert len(edits) == 2
    assert edits[0].operation is EditOperation.APPEND
    assert edits[0].target_section is SkillSection.VERIFICATION_CHECKLIST
    assert edits[1].operation is EditOperation.DELETE
