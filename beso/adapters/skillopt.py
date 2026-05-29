"""Concrete SkillOpt-compatible adapters.

These adapters keep the BESO math engine isolated from provider SDKs and from
the upstream SkillOpt package. They implement the protocol shape with plain
Python callables and local files so real LLM clients or benchmark harnesses can
be injected later without changing the optimizer.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from beso.core.types import (
    EditCategory,
    EditOperation,
    EditProposal,
    EvaluationResult,
    SkillArtifact,
    SkillMetadata,
    SkillSection,
    SplitRole,
    Trajectory,
)
from beso.features.featurizer import approx_tokens

LLMGenerate = Callable[[str], str]
TrajectoryScorer = Callable[[Trajectory], float]

SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def llm_generate(prompt: str) -> str:
    """Placeholder LLM boundary.

    Production code should inject a provider-specific callable with the same
    signature. The default is intentionally inert and deterministic.
    """

    return ""


class SkillOptSerializer:
    """Maps ``SkillArtifact`` <-> SkillOpt markdown (`best_skill.md`)."""

    def render(self, skill: SkillArtifact) -> str:
        """Serialize a skill artifact to markdown.

        ``document`` is the source of truth. If present, it is returned exactly
        so existing SkillOpt markdown round-trips without metadata churn.
        """

        if skill.document:
            return skill.document
        parts = [f"# Skill: {skill.name or skill.skill_id}".rstrip()]
        for section in SkillSection:
            if section not in skill.sections:
                continue
            rendered = _render_section_value(skill.sections[section])
            if rendered:
                parts.append(f"## {_section_title(section)}\n{rendered}")
        return "\n\n".join(parts).rstrip() + "\n"

    def parse(self, markdown: str, skill_id: str) -> SkillArtifact:
        """Parse markdown into a typed artifact while preserving raw text."""

        markdown = markdown or ""
        name = _parse_skill_name(markdown) or skill_id
        sections = _parse_sections(markdown)
        return SkillArtifact(
            skill_id=skill_id,
            name=name,
            document=markdown,
            sections=sections,
            metadata=SkillMetadata(token_count=approx_tokens(markdown)),
        )


class SkillOptDatasetProvider:
    """Loads split-directory JSON/JSONL datasets and returns deterministic ids."""

    def __init__(
        self,
        split_dir: str | Path | None = None,
        *,
        items_by_role: Mapping[SplitRole, Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        self.split_dir = Path(split_dir) if split_dir is not None else None
        self._items: dict[SplitRole, list[dict[str, Any]]] = {}
        self._by_id: dict[str, dict[str, Any]] = {}
        if items_by_role is not None:
            for role, items in items_by_role.items():
                self._items[role] = [dict(item) for item in items]
        elif self.split_dir is not None:
            self._load_split_dir(self.split_dir)
        self._index_items()

    def batch(self, role: SplitRole, size: int, seed: int) -> Sequence[str]:
        items = self._items_for_role(role)
        if size <= 0 or not items:
            return []
        rng = random.Random(seed)
        ids = [_item_id(item, i, role) for i, item in enumerate(items)]
        if size >= len(ids):
            shuffled = list(ids)
            rng.shuffle(shuffled)
            return shuffled[:size]
        return rng.sample(ids, size)

    def split_size(self, role: SplitRole) -> int:
        return len(self._items_for_role(role))

    def item(self, example_id: str) -> dict[str, Any]:
        return dict(self._by_id[example_id])

    def _load_split_dir(self, split_dir: Path) -> None:
        split_names = {
            SplitRole.FEEDBACK_TRAIN: "train",
            SplitRole.OPTIMIZATION_MINIBATCH: "train",
            SplitRole.VALIDATION_GATE: "val",
            SplitRole.FINAL_TEST: "test",
        }
        loaded_by_name: dict[str, list[dict[str, Any]]] = {}
        for split_name in set(split_names.values()):
            loaded_by_name[split_name] = _load_items(split_dir / split_name)
        for role, split_name in split_names.items():
            self._items[role] = list(loaded_by_name.get(split_name, []))

    def _items_for_role(self, role: SplitRole) -> list[dict[str, Any]]:
        return self._items.get(role, [])

    def _index_items(self) -> None:
        for role, items in list(self._items.items()):
            normalized: list[dict[str, Any]] = []
            for idx, item in enumerate(items):
                row = dict(item)
                row.setdefault("id", _item_id(row, idx, role))
                normalized.append(row)
                self._by_id[str(row["id"])] = row
            self._items[role] = normalized


class SkillOptHarness:
    """Minimal execution harness using an injectable LLM generator."""

    def __init__(
        self,
        dataset: SkillOptDatasetProvider,
        *,
        llm: LLMGenerate | None = None,
        serializer: SkillOptSerializer | None = None,
        scorer: Callable[[str, dict[str, Any]], float] | None = None,
    ) -> None:
        self.dataset = dataset
        self.llm = llm or llm_generate
        self.serializer = serializer or SkillOptSerializer()
        self.scorer = scorer or exact_match_score

    def rollout(
        self,
        skill: SkillArtifact,
        example_ids: Sequence[str],
        seed: int,
    ) -> list[Trajectory]:
        skill_markdown = self.serializer.render(skill)
        trajectories: list[Trajectory] = []
        for example_id in example_ids:
            item = self.dataset.item(str(example_id))
            task_input = _task_input(item)
            prompt = _compile_prompt(skill_markdown, task_input)
            output = self.llm(prompt)
            score = float(self.scorer(output, item))
            trajectories.append(
                Trajectory(
                    example_id=str(example_id),
                    task_input=task_input,
                    compiled_prompt=prompt,
                    output=output,
                    score=score,
                    feedback=str(item.get("feedback", "")),
                    cost_tokens=float(approx_tokens(prompt) + approx_tokens(output)),
                    valid_output=bool(output.strip()),
                )
            )
        return trajectories


class SkillOptEditApplicator:
    """Deterministic markdown edit applicator using SkillOpt edit semantics."""

    def __init__(self, serializer: SkillOptSerializer | None = None) -> None:
        self.serializer = serializer or SkillOptSerializer()

    def apply(self, parent: SkillArtifact, edit: EditProposal) -> SkillArtifact:
        document = self.serializer.render(parent)
        updated = apply_markdown_edit(document, edit)
        child_id = edit.edit_id or f"{parent.skill_id}_edit"
        parsed = self.serializer.parse(updated, skill_id=child_id)
        parsed.name = parent.name
        parsed.version = parent.version + 1
        parsed.metadata.parent_id = parent.skill_id
        parsed.metadata.lineage_depth = parent.metadata.lineage_depth + 1
        parsed.metadata.edit_summary = edit.rationale or edit.expected_effect
        parsed.metadata.created_by = "optimizer"
        return parsed

    def apply_sequence(
        self,
        parent: SkillArtifact,
        edits: Sequence[EditProposal],
    ) -> SkillArtifact:
        skill = parent
        for edit in edits:
            skill = self.apply(skill, edit)
        return skill


class SkillOptEvaluator:
    """Aggregates scored trajectories from a harness into BESO eval results."""

    def __init__(
        self,
        harness: SkillOptHarness | None = None,
        *,
        scorer: TrajectoryScorer | None = None,
    ) -> None:
        self.harness = harness
        self.scorer = scorer

    def score_trajectory(self, trajectory: Trajectory) -> float:
        if self.scorer is not None:
            return float(self.scorer(trajectory))
        return float(trajectory.score if trajectory.valid_output else 0.0)

    def evaluate(
        self,
        skill: SkillArtifact,
        role: SplitRole,
        example_ids: Sequence[str],
        seed: int,
    ) -> EvaluationResult:
        if self.harness is None:
            raise RuntimeError("SkillOptEvaluator.evaluate requires a harness")
        trajectories = self.harness.rollout(skill, example_ids, seed)
        scores = {t.example_id: self.score_trajectory(t) for t in trajectories}
        return EvaluationResult(
            candidate_id=skill.skill_id,
            split=role,
            per_example_scores=scores,
            trajectories=trajectories,
            invalid_rate=_invalid_rate(trajectories),
            mean_cost_tokens=_mean([t.cost_tokens for t in trajectories]),
            mean_latency_seconds=_mean([t.latency_seconds for t in trajectories]),
        )


class SkillOptReflectionProposer:
    """LLM-backed reflection proposer returning bounded edit pools."""

    def __init__(
        self,
        *,
        llm: LLMGenerate | None = None,
        max_completion_edits: int = 24,
    ) -> None:
        self.llm = llm or llm_generate
        self.max_completion_edits = int(max_completion_edits)

    def propose_pool(
        self,
        parent: SkillArtifact,
        trajectories: Sequence[Trajectory],
        rejected: Sequence[EditProposal],
        pool_size: int,
    ) -> list[EditProposal]:
        prompt = _reflection_prompt(parent, trajectories, rejected, pool_size)
        raw = self.llm(prompt)
        edits = _parse_edit_payload(raw, parent.skill_id)
        return edits[: min(pool_size, self.max_completion_edits)]


def apply_markdown_edit(markdown: str, edit: EditProposal) -> str:
    """Apply one bounded edit to a markdown skill document."""

    op = edit.operation
    content = _strip_slow_update_markers(edit.content.strip())
    target = edit.target or ""
    if target and _is_in_slow_update_region(markdown, target):
        return markdown
    if op is EditOperation.APPEND:
        return _append(markdown, content)
    if op is EditOperation.INSERT_AFTER:
        return _insert_after(markdown, target, content)
    if op is EditOperation.REPLACE:
        return _replace(markdown, target, content, edit.target_section)
    if op is EditOperation.DELETE:
        return _delete(markdown, target, edit.target_section)
    if op is EditOperation.MERGE:
        return _append(markdown, content)
    return markdown


def exact_match_score(output: str, item: Mapping[str, Any]) -> float:
    """Simple exact-accuracy scorer for benchmark scaffolding."""

    expected = _expected_answers(item)
    if not expected:
        return 0.0
    normalized = _normalize_answer(output)
    return 1.0 if any(normalized == _normalize_answer(ans) for ans in expected) else 0.0


def _append(markdown: str, content: str) -> str:
    su_start = markdown.find(SLOW_UPDATE_START)
    if su_start != -1:
        before = markdown[:su_start].rstrip()
        after = markdown[su_start:]
        return before + "\n\n" + content + "\n\n" + after
    return markdown.rstrip() + "\n\n" + content + "\n"


def _insert_after(markdown: str, target: str, content: str) -> str:
    if not target or target not in markdown:
        return _append(markdown, content)
    idx = markdown.index(target) + len(target)
    newline = markdown.find("\n", idx)
    insert_at = newline + 1 if newline != -1 else len(markdown)
    return markdown[:insert_at] + "\n" + content + "\n" + markdown[insert_at:]


def _replace(
    markdown: str,
    target: str,
    content: str,
    section: SkillSection | None,
) -> str:
    if target:
        if target in markdown:
            return markdown.replace(target, content, 1)
        line_replaced = _replace_fuzzy_line(markdown, target, content)
        if line_replaced is not None:
            return line_replaced
    if section is not None:
        return _replace_section(markdown, section, content)
    return markdown


def _delete(
    markdown: str,
    target: str,
    section: SkillSection | None,
) -> str:
    if target:
        if target in markdown:
            return markdown.replace(target, "", 1)
        line_deleted = _replace_fuzzy_line(markdown, target, "")
        if line_deleted is not None:
            return line_deleted
    if section is not None:
        return _replace_section(markdown, section, "")
    return markdown


def _replace_fuzzy_line(
    markdown: str,
    target: str,
    content: str,
) -> str | None:
    target_key = _line_match_key(target)
    if not target_key:
        return None
    parts = markdown.splitlines(keepends=True)
    for idx, line in enumerate(parts):
        line_key = _line_match_key(line)
        if not line_key:
            continue
        if target_key == line_key or (
            len(target_key) >= 8
            and (target_key in line_key or line_key in target_key)
        ):
            newline = "\n" if line.endswith("\n") else ""
            replacement = content.rstrip()
            parts[idx] = f"{replacement}{newline}" if replacement else ""
            return "".join(parts)
    return None


def _replace_section(
    markdown: str,
    section: SkillSection,
    content: str,
) -> str:
    span = _section_span(markdown, section)
    if span is None:
        section_text = f"## {_section_title(section)}"
        if content.strip():
            section_text += f"\n{content.strip()}"
        return _append(markdown, section_text)

    section_start, heading_end, body_start, section_end = span
    replacement = content.strip()
    if _starts_with_heading(replacement):
        before = markdown[:section_start].rstrip()
        after = markdown[section_end:].lstrip("\n")
        return _join_markdown_blocks(before, replacement, after)

    before = markdown[:body_start].rstrip()
    after = markdown[section_end:].lstrip("\n")
    return _join_markdown_blocks(before, replacement, after)


def _section_span(
    markdown: str,
    section: SkillSection,
) -> tuple[int, int, int, int] | None:
    matches = list(_HEADING_RE.finditer(markdown))
    for idx, match in enumerate(matches):
        if len(match.group(1)) != 2 or _section_key(match.group(2)) is not section:
            continue
        section_start = match.start()
        heading_end = match.end()
        body_start = heading_end
        if body_start < len(markdown) and markdown[body_start] == "\n":
            body_start += 1
        section_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        return section_start, heading_end, body_start, section_end
    return None


def _starts_with_heading(text: str) -> bool:
    return bool(_HEADING_RE.match(text.strip()))


def _join_markdown_blocks(*blocks: str) -> str:
    cleaned = [block.strip() for block in blocks if block.strip()]
    return "\n\n".join(cleaned).rstrip() + "\n"


def _line_match_key(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"^[\s>*+\-`]+", "", text)
    text = re.sub(r"`+", "", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _is_in_slow_update_region(markdown: str, target: str) -> bool:
    start_idx = markdown.find(SLOW_UPDATE_START)
    end_idx = markdown.find(SLOW_UPDATE_END)
    if start_idx == -1 or end_idx == -1:
        return False
    target_idx = markdown.find(target)
    if target_idx == -1:
        return False
    return start_idx <= target_idx < end_idx + len(SLOW_UPDATE_END)


def _strip_slow_update_markers(text: str) -> str:
    return text.replace(SLOW_UPDATE_START, "").replace(SLOW_UPDATE_END, "")


def _parse_skill_name(markdown: str) -> str:
    for match in _HEADING_RE.finditer(markdown):
        if match.group(1) == "#":
            title = match.group(2).strip()
            return title.removeprefix("Skill:").strip()
    return ""


def _parse_sections(markdown: str) -> dict[SkillSection, str]:
    matches = list(_HEADING_RE.finditer(markdown))
    sections: dict[SkillSection, str] = {}
    for idx, match in enumerate(matches):
        if len(match.group(1)) != 2:
            continue
        key = _section_key(match.group(2))
        if key is None:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        sections[key] = markdown[start:end].strip()
    return sections


def _section_key(title: str) -> SkillSection | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", title.strip().lower()).strip("_")
    for section in SkillSection:
        if section.value == normalized:
            return section
    return None


def _section_title(section: SkillSection) -> str:
    return section.value.replace("_", " ").title()


def _render_section_value(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)
    return str(value).strip()


def _load_items(split_path: Path) -> list[dict[str, Any]]:
    if not split_path.exists():
        return []
    json_path = split_path / "items.json"
    jsonl_path = split_path / "items.jsonl"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{json_path} must contain a JSON array")
        return [dict(item) for item in data]
    if jsonl_path.exists():
        rows = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(dict(json.loads(line)))
        return rows
    return []


def _item_id(item: Mapping[str, Any], idx: int, role: SplitRole) -> str:
    return str(item.get("id") or f"{role.value}_{idx}")


def _task_input(item: Mapping[str, Any]) -> str:
    for key in ("input", "question", "prompt", "task", "task_input"):
        if key in item:
            return str(item[key])
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


def _expected_answers(item: Mapping[str, Any]) -> list[str]:
    for key in ("answers", "expected", "answer", "label", "target", "output"):
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, list):
            return [str(v) for v in value]
        return [str(value)]
    return []


def _compile_prompt(skill_markdown: str, task_input: str) -> str:
    return f"{skill_markdown.rstrip()}\n\n## Task\n{task_input}\n"


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _invalid_rate(trajectories: Sequence[Trajectory]) -> float:
    if not trajectories:
        return 0.0
    invalid = sum(1 for t in trajectories if not t.valid_output)
    return invalid / len(trajectories)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _reflection_prompt(
    parent: SkillArtifact,
    trajectories: Sequence[Trajectory],
    rejected: Sequence[EditProposal],
    pool_size: int,
) -> str:
    failures = [t for t in trajectories if t.score <= 0.0]
    rejected_lines = [
        f"- {e.operation.value}: {e.content[:160]}" for e in rejected[-10:]
    ]
    failure_lines = [
        (
            f"- {t.example_id}: score={t.score} "
            f"input={t.task_input[:160]!r} output={t.output[:160]!r} "
            f"feedback={t.feedback[:160]!r}"
        )
        for t in failures[:10]
    ]
    return (
        "Propose bounded SkillOpt edits as JSON with an 'edits' array. "
        "Each edit must include op, content, and optional target.\n\n"
        f"Pool size: {pool_size}\n\n"
        f"## Current Skill\n{parent.document}\n\n"
        "## Recent Failures\n"
        + ("\n".join(failure_lines) or "- none")
        + "\n\n## Rejected Edits\n"
        + ("\n".join(rejected_lines) or "- none")
    )


def _parse_edit_payload(raw: str, parent_skill_id: str) -> list[EditProposal]:
    payload = _extract_json(raw)
    if payload is None:
        return []
    rows = payload.get("edits", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    edits: list[EditProposal] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        op = _parse_operation(row.get("op") or row.get("operation"))
        if op is None:
            continue
        category = _parse_category(row.get("category"))
        section = _parse_section(row.get("target_section") or row.get("section"))
        edits.append(
            EditProposal(
                edit_id=str(row.get("edit_id") or f"edit_{idx}"),
                parent_skill_id=parent_skill_id,
                operation=op,
                content=str(row.get("content") or row.get("proposed_text") or ""),
                target=str(row.get("target") or ""),
                category=category,
                target_section=section,
                source_type=row.get("source_type"),
                rationale=str(row.get("rationale") or ""),
                expected_effect=str(row.get("expected_effect") or ""),
                risk=str(row.get("risk") or ""),
                estimated_scope=str(row.get("estimated_scope") or ""),
                edit_size_tokens=int(row.get("edit_size_tokens") or 0),
            )
        )
    return edits


def _extract_json(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start_candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not start_candidates:
        return None
    start = min(start_candidates)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _parse_operation(value: object) -> EditOperation | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    aliases = {"add": "append", "remove": "delete"}
    raw = aliases.get(raw, raw)
    try:
        return EditOperation(raw)
    except ValueError:
        return None


def _parse_category(value: object) -> EditCategory | None:
    if value is None:
        return None
    try:
        return EditCategory(str(value).strip().lower())
    except ValueError:
        return None


def _parse_section(value: object) -> SkillSection | None:
    if value is None:
        return None
    return _section_key(str(value))


__all__ = [
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
