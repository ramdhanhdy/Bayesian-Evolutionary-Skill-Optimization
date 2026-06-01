"""Run a local GSM8K mini experiment through the live BESO stack.

Required environment:
    BESO_GSM8K_TRAIN_JSONL=path/to/train.jsonl
    BESO_GSM8K_VALIDATION_JSONL=path/to/validation.jsonl

Optional environment:
    BESO_GSM8K_TEST_JSONL=path/to/test.jsonl
    BESO_GSM8K_LIMIT=32
    BESO_GSM8K_TRACE_PATH=artifacts/gsm8k_mini_experiment.jsonl
"""

from __future__ import annotations

from pathlib import Path

from beso.adapters import GSM8KMiniDatasetProvider
from beso.core.types import RolloutBudget, SkillArtifact
from run_toy_experiment import MODEL, build_optimizer, env_value

INITIAL_GSM8K_SKILL = """# Skill: GSM8K Arithmetic

## Goal
Solve grade-school math word problems accurately.

## Core Procedure
- Read the problem and return 0.

## Verification Checklist
- Do not recalculate the answer.

## Output Rules
- Return only the final number.
"""


def _required_path(name: str) -> Path:
    value = env_value(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this script")
    return Path(value)


def main() -> None:
    train_path = _required_path("BESO_GSM8K_TRAIN_JSONL")
    validation_path = _required_path("BESO_GSM8K_VALIDATION_JSONL")
    test_value = env_value("BESO_GSM8K_TEST_JSONL")
    test_path = Path(test_value) if test_value else None
    limit = int(env_value("BESO_GSM8K_LIMIT", default="32"))
    trace_path = Path(
        env_value(
            "BESO_GSM8K_TRACE_PATH",
            default="artifacts/gsm8k_mini_experiment.jsonl",
        )
    )
    dataset = GSM8KMiniDatasetProvider.from_jsonl(
        train_path,
        validation_path,
        test_path=test_path,
        limit_per_split=limit,
    )
    optimizer = build_optimizer(dataset, trace_path=trace_path)

    print(f"BESO GSM8K mini experiment using LiteLLM model: {MODEL}")
    print(f"train={train_path}")
    print(f"validation={validation_path}")
    print(f"trace={trace_path}")

    result = optimizer.optimize(
        SkillArtifact(
            skill_id="gsm8k_z0",
            name="GSM8K Arithmetic",
            document=INITIAL_GSM8K_SKILL,
        ),
        RolloutBudget(
            max_rollouts=int(env_value("BESO_MAX_ROLLOUTS", default="320"))
        ),
    )
    print("\n=== GSM8K Mini Summary ===")
    print(f"rollouts spent: {result.budget.spent_rollouts}")
    print(f"iterations: {len(result.iterations)}")
    if result.best is None:
        print("best: none")
        return
    print(f"best: {result.best.candidate_id} validation={result.best.validation_mean:.3f}")
    print("\n=== Best Skill ===")
    print(result.best.artifact.document)


if __name__ == "__main__":
    main()
