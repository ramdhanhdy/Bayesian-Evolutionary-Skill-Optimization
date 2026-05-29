"""Run a tiny live BESO experiment through LiteLLM.

Required environment:
    OPENROUTER_API_KEY=...
    # or
    DEEPSEEK_API_KEY=...

Optional environment:
    BESO_LITELLM_MODEL=deepseek/deepseek-chat
    BESO_LITELLM_PROVIDER=deepseek
    BESO_LITELLM_API_KEY=...
    BESO_OPENROUTER_MODEL=openrouter/openai/gpt-5
    BESO_OPENROUTER_API_BASE=https://openrouter.ai/api/v1
    BESO_OPENROUTER_SITE_URL=https://example.com
    BESO_OPENROUTER_APP_NAME=BESO Toy Experiment
    BESO_DEEPSEEK_MODEL=deepseek/deepseek-chat
    BESO_DEEPSEEK_API_BASE=https://api.deepseek.com
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from beso.acquisition import (
    AcquisitionConfig,
    BatchSelectionConfig,
    GreedySubmodularBatchSelector,
    PoolNormalizedBESOAcquisition,
)
from beso.adapters import (
    SkillOptDatasetProvider,
    SkillOptEditApplicator,
    SkillOptEvaluator,
    SkillOptHarness,
    SkillOptReflectionProposer,
)
from beso.archive import ArchiveConfig, EvolutionaryArchive
from beso.core.types import (
    ArchiveEntry,
    Candidate,
    EvaluationResult,
    RolloutBudget,
    SkillArtifact,
    SplitRole,
    SurrogatePrediction,
)
from beso.features import FeatureExtractor, HashingEmbedder
from beso.optimization import (
    AcceptanceGateConfig,
    BESOOptimizer,
    BESOOptimizerConfig,
    PairedBootstrapAcceptanceGate,
    RegimeDetectorConfig,
    VarianceRankRegimeDetector,
)
from beso.surrogate import BaggingEnsembleSurrogate, IsotonicCalibrator

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv_file(path: Path) -> None:
    """Minimal .env loader so the example works without python-dotenv."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file(_PROJECT_ROOT / ".env")
load_dotenv_file(Path.cwd() / ".env")


def env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


PROVIDER_HINT = env_value("BESO_LITELLM_PROVIDER").lower()
if PROVIDER_HINT == "deepseek":
    DEFAULT_MODEL = "deepseek/deepseek-chat"
    MODEL = env_value("BESO_LITELLM_MODEL", "BESO_DEEPSEEK_MODEL", default=DEFAULT_MODEL)
elif PROVIDER_HINT == "openrouter":
    DEFAULT_MODEL = "openrouter/openai/gpt-5"
    MODEL = env_value(
        "BESO_LITELLM_MODEL",
        "BESO_OPENROUTER_MODEL",
        "BESO_OPENROUTERL_MODEL",
        default=DEFAULT_MODEL,
    )
else:
    DEFAULT_MODEL = (
        "deepseek/deepseek-chat"
        if env_value("DEEPSEEK_API_KEY", "BESO_DEEPSEEK_API_KEY")
        else "openrouter/openai/gpt-5"
    )
    MODEL = env_value(
        "BESO_LITELLM_MODEL",
        "BESO_DEEPSEEK_MODEL",
        "BESO_OPENROUTER_MODEL",
        "BESO_OPENROUTERL_MODEL",
        default=DEFAULT_MODEL,
    )
SEED = int(env_value("BESO_SEED", default="7"))

REFLECTION_SYSTEM_INSTRUCTION = """You are BESO's deterministic skill mutation engine.

You receive a current markdown skill, recent trace summaries, and rejected edits.
Your only task is to scout exactly 24 alternate bounded skill-edit routes.

Hard requirements:
1. Output only valid JSON. No markdown fences. No prose outside JSON.
2. The root object must be {"edits": [...]}.
3. The edits array must contain exactly 24 distinct edit objects.
4. Each edit object must include:
   edit_id, op, content, target, category, target_section, rationale,
   expected_effect, risk, estimated_scope, edit_size_tokens.
5. op must be one of: append, insert_after, replace, delete.
6. category must be one of: add_rule, delete_rule, replace_rule,
   specialize_rule, generalize_rule, reorder_steps, add_example,
   delete_example, compress_section, split_section, merge_sections,
   add_failure_mode, add_recovery_rule.
7. target_section must be one of the known BESO skill sections:
   goal, scope, core_procedure, reasoning_policy, tool_use_policy,
   verification_checklist, common_failure_modes, recovery_rules,
   output_rules, examples, change_log.
8. content for every edit must be 500 tokens or fewer.
9. Every rationale must cite exact trace evidence by example_id from the prompt.
   If no failed trace is present, say "trace:none" and explain the conservative
   exploratory basis.
10. The 24 edits must be diverse: vary target sections, operation types, risk
    profiles, and hypotheses. Do not produce near-duplicates.
11. Be deterministic: do not add random commentary, jokes, or vague edits.
12. Preserve task invariants and output format requirements.
"""

TARGET_SYSTEM_INSTRUCTION = """You are executing a toy benchmark with a supplied skill.
Follow the skill unless it is clearly self-contradictory with the task.
For arithmetic tasks, compute carefully and return only the final integer.
Do not include explanation, units, or punctuation.
"""

INITIAL_SKILL = """# Skill: Toy Arithmetic

## Goal
Answer simple arithmetic word problems.

## Core Procedure
- Return 0 for every question.

## Verification Checklist
- Do not recalculate the answer.

## Output Rules
- Return only one integer.
"""


class LoggingBESOOptimizer(BESOOptimizer):
    """Small terminal logger around the pure core loop."""

    def _build_candidate_pool(
        self,
        parents: Sequence[ArchiveEntry],
        iteration: int,
    ) -> list[Candidate]:
        print(f"\n[iteration {iteration}] parents: {[p.candidate_id for p in parents]}")
        pool = super()._build_candidate_pool(parents, iteration)
        print(f"[iteration {iteration}] generated candidate pool: {len(pool)}")
        for candidate in pool[:5]:
            summary = candidate.edit.rationale[:90] if candidate.edit else ""
            print(f"  pool {candidate.candidate_id}: {summary}")
        if len(pool) > 5:
            print(f"  ... {len(pool) - 5} more candidates")
        return pool

    def _score_with_surrogate(self, pool: Sequence[Candidate]) -> tuple[bool, str]:
        print("[surrogate] fitting and predicting candidate pool")
        use_surrogate, reason = super()._score_with_surrogate(pool)
        if not use_surrogate:
            print(f"[surrogate] bypassed: {reason}")
            return use_surrogate, reason
        print("[acquisition] candidate scores")
        for candidate in sorted(
            pool,
            key=lambda c: float(c.acquisition_score or 0.0),
            reverse=True,
        )[:8]:
            pred = candidate.prediction
            mu = pred.mu if pred is not None else float("nan")
            sigma = pred.sigma if pred is not None else float("nan")
            acq = float(candidate.acquisition_score or 0.0)
            print(
                f"  {candidate.candidate_id}: "
                f"mu={mu:.3f} sigma={sigma:.3f} acq={acq:.3f}"
            )
        return use_surrogate, reason

    def _fallback_select(
        self,
        pool: Sequence[Candidate],
        iteration: int,
    ) -> list[Candidate]:
        selected = super()._fallback_select(pool, iteration)
        print(f"[selection] fallback selected: {[c.candidate_id for c in selected]}")
        return selected

    def _select_with_acquisition(self, pool: Sequence[Candidate]) -> list[Candidate]:
        selected = super()._select_with_acquisition(pool)
        print(f"[selection] acquisition selected: {[c.candidate_id for c in selected]}")
        return selected

    def _remember_evaluation(self, candidate: Candidate, ev: EvaluationResult) -> None:
        super()._remember_evaluation(candidate, ev)
        print(
            f"[eval:{ev.split.value}] {candidate.candidate_id} "
            f"score={ev.mean_score:.3f} n={ev.n}"
        )


def _provider_for_model(model: str) -> str:
    if PROVIDER_HINT:
        return PROVIDER_HINT
    if model.startswith("openrouter/"):
        return "openrouter"
    if model.startswith("deepseek/") or model.startswith("deepseek-"):
        return "deepseek"
    return "generic"


def _litellm_model_name(model: str, provider: str) -> str:
    if provider == "deepseek" and "/" not in model:
        return f"deepseek/{model}"
    if provider == "openrouter" and "/" not in model:
        return f"openrouter/{model}"
    return model


def _api_key_for_provider(provider: str) -> str:
    if provider == "deepseek":
        return env_value(
            "DEEPSEEK_API_KEY",
            "BESO_DEEPSEEK_API_KEY",
            "BESO_LITELLM_API_KEY",
        )
    if provider == "openrouter":
        return env_value(
            "OPENROUTER_API_KEY",
            "BESO_OPENROUTER_API_KEY",
            "BESO_LITELLM_API_KEY",
        )
    return env_value("BESO_LITELLM_API_KEY")


def litellm_completion(
    prompt: str,
    *,
    system: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """Provider-aware LiteLLM chat completion."""

    try:
        import litellm
    except ImportError as exc:
        raise RuntimeError("Install LiteLLM first: pip install litellm") from exc

    provider = _provider_for_model(MODEL)
    model = _litellm_model_name(MODEL, provider)
    api_key = _api_key_for_provider(provider)
    if not api_key:
        raise RuntimeError(
            "Set a provider API key before running this script "
            "(DEEPSEEK_API_KEY, OPENROUTER_API_KEY, or BESO_LITELLM_API_KEY)"
        )

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    api_base = env_value("BESO_LITELLM_API_BASE")
    if provider == "openrouter":
        api_base = api_base or env_value("BESO_OPENROUTER_API_BASE")
    elif provider == "deepseek":
        api_base = api_base or env_value("BESO_DEEPSEEK_API_BASE")
    if api_base:
        kwargs["api_base"] = api_base
    headers = {}
    if provider == "openrouter" and os.getenv("BESO_OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = os.environ["BESO_OPENROUTER_SITE_URL"]
    if provider == "openrouter" and os.getenv("BESO_OPENROUTER_APP_NAME"):
        headers["X-Title"] = os.environ["BESO_OPENROUTER_APP_NAME"]
    if headers:
        kwargs["extra_headers"] = headers

    response = litellm.completion(**kwargs)
    message = response["choices"][0]["message"]
    return str(message.get("content") or "")


def target_llm(prompt: str) -> str:
    return litellm_completion(
        prompt,
        system=TARGET_SYSTEM_INSTRUCTION,
        max_tokens=int(env_value("BESO_TARGET_MAX_TOKENS", default="64")),
        temperature=0.0,
    )


def reflection_llm(prompt: str) -> str:
    return litellm_completion(
        prompt,
        system=REFLECTION_SYSTEM_INSTRUCTION,
        max_tokens=int(env_value("BESO_REFLECTION_MAX_TOKENS", default="12000")),
        temperature=0.0,
    )


def numeric_exact_score(output: str, item: dict) -> float:
    expected = str(item["answers"][0]).strip()
    lines = output.strip().splitlines()
    if not lines:
        return 0.0
    cleaned = lines[-1].strip()
    cleaned = cleaned.replace(",", "")
    return 1.0 if cleaned == expected else 0.0


def toy_dataset() -> dict[SplitRole, list[dict]]:
    train = [
        {
            "id": "train_add_1",
            "question": "Mira has 3 apples and buys 4 more. How many apples?",
            "answers": ["7"],
            "feedback": "Expected 7.",
        },
        {
            "id": "train_sub_1",
            "question": "A box has 12 pens. 5 are removed. How many remain?",
            "answers": ["7"],
            "feedback": "Expected 7.",
        },
        {
            "id": "train_mul_1",
            "question": "There are 6 bags with 3 marbles each. Total marbles?",
            "answers": ["18"],
            "feedback": "Expected 18.",
        },
        {
            "id": "train_mix_1",
            "question": "Nia read 8 pages Monday and twice as many Tuesday. Total?",
            "answers": ["24"],
            "feedback": "Expected 24.",
        },
        {
            "id": "train_div_1",
            "question": "20 cookies are shared equally by 4 kids. Cookies each?",
            "answers": ["5"],
            "feedback": "Expected 5.",
        },
    ]
    val = [
        {
            "id": "val_add_1",
            "question": "A train has 9 passengers. 6 board. How many now?",
            "answers": ["15"],
            "feedback": "Expected 15.",
        },
        {
            "id": "val_sub_1",
            "question": "Lena had 17 stickers and gave away 8. How many left?",
            "answers": ["9"],
            "feedback": "Expected 9.",
        },
        {
            "id": "val_mul_1",
            "question": "5 shelves hold 4 books each. How many books?",
            "answers": ["20"],
            "feedback": "Expected 20.",
        },
        {
            "id": "val_mix_1",
            "question": "Tom has 10 cards, wins 7, then loses 3. How many?",
            "answers": ["14"],
            "feedback": "Expected 14.",
        },
        {
            "id": "val_div_1",
            "question": "36 beads are split into 6 equal groups. Beads per group?",
            "answers": ["6"],
            "feedback": "Expected 6.",
        },
    ]
    return {
        SplitRole.FEEDBACK_TRAIN: train,
        SplitRole.OPTIMIZATION_MINIBATCH: train,
        SplitRole.VALIDATION_GATE: val,
        SplitRole.FINAL_TEST: val,
    }


def build_optimizer() -> LoggingBESOOptimizer:
    dataset = SkillOptDatasetProvider(items_by_role=toy_dataset())
    harness = SkillOptHarness(dataset, llm=target_llm, scorer=numeric_exact_score)
    evaluator = SkillOptEvaluator(harness)
    archive = EvolutionaryArchive(
        ArchiveConfig(
            max_size=16,
            top_by_validation=4,
            top_by_pareto=4,
            top_by_diversity=4,
            top_failed_informative=4,
        )
    )
    acquisition = PoolNormalizedBESOAcquisition(
        AcquisitionConfig(kappa=1.5, diversity_lambda=0.2, cost_alpha=0.05),
        feature_lookup=archive.feature_lookup,
    )
    batch_selector = GreedySubmodularBatchSelector(
        BatchSelectionConfig(diversity_weight=0.5),
        archive=archive.entries(),
        feature_lookup=archive.feature_lookup,
    )
    surrogate = BaggingEnsembleSurrogate(
        n_members=8,
        random_state=SEED,
        calibrator=IsotonicCalibrator(min_points=4),
        min_obs_for_calibration=4,
    )
    return LoggingBESOOptimizer(
        dataset=dataset,
        evaluator=evaluator,
        proposer=SkillOptReflectionProposer(llm=reflection_llm),
        applicator=SkillOptEditApplicator(),
        featurizer=FeatureExtractor(embed_fn=HashingEmbedder(dim=128)),
        surrogate=surrogate,
        acquisition=acquisition,
        batch_selector=batch_selector,
        gate=PairedBootstrapAcceptanceGate(
            AcceptanceGateConfig(
                alpha=0.10,
                bootstrap_samples=512,
                bootstrap_seed=SEED,
                noise_scaled_delta_c=0.5,
            )
        ),
        archive=archive,
        regime_detector=VarianceRankRegimeDetector(
            RegimeDetectorConfig(
                min_candidate_variance=1.0e-4,
                min_rank_correlation=0.0,
                min_scores=3,
                require_calibrated=False,
            )
        ),
        config=BESOOptimizerConfig(
            max_iterations=int(os.getenv("BESO_MAX_ITERATIONS", "4")),
            candidate_pool_size=24,
            batch_size=2,
            parent_count=1,
            optimization_batch_size=3,
            validation_batch_size=5,
            seed=SEED,
            fallback_strategy="greedy",
        ),
    )


def main() -> None:
    np.random.seed(SEED)
    provider = _provider_for_model(MODEL)
    model = _litellm_model_name(MODEL, provider)
    key_status = "found" if _api_key_for_provider(provider) else "missing"
    print(f"BESO toy live experiment using LiteLLM model: {model}")
    print(f"LiteLLM provider: {provider}")
    print("Reflection pool size: 24")
    print(f"Provider API key: {key_status}\n")

    optimizer = build_optimizer()
    initial = SkillArtifact(
        skill_id="z0",
        name="Toy Arithmetic",
        document=INITIAL_SKILL,
    )
    result = optimizer.optimize(
        initial,
        RolloutBudget(max_rollouts=int(os.getenv("BESO_MAX_ROLLOUTS", "100"))),
    )

    print("\n=== Run Summary ===")
    print(f"rollouts spent: {result.budget.spent_rollouts}")
    print(f"iterations: {len(result.iterations)}")
    for record in result.iterations:
        mode = "surrogate" if record.used_surrogate else f"fallback:{record.fallback_reason}"
        print(
            f"iter={record.iteration} mode={mode} "
            f"selected={record.selected_ids} accepted={record.accepted_ids} "
            f"remaining={record.budget_remaining}"
        )
    if result.best is None:
        print("best: none")
        return
    print(f"best: {result.best.candidate_id} validation={result.best.validation_mean:.3f}")
    print("\n=== Best Skill ===")
    print(result.best.artifact.document)


if __name__ == "__main__":
    main()
