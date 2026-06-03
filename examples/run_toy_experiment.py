"""Run a tiny live BESO experiment through LiteLLM.

Required environment:
    OPENROUTER_API_KEY=...
    # or
    DEEPSEEK_API_KEY=...
    # or
    OPENAI_API_KEY=...

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
    BESO_OPENAI_MODEL=gpt-4o-mini
    BESO_OPENAI_API_BASE=https://api.openai.com/v1
    BESO_GATE_ALPHA=0.10
    BESO_BH_ALPHA=0.10
    BESO_VALIDATION_BATCH_SIZE=10
    BESO_MAX_ROLLOUTS=160
    BESO_TRACE_PATH=artifacts/toy_experiment.jsonl
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

import numpy as np

from beso.acquisition import (
    AcquisitionConfig,
    BatchSelectionConfig,
    GreedySubmodularBatchSelector,
    PoolNormalizedBESOAcquisition,
    clip_to_bounds,
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
)
from beso.features import FeatureExtractor, HashingEmbedder
from beso.optimization import (
    AcceptanceGateConfig,
    BESOOptimizer,
    BESOOptimizerConfig,
    JSONLLogger,
    PairedBootstrapAcceptanceGate,
    RegimeDetectorConfig,
    VarianceRankRegimeDetector,
    apply_benjamini_hochberg,
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


class _LiteLLMRetryableResponseError(RuntimeError):
    pass


def _response_field(value: object, field: str) -> object:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _response_error_text(value: object) -> str:
    error = _response_field(value, "error")
    if error:
        return str(error)
    return ""


def _validated_litellm_content(response: object) -> str:
    choices = _response_field(response, "choices")
    if not choices:
        error_text = _response_error_text(response)
        if error_text:
            raise _LiteLLMRetryableResponseError(f"LiteLLM response included provider error: {error_text}")
        raise RuntimeError("LiteLLM response did not include any choices")

    choice = choices[0]
    finish_reason = _response_field(choice, "finish_reason")
    if str(finish_reason or "").lower() == "error":
        error_text = _response_error_text(response)
        detail = f": {error_text}" if error_text else ""
        raise _LiteLLMRetryableResponseError(f"LiteLLM response finished with provider error{detail}")

    message = _response_field(choice, "message")
    content = _response_field(message, "content") if message is not None else None
    if content is None or str(content).strip() == "":
        raise _LiteLLMRetryableResponseError("LiteLLM response returned empty content")
    return str(content)


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
elif PROVIDER_HINT == "openai":
    DEFAULT_MODEL = "gpt-4o-mini"
    MODEL = env_value(
        "BESO_LITELLM_MODEL",
        "BESO_OPENAI_MODEL",
        default=DEFAULT_MODEL,
    )
else:
    if env_value("DEEPSEEK_API_KEY", "BESO_DEEPSEEK_API_KEY"):
        DEFAULT_MODEL = "deepseek/deepseek-chat"
    elif env_value("OPENAI_API_KEY", "BESO_OPENAI_API_KEY"):
        DEFAULT_MODEL = "gpt-4o-mini"
    else:
        DEFAULT_MODEL = "openrouter/openai/gpt-5"
    MODEL = env_value(
        "BESO_LITELLM_MODEL",
        "BESO_DEEPSEEK_MODEL",
        "BESO_OPENAI_MODEL",
        "BESO_OPENROUTER_MODEL",
        "BESO_OPENROUTERL_MODEL",
        default=DEFAULT_MODEL,
    )
SEED = int(env_value("BESO_SEED", default="7"))
GATE_ALPHA = float(env_value("BESO_GATE_ALPHA", default="0.10"))
BH_ALPHA = float(env_value("BESO_BH_ALPHA", default=str(GATE_ALPHA)))
TRACE_PATH = Path(
    env_value(
        "BESO_TRACE_PATH",
        default=str(_PROJECT_ROOT / "artifacts" / "toy_experiment.jsonl"),
    )
)

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
13. If the current skill contains a contradictory or degenerate rule that
    directly explains failures, include multiple replace/delete edits that
    target that exact rule or its full section. Prefer replacing the faulty
    core_procedure over merely appending advice.
14. The total skill artifact is allowed to grow. When evidence supports it,
    propose comprehensive multi-rule additions, worked examples, recovery
    guidance, and structured sections. The 500-token limit applies per edit,
    not to the full evolved artifact.
"""

TARGET_SYSTEM_INSTRUCTION = """You are executing a benchmark with a supplied skill.
Use only the supplied skill's procedure to solve the task.
Do not introduce a procedure that is not stated in the skill.
If the skill does not provide enough guidance to derive an answer, return 0.
Return only the final answer. Do not include explanation, units, or punctuation.
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
            mu_raw = pred.mu if pred is not None else float("nan")
            mu_bounded = clip_to_bounds(
                mu_raw,
                self.acquisition.config.metric_bounds,
            )
            sigma = pred.sigma if pred is not None else float("nan")
            acq = float(candidate.acquisition_score or 0.0)
            print(
                f"  {candidate.candidate_id}: "
                f"mu_raw={mu_raw:.3f} mu_bounded={mu_bounded:.3f} "
                f"sigma={sigma:.3f} acq={acq:.3f}"
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
    if model.startswith("gpt-") or model.startswith("openai/"):
        return "openai"
    if "/" in model:
        parts = model.split("/", 1)
        known_providers = {"openai", "deepseek", "openrouter", "azure", "anthropic", "cohere", "bedrock", "vertex_ai", "gemini", "groq", "together", "huggingface"}
        if parts[0].lower() not in known_providers:
            if env_value("OPENROUTER_API_KEY", "BESO_OPENROUTER_API_KEY"):
                return "openrouter"
    return "generic"


def _litellm_model_name(model: str, provider: str) -> str:
    if provider == "deepseek" and not model.startswith("deepseek/"):
        return f"deepseek/{model}"
    if provider == "openrouter" and not model.startswith("openrouter/"):
        return f"openrouter/{model}"
    if provider == "openai" and "/" not in model and not model.startswith("gpt-"):
        return f"openai/{model}"
    return model


def _completion_temperature(model: str, temperature: float) -> float:
    model_name = model.rsplit("/", 1)[-1].lower()
    if model_name.startswith("gpt-5") and not model_name.startswith("gpt-5.1"):
        return 1.0
    return temperature


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
    if provider == "openai":
        return env_value(
            "OPENAI_API_KEY",
            "BESO_OPENAI_API_KEY",
            "BESO_LITELLM_API_KEY",
        )
    return env_value("BESO_LITELLM_API_KEY")


def litellm_completion(
    prompt: str,
    *,
    system: str,
    max_tokens: int,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """Provider-aware LiteLLM chat completion."""

    try:
        import litellm
        litellm.suppress_debug_info = True
    except ImportError as exc:
        raise RuntimeError("Install LiteLLM first: pip install litellm") from exc

    provider = _provider_for_model(MODEL)
    model = _litellm_model_name(MODEL, provider)
    api_key = _api_key_for_provider(provider)
    if not api_key:
        raise RuntimeError(
            "Set a provider API key before running this script "
            "(DEEPSEEK_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY, or BESO_LITELLM_API_KEY)"
        )

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "api_key": api_key,
        "temperature": _completion_temperature(model, temperature),
        "max_tokens": max_tokens,
    }
    if provider != "generic":
        kwargs["custom_llm_provider"] = provider
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    api_base = env_value("BESO_LITELLM_API_BASE")
    if provider == "openrouter":
        api_base = api_base or env_value("BESO_OPENROUTER_API_BASE")
    elif provider == "deepseek":
        api_base = api_base or env_value("BESO_DEEPSEEK_API_BASE")
    elif provider == "openai":
        api_base = api_base or env_value("BESO_OPENAI_API_BASE")
    if api_base:
        kwargs["api_base"] = api_base
    headers = {}
    if provider == "openrouter" and os.getenv("BESO_OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = os.environ["BESO_OPENROUTER_SITE_URL"]
    if provider == "openrouter" and os.getenv("BESO_OPENROUTER_APP_NAME"):
        headers["X-Title"] = os.environ["BESO_OPENROUTER_APP_NAME"]
    if headers:
        kwargs["extra_headers"] = headers

    max_retries = int(env_value("BESO_LITELLM_MAX_RETRIES", default="5"))
    initial_delay = float(env_value("BESO_LITELLM_INITIAL_DELAY", default="2.0"))
    backoff_factor = float(env_value("BESO_LITELLM_BACKOFF_FACTOR", default="2.0"))

    exceptions = getattr(litellm, "exceptions", None)
    RateLimitError = getattr(exceptions, "RateLimitError", None)
    APIConnectionError = getattr(exceptions, "APIConnectionError", None)
    ServiceUnavailableError = getattr(exceptions, "ServiceUnavailableError", None)

    catch_exceptions = tuple(
        exc for exc in [RateLimitError, APIConnectionError, ServiceUnavailableError] if exc is not None
    )

    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            response = litellm.completion(**kwargs)
            return _validated_litellm_content(response)
        except Exception as e:
            is_retryable = False
            if isinstance(e, _LiteLLMRetryableResponseError):
                is_retryable = True
            elif catch_exceptions and isinstance(e, catch_exceptions):
                is_retryable = True
            else:
                err_msg = str(e).lower()
                status_code = getattr(e, "status_code", getattr(e, "http_status", None))
                if (
                    status_code in (429, 500, 502, 503, 504)
                    or "429" in err_msg
                    or "rate limit" in err_msg
                    or "too many requests" in err_msg
                    or "503" in err_msg
                    or "service unavailable" in err_msg
                    or "502" in err_msg
                    or "bad gateway" in err_msg
                ):
                    is_retryable = True

            if is_retryable and attempt < max_retries:
                jitter = random.uniform(0, 0.5 * delay)
                sleep_time = delay + jitter
                print(
                    f"LiteLLM call hit retryable error: {e}. "
                    f"Retrying in {sleep_time:.2f} seconds (attempt {attempt + 1}/{max_retries})...",
                    flush=True,
                )
                time.sleep(sleep_time)
                delay *= backoff_factor
            else:
                raise e


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
        json_mode=True,
    )


def numeric_exact_score(output: str, item: dict) -> float:
    expected = str(item["answers"][0]).strip()
    lines = output.strip().splitlines()
    if not lines:
        return 0.0
    cleaned = lines[-1].strip()
    cleaned = cleaned.replace(",", "")
    return 1.0 if cleaned == expected else 0.0


def log_and_apply_bh(decisions):
    if not decisions:
        return []
    print("[gate] raw paired decisions")
    for decision in decisions:
        print(
            f"  {decision.candidate_id}: accepted={decision.accepted} "
            f"diff={decision.mean_diff:.3f} ci=[{decision.ci_low:.3f}, "
            f"{decision.ci_high:.3f}] p={decision.p_value:.4f} "
            f"threshold={decision.noise_scaled_threshold:.3f} "
            f"reason={decision.reason}"
        )
    corrected = apply_benjamini_hochberg(decisions, alpha=BH_ALPHA)
    print(f"[gate] BH correction alpha={BH_ALPHA:.3f}")
    for decision in corrected:
        print(
            f"  {decision.candidate_id}: accepted={decision.accepted} "
            f"p={decision.p_value:.4f} reason={decision.reason}"
        )
    return corrected


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
        {
            "id": "val_add_2",
            "question": "A jar has 11 buttons and 13 more are added. Total buttons?",
            "answers": ["24"],
            "feedback": "Expected 24.",
        },
        {
            "id": "val_sub_2",
            "question": "There were 25 balloons. 9 popped. How many remain?",
            "answers": ["16"],
            "feedback": "Expected 16.",
        },
        {
            "id": "val_mul_2",
            "question": "7 boxes each contain 3 pencils. How many pencils?",
            "answers": ["21"],
            "feedback": "Expected 21.",
        },
        {
            "id": "val_mix_2",
            "question": "Sara had 6 shells, found 8, then gave away 5. How many?",
            "answers": ["9"],
            "feedback": "Expected 9.",
        },
        {
            "id": "val_div_2",
            "question": "42 stickers are shared equally among 7 kids. Stickers each?",
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


def build_optimizer(
    dataset: SkillOptDatasetProvider | None = None,
    *,
    trace_path: str | Path = TRACE_PATH,
    target_generate: Callable[[str], str] = target_llm,
    scorer: Callable[[str, dict], float] = numeric_exact_score,
    feedback_batch_size: int | None = None,
    harness: SkillOptHarness | None = None,
) -> LoggingBESOOptimizer:
    dataset = dataset or SkillOptDatasetProvider(items_by_role=toy_dataset())
    harness = harness or SkillOptHarness(dataset, llm=target_generate, scorer=scorer)
    evaluator = SkillOptEvaluator(harness)
    archive = EvolutionaryArchive(
        ArchiveConfig(
            max_size=16,
            top_by_validation=4,
            top_by_pareto=4,
            top_by_diversity=4,
            top_failed_informative=4,
            parent_cost_beta=float(
                env_value("BESO_PARENT_COST_BETA", default="0.02")
            ),
        )
    )
    acquisition = PoolNormalizedBESOAcquisition(
        AcquisitionConfig(
            kappa=1.5,
            diversity_lambda=0.2,
            cost_alpha=float(env_value("BESO_COST_ALPHA", default="0.01")),
            metric_bounds=(0.0, 1.0),
        ),
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
                alpha=GATE_ALPHA,
                bootstrap_samples=int(
                    env_value("BESO_BOOTSTRAP_SAMPLES", default="512")
                ),
                bootstrap_seed=SEED,
                noise_scaled_delta_c=float(
                    env_value("BESO_NOISE_DELTA_C", default="0.5")
                ),
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
            max_iterations=int(env_value("BESO_MAX_ITERATIONS", default="4")),
            candidate_pool_size=24,
            batch_size=2,
            parent_count=1,
            optimization_batch_size=int(
                env_value("BESO_OPTIMIZATION_BATCH_SIZE", default="3")
            ),
            feedback_batch_size=(
                int(env_value("BESO_FEEDBACK_BATCH_SIZE", default="3"))
                if feedback_batch_size is None
                else feedback_batch_size
            ),
            validation_batch_size=int(
                env_value("BESO_VALIDATION_BATCH_SIZE", default="10")
            ),
            seed=SEED,
            fallback_strategy="greedy",
        ),
        multiplicity_correction=log_and_apply_bh,
        logger=JSONLLogger(trace_path),
    )


def main() -> None:
    np.random.seed(SEED)
    provider = _provider_for_model(MODEL)
    model = _litellm_model_name(MODEL, provider)
    key_status = "found" if _api_key_for_provider(provider) else "missing"
    print(f"BESO toy live experiment using LiteLLM model: {model}")
    print(f"LiteLLM provider: {provider}")
    print("Reflection pool size: 24")
    print(f"Gate alpha: {GATE_ALPHA:.3f}; BH alpha: {BH_ALPHA:.3f}")
    print(f"JSONL trace: {TRACE_PATH}")
    print(f"Provider API key: {key_status}\n")

    optimizer = build_optimizer()
    initial = SkillArtifact(
        skill_id="z0",
        name="Toy Arithmetic",
        document=INITIAL_SKILL,
    )
    result = optimizer.optimize(
        initial,
        RolloutBudget(max_rollouts=int(env_value("BESO_MAX_ROLLOUTS", default="160"))),
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
