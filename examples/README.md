# BESO Toy Arithmetic Experiment

This directory contains a runnable demonstration of the **Bayesian Evolutionary Skill Optimization (BESO)** pipeline applied to a toy arithmetic task.

The experiment optimizes an LLM's system prompt (or "skill") starting from a degenerate baseline to a highly accurate procedure, demonstrating surrogate-guided pool selection, statistical validation gating, and multi-objective archive management.

## Toy Arithmetic Task Overview

The target task is solving simple, multi-step arithmetic word problems.

- **Initial Degenerate Skill**:
  ```markdown
  # Skill: Toy Arithmetic

  ## Goal
  Answer simple arithmetic word problems.

  ## Core Procedure
  - Return 0 for every question.

  ## Verification Checklist
  - Do not recalculate the answer.

  ## Output Rules
  - Return only one integer.
  ```
- **The Optimization Goal**: Guide the agentic skill through local mutations to learn proper problem-reading, arithmetic step-by-step calculation, and strict format compliance, while optimizing secondary metrics like cost (tokens) and latency.

---

## Technical Architecture in Action

The script `@/2026/Bayesian Evolutionary Skill Optimization/examples/run_toy_experiment.py` orchestrates the core components of the BESO framework:

1. **Reflection Proposer (`SkillOptReflectionProposer`)**:
   - Prompts a reflection LLM to draft a candidate pool of $24$ distinct skill edits (e.g. replacing rules, adding failure modes, or recovery steps).
   - Requests provider JSON-object mode and enforces a strict JSON schema via Pydantic (`ProposedEdit` and `ReflectionOutput`). If the LLM returns broken or fractured JSON, it runs a bounded retry/repair loop to fix it, degrading gracefully if needed.
2. **Evaluator (`SkillOptEvaluator` & `SkillOptHarness`)**:
   - Measures candidate accuracy on mini-batches under strict budget limits.
3. **Surrogate Model (`BaggingEnsembleSurrogate` with `IsotonicCalibrator`)**:
   - Fits on previous evaluations of candidate features (e.g., token counts and semantic hash-embeds).
   - Predicts expected candidate performance (`mu` and `sigma` uncertainty).
4. **Acquisition and Selection (`PoolNormalizedBESOAcquisition` & `GreedySubmodularBatchSelector`)**:
   - Computes pool-normalized acquisition scores using expected quality, uncertainty (exploration), novelty/diversity, cost, and invalid risk.
   - Restricts expected score inflation via an optional `metric_bounds` clipping window.
   - Selects a diverse batch of candidates to run through the validation gate using a submodular greedy optimizer.
5. **Gating and Multiplicity Correction (`PairedBootstrapAcceptanceGate` & `apply_benjamini_hochberg`)**:
   - Runs a paired McNemar or bootstrap hypothesis test to guarantee that improvements over the parent are statistically significant.
   - Features a secondary **Pareto Non-Inferiority Gate** to accept neutral "cleanup" edits (e.g., optimizing token size or latency without degrading accuracy).
   - Applies the Benjamini-Hochberg procedure across candidate gates in a round to control False Discovery Rate (FDR).
6. **Evolutionary Archive (`EvolutionaryArchive`)**:
   - Manages candidates across multiple tiers (`BEST`, `PARETO`, `DIVERSE`, `FAILED`).
   - Dynamically routes Pareto cleanup edits away from the primary deployable `best` slot to protect the functional incumbent.
7. **JSONL Audit Trace (`JSONLLogger`)**:
   - Appends a full per-iteration record to `artifacts/toy_experiment.jsonl` by default.
   - Persists proposals, markdown diffs, predictions, acquisition scores, evaluations, gate decisions, and complete archive snapshots for offline analysis.

---

## Configuration & Environment Variables

The script loads environment variables from a `.env` file in the project root or the current directory. 

### API Key (One of these is required)
- `DEEPSEEK_API_KEY` (Default, recommended if key present)
- `OPENAI_API_KEY` (Default if key present and no DeepSeek key)
- `OPENROUTER_API_KEY`
- `BESO_LITELLM_API_KEY`

### Model Configuration (Optional)
- `BESO_LITELLM_PROVIDER`: E.g., `deepseek`, `openai`, or `openrouter` (auto-detected by model name or key presence if omitted).
- `BESO_LITELLM_MODEL`: E.g., `deepseek/deepseek-chat`, `gpt-4o-mini`, or `openrouter/deepseek/deepseek-chat`.
- `BESO_LITELLM_API_BASE`: Override base URL for proxy or local endpoints.

### Gating and Budget (Optional)
- `BESO_GATE_ALPHA`: False positive rate for the individual acceptance gate (Default: `0.10`).
- `BESO_BH_ALPHA`: False discovery rate for multiplicity correction (Default: same as `BESO_GATE_ALPHA`).
- `BESO_VALIDATION_BATCH_SIZE`: Samples evaluated for validation gating (Default: `10`).
- `BESO_MAX_ROLLOUTS`: Maximum evaluation budget (Default: `160`).
- `BESO_TRACE_PATH`: Append-only toy-run trace path (Default: `artifacts/toy_experiment.jsonl`).

---

## Running the Experiment

Ensure dependencies are installed in your environment:

```bash
uv pip install -e .
uv pip install litellm
```

Execute the toy experiment script:

```bash
uv run python examples/run_toy_experiment.py
```

### Example Run Output
During execution, you will see a detailed trace of the iterations:
- **Iteration 0**: The surrogate is cold and bypassed (regime precheck). Candidates are selected via fallback rules (greedy exploration).
- **Iteration 1**: Initial successful mutations (e.g., teaching the model to perform calculations step-by-step instead of returning 0) are accepted and recorded in the archive.
- **Iteration 2+**: The surrogate fits on historical features and guides candidate selection. The acquisition logs show predicted values (`mu`, `sigma`) and composite scores.
- **Run Summary**: Prints the final optimized skill artifact and performance history.

For a comprehensive, step-by-step guide on how to interpret each component of these logs (including the hypothesis testing, Benjamini-Hochberg correction, active learning acquisition scores, and the regime detector), see the detailed guide in `@/2026/Bayesian Evolutionary Skill Optimization/docs/experiments/results/M1-toy-validation.md`.

---

## Local GSM8K Mini Experiment

`run_gsm8k_mini_experiment.py` reuses the same optimizer and LiteLLM adapters
against local standard GSM8K JSONL rows. Each row must contain `question` and
`answer`, where the answer ends with the canonical `#### final` marker.

```bash
export BESO_GSM8K_TRAIN_JSONL=path/to/train.jsonl
export BESO_GSM8K_VALIDATION_JSONL=path/to/validation.jsonl
export BESO_GSM8K_TEST_JSONL=path/to/test.jsonl
export BESO_GSM8K_LIMIT=32
export BESO_FEEDBACK_BATCH_SIZE=8
export BESO_GSM8K_TARGET_MAX_TOKENS=2048
export BESO_GSM8K_BESO_SEED=minimal
uv run python examples/run_gsm8k_mini_experiment.py
```

The GSM8K runner reports three distinct conditions on one deterministic
validation draw:

1. `literal_no_skill`: the harness omits skill injection entirely.
2. `minimal_seed`: the frozen model receives only a neutral goal and numeric
   output rule.
3. `BESO`: the optimizer evolves the minimal seed under its configured rollout
   budget.

Set `BESO_GSM8K_BESO_SEED=toxic` only for a diagnostic recovery run. The two
frozen baselines still run unchanged. Baseline validation calls are reported
separately from BESO's optimization budget, and the comparison record is
written beside the main trace as `*_conditions.jsonl`.

Reflection receives worked solutions from the training feedback split only.
Validation and test answers remain hidden from the proposer. GSM8K scoring
extracts the final numeric answer, allowing natural-language reasoning output
without treating correct answers as formatting failures.
