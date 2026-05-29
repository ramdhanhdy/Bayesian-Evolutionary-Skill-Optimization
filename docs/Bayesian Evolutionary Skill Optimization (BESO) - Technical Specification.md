## Technical Specification

## 1. Executive Summary

**Bayesian Evolutionary Skill Optimization (BESO)** is a proposed optimization framework for improving frozen LLM-based systems by treating natural-language **skills** as trainable external artifacts.

Instead of directly optimizing model weights, BESO optimizes a structured skill document that encodes reusable behavioral rules, procedures, tool-use policies, verification habits, and failure-avoidance strategies. The skill is then compiled into prompts or injected into an agent runtime.

BESO combines four ideas:

1. **Skill-level abstraction**: optimize reusable skill artifacts rather than brittle raw prompts.
    
2. **Reflection-based mutation**: use trajectory feedback to propose meaningful natural-language edits.
    
3. **Bayesian experiment planning**: use a surrogate model to decide which candidate edits are worth evaluating under a limited rollout budget.
    
4. **Evolutionary archive management**: preserve high-performing and specialized candidates instead of greedily keeping only the single best artifact.
    

In short:

> BESO is a sample-efficient optimizer for frozen LLM systems that evolves natural-language skills using reflection, evaluates candidates under a rollout budget, and uses Bayesian acquisition to choose the most promising or informative candidates to test next.

---

## 2. Motivation

LLM behavior is highly sensitive to instructions. Many downstream failures are not caused by a lack of model capability, but by weak task framing, missing procedural rules, poor tool-use policies, vague output constraints, or absent verification steps.

Traditional improvement paths include:

- manual prompt engineering,
    
- automatic prompt optimization,
    
- reinforcement learning,
    
- fine-tuning,
    
- textual-gradient-style optimization,
    
- evolutionary prompt search,
    
- skill-document optimization.
    

Each has trade-offs.

Manual prompt engineering is interpretable but slow and unscalable. Reinforcement learning and fine-tuning can be powerful but expensive, opaque, and often unavailable for closed-source models. Evolutionary prompt optimization can be flexible but may waste rollout budget by evaluating too many weak candidates. Skill-document optimization is more stable than raw prompt editing, but candidate selection can still be local or greedy.

BESO addresses this by asking:

> Can we use Bayesian optimization to make natural-language skill evolution more sample-efficient?

The key hypothesis is that a Bayesian surrogate can learn from previous skill evaluations and guide future candidate selection toward edits that are either likely to improve performance or likely to reveal useful information.

---

## 3. Core Research Question

**Primary question:**

> Can Bayesian-guided evolutionary optimization of structured skill documents outperform prompt-level evolution and gradient-like skill editing under low rollout budgets?

**Secondary questions:**

1. Are skill documents a better optimization target than raw prompts?
    
2. Does Bayesian candidate selection improve sample efficiency compared with random, greedy, bandit, or Pareto-only selection?
    
3. Which representation of text artifacts works best for the Bayesian surrogate?
    
4. Does the optimized skill transfer across models, tasks, and execution harnesses?
    
5. Does preserving diverse skill variants improve generalization compared with keeping only the highest-average candidate?
    

---

## 4. Design Thesis

BESO is built on the following thesis:

> The best search object for prompt-like optimization is not the final prompt string. It is a structured, reusable skill artifact that can be compiled into runtime prompts.

Raw prompts are brittle because they mix many concerns into one text block: role, task definition, reasoning strategy, tool policy, output format, safety behavior, examples, and failure handling.

A structured skill artifact separates these concerns. This makes editing easier, evaluation more interpretable, and Bayesian modeling more meaningful.

The preferred hierarchy is:

```text
Skill artifact
    ↓ compile / inject
Module prompt(s)
    ↓ execute
LLM system behavior
    ↓ evaluate
Trajectory + score + feedback
    ↓ optimize
Updated skill artifact
```

---

## 5. Scope

### 5.1 In Scope

BESO optimizes:

- natural-language skill documents,
    
- prompt sections,
    
- module-level instruction policies,
    
- failure-mode checklists,
    
- tool-use policies,
    
- reasoning procedures,
    
- output-format rules,
    
- validation or verification habits,
    
- few-shot examples inside the skill artifact.
    

BESO can be applied to:

- single-call LLM tasks,
    
- multi-step reasoning tasks,
    
- tool-use agents,
    
- code-generation agents,
    
- retrieval-augmented generation systems,
    
- structured extraction systems,
    
- classification and judgment systems,
    
- benchmark-solving agents.
    

### 5.2 Out of Scope

The initial version does not optimize:

- model weights,
    
- retrieval indexes,
    
- tool implementations,
    
- benchmark labels,
    
- evaluator prompts unless explicitly configured,
    
- full agent architecture search,
    
- arbitrary code generation for new tools.
    

These may become future extensions.

---

## 6. Key Definitions

### 6.1 Frozen Agent

A frozen agent is an LLM-based system whose underlying model weights are not updated during optimization.

```text
Model weights: fixed
External skill state: trainable
```

### 6.2 Skill Artifact

A skill artifact is a structured natural-language document that encodes reusable behavioral knowledge.

It may contain:

- goal definition,
    
- task scope,
    
- step-by-step procedure,
    
- reasoning strategy,
    
- tool-use policy,
    
- verification checklist,
    
- common failure modes,
    
- recovery rules,
    
- output format,
    
- examples,
    
- constraints.
    

### 6.3 Runtime Prompt

A runtime prompt is the actual prompt passed to the LLM during execution. It may be generated by compiling the skill artifact into a task-specific prompt.

### 6.4 Trajectory

A trajectory is the recorded execution path of the system on one task instance.

It may include:

- input,
    
- selected skill,
    
- compiled prompt,
    
- intermediate model outputs,
    
- reasoning summaries,
    
- tool calls,
    
- tool results,
    
- retrieval context,
    
- final answer,
    
- evaluator score,
    
- textual feedback,
    
- error messages.
    

### 6.5 Rollout

A rollout is one execution of the system on one task example.

### 6.6 Candidate

A candidate is a proposed skill artifact or prompt artifact being evaluated.

### 6.7 Archive

The archive is the stored population of accepted, useful, or diagnostically important candidates.

### 6.8 Surrogate Model

A surrogate model is a learned predictor that estimates candidate performance and uncertainty without fully evaluating every candidate.

### 6.9 Acquisition Function

An acquisition function scores unevaluated candidates based on predicted performance and uncertainty. It determines which candidate should be evaluated next.

---

## 7. Optimization Target

BESO supports three levels of optimization.

### 7.1 Level 1: Raw Prompt Optimization

Artifact:

```text
z = prompt string
```

This is the simplest version. It is useful for single-call tasks but fragile for complex systems.

### 7.2 Level 2: Structured Prompt Section Optimization

Artifact:

```text
z = {
  role,
  task_instruction,
  reasoning_strategy,
  tool_policy,
  output_format,
  constraints,
  examples
}
```

This improves control by allowing targeted edits to specific prompt sections.

### 7.3 Level 3: Skill Artifact Optimization

Artifact:

```text
z = skill document
```

This is the recommended target for the main BESO method.

The skill artifact is reusable and may be compiled into multiple prompts.

### 7.4 Recommended Default

The default BESO configuration should optimize Level 3 artifacts:

> Optimize skill documents first. Compile them into prompts at runtime.

Raw prompt optimization can be included as a baseline or ablation.

---

## 8. Skill Artifact Schema

A default skill artifact should use a structured markdown format.

```markdown
# Skill: <skill_name>

## Goal
Describe what this skill helps the agent accomplish.

## Scope
Define when this skill should and should not be used.

## Core Procedure
1. Step one.
2. Step two.
3. Step three.

## Reasoning Policy
Explain how the agent should reason through the task.

## Tool-Use Policy
Explain when to use tools, when not to use tools, and how to validate tool outputs.

## Verification Checklist
- Check 1.
- Check 2.
- Check 3.

## Common Failure Modes
- Failure mode 1.
- Failure mode 2.
- Failure mode 3.

## Recovery Rules
Explain how to recover from uncertainty, missing evidence, conflicting evidence, or invalid intermediate results.

## Output Rules
Define the final answer format, concision level, citation rules, JSON schema, or other output constraints.

## Examples
Optional few-shot examples.

## Change Log
Track accepted edits and why they were made.
```

### 8.1 Machine-Readable Representation

Internally, the skill can be represented as JSON:

```json
{
  "skill_id": "multi_step_qa_v3",
  "name": "Multi-Step Question Answering",
  "version": 3,
  "sections": {
    "goal": "Answer multi-step questions accurately using evidence.",
    "scope": "Use for questions requiring intermediate facts.",
    "core_procedure": [
      "Identify the main question.",
      "Identify missing intermediate facts.",
      "Resolve intermediate facts before answering.",
      "Verify that the final answer follows from evidence."
    ],
    "reasoning_policy": "Decompose before answering.",
    "tool_use_policy": "Use search or calculator tools when the required fact or calculation is uncertain.",
    "verification_checklist": [
      "Does the answer directly answer the question?",
      "Is each intermediate fact supported?",
      "Are there contradictions?"
    ],
    "common_failure_modes": [
      "Answering from the first retrieved fact only.",
      "Confusing related entities.",
      "Skipping verification."
    ],
    "recovery_rules": [
      "If evidence conflicts, report uncertainty.",
      "If a tool call fails, retry with a narrower query."
    ],
    "output_rules": [
      "Give the final answer first.",
      "Explain only the necessary reasoning."
    ],
    "examples": []
  },
  "metadata": {
    "created_by": "optimizer",
    "parent_id": "multi_step_qa_v2",
    "edit_summary": "Added verification checklist for entity confusion.",
    "token_count": 412
  }
}
```

---

## 9. Mathematical Formulation

### 9.1 System Definition

Let:

```text
x = task input
m = metadata, label, expected answer, rubric, or test case
z = skill artifact
Θ = frozen model weights
Φ = full AI system
μ = scoring function
```

The system output is:

```text
y = Φ(x; z, Θ_frozen)
```

The score is:

```text
μ(Φ(x; z, Θ_frozen), m) ∈ [0, 1]
```

### 9.2 Expected Performance Objective

The true objective is to find the skill artifact that maximizes expected performance across the task distribution:

```text
z* = argmax_z E_(x,m)~T [ μ(Φ(x; z, Θ_frozen), m) ]
```

This means:

> Find the skill artifact that makes the frozen system perform best across many possible examples.

### 9.3 Empirical Objective

Because the true task distribution is unknown, use a dataset:

```text
D = {(x_1, m_1), ..., (x_n, m_n)}
```

Approximate expected performance with empirical average:

```text
J(z) = (1/n) Σ_i μ(Φ(x_i; z, Θ_frozen), m_i)
```

Optimization target:

```text
z* ≈ argmax_z J(z)
```

### 9.4 Rollout Budget Constraint

Each evaluation consumes rollouts.

```text
z* ≈ argmax_z J(z)
subject to #rollouts ≤ B
```

where:

```text
B = maximum rollout budget
```

### 9.5 Noisy Evaluation

In practice, evaluating a candidate on a minibatch gives a noisy estimate:

```text
ŷ_t = J_M(z_t) + ε_t
```

where:

```text
M = minibatch
ε_t = evaluation noise
```

The optimizer therefore maintains an evaluation dataset:

```text
H_t = {(z_1, ŷ_1), ..., (z_t, ŷ_t)}
```

### 9.6 Bayesian Surrogate

BESO learns a surrogate model over artifact performance:

```text
p(f | H_t)
```

where:

```text
f(z) = true but unknown performance function
```

For each candidate z, the surrogate estimates:

```text
μ_t(z) = predicted performance
σ_t(z) = uncertainty
```

### 9.7 Acquisition Function

BESO uses an acquisition function to decide which candidate to evaluate next.

Default acquisition: Upper Confidence Bound.

```text
a(z) = μ_t(z) + κ σ_t(z)
```

where:

```text
κ = exploration weight
```

Candidate selection:

```text
z_(t+1) = argmax_{z ∈ C_t} a(z)
```

where:

```text
C_t = candidate pool generated at iteration t
```

### 9.8 Multi-Objective Extension

For some tasks, performance has multiple objectives:

```text
score_accuracy
score_format
score_latency
score_cost
score_safety
```

Then define a vector objective:

```text
F(z) = [J_accuracy(z), J_format(z), -Cost(z), -Latency(z)]
```

BESO may use either:

1. scalarization,
    
2. Pareto archive selection,
    
3. constrained optimization.
    

Example scalar objective:

```text
J_total(z) = w_acc J_acc(z) + w_fmt J_fmt(z) - w_cost Cost(z) - w_lat Latency(z)
```

---

## 10. Algorithm Overview

### 10.1 High-Level Loop

```text
1. Initialize skill artifact z_0.
2. Evaluate z_0 on a seed validation set.
3. Store trajectories, scores, and feedback.
4. Generate candidate skill edits using reflection.
5. Featurize candidates.
6. Fit or update Bayesian surrogate.
7. Use acquisition function to select candidates for evaluation.
8. Evaluate selected candidates with rollout budget.
9. Accept, reject, or archive candidates.
10. Repeat until budget is exhausted.
11. Return best validated skill artifact.
```

### 10.2 Pseudocode

```python
initialize skill z0
H = []
archive = []
rejected_edits = []

score0, traces0 = evaluate(z0, eval_split="validation_seed")
H.append((z0, score0, traces0))
archive.append(z0)

for t in range(max_iterations):
    parents = select_parents(archive, strategy="pareto_plus_ucb")

    candidate_pool = []
    for parent in parents:
        relevant_traces = retrieve_traces(parent, H)
        edit_proposals = reflection_model.propose_edits(
            skill=parent,
            traces=relevant_traces,
            rejected_edits=rejected_edits
        )
        candidates = apply_bounded_edits(parent, edit_proposals)
        candidate_pool.extend(candidates)

    candidate_pool = filter_invalid_candidates(candidate_pool)
    candidate_pool = deduplicate(candidate_pool)

    X_candidates = featurize(candidate_pool)
    surrogate = fit_surrogate(H)
    acquisition_scores = compute_acquisition(surrogate, X_candidates)

    selected = select_top_k(candidate_pool, acquisition_scores, k=batch_size)

    for z in selected:
        score, traces = evaluate(z, eval_split="optimization_minibatch")
        H.append((z, score, traces))

        validation_score = validate_candidate(z)

        if accept_candidate(z, validation_score, archive):
            archive.append(z)
        else:
            rejected_edits.append(extract_edit_record(z, traces, validation_score))

    archive = prune_archive(archive, strategy="pareto_and_diversity")

return select_best_final_skill(archive, final_validation_set)
```

---

## 11. Candidate Generation

BESO does not ask the Bayesian model to generate text directly. Instead:

```text
Reflection model generates candidate edits.
Bayesian surrogate chooses which edits to evaluate.
```

This separation is important.

The reflection model is good at semantic text editing. The Bayesian surrogate is good at experiment planning under uncertainty.

### 11.1 Candidate Generation Inputs

The reflection model receives:

- parent skill artifact,
    
- successful trajectories,
    
- failed trajectories,
    
- evaluator feedback,
    
- previous accepted edits,
    
- rejected edit buffer,
    
- current failure taxonomy,
    
- edit budget.
    

### 11.2 Edit Proposal Schema

```json
{
  "edit_id": "edit_00017",
  "parent_skill_id": "skill_v4",
  "target_section": "verification_checklist",
  "operation": "add",
  "proposed_text": "Before finalizing, verify that the answer directly resolves the original question rather than an intermediate sub-question.",
  "rationale": "Several failed trajectories answered intermediate facts instead of the final user query.",
  "expected_effect": "Improve final-answer alignment on multi-hop questions.",
  "risk": "May increase verbosity or slow down direct questions.",
  "estimated_scope": "multi_step_reasoning",
  "edit_size_tokens": 27
}
```

### 11.3 Edit Operations

Supported operations:

|Operation|Description|
|---|---|
|add_rule|Add a new rule to a section|
|delete_rule|Remove harmful or redundant instruction|
|replace_rule|Rewrite an existing instruction|
|specialize_rule|Make a general instruction more task-specific|
|generalize_rule|Convert narrow fix into reusable rule|
|reorder_steps|Change procedural order|
|add_example|Add a few-shot example|
|delete_example|Remove misleading example|
|compress_section|Reduce verbosity|
|split_section|Separate overloaded section|
|merge_sections|Combine redundant sections|
|add_failure_mode|Add common error pattern|
|add_recovery_rule|Add fallback behavior|

### 11.4 Bounded Edit Constraints

Each edit must satisfy constraints:

```text
max_added_tokens_per_iteration
max_deleted_tokens_per_iteration
max_replaced_tokens_per_iteration
max_sections_modified_per_iteration
max_examples_added_per_iteration
```

Default constraints:

```yaml
max_added_tokens_per_iteration: 120
max_deleted_tokens_per_iteration: 80
max_replaced_tokens_per_iteration: 160
max_sections_modified_per_iteration: 2
max_examples_added_per_iteration: 1
```

Bounded edits prevent unstable prompt drift.

---

## 12. Candidate Featurization

Bayesian optimization requires numeric candidate representations.

### 12.1 Feature Vector

Each candidate z is converted into:

```text
φ(z) = numeric feature vector
```

Potential features:

#### Text Embedding Features

- embedding of full skill artifact,
    
- embedding of changed section,
    
- embedding of edit rationale,
    
- embedding of proposed delta.
    

#### Structural Features

- total token count,
    
- edit size,
    
- section modified,
    
- number of rules,
    
- number of examples,
    
- number of checklist items,
    
- number of failure modes,
    
- number of recovery rules.
    

#### Semantic LLM-Labeled Features

- decomposition emphasis,
    
- verification emphasis,
    
- tool-use aggressiveness,
    
- caution level,
    
- verbosity level,
    
- specificity level,
    
- format strictness,
    
- uncertainty-handling strength.
    

#### Historical Features

- parent score,
    
- parent variance,
    
- edit type success rate,
    
- section success rate,
    
- similarity to previously failed candidates,
    
- similarity to accepted candidates,
    
- number of previous mutations from same lineage.
    

### 12.2 Example Feature Record

```json
{
  "candidate_id": "skill_v7_candidate_3",
  "parent_score": 0.64,
  "edit_operation": "add_rule",
  "target_section": "tool_use_policy",
  "edit_size_tokens": 44,
  "skill_token_count": 612,
  "num_rules": 18,
  "num_examples": 2,
  "semantic_features": {
    "decomposition_emphasis": 0.72,
    "verification_emphasis": 0.81,
    "tool_use_aggressiveness": 0.43,
    "caution_level": 0.66,
    "verbosity_level": 0.58
  },
  "embedding": "<vector>"
}
```

---

## 13. Bayesian Surrogate Options

BESO should support multiple surrogate models.

### 13.1 Gaussian Process

Useful when the number of evaluated candidates is small and feature dimension is manageable.

Pros:

- principled uncertainty,
    
- strong Bayesian foundation.
    

Cons:

- scales poorly,
    
- struggles with high-dimensional text embeddings.
    

### 13.2 Bayesian Ridge Regression

Useful as a simple baseline.

Pros:

- simple,
    
- fast,
    
- interpretable.
    

Cons:

- limited nonlinearity.
    

### 13.3 Random Forest or Extra Trees Surrogate

Useful for structured features.

Pros:

- robust,
    
- handles mixed features,
    
- uncertainty can be approximated from tree variance.
    

Cons:

- uncertainty is heuristic.
    

### 13.4 TPE-Style Surrogate

Tree-structured Parzen Estimator can model good vs bad regions.

Pros:

- practical,
    
- works well for hyperparameter-like spaces,
    
- robust to mixed feature types.
    

Cons:

- less direct posterior interpretation.
    

### 13.5 Ensemble Surrogate

Recommended default.

Use an ensemble of lightweight models:

```text
surrogate = ensemble(
  BayesianRidge,
  RandomForest,
  KNN_on_embeddings,
  lightweight_neural_regressor_optional
)
```

Estimate uncertainty through prediction disagreement.

Recommended for v0:

> Use ensemble surrogate with structured features + embedding similarity.

---

## 14. Acquisition Functions

BESO should support several acquisition strategies.

### 14.1 Upper Confidence Bound

```text
a(z) = μ_t(z) + κ σ_t(z)
```

Best default because it is simple and explicitly balances exploitation and exploration.

### 14.2 Expected Improvement

```text
EI(z) = E[max(0, f(z) - f_best)]
```

Useful when the goal is improvement over current best.

### 14.3 Probability of Improvement

```text
PI(z) = P(f(z) > f_best + ξ)
```

Simple but can over-exploit.

### 14.4 Thompson Sampling

Sample a possible performance function from the posterior and choose the best candidate under that sample.

Good for batched and noisy settings.

### 14.5 Diversity-Aware Acquisition

To avoid testing near-duplicate candidates:

```text
a_diverse(z) = a(z) + λ diversity(z, archive)
```

where:

```text
diversity(z, archive) = minimum distance from z to archived candidates
```

### 14.6 Risk-Aware Acquisition

Penalize candidates predicted to increase latency, cost, verbosity, or invalid output risk:

```text
a_risk(z) = μ_t(z) + κσ_t(z) - α cost(z) - β latency(z) - γ invalidity_risk(z)
```

Recommended default:

```text
a(z) = μ_t(z) + κσ_t(z) + λ diversity(z) - α cost(z)
```

---

## 15. Archive and Selection Strategy

BESO should not maintain only one best skill.

It should maintain an archive with:

1. best-average candidates,
    
2. Pareto-specialized candidates,
    
3. diverse candidates,
    
4. candidates that reveal useful negative information.
    

### 15.1 Score Matrix

Let:

```text
S[k, i] = score of candidate k on example i
```

For each example:

```text
s_i* = max_k S[k, i]
```

Winner set:

```text
W_i = {z_k : S[k, i] = s_i*}
```

Candidate win count:

```text
f(z_k) = |{i : z_k ∈ W_i}|
```

Selection probability:

```text
P(z_k) = f(z_k) / Σ_j f(z_j)
```

### 15.2 Archive Entry Schema

```json
{
  "candidate_id": "skill_v8",
  "parent_id": "skill_v6",
  "artifact": "<skill document>",
  "scores": {
    "optimization_mean": 0.71,
    "validation_mean": 0.68,
    "format_score": 0.94,
    "cost_per_task": 0.012,
    "latency_seconds": 4.2
  },
  "lineage_depth": 5,
  "winning_examples": ["ex_003", "ex_018", "ex_041"],
  "known_strengths": ["multi-hop decomposition", "format compliance"],
  "known_weaknesses": ["slower on direct questions"],
  "accepted_edit_summary": "Added intermediate-answer verification step.",
  "created_at_iteration": 12
}
```

### 15.3 Archive Pruning

Archive pruning should preserve:

- top-k by validation score,
    
- top-k by Pareto win count,
    
- top-k by diversity,
    
- most informative failed candidates.
    

Default:

```yaml
max_archive_size: 32
top_by_validation: 8
top_by_pareto: 8
top_by_diversity: 8
top_failed_informative: 8
```

---

## 16. Evaluation Protocol

### 16.1 Dataset Splits

Use at least four splits:

|Split|Purpose|
|---|---|
|feedback_train|Generate trajectories and reflections|
|optimization_minibatch|Fast candidate scoring|
|validation_gate|Accept/reject candidates|
|final_test|Report final unbiased results|

### 16.2 Why Separate Splits Matter

If candidates are generated and accepted on the same examples, the optimizer may overfit. A separate validation gate reduces this risk.

### 16.3 Recommended Split

For small datasets:

```text
40% feedback_train
20% optimization_minibatch rotation
20% validation_gate
20% final_test
```

For benchmark datasets with fixed train/test:

```text
train split → feedback + optimization + validation
held-out test split → final test only
```

### 16.4 Evaluation Metrics

Primary metric depends on task:

|Task Type|Metric|
|---|---|
|Math QA|exact answer accuracy|
|Code generation|pass@1 or unit test pass rate|
|Classification|accuracy, macro-F1|
|Extraction|field-level F1|
|JSON output|schema validity + content accuracy|
|Tool-use task|final accuracy + tool efficiency|

Secondary metrics:

- token cost,
    
- latency,
    
- invalid output rate,
    
- number of tool calls,
    
- verbosity,
    
- calibration / uncertainty behavior,
    
- robustness on hard examples.
    

### 16.5 Acceptance Rule

A candidate is accepted if:

```text
validation_score(candidate) > validation_score(parent) + δ
```

where:

```text
δ = minimum improvement threshold
```

Optional statistical guard:

```text
accept if improvement is positive on bootstrap confidence interval
```

### 16.6 Rejection Rule

Reject candidate if:

- validation score decreases,
    
- output validity drops below threshold,
    
- cost increases beyond budget,
    
- skill becomes too long,
    
- candidate violates invariants,
    
- candidate is too similar to archived candidates without improvement.
    

---

## 17. Reflection Module

### 17.1 Role

The reflection module transforms trajectory evidence into candidate edits.

It should not directly decide acceptance. It proposes edits. Evaluation decides whether the edits survive.

### 17.2 Reflection Prompt Inputs

The reflection model receives:

```text
- current skill artifact
- task examples
- successful trajectories
- failed trajectories
- evaluator feedback
- previous accepted edits
- rejected edits
- edit budget
- target section constraints
```

### 17.3 Reflection Output

The reflection model must output structured JSON:

```json
{
  "diagnosis": "The model often answers intermediate facts instead of resolving the original question.",
  "failure_modes": [
    "Premature final answer",
    "Weak final-question alignment"
  ],
  "proposed_edits": [
    {
      "operation": "add_rule",
      "target_section": "verification_checklist",
      "text": "Before finalizing, confirm that the answer resolves the original user question, not merely an intermediate sub-question.",
      "rationale": "Prevents premature answers in multi-hop questions.",
      "risk": "May add unnecessary checking on simple questions."
    }
  ]
}
```

### 17.4 Reflection Quality Checks

Reject reflection outputs if:

- invalid JSON,
    
- proposed edit exceeds budget,
    
- target section does not exist,
    
- rationale is missing,
    
- edit contradicts existing constraints,
    
- edit adds vague advice without operational behavior,
    
- edit is a duplicate of rejected edit.
    

---

## 18. Skill Compiler

### 18.1 Purpose

The skill compiler turns a skill artifact into runtime prompts.

### 18.2 Compiler Inputs

```text
skill artifact
task input
module role
available tools
output schema
runtime constraints
```

### 18.3 Compiler Output

```text
runtime prompt
```

### 18.4 Compiler Modes

#### Full Injection

Inject the full skill into the prompt.

Pros:

- maximum instruction availability.
    

Cons:

- high token cost,
    
- possible distraction.
    

#### Section Selection

Inject only relevant sections.

Pros:

- efficient,
    
- modular.
    

Cons:

- requires routing.
    

#### Distilled Prompt

Summarize the skill into a compact runtime instruction.

Pros:

- low token cost.
    

Cons:

- may lose detail.
    

Recommended v0:

> Use section selection with deterministic templates.

---

## 19. System Architecture

### 19.1 Components

```text
Dataset Manager
Evaluation Runner
Trajectory Logger
Metric Evaluator
Reflection Proposer
Edit Applicator
Candidate Featurizer
Bayesian Surrogate
Acquisition Selector
Validation Gate
Archive Manager
Skill Compiler
Experiment Tracker
```

### 19.2 Architecture Flow

```text
Dataset Manager
    ↓
Evaluation Runner ← Skill Compiler ← Candidate Skill
    ↓
Trajectory Logger
    ↓
Metric Evaluator
    ↓
Reflection Proposer
    ↓
Edit Applicator
    ↓
Candidate Pool
    ↓
Candidate Featurizer
    ↓
Bayesian Surrogate
    ↓
Acquisition Selector
    ↓
Validation Gate
    ↓
Archive Manager
    ↓
Best Skill Artifact
```

---

## 20. Data Models

### 20.1 Task Example

```json
{
  "example_id": "ex_001",
  "input": "Question text or task input",
  "metadata": {
    "expected_answer": "...",
    "rubric": "...",
    "difficulty": "medium",
    "category": "multi_step_reasoning"
  }
}
```

### 20.2 Trajectory Record

```json
{
  "trajectory_id": "traj_001",
  "candidate_id": "skill_v3",
  "example_id": "ex_001",
  "compiled_prompt": "...",
  "model_outputs": [
    {
      "step": 1,
      "type": "reasoning_summary",
      "content": "Identified intermediate entity."
    },
    {
      "step": 2,
      "type": "tool_call",
      "tool_name": "search",
      "arguments": {"query": "..."}
    }
  ],
  "final_output": "...",
  "score": 0.0,
  "feedback": "The answer resolves the intermediate entity but not the final question.",
  "cost": {
    "input_tokens": 1200,
    "output_tokens": 340,
    "tool_calls": 1
  },
  "latency_seconds": 5.3
}
```

### 20.3 Candidate Record

```json
{
  "candidate_id": "skill_v5_candidate_2",
  "parent_id": "skill_v5",
  "artifact": "<skill document>",
  "edit_record": {
    "operation": "add_rule",
    "target_section": "verification_checklist",
    "text": "...",
    "rationale": "..."
  },
  "features": "<feature vector>",
  "surrogate_prediction": {
    "mean": 0.69,
    "uncertainty": 0.08,
    "acquisition_score": 0.77
  },
  "evaluation_result": null
}
```

### 20.4 Evaluation Result

```json
{
  "candidate_id": "skill_v5_candidate_2",
  "split": "validation_gate",
  "mean_score": 0.71,
  "standard_error": 0.04,
  "num_examples": 25,
  "secondary_metrics": {
    "format_validity": 0.96,
    "avg_latency_seconds": 4.9,
    "avg_tool_calls": 1.2,
    "avg_tokens": 1320
  },
  "accepted": true,
  "acceptance_reason": "Validation score improved by 0.04 over parent."
}
```

### 20.5 Rejected Edit Record

```json
{
  "edit_id": "edit_00031",
  "candidate_id": "skill_v9_candidate_1",
  "reason": "Increased verbosity and reduced direct-answer accuracy.",
  "target_section": "core_procedure",
  "operation": "add_rule",
  "text": "...",
  "observed_failure": "The model over-explained simple examples.",
  "do_not_repeat": true
}
```

---

## 21. Baselines

BESO should be evaluated against both prompt-level and skill-level baselines.

### 21.1 No Optimization

Use the initial prompt or initial skill.

### 21.2 Manual Skill

Human-written skill document.

### 21.3 One-Shot LLM Skill

Ask an LLM to generate a skill once, with no iterative optimization.

### 21.4 Random Search

Randomly generate candidate edits and evaluate them.

### 21.5 Greedy Reflection

Always apply the edit that looks best according to reflection, without Bayesian selection.

### 21.6 GEPA-Style Prompt Evolution

Reflectively evolve prompts directly.

### 21.7 SkillOpt-Style Bounded Skill Edits

Use bounded add/delete/replace edits with validation gates, but no Bayesian surrogate.

### 21.8 Bandit Selection

Use multi-armed bandit selection over edit types or parent candidates.

### 21.9 Bayesian Prompt Optimization

Apply Bayesian selection directly to raw prompt candidates.

This baseline tests whether skill-level abstraction matters.

---

## 22. Ablation Studies

Required ablations:

|Ablation|Purpose|
|---|---|
|No Bayesian surrogate|Test value of Bayesian selection|
|No reflection|Test value of semantic edit generation|
|No Pareto archive|Test value of diversity preservation|
|Prompt-only search|Test value of skill abstraction|
|No rejected-edit buffer|Test stability benefit|
|No structured features|Test value of feature engineering|
|Embeddings only|Test whether semantic embeddings are enough|
|Structured features only|Test whether cheap features are enough|
|No validation gate|Test overfitting risk|
|Full skill injection vs section selection|Test compiler strategy|

---

## 23. Experimental Plan

### 23.1 Phase 1: Toy Validation

Goal:

> Verify that the optimizer loop works.

Tasks:

- small arithmetic word problems,
    
- simple classification,
    
- JSON extraction.
    

Budget:

```yaml
rollouts: 50-200
candidate_pool_size: 8-16
archive_size: 8
```

Success criterion:

```text
BESO improves over initial skill and random search.
```

### 23.2 Phase 2: Benchmark Evaluation

Tasks:

- multi-step QA,
    
- math reasoning,
    
- code generation,
    
- structured extraction,
    
- tool-use tasks.
    

Budget:

```yaml
rollouts: 200-1000
candidate_pool_size: 16-64
archive_size: 16-32
```

Success criterion:

```text
BESO beats or matches GEPA-style prompt evolution and SkillOpt-style bounded editing under the same rollout budget.
```

### 23.3 Phase 3: Low-Budget Stress Test

Evaluate at budgets:

```text
25, 50, 100, 200, 500 rollouts
```

Primary hypothesis:

> BESO should show its advantage most clearly under low rollout budgets.

### 23.4 Phase 4: Transfer Test

Train skill on one setting and test on:

- nearby task distribution,
    
- harder benchmark split,
    
- different model,
    
- different execution harness.
    

Success criterion:

```text
Optimized skill retains value outside the exact optimization setting.
```

---

## 24. Metrics

### 24.1 Primary Metrics

- final test accuracy,
    
- validation score under fixed budget,
    
- area under optimization curve,
    
- best score achieved at budget B.
    

### 24.2 Sample Efficiency Metrics

```text
score after 25 rollouts
score after 50 rollouts
score after 100 rollouts
rollouts required to beat baseline
```

### 24.3 Robustness Metrics

- performance on hard examples,
    
- variance across random seeds,
    
- sensitivity to initial skill,
    
- sensitivity to evaluator noise.
    

### 24.4 Cost Metrics

- total tokens used during optimization,
    
- inference-time token overhead,
    
- total model calls,
    
- total tool calls,
    
- latency.
    

### 24.5 Interpretability Metrics

- number of accepted edits,
    
- percentage of accepted edits with clear rationale,
    
- human rating of skill readability,
    
- edit locality,
    
- presence of contradictory rules.
    

---

## 25. Main Hypotheses

### H1: Skill-Level Optimization Beats Prompt-Level Optimization

Structured skill artifacts will produce better generalization and more stable optimization than raw prompt strings.

### H2: Bayesian Acquisition Improves Sample Efficiency

Bayesian candidate selection will outperform random, greedy, and Pareto-only selection under low rollout budgets.

### H3: Reflection Improves Candidate Quality

Reflection-generated edits will outperform mutation operators that do not use trajectory feedback.

### H4: Pareto Archives Improve Robustness

Maintaining diverse specialized candidates will improve final performance compared with keeping only the best-average candidate.

### H5: Optimized Skills Transfer

Optimized skill artifacts will retain some performance gain when transferred to nearby tasks or models.

---

## 26. Failure Modes

### 26.1 Surrogate Miscalibration

The Bayesian model may predict high value for candidates that fail in real evaluation.

Mitigation:

- use uncertainty calibration,
    
- use ensemble disagreement,
    
- periodically evaluate random candidates,
    
- track prediction error.
    

### 26.2 Overfitting to Validation Examples

The skill may become too tailored to validation tasks.

Mitigation:

- separate feedback, optimization, validation, and test splits,
    
- rotate minibatches,
    
- use held-out final test,
    
- penalize overly specific rules.
    

### 26.3 Prompt Bloat

The skill may grow too long.

Mitigation:

- token budget,
    
- compression edits,
    
- cost-aware acquisition,
    
- inference-time section selection.
    

### 26.4 Contradictory Rules

Multiple edits may introduce conflicting instructions.

Mitigation:

- contradiction checker,
    
- skill linting,
    
- reflection prompt must identify conflicts,
    
- periodic consolidation.
    

### 26.5 Reflection Hallucination

The reflection model may invent failure causes not supported by traces.

Mitigation:

- require trace-grounded rationales,
    
- cite trajectory IDs in edit proposals,
    
- reject unsupported edits.
    

### 26.6 Edit Myopia

Small bounded edits may fail to discover larger useful shifts.

Mitigation:

- occasional large mutation,
    
- crossover between archived skills,
    
- meta-edit phase every N iterations.
    

### 26.7 Evaluation Noise

LLM judges and stochastic model outputs may produce noisy scores.

Mitigation:

- repeated evaluation for top candidates,
    
- bootstrap confidence intervals,
    
- deterministic decoding when possible,
    
- judge consistency checks.
    

---

## 27. Invariants and Safety Checks

The optimizer must preserve these invariants:

1. Skill artifact must remain valid under schema.
    
2. Skill must not contradict task rules.
    
3. Output format requirements must remain intact.
    
4. Tool-use policy must not authorize unavailable tools.
    
5. Token count must stay under configured budget.
    
6. No accepted edit may reduce validation score beyond tolerance.
    
7. Final reported score must use untouched final test split.
    

Skill linting checks:

```text
schema_validity
section_presence
token_budget
contradiction_check
unsafe_instruction_check
format_rule_preservation
unavailable_tool_check
```

---

## 28. Implementation Plan

### 28.1 Minimal v0

Goal:

> Build the simplest working optimizer.

Components:

- skill schema,
    
- evaluation runner,
    
- trajectory logger,
    
- reflection edit proposer,
    
- edit applicator,
    
- simple ensemble surrogate,
    
- UCB acquisition,
    
- validation gate,
    
- archive manager.
    

Recommended stack:

```yaml
language: Python
storage: SQLite or local JSONL
experiment_tracking: Weights & Biases, MLflow, or local logs
surrogate_models: scikit-learn
LLM_calls: provider-agnostic adapter
```

### 28.2 v0 Algorithm

```text
Skill artifact only.
Single target model.
Single benchmark.
Single scoring metric.
UCB acquisition.
Ensemble surrogate.
Validation-gated acceptance.
```

### 28.3 v1

Add:

- multi-objective scoring,
    
- Pareto archive,
    
- rejected-edit buffer,
    
- section-selection compiler,
    
- more baselines,
    
- ablation suite.
    

### 28.4 v2

Add:

- transfer evaluation,
    
- hierarchical skill-to-module prompt compilation,
    
- task-adaptive skill routing,
    
- multi-model skill robustness,
    
- inference-time cost optimization.
    

---

## 29. Suggested Repository Structure

```text
beso/
  README.md
  pyproject.toml
  configs/
    default.yaml
    experiments/
  beso/
    __init__.py
    artifacts/
      skill.py
      prompt.py
      schema.py
    compiler/
      skill_compiler.py
      section_selector.py
    evaluation/
      runner.py
      metrics.py
      judge.py
      splits.py
    trajectories/
      logger.py
      store.py
      filters.py
    reflection/
      proposer.py
      prompts.py
      validators.py
    edits/
      operations.py
      applicator.py
      lint.py
    features/
      featurizer.py
      embeddings.py
      semantic_labels.py
    surrogate/
      base.py
      ensemble.py
      gaussian_process.py
      tpe.py
    acquisition/
      ucb.py
      expected_improvement.py
      thompson.py
      diversity.py
    archive/
      manager.py
      pareto.py
      lineage.py
    optimization/
      loop.py
      accept_reject.py
      budget.py
    experiments/
      baselines.py
      ablations.py
      reporting.py
  tests/
    test_skill_schema.py
    test_edit_applicator.py
    test_archive.py
    test_acquisition.py
  examples/
    arithmetic_word_problems/
    json_extraction/
    multi_step_qa/
```

---

## 30. Configuration Example

```yaml
experiment:
  name: beso_multi_step_qa_v0
  seed: 42

artifact:
  type: skill
  max_tokens: 900
  compiler_mode: section_selection

optimization:
  max_rollouts: 300
  max_iterations: 30
  batch_size: 2
  candidate_pool_size: 24
  archive_size: 32
  min_improvement_delta: 0.01

edits:
  max_added_tokens_per_iteration: 120
  max_deleted_tokens_per_iteration: 80
  max_replaced_tokens_per_iteration: 160
  max_sections_modified_per_iteration: 2
  allowed_operations:
    - add_rule
    - delete_rule
    - replace_rule
    - add_failure_mode
    - add_recovery_rule
    - compress_section

surrogate:
  type: ensemble
  models:
    - bayesian_ridge
    - random_forest
    - knn_embedding
  uncertainty: ensemble_disagreement

acquisition:
  type: ucb_diversity_cost
  kappa: 1.5
  diversity_lambda: 0.2
  cost_alpha: 0.1

archive:
  strategy: pareto_and_diversity
  top_by_validation: 8
  top_by_pareto: 8
  top_by_diversity: 8
  top_failed_informative: 8

reflection:
  model: optimizer_model
  require_trace_grounding: true
  include_rejected_edits: true

evaluation:
  target_model: target_model
  decoding_temperature: 0.0
  metric: exact_or_judge_score
  repeated_eval_for_top_candidates: true
```

---

## 31. Acceptance Criteria for v0

BESO v0 is successful if it demonstrates:

1. End-to-end optimization loop runs without manual intervention.
    
2. Skill artifacts remain schema-valid after edits.
    
3. Candidate evaluations are logged with trajectories and scores.
    
4. Bayesian surrogate selects candidates based on acquisition score.
    
5. Final optimized skill improves over initial skill on validation.
    
6. Final test result is reported on untouched test split.
    
7. At least one baseline comparison is included.
    
8. Optimization trace is inspectable.
    

Minimum experimental success:

```text
BESO > initial skill
BESO > random edit search
```

Stronger success:

```text
BESO ≥ SkillOpt-style bounded edits under same rollout budget
BESO ≥ GEPA-style prompt evolution under same rollout budget
```

---

## 32. Reporting Format

Each experiment should report:

```text
- dataset
- target model
- initial skill
- optimizer configuration
- rollout budget
- final optimized skill
- validation score curve
- final test score
- baselines
- ablations
- cost
- accepted edits
- rejected edits
- qualitative failure analysis
```

### 32.1 Optimization Curve

Report score as a function of rollout budget:

```text
rollouts → best validation score
```

This is crucial because BESO’s central claim is sample efficiency.

### 32.2 Edit Trace

Report accepted edits:

|Iteration|Section|Operation|Rationale|Validation Change|
|---|---|---|---|---|
|1|verification_checklist|add_rule|Prevent intermediate-answer errors|+0.03|
|4|output_rules|replace_rule|Reduce verbosity|+0.02|
|9|tool_use_policy|add_rule|Improve calculator use|+0.04|

---

## 33. Research Contribution Claim

The strongest possible contribution is not merely:

> We use Bayesian optimization for prompts.

That is too broad and likely not novel enough.

The stronger claim is:

> We introduce a Bayesian-guided evolutionary optimizer for natural-language skill artifacts, where trajectory-grounded reflection proposes bounded semantic edits and a probabilistic surrogate selects which skill variants to evaluate under a limited rollout budget.

The specific contributions:

1. A skill-level optimization target that is more structured than raw prompts.
    
2. A Bayesian surrogate for predicting candidate skill utility and uncertainty.
    
3. Acquisition-guided selection of natural-language edits.
    
4. A Pareto-diverse archive for preserving specialized skill variants.
    
5. A controlled evaluation against prompt evolution, gradient-like text editing, and random/greedy baselines.
    

---

## 34. Expected Strengths

BESO should be strongest when:

- rollout budget is small,
    
- evaluations are expensive,
    
- trajectories contain useful diagnostic feedback,
    
- the task benefits from reusable procedures,
    
- prompt wording is not enough and higher-level skill policy matters,
    
- there are multiple competing behavioral strategies.
    

---

## 35. Expected Weaknesses

BESO may struggle when:

- the evaluation metric is weak,
    
- feedback is vague,
    
- the task has little reusable structure,
    
- skill edits do not affect the bottleneck,
    
- the surrogate cannot model text-performance relationships,
    
- the candidate generator produces low-quality edits,
    
- the optimized skill overfits to benchmark quirks.
    

---

## 36. Token-Cost Motivation and Cost-Aware BESO

A practical concern with SkillOpt-style skill optimization is token consumption. The cost issue has two separate forms:

1. **Optimization-time token cost**
    
    - target-agent rollouts during training,
        
    - optimizer-model reflection calls,
        
    - candidate validation calls,
        
    - repeated evaluation of candidate edits,
        
    - rejected-edit context,
        
    - slow/meta update context.
        
2. **Inference-time token cost**
    
    - the final skill document injected into the runtime context,
        
    - extra instructions added by the skill,
        
    - longer model outputs caused by more verbose procedures,
        
    - additional tool calls caused by the learned policy.
        

BESO should explicitly address both, but its primary advantage is expected at **optimization time**.

SkillOpt-style optimization can be described as:

```text
rollout → reflect → propose bounded edit → evaluate candidate → accept/reject
```

If the optimizer proposes weak or redundant edits, the system may still spend expensive rollout and validation budget evaluating them.

BESO inserts a budget-aware selection layer:

```text
rollout → reflect → propose many candidate edits
        → surrogate predicts utility + uncertainty + cost
        → acquisition selects only worthwhile candidates
        → evaluate selected candidates
        → accept/reject/archive
```

If reflection proposes `M` candidate edits and each candidate requires `b` rollout examples to evaluate, naive evaluation costs:

```text
Cost_naive ≈ M × b
```

BESO evaluates only `K` selected candidates, where `K << M`:

```text
Cost_BESO ≈ K × b
```

The Bayesian layer is therefore not just about quality. It can be framed as **cost-aware experiment planning**.

### 36.1 Cost-aware acquisition

The acquisition function should include token and rollout cost directly:

```text
a_BESO(z) = μ_t(z) + κσ_t(z) + λ diversity(z)
            - α C_train(z)
            - β C_infer(z)
            - γ invalidity_risk(z)
```

where:

```text
μ_t(z) = predicted candidate utility
σ_t(z) = surrogate uncertainty
diversity(z) = distance from archived candidates
C_train(z) = predicted optimization-time cost
C_infer(z) = predicted inference-time cost
invalidity_risk(z) = predicted probability of schema/output failure
```

Training cost can be decomposed as:

```text
C_train(z)
= rollout_tokens(z)
+ reflection_tokens(z)
+ validation_tokens(z)
+ tool_call_cost(z)
```

Inference cost can be decomposed as:

```text
C_infer(z)
= token_count(compiled_skill)
+ expected_extra_output_tokens(z)
+ expected_extra_tool_tokens(z)
```

This turns candidate selection into the question:

> Is this edit worth the tokens?

not merely:

> Does this edit improve score?

### 36.2 Cost-aware objective

A cost-aware utility can be defined as:

```text
U(z) = J_quality(z) - λ_train C_train(z) - λ_infer C_infer(z)
```

where:

```text
J_quality(z) = expected task score
C_train(z) = optimization-time token/call cost
C_infer(z) = deployment-time token/call cost
```

Then BESO selects candidates by uncertainty-aware expected utility:

```text
z_(t+1) = argmax_{z ∈ C_t} E[U(z) | H_t] + κ Var[U(z) | H_t]^(1/2)
```

This makes token efficiency a first-class optimization target rather than a post-hoc concern.

### 36.3 Hard token constraints

BESO should also support hard constraints:

```text
Tokens(z) ≤ T_skill_max
Tokens(C(z, x, q)) ≤ T_runtime_max
C_train(z) ≤ C_train_max
C_infer(z) ≤ C_infer_max
```

Hard constraints are stricter than cost penalties. A candidate that exceeds the deployment token budget should be invalid even if it improves validation accuracy.

### 36.4 Compression and section-selection compiler

A major way for BESO to address inference-time cost is to optimize not only the skill artifact, but also the compiler that projects the artifact into runtime context.

Instead of always injecting the full skill:

```text
p = C_full(z)
```

BESO can use section selection:

```text
p = C_select(z, x, q)
```

For example, a simple math task may need only:

```text
Core Procedure
Verification Checklist
Output Rules
```

and may not need:

```text
All failure modes
All examples
Full recovery policy
Change log
```

This allows the skill artifact to remain rich while the runtime prompt stays compact.

A stronger BESO formulation optimizes both the skill artifact and its compiler:

```text
z*, C* = argmax_{z,C} E[ μ(Φ(x; C(z,x,q), Θ_frozen), m) ]
```

subject to:

```text
E[ Tokens(C(z,x,q)) ] ≤ T_budget
```

This extends the research question from:

> What skill should we learn?

into:

> What skill should we learn, and which parts of that skill should be injected for this task?

### 36.5 Positioning against SkillOpt

SkillOpt already tries to keep deployed skills compact, but its optimization process can still be token-heavy because it uses rollout batches, optimizer reflection, validation gates, rejected-edit feedback, and slow/meta updates.

BESO addresses this at the experiment-planning layer:

> Instead of evaluating every reflected edit, BESO predicts which edits are worth their rollout cost.

If compiler optimization is included, BESO also addresses deployment cost:

> Instead of injecting the whole learned skill, BESO learns or selects a compact task-relevant skill projection.

A concise contribution claim:

> SkillOpt makes skill learning stable through bounded edits and validation gates, but still spends rollout and reflection budget on candidate updates before knowing whether they are promising. BESO treats candidate evaluation as a budgeted Bayesian experiment: it models expected improvement, uncertainty, and token cost, then evaluates only candidates whose expected value justifies their cost.

An even shorter positioning:

> BESO is cost-aware skill evolution under a rollout and token budget.

### 36.6 Caveat

BESO does not reduce token use automatically. If the acquisition function optimizes only quality, BESO may still learn bloated skills. It may even prefer longer skills if longer skills correlate with higher validation performance.

Therefore token consumption must be included through:

- cost-aware acquisition,
    
- hard token constraints,
    
- compression edits,
    
- section-selection compilation,
    
- inference-time cost reporting.
    

---

## 37. Open Design Questions

1. What is the best feature representation for skill artifacts?
    
2. Should the surrogate model predict absolute performance or improvement over parent?
    
3. Should acquisition happen at candidate level, edit-operation level, or section level?
    
4. Should the archive preserve failed candidates for negative learning?
    
5. How large should bounded edits be?
    
6. How often should the optimizer perform compression or consolidation?
    
7. How should the skill compiler decide which sections to inject?
    
8. Can optimized skills transfer across model families?
    
9. Can Bayesian uncertainty remain useful in high-dimensional text spaces?
    
10. Does Pareto diversity improve final generalization or only optimization-time exploration?
    

---

## 38. Recommended First Prototype

The first prototype should be deliberately small.

### Task

Use one task type with clear scoring, such as:

```text
arithmetic word problems
structured JSON extraction
small code-generation unit tests
```

### Artifact

Use one skill document.

### Optimizer

Use:

```text
reflection-generated bounded edits
ensemble surrogate
UCB acquisition
validation-gated acceptance
simple archive
```

### Baselines

Compare against:

```text
initial skill
random edits
greedy reflection edits
prompt-only Bayesian search
```

### Success Criterion

The prototype is promising if:

```text
BESO achieves higher validation score than random and greedy baselines under the same rollout budget.
```

Do not start with a complex multi-agent task. First prove the optimizer works on a small controlled benchmark.

---

## 39. Final Summary

BESO is a proposed optimizer for frozen LLM systems that treats natural-language skills as trainable external state.

The core loop is:

```text
Run skill → collect trajectories → reflect → generate edits → featurize candidates → predict utility and uncertainty → choose candidates with Bayesian acquisition → evaluate → validate → archive → repeat
```

The central mathematical objective is:

```text
z* = argmax_z E_(x,m)~T [ μ(Φ(x; z, Θ_frozen), m) ]
```

where:

```text
z = skill artifact
Φ = frozen LLM system
μ = task metric
```

The central engineering decision is:

> Optimize structured skills, not raw prompts.

The central research hypothesis is:

> Bayesian-guided skill evolution should be more sample-efficient than purely evolutionary or gradient-like prompt/skill optimization, especially under low rollout budgets.

If validated, BESO would occupy a meaningful middle ground between GEPA-style reflective prompt evolution and SkillOpt-style trainable skill documents: it keeps the interpretability of natural-language skills, the flexibility of evolutionary search, and the sample-efficiency discipline of Bayesian experiment planning.