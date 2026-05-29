---
type: research-note
tags: [beso, bayesian-optimization, evolutionary-search, skill-optimization, llm-agents]
date: 2026-05-28
source: [[Bayesian Evolutionary Skill Optimization (BESO) - Technical Specification]]
status: draft
---

# Bayesian Evolutionary Skill Optimization (BESO): mathematical breakdown

## 0. One-line mathematical summary

BESO optimizes a natural-language skill artifact $z$ for a frozen LLM system $\Phi$ by using trajectory-grounded reflection to generate candidate edits, a Bayesian surrogate to model uncertain candidate utility, an acquisition function to allocate a limited rollout budget, and an evolutionary archive to preserve high-performing or specialized skill variants.

The central objective is:

$$
z^* \in \arg\max_{z \in \mathcal{Z}} \; J(z)
= \arg\max_{z \in \mathcal{Z}} \; \mathbb{E}_{(x,m) \sim \mathcal{T}}\left[\mu\left(\Phi(x; C(z), \Theta_{\mathrm{frozen}}), m\right)\right]
$$

where:

- $z$ is a structured skill artifact.
- $\mathcal{Z}$ is the space of valid skill artifacts.
- $C$ is the skill compiler that converts a skill artifact into runtime prompt material.
- $\Phi$ is the frozen LLM system or agent runtime.
- $\Theta_{\mathrm{frozen}}$ are fixed model weights.
- $x$ is a task input.
- $m$ is the task metadata, label, rubric, expected answer, or test case.
- $\mu$ is the evaluator.
- $\mathcal{T}$ is the unknown task distribution.
- $J(z)$ is the true expected utility of skill $z$.

In practical form, BESO solves a budgeted, noisy, black-box, discrete optimization problem:

$$
\max_{z \in \mathcal{Z}} J(z)
\quad \text{subject to} \quad
\sum_{t=1}^{T} c(z_t, B_t) \le B
$$

where $B$ is the total rollout budget and $c(z_t, B_t)$ is the cost of evaluating candidate $z_t$ on minibatch $B_t$.

---

## 1. Objects and spaces

### 1.1 Task distribution

Let the task distribution be:

$$
\mathcal{T} \in \Delta(\mathcal{X} \times \mathcal{M})
$$

where:

- $\mathcal{X}$ is the input space.
- $\mathcal{M}$ is the space of metadata, labels, rubrics, test cases, or evaluator references.
- $\Delta(\cdot)$ denotes a probability distribution over a space.

A task instance is:

$$
\xi = (x,m) \sim \mathcal{T}
$$

For a multi-hop QA task, $x$ might be the question and $m$ might be the expected answer plus rubric. For code generation, $x$ might be a problem statement and $m$ might be a unit-test suite. For structured extraction, $x$ might be a document and $m$ might be the target JSON fields.

### 1.2 Dataset approximation

The true distribution $\mathcal{T}$ is unknown. We observe a finite dataset:

$$
D = \{\xi_i\}_{i=1}^{n} = \{(x_i,m_i)\}_{i=1}^{n}
$$

BESO should split this dataset into disjoint roles:

$$
D = D_{\mathrm{fb}} \cup D_{\mathrm{opt}} \cup D_{\mathrm{val}} \cup D_{\mathrm{test}}
$$

where:

- $D_{\mathrm{fb}}$ generates trajectories for reflection.
- $D_{\mathrm{opt}}$ gives fast candidate estimates during optimization.
- $D_{\mathrm{val}}$ gates candidate acceptance.
- $D_{\mathrm{test}}$ is untouched until final reporting.

This split matters because BESO edits text artifacts. Text artifacts can overfit by encoding benchmark quirks, overly specific rules, or accidental validation shortcuts.

### 1.3 Frozen system

Let the frozen LLM system be:

$$
\Phi_{\Theta}: \mathcal{X} \times \mathcal{P} \to \mathcal{Y} \times \mathcal{R}
$$

where:

- $\Theta = \Theta_{\mathrm{frozen}}$ are fixed model parameters.
- $\mathcal{P}$ is the runtime prompt / instruction space.
- $\mathcal{Y}$ is the final output space.
- $\mathcal{R}$ is the trajectory / trace space.

Given input $x$ and runtime prompt $p$, the system returns:

$$
(y, \tau) = \Phi_{\Theta_{\mathrm{frozen}}}(x; p)
$$

where:

- $y$ is the final output.
- $\tau$ is the recorded trajectory.

The trajectory may contain prompts, intermediate outputs, tool calls, tool results, retrieved documents, errors, evaluator feedback, costs, and latency. It should not be assumed to contain hidden chain-of-thought; it contains only the observable trace made available by the system.

### 1.4 Skill artifact space

BESO does not directly optimize model weights. It optimizes skill artifacts:

$$
z \in \mathcal{Z}
$$

A skill artifact can be modeled as a structured object:

$$
z = (s_1, s_2, \dots, s_L)
$$

where each $s_{\ell}$ is a section, such as:

- goal,
- scope,
- core procedure,
- reasoning policy,
- tool-use policy,
- verification checklist,
- common failure modes,
- recovery rules,
- output rules,
- examples,
- change log.

The space $\mathcal{Z}$ is not the space of all strings. It is the constrained space of schema-valid skill artifacts:

$$
\mathcal{Z} = \{z \in \Sigma^* : \mathrm{SchemaValid}(z)=1, \; \mathrm{BudgetValid}(z)=1, \; \mathrm{InvariantValid}(z)=1\}
$$

where $\Sigma^*$ is the space of finite text strings.

This distinction is important. Raw prompt optimization searches in an unstructured text space. BESO searches in a typed, sectioned, linted text space.

### 1.5 Skill compiler

The skill artifact is compiled into runtime prompt material:

$$
p = C(z, x, q)
$$

where:

- $C$ is the compiler.
- $z$ is the skill artifact.
- $x$ is the task input.
- $q$ is runtime context: available tools, output schema, module role, cost constraints, or selected skill sections.

The compiler may use one of several modes:

$$
C \in \{C_{\mathrm{full}}, C_{\mathrm{section}}, C_{\mathrm{distill}}\}
$$

where:

- $C_{\mathrm{full}}$ injects the whole skill.
- $C_{\mathrm{section}}$ selects relevant sections.
- $C_{\mathrm{distill}}$ compresses the skill into a compact prompt.

The compiled prompt is not the optimization target. It is the runtime consequence of the skill artifact.

---

## 2. Evaluation and objective functions

### 2.1 Single-example score

Given a skill artifact $z$ and task instance $\xi_i=(x_i,m_i)$:

$$
p_i = C(z, x_i, q_i)
$$

$$
(y_i, \tau_i) = \Phi_{\Theta_{\mathrm{frozen}}}(x_i; p_i)
$$

The evaluator assigns a score:

$$
r_i(z) = \mu(y_i, \tau_i, m_i) \in \mathbb{R}
$$

Often:

$$
r_i(z) \in [0,1]
$$

but the framework can support arbitrary bounded metrics.

The evaluator may depend on the final answer only:

$$
r_i(z) = \mu(y_i,m_i)
$$

or on the trajectory as well:

$$
r_i(z) = \mu(y_i,\tau_i,m_i)
$$

Trajectory-dependent scoring is important for tool-use tasks, invalid-output penalties, cost penalties, and safety constraints.

### 2.2 True expected skill utility

The ideal objective is:

$$
J(z) = \mathbb{E}_{(x,m)\sim\mathcal{T}}\left[\mu\left(y,\tau,m\right)\right]
$$

with:

$$
(y,\tau)=\Phi_{\Theta_{\mathrm{frozen}}}(x; C(z,x,q))
$$

Thus:

$$
J(z) = \mathbb{E}_{(x,m)\sim\mathcal{T}}\left[\mu\left(\Phi_{\Theta_{\mathrm{frozen}}}(x; C(z,x,q)),m\right)\right]
$$

The target skill is:

$$
z^* \in \arg\max_{z\in\mathcal{Z}} J(z)
$$

### 2.3 Empirical risk / empirical utility

Since $\mathcal{T}$ is unknown, define empirical utility on dataset $D$:

$$
\hat{J}_D(z)=\frac{1}{|D|}\sum_{(x_i,m_i)\in D} r_i(z)
$$

For a split $S \subset D$:

$$
\hat{J}_S(z)=\frac{1}{|S|}\sum_{i\in S} r_i(z)
$$

The optimization-time estimate may use minibatches:

$$
\hat{J}_{B_t}(z)=\frac{1}{|B_t|}\sum_{i\in B_t} r_i(z)
$$

where:

$$
B_t \subset D_{\mathrm{opt}}
$$

### 2.4 Noisy observation model

Evaluation is noisy because LLM outputs, judge scores, sampled minibatches, tool results, and retrieval contexts can vary.

Model an observed candidate score as:

$$
\tilde{y}_t = \hat{J}_{B_t}(z_t) + \epsilon_t
$$

where:

$$
\epsilon_t \sim \mathcal{N}(0, \sigma_{\epsilon}^2(z_t, B_t))
$$

This noise can be decomposed:

$$
\sigma_{\epsilon}^2
= \sigma_{\mathrm{model}}^2
+ \sigma_{\mathrm{judge}}^2
+ \sigma_{\mathrm{batch}}^2
+ \sigma_{\mathrm{tool}}^2
$$

where:

- $\sigma_{\mathrm{model}}^2$ comes from stochastic decoding or model variance.
- $\sigma_{\mathrm{judge}}^2$ comes from LLM-as-judge or rubric instability.
- $\sigma_{\mathrm{batch}}^2$ comes from using a minibatch rather than the whole dataset.
- $\sigma_{\mathrm{tool}}^2$ comes from nondeterministic tools, retrieval, network state, or environment state.

For deterministic decoding and deterministic evaluators, some of these terms shrink, but minibatch noise usually remains.

### 2.5 Cost-aware utility

BESO should not optimize quality alone if a skill becomes too expensive at runtime. Define cost:

$$
\mathrm{Cost}(z; x) = c_{\mathrm{tok}}(z,x) + c_{\mathrm{tool}}(z,x) + c_{\mathrm{lat}}(z,x)
$$

A scalarized cost-aware utility can be:

$$
U(z) = J_{\mathrm{quality}}(z) - \lambda_{\mathrm{tok}}J_{\mathrm{tokens}}(z) - \lambda_{\mathrm{tool}}J_{\mathrm{tools}}(z) - \lambda_{\mathrm{lat}}J_{\mathrm{latency}}(z)
$$

where:

$$
J_{\mathrm{tokens}}(z)=\mathbb{E}_{x\sim\mathcal{T}}[\mathrm{Tokens}(C(z,x,q))]
$$

This prevents prompt bloat from masquerading as improvement.

---

## 3. BESO as black-box optimization over structured text

### 3.1 Why this is black-box

The function $J(z)$ is not analytically available. We cannot compute gradients through:

- natural-language skill documents,
- discrete edit operations,
- LLM sampling,
- external tools,
- evaluator logic,
- unit tests,
- human or LLM judges.

So BESO treats $J$ as a black-box objective:

$$
z \mapsto J(z)
$$

We can query the function by evaluating candidates, but each query costs rollouts.

### 3.2 Why this is not ordinary Bayesian optimization

Classical Bayesian optimization usually assumes a continuous input space:

$$
z \in \mathbb{R}^d
$$

BESO has a structured discrete text space:

$$
z \in \mathcal{Z} \subset \Sigma^*
$$

Therefore BESO uses a two-stage strategy:

1. A reflection model proposes a finite candidate set:

$$
\mathcal{C}_t = G(A_t, H_t, \mathcal{T}_t)
$$

2. A Bayesian surrogate ranks candidates inside that finite set:

$$
z_{t+1} \in \arg\max_{z\in\mathcal{C}_t} a_t(z)
$$

This makes BESO closer to Bayesian-guided evolutionary search than pure continuous Bayesian optimization.

The Bayesian model does not need to search all possible text. It only needs to decide which generated candidates are worth evaluating.

---

## 4. History, archive, and state

### 4.1 Optimization history

After $t$ evaluated candidates, the history is:

$$
H_t = \{(z_j, B_j, \tilde{y}_j, \mathcal{R}_j, c_j)\}_{j=1}^{t}
$$

where:

- $z_j$ is an evaluated skill candidate.
- $B_j$ is the evaluation minibatch.
- $\tilde{y}_j$ is the observed score.
- $\mathcal{R}_j = \{\tau_{j,i}\}_{i\in B_j}$ is the set of trajectories.
- $c_j$ is the evaluation cost.

If candidate $z_j$ is evaluated multiple times, store repeated observations:

$$
H_t = \{(z_j, B_{j,k}, \tilde{y}_{j,k}, \mathcal{R}_{j,k}, c_{j,k})\}_{j,k}
$$

Then estimate its mean and uncertainty:

$$
\bar{y}_j = \frac{1}{K_j}\sum_{k=1}^{K_j}\tilde{y}_{j,k}
$$

$$
\widehat{\mathrm{SE}}(z_j)=\frac{\widehat{\sigma}(z_j)}{\sqrt{K_j}}
$$

### 4.2 Archive

The archive stores candidates that are useful for exploitation, diversity, specialization, or negative learning:

$$
A_t \subseteq \{z_1,\dots,z_t\}
$$

The archive may be decomposed:

$$
A_t = A_t^{\mathrm{best}} \cup A_t^{\mathrm{pareto}} \cup A_t^{\mathrm{diverse}} \cup A_t^{\mathrm{failed}}
$$

where:

- $A_t^{\mathrm{best}}$ contains high-average performers.
- $A_t^{\mathrm{pareto}}$ contains candidates that win on specific examples or objectives.
- $A_t^{\mathrm{diverse}}$ contains semantically distinct candidates.
- $A_t^{\mathrm{failed}}$ contains informative failures to avoid repeated mistakes.

### 4.3 Parent selection

Candidate generation starts from parents:

$$
P_t \subset A_t
$$

A parent selection distribution can combine validation score, Pareto win count, and diversity:

$$
\Pr(z \in P_t) \propto
\exp\left(
\beta_1 \hat{J}_{\mathrm{val}}(z)
+ \beta_2 w_t(z)
+ \beta_3 d(z,A_t)
- \beta_4 \mathrm{Cost}(z)
\right)
$$

where:

- $w_t(z)$ is a Pareto win-count or specialization score.
- $d(z,A_t)$ is a diversity measure.
- $\beta_1,\dots,\beta_4$ are selection weights.

---

## 5. Reflection-generated candidate edits

### 5.1 Edit operators

Let $\mathcal{E}$ be the space of valid edit operations:

$$
e \in \mathcal{E}
$$

Examples:

- add rule,
- delete rule,
- replace rule,
- specialize rule,
- generalize rule,
- reorder steps,
- add example,
- delete example,
- compress section,
- add failure mode,
- add recovery rule.

Each edit is a transformation:

$$
e: \mathcal{Z} \to \mathcal{Z}
$$

Applying edit $e$ to skill $z$ gives:

$$
z' = e(z)
$$

A sequence of edits $\mathbf{e}=(e_1,\dots,e_K)$ gives:

$$
z' = e_K \circ e_{K-1} \circ \cdots \circ e_1(z)
$$

### 5.2 Valid edit constraint

Not all edits are allowed. Define an edit validity indicator:

$$
\nu(z,e) \in \{0,1\}
$$

where:

$$
\nu(z,e)=1
$$

if and only if the edit preserves schema, token budget, section constraints, output format rules, tool availability, and task invariants.

Thus the feasible edit set for skill $z$ is:

$$
\mathcal{E}(z)=\{e\in\mathcal{E}: \nu(z,e)=1\}
$$

The feasible one-step neighborhood is:

$$
\mathcal{N}(z)=\{e(z): e\in\mathcal{E}(z)\}
$$

### 5.3 Reflection as proposal distribution

The reflection module is a proposal distribution over edits:

$$
Q_{\psi}(e \mid z, \mathcal{R}, F, A_t, H_t)
$$

where:

- $\psi$ are the reflection model parameters.
- $z$ is the parent skill.
- $\mathcal{R}$ is trajectory evidence.
- $F$ is evaluator feedback.
- $A_t$ is the archive.
- $H_t$ is the history.

The candidate pool is sampled from this distribution:

$$
e_{t,1},\dots,e_{t,M} \sim Q_{\psi}(\cdot \mid z, \mathcal{R}, F, A_t,H_t)
$$

$$
\mathcal{C}_t = \{e_{t,j}(z): j=1,\dots,M, \; \nu(z,e_{t,j})=1\}
$$

BESO can also use multiple parents:

$$
\mathcal{C}_t = \bigcup_{z\in P_t}\{e(z): e\sim Q_{\psi}(\cdot\mid z,\mathcal{R}_z,F_z,A_t,H_t),\; \nu(z,e)=1\}
$$

### 5.4 Crossover / semantic merge

For two parent skills $z_a,z_b\in A_t$, define a semantic merge operator:

$$
M_{\psi}: \mathcal{Z}\times\mathcal{Z}\times\mathcal{R} \to \mathcal{Z}
$$

The merged candidate is:

$$
z' = M_{\psi}(z_a,z_b,\mathcal{R}_{a,b})
$$

The merge should preserve complementary strengths:

$$
z' \approx \operatorname{Combine}\left(\mathrm{Strengths}(z_a), \mathrm{Strengths}(z_b)\right)
$$

but it must still satisfy:

$$
z' \in \mathcal{Z}
$$

This is not literal string crossover. It is semantic recombination under skill-schema constraints.

---

## 6. Candidate featurization

### 6.1 Feature map

Bayesian surrogates require numeric inputs. Define a feature map:

$$
\varphi: \mathcal{Z} \to \mathbb{R}^{d}
$$

A candidate skill becomes:

$$
\mathbf{x}_z = \varphi(z)
$$

To avoid confusion with task input $x$, use $\mathbf{x}_z$ for candidate features.

### 6.2 Feature decomposition

The feature map can be decomposed:

$$
\varphi(z) = \left[
\varphi_{\mathrm{text}}(z),
\varphi_{\mathrm{struct}}(z),
\varphi_{\mathrm{edit}}(z),
\varphi_{\mathrm{hist}}(z),
\varphi_{\mathrm{sem}}(z)
\right]
$$

where:

- $\varphi_{\mathrm{text}}(z)$ is an embedding of the skill text or changed section.
- $\varphi_{\mathrm{struct}}(z)$ includes token count, number of rules, number of examples, etc.
- $\varphi_{\mathrm{edit}}(z)$ encodes edit operation, target section, edit size, and parent ID.
- $\varphi_{\mathrm{hist}}(z)$ includes parent score, lineage depth, and edit-type success rates.
- $\varphi_{\mathrm{sem}}(z)$ includes LLM-labeled semantic properties such as verification emphasis or tool-use aggressiveness.

A useful first representation is:

$$
\varphi(z) = [E(z_{\Delta}); \; h(z); \; g(e); \; \ell(z)]
$$

where:

- $E(z_{\Delta})$ is the embedding of the changed text.
- $h(z)$ is structural metadata.
- $g(e)$ is edit-operation metadata.
- $\ell(z)$ is lineage and historical metadata.

### 6.3 Similarity kernel

If using a Gaussian-process-like surrogate, define a kernel over candidates:

$$
k(z,z') = k_{\mathrm{text}}(z,z') + k_{\mathrm{struct}}(z,z') + k_{\mathrm{edit}}(z,z')
$$

Text kernel:

$$
k_{\mathrm{text}}(z,z') = \exp\left(-\frac{\|E(z)-E(z')\|_2^2}{2\ell_{\mathrm{text}}^2}\right)
$$

Structural kernel:

$$
k_{\mathrm{struct}}(z,z') = \exp\left(-\frac{\|h(z)-h(z')\|_2^2}{2\ell_{\mathrm{struct}}^2}\right)
$$

Edit kernel:

$$
k_{\mathrm{edit}}(z,z') = \mathbb{1}[\mathrm{op}(z)=\mathrm{op}(z')] \cdot \mathbb{1}[\mathrm{section}(z)=\mathrm{section}(z')]
$$

A weighted composite kernel:

$$
k(z,z') = \alpha k_{\mathrm{text}}(z,z') + \beta k_{\mathrm{struct}}(z,z') + \gamma k_{\mathrm{edit}}(z,z')
$$

with:

$$
\alpha,\beta,\gamma \ge 0
$$

---

## 7. Bayesian surrogate

### 7.1 Unknown performance function

Let:

$$
f(z)=J(z)
$$

BESO cannot observe $f(z)$ directly. It observes noisy minibatch scores:

$$
\tilde{y}_t = f(z_t) + \epsilon_t
$$

or, more precisely:

$$
\tilde{y}_t = \hat{J}_{B_t}(z_t) + \epsilon_t
$$

The surrogate maintains a posterior:

$$
p(f \mid H_t)
$$

From this posterior, for any candidate $z$, it estimates:

$$
\mu_t(z)=\mathbb{E}[f(z)\mid H_t]
$$

$$
\sigma_t^2(z)=\mathrm{Var}[f(z)\mid H_t]
$$

### 7.2 Gaussian-process surrogate

If using a Gaussian process:

$$
f \sim \mathcal{GP}(m_0,k)
$$

Given evaluated feature vectors:

$$
X_t = [\varphi(z_1),\dots,\varphi(z_t)]^\top
$$

and observations:

$$
\mathbf{y}_t = [\tilde{y}_1,\dots,\tilde{y}_t]^\top
$$

The posterior predictive mean for candidate $z$ is:

$$
\mu_t(z)=m_0(z)+\mathbf{k}_t(z)^\top (K_t+\sigma_\epsilon^2 I)^{-1}(\mathbf{y}_t-m_0(X_t))
$$

The posterior predictive variance is:

$$
\sigma_t^2(z)=k(z,z)-\mathbf{k}_t(z)^\top (K_t+\sigma_\epsilon^2 I)^{-1}\mathbf{k}_t(z)
$$

where:

$$
\mathbf{k}_t(z)=[k(z_1,z),\dots,k(z_t,z)]^\top
$$

and:

$$
K_t[i,j]=k(z_i,z_j)
$$

A GP is elegant but may struggle if embeddings are high-dimensional and the number of candidates grows.

### 7.3 Bayesian linear / ridge surrogate

A simpler model assumes:

$$
f(z)=\mathbf{w}^\top \varphi(z)+\epsilon
$$

with prior:

$$
\mathbf{w}\sim\mathcal{N}(0,\lambda^{-1}I)
$$

and noise:

$$
\epsilon\sim\mathcal{N}(0,\sigma^2)
$$

Posterior:

$$
p(\mathbf{w}\mid H_t)=\mathcal{N}(\mathbf{m}_t, S_t)
$$

Then:

$$
\mu_t(z)=\mathbf{m}_t^\top\varphi(z)
$$

$$
\sigma_t^2(z)=\varphi(z)^\top S_t \varphi(z) + \sigma^2
$$

This is less expressive but robust and easy to inspect.

### 7.4 Ensemble surrogate

For v0, an ensemble surrogate may be more practical.

Let there be $M$ predictive models:

$$
\{g_m\}_{m=1}^{M}
$$

Each model predicts:

$$
\hat{f}_m(z)=g_m(\varphi(z))
$$

Mean prediction:

$$
\mu_t(z)=\frac{1}{M}\sum_{m=1}^{M}\hat{f}_m(z)
$$

Uncertainty from disagreement:

$$
\sigma_t^2(z)=\frac{1}{M-1}\sum_{m=1}^{M}(\hat{f}_m(z)-\mu_t(z))^2
$$

This uncertainty is not fully Bayesian, but it is often usable for acquisition when true posterior modeling is difficult.

### 7.5 Improvement model instead of absolute score model

Instead of modeling absolute utility $J(z)$, BESO can model improvement over parent:

$$
\Delta(z,z_p)=J(z)-J(z_p)
$$

Observation:

$$
\tilde{\Delta}_t = \tilde{y}(z_t)-\tilde{y}(z_{p(t)})
$$

The surrogate becomes:

$$
p(\Delta \mid H_t)
$$

This may be easier because candidate quality is often relative to the parent skill. An edit that helps one parent may not help another.

A useful hybrid score is:

$$
\mu_t^{\mathrm{hybrid}}(z)=\mu_t^{\mathrm{abs}}(z)+\rho\mu_t^{\Delta}(z,z_p)
$$

where $\rho$ controls how much the optimizer trusts improvement modeling.

---

## 8. Acquisition functions

### 8.1 Acquisition as budget allocation

The acquisition function decides which candidates deserve expensive evaluation.

At iteration $t$, candidate pool:

$$
\mathcal{C}_t=\{z_{t,1},\dots,z_{t,M}\}
$$

BESO selects:

$$
z_{t+1}\in\arg\max_{z\in\mathcal{C}_t}a_t(z)
$$

or, for batch size $K$:

$$
S_t\in\arg\max_{S\subset\mathcal{C}_t, |S|=K}\sum_{z\in S}a_t(z) - \eta\sum_{z,z'\in S}\mathrm{sim}(z,z')
$$

The second form discourages selecting near-duplicate candidates in the same batch.

### 8.2 Upper Confidence Bound

Default acquisition:

$$
a_{\mathrm{UCB}}(z)=\mu_t(z)+\kappa_t\sigma_t(z)
$$

where:

- $\mu_t(z)$ rewards predicted performance.
- $\sigma_t(z)$ rewards uncertainty.
- $\kappa_t$ controls exploration.

High $\kappa_t$ means "try uncertain candidates." Low $\kappa_t$ means "exploit predicted winners."

A decaying exploration schedule:

$$
\kappa_t = \kappa_0 \sqrt{\frac{\log(t+1)}{t+1}}
$$

A non-decaying schedule may be better if candidate generation distribution shifts heavily over time.

### 8.3 Expected Improvement

Let the current best validated score be:

$$
f^+ = \max_{z\in A_t}\hat{J}_{\mathrm{val}}(z)
$$

Improvement random variable:

$$
I(z)=\max(0,f(z)-f^+-\xi)
$$

Expected improvement:

$$
\mathrm{EI}(z)=\mathbb{E}[I(z)]
$$

If $f(z)\sim\mathcal{N}(\mu_t(z),\sigma_t^2(z))$, then:

$$
\mathrm{EI}(z)=(\mu_t(z)-f^+-\xi)\Phi(\gamma)+\sigma_t(z)\phi(\gamma)
$$

where:

$$
\gamma=\frac{\mu_t(z)-f^+-\xi}{\sigma_t(z)}
$$

and $\Phi$ and $\phi$ are the standard normal CDF and PDF.

EI is useful when BESO mainly wants to beat the current best skill, but it may over-focus on short-term improvement.

### 8.4 Probability of Improvement

$$
\mathrm{PI}(z)=\Pr(f(z)>f^+ + \xi)
$$

Assuming Gaussian posterior:

$$
\mathrm{PI}(z)=\Phi\left(\frac{\mu_t(z)-f^+-\xi}{\sigma_t(z)}\right)
$$

PI is simple but often too greedy.

### 8.5 Thompson sampling

Sample a plausible function from the posterior:

$$
\tilde{f}_t \sim p(f\mid H_t)
$$

Select:

$$
z_{t+1}\in\arg\max_{z\in\mathcal{C}_t}\tilde{f}_t(z)
$$

For ensemble surrogates, a practical approximation is:

1. sample one model $g_m$ from the ensemble,
2. choose the best candidate according to $g_m$.

Thompson sampling is useful in batched settings because it naturally diversifies choices.

### 8.6 Diversity-aware acquisition

Let distance from archive be:

$$
d(z,A_t)=\min_{z'\in A_t}D(\varphi(z),\varphi(z'))
$$

where $D$ can be cosine distance, edit distance, section-level distance, or embedding distance.

Then:

$$
a_{\mathrm{div}}(z)=\mu_t(z)+\kappa_t\sigma_t(z)+\lambda d(z,A_t)
$$

This rewards candidates that are both promising and not redundant.

### 8.7 Cost-aware acquisition

Let predicted runtime cost be:

$$
\widehat{c}_t(z)
$$

Then:

$$
a_{\mathrm{cost}}(z)=\mu_t(z)+\kappa_t\sigma_t(z)-\alpha\widehat{c}_t(z)
$$

A richer form:

$$
a(z)=\mu_t(z)+\kappa_t\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{\mathrm{Cost}}(z)-\gamma\widehat{\mathrm{InvalidRisk}}(z)
$$

This is a strong default for BESO:

$$
\boxed{
a_{\mathrm{BESO}}(z)=\mu_t(z)+\kappa_t\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)-\gamma\widehat{q}_{\mathrm{invalid}}(z)
}
$$

---

## 9. Acceptance, rejection, and validation gates

### 9.1 Parent-relative acceptance

Let $p(z)$ be the parent of candidate $z$.

A simple acceptance rule:

$$
\mathrm{Accept}(z)=1
\iff
\hat{J}_{D_{\mathrm{val}}}(z) > \hat{J}_{D_{\mathrm{val}}}(p(z)) + \delta
$$

where $\delta$ is a minimum improvement threshold.

### 9.2 Bootstrap acceptance

Because validation estimates are noisy, use bootstrap resampling.

For candidate $z$ and parent $p$, compute per-example differences:

$$
d_i = r_i(z)-r_i(p)
$$

Mean difference:

$$
\bar{d}=\frac{1}{|D_{\mathrm{val}}|}\sum_{i\in D_{\mathrm{val}}}d_i
$$

Bootstrap confidence interval:

$$
\mathrm{CI}_{1-\alpha}(\bar{d})=[L,U]
$$

Accept if:

$$
L > 0
$$

or, less strictly:

$$
\bar{d}>\delta \quad \text{and} \quad L > -\epsilon
$$

This avoids accepting edits that look good only due to minibatch luck.

### 9.3 Risk-constrained acceptance

A candidate must also satisfy constraints:

$$
\mathrm{Accept}(z)=1
$$

only if:

$$
\hat{J}_{\mathrm{val}}(z) \ge \hat{J}_{\mathrm{val}}(p(z))+\delta
$$

$$
\mathrm{InvalidRate}(z) \le \rho_{\max}
$$

$$
\mathrm{Cost}(z) \le C_{\max}
$$

$$
\mathrm{Tokens}(z) \le T_{\max}
$$

$$
\mathrm{SchemaValid}(z)=1
$$

$$
\mathrm{InvariantValid}(z)=1
$$

This prevents the optimizer from improving one score while breaking output format, cost, or safety invariants.

### 9.4 Rejected edits as negative evidence

Rejected edits are not useless. They provide information about harmful regions of the edit space.

Let $R_t$ be the rejected-edit buffer:

$$
R_t=\{(e_j,z_j,\mathrm{reason}_j)\}_{j=1}^{r_t}
$$

Reflection should condition on $R_t$:

$$
Q_{\psi}(e\mid z,\mathcal{R},F,A_t,H_t,R_t)
$$

The surrogate can also use similarity to rejected candidates:

$$
\varphi_{\mathrm{reject}}(z)=\min_{z'\in R_t}D(\varphi(z),\varphi(z'))
$$

or a penalty:

$$
\mathrm{RejectPenalty}(z)=\exp\left(-\min_{z'\in R_t}\frac{D(\varphi(z),\varphi(z'))^2}{2\ell_R^2}\right)
$$

Then acquisition becomes:

$$
a(z)=\mu_t(z)+\kappa_t\sigma_t(z)+\lambda d(z,A_t)-\eta\mathrm{RejectPenalty}(z)
$$

---

## 10. Pareto archive mathematics

### 10.1 Instance-level specialization

Let candidate $z_k$ have score on example $i$:

$$
S[k,i]=r_i(z_k)
$$

For each example:

$$
s_i^*=\max_k S[k,i]
$$

Winner set:

$$
W_i=\{z_k:S[k,i]=s_i^*\}
$$

Candidate win count:

$$
w(z_k)=\sum_{i=1}^{n}\mathbb{1}[z_k\in W_i]
$$

Selection probability:

$$
P_{\mathrm{win}}(z_k)=\frac{w(z_k)+\epsilon}{\sum_j(w(z_j)+\epsilon)}
$$

The small $\epsilon$ prevents zero probability for candidates that are good but not exact winners.

### 10.2 Pareto dominance over objective vectors

For multi-objective evaluation, define:

$$
F(z)=\left(F_1(z),F_2(z),\dots,F_K(z)\right)
$$

Example:

$$
F(z)=\left(J_{\mathrm{acc}}(z),J_{\mathrm{format}}(z),-\mathrm{Cost}(z),-\mathrm{Latency}(z),-\mathrm{InvalidRate}(z)\right)
$$

Candidate $z_a$ Pareto-dominates $z_b$ if:

$$
z_a \succ z_b
$$

when:

$$
F_k(z_a)\ge F_k(z_b) \quad \forall k
$$

and:

$$
\exists k: F_k(z_a)>F_k(z_b)
$$

The Pareto front is:

$$
\mathcal{P}_t=\{z\in A_t:\nexists z'\in A_t \text{ such that } z'\succ z\}
$$

This preserves candidates that represent different trade-offs: high accuracy, low cost, strong format validity, fast latency, or robustness on hard examples.

### 10.3 Archive pruning as constrained subset selection

If the archive is too large, choose a subset:

$$
A_t' \subset A_t
$$

with:

$$
|A_t'|\le M_A
$$

One objective:

$$
A_t' \in \arg\max_{A\subseteq A_t, |A|\le M_A}
\left[
\sum_{z\in A}\hat{J}_{\mathrm{val}}(z)
+ \lambda_1 \mathrm{Coverage}(A)
+ \lambda_2 \mathrm{Diversity}(A)
+ \lambda_3 \mathrm{ParetoValue}(A)
\right]
$$

Diversity can be:

$$
\mathrm{Diversity}(A)=\frac{1}{|A|(|A|-1)}\sum_{z\ne z'\in A}D(\varphi(z),\varphi(z'))
$$

Coverage can be instance-level:

$$
\mathrm{Coverage}(A)=\sum_{i=1}^{n}\max_{z\in A}S[z,i]
$$

This says: keep an archive that collectively covers many examples well, not just one average winner.

---

## 11. Multi-objective and constrained formulations

### 11.1 Scalarized objective

For a single scalar training objective:

$$
J_{\mathrm{scalar}}(z)=\sum_{k=1}^{K}w_kJ_k(z)
$$

with:

$$
\sum_{k=1}^{K}w_k=1, \quad w_k\ge0
$$

Example:

$$
J_{\mathrm{scalar}}(z)=
w_{\mathrm{acc}}J_{\mathrm{acc}}(z)
+w_{\mathrm{fmt}}J_{\mathrm{format}}(z)
-w_{\mathrm{cost}}J_{\mathrm{cost}}(z)
-w_{\mathrm{lat}}J_{\mathrm{latency}}(z)
$$

### 11.2 Constrained objective

A cleaner formulation is often constrained optimization:

$$
\max_{z\in\mathcal{Z}} J_{\mathrm{acc}}(z)
$$

subject to:

$$
J_{\mathrm{format}}(z)\ge \rho_{\mathrm{format}}
$$

$$
\mathrm{Cost}(z)\le C_{\max}
$$

$$
\mathrm{Latency}(z)\le L_{\max}
$$

$$
\mathrm{InvalidRate}(z)\le \rho_{\mathrm{invalid}}
$$

This is useful when format validity or safety is non-negotiable.

### 11.3 Lagrangian relaxation

The constrained problem can be relaxed:

$$
\mathcal{L}(z,\lambda)=J_{\mathrm{acc}}(z)
-\lambda_1\max(0,\rho_{\mathrm{format}}-J_{\mathrm{format}}(z))
-\lambda_2\max(0,\mathrm{Cost}(z)-C_{\max})
-\lambda_3\max(0,\mathrm{InvalidRate}(z)-\rho_{\mathrm{invalid}})
$$

Then the acquisition function can use predicted Lagrangian utility:

$$
a(z)=\mathbb{E}[\mathcal{L}(z,\lambda)\mid H_t]+\kappa\sqrt{\mathrm{Var}[\mathcal{L}(z,\lambda)\mid H_t]}
$$

---

## 12. Budget accounting and sample efficiency

### 12.1 Rollout cost

One rollout is one execution of the system on one task example.

If candidate $z_t$ is evaluated on minibatch $B_t$:

$$
\mathrm{Rollouts}(z_t,B_t)=|B_t|
$$

Total rollout budget:

$$
\sum_{t=1}^{T}|B_t|\le B
$$

If repeated evaluation is used for top candidates:

$$
\sum_{t=1}^{T}\sum_{k=1}^{K_t}|B_{t,k}|\le B
$$

### 12.2 Token and money budget

Rollouts may have variable cost. Define:

$$
c_t = c_{\mathrm{in}}\cdot \mathrm{InputTokens}_t + c_{\mathrm{out}}\cdot \mathrm{OutputTokens}_t + c_{\mathrm{tool}}\cdot \mathrm{ToolCalls}_t
$$

Total cost constraint:

$$
\sum_{t=1}^{T}c_t\le C_{\mathrm{total}}
$$

BESO can therefore optimize under either rollout budget, token budget, monetary budget, or wall-clock budget.

### 12.3 Optimization curve

Let best validation score after $b$ rollouts be:

$$
V(b)=\max_{z\in A(b)}\hat{J}_{\mathrm{val}}(z)
$$

where $A(b)$ is the archive after spending $b$ rollouts.

Sample efficiency can be measured by area under the optimization curve:

$$
\mathrm{AUC}_{B}=\frac{1}{B}\int_{0}^{B}V(b)\,db
$$

In discrete form:

$$
\mathrm{AUC}_{B}\approx\frac{1}{B}\sum_{t=1}^{T}V(b_t)(b_t-b_{t-1})
$$

BESO's central empirical claim should be about this curve, not only the final score.

### 12.4 Rollouts-to-threshold

Given a target score $\gamma$:

$$
\tau_{\gamma}=\min\{b:V(b)\ge \gamma\}
$$

If BESO is sample-efficient, it should have lower $\tau_{\gamma}$ than baselines.

---

## 13. Generalization and transfer

### 13.1 Train-test gap

Optimization score:

$$
\hat{J}_{\mathrm{opt}}(z)
$$

Final test score:

$$
\hat{J}_{\mathrm{test}}(z)
$$

Generalization gap:

$$
G(z)=\hat{J}_{\mathrm{opt}}(z)-\hat{J}_{\mathrm{test}}(z)
$$

Large positive $G(z)$ indicates overfitting.

### 13.2 Skill specificity penalty

Because skills are text, overfitting often appears as overly specific rules. Define a specificity score:

$$
\mathrm{Spec}(z) \in [0,1]
$$

estimated by structural or LLM-labeled features.

A regularized objective:

$$
J_{\mathrm{reg}}(z)=J(z)-\lambda_{\mathrm{spec}}\mathrm{Spec}(z)-\lambda_{\mathrm{len}}\mathrm{Length}(z)
$$

This encourages general rules rather than benchmark memorization.

### 13.3 Transfer across models or tasks

Let $\mathcal{T}_a$ be the source task distribution and $\mathcal{T}_b$ be the target distribution.

Source-optimized skill:

$$
z_a^*\in\arg\max_z J_{\mathcal{T}_a}(z)
$$

Transfer utility:

$$
\mathrm{Transfer}(z_a^*;\mathcal{T}_b)=J_{\mathcal{T}_b}(z_a^*)-J_{\mathcal{T}_b}(z_0)
$$

For cross-model transfer, compare:

$$
\mathrm{TransferModel}(z^*;\Theta_a,\Theta_b)=J_{\Theta_b}(z^*)-J_{\Theta_b}(z_0)
$$

The research question is whether skill artifacts encode reusable behavior or only model-specific prompt hacks.

---

## 14. Relation to GEPA-style prompt evolution

GEPA-style reflective prompt evolution can be abstracted as:

$$
p' = R(p,\tau,s,f)
$$

where $p$ is a prompt and $R$ is a reflection-based prompt updater.

BESO shifts the object from prompt $p$ to skill artifact $z$:

$$
z' = R_z(z,\tau,s,f)
$$

and adds Bayesian selection:

$$
z_{t+1}\in\arg\max_{z\in\mathcal{C}_t}a_t(z)
$$

So the key distinction is:

$$
\text{GEPA-like: reflect } \rightarrow \text{ mutate prompt } \rightarrow \text{ evaluate}
$$

$$
\text{BESO: reflect } \rightarrow \text{ generate skill edits } \rightarrow \text{ predict utility/uncertainty } \rightarrow \text{ selectively evaluate}
$$

BESO's proposed advantage is not that it can generate better edits automatically. The reflection model may generate the same candidate edits as a GEPA-like system. The advantage is that BESO spends the rollout budget more selectively.

---

## 15. Relation to bandits

If each edit type is treated as an arm:

$$
a \in \mathcal{A}=\{\mathrm{add\_rule},\mathrm{replace\_rule},\mathrm{compress},\dots\}
$$

then BESO could be reduced to a contextual bandit:

$$
\Pr(a_t\mid \mathrm{context}_t)
$$

with reward:

$$
r_t=\hat{J}(z_{t+1})-\hat{J}(z_t)
$$

But BESO is richer than a simple bandit because:

- actions are not only edit types; they are concrete semantic edits,
- the candidate space changes every iteration,
- candidates have text embeddings and lineage,
- evaluation can be multi-objective,
- archive diversity matters.

A contextual bandit baseline is still useful:

$$
a_t = \arg\max_a \mathrm{UCB}_t(a,\mathrm{context}_t)
$$

But BESO should outperform it if candidate-level semantic features matter.

---

## 16. Full BESO objective as nested optimization

BESO can be written as a nested optimization process.

The outer goal:

$$
z^* = \arg\max_{z\in\mathcal{Z}}J(z)
$$

The optimizer cannot search $\mathcal{Z}$ directly. At each iteration it constructs a local candidate set:

$$
\mathcal{C}_t = \mathcal{G}_{\psi}(A_t,H_t,R_t)
$$

where $\mathcal{G}_{\psi}$ is reflection-guided candidate generation.

Then it selects candidates by acquisition:

$$
S_t = \operatorname{TopK}_{z\in\mathcal{C}_t} a_t(z)
$$

Then it evaluates:

$$
\tilde{y}_{t,z}=\hat{J}_{B_t}(z)+\epsilon_{t,z}, \quad z\in S_t
$$

Then it updates:

$$
H_{t+1}=H_t\cup\{(z,B_t,\tilde{y}_{t,z},\mathcal{R}_{t,z},c_{t,z}):z\in S_t\}
$$

$$
A_{t+1}=\operatorname{ArchiveUpdate}(A_t,S_t,H_{t+1})
$$

$$
p(f\mid H_{t+1})=\operatorname{SurrogateUpdate}(p(f\mid H_t),H_{t+1})
$$

The final output is:

$$
z_{\mathrm{final}}=\arg\max_{z\in A_T}\hat{J}_{D_{\mathrm{val}}}(z)
$$

and the final reported score is:

$$
\hat{J}_{D_{\mathrm{test}}}(z_{\mathrm{final}})
$$

---

## 17. Algorithm in mathematical pseudocode

Input:

$$
D, z_0, B, M, K, \Theta_{\mathrm{frozen}}, C, \mu
$$

Initialize:

$$
A_0=\{z_0\}, \quad H_0=\emptyset, \quad R_0=\emptyset
$$

Evaluate seed skill:

$$
\tilde{y}_0,\mathcal{R}_0=\operatorname{Eval}(z_0,D_{\mathrm{val}})
$$

$$
H_0\leftarrow H_0\cup\{(z_0,D_{\mathrm{val}},\tilde{y}_0,\mathcal{R}_0)
\}
$$

For $t=0,1,\dots,T-1$ while budget remains:

1. Select parents:

$$
P_t\sim \pi_{\mathrm{parent}}(\cdot\mid A_t,H_t)
$$

2. Generate candidate edits:

$$
\mathcal{C}_t=\bigcup_{z_p\in P_t}\{e(z_p):e\sim Q_{\psi}(\cdot\mid z_p,H_t,R_t),\nu(z_p,e)=1\}
$$

3. Featurize candidates:

$$
X_t=\{\varphi(z):z\in\mathcal{C}_t\}
$$

4. Fit/update surrogate:

$$
p_t(f)=p(f\mid H_t)
$$

5. Score candidates by acquisition:

$$
a_t(z)=\mu_t(z)+\kappa_t\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)
$$

6. Select batch:

$$
S_t=\operatorname{TopK}_{z\in\mathcal{C}_t}(a_t(z),K)
$$

7. Evaluate selected candidates:

$$
(\tilde{y}_z,\mathcal{R}_z,c_z)=\operatorname{Eval}(z,B_t), \quad z\in S_t
$$

8. Validate candidates:

$$
\mathrm{accept}_z=\operatorname{Gate}(z,p(z),D_{\mathrm{val}})
$$

9. Update archive:

$$
A_{t+1}=\operatorname{Prune}\left(A_t\cup\{z\in S_t:\mathrm{accept}_z=1\}\right)
$$

10. Update rejection buffer:

$$
R_{t+1}=R_t\cup\{z\in S_t:\mathrm{accept}_z=0\}
$$

Return:

$$
z_{\mathrm{final}}=\arg\max_{z\in A_T}\hat{J}_{D_{\mathrm{val}}}(z)
$$

Report:

$$
\hat{J}_{D_{\mathrm{test}}}(z_{\mathrm{final}})
$$

---

## 18. What exactly is Bayesian in BESO?

BESO is Bayesian in the candidate-selection layer, not necessarily in the text-generation layer.

The reflection model gives proposals:

$$
\mathcal{C}_t \sim Q_{\psi}(\cdot\mid H_t,A_t,R_t)
$$

The Bayesian layer asks:

$$
\text{Given previous evaluations, what do we believe about the utility of each candidate?}
$$

That belief is:

$$
p(f\mid H_t)
$$

The acquisition function converts belief into action:

$$
a_t(z)=\text{value of evaluating }z\text{ next}
$$

So BESO's Bayesian claim should be phrased carefully:

> BESO uses a probabilistic surrogate to guide which reflection-generated skill variants should be evaluated under a limited rollout budget.

It should not claim that the LLM reflection process itself is Bayesian unless that is explicitly modeled.

---

## 19. Key research hypotheses in mathematical form

### H1: Skill optimization beats prompt optimization

Let $\mathcal{Z}_{\mathrm{skill}}$ be skill artifacts and $\mathcal{Z}_{\mathrm{prompt}}$ be raw prompts.

Hypothesis:

$$
\mathbb{E}_{s}\left[\max_{z\in\mathcal{Z}_{\mathrm{skill}},\; c\le B} \hat{J}_{\mathrm{test}}(z)\right]
>
\mathbb{E}_{s}\left[\max_{p\in\mathcal{Z}_{\mathrm{prompt}},\; c\le B} \hat{J}_{\mathrm{test}}(p)\right]
$$

where expectation is over random seeds, datasets, and optimizer stochasticity.

### H2: Bayesian acquisition improves sample efficiency

Let $V_{\mathrm{BESO}}(b)$ be the best validation score after $b$ rollouts.

Let $V_{\mathrm{baseline}}(b)$ be the corresponding curve for random, greedy, or Pareto-only selection.

Hypothesis:

$$
\mathrm{AUC}_B(V_{\mathrm{BESO}})>
\mathrm{AUC}_B(V_{\mathrm{baseline}})
$$

and:

$$
\tau_{\gamma}^{\mathrm{BESO}}<\tau_{\gamma}^{\mathrm{baseline}}
$$

for meaningful thresholds $\gamma$.

### H3: Reflection improves candidate proposal quality

Let $\mathcal{C}_t^{\mathrm{reflect}}$ be reflection-generated candidates and $\mathcal{C}_t^{\mathrm{random}}$ be randomly mutated candidates.

Hypothesis:

$$
\mathbb{E}\left[\max_{z\in\mathcal{C}_t^{\mathrm{reflect}}}J(z)\right]
>
\mathbb{E}\left[\max_{z\in\mathcal{C}_t^{\mathrm{random}}}J(z)\right]
$$

### H4: Pareto archive improves robustness

Let $A_t^{\mathrm{pareto}}$ be archive with diversity / Pareto preservation and $A_t^{\mathrm{best}}$ be best-only archive.

Hypothesis:

$$
\hat{J}_{\mathrm{hard}}(z_{\mathrm{final}}^{\mathrm{pareto}})
>
\hat{J}_{\mathrm{hard}}(z_{\mathrm{final}}^{\mathrm{best}})
$$

where $D_{\mathrm{hard}}$ is a hard or rare-example subset.

### H5: Optimized skills transfer

For source task distribution $\mathcal{T}_a$ and target distribution $\mathcal{T}_b$:

$$
J_{\mathcal{T}_b}(z_a^*)-J_{\mathcal{T}_b}(z_0)>0
$$

This says the source-optimized skill remains useful on nearby target tasks.

---

## 20. Minimal v0 mathematical design

A clean first prototype should use the simplest version of the math.

### 20.1 Skill space

One skill document:

$$
z\in\mathcal{Z}_{\mathrm{skill}}
$$

Bounded single-section edits:

$$
|\Delta z|\le L_{\max}
$$

### 20.2 Objective

Single scalar metric:

$$
J(z)=\mathbb{E}_{(x,m)\sim\mathcal{T}}[\mu(\Phi(x;C(z),\Theta),m)]
$$

Empirical minibatch estimate:

$$
\hat{J}_{B_t}(z)=\frac{1}{|B_t|}\sum_{i\in B_t}r_i(z)
$$

### 20.3 Surrogate

Use ensemble mean and disagreement:

$$
\mu_t(z)=\frac{1}{M}\sum_{m=1}^{M}g_m(\varphi(z))
$$

$$
\sigma_t(z)=\sqrt{\frac{1}{M-1}\sum_{m=1}^{M}(g_m(\varphi(z))-\mu_t(z))^2}
$$

### 20.4 Acquisition

Use UCB with cost penalty:

$$
a_t(z)=\mu_t(z)+\kappa\sigma_t(z)-\alpha\mathrm{Length}(z)
$$

### 20.5 Acceptance

Use validation gate:

$$
\hat{J}_{\mathrm{val}}(z)>\hat{J}_{\mathrm{val}}(p(z))+\delta
$$

with schema validity:

$$
z\in\mathcal{Z}
$$

### 20.6 Final report

Report:

$$
\hat{J}_{\mathrm{test}}(z_{\mathrm{final}})
$$

plus optimization curve:

$$
V(b)=\max_{z\in A(b)}\hat{J}_{\mathrm{val}}(z)
$$

This is enough to test the core claim without overbuilding the system.

---

## 21. Where the mathematical novelty lives

The research should not be framed as "Bayesian optimization for prompts". That is too broad.

The sharper mathematical object is:

$$
\text{Bayesian optimization over reflection-generated neighborhoods of structured skill artifacts.}
$$

More explicitly:

1. BESO does not optimize all text directly.

$$
\mathcal{Z} \ne \Sigma^*
$$

It optimizes constrained skill artifacts:

$$
\mathcal{Z}=\{z:\mathrm{SchemaValid}(z),\mathrm{InvariantValid}(z),\mathrm{BudgetValid}(z)\}
$$

2. BESO does not require the Bayesian surrogate to generate language.

$$
\mathcal{C}_t = G_{\psi}(A_t,H_t,R_t)
$$

Reflection proposes. Bayesian acquisition selects.

3. BESO performs local Bayesian optimization over candidate neighborhoods:

$$
z_{t+1}\in\arg\max_{z\in\mathcal{C}_t}a_t(z)
$$

4. BESO maintains an evolutionary archive rather than a single incumbent:

$$
A_t = A_t^{\mathrm{best}}\cup A_t^{\mathrm{pareto}}\cup A_t^{\mathrm{diverse}}\cup A_t^{\mathrm{failed}}
$$

5. The system is evaluated by sample-efficiency curves, not just final performance:

$$
\mathrm{AUC}_B(V_{\mathrm{BESO}}) > \mathrm{AUC}_B(V_{\mathrm{baseline}})
$$

That combination is the real research contribution.

---

## 22. Final compact formulation

BESO can be summarized as the following constrained stochastic optimization problem:

$$
\begin{aligned}
&\underset{z\in\mathcal{Z}}{\text{maximize}}
&& J(z)=\mathbb{E}_{(x,m)\sim\mathcal{T}}\left[\mu\left(\Phi_{\Theta_{\mathrm{frozen}}}(x;C(z,x,q)),m\right)\right] \\
&\text{subject to}
&& \sum_{t=1}^{T}|B_t|\le B, \\
&&& \mathrm{SchemaValid}(z)=1, \\
&&& \mathrm{InvariantValid}(z)=1, \\
&&& \mathrm{Tokens}(z)\le T_{\max}, \\
&&& \mathrm{Cost}(z)\le C_{\max}. \\
\end{aligned}
$$

Because $J$ is unknown, noisy, expensive, and defined over structured text, BESO uses the iterative approximation:

$$
\mathcal{C}_t=G_{\psi}(A_t,H_t,R_t)
$$

$$
p_t(f)=p(f\mid H_t)
$$

$$
z_{t+1}=\arg\max_{z\in\mathcal{C}_t}\left[\mu_t(z)+\kappa_t\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)\right]
$$

$$
H_{t+1}=H_t\cup\operatorname{Eval}(z_{t+1})
$$

$$
A_{t+1}=\operatorname{ArchiveUpdate}(A_t,z_{t+1},H_{t+1})
$$

Final selection:

$$
z_{\mathrm{final}}=\arg\max_{z\in A_T}\hat{J}_{D_{\mathrm{val}}}(z)
$$

Final report:

$$
\hat{J}_{D_{\mathrm{test}}}(z_{\mathrm{final}})
$$

In plain language:

> BESO treats a skill document as trainable external state. Reflection proposes possible semantic edits. A Bayesian surrogate estimates which edits are promising or informative. An acquisition function spends the rollout budget. A Pareto-diverse archive keeps useful variants alive. The final skill is selected by validation and judged on untouched test data.
