---
type: research-note
tags: [beso, gepa, skillopt, bayesian-optimization, evolutionary-search, skill-optimization, llm-agents]
date: 2026-05-28
source:
  - [Technical specification](technical-specification.md)
  - [Mathematical breakdown](mathematical-breakdown.md)
external_sources:
  - https://gepa-ai.github.io/gepa/
  - https://arxiv.org/abs/2507.19457
  - https://microsoft.github.io/SkillOpt/
  - https://arxiv.org/html/2605.23904v2
status: draft
---

# GEPA → SkillOpt → BESO: mathematical lineage and contribution map

## 0. Executive summary

This note formalizes the line from GEPA to SkillOpt to BESO.

The short version:

```text
GEPA:
  Optimize textual parameters, usually prompts, using reflective mutation and Pareto-aware evolutionary search.

SkillOpt:
  Move the optimization target from raw prompts to compact reusable skill documents. Add controlled bounded edits, validation gates, rejected-edit feedback, and slow/meta updates.

BESO:
  Keep SkillOpt's skill-level abstraction, but add Bayesian experiment planning: learn a probabilistic surrogate over candidate skill edits and use acquisition functions to decide which candidates deserve expensive rollout evaluation.
```

Mathematically, all three methods optimize external text for a frozen LLM system:

$$
\max_{z \in \mathcal{Z}} J(z)
= \max_{z \in \mathcal{Z} }
\mathbb{E}_{(x,m)\sim\mathcal{T}}
\left[\mu\left(\Phi(x; z, \Theta_{\mathrm{frozen}}),m\right)\right]
$$

The difference is what $z$ means and how the next $z$ is chosen.

| Method   | Optimized object $z$                          | Candidate generation                                      | Candidate selection                            | Main contribution                                                                    |
| -------- | --------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------ |
| GEPA     | Prompt / textual parameter / system component | LLM reflection over trajectories + mutation / merge       | Pareto-aware evolutionary selection            | Language reflection as a rich learning signal; Pareto search over textual candidates |
| SkillOpt | Skill document as external trainable state    | Optimizer model proposes bounded add/delete/replace edits | Held-out validation gate; strict accept/reject | Train reusable skills like model parameters, but in text space                       |
| BESO     | Structured skill artifact                     | Reflection-generated bounded skill edits                  | Bayesian surrogate + acquisition + archive     | Sample-efficient experiment planning for skill evolution under low rollout budgets   |

BESO is not mainly a new reflection mechanism. It is a new selection and budget-allocation layer over SkillOpt-style candidate skill edits.

The clean contribution claim is:

> BESO extends GEPA-style reflective text evolution and SkillOpt-style skill-document training by adding Bayesian candidate selection: a probabilistic surrogate predicts candidate utility and uncertainty, then an acquisition function chooses which skill variants to evaluate under a limited rollout budget.

---

## 1. Shared mathematical foundation

### 1.1 Frozen LLM system

Let the frozen LLM system be:

$$
\Phi_{\Theta}: \mathcal{X}\times\mathcal{A}\to\mathcal{Y}\times\mathcal{R}
$$

where:

- $\mathcal{X}$ is the task-input space.
- $\mathcal{A}$ is the external artifact space: prompt, textual parameter, or skill.
- $\mathcal{Y}$ is the final-output space.
- $\mathcal{R}$ is the trajectory / trace space.
- $\Theta=\Theta_{\mathrm{frozen}}$ are fixed model weights.

For task input $x$ and artifact $z$:

$$
(y,\tau)=\Phi_{\Theta_{\mathrm{frozen}}}(x;z)
$$

where:

- $y$ is the final answer or action result.
- $\tau$ is the observable execution trace: prompts, messages, tool calls, tool results, verifier output, error messages, intermediate outputs, cost, and latency.

Important: $\tau$ should be treated as observable trace, not hidden chain-of-thought.

### 1.2 Task distribution and evaluator

Let task instances be:

$$
\xi=(x,m)\sim\mathcal{T}
$$

where:

- $x$ is the task input.
- $m$ is metadata, expected answer, rubric, verifier, test suite, or scoring reference.
- $\mathcal{T}$ is the task distribution.

The evaluator is:

$$
\mu: \mathcal{Y}\times\mathcal{R}\times\mathcal{M}\to\mathbb{R}
$$

A single-task score is:

$$
r(z; x,m)=\mu(y,\tau,m)
$$

with:

$$
(y,\tau)=\Phi_{\Theta_{\mathrm{frozen}}}(x;z)
$$

Most papers use scores in $[0,1]$, but the framework can support arbitrary bounded reward.

### 1.3 Common optimization objective

The true objective is:

$$
J(z)=\mathbb{E}_{(x,m)\sim\mathcal{T}}[r(z;x,m)]
$$

The optimizer wants:

$$
z^*\in\arg\max_{z\in\mathcal{Z}}J(z)
$$

Because $\mathcal{T}$ is unknown, use a dataset:

$$
D=\{(x_i,m_i)\}_{i=1}^{n}
$$

Empirical score:

$$
\hat{J}_D(z)=\frac{1}{n}\sum_{i=1}^{n}r(z;x_i,m_i)
$$

In practice, all three methods face the same black-box setting:

$$
z \mapsto \hat{J}_D(z)
$$

is expensive, noisy, nondifferentiable, and defined over text.

---

## 2. GEPA: reflective prompt/text evolution

### 2.1 Optimization target

In GEPA, the optimized artifact is a textual candidate:

$$
p\in\mathcal{P}
$$

The candidate may be a prompt, a module instruction, a DSPy prompt, a textual system component, or in broader GEPA usage, another textual artifact.

For the canonical prompt-optimization case:

$$
y=\Phi(x;p,\Theta_{\mathrm{frozen}})
$$

Objective:

$$
p^*\in\arg\max_{p\in\mathcal{P}}
\mathbb{E}_{(x,m)\sim\mathcal{T}}
\left[\mu\left(\Phi(x;p,\Theta_{\mathrm{frozen}}),m\right)\right]
$$

### 2.2 Trajectory-conditioned reflection

GEPA's key move is that it does not reduce the rollout to only a scalar reward. It uses trajectory and side information.

A rollout produces:

$$
(y_i,\tau_i,s_i,f_i)
$$

where:

- $y_i$ is output,
- $\tau_i$ is trace,
- $s_i=\mu(y_i,\tau_i,m_i)$ is score,
- $f_i$ is actionable side information: errors, judge feedback, profiling data, test failures, reasoning summaries, etc.

A reflection operator proposes an improved candidate:

$$
p' = R_{\psi}(p, \{(x_i,m_i,y_i,\tau_i,s_i,f_i)\}_{i\in B})
$$

where $B$ is a minibatch and $R_{\psi}$ is usually an optimizer/reflection LLM.

The conceptual analogy is:

```text
scalar reward: tells the optimizer that the candidate failed
trajectory + side information: helps the optimizer infer why it failed
reflection: turns the why into a text update
```

### 2.3 Evolutionary population

GEPA maintains a population of candidates:

$$
P_t=\{p_{t,1},p_{t,2},\dots,p_{t,K}\}
$$

Each candidate is evaluated on examples, producing a score matrix:

$$
S_t[k,i]=r(p_{t,k};x_i,m_i)
$$

The population is updated through reflective mutation and sometimes merge / recombination:

$$
p_{\mathrm{mut}}=R_{\psi}(p_{\mathrm{parent}},\mathcal{D}_{\mathrm{trace}})
$$

$$
p_{\mathrm{merge}}=M_{\psi}(p_a,p_b,\mathcal{D}_{\mathrm{trace}})
$$

### 2.4 Pareto-aware candidate preservation

A central GEPA idea is that the best average candidate is not always the best search parent.

For each example $i$, define the best score achieved by any candidate:

$$
s_i^*=\max_k S_t[k,i]
$$

Winner set:

$$
W_i=\{p_{t,k}:S_t[k,i]=s_i^*\}
$$

Candidate win count:

$$
w(p_{t,k})=\sum_i\mathbb{1}[p_{t,k}\in W_i]
$$

Candidates with nonzero win counts may represent useful specialists. Selection can favor them even if they are not the global average winner.

A simplified parent-selection probability is:

$$
\Pr(p_{t,k})=\frac{w(p_{t,k})+\epsilon}{\sum_j(w(p_{t,j})+\epsilon)}
$$

This preserves diversity across task instances.

### 2.5 GEPA's core mathematical identity

GEPA can be summarized as:

$$
\boxed{
\text{GEPA} = \text{ReflectiveMutation}(\tau,f) + \text{ParetoEvolution}(P_t)
}
$$

More explicitly:

$$
\mathcal{C}_t=\{R_{\psi}(p,\mathcal{D}_p):p\sim\pi_{\mathrm{pareto}}(P_t)\}
$$

$$
P_{t+1}=\operatorname{ParetoUpdate}(P_t\cup\mathcal{C}_t)
$$

where candidates are evaluated by rollout calls.

### 2.6 GEPA limitation that motivates the next abstraction

GEPA is strong because it learns from rich traces. But prompt-level candidates can be brittle:

- A prompt mixes role, procedure, examples, output format, safety rules, tool policy, and failure handling into one text block.
- Mutations may improve one part while accidentally damaging another.
- Learned behavior may be less reusable if it is embedded in one prompt string.
- Candidate selection is mainly evolutionary / Pareto, not explicitly probabilistic experiment planning.

SkillOpt addresses the first three issues by changing the artifact. BESO addresses the fourth by changing the candidate-selection layer.

---

## 3. SkillOpt: train skills as external text-state

### 3.1 Shift in optimization target

SkillOpt moves from prompt optimization to skill-document optimization.

Instead of:

$$
p\in\mathcal{P}
$$

SkillOpt optimizes:

$$
s\in\mathcal{S}
$$

where $s$ is a compact natural-language skill document, often deployed as `best_skill.md`.

A skill can encode:

- procedure,
- tool-use policy,
- evidence-gathering rules,
- verification routines,
- output-format constraints,
- known failure modes,
- recovery rules.

The frozen agent executes with skill $s$:

$$
(\tau(s),r(s))=h(M,x,s),\qquad r(s)\in[0,1]
$$

using SkillOpt paper notation:

- $M$ is the frozen target model,
- $h$ is the execution harness,
- $x$ is the task,
- $s$ is the skill,
- $\tau(s)$ is the trajectory,
- $r(s)$ is the score.

The expected objective is:

$$
s^*\in\arg\max_{s\in\mathcal{S}}J(s)
=\arg\max_{s\in\mathcal{S}}\mathbb{E}_{x\sim\mathcal{T}}[r(s;x)]
$$

### 3.2 Skill as external trainable state

SkillOpt's key abstraction is:

$$
\text{trainable state} = s
$$

while:

$$
\Theta_{\mathrm{target}} \text{ is fixed}
$$

The target model, backend, and harness do not change. Only the skill document changes.

This means the deployment object is not a new model and not an online optimizer. It is a static skill artifact:

$$
s_{\mathrm{deploy}}=\texttt{best\_skill.md}
$$

with no extra inference-time model calls.

### 3.3 Bounded edit operators

SkillOpt uses structured edits:

$$
e\in\mathcal{E}=\{\mathrm{add},\mathrm{delete},\mathrm{replace}\}
$$

An edit transforms a skill:

$$
s'=e(s)
$$

A bounded edit budget acts like a textual learning rate.

Let edit distance or edit size be:

$$
\Delta(s,s')
$$

SkillOpt constrains updates:

$$
\Delta(s,s')\le \eta_{\mathrm{text}}
$$

where $\eta_{\mathrm{text}}$ is the textual learning-rate budget.

This prevents the optimizer from performing destructive broad rewrites.

### 3.4 Reflection as text-space backward pass

SkillOpt's loop can be written:

1. Rollout with current skill:

$$
\{(\tau_i,r_i)\}_{i\in B}=\operatorname{Rollout}(M,h,s_t,B)
$$

2. Reflect over successes and failures:

$$
\mathcal{E}_t=R_{\psi}(s_t,\{(\tau_i,r_i)\}_{i\in B},R_t)
$$

where $R_t$ may include rejected edits or optimizer memory.

3. Merge/rank bounded edits into a candidate:

$$
s_t'=\operatorname{ApplyBounded}(s_t,\mathcal{E}_t,\eta_{\mathrm{text}})
$$

4. Gate on held-out validation:

$$
s_{t+1}=\begin{cases}
s_t' & \text{if } \hat{J}_{\mathrm{sel}}(s_t')>\hat{J}_{\mathrm{sel}}(s_t) \\
s_t & \text{otherwise}
\end{cases}
$$

where $D_{\mathrm{sel}}$ is the held-out selection / validation split.

This is the mathematical heart of SkillOpt.

### 3.5 Validation gate

SkillOpt is not just self-revision. It is propose-and-test.

Acceptance rule:

$$
\operatorname{Accept}(s_t')=1
\iff
\hat{J}_{D_{\mathrm{sel}}}(s_t')>\hat{J}_{D_{\mathrm{sel}}}(s_t)
$$

A stricter rule can include threshold $\delta$:

$$
\hat{J}_{D_{\mathrm{sel}}}(s_t')>\hat{J}_{D_{\mathrm{sel}}}(s_t)+\delta
$$

Rejected candidates are stored:

$$
\mathcal{R}_{t+1}=\mathcal{R}_t\cup\{(s_t',\mathrm{reason}) : \operatorname{Accept}(s_t')=0\}
$$

The rejected-edit buffer gives negative feedback to future reflection.

### 3.6 Slow/meta update

SkillOpt also uses longer-horizon optimizer-side memory, described as slow update or meta skill.

We can represent optimizer memory as:

$$
m_t\in\mathcal{M}_{\mathrm{opt}}
$$

Reflection becomes:

$$
\mathcal{E}_t=R_{\psi}(s_t,\mathcal{B}_t,\mathcal{R}_t,m_t)
$$

and memory updates epoch-wise:

$$
m_{e+1}=U_{\psi}(m_e,\mathcal{H}_e)
$$

where $\mathcal{H}_e$ is the history from epoch $e$.

The deployed skill remains compact; optimizer memory does not need to be included at inference time.

### 3.7 SkillOpt's core mathematical identity

SkillOpt can be summarized as:

$$
\boxed{
\text{SkillOpt} = \text{SkillArtifact}(s) + \text{BoundedTextEdits}(\eta_{\mathrm{text}}) + \text{ValidationGate} + \text{RejectedEditMemory}
}
$$

Algorithmically:

$$
s_t' = \operatorname{ApplyBounded}\left(s_t, R_{\psi}(s_t,\mathcal{B}_t,\mathcal{R}_t,m_t), \eta_{\mathrm{text}}\right)
$$

$$
s_{t+1}=\begin{cases}
s_t' & \hat{J}_{\mathrm{sel}}(s_t')>\hat{J}_{\mathrm{sel}}(s_t) \\
s_t & \text{otherwise}
\end{cases}
$$

### 3.8 What SkillOpt contributes over GEPA

SkillOpt inherits the GEPA insight:

$$
\text{trajectory feedback} \rightarrow \text{natural-language reflection} \rightarrow \text{text update}
$$

But it changes the abstraction level:

$$
\text{prompt candidate }p
\quad\Longrightarrow\quad
\text{skill document }s
$$

This matters because skill documents are:

- reusable across tasks,
- deployable as standalone artifacts,
- more modular than prompts,
- easier to edit with add/delete/replace operations,
- easier to gate by validation,
- often transferable across models and harnesses.

So the GEPA → SkillOpt link is:

```text
GEPA learns textual candidates by reflection.
SkillOpt applies that idea to skills as the trainable text-state and adds training discipline.
```

---

## 4. BESO: Bayesian Evolutionary Skill Optimization

### 4.1 BESO starts from SkillOpt's abstraction

BESO accepts SkillOpt's main thesis:

$$
\text{the optimized object should be a skill artifact, not just a prompt string.}
$$

Let BESO's skill artifact be:

$$
z\in\mathcal{Z}_{\mathrm{skill}}
$$

with structured sections:

$$
z=(z_{\mathrm{goal}},z_{\mathrm{scope}},z_{\mathrm{procedure}},z_{\mathrm{tool}},z_{\mathrm{verify}},z_{\mathrm{failures}},z_{\mathrm{output}},z_{\mathrm{examples}})
$$

A compiler converts the skill into runtime prompt material:

$$
p=C(z,x,q)
$$

The frozen system runs:

$$
(y,\tau)=\Phi_{\Theta_{\mathrm{frozen}}}(x;C(z,x,q))
$$

The expected objective is:

$$
z^*\in\arg\max_{z\in\mathcal{Z}_{\mathrm{skill}}}
\mathbb{E}_{(x,m)\sim\mathcal{T}}
\left[\mu\left(\Phi_{\Theta_{\mathrm{frozen}}}(x;C(z,x,q)),m\right)\right]
$$

### 4.2 The problem BESO targets

SkillOpt proposes bounded edits and uses validation to accept/reject them. But candidate evaluation is expensive.

If an optimizer proposes $M$ candidate skill edits per round and each candidate needs $b$ rollouts, the naive cost is:

$$
\mathrm{Cost}_{\mathrm{naive}}=M\cdot b
$$

Under small rollout budget $B$, many candidates cannot be tested.

BESO asks:

> Given a pool of reflection-generated skill edits, which candidates should we evaluate first?

This is a Bayesian experiment-planning problem.

### 4.3 Candidate pool generation

Like SkillOpt, BESO uses reflection to generate candidate edits.

For parent skill $z_p$:

$$
\mathcal{C}_t=\{z_{t,1},\dots,z_{t,M}\}
$$

where:

$$
z_{t,j}=e_{t,j}(z_p)
$$

and:

$$
e_{t,j}\sim Q_{\psi}(e\mid z_p,H_t,R_t,A_t)
$$

$Q_{\psi}$ is the reflection proposal distribution.

So far, this is close to SkillOpt.

### 4.4 BESO's extra layer: probabilistic surrogate

BESO adds a surrogate model over candidate performance:

$$
f(z)=J(z)
$$

The true $f(z)$ is unknown. BESO observes noisy evaluations:

$$
\tilde{y}_i=f(z_i)+\epsilon_i
$$

History:

$$
H_t=\{(z_i,\tilde{y}_i,\tau_i,c_i)\}_{i=1}^{t}
$$

Bayesian posterior:

$$
p(f\mid H_t)
$$

For any candidate $z$:

$$
\mu_t(z)=\mathbb{E}[f(z)\mid H_t]
$$

$$
\sigma_t^2(z)=\operatorname{Var}[f(z)\mid H_t]
$$

Interpretation:

- $\mu_t(z)$ estimates how good the candidate is likely to be.
- $\sigma_t(z)$ estimates how uncertain the optimizer is.

### 4.5 Candidate featurization

Because $z$ is text, BESO needs a feature map:

$$
\varphi:\mathcal{Z}_{\mathrm{skill}}\to\mathbb{R}^{d}
$$

A useful decomposition:

$$
\varphi(z)=
[\varphi_{\mathrm{text}}(z),
\varphi_{\mathrm{struct}}(z),
\varphi_{\mathrm{edit}}(z),
\varphi_{\mathrm{history}}(z),
\varphi_{\mathrm{semantic}}(z)]
$$

Features can include:

- embedding of changed text,
- target section,
- edit operation,
- edit size,
- parent score,
- token count,
- number of rules,
- similarity to accepted candidates,
- similarity to rejected candidates,
- LLM-labeled emphasis on verification, decomposition, tool use, caution, etc.

This is a key BESO design point: Bayesian optimization does not operate directly on raw text; it operates on representations of candidate skill artifacts.

### 4.6 Acquisition function

BESO chooses candidates by acquisition:

$$
z_{t+1}\in\arg\max_{z\in\mathcal{C}_t}a_t(z)
$$

A default Upper Confidence Bound acquisition:

$$
a_{\mathrm{UCB}}(z)=\mu_t(z)+\kappa\sigma_t(z)
$$

A BESO-specific practical acquisition:

$$
\boxed{
a_{\mathrm{BESO}}(z)
=\mu_t(z)+\kappa\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)-\gamma\widehat{q}_{\mathrm{invalid}}(z)
}
$$

where:

- $d(z,A_t)$ rewards diversity from the archive.
- $\widehat{c}(z)$ penalizes cost or token bloat.
- $\widehat{q}_{\mathrm{invalid}}(z)$ penalizes predicted invalid output or schema breakage.
- $\kappa$ controls exploration.

This is the mathematical heart of BESO.

### 4.7 Archive management

BESO should maintain an archive:

$$
A_t=A_t^{\mathrm{best}}\cup A_t^{\mathrm{pareto}}\cup A_t^{\mathrm{diverse}}\cup A_t^{\mathrm{failed}}
$$

This combines GEPA's Pareto preservation with SkillOpt's rejected-edit memory.

- $A_t^{\mathrm{best}}$: high validation score.
- $A_t^{\mathrm{pareto}}$: specialists on different examples or objectives.
- $A_t^{\mathrm{diverse}}$: semantically distinct skills.
- $A_t^{\mathrm{failed}}$: informative failures that teach the surrogate and reflection module what not to repeat.

### 4.8 BESO's update recurrence

At iteration $t$:

1. Select parent skills:

$$
P_t\sim\pi_{\mathrm{parent}}(A_t,H_t)
$$

2. Generate candidates through reflection:

$$
\mathcal{C}_t=G_{\psi}(P_t,H_t,R_t,A_t)
$$

3. Fit / update Bayesian surrogate:

$$
p_t(f)=p(f\mid H_t)
$$

4. Select candidates to evaluate:

$$
S_t=\operatorname{TopK}_{z\in\mathcal{C}_t}a_t(z)
$$

5. Evaluate selected candidates:

$$
\tilde{y}_{z}=\hat{J}_{B_t}(z)+\epsilon_z,\qquad z\in S_t
$$

6. Gate candidates:

$$
\operatorname{Accept}(z)=1
\iff
\hat{J}_{D_{\mathrm{val}}}(z)>\hat{J}_{D_{\mathrm{val}}}(p(z))+\delta
$$

7. Update history and archive:

$$
H_{t+1}=H_t\cup\{(z,\tilde{y}_z,\tau_z,c_z):z\in S_t\}
$$

$$
A_{t+1}=\operatorname{ArchiveUpdate}(A_t,S_t,H_{t+1})
$$

### 4.9 BESO's core mathematical identity

BESO can be summarized as:

$$
\boxed{
\text{BESO} = \text{SkillOpt-style bounded skill edits} + \text{Bayesian acquisition over candidate skill variants} + \text{evolutionary archive}
}
$$

Or:

$$
\boxed{
\text{BESO} = \text{GEPA reflection} + \text{SkillOpt skill abstraction} + \text{Bayesian experiment planning}
}
$$

---

## 5. The lineage: what is inherited and what is new

### 5.1 From GEPA to SkillOpt

GEPA establishes:

$$
\tau + f \rightarrow R_{\psi}(\cdot) \rightarrow \text{textual candidate improvement}
$$

SkillOpt inherits that idea but changes the state variable:

$$
p \in \mathcal{P}
\quad\longrightarrow\quad
s\in\mathcal{S}
$$

So the shift is:

```text
GEPA: text candidate is often a prompt or textual system parameter.
SkillOpt: text candidate is a deployable skill document.
```

SkillOpt adds:

$$
\Delta(s,s')\le\eta_{\mathrm{text}}
$$

and:

$$
s_{t+1}=s_t' \;\text{only if}\; \hat{J}_{\mathrm{sel}}(s_t')>\hat{J}_{\mathrm{sel}}(s_t)
$$

That gives skill learning the discipline of:

- minibatches,
- learning-rate-like bounded edits,
- validation gates,
- rejected update memory,
- train/deploy separation.

### 5.2 From SkillOpt to BESO

SkillOpt generates and tests candidate edits. BESO asks which candidate edits deserve testing.

SkillOpt selection is primarily validation-gated:

$$
s_t' \rightarrow \operatorname{Eval}(s_t') \rightarrow \operatorname{Accept/Reject}
$$

BESO inserts a surrogate before expensive evaluation:

$$
\mathcal{C}_t \rightarrow p(f\mid H_t) \rightarrow a_t(z) \rightarrow \operatorname{Eval}(\operatorname{TopK}(a_t))
$$

So the shift is:

```text
SkillOpt: propose bounded edit, evaluate, accept if validation improves.
BESO: propose many bounded edits, predict value/uncertainty, evaluate the most promising or informative ones, then validate/archive.
```

Mathematically:

SkillOpt:

$$
z_t'=G_{\psi}(z_t,H_t,R_t),\qquad z_{t+1}=\operatorname{Gate}(z_t',z_t)
$$

BESO:

$$
\mathcal{C}_t=G_{\psi}(z_t,H_t,R_t)
$$

$$
z_t'\in\arg\max_{z\in\mathcal{C}_t}\left[\mu_t(z)+\kappa\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)\right]
$$

$$
z_{t+1}=\operatorname{Gate}(z_t',z_t)
$$

BESO's novelty is between proposal and evaluation.

---

## 6. Contribution map by component

| Component | GEPA | SkillOpt | BESO contribution |
|---|---|---|---|
| Frozen target model | Yes | Yes | Same assumption |
| Optimized text artifact | Prompt / arbitrary textual parameter | Skill document | Structured skill artifact, possibly compiled into runtime prompts |
| Reflection over traces | Core mechanism | Core mechanism | Inherited; used for candidate generation |
| Actionable side information | Core GEPA idea | Rollout/verifier feedback | Inherited; becomes surrogate and edit features too |
| Pareto diversity | Core selection mechanism | Less central; focus on best skill via validation | Reintroduced as archive layer with diversity and specialization |
| Bounded edits | Not the main framing | Core textual learning rate | Inherited from SkillOpt |
| Validation gate | Present in many optimizer loops | Core accept/reject mechanism | Inherited; can add statistical / risk-aware gates |
| Rejected-edit buffer | Not central | Core stability mechanism | Inherited; also used as negative features for surrogate/acquisition |
| Slow/meta update | Not central | SkillOpt stability mechanism | Optional; can condition reflection/candidate generation |
| Bayesian surrogate | No | No / not central | Main BESO addition |
| Acquisition function | No | No / not central | Main BESO addition |
| Low-budget experiment planning | Indirect via sample-efficient reflection | Indirect via bounded/gated edits | Directly optimized objective |
| Candidate featurization | Not central | Not central | Required for surrogate modeling |
| Uncertainty-aware exploration | Pareto diversity gives implicit exploration | Validation gate gives stability | Explicit through $\sigma_t(z)$ and acquisition |

---

## 7. BESO's contribution precisely stated

### 7.1 Not the contribution

BESO should not claim:

> We invented reflective prompt/skill mutation.

GEPA and SkillOpt already use trajectory-grounded reflection.

BESO should not claim:

> We invented skill documents as trainable state.

SkillOpt already makes this central.

BESO should not claim:

> We invented validation-gated skill updates.

SkillOpt already does this.

### 7.2 Actual contribution

BESO's contribution is:

> A Bayesian experiment-planning layer for skill-document evolution.

More formally:

Given a reflection-generated candidate set:

$$
\mathcal{C}_t=\{z_{t,1},\dots,z_{t,M}\}
$$

and a limited budget allowing only $K\ll M$ evaluations, BESO chooses:

$$
S_t\in\arg\max_{S\subset\mathcal{C}_t, |S|=K}
\sum_{z\in S}a_t(z)-\eta\sum_{z\ne z'\in S}\operatorname{sim}(z,z')
$$

where:

$$
a_t(z)=\mu_t(z)+\kappa\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)-\gamma\widehat{q}_{\mathrm{invalid}}(z)
$$

This turns skill evolution into an active learning / Bayesian optimization problem.

### 7.3 The scientific hypothesis

BESO's core testable hypothesis:

$$
\mathrm{AUC}_B(V_{\mathrm{BESO}})>
\mathrm{AUC}_B(V_{\mathrm{SkillOpt}})
$$

under the same rollout budget $B$, where:

$$
V(b)=\max_{z\in A(b)}\hat{J}_{\mathrm{val}}(z)
$$

That is: BESO should reach good skills faster, not merely produce a good final skill after unlimited trials.

Another hypothesis:

$$
\tau_{\gamma}^{\mathrm{BESO}} < \tau_{\gamma}^{\mathrm{SkillOpt}}
$$

where:

$$
\tau_{\gamma}=\min\{b:V(b)\ge\gamma\}
$$

BESO wins if it needs fewer rollouts to reach the same quality threshold.

---

## 8. Mathematical comparison of update rules

### 8.1 GEPA update

Prompt / text candidate population:

$$
P_t=\{p_{t,k}\}_{k=1}^{K}
$$

Candidate generation:

$$
\mathcal{C}_t=\{R_{\psi}(p,\mathcal{D}_{p}) : p\sim\pi_{\mathrm{pareto}}(P_t)\}
$$

Population update:

$$
P_{t+1}=\operatorname{ParetoUpdate}(P_t\cup\mathcal{C}_t)
$$

### 8.2 SkillOpt update

Single incumbent skill:

$$
s_t
$$

Reflection-generated bounded edit:

$$
s_t'=\operatorname{ApplyBounded}(s_t,R_{\psi}(s_t,\mathcal{B}_t,R_t,m_t),\eta_{\mathrm{text}})
$$

Validation-gated update:

$$
s_{t+1}=\begin{cases}
s_t' & \hat{J}_{\mathrm{sel}}(s_t')>\hat{J}_{\mathrm{sel}}(s_t) \\
s_t & \text{otherwise}
\end{cases}
$$

Rejected buffer:

$$
R_{t+1}=R_t\cup\{s_t' : \hat{J}_{\mathrm{sel}}(s_t')\le\hat{J}_{\mathrm{sel}}(s_t)\}
$$

### 8.3 BESO update

Archive and history:

$$
A_t, H_t, R_t
$$

Candidate pool:

$$
\mathcal{C}_t=G_{\psi}(A_t,H_t,R_t)
$$

Surrogate posterior:

$$
p_t(f)=p(f\mid H_t)
$$

Acquisition ranking:

$$
z_{t,j}\mapsto a_t(z_{t,j})
$$

Selected evaluation batch:

$$
S_t=\operatorname{TopK}_{z\in\mathcal{C}_t}a_t(z)
$$

Evaluation:

$$
\tilde{y}_z=\hat{J}_{B_t}(z)+\epsilon_z
$$

Archive update:

$$
A_{t+1}=\operatorname{ArchiveUpdate}(A_t,S_t,H_{t+1})
$$

So BESO generalizes SkillOpt from:

$$
\text{one proposed candidate} \rightarrow \text{evaluate/gate}
$$

to:

$$
\text{many proposed candidates} \rightarrow \text{Bayesian rank/select} \rightarrow \text{evaluate/gate/archive}
$$

---

## 9. Where BESO can empirically win

BESO should be strongest when:

1. Rollouts are expensive.

$$
B \text{ is small relative to } |\mathcal{C}_t|
$$

2. Reflection can generate many plausible edits.

$$
|\mathcal{C}_t|\gg K
$$

3. Candidate quality varies significantly.

$$
\operatorname{Var}_{z\in\mathcal{C}_t}[J(z)] \text{ is large}
$$

4. Features predict some performance signal.

$$
I(\varphi(z);J(z))>0
$$

where $I$ denotes mutual information.

5. Uncertainty-aware exploration matters.

$$
\exists z: \mu_t(z) \text{ moderate but } \sigma_t(z) \text{ high and } J(z) \text{ high}
$$

If all reflection-generated edits are similar, cheap to evaluate, or impossible to predict, BESO's Bayesian layer may add overhead without gain.

---

## 10. Experimental design to prove BESO's contribution

### 10.1 Baselines

BESO should compare against:

1. No skill.
2. Human skill.
3. One-shot LLM skill.
4. GEPA-style prompt evolution.
5. GEPA-style skill evolution if available.
6. SkillOpt-style bounded skill editing.
7. Random candidate selection from reflection-generated edits.
8. Greedy reflection-ranked candidate selection.
9. Bandit over edit types.
10. BESO without Pareto archive.
11. BESO without uncertainty term.
12. BESO without structured features.

### 10.2 Main metric: optimization curve

The most important curve is:

$$
V(b)=\max_{z\in A(b)}\hat{J}_{\mathrm{val}}(z)
$$

where $b$ is rollout budget spent.

Report:

$$
\mathrm{AUC}_B=\frac{1}{B}\int_0^B V(b)\,db
$$

and:

$$
\hat{J}_{\mathrm{test}}(z_{\mathrm{final}})
$$

BESO's first claim should be sample efficiency:

```text
same final budget: BESO reaches good skills earlier
same low budget: BESO finds better skills
same target score: BESO uses fewer rollouts
```

### 10.3 Ablations specific to BESO

| Ablation | Mathematical question |
|---|---|
| No surrogate | Does $p(f\mid H_t)$ add value? |
| Mean-only acquisition | Is $\sigma_t(z)$ useful? |
| No diversity term | Does $d(z,A_t)$ prevent local collapse? |
| No cost penalty | Does $\widehat{c}(z)$ control bloat? |
| Embeddings only | Is semantic text similarity enough? |
| Structured features only | Are cheap edit/section features enough? |
| Absolute-score model only | Is modeling $J(z)$ enough? |
| Delta model only | Is modeling $J(z)-J(p(z))$ better? |
| No rejected buffer features | Do failed edits improve future selection? |

### 10.4 Hypotheses as equations

BESO vs SkillOpt under budget $B$:

$$
\mathbb{E}[\hat{J}_{\mathrm{test}}(z_{\mathrm{BESO}}(B))]
>
\mathbb{E}[\hat{J}_{\mathrm{test}}(z_{\mathrm{SkillOpt}}(B))]
$$

especially for small $B$.

Sample efficiency:

$$
\mathrm{AUC}_B(V_{\mathrm{BESO}})>
\mathrm{AUC}_B(V_{\mathrm{SkillOpt}})
$$

Surrogate usefulness:

$$
\operatorname{corr}(\mu_t(z),\hat{J}(z))>0
$$

Uncertainty calibration:

$$
\mathbb{E}[(\hat{J}(z)-\mu_t(z))^2\mid \sigma_t(z)=s] \approx s^2
$$

Acquisition usefulness:

$$
\mathbb{E}_{z\sim\pi_{\mathrm{BESO}}}[J(z)]>
\mathbb{E}_{z\sim\pi_{\mathrm{random}}}[J(z)]
$$

where:

$$
\pi_{\mathrm{BESO}}(z)\propto\exp(a_t(z))
$$

---

## 11. Final compact formulation

All three methods solve:

$$
\max_{z\in\mathcal{Z}}\mathbb{E}_{(x,m)\sim\mathcal{T}}
\left[\mu\left(\Phi_{\Theta_{\mathrm{frozen}}}(x;z),m\right)\right]
$$

but choose different $\mathcal{Z}$ and different update rules.

### GEPA

$$
\mathcal{Z}=\mathcal{P}\quad\text{or broad textual parameter space}
$$

$$
z_{t+1}=R_{\psi}(z_t,\tau_t,f_t)
$$

with Pareto population update:

$$
P_{t+1}=\operatorname{ParetoUpdate}(P_t\cup\mathcal{C}_t)
$$

### SkillOpt

$$
\mathcal{Z}=\mathcal{S}\quad\text{skill-document space}
$$

$$
z_t'=\operatorname{ApplyBounded}(z_t,R_{\psi}(z_t,\mathcal{B}_t,R_t,m_t),\eta_{\mathrm{text}})
$$

$$
z_{t+1}=z_t'\quad\text{iff}\quad \hat{J}_{\mathrm{sel}}(z_t')>\hat{J}_{\mathrm{sel}}(z_t)
$$

### BESO

$$
\mathcal{Z}=\mathcal{S}_{\mathrm{structured}}\quad\text{structured skill artifacts}
$$

Generate many reflection candidates:

$$
\mathcal{C}_t=G_{\psi}(A_t,H_t,R_t)
$$

Model unknown utility:

$$
p(f\mid H_t),\qquad \mu_t(z),\sigma_t(z)
$$

Select by acquisition:

$$
z_{t+1}^{\mathrm{eval}}=\arg\max_{z\in\mathcal{C}_t}
\left[\mu_t(z)+\kappa\sigma_t(z)+\lambda d(z,A_t)-\alpha\widehat{c}(z)-\gamma\widehat{q}_{\mathrm{invalid}}(z)\right]
$$

Then validate and archive:

$$
A_{t+1}=\operatorname{ArchiveUpdate}(A_t,z_{t+1}^{\mathrm{eval}},H_{t+1})
$$

---

## 12. Clean contribution statement for the project

A strong paper-style contribution statement for BESO would be:

> We study Bayesian Evolutionary Skill Optimization (BESO), a sample-efficient optimizer for frozen LLM agents that treats structured natural-language skills as trainable external state. BESO inherits trajectory-grounded reflective mutation from GEPA and skill-document training from SkillOpt, but adds a probabilistic experiment-planning layer: reflection generates bounded candidate edits, a surrogate model predicts their utility and uncertainty from semantic, structural, edit, and history features, and an acquisition function selects which variants to evaluate under a limited rollout budget. This directly targets the evaluation bottleneck in skill evolution and should be most beneficial in low-budget regimes where many plausible edits cannot all be tested.

The key novelty in one sentence:

> GEPA teaches text candidates from trajectories; SkillOpt trains skills as external state; BESO decides which skill edits are worth spending rollouts on.
