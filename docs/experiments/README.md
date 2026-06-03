# Experiments

This directory stores durable experiment protocol notes and compact result
summaries. Generated traces remain under `artifacts/` and should not be
committed unless a small artifact is explicitly needed for review.

## GSM8K Mini Conditions

`examples/run_gsm8k_mini_experiment.py` now reports three conditions on one
deterministic validation draw:

- `literal_no_skill`: no skill markdown is injected into the target prompt;
- `minimal_seed`: a neutral goal and output-format skill is injected unchanged;
- `BESO`: the optimizer evolves the minimal seed under its rollout budget.

Frozen-baseline rollouts are reported separately from the BESO budget. Each run
writes a `*_conditions.jsonl` comparison record beside its detailed BESO trace.

## Results

- [M1 toy validation](results/M1-toy-validation.md)
- [M2 GSM8K mini high-performance plateau](results/M2-gsm8k-mini-high-performance-plateau.md)
