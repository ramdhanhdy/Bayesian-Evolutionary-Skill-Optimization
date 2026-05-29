"""The BESO optimization loop and budget control.

Orchestrates parents -> reflection pool -> hard filter -> featurize -> surrogate
-> acquisition -> submodular batch -> evaluate -> gate -> archive update, under a
rollout budget. Hosts the ``RegimeDetector`` that auto-disables the surrogate
when it is not yet predictive (cold start / negligible candidate variance).

Planned modules: ``loop.py``, ``accept_reject.py``, ``budget.py``, ``regime.py``.
"""
