"""Acquisition and batch selection (``AcquisitionFunction`` + ``BatchSelector``).

Implements the pool-normalized a_BESO:

    a(z) = mu~ + kappa*sigma~ + lambda*d~(z,A) - alpha*c~(z) - gamma*q~_invalid(z)

with every term normalized over the current pool C_t (dimensionless weights),
plus submodular max-min / DPP batch selection that updates the reference set
with already-selected members to avoid intra-batch near-duplicates.

Planned modules: ``base.py``, ``composite.py``, ``ucb.py``,
``expected_improvement.py``, ``thompson.py``, ``diversity.py``.
"""
