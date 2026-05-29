"""BESO: Bayesian Evolutionary Skill Optimization.

Optimizes a structured natural-language skill artifact ``z`` for a frozen LLM
system ``Phi`` by combining trajectory-grounded reflective mutation, a Bayesian
surrogate over candidate utility, an acquisition layer for budgeted rollout
allocation, and an evolutionary archive.

Central objective (Mathematical Breakdown S0):

    z* in argmax_{z in Z}  J(z)
        = argmax_{z in Z}  E_{(x,m)~T}[ mu(Phi(x; C(z), Theta_frozen), m) ]
"""

__version__ = "0.0.1"
