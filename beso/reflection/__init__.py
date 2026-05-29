"""Reflection: parallel candidate-pool generation (the ``ReflectionProposer``).

Replaces SkillOpt's single-best-patch proposal with a dense pool of 24-50 bounded
edits per optimizer call: e_{t,j} ~ Q_psi(e | z_p, traces, feedback, archive,
rejected). Output edits are SkillOpt-compatible ({op, content, target}).

Planned modules: ``proposer.py``, ``prompts.py``, ``validators.py``.
"""
