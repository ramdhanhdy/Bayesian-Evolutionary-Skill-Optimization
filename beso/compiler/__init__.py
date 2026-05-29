"""Skill compiler C: skill artifact -> runtime prompt material.

Implements compiler modes C_full / C_section / C_distill (Breakdown S1.5).
For the v0 fork this is largely identity (the markdown document is injected
directly by SkillOpt's harness); section-selection compilation is deferred to v1.

Planned modules: ``skill_compiler.py``, ``section_selector.py``.
"""
