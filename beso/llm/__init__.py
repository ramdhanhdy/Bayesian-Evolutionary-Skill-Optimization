"""Provider-agnostic LLM adapter for the optimizer/reflection model.

Wraps target/optimizer model calls behind a single interface so reflection and
semantic labeling are decoupled from any specific provider. In the SkillOpt fork
this can delegate to ``skillopt.model`` (chat_optimizer / chat_target).

Planned modules: ``adapter.py``.
"""
