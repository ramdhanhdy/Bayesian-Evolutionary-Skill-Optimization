from __future__ import annotations

import importlib
import sys
import types


def test_litellm_completion_uses_supported_gpt5_temperature(monkeypatch) -> None:
    experiment = importlib.import_module("examples.run_toy_experiment")
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    monkeypatch.setattr(experiment, "MODEL", "gpt-5-nano-2025-08-07")
    monkeypatch.setattr(experiment, "PROVIDER_HINT", "")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    output = experiment.litellm_completion(
        "prompt",
        system="system",
        max_tokens=10,
        temperature=0.0,
    )

    assert output == "ok"
    assert captured["model"] == "gpt-5-nano-2025-08-07"
    assert captured["temperature"] == 1.0


def test_litellm_completion_preserves_non_gpt5_temperature(monkeypatch) -> None:
    experiment = importlib.import_module("examples.run_toy_experiment")
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    monkeypatch.setattr(experiment, "MODEL", "gpt-4o-mini")
    monkeypatch.setattr(experiment, "PROVIDER_HINT", "")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    output = experiment.litellm_completion(
        "prompt",
        system="system",
        max_tokens=10,
        temperature=0.2,
    )

    assert output == "ok"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["temperature"] == 0.2
