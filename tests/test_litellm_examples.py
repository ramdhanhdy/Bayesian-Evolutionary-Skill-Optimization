from __future__ import annotations

import importlib
import sys
import time
import types

import pytest


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


def test_litellm_completion_openrouter_auto_detect(monkeypatch) -> None:
    experiment = importlib.import_module("examples.run_toy_experiment")
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    monkeypatch.setattr(experiment, "MODEL", "poolside/laguna-xs.2:free")
    monkeypatch.setattr(experiment, "PROVIDER_HINT", "")
    monkeypatch.setenv("BESO_OPENROUTER_API_KEY", "test-key-openrouter")

    output = experiment.litellm_completion(
        "prompt",
        system="system",
        max_tokens=10,
        temperature=0.0,
    )

    assert output == "ok"
    assert captured["model"] == "openrouter/poolside/laguna-xs.2:free"


def test_litellm_completion_openrouter_explicit_provider(monkeypatch) -> None:
    experiment = importlib.import_module("examples.run_toy_experiment")
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    monkeypatch.setattr(experiment, "MODEL", "poolside/laguna-xs.2:free")
    monkeypatch.setattr(experiment, "PROVIDER_HINT", "openrouter")
    monkeypatch.setenv("BESO_LITELLM_API_KEY", "test-key-litellm")

    output = experiment.litellm_completion(
        "prompt",
        system="system",
        max_tokens=10,
        temperature=0.0,
    )

    assert output == "ok"
    assert captured["model"] == "openrouter/poolside/laguna-xs.2:free"


def test_litellm_completion_retry_and_backoff(monkeypatch) -> None:
    experiment = importlib.import_module("examples.run_toy_experiment")

    call_count = 0
    def completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Rate limit exceeded: 429")
        return {"choices": [{"message": {"content": f"success_on_attempt_{call_count}"}}]}

    # Mock litellm module and exceptions
    class MockRateLimitError(Exception):
        pass

    mock_exceptions = types.SimpleNamespace(
        RateLimitError=MockRateLimitError,
        APIConnectionError=Exception,
        ServiceUnavailableError=Exception,
    )
    mock_litellm = types.SimpleNamespace(
        completion=completion,
        exceptions=mock_exceptions,
    )

    monkeypatch.setitem(sys.modules, "litellm", mock_litellm)
    monkeypatch.setattr(experiment, "MODEL", "poolside/laguna-xs.2:free")
    monkeypatch.setattr(experiment, "PROVIDER_HINT", "openrouter")
    monkeypatch.setenv("BESO_LITELLM_API_KEY", "test-key-litellm")

    # Configure low retry delay for fast test and set retries to 3
    monkeypatch.setenv("BESO_LITELLM_MAX_RETRIES", "3")
    monkeypatch.setenv("BESO_LITELLM_INITIAL_DELAY", "0.01")
    monkeypatch.setenv("BESO_LITELLM_BACKOFF_FACTOR", "2.0")

    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda x: sleeps.append(x))

    output = experiment.litellm_completion(
        "prompt",
        system="system",
        max_tokens=10,
        temperature=0.0,
    )

    # It should have failed twice and succeeded on the 3rd call
    assert output == "success_on_attempt_3"
    assert call_count == 3
    assert len(sleeps) == 2
    assert sleeps[0] >= 0.01


def test_litellm_completion_retries_provider_error_finish_reason(monkeypatch) -> None:
    experiment = importlib.import_module("examples.run_toy_experiment")

    call_count = 0

    def completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "choices": [
                    {
                        "finish_reason": "error",
                        "message": {"content": ""},
                    }
                ],
                "error": {"message": "upstream provider failed"},
            }
        return {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    monkeypatch.setattr(experiment, "MODEL", "poolside/laguna-xs.2:free")
    monkeypatch.setattr(experiment, "PROVIDER_HINT", "openrouter")
    monkeypatch.setenv("BESO_LITELLM_API_KEY", "test-key-litellm")
    monkeypatch.setenv("BESO_LITELLM_MAX_RETRIES", "1")
    monkeypatch.setenv("BESO_LITELLM_INITIAL_DELAY", "0.01")
    monkeypatch.setattr(time, "sleep", lambda x: None)

    output = experiment.litellm_completion(
        "prompt",
        system="system",
        max_tokens=10,
        temperature=0.0,
    )

    assert output == "ok"
    assert call_count == 2


def test_litellm_completion_rejects_empty_content(monkeypatch) -> None:
    experiment = importlib.import_module("examples.run_toy_experiment")

    def completion(**kwargs):
        return {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    monkeypatch.setattr(experiment, "MODEL", "poolside/laguna-xs.2:free")
    monkeypatch.setattr(experiment, "PROVIDER_HINT", "openrouter")
    monkeypatch.setenv("BESO_LITELLM_API_KEY", "test-key-litellm")
    monkeypatch.setenv("BESO_LITELLM_MAX_RETRIES", "0")

    with pytest.raises(experiment._LiteLLMRetryableResponseError, match="empty content"):
        experiment.litellm_completion(
            "prompt",
            system="system",
            max_tokens=10,
            temperature=0.0,
        )

