"""Model fallback under the default OpenAI/Parley provider.

OpenAI path: only explicit fallback_models are tried (no live OpenRouter free
list). OpenRouter free-model discovery helpers remain covered for the optional
openrouter provider. Offline — the client and fetches are mocked.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

import app.utils as utils
from app.utils import _model_candidates, call_llm, classify_llm_error, order_free_models_for_demo


@pytest.fixture(autouse=True)
def _reset_model_cache(monkeypatch, tmp_path):
    """The free-model list is cached in a module global; reset between tests."""
    monkeypatch.setattr(utils, "OPENROUTER_MODEL_CACHE_PATH", str(tmp_path / "openrouter_free_models.json"))
    utils._free_models_cache = None
    utils._free_models_cache_checked_at = None
    yield
    utils._free_models_cache = None
    utils._free_models_cache_checked_at = None


def _ok_completion(content: str):
    completion = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    completion.choices = [choice]
    return completion


def _client_that(behavior):
    """Mocked OpenAI client whose create() dispatches on the model kwarg."""
    client = MagicMock()

    def create(model=None, messages=None, temperature=None):
        return behavior(model)

    client.chat.completions.create.side_effect = create
    return client


def _raise(msg):
    def _b(model):
        raise Exception(msg)

    return _b


# --- candidate ordering ---


def test_model_candidates_dedup_and_order():
    assert _model_candidates("a", ["b", "a", "c"]) == ["a", "b", "c"]
    assert _model_candidates("a", None) == ["a"]
    assert _model_candidates("a", []) == ["a"]


def test_order_free_models_for_demo_prefers_dynamically_detected_compact_models():
    models = [
        "vendor/model-70b:free",
        "paid/model",
        "vendor/model-3b:free",
        "vendor/model-8b:free",
        "vendor/model-unknown:free",
        "vendor/model-3b:free",
    ]

    assert order_free_models_for_demo(models) == [
        "vendor/model-3b:free",
        "vendor/model-8b:free",
        "vendor/model-70b:free",
        "vendor/model-unknown:free",
    ]


# --- fetch_free_models: filtering + caching, never raises ---


def test_fetch_free_models_filters_and_caches(monkeypatch):
    payload = {"data": [{"id": "x/y:free"}, {"id": "a/b"}, {"id": "c/d:free"}, {"id": None}]}
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    with patch.object(utils.requests, "get", return_value=resp) as mock_get:
        first = utils.fetch_free_models()
        second = utils.fetch_free_models()  # cached — no second HTTP call
    assert first == ["c/d:free", "x/y:free"]
    assert second == first
    assert mock_get.call_count == 1


def test_fetch_free_models_uses_fresh_weekly_cache():
    utils._write_cached_free_models(["vendor/model-70b:free", "vendor/model-3b:free"])

    with patch.object(utils.requests, "get") as mock_get:
        result = utils.fetch_free_models()

    assert result == ["vendor/model-3b:free", "vendor/model-70b:free"]
    mock_get.assert_not_called()


def test_fetch_free_models_refreshes_stale_cache(monkeypatch):
    cache_path = utils.Path(utils.OPENROUTER_MODEL_CACHE_PATH)
    cache_path.write_text(
        json.dumps(
            {
                "fetched_at": time.time() - 10,
                "models": ["old/model-3b:free"],
            }
        )
    )
    monkeypatch.setattr(utils, "OPENROUTER_MODEL_CACHE_TTL_SECONDS", 1)
    payload = {"data": [{"id": "new/model-7b:free"}]}
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None

    with patch.object(utils.requests, "get", return_value=resp) as mock_get:
        result = utils.fetch_free_models()

    assert result == ["new/model-7b:free"]
    assert mock_get.call_count == 1


def test_fetch_free_models_refreshes_stale_in_memory_cache(monkeypatch):
    utils._free_models_cache = ["old/model-3b:free"]
    utils._free_models_cache_checked_at = time.time() - 10
    monkeypatch.setattr(utils, "OPENROUTER_MODEL_CACHE_TTL_SECONDS", 1)
    payload = {"data": [{"id": "new/model-7b:free"}]}
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None

    with patch.object(utils.requests, "get", return_value=resp) as mock_get:
        result = utils.fetch_free_models()

    assert result == ["new/model-7b:free"]
    assert mock_get.call_count == 1


def test_fetch_free_models_uses_stale_cache_when_refresh_fails(monkeypatch):
    cache_path = utils.Path(utils.OPENROUTER_MODEL_CACHE_PATH)
    cache_path.write_text(
        json.dumps(
            {
                "fetched_at": time.time() - 10,
                "models": ["stale/model-3b:free"],
            }
        )
    )
    monkeypatch.setattr(utils, "OPENROUTER_MODEL_CACHE_TTL_SECONDS", 1)

    with patch.object(utils.requests, "get", side_effect=Exception("network down")):
        assert utils.fetch_free_models() == ["stale/model-3b:free"]


def test_fetch_free_models_returns_empty_on_error(monkeypatch):
    with patch.object(utils.requests, "get", side_effect=Exception("network down")):
        assert utils.fetch_free_models() == []


def test_get_fallback_models_excludes_primary_from_dynamic_list(monkeypatch):
    with patch.object(utils, "fetch_free_models", return_value=["primary/model:free", "other/model:free"]):
        fallbacks = utils.get_fallback_models(primary="primary/model:free")

    assert fallbacks == ["other/model:free"]


# --- fallback behavior (explicit list) ---


def test_falls_back_to_working_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    def behavior(model):
        if model == "gpt-5-mini":
            return _ok_completion("RECOVERED CONTENT")
        raise Exception("No endpoints found for model")

    with patch.object(utils, "OpenAI", return_value=_client_that(behavior)):
        result = call_llm("prompt", model="dead-model", fallback_models=["gpt-5-mini"])
    assert result == "RECOVERED CONTENT"


def test_no_fallback_on_auth_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    client = _client_that(_raise("Error code: 401 - No auth credentials found"))
    with patch.object(utils, "OpenAI", return_value=client):
        result = call_llm("prompt", model="primary", fallback_models=["fallback"])
    assert "401" in result or "Authentication with OpenAI failed" in result
    assert client.chat.completions.create.call_count == 1  # never tries other models


def test_primary_success_skips_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    client = _client_that(lambda model: _ok_completion("PRIMARY OK"))
    with patch.object(utils, "OpenAI", return_value=client):
        result = call_llm("prompt", model="primary", fallback_models=["fallback"])
    assert result == "PRIMARY OK"
    assert client.chat.completions.create.call_count == 1  # no fallback attempt


# --- OpenAI path: no automatic live free-model list ---


def test_call_llm_leaves_temperature_at_provider_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    client = MagicMock()
    client.chat.completions.create.return_value = _ok_completion("OK")

    with patch.object(utils, "OpenAI", return_value=client):
        assert call_llm("prompt", temperature=0.7, model="gpt-5-mini") == "OK"

    assert "temperature" not in client.chat.completions.create.call_args.kwargs


def test_openai_does_not_auto_fallback_to_live_free_models(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    client = _client_that(_raise("No endpoints found for model"))

    with (
        patch.object(utils, "fetch_free_models", return_value=["live-a:free", "live-b:free"]) as mock_fetch,
        patch.object(utils, "OpenAI", return_value=client),
    ):
        result = call_llm("prompt", model="dead-primary")  # no explicit fallback_models
    assert classify_llm_error(result) == "Model unavailable or delisted"
    assert client.chat.completions.create.call_count == 1
    mock_fetch.assert_not_called()


def test_no_hardcoded_fallback_without_explicit_list(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    client = _client_that(_raise("No endpoints found for model"))

    with patch.object(utils, "OpenAI", return_value=client):
        result = call_llm("prompt", model="dead-primary")
    assert classify_llm_error(result) == "Model unavailable or delisted"
    assert client.chat.completions.create.call_count == 1


def test_all_unavailable_surfaces_clear_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    client = _client_that(_raise("No endpoints found"))
    with patch.object(utils, "OpenAI", return_value=client):
        result = call_llm("prompt", model="a", fallback_models=["b", "c"])
    assert classify_llm_error(result) == "Model unavailable or delisted"
    assert client.chat.completions.create.call_count == 3  # primary + 2 explicit fallbacks


def test_fallback_attempts_are_bounded(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    client = _client_that(_raise("No endpoints found"))
    many = [f"m{i}" for i in range(20)]
    with patch.object(utils, "OpenAI", return_value=client):
        call_llm("prompt", model="a", fallback_models=many)
    # primary + at most MAX_FALLBACK_ATTEMPTS, never all 20.
    assert client.chat.completions.create.call_count == 1 + utils.MAX_FALLBACK_ATTEMPTS


def test_missing_key_still_short_circuits(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch.object(utils, "OpenAI") as mock_openai:
        result = call_llm("prompt", model="x", fallback_models=["y"])
    assert result.startswith("Error:") and "key" in result.lower()
    mock_openai.assert_not_called()


# --- rate-limited / timed-out models: move to the next explicit fallback ---


def test_rate_limited_primary_falls_back(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    def behavior(model):
        if model == "good":
            return _ok_completion("OK AFTER RATE LIMIT")
        raise Exception("Error code: 429 - temporarily rate-limited upstream")

    with patch.object(utils, "OpenAI", return_value=_client_that(behavior)):
        result = call_llm("prompt", model="busy", fallback_models=["good"])
    assert result == "OK AFTER RATE LIMIT"


def test_all_rate_limited_surfaces_clear_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    client = _client_that(_raise("Error code: 429 - rate-limited"))
    with patch.object(utils, "OpenAI", return_value=client):
        result = call_llm("prompt", model="a", fallback_models=["b"])
    assert classify_llm_error(result) == "Rate limited by the model provider"
    assert client.chat.completions.create.call_count == 2  # primary + 1 fallback, no long retry


def test_timed_out_primary_falls_back(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    def behavior(model):
        if model == "good":
            return _ok_completion("OK AFTER TIMEOUT")
        raise Exception("Request timed out while waiting for provider")

    with patch.object(utils, "OpenAI", return_value=_client_that(behavior)):
        result = call_llm("prompt", model="slow", fallback_models=["good"])
    assert result == "OK AFTER TIMEOUT"


def test_all_timed_out_surfaces_clear_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    client = _client_that(_raise("Request timeout from provider"))
    with patch.object(utils, "OpenAI", return_value=client):
        result = call_llm("prompt", model="a", fallback_models=["b"])
    assert classify_llm_error(result) == "Model provider timed out"
    assert client.chat.completions.create.call_count == 2


def test_generic_provider_error_also_falls_back(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    def behavior(model):
        if model == "good":
            return _ok_completion("RECOVERED")
        raise Exception("Error code: 502 - Provider returned error")

    with patch.object(utils, "OpenAI", return_value=_client_that(behavior)):
        result = call_llm("prompt", model="broken", fallback_models=["good"])
    assert result == "RECOVERED"
