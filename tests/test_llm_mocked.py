"""Offline tests of the LLM boundary: parsing and error propagation.

call_llm uses the OpenAI SDK against the configured OpenAI-compatible endpoint
(Parley by default). These tests mock at that boundary so no key and no network
are needed.
"""

import json
from unittest.mock import MagicMock, patch

import app.utils as utils
from app.agents import call_llm_for_generation, call_llm_for_reflection

PARLEY_BASE_URL = "https://parley.api.mit.edu/v1"


def _completion(content: str):
    completion = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    completion.choices = [choice]
    return completion


def test_generation_happy_path_parses_hypotheses(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    payload = json.dumps(
        [
            {"title": "Hypothesis A", "text": "Perovskite tandem cells."},
            {"title": "Hypothesis B", "text": "Bifacial panel coatings."},
        ]
    )
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _completion(payload)
        result = call_llm_for_generation("test goal", num_hypotheses=2, temperature=0.7)

    assert [h["title"] for h in result] == ["Hypothesis A", "Hypothesis B"]
    assert all("text" in h for h in result)


def test_generation_handles_markdown_fenced_json(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    payload = '```json\n[{"title": "T", "text": "X"}]\n```'
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _completion(payload)
        result = call_llm_for_generation("test goal")

    assert result == [{"title": "T", "text": "X"}]


def test_generation_passes_selected_model_to_llm_boundary():
    payload = '[{"title": "T", "text": "X"}]'
    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        result = call_llm_for_generation("test goal", model="gpt-5-mini")

    assert result == [{"title": "T", "text": "X"}]
    assert mock_call.call_args.kwargs["model"] == "gpt-5-mini"


def test_401_propagates_as_error_hypothesis(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "invalid-key")
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.side_effect = Exception(
            "Error code: 401 - No auth credentials found"
        )
        result = call_llm_for_generation("test goal", num_hypotheses=2)

    assert len(result) == 1
    assert result[0]["title"] == "Error"
    assert "OpenAI" in result[0]["text"] or "401" in result[0]["text"]


def test_missing_key_short_circuits_without_call(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch.object(utils, "OpenAI") as mock_openai:
        response = utils.call_llm("prompt")

    assert response.startswith("Error:")
    assert "key" in response.lower()
    mock_openai.assert_not_called()


def test_openai_provider_uses_openai_key_endpoint_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _completion("response")
        response = utils.call_llm("prompt")

    assert response == "response"
    assert mock_openai.call_args.kwargs["api_key"] == "fake-openai-key"
    assert mock_openai.call_args.kwargs["base_url"] == PARLEY_BASE_URL
    assert mock_openai.return_value.chat.completions.create.call_args.kwargs["model"] == "gpt-5-mini"


def test_openai_provider_requires_openai_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key-is-not-valid-here")

    with patch.object(utils, "OpenAI") as mock_openai:
        response = utils.call_llm("prompt")

    assert response == "Error: OpenAI API key not set."
    mock_openai.assert_not_called()


def test_openai_provider_does_not_fetch_openrouter_fallbacks(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setitem(utils.config, "max_retries", 1)

    with (
        patch.object(utils, "fetch_free_models") as mock_fetch,
        patch.object(utils, "OpenAI") as mock_openai,
    ):
        mock_openai.return_value.chat.completions.create.side_effect = Exception("No endpoints found")
        response = utils.call_llm("prompt")

    assert utils.classify_llm_error(response) == "Model unavailable or delisted"
    mock_fetch.assert_not_called()


def test_reflection_error_returns_not_reviewed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    # call_llm is imported into app.agents' namespace, so patch it there.
    with patch("app.agents.call_llm", return_value="Error: API call failed"):
        review = call_llm_for_reflection("some hypothesis")

    assert review["novelty_review"] == "Not reviewed"
    assert review["feasibility_review"] == "Not reviewed"
    assert review["references"] == []


def test_reflection_passes_selected_model_to_llm_boundary():
    payload = json.dumps(
        {
            "novelty_review": "HIGH",
            "feasibility_review": "MEDIUM",
            "comment": "Looks plausible.",
            "references": [],
        }
    )
    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        review = call_llm_for_reflection("some hypothesis", model="gpt-5-mini")

    assert review["novelty_review"] == "HIGH"
    assert mock_call.call_args.kwargs["model"] == "gpt-5-mini"
