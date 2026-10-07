"""Offline tests for rich peer-review reflection (mocked at the LLM boundary)."""

import json
from unittest.mock import patch

from app.agents import ReflectionAgent, call_llm_for_reflection
from app.models import REVIEW_SCORE_KEYS, ContextMemory, Hypothesis, ResearchGoal


def _rich_payload(**overrides):
    payload = {
        "review_scores": {
            "scientific_soundness": 4,
            "novelty": 5,
            "relevance": 4,
            "feasibility": 2,
            "testability": 5,
            "clarity": 3,
            "potential_impact": 4,
        },
        "review_strengths": ["Clear mechanism"],
        "review_weaknesses": ["Materials cost"],
        "critical_assumptions": ["Stable interface assumed"],
        "falsification_conditions": ["If efficiency gain < 1% in controlled test, reject"],
        "safety_ethical_concerns": ["Lead-containing precursors"],
        "recommended_improvements": ["Specify encapsulation strategy"],
        "comment": "Promising but feasibility is a concern.",
        "references": ["2301.12345"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_reflection_parses_rich_schema_and_derives_ordinals():
    with patch("app.agents.call_llm", return_value=_rich_payload()) as mock_call:
        review = call_llm_for_reflection(
            "Use perovskite tandems.",
            model="gpt-5-mini",
            research_goal="Increase solar panel efficiency",
        )

    assert review["novelty_review"] == "HIGH"
    assert review["feasibility_review"] == "LOW"
    assert review["review_scores"]["novelty"] == 5
    assert review["review_scores"]["feasibility"] == 2
    assert set(review["review_scores"]) == set(REVIEW_SCORE_KEYS)
    assert review["critical_assumptions"] == ["Stable interface assumed"]
    assert review["falsification_conditions"][0].startswith("If efficiency")
    assert review["safety_ethical_concerns"] == ["Lead-containing precursors"]
    prompt = mock_call.call_args.args[0]
    assert "Increase solar panel efficiency" in prompt
    assert "critical_assumptions" in prompt
    assert "falsification_conditions" in prompt
    assert mock_call.call_args.kwargs["model"] == "gpt-5-mini"


def test_reflection_handles_fenced_json():
    payload = "```json\n" + _rich_payload() + "\n```"
    with patch("app.agents.call_llm", return_value=payload):
        review = call_llm_for_reflection("hypothesis text")

    assert review["review_scores"]["testability"] == 5
    assert review["recommended_improvements"] == ["Specify encapsulation strategy"]


def test_reflection_legacy_ordinal_payload_still_works():
    payload = json.dumps(
        {
            "novelty_review": "HIGH",
            "feasibility_review": "MEDIUM",
            "comment": "Legacy shape.",
            "references": [],
        }
    )
    with patch("app.agents.call_llm", return_value=payload):
        review = call_llm_for_reflection("hypothesis text")

    assert review["novelty_review"] == "HIGH"
    assert review["feasibility_review"] == "MEDIUM"
    assert review["review_scores"]["novelty"] == 5
    assert review["review_scores"]["feasibility"] == 3


def test_reflection_error_returns_not_reviewed_rich_defaults():
    with patch("app.agents.call_llm", return_value="Error: API call failed"):
        review = call_llm_for_reflection("hypothesis text")

    assert review["novelty_review"] == "Not reviewed"
    assert review["feasibility_review"] == "Not reviewed"
    assert review["review_scores"] == {k: 0 for k in REVIEW_SCORE_KEYS}
    assert review["critical_assumptions"] == []
    assert review["references"] == []
    assert review["error"] == "Error: API call failed"


def test_reflection_unparsable_output_is_an_error_not_medium():
    with patch("app.agents.call_llm", return_value="I think it's pretty good."):
        review = call_llm_for_reflection("hypothesis text")

    assert review["error"].startswith("Could not parse LLM response")
    assert review["novelty_review"] == "Not reviewed"
    assert review["review_scores"] == {k: 0 for k in REVIEW_SCORE_KEYS}


def test_reflection_agent_reports_failure_and_keeps_prior_review():
    goal = ResearchGoal(description="Goal")
    context = ContextMemory()
    reviewed = Hypothesis("G1", "Reviewed", "body one")
    reviewed.review_scores = {"novelty": 4}
    reviewed.novelty_review = "HIGH"
    fresh = Hypothesis("G2", "Fresh", "body two")
    context.add_hypothesis(reviewed)
    context.add_hypothesis(fresh)

    with patch("app.agents.call_llm", return_value="Error: Rate limit exceeded: slow down"):
        errors = ReflectionAgent().review_hypotheses([reviewed, fresh], context, goal)

    assert len(errors) == 1
    assert "G1" in errors[0] and "G2" in errors[0]
    assert "Rate limit exceeded" in errors[0]
    assert reviewed.review_scores == {"novelty": 4}
    assert reviewed.novelty_review == "HIGH"
    assert fresh.novelty_review == "Not reviewed"


def test_reflection_agent_populates_hypothesis_fields():
    goal = ResearchGoal(description="Real research goal")
    context = ContextMemory()
    hypo = Hypothesis("G1", "Title", "Hypothesis body")
    context.add_hypothesis(hypo)

    with patch("app.agents.call_llm", return_value=_rich_payload()):
        errors = ReflectionAgent().review_hypotheses([hypo], context, goal)

    assert errors == []
    assert hypo.novelty_review == "HIGH"
    assert hypo.feasibility_review == "LOW"
    assert hypo.review_scores["scientific_soundness"] == 4
    assert hypo.critical_assumptions == ["Stable interface assumed"]
    assert hypo.falsification_conditions
    assert hypo.safety_ethical_concerns == ["Lead-containing precursors"]
    assert hypo.recommended_improvements == ["Specify encapsulation strategy"]
    assert hypo.references == ["2301.12345"]
    assert any("Promising but feasibility" in c for c in hypo.review_comments)
