"""Offline tests for the LLM tournament judge (mocked at the LLM boundary)."""

import json
from unittest.mock import patch

from app.agents import (
    RankingAgent,
    _score_based_winner,
    judge_pair,
    run_pairwise_debate,
    update_elo,
)
from app.models import ContextMemory, Hypothesis, ResearchGoal


def _hypo(hid: str, title: str, text: str, novelty="HIGH", feasibility="MEDIUM") -> Hypothesis:
    h = Hypothesis(hid, title, text)
    h.novelty_review = novelty
    h.feasibility_review = feasibility
    h.review_comments = [f"Comment on {hid}"]
    return h


def test_judge_pair_passes_real_research_goal_description():
    goal = ResearchGoal(description="Increase solar panel efficiency", llm_model="test/model")
    a = _hypo("G1", "Perovskite", "Use perovskite tandem cells.")
    b = _hypo("G2", "Coatings", "Use anti-reflective coatings.", novelty="LOW", feasibility="HIGH")
    payload = json.dumps(
        {
            "winner": "A",
            "confidence": 0.8,
            "reasoning": "A is more novel and relevant.",
            "criterion_scores": {"novelty": {"A": 5, "B": 2}},
        }
    )

    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        result = judge_pair(goal, a, b)

    assert result["winner"] == "A"
    assert result["confidence"] == 0.8
    prompt = mock_call.call_args.args[0]
    assert "Increase solar panel efficiency" in prompt
    assert "Compare hypotheses for tournament" not in prompt
    assert "Use perovskite tandem cells." in prompt
    assert "Use anti-reflective coatings." in prompt
    assert "Novelty: HIGH" in prompt
    assert "Novelty: LOW" in prompt
    assert mock_call.call_args.kwargs["model"] == "test/model"


def test_judge_pair_parses_fenced_json_and_tie():
    goal = ResearchGoal(description="Goal X")
    a = _hypo("G1", "A", "text a")
    b = _hypo("G2", "B", "text b")
    payload = '```json\n{"winner": "TIE", "confidence": 0.5, "reasoning": "Close.", "criterion_scores": {}}\n```'

    with patch("app.agents.call_llm", return_value=payload):
        result = judge_pair(goal, a, b)

    assert result["winner"] == "TIE"
    assert result["confidence"] == 0.5


def test_judge_pair_llm_error_returns_error_winner():
    goal = ResearchGoal(description="Goal X")
    a = _hypo("G1", "A", "text a")
    b = _hypo("G2", "B", "text b")

    with patch("app.agents.call_llm", return_value="Error: API call failed"):
        result = judge_pair(goal, a, b)

    assert result["winner"] == "ERROR"
    assert "API call failed" in result["reasoning"]


def test_run_pairwise_debate_maps_llm_winner_and_handles_swap():
    goal = ResearchGoal(description="Goal X")
    a = _hypo("G1", "A", "text a", novelty="HIGH", feasibility="HIGH")
    b = _hypo("G2", "B", "text b", novelty="LOW", feasibility="LOW")
    payload = json.dumps({"winner": "A", "confidence": 0.9, "reasoning": "A wins.", "criterion_scores": {}})

    # Force no swap so LLM "A" maps to hypoA.
    with (
        patch("app.agents.random.random", return_value=0.0),
        patch("app.agents.call_llm", return_value=payload),
    ):
        winner, judgment = run_pairwise_debate(a, b, research_goal=goal)

    assert winner is a
    assert judgment["winner"] == "A"
    assert judgment["method"] == "llm"


def test_run_pairwise_debate_swap_remaps_winner_to_original_labels():
    goal = ResearchGoal(description="Goal X")
    a = _hypo("G1", "A", "text a")
    b = _hypo("G2", "B", "text b")
    # After swap, presented_a is hypoB. LLM says "A" => hypoB wins.
    payload = json.dumps({"winner": "A", "confidence": 0.7, "reasoning": "Presented A wins.", "criterion_scores": {}})

    with (
        patch("app.agents.random.random", return_value=0.9),  # trigger swap
        patch("app.agents.call_llm", return_value=payload),
    ):
        winner, judgment = run_pairwise_debate(a, b, research_goal=goal)

    assert winner is b
    assert judgment["winner"] == "B"
    assert judgment["method"] == "llm"


def test_run_pairwise_debate_tie_skips_winner():
    goal = ResearchGoal(description="Goal X")
    a = _hypo("G1", "A", "text a")
    b = _hypo("G2", "B", "text b")
    payload = json.dumps({"winner": "TIE", "confidence": 0.4, "reasoning": "Even.", "criterion_scores": {}})

    with patch("app.agents.call_llm", return_value=payload):
        winner, judgment = run_pairwise_debate(a, b, research_goal=goal)

    assert winner is None
    assert judgment["winner"] == "TIE"


def test_run_pairwise_debate_falls_back_on_llm_failure():
    goal = ResearchGoal(description="Goal X")
    a = _hypo("G1", "A", "text a", novelty="HIGH", feasibility="HIGH")
    b = _hypo("G2", "B", "text b", novelty="LOW", feasibility="LOW")

    with patch("app.agents.call_llm", return_value="Error: rate limited"):
        winner, judgment = run_pairwise_debate(a, b, research_goal=goal)

    assert winner is a  # higher novelty+feasibility
    assert judgment["method"] == "score_fallback"


def test_score_based_winner_prefers_higher_reviews():
    a = _hypo("G1", "A", "text a", novelty="HIGH", feasibility="HIGH")
    b = _hypo("G2", "B", "text b", novelty="LOW", feasibility="LOW")
    assert _score_based_winner(a, b) is a


def test_update_elo_still_adjusts_scores():
    a = _hypo("G1", "A", "text a")
    b = _hypo("G2", "B", "text b")
    a.elo_score = 1200.0
    b.elo_score = 1200.0
    update_elo(a, b, k_factor=32)
    assert a.elo_score > 1200.0
    assert b.elo_score < 1200.0


def test_tournament_uses_llm_judge_and_records_judgment():
    goal = ResearchGoal(description="Real research question about solar cells")
    context = ContextMemory()
    a = _hypo("G1", "A", "hypothesis a text", novelty="HIGH", feasibility="MEDIUM")
    b = _hypo("G2", "B", "hypothesis b text", novelty="LOW", feasibility="HIGH")
    context.add_hypothesis(a)
    context.add_hypothesis(b)

    payload = json.dumps(
        {
            "winner": "A",
            "confidence": 0.85,
            "reasoning": "A better matches the goal.",
            "criterion_scores": {"relevance": {"A": 5, "B": 3}},
        }
    )

    with (
        patch("app.agents.random.shuffle"),  # keep deterministic pair order
        patch("app.agents.random.random", return_value=0.0),  # no swap
        patch("app.agents.call_llm", return_value=payload) as mock_call,
    ):
        RankingAgent().run_tournament([a, b], context, goal)

    assert len(context.tournament_results) == 1
    record = context.tournament_results[0]
    assert record["winner"] == "G1"
    assert record["loser"] == "G2"
    assert record["tie"] is False
    assert record["judgment"]["method"] == "llm"
    assert record["judgment"]["confidence"] == 0.85
    assert a.elo_score > b.elo_score
    prompt = mock_call.call_args.args[0]
    assert "Real research question about solar cells" in prompt


def test_tournament_tie_does_not_update_elo():
    goal = ResearchGoal(description="Goal X")
    context = ContextMemory()
    a = _hypo("G1", "A", "text a")
    b = _hypo("G2", "B", "text b")
    a.elo_score = 1300.0
    b.elo_score = 1100.0
    context.add_hypothesis(a)
    context.add_hypothesis(b)
    payload = json.dumps({"winner": "TIE", "confidence": 0.5, "reasoning": "Tie.", "criterion_scores": {}})

    with (
        patch("app.agents.random.shuffle"),
        patch("app.agents.call_llm", return_value=payload),
    ):
        RankingAgent().run_tournament([a, b], context, goal)

    assert a.elo_score == 1300.0
    assert b.elo_score == 1100.0
    assert context.tournament_results[0]["tie"] is True
    assert context.tournament_results[0]["winner"] is None
